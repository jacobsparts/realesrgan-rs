//! The RRDBNet graph: geometry, weights, and the CPU kernels that mirror the
//! GPU ones.
//!
//! The architecture is 350 3x3 convolutions with LeakyReLU(0.2), arranged as
//! 23 RRDB blocks (each 3 residual-dense blocks, each 5 convs), then a body
//! conv with a global residual, two nearest-neighbour 2x upsampling stages, and
//! a final conv. Both released models (`x4plus`, `x2plus`) are this same
//! network; the differences are the input channel count of `conv_first`
//! (3 vs 12, the x2 model entering through a pixel_unshuffle) and the output
//! scale.
//!
//! TWO RESIDUALS, BOTH SCALED BY 0.2. An RDB returns `conv5(cat(...)) * 0.2 + x`,
//! and an RRDB returns `rdb3(rdb2(rdb1(x))) * 0.2 + x` - i.e. the scaling is
//! applied at both levels, which is easy to get wrong in a way that still
//! produces a plausible image.
//!
//! CONCATENATION IS A LAYOUT, NOT AN OP. In an RDB, conv2..conv5 read the
//! concatenation of everything computed so far. Rather than copying channels
//! into a new tensor at each step, the dense block owns ONE
//! (num_feat + 4*num_grow_ch)-channel buffer and writes each result into its own
//! slot; each conv then reads a longer prefix of that one buffer. The kernel
//! takes a base pointer plus a channel count, so "the concatenation" costs
//! nothing, and the workspace is allocated once and reused by all 69 blocks.
//!
//! WEIGHT LAYOUT. The checkpoint stores conv weights as [c_out][c_in][3][3] -
//! `[oc][ci][ky][kx]` with the nine taps innermost - and that is the layout BOTH
//! backends take now: the CUDA kernel indexes it directly and
//! `lightgpu::ops::cpu::conv3x3s1p1` taps it as `w[(oc*c_in + ci)*9 + ky*3 + kx]`.
//! A second, ci-innermost copy used to be built for a private CPU kernel; the
//! kernel is gone and so is the copy. Both backends accumulate in (ky, kx, ci)
//! order, which is what makes them agree on these short reductions.

use crate::weights::Weights;

/// The lightgpu convolution twins. These are the whole reason this engine no
/// longer carries a convolution kernel of its own: `lightgpu/src/ops/cpu.rs`
/// holds ONE measured parallel + AVX2 3x3 and one 1x1, and every engine in the
/// family uses them rather than keeping a private copy that drifts. The section
/// below is the reference implementation the CUDA selftest and `--bench-conv`
/// still measure against, and the fallback for shapes the twin does not take.
use lightgpu::ops::cpu::conv3x3s1p1 as lg_conv3x3s1p1;


/// Model geometry, read from the checkpoint metadata.
#[derive(Clone, Debug)]
pub struct Geometry {
    pub scale: usize,
    pub num_feat: usize,
    pub num_block: usize,
    pub num_grow_ch: usize,
    pub in_ch: usize,
    pub out_ch: usize,
}

impl Geometry {
    /// Input channels of `conv_first`: the x2 model pre-unshuffles by 2x2.
    ///
    /// Only the GPU path needs this: the CPU forward pass derives the same
    /// number from `pixel_unshuffle2`'s output channel count.
    #[cfg(feature = "cuda")]
    pub fn first_in_ch(&self) -> usize {
        if self.scale == 2 {
            self.in_ch * 4
        } else {
            self.in_ch
        }
    }

    /// The dense-block workspace width.
    pub fn dense_width(&self) -> usize {
        self.num_feat + 4 * self.num_grow_ch
    }
}

/// A single 3x3 convolution's parameters.
pub struct Conv {
    /// [c_out][c_in][3][3] - the checkpoint's own order, which is also the
    /// toolkit kernel's. The nine taps of one (oc, ci) pair are contiguous and
    /// the channel stride is 9; the CPU twin's tap loop reads exactly that, so
    /// there is no transposed copy to hold.
    pub w: Vec<f32>,
    pub bias: Vec<f32>,
    pub c_out: usize,
}

impl Conv {
    /// Build a Conv from raw slices (weight in [c_out][c_in][3][3] order).
    /// Used by the CUDA selftest, which needs a Conv for an arbitrary shape
    /// rather than one read from a checkpoint.
    pub fn from_parts(w: &[f32], bias: &[f32], c_in: usize, c_out: usize) -> Conv {
        Conv::new(w, bias, c_in, c_out)
    }

    fn new(w: &[f32], bias: &[f32], c_in: usize, c_out: usize) -> Conv {
        assert_eq!(w.len(), c_out * c_in * 9, "conv weight size");
        assert_eq!(bias.len(), c_out, "conv bias size");
        Conv { w: w.to_vec(), bias: bias.to_vec(), c_out }
    }

    fn load(w: &Weights, name: &str) -> Result<Conv, String> {
        let wk = format!("{name}.weight");
        let bk = format!("{name}.bias");
        let shape = w.shape(&wk)?.to_vec();
        if shape.len() != 4 || shape[2] != 3 || shape[3] != 3 {
            return Err(format!("{}: expected a [c_out, c_in, 3, 3] weight, got {:?}", wk, shape));
        }
        Ok(Conv::new(w.get(&wk)?, w.get(&bk)?, shape[1], shape[0]))
    }
}

/// One residual dense block: 5 convs over a growing channel prefix.
pub struct DenseBlock {
    pub convs: [Conv; 5],
}

/// One RRDB: 3 dense blocks with a scaled residual.
pub struct Rrdb {
    pub rdbs: [DenseBlock; 3],
}

/// Everything the forward pass needs, in canonical order.
pub struct Model {
    pub geom: Geometry,
    pub conv_first: Conv,
    pub blocks: Vec<Rrdb>,
    pub conv_body: Conv,
    pub conv_up1: Conv,
    pub conv_up2: Conv,
    pub conv_hr: Conv,
    pub conv_last: Conv,
}

impl Model {
    pub fn load(w: &Weights) -> Result<Model, String> {
        let c = &w.config;
        let geom = Geometry {
            scale: c.scale,
            num_feat: c.num_feat,
            num_block: c.num_block,
            num_grow_ch: c.num_grow_ch,
            in_ch: c.in_ch,
            out_ch: c.out_ch,
        };
        let mut blocks = Vec::with_capacity(geom.num_block);
        for b in 0..geom.num_block {
            let mut rdbs = Vec::with_capacity(3);
            for r in 1..=3 {
                let base = format!("body.{b}.rdb{r}");
                let mut convs = Vec::with_capacity(5);
                for k in 1..=5 {
                    convs.push(Conv::load(w, &format!("{base}.conv{k}"))?);
                }
                rdbs.push(DenseBlock {
                    convs: convs.try_into().map_err(|_| "expected 5 convs")?,
                });
            }
            blocks.push(Rrdb {
                rdbs: rdbs.try_into().map_err(|_| "expected 3 rdbs")?,
            });
        }
        Ok(Model {
            geom,
            conv_first: Conv::load(w, "conv_first")?,
            blocks,
            conv_body: Conv::load(w, "conv_body")?,
            conv_up1: Conv::load(w, "conv_up1")?,
            conv_up2: Conv::load(w, "conv_up2")?,
            conv_hr: Conv::load(w, "conv_hr")?,
            conv_last: Conv::load(w, "conv_last")?,
        })
    }

    /// Bytes of f32 weights, for the startup banner.
    pub fn weight_bytes(&self) -> usize {
        let mut n = 0usize;
        let mut add = |c: &Conv| n += (c.w.len() + c.bias.len()) * 4;
        add(&self.conv_first);
        for b in &self.blocks {
            for r in &b.rdbs {
                for c in &r.convs {
                    add(c);
                }
            }
        }
        add(&self.conv_body);
        add(&self.conv_up1);
        add(&self.conv_up2);
        add(&self.conv_hr);
        add(&self.conv_last);
        n
    }
}

// ===========================================================================
// CPU path
// ===========================================================================

/// A planar tensor [c][h][w].
#[derive(Clone)]
pub struct Plane {
    pub c: usize,
    pub h: usize,
    pub w: usize,
    pub data: Vec<f32>,
}

impl Plane {
    pub fn new(c: usize, h: usize, w: usize) -> Plane {
        Plane { c, h, w, data: vec![0.0; c * h * w] }
    }

    #[inline]
    pub fn hw(&self) -> usize {
        self.h * self.w
    }

    /// Reshape in place, keeping the (possibly larger) allocation.
    fn set(&mut self, c: usize, h: usize, w: usize) {
        self.c = c;
        self.h = h;
        self.w = w;
        self.data.resize(c * h * w, 0.0);
    }
}

/// A view of `ch` channels of `ws` starting at channel `off`, as an owned Plane
/// (a debugging convenience for the stage dump; it copies).
fn ws_slice(ws: &Plane, off: usize, ch: usize, _nf: usize, hw: usize) -> Plane {
    let mut p = Plane::new(ch, ws.h, ws.w);
    p.data.copy_from_slice(&ws.data[off * hw..(off + ch) * hw]);
    p
}

#[inline]
pub fn leaky_relu(v: f32) -> f32 {
    if v >= 0.0 { v } else { 0.2 * v }
}

/// 3x3, stride 1, pad 1, reading the first `c_in` channels of `src` (which may
/// be a wider workspace) and writing into `dst`'s `c_out` channels.
///
/// `res` adds `scale * res[i]` after the optional activation.
///
/// THE CONVOLUTION IS THE TOOLKIT'S, NOT THIS ENGINE'S. There used to be a kernel
/// here: a row-band-parallel AVX2 3x3 with the output channels blocked three at a
/// time, an interior register tile, and an `axpy_span` fallback for the row ends.
/// `lightgpu::ops::cpu::conv3x3s1p1` is that same kernel - two output channels per
/// task, four `__m256` accumulators each, all 27 taps in registers - and it is the
/// one every engine in the family calls, so the copy is gone. What is left here is
/// the adapter: the twin for the convolution, then the activation and the residual,
/// which are graph concerns rather than convolution ones. The whole 350-conv
/// network therefore runs through one kernel that other engines exercise too, and
/// `--bench-conv` measures the same one.
///
/// THE ARITHMETIC AGREES TO ROUND-OFF, NOT TO THE BIT. Both accumulate the bias
/// first and then one fused multiply-add per tap in (ky, kx, ci) order, so the
/// twin's output on `conv_first` is EXACTLY this engine's `conv3x3_reference`
/// scalar (checked element by element) - but the twin's vector body is a
/// different instruction stream from the OLD kernel's, and the old kernel's
/// output differs from both by about 2.4e-07 on that first stage. Over 350
/// convolutions that compounds: measured against the pre-adoption binary on a
/// 128x128 input, 7.4e-05 on the last feature plane before the head, which is ONE
/// 8-bit level on 24 pixels out of 786432 of the final PNG. `REALESRGAN_REF=1`
/// runs `conv3x3_reference` instead, to see the scalar end of the same spread.
#[allow(clippy::too_many_arguments)]
fn conv3x3_prefix(
    src: &[f32],
    c_in: usize,
    dst: &mut [f32],
    conv: &Conv,
    h: usize,
    w: usize,
    act: bool,
    res: Option<(&[f32], f32)>,
) {
    let hw = h * w;
    let c_out = conv.c_out;
    // LA_PROFILE=1 accumulates wall time per (c_in, c_out, h, w) shape, so a
    // change here can be attributed to a shape instead of guessed at. The same
    // variable also brackets each CUDA launch in cuda.rs, which is why the
    // variable and the report outlived the kernel it was written for.
    let profiling = std::env::var("LA_PROFILE").map(|v| v == "1").unwrap_or(false);
    let t0 = if profiling { Some(std::time::Instant::now()) } else { None };
    if std::env::var("REALESRGAN_REF").map(|v| v == "1").unwrap_or(false) {
        conv3x3_reference(src, c_in, dst, conv, h, w, act, res);
        if let Some(t0) = t0 {
            CPU_PROFILE.lock().unwrap().push((c_in, c_out, h, w, act, t0.elapsed().as_secs_f64()));
        }
        return;
    }
    // The twin writes `c_out * hw` elements and folds the bias in first, so the
    // epilogue below must NOT add it again.
    lg_conv3x3s1p1(src, &conv.w, &conv.bias, dst, c_in, c_out, h, w);
    // The epilogue is the graph's, not the convolution's: the twin has no
    // activation and no residual, and folding the bias into the accumulator is
    // the one thing it does that used to live in the kernel's tail.
    if act || res.is_some() {
        for oc in 0..c_out {
            let row = &mut dst[oc * hw..(oc + 1) * hw];
            for (i, v) in row.iter_mut().enumerate() {
                let mut x = *v;
                if act {
                    x = leaky_relu(x);
                }
                if let Some((r, s)) = res {
                    x += s * r[oc * hw + i];
                }
                *v = x;
            }
        }
    }
    if let Some(t0) = t0 {
        CPU_PROFILE.lock().unwrap().push((c_in, c_out, h, w, act, t0.elapsed().as_secs_f64()));
    }
}

/// Per-shape CPU conv timings, filled when LA_PROFILE=1 and reported at exit.
pub static CPU_PROFILE: std::sync::Mutex<Vec<(usize, usize, usize, usize, bool, f64)>> =
    std::sync::Mutex::new(Vec::new());

/// Print the accumulated CPU conv profile, largest total first.
pub fn cpu_profile_report() {
    let g = CPU_PROFILE.lock().unwrap();
    if g.is_empty() {
        return;
    }
    let mut by_shape: std::collections::HashMap<(usize, usize, usize, usize, bool), (f64, usize)> =
        std::collections::HashMap::new();
    let total: f64 = g.iter().map(|e| e.5).sum();
    for &(ci, co, h, w, act, t) in g.iter() {
        let e = by_shape.entry((ci, co, h, w, act)).or_insert((0.0, 0));
        e.0 += t;
        e.1 += 1;
    }
    let mut v: Vec<_> = by_shape.into_iter().collect();
    v.sort_by(|a, b| b.1 .0.partial_cmp(&a.1 .0).unwrap());
    println!("cpu conv profile: {:.2}s total over {} calls", total, g.len());
    println!("    {:>5} {:>5} {:>5} {:>5} {:>6} {:>9} {:>5} {:>7}", "c_in", "c_out", "h", "w", "act", "seconds", "n", "share");
    for ((ci, co, h, w, act), (t, n)) in v.iter().take(12) {
        println!(
            "    {:>5} {:>5} {:>5} {:>5} {:>6} {:>8.3}s {:>5} {:>6.1}%",
            ci,
            co,
            h,
            w,
            if *act { "yes" } else { "no" },
            t,
            n,
            100.0 * t / total
        );
    }
}

/// The SCALAR reference the toolkit twin is measured against: `(ky, kx, ci)`, one
/// output pixel at a time, with the bias as the accumulator's initial value.
///
/// This is reached with `REALESRGAN_REF=1` and exists so a swap of the kernel under
/// `conv3x3_prefix` can be JUDGED rather than merely timed. Run the same image
/// twice, once on the twin and once here, and diff the two stage dumps. What that
/// showed when the twin replaced this engine's own AVX2 kernel on a 128x128 input:
/// the twin is element-for-element EXACTLY this scalar on `conv_first`, while the
/// kernel it replaced was off by 2.4e-07 there; by the last feature plane before
/// the head the twin and the old kernel differ by 7.4e-05, which is one 8-bit level
/// on 24 pixels out of 786432 of the final PNG. So this is a round-off-level
/// change in an accumulation order, not an implementation of a different op - and
/// the numbers above are the whole size of it, not an impression of it.
///
/// It is deliberately not vectorised and not parallel: it has one job, which is to
/// be obviously right.
#[allow(clippy::too_many_arguments)]
fn conv3x3_reference(
    src: &[f32],
    c_in: usize,
    dst: &mut [f32],
    conv: &Conv,
    h: usize,
    w: usize,
    act: bool,
    res: Option<(&[f32], f32)>,
) {
    let hw = h * w;
    for oc in 0..conv.c_out {
        let b = conv.bias[oc];
        for y in 0..h {
            for x in 0..w {
                let mut acc = b;
                for ky in 0..3usize {
                    let iy = match y_iter(y, ky, h) {
                        Some(v) => v,
                        None => continue,
                    };
                    for kx in 0..3usize {
                        let ix = x as isize + kx as isize - 1;
                        if ix < 0 || ix as usize >= w {
                            continue;
                        }
                        let ix = ix as usize;
                        let k = ky * 3 + kx;
                        for ci in 0..c_in {
                            acc += conv.w[(oc * c_in + ci) * 9 + k] * src[ci * hw + iy * w + ix];
                        }
                    }
                }
                let mut v = if act { leaky_relu(acc) } else { acc };
                if let Some((r, s)) = res {
                    v += s * r[oc * hw + y * w + x];
                }
                dst[oc * hw + y * w + x] = v;
            }
        }
    }
}

/// Row index for tap `ky` at output row `y`, or None when it falls outside.
#[inline]
fn y_iter(y: usize, ky: usize, h: usize) -> Option<usize> {
    let iy = y as isize + ky as isize - 1;
    if iy < 0 || iy as usize >= h {
        None
    } else {
        Some(iy as usize)
    }
}

/// The CPU 3x3 conv, exposed for the CUDA selftest.
#[allow(clippy::too_many_arguments)]
pub fn conv3x3_prefix_pub(
    src: &[f32],
    c_in: usize,
    dst: &mut [f32],
    conv: &Conv,
    h: usize,
    w: usize,
    act: bool,
    res: Option<(&[f32], f32)>,
) {
    conv3x3_prefix(src, c_in, dst, conv, h, w, act, res);
}

/// conv3x3_prefix with the destination being all of a fresh plane.
fn conv_to(src: &[f32], c_in: usize, dst: &mut Plane, conv: &Conv, act: bool) {
    dst.set(conv.c_out, dst.h, dst.w);
    let (h, w) = (dst.h, dst.w);
    conv3x3_prefix(src, c_in, &mut dst.data, conv, h, w, act, None);
}

/// One RDB: the growing prefix is the workspace itself. `dst` receives the
/// result of conv5 (before the residual, which the caller applies).
fn dense_block_cpu(blk: &DenseBlock, nf: usize, ng: usize, ws: &mut Plane, dst: &mut Plane) {
    let (h, w) = (ws.h, ws.w);
    let hw = h * w;
    let total = nf + 4 * ng;
    ws.data.resize(total * hw, 0.0);
    // ws already holds x in channels [0, nf).

    // conv1: reads 64, writes 32 into [nf, nf+ng)
    {
        let (head, tail) = ws.data.split_at_mut(nf * hw);
        conv3x3_prefix(head, nf, &mut tail[..ng * hw], &blk.convs[0], h, w, true, None);
    }
    // conv2: reads 96, writes 32 into [nf+ng, nf+2ng)
    {
        let (head, tail) = ws.data.split_at_mut((nf + ng) * hw);
        conv3x3_prefix(head, nf + ng, &mut tail[..ng * hw], &blk.convs[1], h, w, true, None);
    }
    // conv3: reads 128, writes 32 into [nf+2ng, nf+3ng)
    {
        let (head, tail) = ws.data.split_at_mut((nf + 2 * ng) * hw);
        conv3x3_prefix(head, nf + 2 * ng, &mut tail[..ng * hw], &blk.convs[2], h, w, true, None);
    }
    // conv4: reads 160, writes 32 into [nf+3ng, nf+4ng)
    {
        let (head, tail) = ws.data.split_at_mut((nf + 3 * ng) * hw);
        conv3x3_prefix(head, nf + 3 * ng, &mut tail[..ng * hw], &blk.convs[3], h, w, true, None);
    }
    // conv5: reads all 192, no activation.
    dst.set(nf, h, w);
    conv3x3_prefix(&ws.data[..total * hw], total, &mut dst.data, &blk.convs[4], h, w, false, None);
}

/// Nearest-neighbour 2x upsample with PyTorch's integer-division mapping.
pub fn upsample2x_nearest(inp: &Plane) -> Plane {
    let mut out = Plane::new(inp.c, inp.h * 2, inp.w * 2);
    let (oh, ow) = (out.h, out.w);
    let (ih, iw) = (inp.h, inp.w);
    for ci in 0..inp.c {
        let ip = &inp.data[ci * ih * iw..(ci + 1) * ih * iw];
        let op = &mut out.data[ci * oh * ow..(ci + 1) * oh * ow];
        for y in 0..oh {
            let sy = y / 2;
            for x in 0..ow {
                op[y * ow + x] = ip[sy * iw + x / 2];
            }
        }
    }
    out
}

/// PyTorch's pixel_unshuffle(2): [c][h][w] -> [c*4][h/2][w/2] with output
/// channel index c*4 + dy*2 + dx.
pub fn pixel_unshuffle2(inp: &Plane) -> Plane {
    let (oh, ow) = (inp.h / 2, inp.w / 2);
    let iw = inp.w;
    let mut out = Plane::new(inp.c * 4, oh, ow);
    for ci in 0..inp.c {
        let ip = &inp.data[ci * inp.hw()..(ci + 1) * inp.hw()];
        for dy in 0..2 {
            for dx in 0..2 {
                let oc = ci * 4 + dy * 2 + dx;
                let op = &mut out.data[oc * oh * ow..(oc + 1) * oh * ow];
                for y in 0..oh {
                    for x in 0..ow {
                        op[y * ow + x] = ip[(2 * y + dy) * iw + (2 * x + dx)];
                    }
                }
            }
        }
    }
    out
}

/// Write a plane as raw little-endian f32 with a (c,h,w) i32 header, the form
/// the parity scripts read. This is a debugging facility: it costs nothing when
/// unused and is the only way to localize a divergence to a stage.
pub fn dump_plane(dir: &str, name: &str, p: &Plane) {
    use std::io::Write;
    let path = format!("{dir}/{name}.f32");
    if let Ok(mut f) = std::fs::File::create(&path) {
        let _ = f.write_all(&(p.c as i32).to_le_bytes());
        let _ = f.write_all(&(p.h as i32).to_le_bytes());
        let _ = f.write_all(&(p.w as i32).to_le_bytes());
        let mut bytes = Vec::with_capacity(p.data.len() * 4);
        for v in &p.data {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let _ = f.write_all(&bytes);
    }
}

/// The full forward pass on the CPU. `progress` is called once per stage.
pub fn forward_cpu(model: &Model, input: &Plane, progress: &mut dyn FnMut(&str)) -> Plane {
    let g = model.geom.clone();
    let nf = g.num_feat;
    let ng = g.num_grow_ch;

    let dump_dir = std::env::var("REALESRGAN_DUMP").ok();
    let dump = |name: &str, p: &Plane| {
        if let Some(d) = &dump_dir {
            dump_plane(d, name, p);
        }
    };
    let mut feat = if g.scale == 2 {
        progress("pixel_unshuffle");
        let f = pixel_unshuffle2(input);
        dump("00_unshuffle", &f);
        f
    } else {
        input.clone()
    };

    progress("conv_first");
    let mut cur = Plane::new(nf, feat.h, feat.w);
    conv_to(&feat.data, feat.c, &mut cur, &model.conv_first, false);
    dump("01_conv_first", &cur);

    let hw = cur.hw();
    // The conv_first output is the target of the network's global skip
    // connection, so it must survive the body untouched.
    let skip = cur.clone();
    let mut ws = Plane::new(g.dense_width(), cur.h, cur.w);
    let mut x5 = Plane::new(nf, cur.h, cur.w);
    let mut rdb_out = Plane::new(nf, cur.h, cur.w);

    progress("body");
    for b in 0..g.num_block {
        // x is the input to this RRDB.
        let rrdb_in = cur.data.clone();
        let mut x = Plane::new(nf, cur.h, cur.w);
        x.data.copy_from_slice(&cur.data[..nf * hw]);
        for r in 0..3 {
            ws.data[..nf * hw].copy_from_slice(&x.data);
            dense_block_cpu(&model.blocks[b].rdbs[r], nf, ng, &mut ws, &mut x5);
            if b == 0 && r == 0 {
                dump("02_b0r0_conv1", &ws_slice(&ws, nf, ng, nf, hw));
                dump("03_b0r0_conv2", &ws_slice(&ws, nf + ng, ng, ng, hw));
                dump("04_b0r0_conv5", &x5);
            }
            // RDB residual: x = x5 * 0.2 + x
            for i in 0..nf * hw {
                rdb_out.data[i] = x5.data[i] * 0.2 + x.data[i];
            }
            std::mem::swap(&mut x, &mut rdb_out);
        }
        // RRDB residual: cur = x * 0.2 + rrdb_in
        for i in 0..nf * hw {
            cur.data[i] = x.data[i] * 0.2 + rrdb_in[i];
        }
    }

    dump("05_after_body", &cur);
    progress("conv_body");
    // Global skip: feat = conv_first(x) + conv_body(body(conv_first(x))).
    // The `skip` copy holds conv_first's output; `cur` holds the body's.
    let mut body_feat = Plane::new(nf, cur.h, cur.w);
    conv_to(&cur.data, nf, &mut body_feat, &model.conv_body, false);
    for i in 0..nf * hw {
        cur.data[i] = skip.data[i] + body_feat.data[i];
    }
    dump("06_after_residual", &cur);
    feat = cur;

    progress("upsample");
    for up in 0..2 {
        let u = upsample2x_nearest(&feat);
        let conv = if up == 0 { &model.conv_up1 } else { &model.conv_up2 };
        let mut o = Plane::new(nf, u.h, u.w);
        conv_to(&u.data, u.c, &mut o, conv, true);
        if up == 0 {
            dump("07_up1", &o);
        } else {
            dump("08_up2", &o);
        }
        feat = o;
    }

    progress("head");
    let mut hr = Plane::new(nf, feat.h, feat.w);
    conv_to(&feat.data, nf, &mut hr, &model.conv_hr, true);
    dump("09_hr", &hr);
    let mut out = Plane::new(g.out_ch, hr.h, hr.w);
    conv_to(&hr.data, nf, &mut out, &model.conv_last, false);
    dump("10_out", &out);
    out
}
