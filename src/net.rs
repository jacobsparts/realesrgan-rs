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
//! WEIGHT LAYOUT. The checkpoint stores conv weights as [c_out][c_in][3][3],
//! which is what the CUDA kernel wants; the CPU twin needs a second, ci-innermost
//! copy (see `Conv::w_cpu_ci`), made once at load. Both backends accumulate in
//! (ky, kx, ci) order, which is what makes them agree on these short
//! reductions.

use crate::weights::Weights;
use rayon::prelude::*;

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

/// A single 3x3 convolution's parameters, in the layout each backend wants.
pub struct Conv {
    /// [c_out][c_in][3][3] - the checkpoint's own order (CUDA layout).
    pub w: Vec<f32>,
    /// [c_out][3][3][c_in], the INPUT CHANNEL innermost, for the CPU twin: the
    /// reduction over ci is its inner loop, and here consecutive ci are
    /// consecutive addresses, so each tap's weights are one contiguous run the
    /// vectoriser can load a block at a time.
    ///
    /// The checkpoint's own [c_out][c_in][3][3] (in `w`) makes that same loop
    /// stride by 9 floats per ci, so every weight load lands in a different cache
    /// line and nothing vectorises. Reordering by (ky, kx, ci) for one output
    /// element does not change: only the address of each weight moves.
    pub w_cpu_ci: Vec<f32>,
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
        let mut w_cpu_ci = vec![0.0f32; w.len()];
        for oc in 0..c_out {
            for ci in 0..c_in {
                for k in 0..9 {
                    w_cpu_ci[(oc * 9 + k) * c_in + ci] = w[(oc * c_in + ci) * 9 + k];
                }
            }
        }
        Conv { w: w.to_vec(), w_cpu_ci, bias: bias.to_vec(), c_out }
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
/// `res` adds `scale * res[i]` after the optional activation; `bias + sum`
/// is accumulated in the order (ky, kx, ci) so this matches the CUDA kernel
/// bit-for-bit on the short reductions the network is built from.
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
    // change here can be attributed to a shape instead of guessed at.
    let profiling = std::env::var("LA_PROFILE").map(|v| v == "1").unwrap_or(false);
    let t0 = if profiling { Some(std::time::Instant::now()) } else { None };
    // Output channels are INDEPENDENT - each writes its own plane of `dst` and
    // reads only `src` and its own weight slice - so this parallelises over oc
    // with no synchronisation, and every element still accumulates in (ky, kx,
    // ci) order.
    //
    // The unit of work is a ROW BAND of one channel, not a whole channel plane:
    // there are only c_out planes (32 or 64 here) against 24 hardware threads,
    // which is too coarse to load-balance. `band_rows` must divide `h` so no band
    // spans two channels; when no split exists (h prime, as in the selftest's
    // 37x29 shape) nband is 1 and this is the old whole-plane task.
    //
    // Each task holds a row of TX output pixels in registers. For one tap the
    // weight is a single scalar shared by all TX pixels and the input is a
    // contiguous span of TX+2 floats, so the inner loop is a streaming AXPY that
    // vectorises - unlike the pixel-at-a-time form, whose weight index steps by
    // 9 and whose input index steps by hw.
    //
    // The (ky, kx, ci) accumulation order is load-bearing: the selftest and the
    // stage-dump parity checks are established against it. Preserving it is NOT
    // the same as being bit-identical to the scalar code: the vector path uses a
    // real fused multiply-add while a scalar build emits a separate multiply and
    // add, which is 1 unit out of 255 on 455 of 1,048,576 pixels at this size.
    // The fused form is the more accurate of the two, so the difference is in
    // the right direction, but it is a difference.
    // The tile width, and the ONLY source of truth for both the kernel's N
    // template argument and the accumulator arrays' size. There is no runtime
    // override: the two numbers must agree, and the GPU side has paid for a
    // duplicated number three times.
    const TX: usize = 32;
    let txv: usize = TX;
    let w_ci = &conv.w_cpu_ci[..];
    // Aim for several tasks per hardware thread; more tasks than that only adds
    // scheduling overhead, and the band loop itself is the same either way.
    let nthreads = rayon::current_num_threads().max(1);
    let want = (4 * nthreads).div_ceil(c_out.max(1)).max(1);
    let nband = [8usize, 4, 2, 1]
        .into_iter()
        .find(|&n| h % n == 0 && n >= want)
        .unwrap_or(1);
    let band_rows = h / nband;
    // OUTPUT-CHANNEL BLOCKING. OC adjacent channels are computed in ONE pass, so
    // a loaded input vector feeds OC FMAs instead of one. Three is the measured
    // optimum (end-to-end 256px: OC=1 9.53 s, OC=2 7.12 s, OC=3 6.71 s); OC=4
    // loses because 4 channels x 4 ymm accumulators fill the 16-register file and
    // leave nothing for the weight broadcast.
    //
    // The accumulation order per output element is untouched - each channel still
    // sees the (ky, kx, ci) sequence, merely interleaved with another channel's -
    // so the result matches the OC=1 code, which the stage-dump parity check
    // against torch verifies.
    //
    // LA_OC=1/2 force the other widths, so they can be interleaved in time.
    let oc_group: usize = match std::env::var("LA_OC").ok().and_then(|v| v.parse::<usize>().ok()) {
        Some(1) => 1,
        Some(2) => 2,
        _ => 3,
    };
    // Rows and output channels must both be parallel, and a row band spanning
    // several channels is not contiguous in `dst`, so the task list is built
    // explicitly: one band of one channel group per task. Task count is
    // (c_out/oc_group) * nband - chunking by channel group alone left only 32
    // tasks for a 64-channel conv and cost more in parallelism than the blocking
    // gained. The slices are disjoint by construction, so no unsafe is needed.
    //
    // Each channel plane is MOVED out of `planes` one group at a time, so the
    // borrows going into a task are the only live ones for those planes.
    let mut planes: Vec<&mut [f32]> = dst[..c_out * hw].chunks_mut(hw).collect();
    let mut tasks: Vec<BandTask> = Vec::with_capacity((c_out / oc_group) * nband);
    // One Vec of band slices per band index, across all channels: each plane is
    // taken out of `planes` and split, so the band slices are owned and do not
    // borrow the loop's locals.
    let mut by_band: Vec<Vec<&mut [f32]>> = (0..nband).map(|_| Vec::with_capacity(c_out)).collect();
    for o in 0..c_out {
        let plane = std::mem::take(&mut planes[o]);
        for (bi, chunk) in plane.chunks_mut(band_rows * w).enumerate() {
            by_band[bi].push(chunk);
        }
    }
    drop(planes);
    let mut bands_iter: Vec<std::vec::IntoIter<&mut [f32]>> =
        by_band.into_iter().map(|v| v.into_iter()).collect();
    // A CEILING number of groups, so the last may hold fewer than oc_group
    // channels. A partial group takes the general path, which re-reads the input
    // for its one channel - 1/c_out of the work at the un-blocked rate.
    let ngroups = c_out.div_ceil(oc_group);
    for g in 0..ngroups {
        let oc_base = g * oc_group;
        let n_oc = oc_group.min(c_out - oc_base);
        let mut pending: Vec<(usize, Vec<&mut [f32]>)> = Vec::with_capacity(nband);
        for (bi, it) in bands_iter.iter_mut().enumerate() {
            let mut per: Vec<&mut [f32]> = Vec::with_capacity(n_oc);
            for _ in 0..n_oc {
                per.push(it.next().expect("band has one slice per channel"));
            }
            pending.push((bi, per));
        }
        for (bi, per) in pending {
            tasks.push(BandTask { oc_base, y0: bi * band_rows, per });
        }
    }
    tasks.into_par_iter().for_each(|t| {
        let BandTask { oc_base, y0, mut per } = t;
        let n_oc = per.len();
        let y1 = y0 + band_rows;
        let mut acc = [0f32; TX];
        let mut acc1 = [0f32; TX];
        let mut acc2 = [0f32; TX];
        for y in y0..y1 {
            let yy = y - y0;
            let mut x0 = 0usize;
            while x0 < w {
                let nx = txv.min(w - x0);
                #[cfg(target_arch = "x86_64")]
                let interior = nx == txv
                    && x0 > 0
                    && x0 + txv < w
                    && is_x86_feature_detected!("avx2")
                    && is_x86_feature_detected!("fma");
                #[cfg(not(target_arch = "x86_64"))]
                let interior = false;
                // ---- fused interior tile: `n_oc` channels in one tap pass ----
                #[cfg(target_arch = "x86_64")]
                if interior && n_oc >= 1 {
                    // N is the module's TX, so the kernel's width and the
                    // accumulator arrays' length are the same number by
                    // construction - nothing here can ask for a width the arrays
                    // do not have. The width is a const generic, so the
                    // accumulator sets stay in registers.
                    unsafe {
                        match n_oc {
                            1 => conv3x3_tile_interior_oc_avx2::<{ TX }, 1>(
                                src, w_ci, [conv.bias[oc_base]], c_in, oc_base, hw, h, w, y, x0,
                                &mut [&mut acc[..]],
                            ),
                            2 => conv3x3_tile_interior_oc_avx2::<{ TX }, 2>(
                                src, w_ci, [conv.bias[oc_base], conv.bias[oc_base + 1]], c_in,
                                oc_base, hw, h, w, y, x0,
                                &mut [&mut acc[..], &mut acc1[..]],
                            ),
                            _ => conv3x3_tile_interior_oc_avx2::<{ TX }, 3>(
                                src, w_ci,
                                [conv.bias[oc_base], conv.bias[oc_base + 1], conv.bias[oc_base + 2]],
                                c_in, oc_base, hw, h, w, y, x0,
                                &mut [&mut acc[..], &mut acc1[..], &mut acc2[..]],
                            ),
                        }
                    };
                    let accs = [&acc, &acc1, &acc2];
                    for o in 0..n_oc {
                        let oc = oc_base + o;
                        let row = &mut per[o][yy * w..yy * w + w];
                        for t in 0..nx {
                            let mut v = if act { leaky_relu(accs[o][t]) } else { accs[o][t] };
                            if let Some((r, s)) = res {
                                v += s * r[oc * hw + y * w + x0 + t];
                            }
                            row[x0 + t] = v;
                        }
                    }
                    x0 += nx;
                    continue;
                }
                // ---- general path, one channel at a time: the first and last
                // tile of a row, remainder channels, and non-AVX2 builds ----
                for o in 0..n_oc {
                    let oc = oc_base + o;
                    let b = conv.bias[oc];
                    acc[..nx].fill(b);
                    for ky in 0..3usize {
                        let iy = y_iter(y, ky, h);
                        let iy = match iy {
                            Some(v) => v,
                            None => continue,
                        };
                        for kx in 0..3usize {
                            let k = ky * 3 + kx;
                            let mut lo = 0usize;
                            let mut hi = nx;
                            if kx == 0 && x0 == 0 {
                                lo = 1;
                            }
                            if kx == 2 && x0 + nx == w {
                                hi = nx - 1;
                            }
                            if lo >= hi {
                                continue;
                            }
                            let base = x0 + lo + kx - 1;
                            let wtap = &w_ci[(oc * 9 + k) * c_in..][..c_in];
                            for ci in 0..c_in {
                                let wv = wtap[ci];
                                let span = &src[ci * hw + iy * w + base
                                                ..ci * hw + iy * w + base + (hi - lo)];
                                axpy_span(wv, span, &mut acc[lo..hi]);
                            }
                        }
                    }
                    let row = &mut per[o][yy * w..yy * w + w];
                    for t in 0..nx {
                        let mut v = if act { leaky_relu(acc[t]) } else { acc[t] };
                        if let Some((r, s)) = res {
                            v += s * r[oc * hw + y * w + x0 + t];
                        }
                        row[x0 + t] = v;
                    }
                }
                x0 += nx;
            }
        }
    });
    if let Some(t0) = t0 {
        // Only the calling thread's view is recorded; rayon tasks are joined
        // above, so the elapsed time spans the whole parallel region.
        CPU_PROFILE.lock().unwrap().push((
            c_in,
            c_out,
            h,
            w,
            act,
            t0.elapsed().as_secs_f64(),
        ));
    }
}

/// One parallel unit of the CPU conv: a row band, in every channel of one
/// output-channel group.
struct BandTask<'a> {
    oc_base: usize,
    y0: usize,
    /// One band slice per channel of the group, already offset to row `y0`.
    per: Vec<&'a mut [f32]>,
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

/// Up to three output channels of one interior N-pixel row tile in ONE pass over
/// the taps, so a loaded input vector feeds OC FMAs instead of one.
///
/// NUMERICS: each output element still accumulates in (ky, kx, ci) order, the
/// channels' chains merely interleaved in the instruction stream, so lane j of
/// channel o receives the sequence it would have alone - identical to the
/// one-channel kernel, which the stage-dump parity check against torch verifies.
/// OC is a const generic, so the accumulator sets stay in registers.
///
/// Interior tiles only (every lane valid for every tap); the caller uses the
/// general path for a row's first and last tile and for a partly-filled group.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn conv3x3_tile_interior_oc_avx2<const N: usize, const OC: usize>(
    src: &[f32],
    w_ci: &[f32],
    biases: [f32; OC],
    c_in: usize,
    oc: usize,
    hw: usize,
    h: usize,
    w: usize,
    y: usize,
    x0: usize,
    outs: &mut [&mut [f32]; OC],
) {
    use core::arch::x86_64::*;
    // One accumulator register set per output channel; with OC and N const, this
    // is a fixed register allocation and nothing touches memory in the tap loop.
    let mut a = [[_mm256_set1_ps(0.0); 8]; OC];
    for o in 0..OC {
        let mut q = 0usize;
        while q < N {
            a[o][q / 8] = _mm256_set1_ps(biases[o]);
            q += 8;
        }
    }
    for ky in 0..3usize {
        let iy = y as isize + ky as isize - 1;
        if iy < 0 || iy as usize >= h {
            continue;
        }
        let iy = iy as usize;
        for kx in 0..3usize {
            let k = ky * 3 + kx;
            let base = x0 + kx - 1;
            let mut wt = [core::ptr::null::<f32>(); OC];
            for o in 0..OC {
                wt[o] = w_ci.as_ptr().add(((oc + o) * 9 + k) * c_in);
            }
            for ci in 0..c_in {
                let row = src.get_unchecked(ci * hw + iy * w + base..);
                let mut j = 0usize;
                while j < N {
                    let v = _mm256_loadu_ps(row.as_ptr().add(j));
                    for o in 0..OC {
                        let wv = _mm256_set1_ps(*wt[o].add(ci));
                        a[o][j / 8] = _mm256_fmadd_ps(wv, v, a[o][j / 8]);
                    }
                    j += 8;
                }
            }
        }
    }
    for o in 0..OC {
        let mut j = 0usize;
        while j < N {
            _mm256_storeu_ps(outs[o].as_mut_ptr().add(j), a[o][j / 8]);
            j += 8;
        }
    }
}

/// `acc[j] += w * span[j]` for every j: the general path's inner loop.
///
/// The AVX2 path issues eight independent updates as one FMA, which preserves the
/// per-lane accumulation sequence - lane j accumulates `w * span[j]` exactly where
/// the scalar code would - so no reassociation is introduced.
#[inline]
fn axpy_span(w: f32, span: &[f32], acc: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        // The feature test is hoisted by the compiler into a cfence-guarded
        // branch; is_x86_feature_detected caches its result after the first call.
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe { axpy_span_avx2(w, span, acc) };
            return;
        }
    }
    for (a, s) in acc.iter_mut().zip(span.iter()) {
        *a += w * s;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn axpy_span_avx2(w: f32, span: &[f32], acc: &mut [f32]) {
    use core::arch::x86_64::*;
    debug_assert_eq!(span.len(), acc.len());
    let n = span.len();
    let wv = _mm256_set1_ps(w);
    let mut j = 0usize;
    while j + 8 <= n {
        let v = _mm256_loadu_ps(span.as_ptr().add(j));
        let a = _mm256_loadu_ps(acc.as_ptr().add(j));
        _mm256_storeu_ps(acc.as_mut_ptr().add(j), _mm256_fmadd_ps(wv, v, a));
        j += 8;
    }
    // Always evaluate the remainder, even when it is empty: n is a runtime value
    // that the compiler cannot prove is a multiple of 8, and the bounds are
    // re-checked here rather than assumed.
    while j < n {
        *acc.get_unchecked_mut(j) += w * *span.get_unchecked(j);
        j += 1;
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
