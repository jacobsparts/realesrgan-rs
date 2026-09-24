//! The CUDA forward pass.
//!
//! This mirrors `net::forward_cpu` stage for stage, so the two can be compared
//! op by op (`--cuda-selftest` does exactly that on random data).
//!
//! THERE ARE TWO 3x3 CONV IMPLEMENTATIONS, AND THE TABLE PICKS.
//!
//!   1. `lg_conv3x3_winograd` - the toolkit's F(4x4,3x3) Winograd op, promoted
//!      out of this repository, and the DEFAULT for every shape the table covers.
//!   2. `lg_conv3x3_res` - one un-templated direct conv, the fallback for a
//!      non-3x3/non-stride-1 conv and for a tensor too small for a Winograd CTA.
//!      It is a correctness fallback, not a performance one.
//!
//! `ConvShape` is the ONE TABLE both paths are launched from - the kernel name,
//! the geometry (grid/block/shared), the channel-blocking numbers and the
//! ARGUMENT LIST - and the selftest reads it through `by_index`, so the shape
//! the network launches is the shape that gets checked.
//!
//! THREE STRUCTURAL THINGS ARE WORTH NAMING.
//!
//! 1. THE DENSE WORKSPACE IS ONE BUFFER. Every RDB writes its 32-channel result
//!    into its own slot of a single (num_feat + 4*num_grow_ch)-channel buffer,
//!    and that buffer IS the concatenation the next conv reads. The kernels take
//!    a base pointer plus a channel count, so "the concatenation" costs nothing
//!    and the workspace is allocated once for the whole network.
//!
//! 2. ALL WEIGHTS LIVE IN ONE UPLOAD. The 350 convs are concatenated into one
//!    device buffer in a fixed order; a launch addresses its slice by a byte
//!    offset.
//!
//! 3. A CONV'S READ AND WRITE RANGES MUST NOT OVERLAP. Inside a dense block the
//!    destination slot lies *beyond* the read prefix in the same buffer, which
//!    is what makes the zero-copy concatenation safe: conv1 reads channels
//!    [0,64) and writes [64,96), conv2 reads [0,96) and writes [96,128), and so
//!    on. `conv` therefore takes a channel offset for each side, and the dense
//!    block's call sites are the only place those offsets are non-zero.

use crate::cuda::{Cuda, DevBuf};
use crate::net::{Conv, Model};
use lightgpu::vm::Launch;


/// The launch geometry of one conv kernel: its name, its tile/blocking numbers,
/// its per-channel shared-memory cost and its ARGUMENT LIST.
///
/// The network path and the selftest both read THIS struct, so the shape a
/// kernel is checked with is the shape it is launched with. A host/device
/// disagreement about tile shape, grid or argument order is silence, not an
/// error - the kernel happily computes the tiles it is given.
#[derive(Clone, Copy)]
struct ConvShape {
    name: &'static str,
    /// Tile height in output pixels per CTA along y.
    ty: u32,
    /// Output columns a thread accumulates along x.
    cx: u32,
    /// Threads in x. Must equal the kernel's TX template argument.
    tx: u32,
    /// Shared-memory buffers the kernel stages into. Both rows use 1; the field
    /// is part of the smem cap arithmetic, so a double-buffered kernel declares
    /// its second buffer HERE rather than at the launch site.
    buffers: u32,
    /// Floats of shared memory per channel of the chunk, PER BUFFER. The tiled
    /// family stages a (span+2) x (ty+2) patch; Winograd stages a transformed
    /// tile block. Kept here so the cap in `chunk_and_smem` and the launch size
    /// cannot disagree.
    smem_per_chunk: u32,
    /// Output channels per CTA, passed as a RUNTIME ARGUMENT to the Winograd op.
    /// A host/device disagreement here leaves output channels unwritten with no
    /// error reported.
    ocb: u32,
    /// Output CHANNELS one CTA covers, which is what grid.z divides by. A field
    /// rather than `ty * oc` because the two families do not relate them the
    /// same way: the Winograd op's ty is the tile block and its oc per thread is
    /// 1, so the product is not its channel coverage.
    oc_per_cta: u32,
}

impl ConvShape {
    /// A tile-parameterised shape: the shared cost per channel is the staged
    /// (span+2) x (ty+2) patch.
    fn tiled(name: &'static str, ty: u32, cx: u32, tx: u32, buffers: u32) -> ConvShape {
        let span = tx * cx;
        ConvShape {
            name, ty, cx, tx, buffers,
            smem_per_chunk: (span + 2) * (ty + 2),
            // This family's kernel has no runtime ocb argument; the field is kept
            // equal to oc_per_cta so the launch has one number to read.
            ocb: ty,
            oc_per_cta: ty,
        }
    }

    /// THE WINOGRAD ROW - the default, and the only tabled 3x3 with a twelve-
    /// argument list. Written out rather than derived, because this family stages
    /// the TRANSFORMED tile block (16 tiles x 36 values per channel) and its CTA
    /// covers 4x4 tiles = 16x16 OUTPUT pixels, neither of which the tiled formula
    /// produces.
    fn winograd() -> ConvShape {
        ConvShape {
            name: "lg_conv3x3_winograd",
            ty: 16, cx: 4, tx: 16, buffers: 1,
            // TWO 36-value arrays per channel of the chunk, not one: the
            // transformed INPUT (16 tiles x 36) and the transformed WEIGHTS
            // (16 oc x 36). The per-channel cost assumes ocb, so ocb and this
            // field must change together.
            smem_per_chunk: 2 * 16 * 36,
            // 256 threads = 16 tiles x 16 output channels.
            oc_per_cta: 16,
            // OCB is a RUNTIME argument of the toolkit op; this is the number the
            // launch passes.
            ocb: 16,
        }
    }

    /// Is this the toolkit's Winograd op? It is the one row that takes the
    /// twelve-argument list and the 256-thread block, so this predicate decides
    /// both. A NAME EQUALITY, not a search for "winograd": a substring test is
    /// the kind of host/device disagreement this table exists to prevent.
    fn is_toolkit_winograd(&self) -> bool {
        self.name == "lg_conv3x3_winograd"
    }

    /// Output pixels a CTA covers along x: one thread's `cx` columns times the
    /// threads in x. This is the WIDTH the grid must divide by, and it is derived
    /// from `tx * cx` rather than stored because the two numbers are already in
    /// the table in the form a reader checks them against the kernel's template
    /// arguments.
    fn span(&self) -> u32 {
        self.tx * self.cx
    }

    /// Shared memory the kernel may use, and the channel chunk that fits. The
    /// cap is 48 KB, which is a MEASURED limit on this card rather than a nominal
    /// one: a larger request dies with CUDA_ERROR_INVALID_VALUE even though the
    /// driver accepts the opt-in attribute. `requested` (LA_CCHUNK) is a
    /// preference and is clamped, never trusted - a chunk larger than the staging
    /// can hold stages past the buffer and corrupts the reduction silently.
    ///
    /// `buffers` is multiplied in because a double-buffered kernel stages TWO
    /// copies per channel; both rows here use 1.
    fn chunk_and_smem(&self, requested: i32) -> (i32, u32) {
        let per_chunk = self.smem_per_chunk * 4;
        let cap: i32 = ((48 * 1024) / (per_chunk * self.buffers)) as i32;
        let c_chunk = requested.min(cap).max(1);
        (c_chunk, (c_chunk as u32) * per_chunk * self.buffers)
    }

    /// The CTA's block shape. The Winograd op is a fixed 256 threads (16x16
    /// tile x 16 channels), NOT `tx * ty`; every other row is a (tx, ty) tile of
    /// threads, one per pixel tile.
    fn block(&self) -> (u32, u32, u32) {
        if self.is_toolkit_winograd() {
            (256, 1, 1)
        } else {
            (self.tx, self.ty, 1)
        }
    }

    /// The full launch geometry: the grid from the CTA's output coverage, the
    /// block from `block()`, and the channel blocking from `oc_per_cta`. One
    /// method so the network path, the bench and the selftest cannot disagree.
    fn launch(&self, w: usize, h: usize, c_out: usize) -> Launch {
        let (cw, ch) = self.cta_out();
        let (bx, by, _) = self.block();
        Launch::new(
            (
                (w as u32).div_ceil(cw),
                (h as u32).div_ceil(ch),
                (c_out as u32).div_ceil(self.oc_per_cta),
            ),
            (bx, by, 1),
        )
    }

    /// The command-line name of a row, for LA_CONV. `None` for anything else, so
    /// an unrecognised variant is a hard error at the call site.
    fn by_name(name: &str) -> Option<ConvShape> {
        match name {
            "wg" => Some(ConvShape::winograd()),
            "direct" => Some(ConvShape::tiled("lg_conv3x3_res", 8, 1, 32, 1)),
            _ => None,
        }
    }

    /// The row a selftest variant number refers to. The numbers are historical, so
    /// printed row labels stay stable. A row exists only if the selftest's list
    /// drives it.
    fn by_index(v: usize) -> ConvShape {
        match v {
            // The direct conv fallback: one thread per output pixel, no staging.
            0 => ConvShape::tiled("lg_conv3x3_res", 8, 1, 32, 1),
            // The toolkit's F(4x4,3x3) Winograd op - the DEFAULT.
            _ => ConvShape::winograd(),
        }
    }

    /// The kernel's ARGUMENT LIST, as data in the table's terms: the third thing
    /// the table owns, beside the name and the geometry. The two rows take
    /// different lists - twelve arguments against nine - and a mismatch at the
    /// driver boundary is a wrong number in the wrong slot, not a compile error.
    /// `rp`/`rs` are accepted and ignored: neither row has a residual operand.
    fn args(
        &self,
        sp: u64, wp: u64, bp: u64, rp: u64, dp: u64,
        c_in: usize, c_out: usize, h: usize, w: usize,
        act: bool, rs: f32, c_chunk: i32,
    ) -> lightgpu::vm::Args {
        let mut a = lightgpu::vm::Args::new();
        if self.is_toolkit_winograd() {
            // act codes: 0 none, 1 relu, 2 lrelu. This network only asks for
            // lrelu (0.2).
            let code: i32 = if act { 2 } else { 0 };
            a.ptr(sp).ptr(wp).ptr(bp).ptr(dp)
                .i32(c_in as i32).i32(c_out as i32).i32(h as i32).i32(w as i32)
                .i32(c_chunk).i32(self.ocb as i32)
                .i32(code).f32(0.2);
        } else {
            // The fallback's list: (in, w, bias, out, c_in, c_out, h, wd, act) -
            // no residual operand and no c_chunk, because it is un-templated.
            let _ = (rp, rs, c_chunk);
            a.ptr(sp).ptr(wp).ptr(bp).ptr(dp)
                .i32(c_in as i32).i32(c_out as i32).i32(h as i32).i32(w as i32)
                .i32(act as i32);
        }
        a
    }

    /// Output pixels one CTA covers in x and y. Winograd is flagged by its name
    /// because its transform size, not its tile geometry, sets this.
    fn cta_out(&self) -> (u32, u32) {
        if self.is_toolkit_winograd() {
            // 4 pixels per output tile x a 4x4 block of tiles.
            (4 * self.cx, 4 * self.cx)
        } else {
            (self.span(), self.ty)
        }
    }
}

/// Time the FUSED Winograd op on the network's real conv shapes, using the SAME
/// shape table, the same argument builder and the same grid/block/shared
/// arithmetic the production launch uses.
///
/// This exists to give `--bench-gemm` something measured the same way to compare
/// against. A GEMM row on its own cannot answer whether a GEMM-fed Winograd
/// would be faster: the GEMM route also owes an input transform, a weight
/// transform, an inverse transform and 36 launches per conv, none of which the
/// fused kernel pays, so the comparison has to be between TIMES on identical
/// shapes and identical instruments - which is what this is.
///
/// The GFLOP/s printed is DIRECT-EQUIVALENT (2*c_out*c_in*9*pixels), the same
/// denominator `--bench-gemm` uses, so the two are comparable - but it
/// overstates the real rate by 4x, because F(4x4,3x3) performs 36 multiplies per
/// 4x4x3x3 = 144-result tile. Read the TIME and the utilisation, not the rate.
pub fn bench_fused_conv(cuda: &Cuda) {
    println!("\n  -- the FUSED Winograd op, same instrument (direct-equivalent GFLOP) --");
    // (c_in, c_out, h, w): the dense body at 256px, the network's first conv
    // (the c_in=3 case the OPS entry warns about), and the 1024x1024 stage.
    for &(c_in, c_out, h, w) in &[
        (64usize, 64usize, 256usize, 256usize),
        (3usize, 64usize, 256usize, 256usize),
        (64, 64, 1024, 1024),
        // Equal multiply work (c_in*c_out = 1024 per tile) but different staging
        // work: c_out=16 means grid.z=1, so the transformed input is built once
        // per tile, while c_out=64 has four z-CTAs rebuilding it.
        (64, 16, 256, 256),
        (16, 64, 256, 256),
        (64, 32, 256, 256),
        (32, 64, 256, 256),
        // The shape the network spends nearly all its time on: 276 of its 349
        // convs run at 16x16, where the kernel's 16x16 CTA is exact.
        (64, 64, 16, 16),
        (64, 32, 16, 16),
        (16, 16, 16, 16),
        // and the 64x64 pair, so the 16x16 and the 64x64 CTA cases are bracketed
        // at one channel shape rather than compared across a channel change.
        (64, 64, 64, 64),
    ] {
        let hw = h * w;
        // The weights and bias are one buffer laid out as the network lays it
        // out: all `c_out*c_in*9` weight floats, then the bias. The offset
        // arithmetic below is the same as `Layout`'s.
        let mut cat = vec![0.01f32; c_out * c_in * 9];
        cat.extend(std::iter::repeat(0.0f32).take(c_out));
        let dw = match DevBuf::from_host(&cat) {
            Ok(b) => b,
            Err(e) => { println!("  ({e})"); continue; }
        };
        let din = vec![0.01f32; c_in * hw];
        let src = match DevBuf::from_host(&din) {
            Ok(b) => b,
            Err(e) => { println!("  ({e})"); continue; }
        };
        let dst = match DevBuf::alloc(c_out * hw) {
            Ok(b) => b,
            Err(e) => { println!("  ({e})"); continue; }
        };
        // The production launch, from the production table: the same row the
        // network resolves from LA_CONV, the same chunk arithmetic, the same
        // argument builder.
        // THE SHAPE COMES FROM LA_CONV, like the production path, so this bench
        // can time a VARIANT (the occupancy rows) as well as the default. It used
        // to hardcode ConvShape::winograd(), which meant a new row in the table
        // could not be measured by the instrument built to measure it.
        let shape = match std::env::var("LA_CONV").ok().and_then(|v| ConvShape::by_name(&v)) {
            Some(s) => s,
            None => ConvShape::winograd(),
        };
        let (c_chunk, smem) = shape.chunk_and_smem(16);
        let l = shape.launch(w, h, c_out).shared(smem);
        // ONE launch of the row, written out where it is timed rather than in a
        // closure, and `Args` rebuilt per launch because it is a list of argument
        // slots with no Clone - as every other launch site in this file does. The
        // table is the thing that is shared, not the marshalled buffer.
        let one = || -> Result<(), String> {
            let args = shape.args(
                src.ptr,
                dw.ptr,
                dw.ptr + (c_out * c_in * 9 * 4) as u64,
                0u64,
                dst.ptr,
                c_in, c_out, h, w, false, 0.0, c_chunk,
            );
            cuda.launch_raw(shape.name, l, args)?;
            cuda.sync()
        };
        if one().is_err() {
            println!("  (launch failed for {c_in}->{c_out} {h}x{w})");
            continue;
        }
        let mut best = f64::INFINITY;
        let mut ok = true;
        for _ in 0..7 {
            let t = std::time::Instant::now();
            if one().is_err() {
                ok = false;
                break;
            }
            let dt = t.elapsed().as_secs_f64();
            if dt < best {
                best = dt;
            }
        }
        if !ok {
            println!("  (launch failed for {c_in}->{c_out} {h}x{w})");
            continue;
        }
        let flops = 2.0 * (c_in * c_out * 9 * hw) as f64;
        // The denominator is the MEASURED fp32 FFMA ceiling of this card, not the
        // nominal 20 SMs x 128 lanes x 2: quoting against a peak the hardware does
        // not reach makes every kernel look worse than the machine allows.
        const FMA_PEAK: f64 = 7653.0;
        println!(
            "  {:<10} {c_in:>3}->{c_out:<3} {h:>4}x{w:<4}: best {:.6}s ({:.1} us), {:.1} direct-GFLOP/s = {:.1}%",
            shape.name, best, best * 1e6, flops / best / 1e9, 100.0 * (flops / best) / 1e9 / FMA_PEAK
        );
    }
}

/// Element offset of a conv's weights and biases within the weight buffer.
struct ConvLoc {
    w: usize,
    b: usize,
}

pub struct Layout {
    conv_first: ConvLoc,
    /// [num_block][3][5]
    blocks: Vec<Vec<Vec<ConvLoc>>>,
    conv_body: ConvLoc,
    conv_up1: ConvLoc,
    conv_up2: ConvLoc,
    conv_hr: ConvLoc,
    conv_last: ConvLoc,
    pub floats: usize,
}

impl Layout {
    fn new(model: &Model) -> Layout {
        let mut off = 0usize;
        let mut place = |c: &Conv| -> ConvLoc {
            let w = off;
            off += c.w.len();
            let b = off;
            off += c.bias.len();
            ConvLoc { w, b }
        };
        let conv_first = place(&model.conv_first);
        let mut blocks = Vec::new();
        for blk in &model.blocks {
            let mut row = Vec::new();
            for r in &blk.rdbs {
                let mut five = Vec::new();
                for c in &r.convs {
                    five.push(place(c));
                }
                row.push(five);
            }
            blocks.push(row);
        }
        let conv_body = place(&model.conv_body);
        let conv_up1 = place(&model.conv_up1);
        let conv_up2 = place(&model.conv_up2);
        let conv_hr = place(&model.conv_hr);
        let conv_last = place(&model.conv_last);
        Layout { conv_first, blocks, conv_body, conv_up1, conv_up2, conv_hr, conv_last, floats: off }
    }
}

/// The activation planes, one allocation each, sized for the largest stage.
struct Acts {
    /// (num_feat + 4*num_grow_ch) channels at base area: the dense workspace.
    ws: DevBuf,
    /// num_feat channels at base area.
    cur: DevBuf,
    x5: DevBuf,
    rdb: DevBuf,
    /// The RRDB block input, saved for the block-level residual.
    save: DevBuf,
    /// The conv_first output, saved for the network's global skip connection.
    skip: DevBuf,
    body: DevBuf,
    /// num_feat channels at 4x base area: the first upsample's output.
    up1: DevBuf,
    /// num_feat channels at 16x base area: the second output and the head plane.
    up2: DevBuf,
    hr: DevBuf,
    out: DevBuf,
    /// first_in_ch channels at base area: after the x2 model's unshuffle.
    feat: DevBuf,
}

impl Acts {
    /// Allocate every activation plane for an input of `w x h`.
    ///
    /// This is the only size-dependent part of a `Gpu`: the weights, the module
    /// handles and the model are size-independent, so tiling reuses one `Gpu`
    /// and only re-runs this.
    fn alloc(g: &crate::net::Geometry, w: usize, h: usize) -> Result<Acts, String> {
        let (net_w, net_h) = if g.scale == 2 { (w / 2, h / 2) } else { (w, h) };
        let base = net_w * net_h;
        let nf = g.num_feat;
        Ok(Acts {
            ws: DevBuf::alloc(g.dense_width() * base)?,
            cur: DevBuf::alloc(nf * base)?,
            x5: DevBuf::alloc(nf * base)?,
            rdb: DevBuf::alloc(nf * base)?,
            save: DevBuf::alloc(nf * base)?,
            skip: DevBuf::alloc(nf * base)?,
            body: DevBuf::alloc(nf * base)?,
            up1: DevBuf::zeros(nf * 4 * base)?,
            up2: DevBuf::zeros(nf * 16 * base)?,
            hr: DevBuf::alloc(nf * 16 * base)?,
            out: DevBuf::alloc(g.out_ch * 16 * base)?,
            feat: DevBuf::alloc(g.first_in_ch() * base)?,
        })
    }
}

pub struct Gpu {
    pub cuda: Cuda,
    pub model: Model,
    loc: Layout,
    wbuf: DevBuf,
    acts: Acts,
    pub base_w: usize,
    pub base_h: usize,
    /// The network's working resolution (post-unshuffle: half for the x2 model).
    pub net_w: usize,
    pub net_h: usize,
}

impl Gpu {
    pub fn new(cuda: Cuda, model: Model, w: usize, h: usize) -> Result<Gpu, String> {
        let g = model.geom.clone();
        let loc = Layout::new(&model);
        let wbuf = DevBuf::alloc(loc.floats)?;

        let mut host = vec![0f32; loc.floats];
        {
            let mut put = |l: &ConvLoc, c: &Conv| {
                host[l.w..l.w + c.w.len()].copy_from_slice(&c.w);
                host[l.b..l.b + c.bias.len()].copy_from_slice(&c.bias);
            };
            put(&loc.conv_first, &model.conv_first);
            for (bi, blk) in model.blocks.iter().enumerate() {
                for (ri, r) in blk.rdbs.iter().enumerate() {
                    for (ci, c) in r.convs.iter().enumerate() {
                        put(&loc.blocks[bi][ri][ci], c);
                    }
                }
            }
            put(&loc.conv_body, &model.conv_body);
            put(&loc.conv_up1, &model.conv_up1);
            put(&loc.conv_up2, &model.conv_up2);
            put(&loc.conv_hr, &model.conv_hr);
            put(&loc.conv_last, &model.conv_last);
        }
        wbuf.upload(&host)?;

        let acts = Acts::alloc(&g, w, h)?;
        let (net_w, net_h) = if g.scale == 2 { (w / 2, h / 2) } else { (w, h) };

        Ok(Gpu {
            cuda, model, loc, wbuf, acts,
            base_w: w, base_h: h, net_w, net_h,
        })
    }

    /// Re-point the activation arena at a new input size, reusing the loaded
    /// weights and modules. This is what makes tiling cheap: only the buffers
    /// below the weights are rebuilt.
    pub fn resize(&mut self, w: usize, h: usize) -> Result<(), String> {
        let g = self.model.geom.clone();
        self.acts = Acts::alloc(&g, w, h)?;
        self.base_w = w;
        self.base_h = h;
        let (nw, nh) = if g.scale == 2 { (w / 2, h / 2) } else { (w, h) };
        self.net_w = nw;
        self.net_h = nh;
        Ok(())
    }

    #[inline]
    fn at(buf: &DevBuf, ch: usize, area: usize) -> u64 {
        buf.ptr + (ch * area * 4) as u64
    }

    /// The fused 3x3 conv. `src_ch`/`dst_ch` are channel offsets: the kernel
    /// sees `src.ptr + src_ch*area` as its input plane and writes
    /// `dst.ptr + dst_ch*area` as its output plane, so a call can read a prefix
    /// of the dense workspace and write a later slot of the same buffer.
    ///
    /// The two ranges must be disjoint; the dense block is the only caller that
    /// relies on this and it walks the offsets forward.
    #[allow(clippy::too_many_arguments)]
    fn conv(
        &self,
        loc: &ConvLoc,
        src: &DevBuf,
        src_ch: usize,
        dst: &DevBuf,
        dst_ch: usize,
        c_in: usize,
        c_out: usize,
        w: usize,
        h: usize,
        act: bool,
        res: Option<(&DevBuf, usize, f32)>,
    ) -> Result<(), String> {
        let area = w * h;
        let wb = self.wbuf.ptr;
        let wp = wb + (loc.w * 4) as u64;
        let bp = wb + (loc.b * 4) as u64;
        let sp = Self::at(src, src_ch, area);
        let dp = Self::at(dst, dst_ch, area);
        let (rp, rs) = match res {
            Some((b, ch, s)) => (Self::at(b, ch, area), s),
            None => (0u64, 0f32),
        };

        // The shape comes from the ONE table below - grid, block, shared size,
        // channel chunk and argument list - and the selftest reads the SAME table
        // through `by_index`, so the shape a kernel is launched with cannot differ
        // between the two paths. An unrecognised LA_CONV is a HARD ERROR, not a
        // silent fallback to a slower row.
        let variant = std::env::var("LA_CONV").unwrap_or_else(|_| "wg".into());
        let tpl: Option<ConvShape> = ConvShape::by_name(&variant);
        if tpl.is_none() {
            return Err(format!(
                "unknown LA_CONV `{variant}` (see ConvShape::by_name: \"wg\" or \"direct\")"
            ));
        }
        if let Some(shape) = tpl {
            if w >= 16 && h >= 8 {
                let requested: i32 = std::env::var("LA_CCHUNK")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(16);
                let (c_chunk, smem) = shape.chunk_and_smem(requested);
                let l = shape.launch(w, h, c_out).shared(smem);
                // The argument list comes from the table too - the two rows take
                // different lists (twelve arguments against nine), and a mismatch
                // at the driver boundary is a wrong number in the wrong slot, not
                // a compile error.
                let args = shape.args(sp, wp, bp, rp, dp, c_in, c_out, h, w, act, rs, c_chunk);
                return self.cuda.launch_raw(shape.name, l, args);
            }
        }
        // The shapes the TABLE does not cover: a non-3x3 conv, or w < 16 or
        // h < 8, where the Winograd op cannot be used.
        const TX: u32 = 32;
        const TY: u32 = 8;
        let l = Launch::new(
            (((w as u32 + TX - 1) / TX).max(1), ((h as u32 + TY - 1) / TY).max(1), 1),
            (TX, TY, 1),
        );
        self.cuda.run("lg_conv3x3_res", l, |a| {
            a.ptr(sp).ptr(wp).ptr(bp).ptr(dp)
                .i32(c_in as i32).i32(c_out as i32).i32(h as i32).i32(w as i32)
                .i32(act as i32);
        })
    }

    /// dst = a + scale * b over `n` elements.
    fn add_scaled(&self, a: &DevBuf, b: &DevBuf, dst: &DevBuf, n: usize, s: f32) -> Result<(), String> {
        // The toolkit kernel takes its length as `long`; marshalling it as i32
        // would truncate a plane count past 2^31 and, worse, read the scale
        // argument from the wrong slot.
        self.cuda.run_n("lg_add_scaled", n, |x| {
            x.ptr(a.ptr).ptr(b.ptr).ptr(dst.ptr).i64(n as i64).f32(s);
        })
    }

    /// dst = src elementwise (device-side copy; the toolkit's lg_copy).
    fn copy(&self, src: &DevBuf, src_ch: usize, dst: &DevBuf, dst_ch: usize, area: usize, n: usize) -> Result<(), String> {
        let sp = Self::at(src, src_ch, area);
        let dp = Self::at(dst, dst_ch, area);
        self.cuda.run_n("lg_copy", n, |a| {
            a.ptr(sp).ptr(dp).i64(n as i64);
        })
    }

    fn upsample(&self, src: &DevBuf, dst: &DevBuf, c: usize, w: usize, h: usize) -> Result<(), String> {
        const TX: u32 = 32;
        const TY: u32 = 8;
        let (ow, oh) = (w * 2, h * 2);
        let l = Launch::new(
            (((ow as u32 + TX - 1) / TX).max(1), ((oh as u32 + TY - 1) / TY).max(1), 1),
            (TX, TY, 1),
        );
        self.cuda.run("lg_upsample2x_nearest", l, |a| {
            a.ptr(src.ptr).ptr(dst.ptr).i32(c as i32).i32(h as i32).i32(w as i32);
        })
    }

    /// Upload RGB planar f32 at base resolution and run the whole network.
    /// Returns the planar output at `scale`x.
    pub fn forward(&self, input: &[f32]) -> Result<Vec<f32>, String> {
        let g = self.model.geom.clone();
        let nf = g.num_feat;
        let ng = g.num_grow_ch;
        let (nw, nh) = (self.net_w, self.net_h);
        let base = nw * nh;

        // The x2 model begins with pixel_unshuffle, whose output is 4x the
        // channel count: the source and destination cannot share a buffer
        // because the kernel reads (2y+dy, 2x+dx) while writing (y, x) of a
        // different plane set. `feat` is the 12-channel destination and a
        // same-sized staging buffer carries the 3-channel input.
        if g.scale == 2 {
            // The unshuffle READS a full-resolution plane and WRITES the
            // half-resolution 4x-channel plane, so the staging buffer is
            // sized by the input dimensions (base_w x base_h), not by the
            // network's working resolution.
            let st = DevBuf::alloc(g.in_ch * self.base_w * self.base_h)?;
            st.upload(input)?;
            // The kernel is given the INPUT plane's dimensions and derives the
            // output (h/2, w/2) itself.
            self.cuda.run(
                "lg_pixel_unshuffle2",
                Launch::new(
                    (((nw as u32 + 31) / 32).max(1), ((nh as u32 + 7) / 8).max(1), 1),
                    (32, 8, 1),
                ),
                |a| {
                    a.ptr(st.ptr).ptr(self.acts.feat.ptr)
                        .i32(g.in_ch as i32)
                        .i32(self.base_h as i32)
                        .i32(self.base_w as i32);
                },
            )?;
        } else {
            self.acts.feat.upload(input)?;
        }

        // conv_first: reads first_in_ch channels, writes nf.
        self.conv(&self.loc.conv_first, &self.acts.feat, 0, &self.acts.cur, 0,
                  g.first_in_ch(), nf, nw, nh, false, None)?;
        // conv_first's output is the target of the global skip connection, so
        // it is preserved before the body overwrites `cur`.
        self.copy(&self.acts.cur, 0, &self.acts.skip, 0, base, nf * base)?;

        // Body: 23 RRDB blocks.
        for bi in 0..g.num_block {
            // The RRDB's own residual needs its input, so it is saved first.
            self.copy(&self.acts.cur, 0, &self.acts.save, 0, base, nf * base)?;
            for ri in 0..3 {
                let row = &self.loc.blocks[bi][ri];
                // ws[0..nf] = cur (the concatenation's first term).
                self.copy(&self.acts.cur, 0, &self.acts.ws, 0, base, nf * base)?;
                // Four grow-channel convs, each reading the prefix so far and
                // writing the NEXT slot.
                self.conv(&row[0], &self.acts.ws, 0, &self.acts.ws, nf, nf, ng, nw, nh, true, None)?;
                self.conv(&row[1], &self.acts.ws, 0, &self.acts.ws, nf + ng, nf + ng, ng, nw, nh, true, None)?;
                self.conv(&row[2], &self.acts.ws, 0, &self.acts.ws, nf + 2 * ng, nf + 2 * ng, ng, nw, nh, true, None)?;
                self.conv(&row[3], &self.acts.ws, 0, &self.acts.ws, nf + 3 * ng, nf + 3 * ng, ng, nw, nh, true, None)?;
                // conv5 reads the whole workspace, writes to x5 with no activation.
                self.conv(&row[4], &self.acts.ws, 0, &self.acts.x5, 0, nf + 4 * ng, nf, nw, nh, false, None)?;
                // RDB residual: cur = x5 * 0.2 + cur
                self.add_scaled(&self.acts.cur, &self.acts.x5, &self.acts.rdb, nf * base, 0.2)?;
                self.copy(&self.acts.rdb, 0, &self.acts.cur, 0, base, nf * base)?;
            }
            // RRDB residual: cur = cur * 0.2 + save. The loop above left the
            // third RDB's output in cur, and `save` is this block's input.
            //
            // The residual is (input + 0.2 * rdb3_out), and add_scaled computes
            // a + s*b, so the terms are swapped relative to the naming: pass
            // save as `a` and cur as `b` with s = 0.2.
            self.add_scaled(&self.acts.save, &self.acts.cur, &self.acts.rdb, nf * base, 0.2)?;
            self.copy(&self.acts.rdb, 0, &self.acts.cur, 0, base, nf * base)?;
        }

        // conv_body + the global residual: cur = skip + conv_body(body).
        self.conv(&self.loc.conv_body, &self.acts.cur, 0, &self.acts.body, 0, nf, nf, nw, nh, false, None)?;
        self.add_scaled(&self.acts.skip, &self.acts.body, &self.acts.cur, nf * base, 1.0)?;

        // Upsampling: two 2x stages, each nearest-2x then conv + lrelu.
        let mut w = nw;
        let mut h = nh;
        self.upsample(&self.acts.cur, &self.acts.up1, nf, w, h)?;
        w *= 2;
        h *= 2;
        self.conv(&self.loc.conv_up1, &self.acts.up1, 0, &self.acts.hr, 0, nf, nf, w, h, true, None)?;

        self.upsample(&self.acts.hr, &self.acts.up2, nf, w, h)?;
        w *= 2;
        h *= 2;
        // conv_up2 writes into hr (which the head then uses).
        self.conv(&self.loc.conv_up2, &self.acts.up2, 0, &self.acts.hr, 0, nf, nf, w, h, true, None)?;

        // Head: conv_hr (act) then conv_last.
        self.conv(&self.loc.conv_hr, &self.acts.hr, 0, &self.acts.up2, 0, nf, nf, w, h, true, None)?;
        self.conv(&self.loc.conv_last, &self.acts.up2, 0, &self.acts.out, 0, nf, g.out_ch, w, h, false, None)?;

        self.cuda.sync()?;
        let mut out = vec![0f32; g.out_ch * w * h];
        self.acts.out.download(&mut out)?;
        Ok(out)
    }

    /// Compare each project kernel against its CPU twin on random data.
    ///
    /// The elementwise and layout kernels must agree EXACTLY (they are the same
    /// arithmetic in the same order). The convolution's two implementations sum
    /// in the same (ky, kx, ci) order, but FMA contraction and the different
    /// block shapes still move the last bits, so it is compared with a
    /// tolerance scaled by sqrt(reduction length) - the same shape of bound the
    /// other engines in this tree use.
    pub fn selftest(&self) -> Result<(usize, Vec<String>), String> {
        use crate::net::{self, Plane};
        let mut compared = 0usize;
        let mut bad: Vec<String> = Vec::new();
        let (w, h) = (37usize, 29usize); // deliberately not tile multiples
        let base = w * h;
        let rng = |n: usize, seed: u32| -> Vec<f32> {
            let mut s = seed.wrapping_mul(2654435761).wrapping_add(1);
            (0..n)
                .map(|_| {
                    s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                    ((s >> 8) as f32 / 16777216.0) - 0.5
                })
                .collect()
        };

        // ---- the 3x3 conv, both rows ----
        // Each row is (c_in, c_out, act, variant) and the variant indexes the
        // table above, which supplies the kernel name and its EXACT geometry. The
        // shape is deliberately not re-derived from the variant number: two
        // copies of that arithmetic is how the selftest and the network path came
        // to disagree about the shared size.
        //
        //   0  = the direct conv fallback
        //   14 = the toolkit's F(4x4,3x3) Winograd op, the default
        //
        // Every row agrees with the SAME CPU twin, and 37x29 is a multiple of
        // neither tile, so the border and remainder paths are exercised.
        for &(c_in, c_out, act, variant) in &[
            // The direct fallback, including the 3-channel case it exists for.
            (3usize, 64usize, false, 0usize),
            (64, 32, true, 0),
            (96, 32, true, 0),
            (192, 64, false, 0),
            (64, 64, true, 0),
            (160, 32, true, 0),
            // The Winograd default, on the shapes the network launches. This is
            // a TOLERANCE check rather than a bit-identity one: the transforms
            // are exact in exact arithmetic but ill conditioned in fp32.
            (3, 64, false, 14),
            (64, 32, true, 14),
            (96, 32, true, 14),
            (192, 64, false, 14),
            (64, 64, true, 14),
            (160, 32, true, 14),
            (128, 32, true, 14),
            (32, 128, true, 14),
            // Small shapes: c_in <= c_chunk here, so the chunk loop runs once
            // and the single-chunk staging path is isolated. The (5,4) row covers
            // both rows, so a failure there is a harness or CPU-twin defect
            // rather than a kernel one.
            (9, 8, false, 14),
            (5, 4, false, 14),
            (5, 4, false, 0),
        ] {
            let x = rng(c_in * base, c_in as u32 * 7 + c_out as u32);
            let ww = rng(c_out * c_in * 9, c_out as u32 * 13 + 5);
            let bb = rng(c_out, c_out as u32 + 99);
            let din = DevBuf::from_host(&x)?;
            let mut cat = bb.clone();
            cat.extend_from_slice(&ww); // bias first, then weights
            let wcon = DevBuf::from_host(&cat)?;
            let w_off = bb.len();
            let dout = DevBuf::alloc(c_out * base)?;
            let loc = ConvLoc { w: w_off, b: 0 };
            // The shape, the grid, the block, the shared size and the argument
            // list all come from the SAME table the network path uses, so the two
            // cannot disagree.
            let shape = ConvShape::by_index(variant);
            let name = shape.name;
            // The chunk the network's path asks for; `chunk_and_smem` clamps it
            // to what the shared-memory cap holds.
            let (c_chunk, smem) = shape.chunk_and_smem(16);
            let l = shape.launch(w, h, c_out).shared(smem);
            let args = shape.args(
                din.ptr,
                wcon.ptr + (loc.w * 4) as u64,
                wcon.ptr + (loc.b * 4) as u64,
                0u64,
                dout.ptr,
                c_in, c_out, h, w, act, 0.0, c_chunk,
            );
            self.cuda.launch_raw(name, l, args)?;
            self.cuda.sync()?;
            let mut got = vec![0f32; c_out * base];
            dout.download(&mut got)?;

            let mut cpu_out = Plane::new(c_out, h, w);
            let conv = crate::net::Conv::from_parts(&ww, &bb, c_in, c_out);
            net::conv3x3_prefix_pub(&x, c_in, &mut cpu_out.data, &conv, h, w, act, None);
            let max = got.iter().zip(cpu_out.data.iter()).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
            let limit = 1e-4 * ((c_in * 9) as f32).sqrt().max(1.0);
            compared += 1;
            if max > limit {
                bad.push(format!("{name} {c_in}->{c_out} act={act}: max |diff| {max:.3e} > {limit:.1e}"));
            } else {
                println!("  {name:<22} {c_in:>3}->{c_out:<3} act={act:<5} max |diff| {max:.2e}");
            }
        }

        // ---- pixel_unshuffle2 (exact) ----
        {
            let (c, hh, ww2) = (3usize, 24usize, 20usize);
            let x = rng(c * hh * ww2, 11);
            let din = DevBuf::from_host(&x)?;
            let dout = DevBuf::alloc(c * 4 * (hh / 2) * (ww2 / 2))?;
            self.cuda.run(
                "lg_pixel_unshuffle2",
                lightgpu::vm::Launch::new((((ww2 / 2) as u32 + 31) / 32, ((hh / 2) as u32 + 7) / 8, 1), (32, 8, 1)),
                |a| { a.ptr(din.ptr).ptr(dout.ptr).i32(c as i32).i32(hh as i32).i32(ww2 as i32); },
            )?;
            self.cuda.sync()?;
            let mut got = vec![0f32; c * 4 * (hh / 2) * (ww2 / 2)];
            dout.download(&mut got)?;
            let cpu = net::pixel_unshuffle2(&Plane { c, h: hh, w: ww2, data: x });
            let max = got.iter().zip(cpu.data.iter()).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
            compared += 1;
            if max != 0.0 { bad.push(format!("lg_pixel_unshuffle2: max |diff| {max:.3e}")); }
            else { println!("  {:<22} 3->12 (24x20->12x10)   exact", "lg_pixel_unshuffle2"); }
        }

        // ---- upsample2x_nearest (exact) ----
        {
            let (c, hh, ww2) = (5usize, 13usize, 9usize);
            let x = rng(c * hh * ww2, 23);
            let din = DevBuf::from_host(&x)?;
            let dout = DevBuf::alloc(c * 4 * hh * ww2)?;
            self.cuda.run(
                "lg_upsample2x_nearest",
                lightgpu::vm::Launch::new((((ww2 * 2) as u32 + 31) / 32, ((hh * 2) as u32 + 7) / 8, 1), (32, 8, 1)),
                |a| { a.ptr(din.ptr).ptr(dout.ptr).i32(c as i32).i32(hh as i32).i32(ww2 as i32); },
            )?;
            self.cuda.sync()?;
            let mut got = vec![0f32; c * 4 * hh * ww2];
            dout.download(&mut got)?;
            let cpu = net::upsample2x_nearest(&Plane { c, h: hh, w: ww2, data: x });
            let max = got.iter().zip(cpu.data.iter()).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
            compared += 1;
            if max != 0.0 { bad.push(format!("lg_upsample2x_nearest: max |diff| {max:.3e}")); }
            else { println!("  {:<22} 5ch 13x9 -> 26x18     exact", "lg_upsample2x_nearest"); }
        }

        // ---- add_scaled, lrelu, copy (exact) ----
        {
            let n = 1000usize;
            let a = rng(n, 31);
            let b = rng(n, 37);
            let da = DevBuf::from_host(&a)?;
            let db = DevBuf::from_host(&b)?;
            let dout = DevBuf::alloc(n)?;
            self.cuda.run_n("lg_add_scaled", n, |x| {
                x.ptr(da.ptr).ptr(db.ptr).ptr(dout.ptr).i64(n as i64).f32(0.2);
            })?;
            self.cuda.sync()?;
            let mut got = vec![0f32; n];
            dout.download(&mut got)?;
            // a + s*b contracts to an FMA on the device, so this is an
            // arithmetic kernel and gets a tolerance; `lg_copy` and
            // `lg_lrelu` below are pure data movement / a comparison and
            // must stay bit-exact.
            let max = got.iter().enumerate().fold(0f32, |m, (i, g)| m.max((g - (a[i] + 0.2 * b[i])).abs()));
            let limit = 1e-6;
            compared += 1;
            if max > limit {
                bad.push(format!("lg_add_scaled: max |diff| {max:.3e} > {limit:.1e}"));
            } else {
                println!("  {:<22} n={n} scale=0.2    max |diff| {max:.2e}", "lg_add_scaled");
            }

            let dout2 = DevBuf::alloc(n)?;
            self.cuda.run_n("lg_lrelu", n, |x| { x.ptr(da.ptr).ptr(dout2.ptr).f32(0.2).i64(n as i64); })?;
            self.cuda.sync()?;
            let mut got2 = vec![0f32; n];
            dout2.download(&mut got2)?;
            let max2 = got2.iter().enumerate().fold(0f32, |m, (i, g)| m.max((g - net::leaky_relu(a[i])).abs()));
            compared += 1;
            if max2 != 0.0 { bad.push(format!("lg_lrelu: max |diff| {max2:.3e}")); }
            else { println!("  {:<22} n={n} slope=0.2    exact", "lg_lrelu"); }

            let dout3 = DevBuf::alloc(n)?;
            self.cuda.run_n("lg_copy", n, |x| { x.ptr(da.ptr).ptr(dout3.ptr).i64(n as i64); })?;
            self.cuda.sync()?;
            let mut got3 = vec![0f32; n];
            dout3.download(&mut got3)?;
            let max3 = got3.iter().zip(a.iter()).fold(0f32, |m, (g, v)| m.max((g - v).abs()));
            compared += 1;
            if max3 != 0.0 { bad.push(format!("lg_copy: max |diff| {max3:.3e}")); }
            else { println!("  {:<22} n={n}             exact", "lg_copy"); }
        }

        Ok((compared, bad))
    }
}
