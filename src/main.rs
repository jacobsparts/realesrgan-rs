//! Real-ESRGAN image upscaling, standalone.
//!
//! Usage:
//!   realesrgan --model RealESRGAN_x4plus.safetensors -i in.png -o out.png
//!   realesrgan --model ... -i in.png --device cpu --tile 256
//!
//! The model file is the output of `tools/convert.py`, not the original .pth.

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
mod gpu;
mod image;
mod net;
mod weights;

use std::time::Instant;

const VERSION: &str = env!("CARGO_PKG_VERSION");

fn usage() -> ! {
    eprintln!(
        "realesrgan {VERSION} - Real-ESRGAN (RRDBNet) upscaling on lightgpu

USAGE:
    realesrgan --model <weights.safetensors> -i <in.png> -o <out.png> [options]

OPTIONS:
    -m, --model <path>    converted .safetensors checkpoint (see tools/convert.py)
    -i, --input <path>    input PNG, or - for stdin (default: stdin)
    -o, --output <path>   output PNG, or - for stdout (default: stdout)
        --device <dev>    gpu or cpu (default: the CUDA build uses gpu, a
                          CPU-only build uses cpu)
        --tile <n>        process in tiles of n pixels, 0 = whole image (default 0)
        --tile-pad <n>    overlap around each tile (default 10)
        --outscale <f>    resize the result to this final scale (default: model scale)
    -q, --quiet           no progress output
        --cuda-selftest   compare each CUDA kernel against its CPU twin and exit
        --self-test       run the CPU graph's internal checks and exit
    -h, --help            this text
    -V, --version         print the version"
    );
    std::process::exit(2)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        usage();
    }
    let mut model_path = None;
    let mut input = None;
    let mut output = None;
    // A CPU-only build has no GPU backend to default to, so it defaults to the
    // one it can actually run.
    let mut device = if cfg!(feature = "cuda") { "gpu" } else { "cpu" }.to_string();
    let mut tile = 0usize;
    let mut tile_pad = 10usize;
    let mut outscale: Option<f64> = None;
    let mut quiet = false;
    let mut selftest = false;
    let mut cpu_selftest = false;
    let mut bench_conv = false;
    let mut bench_gemm = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-m" | "--model" => {
                i += 1;
                model_path = args.get(i).cloned();
            }
            "-i" | "--input" => {
                i += 1;
                input = args.get(i).cloned();
            }
            "-o" | "--output" => {
                i += 1;
                output = args.get(i).cloned();
            }
            "--device" => {
                i += 1;
                device = args.get(i).cloned().unwrap_or_else(|| usage());
            }
            "--tile" => {
                i += 1;
                tile = args.get(i).and_then(|s| s.parse().ok()).unwrap_or_else(|| usage());
            }
            "--tile-pad" => {
                i += 1;
                tile_pad = args.get(i).and_then(|s| s.parse().ok()).unwrap_or_else(|| usage());
            }
            "--outscale" => {
                i += 1;
                outscale = args.get(i).and_then(|s| s.parse().ok());
            }
            "-q" | "--quiet" => quiet = true,
            "--cuda-selftest" => selftest = true,
            "--self-test" => cpu_selftest = true,
            // Time one conv shape in isolation.
            "--bench-conv" => bench_conv = true,
            // Time the toolkit's f32 GEMM at the shapes a Winograd-fed batched
            // GEMM would use. Measurement only - see the comment on
            // TOOLKIT_KERNELS in build.rs.
            "--bench-gemm" => bench_gemm = true,
            "-h" | "--help" => usage(),
            "-V" | "--version" => {
                println!("realesrgan {VERSION}");
                return;
            }
            other => {
                eprintln!("unknown argument: {other}");
                usage();
            }
        }
        i += 1;
    }
    // Both tiling flags are parsed in every build so the CLI surface does not
    // change with the feature set, but only the GPU path can tile; the CPU-only
    // build reads neither value, hence the discard.
    let _ = (tile, tile_pad);

    let model_path = model_path.unwrap_or_else(|| usage());
    let t_load = Instant::now();
    let w = match weights::Weights::open(&model_path) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1)
        }
    };
    if !quiet {
        eprintln!(
            "model: {} ({:.1} MiB, {} tensors, scale x{}, {} blocks)",
            model_path,
            w.total_bytes() as f64 / (1024.0 * 1024.0),
            w.file.order().len(),
            w.config.scale,
            w.config.num_block
        );
    }
    let model = match net::Model::load(&w) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1)
        }
    };
    if !quiet {
        eprintln!(
            "weights bound in {:.2}s ({:.1} MiB)",
            t_load.elapsed().as_secs_f64(),
            model.weight_bytes() as f64 / (1024.0 * 1024.0)
        );
    }

    if bench_conv {
        bench_conv_run();
    }
    if bench_gemm {
        #[cfg(not(feature = "cuda"))]
        {
            eprintln!("error: --bench-gemm needs a build with the `cuda` feature");
            std::process::exit(1);
        }
        #[cfg(feature = "cuda")]
        bench_gemm_run();
    }
    if cpu_selftest {
        cpu_selftest_run(&model, &w);
        return;
    }

    if selftest {
        #[cfg(not(feature = "cuda"))]
        {
            eprintln!("error: --cuda-selftest needs a build with the `cuda` feature");
            std::process::exit(1);
        }
        #[cfg(feature = "cuda")]
        let cuda = match cuda::Cuda::init(!quiet) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("error: {e}");
                std::process::exit(1)
            }
        };
        // A small working resolution is enough: the kernels are compared on
        // their own random data, not on an image.
        #[cfg(feature = "cuda")]
        let g = match gpu::Gpu::new(cuda, model, 64, 64) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("error: {e}");
                std::process::exit(1)
            }
        };
        #[cfg(feature = "cuda")]
        match g.selftest() {
            Ok((n, bad)) => {
                if bad.is_empty() {
                    println!("cuda selftest: {n} op(s) compared, 0 failures");
                } else {
                    println!("cuda selftest: {n} op(s) compared, {} FAILED:", bad.len());
                    for b in &bad {
                        println!("  {b}");
                    }
                    std::process::exit(1);
                }
            }
            Err(e) => {
                eprintln!("error: {e}");
                std::process::exit(1)
            }
        }
        #[cfg(feature = "cuda")]
        return;
    }

    // `-` and an unspecified path both mean the standard stream, so the
    // program composes in a pipe: `... -o - | ...`
    let input = input.unwrap_or_else(|| "-".to_string());
    let output = output.unwrap_or_else(|| "-".to_string());
    let img = match if input == "-" {
        image::load_rgb_stream(std::io::stdin().lock()).map_err(|e| format!("stdin: {e}"))
    } else {
        image::load_rgb(&input)
    } {
        Ok(v) => v,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1)
        }
    };
    if !quiet {
        let src = if input == "-" { "stdin".to_string() } else { input.clone() };
        eprintln!("input: {} ({}x{})", src, img.w, img.h);
    }

    let t_fwd = Instant::now();
    let out = match device.as_str() {
        "cpu" => {
            let p = net::Plane {
                c: 3,
                h: img.h,
                w: img.w,
                data: img.data.clone(),
            };
            let mut n = 0;
            let mut progress = |s: &str| {
                if !quiet {
                    n += 1;
                    eprintln!("  [{n}] {s}");
                }
            };
            let r = net::forward_cpu(&model, &p, &mut progress);
            net::cpu_profile_report();
            r
        }
        "gpu" => {
            #[cfg(not(feature = "cuda"))]
            {
                eprintln!("error: this binary was built without the `cuda` feature; use --device cpu");
                std::process::exit(1);
            }
            #[cfg(feature = "cuda")]
            let cuda = match cuda::Cuda::init(!quiet) {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("error: {e}");
                    std::process::exit(1)
                }
            };
            #[cfg(feature = "cuda")]
            {
                match forward_tiled(cuda, model, &img, tile, tile_pad, quiet) {
                    Ok(v) => v,
                    Err(e) => {
                        eprintln!("error: {e}");
                        std::process::exit(1)
                    }
                }
            }
        }
        other => {
            eprintln!("unknown device: {other}");
            usage()
        }
    };
    if !quiet {
        eprintln!("forward in {:.2}s", t_fwd.elapsed().as_secs_f64());
    }

    let mut result = image::Image { w: out.w, h: out.h, data: out.data.clone() };
    if let Some(s) = outscale {
        let want_w = (img.w as f64 * s).round() as usize;
        let want_h = (img.h as f64 * s).round() as usize;
        if (want_w, want_h) != (result.w, result.h) {
            result = resize_lanczos(&result, want_w, want_h);
        }
    }
    let rgb = result.to_rgb8();
    if let Err(e) = if output == "-" {
        let stdout = std::io::stdout();
        let mut lock = stdout.lock();
        image::save_rgb_stream(&mut lock, result.w, result.h, &rgb)
            .and_then(|()| std::io::Write::flush(&mut lock).map_err(|e| e.to_string()))
            .map_err(|e| format!("stdout: {e}"))
    } else {
        image::save_rgb(&output, result.w, result.h, &rgb)
    } {
        eprintln!("error: {e}");
        std::process::exit(1)
    }
    if !quiet {
        let dst = if output == "-" { "stdout".to_string() } else { output.clone() };
        eprintln!("wrote {} ({}x{})", dst, result.w, result.h);
    }
}

/// The GPU forward pass, tiled when asked.
///
/// Tiling follows the reference: each tile is cropped from the input with
/// `pad` pixels of context on every side, the network runs on that crop, and
/// only the centre is pasted into the output (the pad is discarded). At the
/// image border the crop shrinks rather than extending outside it, which is
/// what the reference does and what keeps the border pixels identical to the
/// untiled run.
///
/// A `Gpu` is built per distinct crop size. Interior tiles all share one size,
/// and the border tiles add at most three more, so a rebuild happens a handful
/// of times. `Model::load` is cheap (it re-points at the memory-mapped
/// checkpoint), so this costs a weight re-upload of 0.4 s at worst - far less
/// than holding several activation arenas live at once.
#[cfg(feature = "cuda")]
fn forward_tiled(
    cuda: cuda::Cuda,
    model: net::Model,
    img: &image::Image,
    tile: usize,
    pad: usize,
    quiet: bool,
) -> Result<net::Plane, String> {
    let (w, h) = (img.w, img.h);
    let scale = model.geom.scale;
    let mut g = gpu::Gpu::new(cuda, model, w, h)?;
    let out_w = w * scale;
    let out_h = h * scale;
    let mut out = net::Plane {
        c: 3,
        h: out_h,
        w: out_w,
        data: vec![0f32; 3 * out_w * out_h],
    };

    // `tile` is the interior size in INPUT pixels; `pad` is context added on
    // each side. A tile of 0 (or one that covers the image) runs in one pass.
    let tile_x = if tile == 0 { w } else { tile.min(w) };
    let tile_y = if tile == 0 { h } else { tile.min(h) };
    if tile_x == w && tile_y == h {
        let v = g.forward(&img.data)?;
        out.data = v;
        g.cuda.profile_report();
        crate::net::cpu_profile_report();
        return Ok(out);
    }

    let mut n_tiles = 0usize;
    let mut y0 = 0usize;
    while y0 < h {
        let y1 = (y0 + tile_y).min(h);
        let y0p = y0.saturating_sub(pad);
        let y1p = (y1 + pad).min(h);
        let mut x0 = 0usize;
        while x0 < w {
            let x1 = (x0 + tile_x).min(w);
            let x0p = x0.saturating_sub(pad);
            let x1p = (x1 + pad).min(w);
            let (cw, ch) = (x1p - x0p, y1p - y0p);

            // Crop the padded region into a planar f32 plane.
            let mut crop = vec![0f32; 3 * cw * ch];
            for c in 0..3 {
                let src = img.plane(c);
                for yy in 0..ch {
                    for xx in 0..cw {
                        crop[c * cw * ch + yy * cw + xx] =
                            src[(y0p + yy) * w + (x0p + xx)];
                    }
                }
            }

            g.resize(cw, ch)?;
            let up = g.forward(&crop)?;
            let (uw, uh) = (cw * scale, ch * scale);

            // Paste the interior (everything except the pad, scaled).
            let px0 = (x0 - x0p) * scale;
            let py0 = (y0 - y0p) * scale;
            let paste_w = (x1 - x0) * scale;
            let paste_h = (y1 - y0) * scale;
            for c in 0..3 {
                for yy in 0..paste_h {
                    let drow = (y0 * scale + yy) * out_w + x0 * scale;
                    let srow = c * uw * uh + (py0 + yy) * uw + px0;
                    out.data[c * out_w * out_h + drow..c * out_w * out_h + drow + paste_w]
                        .copy_from_slice(&up[srow..srow + paste_w]);
                }
            }
            n_tiles += 1;
            if !quiet {
                eprintln!(
                    "  tile {n_tiles}: in ({x0p},{y0p})-({x1p},{y1p}) -> out ({},{}), {}x{}",
                    x0 * scale, y0 * scale, paste_w, paste_h
                );
            }
            x0 = x1;
        }
        y0 = y1;
    }
    if !quiet {
        eprintln!("  {n_tiles} tile(s)");
    }
    g.cuda.profile_report();
    crate::net::cpu_profile_report();
    Ok(out)
}

/// The reference resizes with INTER_LANCZOS4 when `--outscale` differs from the
/// model scale. This is a separable Lanczos-4 resample; it exists so the flag
/// is not a silent no-op.
fn resize_lanczos(src: &image::Image, w: usize, h: usize) -> image::Image {
    fn kernel(x: f64) -> f64 {
        if x == 0.0 {
            return 1.0;
        }
        let px = std::f64::consts::PI * x;
        let a = 4.0;
        if x.abs() >= a {
            return 0.0;
        }
        let pix = px / a;
        (px.sin() / px) * (pix.sin() / pix)
    }
    let mut tmp = image::Image::new(w, src.h);
    for c in 0..3 {
        let ip = src.plane(c);
        for y in 0..src.h {
            for x in 0..w {
                let center = (x as f64 + 0.5) * src.w as f64 / w as f64 - 0.5;
                let lo = (center - 4.0).ceil() as isize;
                let hi = (center + 4.0).floor() as isize;
                let mut acc = 0.0;
                let mut norm = 0.0;
                for i in lo..=hi {
                    let k = kernel(center - i as f64);
                    norm += k;
                    let xi = i.clamp(0, src.w as isize - 1) as usize;
                    acc += k * ip[y * src.w + xi] as f64;
                }
                tmp.data[c * w * src.h + y * w + x] = (acc / norm) as f32;
            }
        }
    }
    let mut out = image::Image::new(w, h);
    for c in 0..3 {
        let tp = &tmp.data[c * w * src.h..(c + 1) * w * src.h];
        for y in 0..h {
            for x in 0..w {
                let center = (y as f64 + 0.5) * src.h as f64 / h as f64 - 0.5;
                let lo = (center - 4.0).ceil() as isize;
                let hi = (center + 4.0).floor() as isize;
                let mut acc = 0.0;
                let mut norm = 0.0;
                for i in lo..=hi {
                    let k = kernel(center - i as f64);
                    norm += k;
                    let yi = i.clamp(0, src.h as isize - 1) as usize;
                    acc += k * tp[yi * w + x] as f64;
                }
                out.data[c * w * h + y * w + x] = (acc / norm) as f32;
            }
        }
    }
    out
}

/// Time the toolkit's v8-style f32 GEMM at the shapes a Winograd-fed batched
/// GEMM would use, and report the fraction of the measured peak.
///
/// One transform point of F(4x4,3x3) is a GEMM with K = c_in, M = c_out and
/// N = the number of 4x4 output tiles, and there are 36 of them. The shapes
/// below are the network's real ones: the dense body at 256x256 (c_in=64,
/// c_out=64) has 4096 tiles per point, the 1024x1024 upsample stage 16384.
#[cfg(feature = "cuda")]
fn bench_gemm_run() {
    let cuda = match cuda::Cuda::init(false) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1)
        }
    };
    // The measured fp32 FFMA ceiling of this GPU, from a dedicated 8-chain
    // probe - not the nominal SM x lane x 2 product, which is unreachable.
    const PEAK_GFLOP_S: f64 = 7653.0;
    // Correctness first: a utilisation number from a kernel that computes the
    // WRONG answer would read exactly the same, and this project's rule is that
    // a kernel which can be selected is also checked.
    bench_gemm_verify(&cuda);
    println!("GEMM utilisation, against a measured peak of {PEAK_GFLOP_S:.0} GFLOP/s");
    println!("  (one Winograd F(4x4,3x3) transform point: y[oc, tiles] = W[oc, ci] * x[ci, tiles])");
    // Two groups, because per-launch cost and throughput are different
    // questions: the network's per-point shapes are small (M=64, N=4096 is 33.5
    // MFLOP, ~40 us of work), so their times are dominated by launch and sync
    // overhead and are not a utilisation figure. The second group gives the
    // kernel enough work to reach one.
    println!("  -- the network's per-point shapes (launch-overhead dominated) --");
    for &(c_in, c_out, tiles) in &[
        (64usize, 64usize, 4096usize),
        (64, 64, 16384),
        (192, 64, 4096),
    ] {
        bench_gemm_one(&cuda, c_in, c_out, tiles, PEAK_GFLOP_S);
    }
    println!("  -- large-N shapes: what the kernel reaches when it has work --");
    for &(c_in, c_out, tiles) in &[
        (64usize, 64usize, 65536usize),
        (64, 64, 262144),
        (192, 64, 262144),
        (64, 128, 262144),
    ] {
        bench_gemm_one(&cuda, c_in, c_out, tiles, PEAK_GFLOP_S);
    }
    println!("\nnote: 36 such points = one 256x256 conv stage; multiply a row's time by 36");

    // THE COMPARISON THAT DECIDES IT: the same instrument, the same shapes, the
    // same best-of-7 discipline, but on the FUSED Winograd kernel that the
    // network actually runs. A GEMM time on its own says nothing about whether
    // the GEMM ROUTE is faster - the route also owes three transform passes and
    // 36 launches per conv, and the fused kernel pays none of that.
    //
    // The fused kernel is timed by `gpu.rs`, not here: its launch has to go
    // through the same shape table the network uses, and reproducing that launch
    // in this file would measure a copy of the production path instead of the
    // production path.
    gpu::bench_fused_conv(&cuda);
}

#[cfg(feature = "cuda")]
fn bench_gemm_one(cuda: &cuda::Cuda, c_in: usize, c_out: usize, tiles: usize, peak: f64) {
    let ne0 = c_in;      // reduction (contiguous in both operands)
    let ne1 = c_out;     // rows of W
    let ncols = tiles;   // columns of x / y
    let w = vec![0.01f32; ne1 * ne0];
    let x = vec![0.01f32; ncols * ne0];
    let dw = match cuda::DevBuf::from_host(&w) { Ok(b) => b, Err(e) => { eprintln!("{e}"); return } };
    let dx = match cuda::DevBuf::from_host(&x) { Ok(b) => b, Err(e) => { eprintln!("{e}"); return } };
    let dy = match cuda::DevBuf::alloc(ncols * ne1) { Ok(b) => b, Err(e) => { eprintln!("{e}"); return } };
    let grid = (((ne1 + 63) / 64) as u32, ((ncols + 31) / 32) as u32, 1);
    let launch = || {
        cuda.run("lg_f32_gemm_tiled", lightgpu::vm::Launch::new(grid, (256, 1, 1)), |a| {
            a.ptr(dw.ptr).ptr(dx.ptr).ptr(dy.ptr)
                .i32(ne0 as i32).i32(ne1 as i32).i32(ncols as i32);
        })
    };
    if let Err(e) = launch() {
        eprintln!("launch failed: {e}");
        return;
    }
    if let Err(e) = cuda.sync() {
        eprintln!("sync failed: {e}");
        return;
    }
    let mut best = f64::INFINITY;
    for _ in 0..7 {
        let t = Instant::now();
        if let Err(e) = launch() {
            eprintln!("launch failed: {e}");
            return;
        }
        if cuda.sync().is_err() {
            return;
        }
        let dt = t.elapsed().as_secs_f64();
        if dt < best {
            best = dt;
        }
    }
    let flops = 2.0 * (ne0 * ne1 * ncols) as f64;
    let gf = flops / best / 1e9;
    println!(
        "  K={ne0:>4} M={ne1:>4} N={ncols:>6}: best {:.6}s ({:.1} us), {:.1} GFLOP/s = {:.1}% of peak",
        best, best * 1e6, gf, 100.0 * gf / peak
    );
}

/// A single small GEMM, checked against a CPU reference.
///
/// The utilisation bench above answers "how fast", and a bench that is fast and
/// WRONG would answer it anyway - so this one answers "the same answer". The
/// toolkit's own op has a CPU twin (`cpu::f32_gemm`), and the shape here is
/// small enough to reproduce with a triple loop in the test itself: no shared
/// code path between the two sides means the comparison is evidence.
#[cfg(feature = "cuda")]
fn bench_gemm_verify(cuda: &cuda::Cuda) {
    let (ne0, ne1, ncols) = (8usize, 5usize, 7usize);
    let w: Vec<f32> = (0..ne1 * ne0).map(|i| ((i * 7 % 11) as f32) * 0.125 - 0.5).collect();
    let x: Vec<f32> = (0..ncols * ne0).map(|i| ((i * 5 % 13) as f32) * 0.25 - 1.0).collect();
    // ne0 % 4 == 0 is the kernel's stated requirement, hence 8 and not 7.
    let dw = cuda::DevBuf::from_host(&w).unwrap();
    let dx = cuda::DevBuf::from_host(&x).unwrap();
    let dy = cuda::DevBuf::alloc(ncols * ne1).unwrap();
    let grid = (((ne1 + 63) / 64) as u32, ((ncols + 31) / 32) as u32, 1);
    cuda.run("lg_f32_gemm_tiled", lightgpu::vm::Launch::new(grid, (256, 1, 1)), |a| {
        a.ptr(dw.ptr).ptr(dx.ptr).ptr(dy.ptr)
            .i32(ne0 as i32).i32(ne1 as i32).i32(ncols as i32);
    })
    .unwrap();
    cuda.sync().unwrap();
    let mut got = vec![0f32; ncols * ne1];
    dy.download(&mut got).unwrap();
    // y[col][row] = sum_k W[row][k] * x[col][k] - the layout the caller in
    // locate-anything-rs uses, restated here from the op's contract.
    let mut want = vec![0f32; ncols * ne1];
    for col in 0..ncols {
        for row in 0..ne1 {
            let mut acc = 0f32;
            for k in 0..ne0 {
                acc += w[row * ne0 + k] * x[col * ne0 + k];
            }
            want[col * ne1 + row] = acc;
        }
    }
    let max = got.iter().zip(want.iter()).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
    println!("  correctness K=8 M=5 N=7: max |diff| vs CPU reference {max:.3e}");
}

/// Time one conv shape repeatedly and report GFLOP/s. Best-of-N rather than a
/// mean, because the interesting quantity is the achievable rate when nothing
/// else interferes, and on this box a mean is dominated by scheduler noise.
/// This times the CPU REFERENCE conv (`net::conv3x3_prefix_pub`), not the GPU
/// path - the GPU conv has its own numbers in `gpu.rs` and `--bench-gemm`.
fn bench_conv_run() {
    let (c_in, c_out, h, w) = (192usize, 64usize, 256usize, 256usize);
    let hw = h * w;
    let wts: Vec<f32> = (0..c_out * c_in * 9)
        .map(|i| ((i % 17) as f32) * 0.01 - 0.08)
        .collect();
    let bias: Vec<f32> = (0..c_out).map(|i| (i % 5) as f32 * 0.01).collect();
    let conv = net::Conv::from_parts(&wts, &bias, c_in, c_out);
    let src: Vec<f32> = (0..c_in * hw).map(|i| ((i % 29) as f32) * 0.03 - 0.4).collect();
    let mut dst = vec![0f32; c_out * hw];
    let flops = 2.0 * (c_in * c_out * 9 * hw) as f64;
    let threads = rayon::current_num_threads();

    net::conv3x3_prefix_pub(&src, c_in, &mut dst, &conv, h, w, false, None);
    let mut best = f64::INFINITY;
    for _ in 0..7 {
        let t = Instant::now();
        net::conv3x3_prefix_pub(&src, c_in, &mut dst, &conv, h, w, false, None);
        let dt = t.elapsed().as_secs_f64();
        if dt < best {
            best = dt;
        }
    }
    let checksum: f64 = dst.iter().step_by(997).map(|v| *v as f64).sum();
    println!(
        "conv {}->{} {}x{}: best {:.4}s, {:.1} GFLOP/s over {} thread(s) = {:.1} GFLOP/s/thread [chk {:.4}]",
        c_in,
        c_out,
        h,
        w,
        best,
        flops / best / 1e9,
        threads,
        flops / best / 1e9 / threads as f64,
        checksum
    );
}

fn cpu_selftest_run(model: &net::Model, w: &weights::Weights) {
    let geom = &model.geom;
    println!("cpu selftest: scale x{}, {} blocks, {} feat, {} grow",
             geom.scale, geom.num_block, geom.num_feat, geom.num_grow_ch);
    let (w0, h0) = (16usize, 16usize);
    let img = net::Plane { c: geom.in_ch, h: h0, w: w0, data: vec![0.5f32; geom.in_ch * w0 * h0] };
    let mut stages = Vec::new();
    let mut progress = |s: &str| stages.push(s.to_string());
    let out = net::forward_cpu(model, &img, &mut progress);
    println!("stages: {}", stages.join(" -> "));
    println!("output: {}x{}x{}", out.c, out.h, out.w);
    assert_eq!(out.c, geom.out_ch, "output channels");
    assert_eq!(out.h, h0 * geom.scale, "output height");
    assert_eq!(out.w, w0 * geom.scale, "output width");
    let finite = out.data.iter().all(|v| v.is_finite());
    assert!(finite, "output has a non-finite value");
    let mn = out.data.iter().cloned().fold(f32::INFINITY, f32::min);
    let mx = out.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    println!("range [{mn:.4}, {mx:.4}]");
    // A constant 0.5 input must not produce a constant output: the network is
    // not shift-invariant-free, so a constant result means a stage was skipped.
    let spread = mx - mn;
    assert!(spread > 1e-6, "output is constant: spread {spread}");
    println!(
        "weights: {} tensors, {:.1} MiB",
        w.file.order().len(),
        w.total_bytes() as f64 / 1048576.0
    );
    println!("cpu selftest ok");
}
