//! CUDA backend: device memory, the two embedded fatbin modules, launches.
//!
//! The driver bindings, the context, the module and the argument marshalling
//! come from `lightgpu`; what is here is the f32-facing surface the graph code
//! uses (buffers counted in ELEMENTS rather than bytes, and name resolution
//! across the two modules), plus the ASCII-art of this engine's fatbins.
//!
//! Compiled only with `--features cuda`. `libcuda.so.1` is `dlopen`ed by
//! lightgpu at run time, so the CPU-only build needs no NVIDIA driver at all.

#![allow(dead_code)]

use lightgpu::ffi::CUfunction;
use lightgpu::vm;
use lightgpu::vm::{Args, Launch};
use std::cell::RefCell;
use std::collections::HashMap;

thread_local! {
    /// Per-(kernel, launch grid) device time in ms and call count, filled only
    /// under `LA_PROFILE=1`.
    ///
    /// The grid is part of the key because one kernel serves very different
    /// geometries: the 256x256 body convs (8x16 blocks) and the 1024x1024 head
    /// convs (32x64 blocks) run the same code at very different arithmetic
    /// intensity, so a name-only total hides which regime is losing time.
    static PROFILE: RefCell<HashMap<String, (f64, usize)>> = RefCell::new(HashMap::new());
}

fn profiling() -> bool {
    std::env::var("LA_PROFILE").map(|v| v == "1").unwrap_or(false)
}

/// The toolkit's generic ops, compiled down to the subset this engine calls:
/// sm_61 / sm_75 / sm_80 SASS plus compute_80 PTX, so the binary runs on
/// Kepler-and-later cards and JITs forward on newer ones.
#[cfg(feature = "cuda")]
pub static TOOLKIT_FATBIN: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/realesrgan_toolkit.fatbin"));

/// This project's own kernel family (`cuda/realesrgan.cu`), loaded as a SECOND
/// module. Separate modules are separate namespaces: a name in one cannot
/// shadow a name in the other, and a name in neither fails at startup through
/// `KERNEL_NAMES` rather than in the middle of a forward pass.
#[cfg(feature = "cuda")]
pub static PROJECT_FATBIN: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/realesrgan_project.fatbin"));

/// Every kernel this engine launches, resolved eagerly at startup from
/// whichever module defines it. Keeping this as an explicit list turns a
/// renamed or mis-filed kernel into a startup error; the per-launch path then
/// costs one hash lookup.
pub const KERNEL_NAMES: &[&str] = &[
    // From cuda/realesrgan.cu: the direct conv fallback for shapes the
    // launcher's table does not cover (a non-3x3 conv, or a tensor too small for
    // a Winograd CTA).
    "lg_conv3x3_res",
    // From the lightgpu toolkit.
    "lg_add",
    "lg_copy",
    "lg_scale",
    "lg_noop",
    "lg_add_scaled",
    "lg_lrelu",
    "lg_upsample2x_nearest",
    "lg_pixel_unshuffle2",
    // The production 3x3 conv: F(4x4,3x3) Winograd, with the bias and the
    // activation as arguments. This is the DEFAULT in gpu.rs.
    "lg_conv3x3_winograd",
];

/// Device information reported at startup.
#[derive(Clone, Debug)]
pub struct DeviceInfo {
    pub name: String,
    pub cc_major: i32,
    pub cc_minor: i32,
    pub sm_count: i32,
    pub smem_per_block: i32,
    pub free_vram: usize,
}

pub struct Cuda {
    pub project: vm::Module,
    pub toolkit: vm::Module,
    pub info: DeviceInfo,
}

impl Cuda {
    pub fn init(verbose: bool) -> Result<Cuda, String> {
        vm::init()?;
        let dev = vm::device()?;
        let project = vm::Module::load(PROJECT_FATBIN)?;
        let toolkit = vm::Module::load(TOOLKIT_FATBIN)?;
        let c = Cuda {
            project,
            toolkit,
            info: DeviceInfo {
                name: dev.name.clone(),
                cc_major: dev.cc_major,
                cc_minor: dev.cc_minor,
                sm_count: dev.sm_count,
                smem_per_block: dev.smem_per_block,
                free_vram: vm::free_vram()?,
            },
        };
        // Fail here, not mid-forward, if a kernel is missing or mis-filed.
        for k in KERNEL_NAMES {
            c.func(k)?;
        }
        if verbose {
            eprintln!(
                "cuda: {} cc {}.{} ({} SMs, {} MiB free, fatbins {} + {} bytes)",
                c.info.name,
                c.info.cc_major,
                c.info.cc_minor,
                c.info.sm_count,
                c.info.free_vram / (1024 * 1024),
                PROJECT_FATBIN.len(),
                TOOLKIT_FATBIN.len()
            );
        }
        Ok(c)
    }

    /// Resolve a kernel from whichever module defines it. The project module is
    /// tried first: it holds the names this engine owns, and the toolkit is the
    /// fallback. `Module::func` memoizes, so this is one hash lookup per launch
    /// once warm.
    pub fn func(&self, name: &str) -> Result<CUfunction, String> {
        match self.project.func(name) {
            Ok(f) => Ok(f),
            Err(_) => self.toolkit.func(name),
        }
    }

    /// Which module owns `name`. Used to launch through the module that has it
    /// (`Args::launch` takes a module, not a bare handle).
    pub fn module_of(&self, name: &str) -> Result<&vm::Module, String> {
        if self.project.func(name).is_ok() {
            return Ok(&self.project);
        }
        if self.toolkit.func(name).is_ok() {
            return Ok(&self.toolkit);
        }
        Err(format!("kernel {name} is in neither module"))
    }

    /// Launch `name` with `build` filling the argument list.
    ///
    /// With `LA_PROFILE=1` each launch is bracketed by CUDA events and its
    /// device time accumulated per kernel name. Events rather than a host sync:
    /// 355 launches each costing tens of microseconds would be swamped by the
    /// synchronisation that measuring them the naive way requires.
    pub fn run<F: FnOnce(&mut Args)>(&self, name: &str, l: Launch, build: F) -> Result<(), String> {
        let m = self.module_of(name)?;
        let mut a = Args::new();
        build(&mut a);
        if !profiling() {
            return a.launch(m, name, l);
        }
        let a_ev = vm::Event::new()?;
        let b_ev = vm::Event::new()?;
        a_ev.record()?;
        a.launch(m, name, l)?;
        b_ev.record()?;
        // cuEventElapsedTime needs BOTH events complete. Waiting on the closing
        // event serialises the launches, which is acceptable only because this
        // engine already runs everything on one stream - there is no overlap
        // left to lose - and it is what makes each kernel's device time exact
        // rather than a fraction of a queue.
        b_ev.synchronize()?;
        let ms = a_ev.elapsed_ms(&b_ev)? as f64;
        let key = format!("{} g{}x{}x{} b{}x{}", name, l.grid.0, l.grid.1, l.grid.2,
                          l.block.0, l.block.1);
        PROFILE.with(|p| {
            let mut p = p.borrow_mut();
            let e = p.entry(key).or_insert((0.0, 0usize));
            e.0 += ms;
            e.1 += 1;
        });
        Ok(())
    }

    /// Print the accumulated per-kernel device time, heaviest first.
    pub fn profile_report(&self) {
        PROFILE.with(|p| {
            let p = p.borrow();
            if p.is_empty() {
                return;
            }
            let mut rows: Vec<(&String, &(f64, usize))> = p.iter().collect();
            rows.sort_by(|a, b| b.1 .0.partial_cmp(&a.1 .0).unwrap());
            let total: f64 = p.values().map(|v| v.0).sum();
            eprintln!("--- kernel profile (device ms, {} total) ---", total);
            for (name, (ms, n)) in rows {
                eprintln!("  {:<24} {:9.1} ms in {:5} calls  ({:6.2} µs each, {:4.1}%)",
                          name, ms, n, ms * 1000.0 / *n as f64, 100.0 * ms / total);
            }
        });
    }

    /// Launch with an argument list already built by the caller. Needed where
    /// one call site selects between kernels with DIFFERENT signatures (the
    /// selftest runs all three conv variants, and v2 takes one more argument
    /// than the others), which a single closure cannot express.
    pub fn launch_raw(&self, name: &str, l: Launch, mut args: Args) -> Result<(), String> {
        let m = self.module_of(name)?;
        if !profiling() {
            return args.launch(m, name, l);
        }
        // The bracket has to SURROUND the launch, not follow it: an event pair
        // recorded after the kernel would time the empty interval between two
        // event records and report a number near zero for the hottest kernel in
        // the engine. `run` brackets by hand for the same reason.
        let a_ev = vm::Event::new()?;
        let b_ev = vm::Event::new()?;
        a_ev.record()?;
        args.launch(m, name, l)?;
        b_ev.record()?;
        b_ev.synchronize()?;
        let ms = a_ev.elapsed_ms(&b_ev)? as f64;
        let key = format!("{} g{}x{}x{} b{}x{}", name, l.grid.0, l.grid.1, l.grid.2,
                          l.block.0, l.block.1);
        PROFILE.with(|p| {
            let mut p = p.borrow_mut();
            let e = p.entry(key).or_insert((0.0, 0usize));
            e.0 += ms;
            e.1 += 1;
        });
        Ok(())
    }

    /// Launch over `n` elements with 256-thread blocks (the shape of every
    /// elementwise kernel here).
    pub fn run_n<F: FnOnce(&mut Args)>(&self, name: &str, n: usize, build: F) -> Result<(), String> {
        let block = 256u64;
        let grid = ((n as u64 + block - 1) / block).max(1) as u32;
        self.run(name, Launch::new((grid, 1, 1), (block as u32, 1, 1)), build)
    }

    pub fn sync(&self) -> Result<(), String> {
        vm::sync()
    }
}

/// A device buffer of f32 counted in ELEMENTS (lightgpu counts bytes; the graph
/// code thinks in tensor lengths, so the conversion stays in this one place).
pub struct DevBuf {
    pub ptr: lightgpu::ffi::CUdeviceptr,
    pub len: usize,
    inner: Option<vm::DevBuf>,
}

impl DevBuf {
    pub fn empty() -> DevBuf {
        DevBuf { ptr: 0, len: 0, inner: None }
    }

    pub fn alloc(len: usize) -> Result<DevBuf, String> {
        if len == 0 {
            return Ok(DevBuf::empty());
        }
        let b = vm::DevBuf::alloc(len * std::mem::size_of::<f32>())?;
        Ok(DevBuf { ptr: b.ptr, len, inner: Some(b) })
    }

    /// A zeroed buffer. Zeroing matters where a plane is read before every
    /// element is provably written (the upsampled planes' edges during the
    /// head convs), so the allocation and the zeroing are one operation.
    pub fn zeros(len: usize) -> Result<DevBuf, String> {
        if len == 0 {
            return Ok(DevBuf::empty());
        }
        let b = vm::DevBuf::zeros(len * std::mem::size_of::<f32>())?;
        Ok(DevBuf { ptr: b.ptr, len, inner: Some(b) })
    }

    pub fn from_host(v: &[f32]) -> Result<DevBuf, String> {
        let b = DevBuf::alloc(v.len())?;
        b.upload(v)?;
        Ok(b)
    }

    pub fn from_bytes_u8(v: &[u8]) -> Result<DevBuf, String> {
        if v.is_empty() {
            return Ok(DevBuf::empty());
        }
        let g = vm::DevBuf::alloc(v.len())?;
        vm::copy_htod(g.ptr, v)?;
        Ok(DevBuf { ptr: g.ptr, len: v.len(), inner: Some(g) })
    }

    pub fn download_bytes_u8(&self, out: &mut [u8]) -> Result<(), String> {
        if out.len() != self.len {
            return Err(format!("download size {} != buffer {}", out.len(), self.len));
        }
        vm::copy_dtoh(out, self.ptr)
    }

    pub fn upload(&self, v: &[f32]) -> Result<(), String> {
        if v.len() != self.len {
            return Err(format!("upload size {} != buffer {}", v.len(), self.len));
        }
        match &self.inner {
            Some(b) => b.upload(v),
            None => Ok(()),
        }
    }

    pub fn download(&self, out: &mut [f32]) -> Result<(), String> {
        if out.len() != self.len {
            return Err(format!("download size {} != buffer {}", out.len(), self.len));
        }
        match &self.inner {
            Some(b) => b.download(out),
            None => Ok(()),
        }
    }
}
