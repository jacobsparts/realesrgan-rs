//! Compiles this engine's kernels into two modules: the shared `lightgpu`
//! toolkit's `cuda/kernels.cu` (TOOLKIT_KERNELS) and this project's own
//! `cuda/realesrgan.cu` (PROJECT_KERNELS). Each compiles to its own fatbin with
//! its own `--entries` list and `src/cuda.rs` loads them as separate modules, so
//! neither can shadow a name in the other.
//!
//! A kernel missing from its list is PRUNED from the fatbin and fails at launch
//! rather than at build time, so both lists are checked against the source they
//! are compiled from before nvcc runs: a typo, or a kernel moved between files,
//! fails the build.

/// Generic ops from the shared toolkit.
const TOOLKIT_KERNELS: &[&str] = &[
    "lg_add",
    "lg_copy",
    "lg_scale",
    "lg_noop",
    "lg_add_scaled",
    "lg_lrelu",
    "lg_upsample2x_nearest",
    "lg_pixel_unshuffle2",
    // The Winograd convolution, which is this engine's production 3x3 conv.
    "lg_conv3x3_winograd",
    // Measurement only: `--bench-gemm` times this against the fused conv.
    "lg_f32_gemm_tiled",
];

/// This project's own kernels, in `cuda/realesrgan.cu`: the direct conv fallback.
const PROJECT_KERNELS: &[&str] = &[
    "lg_conv3x3_res",
];

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=cuda/realesrgan.cu");

    let toolkit = lightgpu_build::toolkit_kernels_cu()
        .expect("locate lightgpu's cuda/kernels.cu (set LA_GPU_DIR to override)");
    let toolkit = toolkit.to_string_lossy().into_owned();

    for k in TOOLKIT_KERNELS {
        assert!(
            lightgpu_build::known_kernel(k),
            "unknown kernel `{k}`: not defined by the toolkit - did it move into cuda/realesrgan.cu?"
        );
    }
    let src = std::fs::read_to_string("cuda/realesrgan.cu").expect("read cuda/realesrgan.cu");
    let defined = lightgpu_build::kernel_names_in(&src);
    for k in PROJECT_KERNELS {
        assert!(
            defined.iter().any(|d| d == k),
            "`{k}` is not defined in cuda/realesrgan.cu (it has {})",
            defined.join(", ")
        );
    }

    lightgpu_build::fatbin_modules(&[
        lightgpu_build::Source {
            path: &toolkit,
            out_name: "realesrgan_toolkit.fatbin",
            entries: Some(TOOLKIT_KERNELS),
        },
        lightgpu_build::Source {
            path: "cuda/realesrgan.cu",
            out_name: "realesrgan_project.fatbin",
            entries: Some(PROJECT_KERNELS),
        },
    ]);
}
