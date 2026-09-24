//! Weights: the Real-ESRGAN reading of a converted `.safetensors` checkpoint.
//!
//! The container itself - `mmap`, the JSON header, tensor offsets and borrowed
//! slices - is `lightgpu::safetensors`, shared with every other engine in the
//! family. What is left here is what is specific to RRDBNet:
//!
//! * the architecture constants (`scale`, `num_feat`, `num_block`,
//!   `num_grow_ch`, `in_ch`, `out_ch`) are read out of `__metadata__`;
//! * every tensor the graph walks is shape-checked against those constants at
//!   load, because a mis-converted checkpoint should fail here and not produce a
//!   plausible-looking image;
//! * `first_in_ch` tells the caller whether the file is the x2 model (which
//!   pre-unshuffles by 2x2, so `conv_first` sees 12 input channels) or the x4
//!   model (3).
//!
//! `get()` returns a zero-copy `&[f32]` into the mapping.
use lightgpu::safetensors::File;

/// The architecture constants the engine needs before it can walk the graph.
#[derive(Clone, Debug)]
pub struct Config {
    pub scale: usize,
    pub num_feat: usize,
    pub num_block: usize,
    pub num_grow_ch: usize,
    pub in_ch: usize,
    pub out_ch: usize,
}

impl Config {
    /// Input channels of `conv_first`: the x2 model pre-unshuffles by 2x2.
    pub fn first_in_ch(&self) -> usize {
        if self.scale == 2 {
            self.in_ch * 4
        } else {
            self.in_ch
        }
    }
}

pub struct Weights {
    pub file: File,
    pub config: Config,
}

impl Weights {
    pub fn open(path: &str) -> Result<Weights, String> {
        let file = File::open(path).map_err(|e| format!("{e} (not a converted Real-ESRGAN file?)"))?;
        let get = |k: &str| -> Result<usize, String> {
            file.metadata_usize(k).map_err(|_| {
                format!("checkpoint metadata is missing `{k}` (not a converted Real-ESRGAN file?)")
            })
        };
        let config = Config {
            scale: get("scale")?,
            num_feat: get("num_feat")?,
            num_block: get("num_block")?,
            num_grow_ch: get("num_grow_ch")?,
            in_ch: get("in_ch")?,
            out_ch: get("out_ch")?,
        };
        let w = Weights { file, config };
        w.check_shapes()?;
        Ok(w)
    }

    pub fn get(&self, name: &str) -> Result<&[f32], String> {
        self.file.f32(name)
    }

    pub fn shape(&self, name: &str) -> Result<&[usize], String> {
        self.file.shape(name)
    }

    /// Sum of the tensor payloads, for the load message.
    pub fn total_bytes(&self) -> usize {
        self.file.payload_bytes()
    }

    /// Check the shapes against the architecture the metadata claims. A
    /// mismatch here means a corrupted or wrongly-converted checkpoint, and
    /// catching it at load is far cheaper than debugging a wrong image.
    fn check_shapes(&self) -> Result<(), String> {
        let c = &self.config;
        let nf = c.num_feat;
        let ng = c.num_grow_ch;
        let cin = c.first_in_ch();

        let expect = |name: &str, want: &[usize]| -> Result<(), String> {
            let got = self.shape(name)?;
            if got != want {
                return Err(format!("{}: shape {:?}, expected {:?}", name, got, want));
            }
            Ok(())
        };

        expect("conv_first.weight", &[nf, cin, 3, 3])?;
        expect("conv_first.bias", &[nf])?;
        for b in 0..c.num_block {
            for r in 1..=3 {
                let base = format!("body.{b}.rdb{r}");
                expect(&format!("{base}.conv1.weight"), &[ng, nf, 3, 3])?;
                expect(&format!("{base}.conv2.weight"), &[ng, nf + ng, 3, 3])?;
                expect(&format!("{base}.conv3.weight"), &[ng, nf + 2 * ng, 3, 3])?;
                expect(&format!("{base}.conv4.weight"), &[ng, nf + 3 * ng, 3, 3])?;
                expect(&format!("{base}.conv5.weight"), &[nf, nf + 4 * ng, 3, 3])?;
            }
        }
        expect("conv_body.weight", &[nf, nf, 3, 3])?;
        expect("conv_up1.weight", &[nf, nf, 3, 3])?;
        expect("conv_up2.weight", &[nf, nf, 3, 3])?;
        expect("conv_hr.weight", &[nf, nf, 3, 3])?;
        expect("conv_last.weight", &[c.out_ch, nf, 3, 3])?;
        Ok(())
    }
}
