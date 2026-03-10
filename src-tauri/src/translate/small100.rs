use anyhow::{anyhow, Context, Result};
use ort::session::{builder::GraphOptimizationLevel, Session};
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

const DEFAULT_MODEL_PATH: &str = "models/small100/model.onnx";
const LFS_POINTER_PREFIX: &str = "version https://git-lfs.github.com/spec/v1";

#[derive(Debug, Clone)]
pub(crate) struct Small100Paths {
    pub(crate) model: PathBuf,
}

#[derive(Debug)]
pub(crate) struct Small100Runtime {
    pub(crate) session: Session,
}

#[derive(Debug, Clone)]
pub(crate) struct Small100InitOptions {
    pub(crate) model_dir: PathBuf,
    pub(crate) intra_threads: usize,
    pub(crate) inter_threads: usize,
    pub(crate) optimization: GraphOptimizationLevel,
}

impl Default for Small100InitOptions {
    fn default() -> Self {
        Self {
            model_dir: PathBuf::from(DEFAULT_MODEL_PATH),
            intra_threads: 1,
            inter_threads: 1,
            optimization: GraphOptimizationLevel::Level3,
        }
    }
}

pub(crate) fn init_runtime(opts: &Small100InitOptions) -> Result<Small100Runtime> {
    ensure_ort_initialized();

    let paths = resolve_default_paths(&opts.model_dir);
    validate_model_file(&paths.model).context("model.onnx")?;

    let builder = Session::builder().map_err(|e| anyhow!(e.to_string()))?;
    let builder = builder
        .with_optimization_level(opts.optimization)
        .map_err(|e| anyhow!(e.to_string()))?;
    let builder = builder
        .with_intra_threads(opts.intra_threads)
        .map_err(|e| anyhow!(e.to_string()))?;
    let mut builder = builder
        .with_inter_threads(opts.inter_threads)
        .map_err(|e| anyhow!(e.to_string()))?;
    let session = builder
        .commit_from_file(&paths.model)
        .map_err(|e| anyhow!(e.to_string()))?;

    Ok(Small100Runtime { session })
}

fn resolve_default_paths(model_dir: &Path) -> Small100Paths {
    if model_dir.is_dir() {
        Small100Paths {
            model: model_dir.join("model.onnx"),
        }
    } else {
        Small100Paths {
            model: model_dir.to_path_buf(),
        }
    }
}

fn ensure_ort_initialized() {
    static ORT_INIT: OnceLock<()> = OnceLock::new();
    ORT_INIT.get_or_init(|| {
        let _ = ort::init().commit();
    });
}

fn validate_model_file(path: &Path) -> Result<()> {
    let meta =
        fs::metadata(path).with_context(|| format!("model file not found: {}", path.display()))?;

    if meta.len() < 1024 {
        return Err(anyhow!(
            "model file too small ({} bytes): {}",
            meta.len(),
            path.display()
        ));
    }

    let mut file = fs::File::open(path)
        .with_context(|| format!("failed to open model file: {}", path.display()))?;
    let mut buf = [0u8; 64];
    let n = file.read(&mut buf)?;
    let header = std::str::from_utf8(&buf[..n]).unwrap_or("");
    if header.starts_with(LFS_POINTER_PREFIX) {
        return Err(anyhow!(
            "model file is a Git LFS pointer; run `git lfs pull` for {}",
            path.display()
        ));
    }

    Ok(())
}
