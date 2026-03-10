use super::small100;
use anyhow::{anyhow, Result};
use std::path::PathBuf;

#[derive(Debug)]
pub struct TranslateService {
  runtime: small100::Small100Runtime,
  model_dir: PathBuf,
}

impl TranslateService {
  pub fn init_default() -> Result<Self> {
    Self::init(PathBuf::from("models/small100"))
  }

  pub fn init(model_dir: PathBuf) -> Result<Self> {
    let opts = small100::Small100InitOptions {
      model_dir: model_dir.clone(),
      ..Default::default()
    };
    let runtime = small100::init_runtime(&opts)?;
    Ok(Self { runtime, model_dir })
  }

  pub fn translate(&self, _input: &str, _src_lang: &str, _tgt_lang: &str) -> Result<String> {
    let _ = &self.runtime;
    Err(anyhow!("translation pipeline not implemented yet"))
  }

  pub fn model_dir(&self) -> &PathBuf {
    &self.model_dir
  }
}
