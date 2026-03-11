use super::small100;
use super::utils::Small100Tokenizer;
use anyhow::{anyhow, Result};
use ort::value::Tensor;
use std::path::PathBuf;

const MAX_INPUT_TOKENS: usize = 256;
const MAX_OUTPUT_TOKENS: usize = 256;
const SUPPORTED_LANGS: [&str; 2] = ["en", "pt"];
const INPUT_IDS_NAME: &str = "input_ids";
const ATTENTION_MASK_NAME: &str = "attention_mask";
const DECODER_INPUT_IDS_NAME: &str = "decoder_input_ids";
const LOGITS_NAME: &str = "logits";

#[derive(Debug)]
pub struct TranslateService {
  runtime: small100::Small100Runtime,
  model_dir: PathBuf,
  tokenizer: Small100Tokenizer,
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
    let tokenizer = Small100Tokenizer::from_model_dir(&model_dir, "en")?;
    Ok(Self {
      runtime,
      model_dir,
      tokenizer,
    })
  }

  pub fn translate(&self, input: &str, src_lang: &str, tgt_lang: &str) -> Result<String> {
    validate_lang(src_lang)?;
    validate_lang(tgt_lang)?;
    self.ensure_expected_io()?;

    let trimmed = input.trim();
    if trimmed.is_empty() {
      return Ok(String::new());
    }

    if src_lang == tgt_lang {
      return Ok(trimmed.to_string());
    }

    let mut tokenizer = self.tokenizer.clone();
    tokenizer.set_tgt_lang(tgt_lang)?;

    let mut input_ids = tokenizer.encode(trimmed, true)?;
    if input_ids.len() > MAX_INPUT_TOKENS {
      input_ids.truncate(MAX_INPUT_TOKENS);
    }
    let attention_mask = vec![1i64; input_ids.len()];

    let mut decoder_ids = vec![tokenizer.eos_token_id()];
    for _ in 0..MAX_OUTPUT_TOKENS {
      let next_id = self.next_token_id(&input_ids, &attention_mask, &decoder_ids)?;
      decoder_ids.push(next_id);
      if next_id == tokenizer.eos_token_id() {
        break;
      }
    }

    tokenizer.decode(&decoder_ids, true)
  }

  pub fn model_dir(&self) -> &PathBuf {
    &self.model_dir
  }

  fn next_token_id(
    &self,
    input_ids: &[i64],
    attention_mask: &[i64],
    decoder_ids: &[i64],
  ) -> Result<i64> {
    let input_tensor =
      Tensor::<i64>::from_array((vec![1usize, input_ids.len()], input_ids.to_vec()))?;
    let mask_tensor =
      Tensor::<i64>::from_array((vec![1usize, attention_mask.len()], attention_mask.to_vec()))?;
    let decoder_tensor =
      Tensor::<i64>::from_array((vec![1usize, decoder_ids.len()], decoder_ids.to_vec()))?;

    let outputs = self.runtime.session.run(ort::inputs! {
      INPUT_IDS_NAME => input_tensor,
      ATTENTION_MASK_NAME => mask_tensor,
      DECODER_INPUT_IDS_NAME => decoder_tensor,
    })?;

    let logits_value = outputs.get(LOGITS_NAME).unwrap_or(&outputs[0]);
    let (shape, logits) = logits_value.try_extract_tensor::<f32>()?;
    if shape.len() != 3 {
      return Err(anyhow!("unexpected logits shape: {shape:?}"));
    }

    let batch = shape[0] as usize;
    let seq_len = shape[1] as usize;
    let vocab = shape[2] as usize;
    if batch != 1 || seq_len == 0 || vocab == 0 {
      return Err(anyhow!("invalid logits shape: {shape:?}"));
    }

    let row_start = (seq_len - 1) * vocab;
    let row_end = row_start + vocab;
    let row = logits
      .get(row_start..row_end)
      .ok_or_else(|| anyhow!("logits slice out of bounds"))?;

    let mut best_id = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for (idx, val) in row.iter().enumerate() {
      if *val > best_val {
        best_val = *val;
        best_id = idx;
      }
    }

    Ok(best_id as i64)
  }
}

fn validate_lang(lang: &str) -> Result<()> {
  if SUPPORTED_LANGS.contains(&lang) {
    Ok(())
  } else {
    Err(anyhow!(
      "unsupported language: {lang}. Supported: {}",
      SUPPORTED_LANGS.join(", ")
    ))
  }
}

impl TranslateService {
  fn ensure_expected_io(&self) -> Result<()> {
    let input_names: Vec<&str> = self.runtime.session.inputs().iter().map(|i| i.name()).collect();
    let mut missing = Vec::new();
    for name in [INPUT_IDS_NAME, ATTENTION_MASK_NAME, DECODER_INPUT_IDS_NAME] {
      if !input_names.iter().any(|n| n == &name) {
        missing.push(name);
      }
    }
    if !missing.is_empty() {
      return Err(anyhow!(
        "model inputs missing {:?}. Available: {:?}",
        missing,
        input_names
      ));
    }

    let output_names: Vec<&str> = self.runtime.session.outputs().iter().map(|o| o.name()).collect();
    if output_names.is_empty() {
      return Err(anyhow!("model has no outputs"));
    }
    Ok(())
  }
}
