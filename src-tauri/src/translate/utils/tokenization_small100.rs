use anyhow::{anyhow, Context, Result};
use sentencepiece::SentencePieceProcessor;
use serde_json::from_reader;
use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};

const FAIRSEQ_LANGUAGE_CODES: [&str; 100] = [
  "af", "am", "ar", "ast", "az", "ba", "be", "bg", "bn", "br", "bs", "ca", "ceb", "cs", "cy",
  "da", "de", "el", "en", "es", "et", "fa", "ff", "fi", "fr", "fy", "ga", "gd", "gl", "gu", "ha",
  "he", "hi", "hr", "ht", "hu", "hy", "id", "ig", "ilo", "is", "it", "ja", "jv", "ka", "kk", "km",
  "kn", "ko", "lb", "lg", "ln", "lo", "lt", "lv", "mg", "mk", "ml", "mn", "mr", "ms", "my", "ne",
  "nl", "no", "ns", "oc", "or", "pa", "pl", "ps", "pt", "ro", "ru", "sd", "si", "sk", "sl", "so",
  "sq", "sr", "ss", "su", "sv", "sw", "ta", "th", "tl", "tn", "tr", "uk", "ur", "uz", "vi", "wo",
  "xh", "yi", "yo", "zh", "zu",
];

const BOS_TOKEN: &str = "<s>";
const EOS_TOKEN: &str = "</s>";
const PAD_TOKEN: &str = "<pad>";
const UNK_TOKEN: &str = "<unk>";
const NUM_MADEUP_WORDS: usize = 8;

#[derive(Debug, Clone)]
pub struct Small100Tokenizer {
  sp_model: SentencePieceProcessor,
  encoder: HashMap<String, i64>,
  decoder: Vec<String>,
  lang_code_to_token: HashMap<String, String>,
  lang_token_to_id: HashMap<String, i64>,
  lang_code_to_id: HashMap<String, i64>,
  id_to_lang_token: HashMap<i64, String>,
  prefix_tokens: Vec<i64>,
  suffix_tokens: Vec<i64>,
  tgt_lang: String,
  encoder_size: usize,
  bos_token_id: i64,
  eos_token_id: i64,
  pad_token_id: i64,
  unk_token_id: i64,
}

impl Small100Tokenizer {
  pub fn from_model_dir(model_dir: &Path, tgt_lang: &str) -> Result<Self> {
    let vocab_path = model_dir.join("vocab.json");
    let spm_path = model_dir.join("sentencepiece.bpe.model");
    Self::from_files(&vocab_path, &spm_path, tgt_lang)
  }

  pub fn from_files(vocab_path: &Path, spm_path: &Path, tgt_lang: &str) -> Result<Self> {
    let encoder = load_vocab(vocab_path).with_context(|| format!("failed to load vocab: {}", vocab_path.display()))?;
    let decoder = build_decoder(&encoder)?;
    let encoder_size = encoder.len();

    let sp_model = SentencePieceProcessor::open(spm_path)
      .map_err(|e| anyhow!("failed to load sentencepiece model: {e}"))?;

    let (lang_code_to_token, lang_token_to_id, lang_code_to_id, id_to_lang_token) =
      build_language_maps(encoder_size as i64);

    let bos_token_id = *encoder.get(BOS_TOKEN).context("missing <s> in vocab")?;
    let eos_token_id = *encoder.get(EOS_TOKEN).context("missing </s> in vocab")?;
    let pad_token_id = *encoder.get(PAD_TOKEN).context("missing <pad> in vocab")?;
    let unk_token_id = *encoder.get(UNK_TOKEN).context("missing <unk> in vocab")?;

    let mut tokenizer = Self {
      sp_model,
      encoder,
      decoder,
      lang_code_to_token,
      lang_token_to_id,
      lang_code_to_id,
      id_to_lang_token,
      prefix_tokens: Vec::new(),
      suffix_tokens: vec![eos_token_id],
      tgt_lang: tgt_lang.to_string(),
      encoder_size,
      bos_token_id,
      eos_token_id,
      pad_token_id,
      unk_token_id,
    };

    tokenizer.set_tgt_lang(tgt_lang)?;

    Ok(tokenizer)
  }

  pub fn vocab_size(&self) -> usize {
    self.encoder_size + self.lang_token_to_id.len() + NUM_MADEUP_WORDS
  }

  pub fn tgt_lang(&self) -> &str {
    &self.tgt_lang
  }

  pub fn set_tgt_lang(&mut self, lang: &str) -> Result<()> {
    let lang_id = self
      .lang_code_to_id
      .get(lang)
      .copied()
      .with_context(|| format!("unsupported language code: {lang}"))?;
    self.tgt_lang = lang.to_string();
    self.prefix_tokens = vec![lang_id];
    self.suffix_tokens = vec![self.eos_token_id];
    Ok(())
  }

  pub fn get_lang_token(&self, lang: &str) -> Result<&str> {
    self
      .lang_code_to_token
      .get(lang)
      .map(|s| s.as_str())
      .with_context(|| format!("unsupported language code: {lang}"))
  }

  pub fn get_lang_id(&self, lang: &str) -> Result<i64> {
    self
      .lang_code_to_id
      .get(lang)
      .copied()
      .with_context(|| format!("unsupported language code: {lang}"))
  }

  pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<i64>> {
    let pieces = self
      .sp_model
      .encode(text)
      .map_err(|e| anyhow!("sentencepiece encode failed: {e}"))?;

    let mut ids = Vec::with_capacity(pieces.len());
    for piece in pieces {
      let id = self.encoder.get(&piece).copied().unwrap_or(self.unk_token_id);
      ids.push(id);
    }

    if add_special_tokens {
      Ok(self.build_inputs_with_special_tokens(&ids))
    } else {
      Ok(ids)
    }
  }

  pub fn decode(&self, ids: &[i64], skip_special_tokens: bool) -> Result<String> {
    let mut lang_tokens = Vec::new();
    let mut pieces = Vec::new();

    for id in ids {
      if let Some(lang_token) = self.id_to_lang_token.get(id) {
        if !skip_special_tokens {
          lang_tokens.push(lang_token.clone());
        }
        continue;
      }

      if skip_special_tokens && self.is_special_id(*id) {
        continue;
      }

      pieces.push(self.id_to_token(*id));
    }

    let text = self
      .sp_model
      .decode(&pieces)
      .map_err(|e| anyhow!("sentencepiece decode failed: {e}"))?;

    if skip_special_tokens || lang_tokens.is_empty() {
      Ok(text)
    } else if text.is_empty() {
      Ok(lang_tokens.join(" "))
    } else {
      Ok(format!("{} {}", lang_tokens.join(" "), text))
    }
  }

  pub fn build_inputs_with_special_tokens(&self, token_ids: &[i64]) -> Vec<i64> {
    let mut out = Vec::with_capacity(self.prefix_tokens.len() + token_ids.len() + self.suffix_tokens.len());
    out.extend_from_slice(&self.prefix_tokens);
    out.extend_from_slice(token_ids);
    out.extend_from_slice(&self.suffix_tokens);
    out
  }

  pub fn eos_token_id(&self) -> i64 {
    self.eos_token_id
  }

  fn id_to_token(&self, id: i64) -> String {
    if let Some(lang_token) = self.id_to_lang_token.get(&id) {
      return lang_token.clone();
    }

    let index = id as usize;
    if index < self.decoder.len() {
      self.decoder[index].clone()
    } else {
      UNK_TOKEN.to_string()
    }
  }

  fn is_special_id(&self, id: i64) -> bool {
    id == self.bos_token_id || id == self.eos_token_id || id == self.pad_token_id || id == self.unk_token_id
  }

  pub fn supported_languages() -> &'static [&'static str] {
    &FAIRSEQ_LANGUAGE_CODES
  }
}

fn build_language_maps(
  base_size: i64,
) -> (
  HashMap<String, String>,
  HashMap<String, i64>,
  HashMap<String, i64>,
  HashMap<i64, String>,
) {
  let mut lang_code_to_token = HashMap::new();
  let mut lang_token_to_id = HashMap::new();
  let mut lang_code_to_id = HashMap::new();
  let mut id_to_lang_token = HashMap::new();

  for (i, lang_code) in FAIRSEQ_LANGUAGE_CODES.iter().enumerate() {
    let token = format!("__{}__", lang_code);
    let id = base_size + i as i64;
    lang_code_to_token.insert(lang_code.to_string(), token.clone());
    lang_token_to_id.insert(token.clone(), id);
    lang_code_to_id.insert(lang_code.to_string(), id);
    id_to_lang_token.insert(id, token);
  }

  (lang_code_to_token, lang_token_to_id, lang_code_to_id, id_to_lang_token)
}

fn load_vocab(path: &Path) -> Result<HashMap<String, i64>> {
  let file = File::open(path)?;
  let vocab: HashMap<String, i64> = from_reader(file)?;
  Ok(vocab)
}

fn build_decoder(encoder: &HashMap<String, i64>) -> Result<Vec<String>> {
  let mut decoder = vec![String::new(); encoder.len()];
  for (token, id) in encoder {
    let index = *id as usize;
    if index >= decoder.len() {
      return Err(anyhow!("vocab id out of bounds: {id}"));
    }
    decoder[index] = token.clone();
  }

  if decoder.iter().any(|token| token.is_empty()) {
    return Err(anyhow!("vocab has missing ids"));
  }

  Ok(decoder)
}

#[cfg(test)]
mod tests {
  use super::*;

  fn load_tokenizer() -> Small100Tokenizer {
    let model_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/small100");
    Small100Tokenizer::from_model_dir(&model_dir, "en").expect("tokenizer to load")
  }

  #[test]
  fn loads_vocab_and_lang_ids() {
    let tokenizer = load_tokenizer();
    let expected = tokenizer.encoder_size + FAIRSEQ_LANGUAGE_CODES.len() + NUM_MADEUP_WORDS;
    assert_eq!(tokenizer.vocab_size(), expected);
    let en_id = tokenizer.get_lang_id("en").unwrap();
    assert_eq!(en_id, tokenizer.encoder_size as i64 + 18);
  }

  #[test]
  fn encode_adds_prefix_suffix() {
    let mut tokenizer = load_tokenizer();
    tokenizer.set_tgt_lang("fr").unwrap();
    let ids = tokenizer.encode("Hello", true).unwrap();
    assert_eq!(ids.first().copied(), tokenizer.get_lang_id("fr").ok());
    assert_eq!(ids.last().copied(), Some(tokenizer.eos_token_id));
  }

  #[test]
  fn decode_skips_specials() {
    let tokenizer = load_tokenizer();
    let mut ids = Vec::new();
    ids.push(tokenizer.get_lang_id("en").unwrap());
    ids.push(tokenizer.bos_token_id);
    ids.extend(tokenizer.encode("Hello", false).unwrap());
    ids.push(tokenizer.eos_token_id);
    let text = tokenizer.decode(&ids, true).unwrap();
    assert!(!text.contains("__en__"));
  }
}
