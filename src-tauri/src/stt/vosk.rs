use anyhow::{anyhow, Result};
use vosk::{CompleteResult, DecodingState, Model, Recognizer};

const PORTUGUESE_MODEL_PATH: &str = "models/vosk-model-small-pt-0.3";
const TARGET_SAMPLE_RATE: f32 = 16000.0;

fn load_model() -> Result<Model> {
    Model::new(PORTUGUESE_MODEL_PATH)
        .ok_or_else(|| anyhow!("Failed to load Vosk model at '{PORTUGUESE_MODEL_PATH}'"))
}

fn f32_to_i16_samples(samples: &[f32]) -> Vec<i16> {
    samples
        .iter()
        .map(|&sample| {
            let clamped = sample.clamp(-1.0, 1.0);
            (clamped * i16::MAX as f32) as i16
        })
        .collect()
}

pub struct VoskStream {
    _model: Model,
    recognizer: Recognizer,
}

impl VoskStream {
    pub fn new() -> Result<Self> {
        let model = load_model()?;
        let mut recognizer = Recognizer::new(&model, TARGET_SAMPLE_RATE)
            .ok_or_else(|| anyhow!("Failed to create Vosk recognizer"))?;

        recognizer.set_max_alternatives(10);
        recognizer.set_words(true);
        recognizer.set_partial_words(true);

        Ok(Self {
            _model: model,
            recognizer,
        })
    }

    pub fn transcribe_audio_streaming(&mut self, audio: &[f32]) -> Result<String> {
        let pcm = f32_to_i16_samples(audio);

        let state = self
            .recognizer
            .accept_waveform(&pcm)
            .map_err(|e| anyhow!("Vosk accept_waveform failed: {e}"))?;

        match state {
            DecodingState::Running => {
                let partial = self.recognizer.partial_result().partial.trim();
                if partial.is_empty() {
                    Ok(String::new())
                } else {
                    Ok(partial.to_string())
                }
            }
            DecodingState::Finalized => {
                let result = self.recognizer.result();
                let text = match result {
                    CompleteResult::Single(single) => single.text,
                    CompleteResult::Multiple(multiple) => multiple
                        .alternatives
                        .first()
                        .map(|alt| alt.text)
                        .unwrap_or(""),
                };
                Ok(text.trim().to_string())
            }
            DecodingState::Failed => Err(anyhow!("Vosk decoding failed")),
        }
    }
}
