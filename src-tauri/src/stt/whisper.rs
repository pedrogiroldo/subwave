use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

pub fn load_model(path: &str) -> anyhow::Result<WhisperContext> {
    let params = WhisperContextParameters::default();
    let ctx = WhisperContext::new_with_params(path, params)?;
    Ok(ctx)
}

pub fn transcribe(ctx: &WhisperContext, audio: &[f32]) -> anyhow::Result<String> {
    let mut state = ctx.create_state()?;

    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
    params.set_language(Some("auto"));
    params.set_no_timestamps(true);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_special(false);

    state.full(params, audio)?;

    let num_segments = state.full_n_segments();
    let mut text = String::new();

    for i in 0..num_segments {
        if let Some(segment) = state.get_segment(i) {
            let segment_text = segment.to_str()?;
            text.push_str(segment_text.trim());
            text.push(' ');
        }
    }

    Ok(text.trim().to_string())
}
