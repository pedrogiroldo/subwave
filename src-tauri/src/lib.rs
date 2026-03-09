use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Condvar, Mutex};
use tauri::{Emitter, Manager, State};
use whisper_rs::WhisperContext;

mod audio;
mod stt;

use audio::capture;
use stt::vosk;
use stt::whisper;

struct AppState {
    whisper_ctx: Mutex<Option<WhisperContext>>,
    audio_stream: Mutex<Option<capture::CaptureStream>>,
    is_capturing: Mutex<bool>,
    transcription_queue: Mutex<VecDeque<Vec<f32>>>,
    transcription_condvar: Condvar,
    transcription_worker_started: Mutex<bool>,
    transcription_running: AtomicBool,
}

const MAX_TRANSCRIPTION_QUEUE: usize = 20;

#[derive(Debug, Serialize, Deserialize)]
pub struct AudioDevice {
    pub name: String,
    pub is_default: bool,
}

fn resolve_model_path(input: &str) -> Result<PathBuf, String> {
    let raw = PathBuf::from(input);
    if raw.is_absolute() && raw.exists() {
        return Ok(raw);
    }

    let cwd = std::env::current_dir().map_err(|e| e.to_string())?;
    let direct = cwd.join(&raw);
    if direct.exists() {
        return Ok(direct);
    }

    if let Ok(stripped) = Path::new(input).strip_prefix("src-tauri") {
        let stripped_candidate = cwd.join(stripped);
        if stripped_candidate.exists() {
            return Ok(stripped_candidate);
        }
    }

    let nested = cwd.join("src-tauri").join(&raw);
    if nested.exists() {
        return Ok(nested);
    }

    Err(format!(
        "Model file not found. Tried: '{}' | '{}' | '{}'. Current dir: '{}'",
        raw.display(),
        direct.display(),
        nested.display(),
        cwd.display()
    ))
}

fn enqueue_transcription_chunk(state: &AppState, audio_chunk: Vec<f32>) {
    let mut queue = state
        .transcription_queue
        .lock()
        .unwrap_or_else(|e| e.into_inner());

    if queue.len() >= MAX_TRANSCRIPTION_QUEUE {
        queue.pop_front();
    }

    queue.push_back(audio_chunk);
    state.transcription_condvar.notify_one();
}

#[tauri::command]
fn get_audio_devices() -> Result<Vec<AudioDevice>, String> {
    let sources = capture::list_input_sources()?;
    Ok(sources
        .into_iter()
        .map(|(name, is_default)| AudioDevice { name, is_default })
        .collect())
}

#[tauri::command]
fn load_whisper_model(path: String, state: State<'_, AppState>) -> Result<String, String> {
    let resolved = resolve_model_path(&path)?;
    let resolved_str = resolved
        .to_str()
        .ok_or("Invalid model path encoding")?
        .to_string();

    let ctx = whisper::load_model(&resolved_str).map_err(|e| e.to_string())?;

    let mut whisper_ctx = state.whisper_ctx.lock().map_err(|e| e.to_string())?;
    *whisper_ctx = Some(ctx);

    Ok("Model loaded successfully".to_string())
}

#[tauri::command]
fn is_model_loaded(state: State<'_, AppState>) -> Result<bool, String> {
    let whisper_ctx = state.whisper_ctx.lock().map_err(|e| e.to_string())?;
    Ok(whisper_ctx.is_some())
}

#[tauri::command]
fn transcribe_audio(audio: Vec<f32>, state: State<'_, AppState>) -> Result<String, String> {
    let whisper_ctx = state.whisper_ctx.lock().map_err(|e| e.to_string())?;

    let ctx = whisper_ctx.as_ref().ok_or("Model not loaded")?;

    let text = whisper::transcribe(ctx, &audio).map_err(|e| e.to_string())?;

    Ok(text)
}

#[tauri::command]
fn start_capture(window: tauri::Window, state: State<'_, AppState>) -> Result<String, String> {
    let mut is_capturing = state.is_capturing.lock().map_err(|e| e.to_string())?;

    if *is_capturing {
        return Ok("Already capturing".to_string());
    }

    let (tx, rx) = std::sync::mpsc::channel::<Vec<f32>>();
    let app_handle = window.app_handle().clone();
    let window_label = window.label().to_string();

    if let Err(err) = capture::ensure_loopback_available() {
        return Err(format!(
            "Nenhum monitor de audio do sistema encontrado. Verifique se o PulseAudio/pipewire-pulse esta instalado e se existe um source monitor (ex.: *.monitor). Detalhes: {err}"
        ));
    }

    let stream = capture::start_capture(tx)?;

    state.transcription_running.store(true, Ordering::SeqCst);
    let mut worker_started = state
        .transcription_worker_started
        .lock()
        .map_err(|e| e.to_string())?;

    if !*worker_started {
        *worker_started = true;
        let app_handle_for_worker = app_handle.clone();
        let window_label_for_worker = window_label.clone();
        std::thread::spawn(move || {
            let state = app_handle_for_worker.state::<AppState>();
            loop {
                if !state.transcription_running.load(Ordering::SeqCst) {
                    break;
                }

                let mut queue = match state.transcription_queue.lock() {
                    Ok(guard) => guard,
                    Err(err) => err.into_inner(),
                };

                while queue.is_empty() && state.transcription_running.load(Ordering::SeqCst) {
                    queue = match state.transcription_condvar.wait(queue) {
                        Ok(guard) => guard,
                        Err(err) => err.into_inner(),
                    };
                }

                if !state.transcription_running.load(Ordering::SeqCst) {
                    break;
                }

                let audio_chunk = queue.pop_front();
                drop(queue);

                let Some(audio_chunk) = audio_chunk else {
                    continue;
                };

                let text_result = {
                    let whisper_ctx = match state.whisper_ctx.lock() {
                        Ok(guard) => guard,
                        Err(err) => err.into_inner(),
                    };

                    let Some(ctx) = whisper_ctx.as_ref() else {
                        eprintln!("Model not loaded while transcribing audio");
                        continue;
                    };

                    whisper::transcribe(ctx, &audio_chunk)
                };

                match text_result {
                    Ok(text) => {
                        if text.trim().is_empty() {
                            continue;
                        }
                        if let Some(window) =
                            app_handle_for_worker.get_webview_window(&window_label_for_worker)
                        {
                            if let Err(err) = window.emit("transcription", text) {
                                eprintln!("Failed to emit transcription: {err}");
                            }
                        } else {
                            eprintln!("Transcription window not found");
                        }
                    }
                    Err(err) => {
                        eprintln!("Transcription error: {err}");
                    }
                }
            }
        });
    }

    let app_handle_for_thread = app_handle.clone();
    std::thread::spawn(move || {
        let state = app_handle_for_thread.state::<AppState>();
        while let Ok(audio_chunk) = rx.recv() {
            enqueue_transcription_chunk(&state, audio_chunk);
        }
    });

    let mut audio_stream = state.audio_stream.lock().map_err(|e| e.to_string())?;
    *audio_stream = Some(stream);
    *is_capturing = true;

    Ok("Audio capture started".to_string())
}

#[tauri::command]
fn stop_capture(state: State<'_, AppState>) -> Result<String, String> {
    let mut is_capturing = state.is_capturing.lock().map_err(|e| e.to_string())?;
    let mut audio_stream = state.audio_stream.lock().map_err(|e| e.to_string())?;

    if let Some(stream) = audio_stream.take() {
        drop(stream);
    }

    *is_capturing = false;
    state.transcription_running.store(false, Ordering::SeqCst);
    {
        let mut queue = match state.transcription_queue.lock() {
            Ok(guard) => guard,
            Err(err) => err.into_inner(),
        };
        queue.clear();
        state.transcription_condvar.notify_all();
    }
    if let Ok(mut worker_started) = state.transcription_worker_started.lock() {
        *worker_started = false;
    }

    Ok("Audio capture stopped".to_string())
}

#[tauri::command]
fn is_capturing(state: State<'_, AppState>) -> Result<bool, String> {
    let is_capturing = state.is_capturing.lock().map_err(|e| e.to_string())?;
    Ok(*is_capturing)
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .manage(AppState {
            whisper_ctx: Mutex::new(None),
            audio_stream: Mutex::new(None),
            is_capturing: Mutex::new(false),
            transcription_queue: Mutex::new(VecDeque::new()),
            transcription_condvar: Condvar::new(),
            transcription_worker_started: Mutex::new(false),
            transcription_running: AtomicBool::new(false),
        })
        .invoke_handler(tauri::generate_handler![
            get_audio_devices,
            load_whisper_model,
            is_model_loaded,
            transcribe_audio,
            start_capture,
            stop_capture,
            is_capturing,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
