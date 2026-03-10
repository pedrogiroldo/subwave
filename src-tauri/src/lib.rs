use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Condvar, Mutex};
use tauri::{Emitter, Manager, State};

mod audio;
mod stt;
pub mod translate;

use audio::capture;
use stt::vosk;

struct AppState {
    vosk_stream: Mutex<Option<vosk::VoskStream>>,
    audio_stream: Mutex<Option<capture::CaptureStream>>,
    is_capturing: Mutex<bool>,
    transcription_queue: Mutex<VecDeque<Vec<f32>>>,
    transcription_condvar: Condvar,
    transcription_worker_started: Mutex<bool>,
    transcription_running: AtomicBool,
    translate_service: Mutex<Option<translate::TranslateService>>,
}

const MAX_TRANSCRIPTION_QUEUE: usize = 20;

#[derive(Debug, Serialize, Deserialize)]
pub struct AudioDevice {
    pub name: String,
    pub is_default: bool,
}

fn enqueue_transcription_chunk(state: &AppState, audio_chunk: Vec<f32>) {
    let mut queue = state
        .transcription_queue
        .lock()
        .unwrap_or_else(|e| e.into_inner());

    while queue.len() >= MAX_TRANSCRIPTION_QUEUE
        && state.transcription_running.load(Ordering::SeqCst)
    {
        queue = match state.transcription_condvar.wait(queue) {
            Ok(guard) => guard,
            Err(err) => err.into_inner(),
        };
    }

    if !state.transcription_running.load(Ordering::SeqCst) {
        return;
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
    let _ = path;
    let stream = vosk::VoskStream::new().map_err(|e| e.to_string())?;
    let mut vosk_stream = state.vosk_stream.lock().map_err(|e| e.to_string())?;
    *vosk_stream = Some(stream);

    Ok("Model loaded successfully".to_string())
}

#[tauri::command]
fn is_model_loaded(state: State<'_, AppState>) -> Result<bool, String> {
    let vosk_stream = state.vosk_stream.lock().map_err(|e| e.to_string())?;
    Ok(vosk_stream.is_some())
}

#[tauri::command]
fn transcribe_audio(audio: Vec<f32>, state: State<'_, AppState>) -> Result<String, String> {
    let mut vosk_stream = state.vosk_stream.lock().map_err(|e| e.to_string())?;
    let stream = vosk_stream.as_mut().ok_or("Model not loaded")?;
    stream
        .transcribe_audio_streaming(&audio)
        .map_err(|e| e.to_string())
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
                state.transcription_condvar.notify_one();

                let Some(audio_chunk) = audio_chunk else {
                    continue;
                };

                let text_result = {
                    let mut vosk_stream = match state.vosk_stream.lock() {
                        Ok(guard) => guard,
                        Err(err) => err.into_inner(),
                    };

                    if vosk_stream.is_none() {
                        match vosk::VoskStream::new() {
                            Ok(stream) => {
                                *vosk_stream = Some(stream);
                            }
                            Err(err) => {
                                eprintln!("Failed to load Vosk model: {err}");
                                continue;
                            }
                        }
                    }

                    let Some(stream) = vosk_stream.as_mut() else {
                        eprintln!("Model not loaded while transcribing audio");
                        continue;
                    };

                    stream.transcribe_audio_streaming(&audio_chunk)
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
fn stop_capture(window: tauri::Window, state: State<'_, AppState>) -> Result<String, String> {
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

    let final_text = {
        let mut vosk_stream = state.vosk_stream.lock().map_err(|e| e.to_string())?;
        if let Some(stream) = vosk_stream.as_mut() {
            stream.finalize().ok()
        } else {
            None
        }
    };

    if let Some(text) = final_text {
        if !text.trim().is_empty() {
            if let Err(err) = window.emit("transcription", text) {
                eprintln!("Failed to emit final transcription: {err}");
            }
        }
    }

    Ok("Audio capture stopped".to_string())
}

#[tauri::command]
fn is_capturing(state: State<'_, AppState>) -> Result<bool, String> {
    let is_capturing = state.is_capturing.lock().map_err(|e| e.to_string())?;
    Ok(*is_capturing)
}

#[tauri::command]
fn init_translate_model(state: State<'_, AppState>) -> Result<String, String> {
    let mut service = state
        .translate_service
        .lock()
        .map_err(|e| e.to_string())?;
    if service.is_some() {
        return Ok("Translate model already loaded".to_string());
    }
    let instance = translate::TranslateService::init_default().map_err(|e| e.to_string())?;
    *service = Some(instance);
    Ok("Translate model loaded".to_string())
}

#[tauri::command]
fn translate_text(
    text: String,
    src_lang: String,
    tgt_lang: String,
    state: State<'_, AppState>,
) -> Result<String, String> {
    let service = state
        .translate_service
        .lock()
        .map_err(|e| e.to_string())?;
    let Some(service) = service.as_ref() else {
        return Err("Translate model not loaded".to_string());
    };
    service
        .translate(&text, &src_lang, &tgt_lang)
        .map_err(|e| e.to_string())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .manage(AppState {
            vosk_stream: Mutex::new(None),
            audio_stream: Mutex::new(None),
            is_capturing: Mutex::new(false),
            transcription_queue: Mutex::new(VecDeque::new()),
            transcription_condvar: Condvar::new(),
            transcription_worker_started: Mutex::new(false),
            transcription_running: AtomicBool::new(false),
            translate_service: Mutex::new(None),
        })
        .invoke_handler(tauri::generate_handler![
            get_audio_devices,
            load_whisper_model,
            is_model_loaded,
            transcribe_audio,
            start_capture,
            stop_capture,
            is_capturing,
            init_translate_model,
            translate_text,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
