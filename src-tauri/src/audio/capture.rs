use rubato::{Async, Resampler, SincInterpolationParameters, SincInterpolationType, WindowFunction};
use std::sync::mpsc;
use audioadapter_buffers::direct::InterleavedSlice;

const TARGET_SAMPLE_RATE: u32 = 16000;
const CHUNK_MILLIS: usize = 200;
const TARGET_SAMPLES: usize = TARGET_SAMPLE_RATE as usize * CHUNK_MILLIS / 1000;
const SILENCE_THRESHOLD: f32 = 0.01;
const RESAMPLE_CHUNK_FRAMES: usize = 1024;

pub struct ResampleState {
    input_rate: u32,
    resampler: Async<f32>,
    input_buffer: Vec<f32>,
}

impl ResampleState {
    pub fn new(input_rate: u32) -> Self {
        Self {
            input_rate,
            resampler: build_resampler(input_rate, RESAMPLE_CHUNK_FRAMES),
            input_buffer: Vec::new(),
        }
    }
}

pub struct CaptureStream {
    #[cfg(target_os = "linux")]
    running: std::sync::Arc<std::sync::atomic::AtomicBool>,
    #[cfg(target_os = "linux")]
    join_handle: Option<std::thread::JoinHandle<()>>,
    #[cfg(not(target_os = "linux"))]
    stream: cpal::Stream,
}

impl Drop for CaptureStream {
    fn drop(&mut self) {
        #[cfg(target_os = "linux")]
        {
            self.running
                .store(false, std::sync::atomic::Ordering::SeqCst);
            if let Some(handle) = self.join_handle.take() {
                let _ = handle.join();
            }
        }
    }
}

pub fn stereo_to_mono(samples: &[f32]) -> Vec<f32> {
    samples
        .chunks_exact(2)
        .map(|chunk| (chunk[0] + chunk[1]) / 2.0)
        .collect()
}

pub fn build_resampler(input_rate: u32, chunk_size: usize) -> Async<f32> {
    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };

    Async::<f32>::new_sinc(
        TARGET_SAMPLE_RATE as f64 / input_rate as f64,
        2.0,
        &params,
        chunk_size,
        1,
        rubato::FixedAsync::Output,
    )
    .expect("Falha ao criar resampler")
}

pub fn is_silence(samples: &[f32], threshold: f32) -> bool {
    if samples.is_empty() {
        return true;
    }

    let sq_sum: f32 = samples.iter().map(|&s| s * s).sum();
    let rms = (sq_sum / samples.len() as f32).sqrt();
    rms < threshold
}

fn process_input_data(
    data: &[f32],
    channels: usize,
    input_rate: u32,
    resample_state: &mut ResampleState,
    buffer: &mut Vec<f32>,
    tx: &mpsc::Sender<Vec<f32>>,
) {
    let mono_data = if channels > 1 {
        data.chunks_exact(channels)
            .map(|frame| frame.iter().sum::<f32>() / channels as f32)
            .collect::<Vec<f32>>()
    } else {
        data.to_vec()
    };

    let resampled_data = if input_rate == TARGET_SAMPLE_RATE {
        mono_data
    } else {
        resample_state.input_buffer.extend_from_slice(&mono_data);
        let mut output = Vec::new();

        loop {
            let needed = resample_state.resampler.input_frames_next();
            if resample_state.input_buffer.len() < needed {
                break;
            }

            let input_slice = &resample_state.input_buffer[..needed];
            let input_adapter = match InterleavedSlice::new(input_slice, 1, needed) {
                Ok(adapter) => adapter,
                Err(err) => {
                    eprintln!("Erro ao criar adapter de entrada: {err:?}");
                    break;
                }
            };

            let resampled = match resample_state.resampler.process(&input_adapter, 0, None) {
                Ok(buffer) => buffer,
                Err(err) => {
                    eprintln!("Erro ao resamplear audio: {err:?}");
                    break;
                }
            };

            output.extend_from_slice(&resampled.take_data());
            resample_state.input_buffer.drain(..needed);
        }

        output
    };

    buffer.extend_from_slice(&resampled_data);

    if buffer.len() >= TARGET_SAMPLES {
        if !is_silence(buffer, SILENCE_THRESHOLD) {
            let chunk = std::mem::take(buffer);
            let _ = tx.send(chunk);
        } else {
            buffer.clear();
        }
    }
}

#[cfg(target_os = "linux")]
mod pulse_backend {
    use super::{
        process_input_data, CaptureStream, ResampleState, TARGET_SAMPLES, TARGET_SAMPLE_RATE,
    };
    use libpulse_binding::callbacks::ListResult;
    use libpulse_binding::context::{Context, FlagSet as ContextFlagSet};
    use libpulse_binding::mainloop::standard::Mainloop;
    use libpulse_binding::sample::{Format, Spec};
    use libpulse_binding::stream::Direction;
    use libpulse_simple_binding::Simple;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{mpsc, Arc, Mutex};
    use std::thread;
    use std::time::Duration;

    fn wait_for_context_ready(mainloop: &mut Mainloop, context: &Context) -> Result<(), String> {
        loop {
            match context.get_state() {
                libpulse_binding::context::State::Ready => return Ok(()),
                libpulse_binding::context::State::Failed => {
                    return Err("Falha ao conectar ao PulseAudio".to_string())
                }
                libpulse_binding::context::State::Terminated => {
                    return Err("Conexao com PulseAudio foi encerrada".to_string())
                }
                _ => {}
            }

            if mainloop.iterate(false).is_error() {
                return Err("Falha no mainloop do PulseAudio".to_string());
            }
            thread::sleep(Duration::from_millis(10));
        }
    }

    fn with_context<T>(
        f: impl FnOnce(&mut Mainloop, &Context) -> Result<T, String>,
    ) -> Result<T, String> {
        let mut mainloop = Mainloop::new().ok_or("Falha ao criar mainloop do PulseAudio")?;
        let mut context = Context::new(&mainloop, "subwave").ok_or("Falha ao criar contexto")?;
        context
            .connect(None, ContextFlagSet::NOFLAGS, None)
            .map_err(|e| format!("Falha ao conectar ao PulseAudio: {e}"))?;

        wait_for_context_ready(&mut mainloop, &context)?;

        let result = f(&mut mainloop, &context);
        context.disconnect();
        result
    }

    fn get_default_sink_name(mainloop: &mut Mainloop, context: &Context) -> Result<String, String> {
        let default_sink = Arc::new(Mutex::new(None::<String>));
        let done = Arc::new(AtomicBool::new(false));

        let default_sink_clone = Arc::clone(&default_sink);
        let done_clone = Arc::clone(&done);
        context.introspect().get_server_info(move |info| {
            let name = info
                .default_sink_name
                .as_ref()
                .map(|value| value.to_string());
            *default_sink_clone.lock().unwrap() = name;
            done_clone.store(true, Ordering::SeqCst);
        });

        while !done.load(Ordering::SeqCst) {
            if mainloop.iterate(false).is_error() {
                return Err("Falha no mainloop ao obter sink default".to_string());
            }
            thread::sleep(Duration::from_millis(10));
        }

        let default_sink_name = default_sink.lock().unwrap().clone();
        default_sink_name.ok_or_else(|| "Nenhum sink default encontrado".to_string())
    }

    fn get_monitor_for_sink(
        mainloop: &mut Mainloop,
        context: &Context,
        sink_name: &str,
    ) -> Result<Option<String>, String> {
        let monitor_name = Arc::new(Mutex::new(None::<String>));
        let done = Arc::new(AtomicBool::new(false));

        let monitor_name_clone = Arc::clone(&monitor_name);
        let done_clone = Arc::clone(&done);

        context
            .introspect()
            .get_sink_info_by_name(sink_name, move |result| match result {
                ListResult::Item(info) => {
                    let name = info
                        .monitor_source_name
                        .as_ref()
                        .map(|value| value.to_string());
                    *monitor_name_clone.lock().unwrap() = name;
                }
                ListResult::End => {
                    done_clone.store(true, Ordering::SeqCst);
                }
                ListResult::Error => {
                    done_clone.store(true, Ordering::SeqCst);
                }
            });

        while !done.load(Ordering::SeqCst) {
            if mainloop.iterate(false).is_error() {
                return Err("Falha no mainloop ao obter monitor".to_string());
            }
            thread::sleep(Duration::from_millis(10));
        }

        let monitor = monitor_name.lock().unwrap().clone();
        Ok(monitor)
    }

    fn find_any_monitor_source(
        mainloop: &mut Mainloop,
        context: &Context,
    ) -> Result<Option<String>, String> {
        let sources = Arc::new(Mutex::new(Vec::new()));
        let done = Arc::new(AtomicBool::new(false));

        let sources_clone = Arc::clone(&sources);
        let done_clone = Arc::clone(&done);

        context
            .introspect()
            .get_source_info_list(move |result| match result {
                ListResult::Item(info) => {
                    if let Some(name) = info.name.as_ref() {
                        sources_clone.lock().unwrap().push(name.to_string());
                    }
                }
                ListResult::End => {
                    done_clone.store(true, Ordering::SeqCst);
                }
                ListResult::Error => {
                    done_clone.store(true, Ordering::SeqCst);
                }
            });

        while !done.load(Ordering::SeqCst) {
            if mainloop.iterate(false).is_error() {
                return Err("Falha no mainloop ao listar sources".to_string());
            }
            thread::sleep(Duration::from_millis(10));
        }

        let sources = sources.lock().unwrap();
        let monitor = sources
            .iter()
            .find(|name| name.contains(".monitor") || name.contains("monitor"));
        Ok(monitor.cloned())
    }

    pub fn find_monitor_source_name() -> Result<String, String> {
        with_context(|mainloop, context| {
            let default_sink = get_default_sink_name(mainloop, context)?;
            if let Some(monitor) = get_monitor_for_sink(mainloop, context, &default_sink)? {
                return Ok(monitor);
            }

            if let Some(monitor) = find_any_monitor_source(mainloop, context)? {
                return Ok(monitor);
            }

            Err("Nenhum source monitor encontrado".to_string())
        })
    }

    pub fn list_sources_with_default() -> Result<Vec<(String, bool)>, String> {
        with_context(|mainloop, context| {
            let default_source = {
                let default_source = Arc::new(Mutex::new(None::<String>));
                let done = Arc::new(AtomicBool::new(false));
                let default_source_clone = Arc::clone(&default_source);
                let done_clone = Arc::clone(&done);
                context.introspect().get_server_info(move |info| {
                    let name = info
                        .default_source_name
                        .as_ref()
                        .map(|value| value.to_string());
                    *default_source_clone.lock().unwrap() = name;
                    done_clone.store(true, Ordering::SeqCst);
                });

                while !done.load(Ordering::SeqCst) {
                    if mainloop.iterate(false).is_error() {
                        return Err("Falha no mainloop ao obter source default".to_string());
                    }
                    thread::sleep(Duration::from_millis(10));
                }

                let default_source_name = default_source.lock().unwrap().clone();
                default_source_name
            };

            let sources = Arc::new(Mutex::new(Vec::new()));
            let done = Arc::new(AtomicBool::new(false));
            let sources_clone = Arc::clone(&sources);
            let done_clone = Arc::clone(&done);

            context
                .introspect()
                .get_source_info_list(move |result| match result {
                    ListResult::Item(info) => {
                        if let Some(name) = info.name.as_ref() {
                            sources_clone.lock().unwrap().push(name.to_string());
                        }
                    }
                    ListResult::End => {
                        done_clone.store(true, Ordering::SeqCst);
                    }
                    ListResult::Error => {
                        done_clone.store(true, Ordering::SeqCst);
                    }
                });

            while !done.load(Ordering::SeqCst) {
                if mainloop.iterate(false).is_error() {
                    return Err("Falha no mainloop ao listar sources".to_string());
                }
                thread::sleep(Duration::from_millis(10));
            }

            let sources = sources.lock().unwrap();
            let mut result = Vec::new();
            for name in sources.iter() {
                let is_default = default_source
                    .as_ref()
                    .map(|default_name| default_name == name)
                    .unwrap_or(false);
                result.push((name.clone(), is_default));
            }
            Ok(result)
        })
    }

    pub fn start_capture(tx: mpsc::Sender<Vec<f32>>) -> Result<CaptureStream, String> {
        let monitor_name = find_monitor_source_name()?;

        let sample_spec = Spec {
            format: Format::S16le,
            channels: 1,
            rate: TARGET_SAMPLE_RATE,
        };

        if !sample_spec.is_valid() {
            return Err("Formato de captura invalido".to_string());
        }

        let simple = Simple::new(
            None,
            "subwave",
            Direction::Record,
            Some(&monitor_name),
            "monitor",
            &sample_spec,
            None,
            None,
        )
        .map_err(|e| format!("Falha ao abrir stream do PulseAudio: {e}"))?;

        let running = Arc::new(AtomicBool::new(true));
        let running_clone = Arc::clone(&running);
        let join_handle = thread::spawn(move || {
            let mut buffer: Vec<u8> = vec![0; 2048];
            let mut float_buffer: Vec<f32> = Vec::with_capacity(TARGET_SAMPLES);
            let mut resample_state = ResampleState::new(TARGET_SAMPLE_RATE);
            while running_clone.load(Ordering::SeqCst) {
                if let Err(err) = simple.read(&mut buffer) {
                    eprintln!("Erro ao ler stream PulseAudio: {err}");
                    break;
                }

                let mut converted: Vec<f32> = Vec::with_capacity(buffer.len() / 2);
                for chunk in buffer.chunks_exact(2) {
                    let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
                    converted.push(sample as f32 / i16::MAX as f32);
                }
                process_input_data(
                    &converted,
                    1,
                    TARGET_SAMPLE_RATE,
                    &mut resample_state,
                    &mut float_buffer,
                    &tx,
                );
            }
        });

        Ok(CaptureStream {
            running,
            join_handle: Some(join_handle),
        })
    }
}

#[cfg(target_os = "linux")]
pub fn ensure_loopback_available() -> Result<(), String> {
    pulse_backend::find_monitor_source_name().map(|_| ())
}

#[cfg(target_os = "linux")]
pub fn list_input_sources() -> Result<Vec<(String, bool)>, String> {
    pulse_backend::list_sources_with_default()
}

#[cfg(target_os = "linux")]
pub fn start_capture(tx: mpsc::Sender<Vec<f32>>) -> Result<CaptureStream, String> {
    pulse_backend::start_capture(tx)
}

#[cfg(not(target_os = "linux"))]
pub fn ensure_loopback_available() -> Result<(), String> {
    find_loopback_device()
        .map(|_| ())
        .ok_or_else(|| "Nenhum dispositivo de loopback/monitor encontrado".to_string())
}

#[cfg(not(target_os = "linux"))]
pub fn list_input_sources() -> Result<Vec<(String, bool)>, String> {
    let host = cpal::default_host();
    let mut devices = Vec::new();

    if let Ok(device_list) = host.input_devices() {
        for (idx, _device) in device_list.enumerate() {
            let name = format!("Device {idx}");
            devices.push((name, false));
        }
    }

    Ok(devices)
}

#[cfg(not(target_os = "linux"))]
pub fn start_capture(tx: mpsc::Sender<Vec<f32>>) -> Result<CaptureStream, String> {
    use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
    use cpal::SampleFormat;

    let device = find_loopback_device()
        .ok_or_else(|| "Nenhum dispositivo de audio encontrado".to_string())?;
    let supported_config = device.default_input_config().map_err(|e| e.to_string())?;
    let sample_format = supported_config.sample_format();
    let config: cpal::StreamConfig = supported_config.into();
    let channels = config.channels as usize;
    let input_rate = config.sample_rate.0;

    let err_fn = |err| eprintln!("Erro no stream de audio: {}", err);

    let stream = match sample_format {
        SampleFormat::F32 => {
            let tx = tx.clone();
            let mut buffer: Vec<f32> = Vec::with_capacity(TARGET_SAMPLES);
            let mut resample_state = ResampleState::new(input_rate);
            device
                .build_input_stream(
                    &config,
                    move |data: &[f32], _| {
                        process_input_data(
                            data,
                            channels,
                            input_rate,
                            &mut resample_state,
                            &mut buffer,
                            &tx,
                        );
                    },
                    err_fn,
                    None,
                )
                .map_err(|e| e.to_string())?
        }
        SampleFormat::I16 => {
            let tx = tx.clone();
            let mut buffer: Vec<f32> = Vec::with_capacity(TARGET_SAMPLES);
            let mut resample_state = ResampleState::new(input_rate);
            device
                .build_input_stream(
                    &config,
                    move |data: &[i16], _| {
                        let converted: Vec<f32> = data
                            .iter()
                            .map(|&sample| sample as f32 / i16::MAX as f32)
                            .collect();
                        process_input_data(
                            &converted,
                            channels,
                            input_rate,
                            &mut resample_state,
                            &mut buffer,
                            &tx,
                        );
                    },
                    err_fn,
                    None,
                )
                .map_err(|e| e.to_string())?
        }
        SampleFormat::U16 => {
            let mut buffer: Vec<f32> = Vec::with_capacity(TARGET_SAMPLES);
            let mut resample_state = ResampleState::new(input_rate);
            device
                .build_input_stream(
                    &config,
                    move |data: &[u16], _| {
                        let converted: Vec<f32> = data
                            .iter()
                            .map(|&sample| (sample as f32 / u16::MAX as f32) * 2.0 - 1.0)
                            .collect();
                        process_input_data(
                            &converted,
                            channels,
                            input_rate,
                            &mut resample_state,
                            &mut buffer,
                            &tx,
                        );
                    },
                    err_fn,
                    None,
                )
                .map_err(|e| e.to_string())?
        }
        _ => return Err("Formato de amostra nao suportado".to_string()),
    };

    stream.play().map_err(|e| e.to_string())?;

    Ok(CaptureStream { stream })
}

#[cfg(not(target_os = "linux"))]
fn find_loopback_device() -> Option<cpal::Device> {
    let host = cpal::default_host();
    let devices: Vec<cpal::Device> = match host.input_devices() {
        Ok(devices) => devices.collect(),
        Err(err) => {
            eprintln!("Falha ao listar dispositivos de entrada: {err}");
            return None;
        }
    };

    let keywords = [
        "monitor",
        "loopback",
        "stereo mix",
        "what u hear",
        "monitor of",
        "alsa_output",
        ".monitor",
    ];

    for device in devices {
        let description = match device.description() {
            Ok(description) => description,
            Err(err) => {
                eprintln!("Falha ao obter descricao do dispositivo: {err}");
                continue;
            }
        };

        let lower = description.to_string().to_lowercase();
        if keywords.iter().any(|keyword| lower.contains(keyword)) {
            return Some(device);
        }
    }

    None
}
