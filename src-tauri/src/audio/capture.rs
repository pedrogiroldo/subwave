use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use rubato::{Async, SincInterpolationParameters, SincInterpolationType, WindowFunction};
use std::sync::mpsc;

const TARGET_SAMPLE_RATE: u32 = 16000;
const CHUNK_SECONDS: usize = 4;
const TARGET_SAMPLES: usize = TARGET_SAMPLE_RATE as usize * CHUNK_SECONDS;
const SILENCE_THRESHOLD: f32 = 0.01;

pub fn find_loopback_device() -> Option<cpal::Device> {
    let host = cpal::default_host();
    let devices = host.input_devices().ok()?;

    for device in devices {
        if let Ok(name) = device.description() {
            // Busca por dispositivos de monitoramento nativos do Linux
            if name.to_string().to_lowercase().contains("monitor") {
                return Some(device);
            }
        }
    }
    // Fallback para o dispositivo de entrada padrão
    host.default_input_device()
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
        1, // Mono
        rubato::FixedAsync::Output,
    )
    .expect("Falha ao criar resampler")
}

pub fn is_silence(samples: &[f32], threshold: f32) -> bool {
    let sq_sum: f32 = samples.iter().map(|&s| s * s).sum();
    let rms = (sq_sum / samples.len() as f32).sqrt();
    rms < threshold
}

pub fn start_capture(tx: mpsc::Sender<Vec<f32>>) -> cpal::Stream {
    let device = find_loopback_device().expect("Nenhum dispositivo de áudio encontrado");
    let config = device.default_input_config().unwrap();
    let input_rate = config.sample_rate();

    let mut buffer: Vec<f32> = Vec::with_capacity(TARGET_SAMPLES);

    let err_fn = |err| eprintln!("Erro no stream de áudio: {}", err);

    let stream = device
        .build_input_stream(
            &config.into(),
            move |data: &[f32], _| {
                // 1. Converter para mono se necessário
                let mono_data = if data.len() > 1 {
                    stereo_to_mono(data)
                } else {
                    data.to_vec()
                };

                // 2. Resampling (Simplificado para o exemplo - na prática exige controle de chunks exatos para o Rubato)
                // Aqui você processaria o `mono_data` através do `build_resampler`
                // Assumindo que o output resampleado vai para `resampled_data`
                let resampled_data = mono_data; // Substitua pela lógica real de iteração do Rubato

                buffer.extend_from_slice(&resampled_data);

                // 3. Verifica tamanho do chunk
                if buffer.len() >= TARGET_SAMPLES {
                    if !is_silence(&buffer, SILENCE_THRESHOLD) {
                        let _ = tx.send(buffer.clone());
                    }
                    buffer.clear();
                }
            },
            err_fn,
            None,
        )
        .unwrap();

    stream.play().unwrap();
    stream
}
