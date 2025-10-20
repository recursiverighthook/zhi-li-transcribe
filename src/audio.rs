use std::io::Write;
use std::path::PathBuf;
use std::process::exit;
use hound::SampleFormat;
use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
use symphonia::core::errors::Error;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::probe::Hint;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};
use symphonia::core::audio::{AudioBufferRef, Signal};
use rubato::{FftFixedInOut, Resampler};

fn write_samples_to_wav(
    writer: &mut hound::WavWriter<std::io::BufWriter<std::fs::File>>,
    samples: &[Vec<f32>],
    channels: u16,
) {
    if samples.is_empty() || samples[0].is_empty() {
        return;
    }

    // Interleave the channels and write to the WAV file
    if channels == 1 {
        for sample in &samples[0] {
            let s = (*sample * i16::MAX as f32).clamp(i16::MIN as f32, i16::MAX as f32) as i16;
            writer.write_sample(s).expect("failed to write mono sample");
        }
    } else if channels >= 2 {
        // Assumes stereo for simplicity, but handles >2 channels by taking the first two.
        for i in 0..samples[0].len() {
            let l = (samples[0][i] * i16::MAX as f32).clamp(i16::MIN as f32, i16::MAX as f32) as i16;
            let r = (samples[1][i] * i16::MAX as f32).clamp(i16::MIN as f32, i16::MAX as f32) as i16;
            writer.write_sample(l).expect("failed to write left sample");
            writer.write_sample(r).expect("failed to write right sample");
        }
    }
}

pub(crate) async fn convert_audio_to_wav(audio_path: &str) -> Result<PathBuf, String> {
    // --- Setup: Open file and probe for format ---
    let src = std::fs::File::open(audio_path).map_err(|e| e.to_string())?;
    let mss = MediaSourceStream::new(Box::new(src), Default::default());
    let mut hint = Hint::new();
    if let Some(ext) = std::path::Path::new(audio_path).extension().and_then(|s| s.to_str()) {
        hint.with_extension(ext);
    }
    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &Default::default(), &Default::default())
        .map_err(|e| format!("Unsupported format: {}", e))?;
    let mut format = probed.format;
    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or_else(|| "No supported audio tracks found.".to_string())?;

    // --- Get Audio Parameters ---
    let channels = track.codec_params.channels.map_or(1, |c| c.count() as u16);
    let sample_rate = track.codec_params.sample_rate.ok_or_else(|| "Sample rate not found.".to_string())?;
    let track_id = track.id;

    println!("-> Detected Sample Rate: {}", sample_rate);
    println!("-> Detected Channels: {}", channels);

    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .map_err(|e| format!("Unsupported codec: {}", e))?;

    let target_sample_rate = 16000;
    let spec = hound::WavSpec {
        channels,
        sample_rate: target_sample_rate,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };
    let output_path = std::path::Path::new(audio_path).with_extension("wav");
    let mut writer = hound::WavWriter::create(&output_path, spec)
        .map_err(|e| format!("Failed to create wav writer: {}", e))?;

    const RESAMPLER_CHUNK_SIZE: usize = 2049;
    let mut rubato_resampler = if sample_rate != target_sample_rate {
        Some(FftFixedInOut::<f32>::new(
            sample_rate as usize,
            target_sample_rate as usize,
            RESAMPLER_CHUNK_SIZE,
            channels as usize,
        ).map_err(|e| format!("Failed to create resampler: {}", e))?)
    } else {
        None
    };
    let mut audio_buffer: Vec<Vec<f32>> = (0..channels).map(|_| Vec::new()).collect();
    loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(Error::IoError(err)) if err.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(err) => return Err(format!("Error reading packet: {}", err)),
        };

        if packet.track_id() != track_id {
            continue;
        }

        match decoder.decode(&packet) {
            Ok(decoded) => {
                match decoded {
                    AudioBufferRef::F32(buf) => {
                        for i in 0..channels as usize {
                            audio_buffer[i].extend_from_slice(buf.chan(i));
                        }
                    }
                    AudioBufferRef::S16(buf) => {
                        for i in 0..channels as usize {
                            audio_buffer[i].extend(buf.chan(i).iter().map(|s| *s as f32 / 32767.0));
                        }
                    }
                    AudioBufferRef::S32(buf) => {
                        for i in 0..channels as usize {
                            audio_buffer[i].extend(buf.chan(i).iter().map(|s| *s as f32 / i32::MAX as f32));
                        }
                    }
                    AudioBufferRef::F64(buf) => {
                        for i in 0..channels as usize {
                            audio_buffer[i].extend(buf.chan(i).iter().map(|s| *s as f32));
                        }
                    }
                    _ => {
                        eprintln!("Unsupported sample format, skipping packet.");
                        continue;
                    }
                }

               if let Some(resampler) = rubato_resampler.as_mut() {
                    while audio_buffer.iter().all(|ch| ch.len() >= RESAMPLER_CHUNK_SIZE) {
                        let mut chunk_to_process: Vec<Vec<f32>> = (0..channels as usize)
                            .map(|i| audio_buffer[i].drain(..RESAMPLER_CHUNK_SIZE).collect())
                            .collect();

                        let resampled = resampler.process(&chunk_to_process, None)
                            .map_err(|e| format!("Resampling error: {}", e))?;

                        write_samples_to_wav(&mut writer, &resampled, channels);
                    }
                }
            }
            Err(Error::DecodeError(err)) => eprintln!("Decode error, skipping packet: {}", err),
            Err(err) => return Err(format!("An unexpected error occurred during decoding: {}", err)),
        }
    }

    // --- Finalize: Process any remaining audio in the buffer ---
    if let Some(resampler) = rubato_resampler.as_mut() {
        let remaining_frames = audio_buffer[0].len();
        if remaining_frames > 0 {
            // Pad the last chunk with silence to meet the required chunk size.
            let mut padded_buffer = audio_buffer;
            for chan_buffer in padded_buffer.iter_mut() {
                chan_buffer.resize(RESAMPLER_CHUNK_SIZE, 0.0);
            }
            let resampled = resampler.process(&padded_buffer, None)
                .map_err(|e| format!("Final resampling error: {}", e))?;

            let expected_output_frames = (remaining_frames as f64 * target_sample_rate as f64 / sample_rate as f64).ceil() as usize;

            let truncated_resampled: Vec<Vec<f32>> = resampled.into_iter().map(|mut chan| {
                chan.truncate(expected_output_frames);
                chan
            }).collect();

            write_samples_to_wav(&mut writer, &truncated_resampled, channels);
        }
    } else {
        // If no resampling was done, write the original buffered content.
        write_samples_to_wav(&mut writer, &audio_buffer, channels);
    }

    writer.finalize().map_err(|e| format!("Failed to finalize wav file: {}", e))?;

    println!(
        "Successfully converted '{}' to '{}'",
        audio_path,
        output_path.to_string_lossy()
    );
    Ok(output_path)
}

pub(crate) async fn get_model(model_size: String) -> String {
    let model_name = format!("models/ggml-{}-v3.bin", model_size);
    let model_path = std::path::Path::new(&model_name);
    let path = std::path::Path::new("models");
    if !path.exists() {
        if let Err(e) = std::fs::create_dir_all(&path) {
            eprintln!("Failed to create models directory: {}", e);
            exit(-4)
        }
    }

    if !model_path.exists() {
        println!("Downloading Whisper model...");
        let api = hf_hub::api::sync::Api::new().expect("Failed to create Hugging Face API");
        let model_file = api.model("ggerganov/whisper.cpp".to_string()).get("ggml-large-v3.bin").expect("Failed to download model");
        std::fs::copy(&model_file, model_path).expect("Failed to copy model to destination");
        println!("Whisper model downloaded successfully.");
    }
    model_path.to_str().unwrap().to_string()
}

pub(crate) async fn transcribe_wav(model: String, wav_file: String) -> String {
    let mut transcription = "".to_string();
    let context_param = WhisperContextParameters::default();

    let ctx = WhisperContext::new_with_params(&model, context_param)
        .expect("failed to load model");
    let mut state = ctx.create_state().expect("failed to create key");

    let mut params = FullParams::new(SamplingStrategy::BeamSearch {
        beam_size: 5,
        patience: -1.0,
    });

    params.set_print_special(false);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);
    params.set_token_timestamps(true);

    let reader = hound::WavReader::open(&wav_file).unwrap_or_else(|e| panic!("failed to open file {}: {}", wav_file, e));
    let hound::WavSpec {
        channels,
        sample_rate,
        ..
    } = reader.spec();

    let samples: Vec<i16> = reader
        .into_samples::<i16>()
        .map(|x| x.expect("Invalid sample"))
        .collect();

    let mut audio = vec![0.0f32; samples.len()];
    whisper_rs::convert_integer_to_float_audio(&samples, &mut audio).expect("Conversion error");

    if channels == 2 {
        audio = whisper_rs::convert_stereo_to_mono_audio(&audio).expect("Conversion error");
    } else if channels != 1 {
        panic!(">2 channels unsupported");
    }

    if sample_rate != 16000 {
        panic!("sample rate must be 16KHz, but was {}", sample_rate);
    }

    state.full(params, &audio[..]).expect("failed to run model");

    let mut file = std::fs::File::create("transcript.txt").expect("failed to create file");

    println!("Found {} segments.", state.as_iter().count());

    for segment in state.as_iter() {
        let start_timestamp = segment.start_timestamp();
        let end_timestamp = segment.end_timestamp();
        let text = segment.to_str().expect("failed to convert segments to str");

        println!("[{} - {}]: {}", start_timestamp, end_timestamp, text);

        let line = format!("[{} - {}]: {}\n", start_timestamp, end_timestamp, text);
        file.write_all(line.as_bytes()).expect("failed to write to file");
        transcription = line
    }
    transcription
}

