use std::io::Write;
use std::path::PathBuf;
use std::process::exit;
use hound::SampleFormat;
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
use symphonia::core::errors::Error;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::probe::Hint;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

pub(crate) async fn convert_audio_to_wav(audio_path: &str) -> Result<PathBuf, String> {
    let src = std::fs::File::open(audio_path).map_err(|e| e.to_string())?;
    let mss = MediaSourceStream::new(Box::new(src), Default::default());
    let mut hint = Hint::new();
    let extension = std::path::Path::new(audio_path)
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("mp3");
    hint.with_extension(extension);

    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &Default::default(), &Default::default())
        .map_err(|e| format!("Unsupported format: {}", e))?;

    let mut format = probed.format;

    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or_else(|| "No supported audio tracks found.".to_string())?;

    let channels = match track.codec_params.channels {
        Some(channels) => channels.count() as u16,
        None => 1,
    };

    let sample_rate = match track.codec_params.sample_rate {
        Some(rate) => rate,
        None => return Err("Sample rate not found.".to_string()),
    };

    println!("-> Detected Sample Rate: {}", sample_rate);
    println!("-> Detected Channels: {}", channels);

    let dec_opts: DecoderOptions = Default::default();
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &dec_opts)
        .map_err(|e| format!("Unsupported codec: {}", e))?;

    let track_id = track.id;

    let spec = hound::WavSpec {
        channels,
        sample_rate,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };

    let output_path = std::path::Path::new(audio_path).with_extension("wav");
    let mut writer = hound::WavWriter::create(&output_path, spec)
        .map_err(|e| format!("Failed to create wav writer: {}", e))?;

    loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(Error::IoError(err)) if err.kind() == std::io::ErrorKind::UnexpectedEof => {
                break;
            }
            Err(err) => {
                eprintln!("Error reading packet: {}", err);
                break;
            }
        };

        if packet.track_id() != track_id {
            continue;
        }

        match decoder.decode(&packet) {
            Ok(decoded) => {
                let mut sample_buf =
                    SampleBuffer::<i16>::new(decoded.capacity() as u64, *decoded.spec());
                sample_buf.copy_interleaved_ref(decoded);
                for sample in sample_buf.samples() {
                    writer.write_sample(*sample).expect("failed to write sample");
                }
            }
            Err(Error::DecodeError(err)) => {
                eprintln!("Decode error: {}", err);
                continue;
            }
            Err(err) => {
                eprintln!("An unexpected error occurred during decoding: {}", err);
                break;
            }
        }
    }

    writer.finalize().map_err(|e| format!("Failed to finalize wav file: {}", e))?;

    println!(
        "Successfully converted '{}' to '{}'",
        audio_path,
        output_path.to_string_lossy()
    );
    Ok(output_path)
}

pub(crate) async fn get_model() -> String {
    let model_path = std::path::Path::new("models/ggml-large-v3.bin");
    let path = std::path::Path::new("models");
    if !path.exists() {
        let create_models_dir = std::fs::create_dir_all(&path);
        if create_models_dir.is_err() {
            eprintln!("Failed to create models directory.");
            exit(-4)
        }
    }

    if !model_path.exists() {
        let api = hf_hub::api::sync::Api::new().unwrap();
        let model = api.model("ggerganov/whisper.cpp".to_string()).get("ggml-large-v3.bin").unwrap();
        std::fs::copy(model.as_path(), model_path).unwrap();
        println!("Whisper model Downloaded successfully.");
    }
    model_path.to_str().unwrap().to_string()
}

pub(crate) async fn transcribe_wav(model: String, wav_file: String) -> String {
    let transcription = "".to_string();
    let mut context_param = WhisperContextParameters::default();

    // Enable DTW token level timestamp for known model by using model preset
    context_param.dtw_parameters.mode = whisper_rs::DtwMode::ModelPreset {
        model_preset: whisper_rs::DtwModelPreset::BaseEn,
    };

    // Enable DTW token level timestamp for unknown model by providing custom aheads
    // see details https://github.com/ggerganov/whisper.cpp/pull/1485#discussion_r1519681143
    // values corresponds to ggml-base.en.bin, result will be the same as with DtwModelPreset::BaseEn
    let custom_aheads = [
        (3, 1),
        (4, 2),
        (4, 3),
        (4, 7),
        (5, 1),
        (5, 2),
        (5, 4),
        (5, 6),
    ]
        .map(|(n_text_layer, n_head)| whisper_rs::DtwAhead {
            n_text_layer,
            n_head,
        });
    context_param.dtw_parameters.mode = whisper_rs::DtwMode::Custom {
        aheads: &custom_aheads,
    };

    let ctx = WhisperContext::new_with_params(
        &model,
        context_param,
    )
        .expect("failed to load model");
    // Create a state
    let mut state = ctx.create_state().expect("failed to create key");

    // Create a params object for running the model.
    // The number of past samples to consider defaults to 0.
    let mut params = FullParams::new(SamplingStrategy::BeamSearch {
        beam_size: 5,   // Number of paths to explore
        patience: -1.0,
    });

    // Edit params as needed.
    // Set the number of threads to use to 1.
    //params.set_n_threads(1);
    // Enable translation.
    //params.set_translate(false);
    // Disable anything that prints to stdout.
    params.set_print_special(false);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);
    // Enable token level timestamps
    params.set_token_timestamps(true);

    // Open the audio file.
    let reader = hound::WavReader::open(wav_file).expect("failed to open file");
    #[allow(unused_variables)]
    let hound::WavSpec {
        channels,
        sample_rate,
        bits_per_sample,
        ..
    } = reader.spec();

    // Convert the audio to floating point samples.
    let samples: Vec<i16> = reader
        .into_samples::<i16>()
        .map(|x| x.expect("Invalid sample"))
        .collect();

    let mut audio = vec![0.0f32; samples.len().try_into().unwrap()];

    whisper_rs::convert_integer_to_float_audio(&samples, &mut audio).expect("Conversion error");

    if channels == 2 {
        audio = whisper_rs::convert_stereo_to_mono_audio(&audio).expect("Conversion error");
    } else if channels != 1 {
        panic!(">2 channels unsupported");
    }

    if sample_rate != 16000 { panic!("sample rate must be 16KHz"); }

    // Run the model.
    state.full(params, &audio[..]).expect("failed to run model");

    // Create a file to write the transcript to.
    let mut file = std::fs::File::create("transcript.txt").expect("failed to create file");

    println!("segment length {}", state.as_iter().count());

    // Iterate through the segments of the transcript.
    for segment in state.as_iter() {
        // Get the transcribed text and timestamps for the current segment.
        let start_timestamp = segment.start_timestamp();
        let end_timestamp = segment.end_timestamp();

        let first_token_dtw_ts = segment.get_token(0).map_or(-1, |t| t.token_data().t_dtw);
        println!(
            "[{} - {} ({})]: {}",
            start_timestamp, end_timestamp, first_token_dtw_ts, segment
        );

        // Format the segment information as a string.
        let line = format!("[{} - {}]: {}\n", start_timestamp, end_timestamp, segment);

        // Write the segment information to the file.
        file.write_all(line.as_bytes())
            .expect("failed to write to file");
    }
    transcription
}