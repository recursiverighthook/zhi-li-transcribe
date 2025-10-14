use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
    extract::Multipart,
    routing::{get, post},
    Json, Router,
};
use dashmap::DashMap;
use serde::Serialize;
use std::{sync::Arc, net::SocketAddr};
use std::io::{Read, Write};
use std::path::PathBuf;
use std::process::exit;
use axum::extract::DefaultBodyLimit;
use tokio_util::io::StreamReader;
use tokio::{sync::mpsc};
use uuid::Uuid;
use futures_util::TryStreamExt;
use hound::{SampleFormat, WavReader};
use symphonia::core::audio::SampleBuffer;
use symphonia::core::errors::Error;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::probe::Hint;
use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
use whisper_rs::{convert_stereo_to_mono_audio, FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};


#[derive(Debug, Serialize, Clone)]
enum JobStatus {
    Pending,
    Processing,
    Completed,
}

#[derive(Debug, Serialize, Clone)]
struct Job {
    id: Uuid,
    file_path: String,
    status: JobStatus,
    result: Option<String>,
}

struct AppState {
    job_store: DashMap<Uuid, Job>,
    job_sender: mpsc::Sender<Job>,
}

#[tokio::main]
async fn main() {

    let model_path = get_model().await;
    let (tx, mut rx) = mpsc::channel::<Job>(100);

    let shared_state = Arc::new(AppState {
        job_store: DashMap::new(),
        job_sender: tx,
    });


    let worker_state = shared_state.clone();
    tokio::spawn(async move {
        while let Some(job) = rx.recv().await {
            println!("WORKER: Received job {}", job.id);
            worker_state
                .job_store
                .entry(job.id)
                .and_modify(|j| j.status = JobStatus::Processing);

            println!("Translating into wav file format");
            let wav_path = match convert_audio_to_wav(&job.file_path).await {
                Ok(path) => path,
                Err(e) => {
                    eprintln!("Failed to convert audio for job {}: {}", job.id, e);
                    worker_state.job_store.entry(job.id).and_modify(|j| {
                        j.status = JobStatus::Completed; // Or a new "Failed" status
                        j.result = Some(format!("Error: {}", e));
                    });
                    continue; // Skip to the next job
                }
            };

            let mut params = FullParams::new(SamplingStrategy::BeamSearch {
                beam_size: 5,
                patience: -1.0,
            });

            params.set_language(Some("en"));

            let audio_file = std::fs::File::open(wav_path).expect("Failed to open WAV file");
            let reader = WavReader::new(audio_file).expect("Failed to create WAV reader");

            #[allow(unused_variables)]
            let hound::WavSpec {
                channels,
                sample_rate,
                bits_per_sample,
                ..
            } = reader.spec();


            let samples: Vec<i16> = reader
                .into_samples::<i16>()
                .map(|x| x.expect("Invalid sample"))
                .collect();
            let mut audio = vec![0.0f32; samples.len().try_into().unwrap()];

            let whisper_context = WhisperContext::new_with_params(&model_path, WhisperContextParameters::default()).unwrap();

            whisper_rs::convert_integer_to_float_audio(&samples, &mut audio).expect("Conversion error");

            let audio = if channels == 1 {
                audio
            } else if channels == 2 {
                let mut output = vec![0.0; audio.len() / 2];
                output = convert_stereo_to_mono_audio(&audio).expect("Conversion error");
                output
            } else {
                panic!(">2 channels unsupported");
            };

            let mut state = whisper_context.create_state().expect("failed to create state");
            let mut file = std::fs::File::create("transcript.txt").expect("failed to create file");
            let st = std::time::Instant::now();
            println!("WORKER: Starting transcription for {}", job.id);
            state.full(params, &audio[..]).expect("Failed to run translation");
            let et = std::time::Instant::now();

            for segment in state.as_iter() {
                let start_timestamp = segment.start_timestamp();
                let end_timestamp = segment.end_timestamp();

                let first_token_dtw_ts = segment.get_token(0).map_or(-1, |t| t.token_data().t_dtw);
                println!(
                    "[{} - {} ({})]: {}",
                    start_timestamp, end_timestamp, first_token_dtw_ts, segment
                );

                let line = format!("[{} - {}]: {}\n", start_timestamp, end_timestamp, segment);
                file.write_all(line.as_bytes())
                    .expect("failed to write to file");
            }
            println!("WORKER: Transcription took: {:?}", et - st);
            println!("WORKER: Finished job {}", job.id);
            worker_state.job_store.entry(job.id).and_modify(|j| {
                j.status = JobStatus::Completed;
                j.result = Some(file.read_to_string(&mut "".to_string()).unwrap().to_string()); // Store the actual result
            });
        }
    });

    let app = Router::new()
        .route("/jobs", post(create_job))
        .route("/jobs/{id}", get(get_job_status))
        .layer(DefaultBodyLimit::disable())
        .with_state(shared_state);

    let addr: SocketAddr = "127.0.0.1:3000".parse().unwrap();
    println!("🚀 Server listening on {}", addr);
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

async fn create_job(
    State(state): State<Arc<AppState>>,
    mut multipart: Multipart,
) -> impl IntoResponse {
    let upload_dir = std::path::Path::new("./uploads");
    if let Err(e) = tokio::fs::create_dir_all(upload_dir).await {
        eprintln!("Failed to create upload directory: {}", e);
        return (StatusCode::INTERNAL_SERVER_ERROR, "Server storage error.").into_response();
    }

    while let Ok(Some(mut field)) = multipart.next_field().await {
        if let Some(original_filename) = field.file_name().map(|name| name.to_string()) {
            let dest_path = upload_dir.join(&original_filename);
            println!("Receiving file: {}", original_filename);
            let file = match tokio::fs::File::create(&dest_path).await {
                Ok(file) => file,
                Err(e) => {
                    eprintln!("Failed to create file on disk: {}", e);
                    return (StatusCode::INTERNAL_SERVER_ERROR, "Could not save file.").into_response();
                }
            };

            if let Err(e) = stream_body_to_file(&mut field, file).await {
                eprintln!("File stream failed: {}", e);
                return (StatusCode::INTERNAL_SERVER_ERROR, "File upload failed mid-stream.").into_response();
            }

            let file_path = match dest_path.to_str() {
                Some(path) => path.to_string(),
                None => {
                    eprintln!("Invalid file path encoding.");
                    return (StatusCode::INTERNAL_SERVER_ERROR, "Invalid file path.").into_response();
                }
            };

            let job_id = Uuid::new_v4();
            let new_job = Job {
                id: job_id,
                file_path,
                status: JobStatus::Pending,
                result: None,
            };

            state.job_store.insert(job_id, new_job.clone());

            if state.job_sender.send(new_job.clone()).await.is_err() {
                eprintln!("Failed to send job to worker channel.");
                return (StatusCode::INTERNAL_SERVER_ERROR, "Failed to queue job.").into_response();
            }

            return (StatusCode::ACCEPTED, Json(new_job)).into_response();
        }
    }

    (StatusCode::BAD_REQUEST, "No video file found in upload.").into_response()
}

async fn stream_body_to_file(
    field: &mut axum::extract::multipart::Field<'_>,
    mut file: tokio::fs::File,
) -> Result<(), std::io::Error> {
    let body_with_io_error =
        field.map_err(|err| std::io::Error::new(std::io::ErrorKind::Other, err));

    let mut field_stream = StreamReader::new(body_with_io_error);
    tokio::io::copy(&mut field_stream, &mut file).await?;
    Ok(())
}


async fn get_job_status(
    State(state): State<Arc<AppState>>,
    Path(job_id): Path<Uuid>,
) -> impl IntoResponse {
    if let Some(job) = state.job_store.get(&job_id) {
        (StatusCode::OK, Json(job.value().clone())).into_response()
    } else {
        StatusCode::NOT_FOUND.into_response()
    }
}

async fn convert_audio_to_wav(audio_path: &str) -> Result<PathBuf, String> {
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

async fn get_model() -> String {
    let model_path = std::path::Path::new("models/ggml-base.en.bin");
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
        let model = api.model("ggerganov/whisper.cpp".to_string()).get("ggml-base.en.bin").unwrap();
        std::fs::copy(model.as_path(), model_path).unwrap();
        println!("Whisper model Downloaded successfully.");
    }
    model_path.to_str().unwrap().to_string()
}