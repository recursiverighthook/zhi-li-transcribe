mod audio;

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
use axum::extract::DefaultBodyLimit;
use tokio_util::io::StreamReader;
use tokio::{sync::mpsc};
use uuid::Uuid;
use futures_util::TryStreamExt;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};


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

    let model_path = audio::get_model().await;
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
            let wav_path = match audio::convert_audio_to_wav(&job.file_path).await {
                Ok(path) => path,
                Err(e) => {
                    eprintln!("Failed to convert audio for job {}: {}", job.id, e);
                    worker_state.job_store.entry(job.id).and_modify(|j| {
                        j.status = JobStatus::Completed; // Or a new "Failed" status
                        j.result = Some(format!("Error: {}", e));
                    });
                    continue;
                }
            };
                let transcript = audio::transcribe_wav(model_path.to_string(), wav_path.to_str().unwrap().to_string()).await;
                worker_state.job_store.entry(job.id).and_modify(|j| {
                j.status = JobStatus::Completed;
                j.result = Some(transcript);
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