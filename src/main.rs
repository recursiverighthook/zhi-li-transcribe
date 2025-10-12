use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::{sync::Arc, net::SocketAddr};
use tokio::sync::mpsc;
use uuid::Uuid;

#[derive(Debug, Serialize, Clone)]
enum JobStatus {
    Pending,
    Processing,
    Completed,
    Failed,
}

#[derive(Debug, Serialize, Clone)]
struct Job {
    id: Uuid,
    file_path: String,
    status: JobStatus,
    result: Option<String>,
}

#[derive(Deserialize)]
struct CreateJobRequest {
    file_path: String,
}

struct AppState {
    job_store: DashMap<Uuid, Job>,
    job_sender: mpsc::Sender<Job>,
}

#[tokio::main]
async fn main() {
    let (tx, mut rx) = mpsc::channel::<Job>(100);

    let shared_state = Arc::new(AppState {
        job_store: DashMap::new(),
        job_sender: tx,
    });


    let worker_state = shared_state.clone();
    tokio::spawn(async move {
        while let Some(job) = rx.recv().await {
            println!("WORKER: Received job {}", job.id);
            worker_state.job_store.entry(job.id).and_modify(|j| j.status = JobStatus::Processing);

            tokio::time::sleep(std::time::Duration::from_secs(15)).await;
            let transcription_result = format!("Transcription complete for {}", job.file_path);

            worker_state.job_store.entry(job.id).and_modify(|j| {
                j.status = JobStatus::Completed;
                j.result = Some(transcription_result);
            });
            println!("WORKER: Finished job {}", job.id);
        }
    });

    let app = Router::new()
        .route("/jobs", post(create_job))
        .route("/jobs/:id", get(get_job_status))
        .with_state(shared_state);

    let addr: SocketAddr = "127.0.0.1:3000".parse().unwrap();
    println!("🚀 Server listening on {}", addr);
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

async fn create_job(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<CreateJobRequest>,
) -> impl IntoResponse {
    let job_id = Uuid::new_v4();
    let new_job = Job {
        id: job_id,
        file_path: payload.file_path,
        status: JobStatus::Pending,
        result: None,
    };

    state.job_store.insert(job_id, new_job.clone());

    if let Err(e) = state.job_sender.send(new_job.clone()).await {
        eprintln!("🔥 Failed to send job to worker: {}", e);
        return (StatusCode::INTERNAL_SERVER_ERROR, "Failed to queue job").into_response();
    }

    (StatusCode::ACCEPTED, Json(new_job)).into_response()
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