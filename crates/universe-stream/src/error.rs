//! Error types for streaming module

use thiserror::Error;

/// Result type for streaming operations
pub type StreamResult<T> = Result<T, StreamError>;

/// Errors that can occur during streaming
#[derive(Error, Debug)]
pub enum StreamError {
    #[error("WebRTC error: {0}")]
    WebRtc(String),

    #[error("Signaling error: {0}")]
    Signaling(String),

    #[error("Encoding error: {0}")]
    Encoding(String),

    #[error("Connection closed")]
    ConnectionClosed,

    #[error("Peer not found: {0}")]
    PeerNotFound(String),

    #[error("Invalid SDP: {0}")]
    InvalidSdp(String),

    #[error("ICE gathering failed: {0}")]
    IceGatheringFailed(String),

    #[error("DTLS handshake failed: {0}")]
    DtlsHandshakeFailed(String),

    #[error("Media track error: {0}")]
    MediaTrack(String),

    #[error("Timeout: {0}")]
    Timeout(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Internal error: {0}")]
    Internal(String),
}

impl From<webrtc::Error> for StreamError {
    fn from(err: webrtc::Error) -> Self {
        StreamError::WebRtc(err.to_string())
    }
}
