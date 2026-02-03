//! WebSocket signaling for WebRTC connection establishment
//!
//! Handles the exchange of SDP offers/answers and ICE candidates
//! between peers via WebSocket connections.

use crate::error::{StreamError, StreamResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{broadcast, mpsc, RwLock};
use tracing::{debug, error, info, warn};

/// Signaling message types
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum SignalingMessage {
    /// Client wants to start a stream session
    Join {
        #[serde(rename = "clientId")]
        client_id: String,
    },

    /// SDP offer from server to client
    Offer {
        sdp: String,
        #[serde(rename = "peerId")]
        peer_id: String,
    },

    /// SDP answer from client to server
    Answer {
        sdp: String,
        #[serde(rename = "peerId")]
        peer_id: String,
    },

    /// ICE candidate exchange
    IceCandidate {
        candidate: serde_json::Value,
        #[serde(rename = "peerId")]
        peer_id: String,
    },

    /// Stream started notification
    StreamStarted {
        #[serde(rename = "peerId")]
        peer_id: String,
        resolution: (u32, u32),
        fps: u32,
    },

    /// Stream stopped notification
    StreamStopped {
        #[serde(rename = "peerId")]
        peer_id: String,
    },

    /// Error message
    Error {
        message: String,
        code: u32,
    },

    /// Ping for keepalive
    Ping,

    /// Pong response
    Pong,

    /// Camera control from client
    CameraControl {
        position: [f64; 3],
        rotation: [f64; 4],
        fov: f64,
    },
}

/// Signaling server for managing WebSocket connections
pub struct SignalingServer {
    /// Active client connections
    clients: Arc<RwLock<HashMap<String, ClientConnection>>>,
    /// Broadcast channel for server events
    event_tx: broadcast::Sender<SignalingMessage>,
}

/// Represents a connected client
struct ClientConnection {
    /// Client ID
    #[allow(dead_code)]
    id: String,
    /// Channel to send messages to this client
    tx: mpsc::Sender<SignalingMessage>,
    /// Associated WebRTC peer ID (if connected)
    #[allow(dead_code)]
    peer_id: Option<String>,
}

impl SignalingServer {
    /// Create a new signaling server
    pub fn new() -> Self {
        let (event_tx, _) = broadcast::channel(100);
        Self {
            clients: Arc::new(RwLock::new(HashMap::new())),
            event_tx,
        }
    }

    /// Register a new client connection
    pub async fn register_client(
        &self,
        client_id: String,
    ) -> (mpsc::Sender<SignalingMessage>, mpsc::Receiver<SignalingMessage>) {
        let (tx, rx) = mpsc::channel::<SignalingMessage>(32);

        {
            let mut clients = self.clients.write().await;
            clients.insert(client_id.clone(), ClientConnection {
                id: client_id.clone(),
                tx: tx.clone(),
                peer_id: None,
            });
        }

        info!("Client {} registered", client_id);
        (tx, rx)
    }

    /// Unregister a client
    pub async fn unregister_client(&self, client_id: &str) {
        let mut clients = self.clients.write().await;
        clients.remove(client_id);
        info!("Client {} unregistered", client_id);
    }

    /// Handle an incoming signaling message
    pub async fn handle_message(
        &self,
        client_id: &str,
        msg: SignalingMessage,
        reply_tx: &mpsc::Sender<SignalingMessage>,
    ) -> StreamResult<()> {
        match msg {
            SignalingMessage::Ping => {
                reply_tx.send(SignalingMessage::Pong).await
                    .map_err(|_| StreamError::ConnectionClosed)?;
            }

            SignalingMessage::Answer { sdp: _, peer_id } => {
                info!("Received SDP answer from {} for peer {}", client_id, peer_id);
            }

            SignalingMessage::IceCandidate { candidate: _, peer_id } => {
                debug!("Received ICE candidate from {} for peer {}", client_id, peer_id);
            }

            SignalingMessage::CameraControl { position, rotation, fov } => {
                debug!("Camera control from {}: pos={:?}, rot={:?}, fov={}", 
                    client_id, position, rotation, fov);
            }

            _ => {
                warn!("Unexpected message type from {}: {:?}", client_id, msg);
            }
        }
        Ok(())
    }

    /// Send a message to a specific client
    pub async fn send_to_client(&self, client_id: &str, msg: SignalingMessage) -> StreamResult<()> {
        let clients = self.clients.read().await;
        if let Some(client) = clients.get(client_id) {
            client.tx.send(msg).await
                .map_err(|_| StreamError::ConnectionClosed)?;
            Ok(())
        } else {
            Err(StreamError::PeerNotFound(client_id.to_string()))
        }
    }

    /// Broadcast a message to all clients
    pub fn broadcast(&self, msg: SignalingMessage) {
        let _ = self.event_tx.send(msg);
    }

    /// Subscribe to broadcast messages
    pub fn subscribe(&self) -> broadcast::Receiver<SignalingMessage> {
        self.event_tx.subscribe()
    }

    /// Get connected client count
    pub async fn client_count(&self) -> usize {
        self.clients.read().await.len()
    }
}

impl Default for SignalingServer {
    fn default() -> Self {
        Self::new()
    }
}
