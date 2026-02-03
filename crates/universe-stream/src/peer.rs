//! WebRTC peer connection management
//!
//! Handles individual WebRTC peer connections, including:
//! - ICE candidate gathering and exchange
//! - DTLS handshake
//! - SRTP media encryption
//! - Video track management

use crate::error::{StreamError, StreamResult};
use parking_lot::RwLock;
use std::sync::Arc;
use tokio::sync::mpsc;
use webrtc::api::media_engine::MediaEngine;
use webrtc::api::APIBuilder;
use webrtc::ice_transport::ice_connection_state::RTCIceConnectionState;
use webrtc::ice_transport::ice_server::RTCIceServer;
use webrtc::peer_connection::configuration::RTCConfiguration;
use webrtc::peer_connection::peer_connection_state::RTCPeerConnectionState;
use webrtc::peer_connection::sdp::session_description::RTCSessionDescription;
use webrtc::peer_connection::RTCPeerConnection;
use webrtc::rtp_transceiver::rtp_codec::RTCRtpCodecCapability;
use webrtc::track::track_local::track_local_static_rtp::TrackLocalStaticRTP;
use webrtc::track::track_local::{TrackLocal, TrackLocalWriter};

/// Configuration for WebRTC peer connections
#[derive(Clone, Debug)]
pub struct PeerConfig {
    /// STUN servers for NAT traversal
    pub stun_servers: Vec<String>,
    /// TURN servers for relay (optional)
    pub turn_servers: Vec<TurnServer>,
    /// Video codec preference
    pub video_codec: VideoCodec,
    /// Maximum bitrate in kbps
    pub max_bitrate_kbps: u32,
    /// Target frame rate
    pub target_fps: u32,
}

/// TURN server configuration
#[derive(Clone, Debug)]
pub struct TurnServer {
    pub url: String,
    pub username: String,
    pub credential: String,
}

/// Supported video codecs
#[derive(Clone, Debug, Default)]
pub enum VideoCodec {
    #[default]
    H264,
    VP8,
    VP9,
    AV1,
}

impl Default for PeerConfig {
    fn default() -> Self {
        Self {
            stun_servers: vec![
                "stun:stun.l.google.com:19302".to_string(),
                "stun:stun1.l.google.com:19302".to_string(),
            ],
            turn_servers: Vec::new(),
            video_codec: VideoCodec::H264,
            max_bitrate_kbps: 15000, // 15 Mbps
            target_fps: 60,
        }
    }
}

/// State of a WebRTC peer connection
#[derive(Clone, Debug, PartialEq)]
pub enum PeerState {
    /// Initial state
    New,
    /// ICE gathering in progress
    Connecting,
    /// Connected and ready for media
    Connected,
    /// Connection failed
    Failed(String),
    /// Connection closed
    Closed,
}

/// Represents a WebRTC peer connection
pub struct WebRtcPeer {
    /// Unique peer ID
    pub id: String,
    /// Underlying WebRTC connection
    peer_connection: Arc<RTCPeerConnection>,
    /// Video track for sending frames
    video_track: Arc<TrackLocalStaticRTP>,
    /// Current connection state
    state: Arc<RwLock<PeerState>>,
    /// Channel for sending ICE candidates to signaling
    ice_candidate_tx: mpsc::Sender<String>,
    /// Configuration
    config: PeerConfig,
}

impl WebRtcPeer {
    /// Create a new WebRTC peer
    pub async fn new(
        id: String,
        config: PeerConfig,
        ice_candidate_tx: mpsc::Sender<String>,
    ) -> StreamResult<Self> {
        // Create media engine with H.264 codec
        let mut media_engine = MediaEngine::default();
        
        let codec_capability = match config.video_codec {
            VideoCodec::H264 => RTCRtpCodecCapability {
                mime_type: "video/H264".to_string(),
                clock_rate: 90000,
                channels: 0,
                sdp_fmtp_line: "level-asymmetry-allowed=1;packetization-mode=1;profile-level-id=42e01f".to_string(),
                rtcp_feedback: vec![],
            },
            VideoCodec::VP8 => RTCRtpCodecCapability {
                mime_type: "video/VP8".to_string(),
                clock_rate: 90000,
                channels: 0,
                sdp_fmtp_line: String::new(),
                rtcp_feedback: vec![],
            },
            VideoCodec::VP9 => RTCRtpCodecCapability {
                mime_type: "video/VP9".to_string(),
                clock_rate: 90000,
                channels: 0,
                sdp_fmtp_line: String::new(),
                rtcp_feedback: vec![],
            },
            VideoCodec::AV1 => RTCRtpCodecCapability {
                mime_type: "video/AV1".to_string(),
                clock_rate: 90000,
                channels: 0,
                sdp_fmtp_line: String::new(),
                rtcp_feedback: vec![],
            },
        };

        media_engine.register_codec(
            webrtc::rtp_transceiver::rtp_codec::RTCRtpCodecParameters {
                capability: codec_capability.clone(),
                payload_type: 96,
                stats_id: String::new(),
            },
            webrtc::rtp_transceiver::rtp_codec::RTPCodecType::Video,
        )?;

        // Build API
        let api = APIBuilder::new()
            .with_media_engine(media_engine)
            .build();

        // Configure ICE servers
        let mut ice_servers = Vec::new();
        
        for stun_url in &config.stun_servers {
            ice_servers.push(RTCIceServer {
                urls: vec![stun_url.clone()],
                ..Default::default()
            });
        }

        for turn in &config.turn_servers {
            ice_servers.push(RTCIceServer {
                urls: vec![turn.url.clone()],
                username: turn.username.clone(),
                credential: turn.credential.clone(),
                ..Default::default()
            });
        }

        let rtc_config = RTCConfiguration {
            ice_servers,
            ..Default::default()
        };

        // Create peer connection
        let peer_connection = Arc::new(api.new_peer_connection(rtc_config).await?);

        // Create video track
        let video_track = Arc::new(TrackLocalStaticRTP::new(
            codec_capability,
            "video".to_string(),
            "helios-stream".to_string(),
        ));

        // Add track to peer connection
        peer_connection
            .add_track(Arc::clone(&video_track) as Arc<dyn TrackLocal + Send + Sync>)
            .await?;

        let state = Arc::new(RwLock::new(PeerState::New));
        let state_clone = Arc::clone(&state);

        // Set up connection state callback
        peer_connection.on_peer_connection_state_change(Box::new(move |s: RTCPeerConnectionState| {
            let state = Arc::clone(&state_clone);
            Box::pin(async move {
                let new_state = match s {
                    RTCPeerConnectionState::New => PeerState::New,
                    RTCPeerConnectionState::Connecting => PeerState::Connecting,
                    RTCPeerConnectionState::Connected => PeerState::Connected,
                    RTCPeerConnectionState::Failed => PeerState::Failed("Connection failed".to_string()),
                    RTCPeerConnectionState::Closed => PeerState::Closed,
                    RTCPeerConnectionState::Disconnected => PeerState::Failed("Disconnected".to_string()),
                    _ => return,
                };
                *state.write() = new_state;
            })
        }));

        // Set up ICE candidate callback
        let ice_tx = ice_candidate_tx.clone();
        let peer_id = id.clone();
        peer_connection.on_ice_candidate(Box::new(move |candidate| {
            let tx = ice_tx.clone();
            let id = peer_id.clone();
            Box::pin(async move {
                if let Some(c) = candidate {
                    if let Ok(json) = c.to_json() {
                        let msg = serde_json::json!({
                            "type": "ice-candidate",
                            "peerId": id,
                            "candidate": json,
                        });
                        let _ = tx.send(msg.to_string()).await;
                    }
                }
            })
        }));

        Ok(Self {
            id,
            peer_connection,
            video_track,
            state,
            ice_candidate_tx,
            config,
        })
    }

    /// Create SDP offer
    pub async fn create_offer(&self) -> StreamResult<RTCSessionDescription> {
        let offer = self.peer_connection.create_offer(None).await?;
        self.peer_connection.set_local_description(offer.clone()).await?;
        Ok(offer)
    }

    /// Handle incoming SDP offer and create answer
    pub async fn handle_offer(&self, offer: RTCSessionDescription) -> StreamResult<RTCSessionDescription> {
        self.peer_connection.set_remote_description(offer).await?;
        let answer = self.peer_connection.create_answer(None).await?;
        self.peer_connection.set_local_description(answer.clone()).await?;
        Ok(answer)
    }

    /// Handle incoming SDP answer
    pub async fn handle_answer(&self, answer: RTCSessionDescription) -> StreamResult<()> {
        self.peer_connection.set_remote_description(answer).await?;
        Ok(())
    }

    /// Add ICE candidate
    pub async fn add_ice_candidate(&self, candidate_json: &str) -> StreamResult<()> {
        let candidate: webrtc::ice_transport::ice_candidate::RTCIceCandidateInit =
            serde_json::from_str(candidate_json)?;
        self.peer_connection.add_ice_candidate(candidate).await?;
        Ok(())
    }

    /// Send RTP packet on video track
    pub async fn send_rtp(&self, data: &[u8]) -> StreamResult<()> {
        let result = self.video_track.write(data).await;
        match result {
            Ok(_) => Ok(()),
            Err(e) => Err(StreamError::MediaTrack(format!("{:?}", e))),
        }
    }

    /// Get current connection state
    pub fn state(&self) -> PeerState {
        self.state.read().clone()
    }

    /// Check if connected
    pub fn is_connected(&self) -> bool {
        matches!(*self.state.read(), PeerState::Connected)
    }

    /// Close the connection
    pub async fn close(&self) -> StreamResult<()> {
        self.peer_connection.close().await?;
        *self.state.write() = PeerState::Closed;
        Ok(())
    }

    /// Get peer ID
    pub fn id(&self) -> &str {
        &self.id
    }
}
