//! WebSocket streaming server (MJPEG-over-WebSocket)

use crate::encoder::{EncodedFrame, EncodedH264Frame, H264Config};
use crate::input::{InputEvent, OutputEvent, parse_input, serialize_output};

use axum::{
    extract::{Query, State, WebSocketUpgrade, ws::{Message, WebSocket}},
    response::{IntoResponse, Html},
    routing::get,
    Router,
};
use tower_http::cors::{CorsLayer, Any};
use tower_http::services::ServeDir;
use std::sync::Arc;
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tokio::sync::{broadcast, mpsc, watch};
use parking_lot::RwLock;
use futures::{StreamExt, SinkExt};
use base64::Engine as _;

const ANON_JUMP_CAPACITY: f64 = 5.0;
const ANON_JUMP_REFILL_PER_SEC: f64 = 1.0 / 30.0; // 1 jump every 30s

struct JumpBucket {
    capacity: f64,
    tokens: f64,
    refill_per_sec: f64,
    last_update: Instant,
}

impl JumpBucket {
    fn new(capacity: f64, refill_per_sec: f64) -> Self {
        Self {
            capacity,
            tokens: capacity,
            refill_per_sec,
            last_update: Instant::now(),
        }
    }

    fn update(&mut self) {
        let dt = self.last_update.elapsed().as_secs_f64();
        self.last_update = Instant::now();
        self.tokens = (self.tokens + dt * self.refill_per_sec).min(self.capacity);
    }

    fn remaining(&mut self) -> u32 {
        self.update();
        self.tokens.floor().max(0.0) as u32
    }

    fn max(&self) -> u32 {
        self.capacity.floor().max(0.0) as u32
    }

    fn try_consume(&mut self, cost: f64) -> bool {
        self.update();
        if self.tokens + 1e-9 >= cost {
            self.tokens -= cost;
            true
        } else {
            false
        }
    }
}

/// Client connection
pub struct Client {
    pub id: String,
    pub input_tx: mpsc::Sender<InputEvent>,
}

/// Server state shared across connections
pub struct StreamingServer {
    /// Frame broadcast channel
    frame_tx: broadcast::Sender<Arc<EncodedFrame>>,
    /// H.264 frame broadcast channel
    h264_tx: broadcast::Sender<Arc<EncodedH264Frame>>,
    /// Latest H.264 config (WebCodecs)
    h264_cfg_tx: watch::Sender<Option<H264Config>>,
    /// Output events (state updates, etc.) broadcast to control clients
    output_tx: broadcast::Sender<OutputEvent>,
    /// Connected clients
    clients: Arc<RwLock<Vec<Client>>>,
    /// Input event sender (receiver is returned from new())
    input_tx: mpsc::Sender<InputEvent>,
}

impl StreamingServer {
    pub fn new() -> (Self, mpsc::Receiver<InputEvent>) {
        let (frame_tx, _) = broadcast::channel(4);
        let (h264_tx, _) = broadcast::channel(64);
        let (h264_cfg_tx, _) = watch::channel(None);
        let (output_tx, _) = broadcast::channel(64);
        let (input_tx, input_rx) = mpsc::channel(256);

        let server = Self {
            frame_tx,
            h264_tx,
            h264_cfg_tx,
            output_tx,
            clients: Arc::new(RwLock::new(Vec::new())),
            input_tx,
        };

        (server, input_rx)
    }

    /// Broadcast frame to all clients
    pub fn broadcast_frame(&self, frame: Arc<EncodedFrame>) {
        let _ = self.frame_tx.send(frame);
    }

    /// Broadcast H.264 frame to all H.264 stream clients
    pub fn broadcast_h264_frame(&self, frame: Arc<EncodedH264Frame>) {
        let _ = self.h264_tx.send(frame);
    }

    /// Publish (or replace) the latest H.264 decoder config
    pub fn set_h264_config(&self, cfg: H264Config) {
        let _ = self.h264_cfg_tx.send_replace(Some(cfg));
    }

    /// Broadcast a state/output event to control clients
    pub fn broadcast_output(&self, event: OutputEvent) {
        let _ = self.output_tx.send(event);
    }

    /// Get client count
    pub fn client_count(&self) -> usize {
        // Best-effort: count stream subscribers as \"viewers\".
        self.frame_tx.receiver_count() + self.h264_tx.receiver_count()
    }

    pub fn mjpeg_viewer_count(&self) -> usize {
        self.frame_tx.receiver_count()
    }

    pub fn h264_viewer_count(&self) -> usize {
        self.h264_tx.receiver_count()
    }

    /// Create router for web server
    pub fn create_router(self: Arc<Self>) -> Router {
        Router::new()
            // DEBUG/DEV ONLY: Pixel streaming endpoints (not used in production).
            // Production uses client-side dataset rendering. These endpoints are retained
            // for debugging server-side rendering or testing video codec compatibility.
            // Phase 0.1: Stream mode is debug-only; client defaults to dataset mode.
            .route("/mjpeg", get(index_handler))
            .route("/stream", get(stream_handler))
            .route("/control", get(control_handler))
            // Serve the on-disk universe dataset (index.json + cells/*.bin) over HTTP.
            // This enables client-side rendering modes that stream dataset chunks instead of video.
            .nest_service("/universe", ServeDir::new("universe"))
            .fallback_service(ServeDir::new("client/dist").append_index_html_on_directories(true))
            .layer(CorsLayer::new().allow_origin(Any).allow_methods(Any).allow_headers(Any))
            .with_state(self)
    }
}

/// DEBUG/DEV ONLY: Legacy MJPEG viewer endpoint.
/// This is not used in production; the main client uses dataset mode.
/// Phase 0.1: Pixel streaming is debug-only.
async fn index_handler() -> Html<&'static str> {
    Html(r#"
<!DOCTYPE html>
<html>
<head>
    <title>Helios Stream</title>
    <style>
        body { margin: 0; background: #000; color: #fff; font-family: monospace; overflow: hidden; }
        #container { display: flex; flex-direction: column; height: 100vh; padding-bottom: 60px; box-sizing: border-box; }
        #video { flex: 1; object-fit: contain; min-height: 0; }
        #controls { padding: 10px; background: #111; position: fixed; bottom: 0; left: 0; right: 0; z-index: 10; }
        #info { padding: 5px; font-size: 12px; }
    </style>
</head>
<body>
    <div id="container">
        <img id="video" />
        <div id="controls">
            <div id="info">Connecting...</div>
            <div>
                WASD: Move | Mouse: Look | Space/Shift: Up/Down | Scroll: Zoom | Q/E: Speed -/+ | P: Pause | ,/.: Time Speed
            </div>
        </div>
    </div>
    <script>
        console.log('🚀 Universe Client Starting...');

        const video = document.getElementById('video');
        const info = document.getElementById('info');

        // Video stream
        console.log('[STREAM] Connecting to ws://' + location.host + '/stream');
        const streamWs = new WebSocket(`ws://${location.host}/stream`);
        streamWs.binaryType = 'arraybuffer';

        streamWs.onopen = () => {
            console.log('[STREAM] ✓ Connected');
            info.textContent = 'Stream connected';
        };

        streamWs.onmessage = (event) => {
            const blob = new Blob([event.data], { type: 'image/jpeg' });
            video.src = URL.createObjectURL(blob);
            console.log('[STREAM] Frame received:', blob.size, 'bytes');
        };

        streamWs.onerror = (err) => {
            console.error('[STREAM] ✗ Error:', err);
            info.textContent = 'Stream error';
        };

        streamWs.onclose = () => {
            console.log('[STREAM] Disconnected');
        };

        // Control channel
        console.log('[CONTROL] Connecting to ws://' + location.host + '/control');
        const controlWs = new WebSocket(`ws://${location.host}/control`);

        controlWs.onopen = () => {
            console.log('[CONTROL] ✓ Connected');
        };

        controlWs.onmessage = (event) => {
            const data = JSON.parse(event.data);
            console.log('[CONTROL] Received:', data.type);
            if (data.type === 'State') {
                info.textContent = `Year: ${data.epoch_jd.toFixed(1)} | FPS: ${data.fps.toFixed(1)} | Pos: (${data.camera_x.toExponential(2)}, ${data.camera_y.toExponential(2)}, ${data.camera_z.toExponential(2)})`;
            }
        };

        controlWs.onerror = (err) => {
            console.error('[CONTROL] ✗ Error:', err);
        };

        controlWs.onclose = () => {
            console.log('[CONTROL] Disconnected');
        };

        // Input handling
        const send = (event) => {
            if (controlWs.readyState === WebSocket.OPEN) {
                console.log('[INPUT] Sending:', event);
                controlWs.send(JSON.stringify(event));
            } else {
                console.warn('[INPUT] Control WS not ready, state:', controlWs.readyState);
            }
        };

        const keys = {};
        document.addEventListener('keydown', (e) => {
            if (!keys[e.code]) {
                keys[e.code] = true;
                console.log('[KEYBOARD] Down:', e.code);
                send({ type: 'Key', code: e.code, pressed: true });
            }
            e.preventDefault();
        });

        document.addEventListener('keyup', (e) => {
            keys[e.code] = false;
            console.log('[KEYBOARD] Up:', e.code);
            send({ type: 'Key', code: e.code, pressed: false });
            e.preventDefault();
        });

        let mouseCapture = false;
        video.addEventListener('click', () => {
            console.log('[MOUSE] Click - requesting pointer lock');
            video.requestPointerLock();
        });

        document.addEventListener('pointerlockchange', () => {
            mouseCapture = document.pointerLockElement === video;
            console.log('[MOUSE] Pointer lock:', mouseCapture ? '🔒 ACTIVE' : '🔓 INACTIVE');
        });

        document.addEventListener('pointerlockerror', (e) => {
            console.error('[MOUSE] ✗ Pointer lock error:', e);
        });

        document.addEventListener('mousemove', (e) => {
            if (mouseCapture && (e.movementX !== 0 || e.movementY !== 0)) {
                const event = { type: 'MouseMove', dx: e.movementX * 0.002, dy: e.movementY * 0.002 };
                send(event);
            }
        });

        // Scroll wheel for zoom
        document.addEventListener('wheel', (e) => {
            e.preventDefault();
            const delta = e.deltaY > 0 ? -1 : 1;  // Scroll up = zoom in (positive)
            send({ type: 'Scroll', delta: delta });
        }, { passive: false });

        console.log('✓ Event listeners registered');
        console.log('📌 Click the Sun image to capture mouse!');
    </script>
</body>
</html>
    "#)
}

/// WebSocket handler for video stream
async fn stream_handler(
    ws: WebSocketUpgrade,
    State(state): State<Arc<StreamingServer>>,
    Query(params): Query<HashMap<String, String>>,
) -> impl IntoResponse {
    let codec = params.get("codec").map(|s| s.as_str()).unwrap_or("mjpeg");
    if codec.eq_ignore_ascii_case("h264") {
        ws.on_upgrade(|socket| handle_stream_h264(socket, state))
    } else {
        ws.on_upgrade(|socket| handle_stream_mjpeg(socket, state))
    }
}

async fn handle_stream_mjpeg(mut socket: WebSocket, state: Arc<StreamingServer>) {
    tracing::info!("🎥 Stream client connected");
    let mut frame_rx = state.frame_tx.subscribe();

    let mut frame_count = 0;
    while let Ok(frame) = frame_rx.recv().await {
        if socket.send(Message::Binary(frame.data.clone())).await.is_err() {
            tracing::info!("🎥 Stream client disconnected");
            break;
        }
        frame_count += 1;
        if frame_count % 60 == 0 {
            tracing::debug!("Streamed {} frames to client", frame_count);
        }
    }
}

async fn handle_stream_h264(mut socket: WebSocket, state: Arc<StreamingServer>) {
    tracing::info!("🎥 H264 stream client connected");

    // Subscribe immediately so the server knows there is an H.264 viewer and starts feeding the encoder.
    // We'll start consuming frames only after we've sent the decoder configuration.
    let mut frame_rx = state.h264_tx.subscribe();

    // Wait for a decoder config, then send it to the client
    let mut cfg_rx = state.h264_cfg_tx.subscribe();
    loop {
        let cfg_opt = { cfg_rx.borrow().clone() };
        if let Some(cfg) = cfg_opt {
            let msg = serde_json::json!({
                "type": "VideoConfig",
                "codec": cfg.codec,
                "avcc": base64::engine::general_purpose::STANDARD.encode(cfg.avcc),
            });
            if socket.send(Message::Text(msg.to_string())).await.is_err() {
                tracing::info!("🎥 H264 client disconnected (sending config)");
                return;
            }
            break;
        }
        if cfg_rx.changed().await.is_err() {
            return;
        }
    }

    let mut frame_count = 0u64;
    while let Ok(frame) = frame_rx.recv().await {
        // Binary packet format:
        // [0] flags (bit0 = keyframe)
        // [1..9] timestamp_us (u64 LE)
        // [9..] annexb access unit bytes
        let mut payload = Vec::with_capacity(1 + 8 + frame.data.len());
        payload.push(if frame.is_key { 1 } else { 0 });
        payload.extend_from_slice(&frame.timestamp_us.to_le_bytes());
        payload.extend_from_slice(&frame.data);

        if socket.send(Message::Binary(payload)).await.is_err() {
            tracing::info!("🎥 H264 stream client disconnected");
            break;
        }
        frame_count += 1;
        if frame_count % 60 == 0 {
            tracing::debug!("H264 streamed {} frames to client", frame_count);
        }
    }
}

/// WebSocket handler for control channel
async fn control_handler(
    ws: WebSocketUpgrade,
    State(state): State<Arc<StreamingServer>>,
    Query(params): Query<HashMap<String, String>>,
) -> impl IntoResponse {
    let registered = params
        .get("registered")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("yes"))
        .unwrap_or(false);

    ws.on_upgrade(move |socket| handle_control(socket, state, registered))
}

async fn handle_control(socket: WebSocket, state: Arc<StreamingServer>, registered: bool) {
    tracing::info!("🎮 Control client connected");
    let (mut sender, mut receiver) = socket.split();
    let mut output_rx = state.output_tx.subscribe();

    // Per-connection jump budget (token bucket).
    let multiplier = if registered { 5.0 } else { 1.0 };
    let mut jump_bucket = JumpBucket::new(
        ANON_JUMP_CAPACITY * multiplier,
        ANON_JUMP_REFILL_PER_SEC * multiplier,
    );
    let mut block_lookat_until: Option<Instant> = None;

    // Send initial jump status
    let initial = OutputEvent::JumpStatus {
        remaining: jump_bucket.remaining(),
        max: jump_bucket.max(),
        registered,
    };
    let _ = sender.send(Message::Text(serialize_output(&initial))).await;

    let mut event_count = 0;
    loop {
        tokio::select! {
            // Server -> client output events (state, etc.)
            recv_res = output_rx.recv() => {
                let event = match recv_res {
                    Ok(e) => e,
                    Err(broadcast::error::RecvError::Lagged(n)) => {
                        tracing::warn!("🎮 Control client lagged by {} messages", n);
                        continue;
                    }
                    Err(broadcast::error::RecvError::Closed) => break,
                };

                let json = serialize_output(&event);
                if sender.send(Message::Text(json)).await.is_err() {
                    tracing::info!("🎮 Control client disconnected (send failed)");
                    break;
                }
            }

            // Client -> server input events
            msg = receiver.next() => {
                let msg = match msg {
                    Some(Ok(Message::Text(text))) => text,
                    Some(Ok(Message::Close(_))) | None => {
                        tracing::info!("🎮 Control client disconnected");
                        break;
                    }
                    Some(Err(e)) => {
                        tracing::warn!("🎮 Control client disconnected (error): {}", e);
                        break;
                    }
                    _ => continue,
                };

                if let Ok(event) = parse_input(&msg) {
                    event_count += 1;

                    // Handle Ping locally (don't broadcast to all clients)
                    if let InputEvent::Ping { client_time } = event {
                        let server_time = std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap()
                            .as_millis() as u64;
                        let pong = OutputEvent::Pong { client_time, server_time };
                        let json = serialize_output(&pong);
                        let _ = sender.send(Message::Text(json)).await;
                        continue;
                    }

                    // Gate LookAt if the preceding Teleport was rejected (prevents camera snapping without moving)
                    if matches!(event, InputEvent::LookAt { .. }) {
                        if let Some(until) = block_lookat_until {
                            if Instant::now() < until {
                                continue;
                            }
                        }
                    }

                    // Jump budgeting (Teleport counts as a \"jump\")
                    if let InputEvent::Teleport { .. } = event {
                        if jump_bucket.try_consume(1.0) {
                            // Allowed: forward to renderer
                            if let Err(e) = state.input_tx.send(event).await {
                                tracing::error!("Failed to send teleport event: {}", e);
                            }
                        } else {
                            // Rejected: block follow-up LookAt briefly and notify client
                            block_lookat_until = Some(Instant::now() + Duration::from_millis(250));
                            let err = OutputEvent::Error {
                                message: "Jump limit reached. Please wait for your jump budget to refill.".to_string(),
                            };
                            let _ = sender.send(Message::Text(serialize_output(&err))).await;
                        }

                        // Always send updated budget to this client
                        let status = OutputEvent::JumpStatus {
                            remaining: jump_bucket.remaining(),
                            max: jump_bucket.max(),
                            registered,
                        };
                        let _ = sender.send(Message::Text(serialize_output(&status))).await;
                        continue;
                    }

                    tracing::debug!("🎮 Event #{}: {:?}", event_count, event);
                    if let Err(e) = state.input_tx.send(event).await {
                        tracing::error!("Failed to send input event: {}", e);
                    }
                } else {
                    tracing::warn!("Failed to parse input: {}", msg);
                }
            }
        }
    }
    tracing::info!("🎮 Control handler exiting, processed {} events", event_count);
}
