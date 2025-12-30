//! Main streaming server

use crate::headless::{HeadlessRenderer, HeadlessConfig};
use crate::streaming::StreamingServer;
use crate::input::OutputEvent;
use crate::capture::CapturedFrame;
use crate::encoder::{EncoderConfig, JpegEncoder, EncodedH264Frame, FfmpegH264Encoder, FfmpegH264Config, h264_config_from_sps_pps};

use std::sync::Arc;
use std::time::{Duration, Instant};
use std::collections::{HashSet, VecDeque};
use parking_lot::RwLock;
use anyhow::Result;
use std::path::PathBuf;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

// --- H.264 Annex-B parsing helpers -------------------------------------------------

fn find_start_code(buf: &[u8], from: usize) -> Option<(usize, usize)> {
    let mut i = from;
    while i + 3 < buf.len() {
        if buf[i] == 0 && buf[i + 1] == 0 {
            if buf[i + 2] == 1 {
                return Some((i, 3));
            }
            if i + 4 < buf.len() && buf[i + 2] == 0 && buf[i + 3] == 1 {
                return Some((i, 4));
            }
        }
        i += 1;
    }
    None
}

fn nal_type_at(buf: &[u8], sc_pos: usize, sc_len: usize) -> Option<u8> {
    let h = sc_pos + sc_len;
    if h >= buf.len() {
        return None;
    }
    Some(buf[h] & 0x1F)
}

struct AnnexBAccessUnitParser {
    buf: Vec<u8>,
}

impl AnnexBAccessUnitParser {
    fn new() -> Self {
        Self { buf: Vec::new() }
    }

    fn push(&mut self, data: &[u8]) {
        self.buf.extend_from_slice(data);
    }

    /// Split stream into access units using AUD (nal_type=9) boundaries.
    /// Keeps the last (possibly incomplete) access unit in the internal buffer.
    fn drain_access_units(&mut self) -> Vec<Vec<u8>> {
        let mut aud_positions: Vec<usize> = Vec::new();
        let mut scan = 0usize;
        while let Some((pos, sc_len)) = find_start_code(&self.buf, scan) {
            if let Some(nt) = nal_type_at(&self.buf, pos, sc_len) {
                if nt == 9 {
                    aud_positions.push(pos);
                }
            }
            scan = pos + sc_len;
        }

        let mut out: Vec<Vec<u8>> = Vec::new();
        if aud_positions.len() >= 2 {
            // First AU may contain SPS/PPS before the first AUD. Include that prefix.
            out.push(self.buf[0..aud_positions[1]].to_vec());
            for w in aud_positions.windows(2).skip(1) {
                out.push(self.buf[w[0]..w[1]].to_vec());
            }
            let keep_from = *aud_positions.last().unwrap();
            self.buf.drain(0..keep_from);
        } else {
            // No AUD found yet; keep buffering.
            // Prevent runaway growth if stream is malformed.
            if self.buf.len() > 4 * 1024 * 1024 {
                self.buf.clear();
            }
        }
        out
    }
}

fn inspect_access_unit(au: &[u8]) -> (bool, Option<Vec<u8>>, Option<Vec<u8>>) {
    let mut is_key = false;
    let mut sps: Option<Vec<u8>> = None;
    let mut pps: Option<Vec<u8>> = None;

    let mut sc = find_start_code(au, 0);
    while let Some((pos, sc_len)) = sc {
        let next = find_start_code(au, pos + sc_len).map(|(p, _)| p).unwrap_or(au.len());
        if let Some(nt) = nal_type_at(au, pos, sc_len) {
            if nt == 5 {
                is_key = true;
            } else if nt == 7 && sps.is_none() {
                // Include 1-byte NAL header, exclude start code
                let start = pos + sc_len;
                sps = Some(au[start..next].to_vec());
            } else if nt == 8 && pps.is_none() {
                let start = pos + sc_len;
                pps = Some(au[start..next].to_vec());
            }
        }
        sc = find_start_code(au, pos + sc_len);
    }

    (is_key, sps, pps)
}

/// Tracks which keys are currently held down
#[derive(Default)]
pub struct InputState {
    pub held_keys: HashSet<String>,
}

/// Streaming server configuration
#[derive(Clone, Debug)]
pub struct StreamConfig {
    pub port: u16,
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    /// Universe dataset directory (contains index.json and cells/)
    pub universe: PathBuf,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            port: 7878,
            width: 1920,
            height: 1080,
            fps: 30, // Lower FPS for streaming
            universe: PathBuf::from("universe"),
        }
    }
}

/// Run the streaming server
pub async fn run_server(config: StreamConfig) -> Result<()> {
    tracing::info!("Starting Universe streaming server on port {}", config.port);

    // Create renderer
    let renderer_config = HeadlessConfig {
        width: config.width,
        height: config.height,
        universe_dir: config.universe.clone(),
    };
    let renderer = Arc::new(RwLock::new(
        HeadlessRenderer::new(renderer_config).await?
    ));

    // Create streaming server
    let (streaming, mut input_rx) = StreamingServer::new(&config.universe);

    // Shared input state for tracking held keys
    let input_state = Arc::new(RwLock::new(InputState::default()));

    // Input processing task
    let renderer_for_input = renderer.clone();
    let input_state_for_task = input_state.clone();
    tokio::spawn(async move {
        while let Some(event) = input_rx.recv().await {
            use crate::input::InputEvent;

            tracing::debug!("Processing input event: {:?}", event);

            match event {
                InputEvent::MouseMove { dx, dy } => {
                    let mut r = renderer_for_input.write();
                    tracing::debug!("Camera rotate: dx={}, dy={}", dx, dy);
                    r.camera.rotate(dx, dy);
                }
                InputEvent::Scroll { delta } => {
                    // Exponential zoom - multiply distance by factor
                    let mut r = renderer_for_input.write();
                    let factor = if delta > 0.0 { 0.85 } else { 1.18 };  // ~15% per scroll
                    let dir = r.camera.forward();
                    let dist = r.camera.position.length();
                    // Move towards/away from origin exponentially
                    r.camera.position = r.camera.position + dir * dist * (1.0 - factor);
                    tracing::debug!("Scroll zoom: factor={}, new_dist={:.2e}", factor, r.camera.position.length());
                }
                InputEvent::Key { code, pressed } => {
                    // Track held keys for continuous movement
                    let mut state = input_state_for_task.write();
                    if pressed {
                        state.held_keys.insert(code.clone());
                    } else {
                        state.held_keys.remove(&code);
                    }

                    // Handle instant actions (non-movement keys)
                    if pressed {
                        let mut r = renderer_for_input.write();
                        match code.as_str() {
                            "KeyP" => r.time_controller.toggle_pause(),
                            "Comma" => {
                                let current = r.time_controller.rate_years_per_second();
                                r.time_controller.set_rate_years_per_second(current * 0.5);
                            }
                            "Period" => {
                                let current = r.time_controller.rate_years_per_second();
                                r.time_controller.set_rate_years_per_second(current * 2.0);
                            }
                            "KeyQ" => {
                                // Decrease camera speed
                                r.camera.speed *= 0.5;
                                tracing::info!("Camera speed: {:.2e} m/s", r.camera.speed);
                            }
                            "KeyE" => {
                                // Increase camera speed
                                r.camera.speed *= 2.0;
                                tracing::info!("Camera speed: {:.2e} m/s", r.camera.speed);
                            }
                            _ => {}
                        }
                    }
                }
                InputEvent::SetTime { jd } => {
                    let mut r = renderer_for_input.write();
                    let epoch = universe_sim::jc_to_epoch((jd - 2451545.0) / 36525.0);
                    r.time_controller.set_time(epoch);
                }
                InputEvent::SetTimeRate { rate } => {
                    let mut r = renderer_for_input.write();
                    r.time_controller.set_rate_years_per_second(rate);
                }
                InputEvent::Teleport { x, y, z } => {
                    use glam::DVec3;
                    let mut r = renderer_for_input.write();
                    r.camera.set_position(DVec3::new(x, y, z));
                }
                _ => {}
            }
        }
    });

    // Start HTTP/WebSocket server
    let streaming = Arc::new(streaming);
    let app = streaming.clone().create_router();
    let addr = std::net::SocketAddr::from(([0, 0, 0, 0], config.port));

    tracing::info!("Server ready at http://0.0.0.0:{}", config.port);
    tracing::info!("Open http://localhost:{} in your browser", config.port);

    let streaming_for_render = streaming.clone();
    let server_handle = tokio::spawn(async move {
        let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
        axum::serve(listener, app).await.unwrap();
    });

    // --- H.264 encoder task (NVENC via ffmpeg) ------------------------------------
    // One encoder instance feeds all H.264 clients (broadcast).
    let (h264_raw_tx, mut h264_raw_rx) = tokio::sync::mpsc::channel::<CapturedFrame>(2);
    let streaming_for_h264 = streaming.clone();
    let h264_cfg = FfmpegH264Config {
        width: config.width,
        height: config.height,
        fps: config.fps,
        bitrate_kbps: 8_000,
        gop: (config.fps.max(1) * 2),
    };

    tokio::spawn(async move {
        let encoder = match FfmpegH264Encoder::new(h264_cfg).await {
            Ok(e) => e,
            Err(e) => {
                tracing::error!("Failed to start ffmpeg/NVENC encoder: {}", e);
                return;
            }
        };
        let (mut child, mut stdin, mut stdout) = encoder.into_parts();

        let mut out_buf = vec![0u8; 64 * 1024];
        let mut parser = AnnexBAccessUnitParser::new();
        let mut ts_queue: VecDeque<u64> = VecDeque::new();

        let mut sps: Option<Vec<u8>> = None;
        let mut pps: Option<Vec<u8>> = None;
        let mut config_published = false;

        loop {
            tokio::select! {
                maybe_frame = h264_raw_rx.recv() => {
                    match maybe_frame {
                        Some(frame) => {
                            ts_queue.push_back(frame.timestamp_us);
                            if let Err(e) = stdin.write_all(&frame.data).await {
                                tracing::error!("H264 encoder stdin write failed: {}", e);
                                break;
                            }
                        }
                        None => break,
                    }
                }
                read_res = stdout.read(&mut out_buf) => {
                    let n = match read_res {
                        Ok(n) => n,
                        Err(e) => {
                            tracing::error!("H264 encoder stdout read failed: {}", e);
                            break;
                        }
                    };
                    if n == 0 {
                        break;
                    }
                    parser.push(&out_buf[..n]);
                    for au in parser.drain_access_units() {
                        let (is_key, au_sps, au_pps) = inspect_access_unit(&au);

                        if sps.is_none() { sps = au_sps; }
                        if pps.is_none() { pps = au_pps; }

                        if !config_published {
                            if let (Some(ref sps), Some(ref pps)) = (&sps, &pps) {
                                if let Ok(cfg) = h264_config_from_sps_pps(sps, pps) {
                                    tracing::info!("Published H264 decoder config: {}", cfg.codec);
                                    streaming_for_h264.set_h264_config(cfg);
                                    config_published = true;
                                }
                            }
                        }

                        let ts = ts_queue.pop_front().unwrap_or(0);
                        streaming_for_h264.broadcast_h264_frame(Arc::new(EncodedH264Frame {
                            data: au,
                            timestamp_us: ts,
                            is_key,
                        }));
                    }
                }
            }
        }

        let _ = child.kill().await;
    });

    // MJPEG encoder (fallback only)
    let mut jpeg_encoder = JpegEncoder::new(EncoderConfig {
        width: config.width,
        height: config.height,
        quality: 80,
    });

    // Render loop
    let frame_duration = Duration::from_secs_f64(1.0 / config.fps as f64);
    let mut frame_count = 0u64;
    let mut last_fps_check = Instant::now();
    let mut last_measured_fps: f32 = config.fps as f32;
    let mut last_frame_time = Instant::now();
    let mut last_state_sent = Instant::now();

    loop {
        let frame_start = Instant::now();
        let dt = last_frame_time.elapsed().as_secs_f64();
        last_frame_time = frame_start;

        // Process held keys for continuous movement
        {
            let state = input_state.read();
            if !state.held_keys.is_empty() {
                let mut r = renderer.write();
                let mut forward = 0.0;
                let mut right = 0.0;
                let mut up = 0.0;

                for key in &state.held_keys {
                    match key.as_str() {
                        "KeyW" => forward += 1.0,
                        "KeyS" => forward -= 1.0,
                        "KeyA" => right -= 1.0,
                        "KeyD" => right += 1.0,
                        "Space" => up += 1.0,
                        "ShiftLeft" | "ShiftRight" => up -= 1.0,
                        _ => {}
                    }
                }

                if forward != 0.0 || right != 0.0 || up != 0.0 {
                    r.camera.translate(forward, right, up, dt);
                }
            }
        }

        // Render frame + snapshot state
        let (captured, epoch_jd, time_rate, cam_pos) = {
            let mut r = renderer.write();
            let captured = r.render_frame()?;
            let epoch_jd = r.time_controller.current().to_jde_utc_days();
            let time_rate = r.time_controller.rate();
            let cam_pos = r.camera.position;
            (captured, epoch_jd, time_rate, cam_pos)
        };

        // Broadcast MJPEG if any MJPEG viewers exist (server index page or legacy clients)
        if streaming_for_render.mjpeg_viewer_count() > 0 {
            if let Ok(encoded) = jpeg_encoder.encode(&captured) {
                streaming_for_render.broadcast_frame(Arc::new(encoded));
            }
        }

        // Feed H.264 encoder if any H.264 viewers exist
        if streaming_for_render.h264_viewer_count() > 0 {
            let _ = h264_raw_tx.try_send(captured);
        }

        // FPS logging
        let fps_sample_frames = config.fps.max(1) as u64;
        if frame_count % fps_sample_frames == 0 && frame_count > 0 {
            let elapsed = Instant::now() - last_fps_check;
            let actual_fps = fps_sample_frames as f64 / elapsed.as_secs_f64();
            last_measured_fps = actual_fps as f32;
            last_fps_check = Instant::now();
            tracing::debug!(
                "FPS: {:.1} | Clients: {}",
                actual_fps,
                streaming_for_render.client_count()
            );
        }

        // State updates (rate-limited)
        if last_state_sent.elapsed() >= Duration::from_millis(200) {
            last_state_sent = Instant::now();
            let clients = streaming_for_render.client_count() as u32;
            streaming_for_render.broadcast_output(OutputEvent::State {
                epoch_jd,
                time_rate,
                camera_x: cam_pos.x,
                camera_y: cam_pos.y,
                camera_z: cam_pos.z,
                fps: last_measured_fps,
                clients,
            });
        }

        // Frame timing
        let elapsed = frame_start.elapsed();
        if elapsed < frame_duration {
            tokio::time::sleep(frame_duration - elapsed).await;
        }

        frame_count += 1;

        // Check if server is still running
        if server_handle.is_finished() {
            break;
        }
    }

    Ok(())
}
