//! Video encoding (MJPEG for WebSocket streaming)

use crate::capture::CapturedFrame;
use anyhow::Result;
use std::io::Cursor;
use tokio::io::{AsyncWriteExt, AsyncReadExt};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use std::process::Stdio;

/// Encoded frame
pub struct EncodedFrame {
    pub data: Vec<u8>,
    pub timestamp_us: u64,
    pub width: u32,
    pub height: u32,
}

/// Encoded H.264 access unit (Annex-B)
pub struct EncodedH264Frame {
    pub data: Vec<u8>,
    pub timestamp_us: u64,
    pub is_key: bool,
}

/// H.264 decoder configuration for WebCodecs (avcC + codec string)
#[derive(Clone, Debug)]
pub struct H264Config {
    pub codec: String,
    pub avcc: Vec<u8>,
}

/// Encoder configuration
#[derive(Clone, Debug)]
pub struct EncoderConfig {
    pub width: u32,
    pub height: u32,
    pub quality: u8,  // JPEG quality 1-100
}

impl Default for EncoderConfig {
    fn default() -> Self {
        Self {
            width: 1920,
            height: 1080,
            quality: 85,
        }
    }
}

/// JPEG encoder for frame streaming
pub struct JpegEncoder {
    config: EncoderConfig,
}

impl JpegEncoder {
    pub fn new(config: EncoderConfig) -> Self {
        Self { config }
    }

    pub fn encode(&mut self, frame: &CapturedFrame) -> Result<EncodedFrame> {
        use image::{ImageBuffer, Rgb, ImageFormat};

        // Convert RGBA to RGB (strip alpha channel)
        let mut rgb_data = Vec::with_capacity((frame.width * frame.height * 3) as usize);
        for chunk in frame.data.chunks_exact(4) {
            rgb_data.push(chunk[0]); // R
            rgb_data.push(chunk[1]); // G
            rgb_data.push(chunk[2]); // B
            // Skip alpha (chunk[3])
        }

        // Create RGB image buffer
        let img: ImageBuffer<Rgb<u8>, _> = ImageBuffer::from_raw(
            frame.width,
            frame.height,
            rgb_data,
        ).ok_or_else(|| anyhow::anyhow!("Failed to create image buffer"))?;

        // Encode as JPEG
        let mut buffer = Cursor::new(Vec::new());
        img.write_to(&mut buffer, ImageFormat::Jpeg)?;

        Ok(EncodedFrame {
            data: buffer.into_inner(),
            timestamp_us: frame.timestamp_us,
            width: frame.width,
            height: frame.height,
        })
    }
}

/// FFmpeg/NVENC-backed H.264 encoder (writes Annex-B to stdout).
///
/// Notes:
/// - This uses `ffmpeg` as an implementation detail for v1.
/// - It still benefits from NVENC for the heavy encode step.
pub struct FfmpegH264Encoder {
    child: Child,
    stdin: ChildStdin,
    stdout: ChildStdout,
}

#[derive(Clone, Debug)]
pub struct FfmpegH264Config {
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    pub bitrate_kbps: u32,
    pub gop: u32,
}

impl Default for FfmpegH264Config {
    fn default() -> Self {
        Self {
            width: 1280,
            height: 720,
            fps: 30,
            bitrate_kbps: 8_000,
            gop: 60,
        }
    }
}

impl FfmpegH264Encoder {
    pub async fn new(config: FfmpegH264Config) -> Result<Self> {
        // NVENC low-latency params: no B-frames, small-ish GOP, AUD inserted for parsing.
        let size_arg = format!("{}x{}", config.width, config.height);
        let fps_arg = config.fps.to_string();
        let bitrate_arg = format!("{}k", config.bitrate_kbps);
        let bufsize_arg = format!("{}k", config.bitrate_kbps * 2);
        let gop_arg = config.gop.to_string();

        let mut cmd = Command::new("ffmpeg");
        cmd.args([
            "-hide_banner",
            "-loglevel", "error",
            "-nostats",
            "-f", "rawvideo",
            "-pix_fmt", "rgba",
            "-s", &size_arg,
            "-r", &fps_arg,
            "-i", "pipe:0",
            "-an",
            "-c:v", "h264_nvenc",
            // Improve decoder compatibility (WebCodecs / mobile):
            // keep to baseline profile and clamp level so `avc1.*` is widely supported.
            "-profile:v", "baseline",
            "-level:v", "4.1",
            "-preset", "p1",
            "-tune", "ll",
            "-bf", "0",
            "-g", &gop_arg,
            "-keyint_min", &gop_arg,
            "-rc", "cbr",
            "-b:v", &bitrate_arg,
            "-maxrate", &bitrate_arg,
            "-bufsize", &bufsize_arg,
            "-pix_fmt", "yuv420p",
            "-bsf:v", "h264_metadata=aud=insert",
            "-f", "h264",
            "pipe:1",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null());

        let mut child = cmd.spawn()?;
        let stdin = child.stdin.take().ok_or_else(|| anyhow::anyhow!("ffmpeg stdin unavailable"))?;
        let stdout = child.stdout.take().ok_or_else(|| anyhow::anyhow!("ffmpeg stdout unavailable"))?;

        Ok(Self { child, stdin, stdout })
    }

    pub async fn write_frame_rgba(&mut self, frame: &CapturedFrame) -> Result<()> {
        self.stdin.write_all(&frame.data).await?;
        Ok(())
    }

    pub async fn read_stdout_chunk(&mut self, buf: &mut [u8]) -> Result<usize> {
        let n = self.stdout.read(buf).await?;
        Ok(n)
    }

    pub fn into_parts(self) -> (Child, ChildStdin, ChildStdout) {
        (self.child, self.stdin, self.stdout)
    }
}

/// Build an avcC record and codec string from SPS/PPS NAL units.
/// The SPS/PPS slices must include the 1-byte NAL header (0x67/0x68) but NOT any start code.
pub fn h264_config_from_sps_pps(sps: &[u8], pps: &[u8]) -> Result<H264Config> {
    if sps.len() < 4 {
        anyhow::bail!("SPS too short");
    }
    // sps[0] is NAL header. Next 3 bytes are profile_idc, compat, level_idc.
    let profile_idc = sps[1];
    let profile_compat = sps[2];
    let level_idc = sps[3];

    let codec = format!("avc1.{:02X}{:02X}{:02X}", profile_idc, profile_compat, level_idc);

    let mut avcc = Vec::with_capacity(11 + sps.len() + pps.len());
    avcc.push(1); // configurationVersion
    avcc.push(profile_idc);
    avcc.push(profile_compat);
    avcc.push(level_idc);
    avcc.push(0xFC | 3); // lengthSizeMinusOne = 3 (4 bytes)
    avcc.push(0xE0 | 1); // numOfSPS = 1
    avcc.extend_from_slice(&(sps.len() as u16).to_be_bytes());
    avcc.extend_from_slice(sps);
    avcc.push(1); // numOfPPS = 1
    avcc.extend_from_slice(&(pps.len() as u16).to_be_bytes());
    avcc.extend_from_slice(pps);

    Ok(H264Config { codec, avcc })
}
