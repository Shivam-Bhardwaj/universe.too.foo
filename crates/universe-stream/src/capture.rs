//! Frame capture from WGPU render target

use wgpu::{Device, Queue, Texture, TextureFormat, TextureUsages};
use anyhow::Result;

/// Captured frame data
pub struct CapturedFrame {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,  // RGBA data
    pub timestamp_us: u64,
}

/// Frame capture from render texture
pub struct FrameCapture {
    /// Staging texture for GPU->CPU copy
    staging_texture: Texture,
    /// Staging buffer
    staging_buffer: wgpu::Buffer,
    /// Dimensions
    width: u32,
    height: u32,
    /// Bytes per row (aligned)
    bytes_per_row: u32,
    /// Frame counter
    frame_count: u64,
}

impl FrameCapture {
    pub fn new(device: &Device, width: u32, height: u32) -> Self {
        // WGPU requires 256-byte row alignment
        let bytes_per_pixel = 4; // RGBA
        let unpadded_bytes_per_row = width * bytes_per_pixel;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let bytes_per_row = (unpadded_bytes_per_row + align - 1) / align * align;

        let staging_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Capture Staging Texture"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: TextureFormat::Rgba8UnormSrgb,
            usage: TextureUsages::COPY_DST | TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Capture Staging Buffer"),
            size: (bytes_per_row * height) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        Self {
            staging_texture,
            staging_buffer,
            width,
            height,
            bytes_per_row,
            frame_count: 0,
        }
    }

    /// Capture frame from render target
    pub fn capture(
        &mut self,
        device: &Device,
        queue: &Queue,
        source_texture: &Texture,
    ) -> Result<CapturedFrame> {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Capture Encoder"),
        });

        // Copy render target to staging texture
        encoder.copy_texture_to_texture(
            wgpu::ImageCopyTexture {
                texture: source_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyTexture {
                texture: &self.staging_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );

        // Copy staging texture to buffer
        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: &self.staging_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &self.staging_buffer,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(self.bytes_per_row),
                    rows_per_image: Some(self.height),
                },
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );

        queue.submit(std::iter::once(encoder.finish()));

        // Map and read buffer
        let buffer_slice = self.staging_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        device.poll(wgpu::Maintain::Wait);
        rx.recv()??;

        let data = buffer_slice.get_mapped_range();

        // Remove row padding
        let mut rgba_data = Vec::with_capacity((self.width * self.height * 4) as usize);
        for row in 0..self.height {
            let start = (row * self.bytes_per_row) as usize;
            let end = start + (self.width * 4) as usize;
            rgba_data.extend_from_slice(&data[start..end]);
        }

        drop(data);
        self.staging_buffer.unmap();

        let timestamp_us = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_micros() as u64;

        self.frame_count += 1;

        Ok(CapturedFrame {
            width: self.width,
            height: self.height,
            data: rgba_data,
            timestamp_us,
        })
    }
}
