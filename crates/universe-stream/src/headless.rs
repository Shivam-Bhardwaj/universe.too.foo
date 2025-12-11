//! Headless renderer for streaming (no window)

use universe_render::Camera;
use universe_sim::{SolarSystem, TimeController, Body};
use universe_render::gpu_types::GpuSplat;
use universe_render::camera::CameraUniform;
use universe_render::streaming::StreamingManager;
use crate::capture::{FrameCapture, CapturedFrame};

use wgpu::{Device, Queue, Texture, TextureDescriptor, TextureFormat, TextureUsages};
use glam::DVec3;
use std::sync::Arc;
use std::path::PathBuf;
use anyhow::Result;

/// Headless renderer configuration
#[derive(Clone, Debug)]
pub struct HeadlessConfig {
    pub width: u32,
    pub height: u32,
    pub universe_dir: PathBuf,
}

impl Default for HeadlessConfig {
    fn default() -> Self {
        Self {
            width: 1920,
            height: 1080,
            universe_dir: PathBuf::from("universe"),
        }
    }
}

/// Headless renderer for streaming
pub struct HeadlessRenderer {
    device: Arc<Device>,
    queue: Arc<Queue>,

    // Render target
    render_texture: Texture,
    render_view: wgpu::TextureView,
    depth_texture: Texture,
    depth_view: wgpu::TextureView,

    // Pipeline
    pipeline: universe_render::pipeline::SplatPipeline,
    camera_buffer: wgpu::Buffer,

    // Capture and encode
    capture: FrameCapture,

    // State
    pub camera: Camera,
    pub solar_system: SolarSystem,
    pub time_controller: TimeController,

    // Data streaming (same dataset format as the native renderer)
    streaming: StreamingManager,

    config: HeadlessConfig,
}

impl HeadlessRenderer {
    pub async fn new(config: HeadlessConfig) -> Result<Self> {
        // Create WGPU instance (headless)
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });

        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None, // Headless!
            force_fallback_adapter: false,
        }).await.ok_or_else(|| anyhow::anyhow!("No GPU adapter"))?;

        let (device, queue) = adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("Universe Headless"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: Default::default(),
            },
            None,
        ).await?;

        let device = Arc::new(device);
        let queue = Arc::new(queue);

        // Create render target texture
        let render_texture = device.create_texture(&TextureDescriptor {
            label: Some("Render Target"),
            size: wgpu::Extent3d {
                width: config.width,
                height: config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: TextureFormat::Rgba8UnormSrgb,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let render_view = render_texture.create_view(&Default::default());

        // Depth texture
        let (depth_texture, depth_view) = universe_render::pipeline::create_depth_texture(
            &device,
            config.width,
            config.height,
        );

        // Pipeline
        let pipeline = universe_render::pipeline::SplatPipeline::new(&device, TextureFormat::Rgba8UnormSrgb);

        // Camera buffer
        let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Camera"),
            size: std::mem::size_of::<CameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Frame capture
        let capture = FrameCapture::new(&device, config.width, config.height);

        // Streaming manager (loads index.json + cells/)
        let streaming = StreamingManager::new(&config.universe_dir, 1000)?;

        Ok(Self {
            device,
            queue,
            render_texture,
            render_view,
            depth_texture,
            depth_view,
            pipeline,
            camera_buffer,
            capture,
            camera: Camera::new(),
            solar_system: SolarSystem::new(),
            time_controller: TimeController::new(),
            streaming,
            config,
        })
    }

    /// Render one frame (returns raw RGBA pixels)
    pub fn render_frame(&mut self) -> Result<CapturedFrame> {
        // Update simulation
        let dt = 1.0 / 60.0;
        let epoch = self.time_controller.tick(dt);
        self.solar_system.set_epoch(epoch);
        self.camera.auto_speed();

        // Collect splats from visible dataset cells
        let visible = self.streaming.get_visible_cells();
        let mut all_splats: Vec<GpuSplat> = Vec::new();

        for cell_id in visible.iter() {
            if let Ok(cell) = self.streaming.cell_cache.get(*cell_id) {
                for splat in &cell.splats {
                    let world_pos = DVec3::new(
                        cell.metadata.bounds.centroid.x + splat.pos[0] as f64,
                        cell.metadata.bounds.centroid.y + splat.pos[1] as f64,
                        cell.metadata.bounds.centroid.z + splat.pos[2] as f64,
                    );
                    let camera_rel = self.camera.world_to_camera_relative(world_pos);

                    all_splats.push(GpuSplat {
                        pos: camera_rel.into(),
                        _pad0: 0.0,
                        scale: splat.scale,
                        _pad1: 0.0,
                        rotation: splat.rotation,
                        color: splat.color,
                        opacity: splat.opacity,
                    });
                }
            }
        }

        // Add planet splats from simulation (for guaranteed visibility even if dataset is star-only)
        const PLANET_VISUAL_SCALE: f32 = 1000.0;
        for body in Body::planets() {
            let pos = self.solar_system.body_position(*body);
            let world_pos = DVec3::new(pos.x, pos.y, pos.z);
            let camera_rel = self.camera.world_to_camera_relative(world_pos);

            let visual_radius = body.radius() as f32 * PLANET_VISUAL_SCALE;
            all_splats.push(GpuSplat {
                pos: camera_rel.into(),
                _pad0: 0.0,
                scale: [visual_radius; 3],
                _pad1: 0.0,
                rotation: [0.0, 0.0, 0.0, 1.0],
                color: match body {
                    Body::Mercury => [0.6, 0.6, 0.6],
                    Body::Venus => [0.9, 0.85, 0.7],
                    Body::Earth => [0.2, 0.4, 0.8],
                    Body::Mars => [0.8, 0.4, 0.2],
                    Body::Jupiter => [0.9, 0.8, 0.6],
                    Body::Saturn => [0.9, 0.85, 0.6],
                    Body::Uranus => [0.6, 0.8, 0.9],
                    Body::Neptune => [0.3, 0.5, 0.9],
                    Body::Pluto => [0.8, 0.75, 0.7],
                    _ => [1.0, 1.0, 1.0],
                },
                opacity: 1.0,
            });
        }

        // Create splat buffer
        let splat_data = bytemuck::cast_slice(&all_splats);
        let splat_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Splat Data"),
            contents: splat_data,
            usage: wgpu::BufferUsages::STORAGE,
        });

        // Update camera uniform
        let aspect = self.config.width as f32 / self.config.height as f32;
        let camera_uniform = CameraUniform::from_camera(&self.camera, aspect);
        self.queue.write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(&camera_uniform));

        // Create bind group
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Splat Bind Group"),
            layout: &self.pipeline.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.camera_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: splat_buffer.as_entire_binding(),
                },
            ],
        });

        // Render
        let mut encoder = self.device.create_command_encoder(&Default::default());

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.render_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0, g: 0.0, b: 0.02, a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(0.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            pass.set_pipeline(&self.pipeline.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.draw(0..6, 0..(all_splats.len() as u32));
        }

        self.queue.submit(std::iter::once(encoder.finish()));

        // Capture frame
        let captured = self.capture.capture(&self.device, &self.queue, &self.render_texture)?;
        Ok(captured)
    }
}

use wgpu::util::DeviceExt;
