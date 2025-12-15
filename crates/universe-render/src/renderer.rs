//! Main renderer orchestrating everything

use crate::camera::{Camera, CameraUniform, RasterCameraUniform};
use crate::gpu_types::GpuSplat;
use crate::pipeline::{SplatPipeline, create_depth_texture};
use crate::tile_pipeline::TilePipeline;
use crate::streaming::{GpuCache, StreamingManager};

use universe_sim::{SolarSystem, TimeController, Body};
use universe_raster::gpu_types::TileParams;

use glam::DVec3;
use winit::window::Window;
use std::sync::Arc;
use anyhow::Result;
use wgpu::util::DeviceExt;

/// GPU buffer capacity (200MB to stay within device limits)
const GPU_BUFFER_CAPACITY: u64 = 200 * 1024 * 1024;

pub struct Renderer {
    // WGPU state
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,

    // Legacy pipeline (billboard rendering)
    pipeline: SplatPipeline,
    depth_texture: wgpu::Texture,
    depth_view: wgpu::TextureView,

    // Tile-based pipeline (proper 3DGS)
    tile_pipeline: TilePipeline,

    // Buffers
    camera_buffer: wgpu::Buffer,
    raster_camera_buffer: wgpu::Buffer,
    tile_params_buffer: wgpu::Buffer,
    #[allow(dead_code)]
    gpu_cache: GpuCache,

    // State
    pub camera: Camera,
    pub solar_system: SolarSystem,
    pub time_controller: TimeController,

    // Data
    streaming: StreamingManager,

    // Current frame data
    splat_bind_group: Option<wgpu::BindGroup>,
    current_splat_count: u32,
    current_splat_buffer: Option<wgpu::Buffer>,

    // Rendering mode
    pub use_tile_pipeline: bool,
}

impl Renderer {
    pub async fn new(window: Arc<Window>, universe_dir: &std::path::Path) -> Result<Self> {
        let size = window.inner_size();

        // WGPU setup
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });

        let surface = instance.create_surface(Arc::clone(&window))?;

        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }).await.ok_or_else(|| anyhow::anyhow!("No GPU adapter found"))?;

        let (device, queue) = adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("Universe Device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: Default::default(),
            },
            None,
        ).await?;

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps.formats.iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);

        tracing::info!("Window inner_size: {}x{}", size.width, size.height);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        tracing::info!("Configuring surface: {}x{}", config.width, config.height);
        surface.configure(&device, &config);

        // Pipeline
        let pipeline = SplatPipeline::new(&device, surface_format);

        // Depth buffer
        let (depth_texture, depth_view) = create_depth_texture(&device, size.width, size.height);

        // Camera buffer (legacy)
        let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Camera Buffer"),
            size: std::mem::size_of::<CameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Raster camera buffer (tile pipeline)
        let raster_camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Raster Camera Buffer"),
            size: std::mem::size_of::<RasterCameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Tile params buffer
        let tile_params = TileParams::new(size.width, size.height);
        let tile_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Tile Params Buffer"),
            contents: bytemuck::bytes_of(&tile_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Tile pipeline
        tracing::info!("Creating tile pipeline with size {}x{}", size.width, size.height);
        let tile_pipeline = TilePipeline::new(&device, size.width, size.height, surface_format);

        // GPU cache
        let gpu_cache = GpuCache::new(&device, GPU_BUFFER_CAPACITY);

        // Streaming manager
        let streaming = StreamingManager::new(universe_dir, 1000)?;

        Ok(Self {
            surface,
            device,
            queue,
            config,
            pipeline,
            depth_texture,
            depth_view,
            tile_pipeline,
            camera_buffer,
            raster_camera_buffer,
            tile_params_buffer,
            gpu_cache,
            camera: Camera::new(),
            solar_system: SolarSystem::new(),
            time_controller: TimeController::new(),
            streaming,
            splat_bind_group: None,
            current_splat_count: 0,
            current_splat_buffer: None,
            use_tile_pipeline: true, // Enable tile pipeline by default
        })
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            tracing::info!("Resizing to {}x{}", new_size.width, new_size.height);
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);

            let (depth_texture, depth_view) = create_depth_texture(
                &self.device,
                new_size.width,
                new_size.height,
            );
            self.depth_texture = depth_texture;
            self.depth_view = depth_view;

            // Resize tile pipeline
            self.tile_pipeline.resize(&self.device, new_size.width, new_size.height);

            // Update tile params
            let tile_params = TileParams::new(new_size.width, new_size.height);
            self.queue.write_buffer(&self.tile_params_buffer, 0, bytemuck::bytes_of(&tile_params));
        }
    }

    /// Update simulation state
    pub fn update(&mut self, dt: f32) {
        // Advance time
        let epoch = self.time_controller.tick(dt as f64);
        self.solar_system.set_epoch(epoch);

        // Update camera speed based on position
        self.camera.auto_speed();
    }

    /// Load visible splats to GPU
    pub fn prepare_frame(&mut self) -> Result<()> {
        // Phase 1.3: Get visible cells using frustum culling
        let camera_forward = self.camera.forward();
        let camera_up = self.camera.up();
        let aspect = self.config.width as f32 / self.config.height as f32;
        let max_distance = self.camera.far as f64 * 0.9; // Use 90% of far plane
        
        let visible = self.streaming.get_visible_cells_frustum(
            self.camera.position,
            camera_forward,
            camera_up,
            self.camera.fov_y,
            aspect,
            max_distance,
        );

        // Collect splats from visible cells
        let mut all_splats: Vec<GpuSplat> = Vec::new();

        for cell_id in visible.iter().take(100) { // Limit for performance
            if let Ok(cell) = self.streaming.cell_cache.get(*cell_id) {
                for splat in &cell.splats {
                    // Transform to camera-relative coordinates
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

        // Add planet splats from simulation
        // Scale factor to make planets visible at solar system scale
        // Real planets are ~10^7 meters, solar system is ~10^12 meters
        // We scale up by 1000x to make them visible as ~1% of orbital distance
        const PLANET_VISUAL_SCALE: f32 = 1000.0;

        for body in Body::planets() {
            let pos = self.solar_system.body_position(*body);
            let world_pos = DVec3::new(pos.x, pos.y, pos.z);
            let camera_rel = self.camera.world_to_camera_relative(world_pos);

            // Scale radius for visibility (real radius * scale factor)
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

        self.current_splat_count = all_splats.len() as u32;

        if all_splats.is_empty() {
            self.splat_bind_group = None;
            return Ok(());
        }

        // Upload to GPU (simple: recreate buffer each frame for now)
        let splat_data = bytemuck::cast_slice(&all_splats);
        let splat_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Splat Data"),
            contents: splat_data,
            usage: wgpu::BufferUsages::STORAGE,
        });

        // Create bind group (for legacy pipeline)
        self.splat_bind_group = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
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
        }));

        // Store splat buffer for tile pipeline (move it last)
        self.current_splat_buffer = Some(splat_buffer);

        Ok(())
    }

    pub fn render(&mut self) -> Result<()> {
        let output = self.surface.get_current_texture()?;
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Update camera uniforms
        let aspect = self.config.width as f32 / self.config.height as f32;
        let camera_uniform = CameraUniform::from_camera(&self.camera, aspect);
        self.queue.write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(&camera_uniform));

        // Update raster camera uniform (for tile pipeline)
        let raster_camera_uniform = RasterCameraUniform::from_camera(
            &self.camera,
            self.config.width,
            self.config.height,
        );
        self.queue.write_buffer(&self.raster_camera_buffer, 0, bytemuck::bytes_of(&raster_camera_uniform));

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });

        if self.use_tile_pipeline {
            // Use tile-based rendering pipeline
            if let Some(splat_buffer) = &self.current_splat_buffer {
                self.tile_pipeline.render(
                    &self.device,
                    &self.queue,
                    &mut encoder,
                    splat_buffer,
                    self.current_splat_count,
                    &self.raster_camera_buffer,
                    &self.tile_params_buffer,
                );

                // Blit output to surface
                self.tile_pipeline.blit_to_surface(&self.device, &mut encoder, &view);
            } else {
                // No splats - just clear
                let _render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Clear Pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: 0.0, g: 0.0, b: 0.02, a: 1.0,
                            }),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
            }
        } else {
            // Use legacy billboard rendering pipeline
            {
                let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Splat Pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: 0.0, g: 0.0, b: 0.02, a: 1.0, // Dark space
                            }),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &self.depth_view,
                        depth_ops: Some(wgpu::Operations {
                            // REVERSE-Z: Clear to 0.0 (far)
                            load: wgpu::LoadOp::Clear(0.0),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });

                if let Some(bind_group) = &self.splat_bind_group {
                    render_pass.set_pipeline(&self.pipeline.pipeline);
                    render_pass.set_bind_group(0, bind_group, &[]);

                    // 6 vertices per splat (2 triangles)
                    render_pass.draw(0..6, 0..self.current_splat_count);
                }
            }
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }

    /// Toggle between tile-based and legacy rendering
    pub fn toggle_pipeline(&mut self) {
        self.use_tile_pipeline = !self.use_tile_pipeline;
        tracing::info!("Rendering pipeline: {}", if self.use_tile_pipeline { "Tile-based 3DGS" } else { "Legacy Billboard" });
    }

    pub fn get_info(&self) -> String {
        format!(
            "Splats: {} | Year: {:.1} | Rate: {:.0}x | Pos: ({:.2e}, {:.2e}, {:.2e})",
            self.current_splat_count,
            self.time_controller.year(),
            self.time_controller.rate() / 86400.0 / 365.25,
            self.camera.position.x,
            self.camera.position.y,
            self.camera.position.z,
        )
    }
}
