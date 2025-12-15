use std::sync::Arc;
use std::time::Instant;
use std::{fs, mem};

use anyhow::{ensure, Context, Result};
use glam::Vec3;
use wgpu::util::DeviceExt;
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::{DeviceEvent, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowId},
};

mod camera;
mod geometry;

mod assets {
    pub mod loader;
    pub mod structs;
}

use crate::assets::loader::AssetLoader;
use crate::assets::structs::{KeplerParams, PackedResidual};
use camera::{Camera, CameraController};

// Matches `MAX_ASTEROIDS` in `compile_belt.py` (the loader also reads dynamically).
const MAX_ASTEROIDS: u32 = 100_000;

fn kepler_params_from_le_bytes(bytes: &[u8]) -> KeplerParams {
    // Must match `KeplerParams` layout: 7x f32 + 1x u32 (32 bytes), little-endian.
    debug_assert_eq!(bytes.len(), mem::size_of::<KeplerParams>());
    let f = |off: usize| f32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
    let u = |off: usize| u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());

    KeplerParams {
        semi_major_axis: f(0),
        eccentricity: f(4),
        inclination: f(8),
        arg_periapsis: f(12),
        long_asc_node: f(16),
        mean_anomaly_0: f(20),
        residual_scale: f(24),
        count: u(28),
    }
}

#[repr(C, align(16))]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct GlobalUniforms {
    view_proj: [[f32; 4]; 4],
    time: f32,
    // WGSL `vec3<f32>` has 16-byte alignment in uniform buffers; insert explicit padding
    // so the host-side struct matches Naga's expected size (96 bytes total).
    _pad_time: [f32; 3],
    _pad0: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],
}

impl Vertex {
    fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &wgpu::vertex_attr_array![0 => Float32x3],
        }
    }
}

fn create_depth_texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Depth Texture"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });

    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

fn solve_kepler(m: f32, e: f32) -> f32 {
    let mut e_anom = m;
    for _ in 0..5 {
        e_anom = e_anom - (e_anom - e * e_anom.sin() - m) / (1.0 - e * e_anom.cos());
    }
    e_anom
}

fn orbit_pos_kepler(params: &KeplerParams, t_days: f32) -> Vec3 {
    // Mirrors the shader's get_orbit_pos_with_residuals(t), but without residuals.
    // Units: AU-ish (whatever your assets encode), time in days.
    let n = 0.017202 / params.semi_major_axis.powf(1.5);
    let m = params.mean_anomaly_0 + n * t_days;
    let e_anom = solve_kepler(m, params.eccentricity);

    let a = params.semi_major_axis;
    let e = params.eccentricity;
    let x_orb = a * (e_anom.cos() - e);
    let y_orb = a * (1.0 - e * e).sqrt() * e_anom.sin();

    let w = params.arg_periapsis;
    let i = params.inclination;
    let o = params.long_asc_node;

    let (sw, cw) = w.sin_cos();
    let (si, ci) = i.sin_cos();
    let (so, co) = o.sin_cos();

    // Rotate by Argument of Periapsis (w) around Z
    let x1 = x_orb * cw - y_orb * sw;
    let y1 = x_orb * sw + y_orb * cw;

    // Rotate by Inclination (i) around X
    let x2 = x1;
    let y2 = y1 * ci;
    let z2 = y1 * si;

    // Rotate by Longitude of Ascending Node (O) around Z
    let x_final = x2 * co - y2 * so;
    let y_final = x2 * so + y2 * co;
    let z_final = z2;

    Vec3::new(x_final, y_final, z_final)
}

struct State {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: PhysicalSize<u32>,

    depth_texture: wgpu::Texture,
    depth_view: wgpu::TextureView,

    pipeline: wgpu::RenderPipeline,
    bind_group_0: wgpu::BindGroup,
    bind_group_1: wgpu::BindGroup,

    global_buffer: wgpu::Buffer,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_count: u32,
    num_bodies: u32,

    start: Instant,

    camera: Camera,
    camera_controller: CameraController,
}

impl State {
    async fn new(window: Arc<Window>) -> Result<Self> {
        let size = window.inner_size();

        // 1) WGPU init
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });

        let surface = instance
            .create_surface(Arc::clone(&window))
            .context("create_surface failed")?;

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .context("No suitable GPU adapters found")?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Neural Planetarium Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: Default::default(),
                },
                None,
            )
            .await
            .context("request_device failed")?;

        let caps = surface.get_capabilities(&adapter);
        let surface_format = caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(caps.formats[0]);

        let present_mode = if caps.present_modes.contains(&wgpu::PresentMode::Mailbox) {
            wgpu::PresentMode::Mailbox
        } else {
            wgpu::PresentMode::Fifo
        };

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let (depth_texture, depth_view) = create_depth_texture(&device, config.width, config.height);

        // 2) Asset loading
        let (_eros_orbit_params, residuals) = AssetLoader::load_orbit_data("assets/eros")
            .context("Failed to load orbit data (assets/eros_*.{json,bin})")?;
        let neural_weights = AssetLoader::load_neural_brain("assets/neural_decoder.bin")
            .context("Failed to load neural weights (assets/neural_decoder.bin)")?;

        // 3) GPU buffers
        // --- 1. LOAD REAL BELT ---
        let mut orbit_data = fs::read("assets/real_belt.bin")
            .context("Failed to read assets/real_belt.bin (run `python compile_belt.py` first)")?;
        let bytes_per_orbit = mem::size_of::<KeplerParams>();
        ensure!(
            orbit_data.len() % bytes_per_orbit == 0,
            "assets/real_belt.bin length {} is not a multiple of {} bytes",
            orbit_data.len(),
            bytes_per_orbit
        );

        let mut num_bodies = (orbit_data.len() / bytes_per_orbit) as u32;
        ensure!(
            num_bodies > 0,
            "assets/real_belt.bin contains 0 asteroids (file is empty)"
        );

        // Safety clamp (keeps runtime memory usage predictable).
        if num_bodies > MAX_ASTEROIDS {
            num_bodies = MAX_ASTEROIDS;
            orbit_data.truncate(num_bodies as usize * bytes_per_orbit);
        }

        println!("Loaded {} Real Asteroids", num_bodies);

        let orbit_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Orbit Params Storage Buffer"),
            contents: &orbit_data,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let residuals_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Residuals Storage Buffer"),
            contents: bytemuck::cast_slice::<PackedResidual, u8>(&residuals),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let neural_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Neural Weights Storage Buffer"),
            contents: bytemuck::cast_slice::<f32, u8>(&neural_weights),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        // Aim the starting camera at the asteroid's actual orbit position at t=0,
        // otherwise it's easy to start out looking at empty space (black screen).
        let first_orbit =
            kepler_params_from_le_bytes(&orbit_data[0..mem::size_of::<KeplerParams>()]);
        let world_center_t0 = orbit_pos_kepler(&first_orbit, 0.0);

        let mut camera = Camera {
            eye: world_center_t0 + Vec3::new(0.0, 0.05, 0.25),
            target: world_center_t0,
            up: Vec3::Y,
            aspect: config.width as f32 / config.height as f32,
            fovy: 45.0f32.to_radians(),
            znear: 0.001, // Allow getting VERY close
            zfar: 1000.0,
        };

        let mut camera_controller = CameraController::new(0.05, 0.002); // Slower speed for precision
        camera_controller.look_at(camera.eye, camera.target);
        camera_controller.update_camera(&mut camera);

        let global = GlobalUniforms {
            view_proj: camera.build_view_projection_matrix().to_cols_array_2d(),
            time: 0.0,
            _pad_time: [0.0; 3],
            _pad0: [0.0; 4],
        };

        let global_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Global Uniform Buffer"),
            contents: bytemuck::bytes_of(&global),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Geometry: procedural sphere (better visual sanity check than a single triangle)
        let (positions, indices) = geometry::generate_uv_sphere(geometry::SphereOptions {
            radius: 1.0,
            stacks: 24,
            slices: 48,
        });
        let verts: Vec<Vertex> = positions
            .into_iter()
            .map(|p| Vertex { position: p })
            .collect();

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Sphere Vertex Buffer"),
            contents: bytemuck::cast_slice(verts.as_slice()),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_count = indices.len() as u32;
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Sphere Index Buffer"),
            contents: bytemuck::cast_slice(indices.as_slice()),
            usage: wgpu::BufferUsages::INDEX,
        });

        // 4) Pipeline
        let shader = device.create_shader_module(wgpu::include_wgsl!("assets/shader.wgsl"));

        let bind_group_layout_0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("BindGroupLayout0 (Orbit/Residuals/Neural)"),
            entries: &[
                // orbit
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // residuals
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // neural weights
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let bind_group_layout_1 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("BindGroupLayout1 (Globals)"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let bind_group_0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("BindGroup0"),
            layout: &bind_group_layout_0,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: orbit_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: residuals_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: neural_buffer.as_entire_binding(),
                },
            ],
        });

        let bind_group_1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("BindGroup1"),
            layout: &bind_group_layout_1,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: global_buffer.as_entire_binding(),
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Planetarium Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout_0, &bind_group_layout_1],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Planetarium Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::layout()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None, // be robust to any winding mistakes in prototype mesh
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        Ok(Self {
            surface,
            device,
            queue,
            config,
            size,
            depth_texture,
            depth_view,
            pipeline,
            bind_group_0,
            bind_group_1,
            global_buffer,
            vertex_buffer,
            index_buffer,
            index_count,
            num_bodies,
            start: Instant::now(),
            camera,
            camera_controller,
        })
    }

    fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width == 0 || new_size.height == 0 {
            return;
        }

        self.size = new_size;
        self.config.width = new_size.width;
        self.config.height = new_size.height;
        self.surface.configure(&self.device, &self.config);

        self.camera.aspect = self.config.width as f32 / self.config.height as f32;

        let (depth_texture, depth_view) = create_depth_texture(&self.device, self.config.width, self.config.height);
        self.depth_texture = depth_texture;
        self.depth_view = depth_view;
    }

    fn update(&mut self) {
        self.camera_controller.update_camera(&mut self.camera);

        let elapsed_s = self.start.elapsed().as_secs_f32();
        let time_days = elapsed_s / 86400.0;

        let view_proj = self.camera.build_view_projection_matrix();
        let globals = GlobalUniforms {
            view_proj: view_proj.to_cols_array_2d(),
            time: time_days,
            _pad_time: [0.0; 3],
            _pad0: [0.0; 4],
        };

        self.queue
            .write_buffer(&self.global_buffer, 0, bytemuck::bytes_of(&globals));
    }

    fn render(&mut self) -> Result<()> {
        let frame = match self.surface.get_current_texture() {
            Ok(f) => f,
            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                self.surface.configure(&self.device, &self.config);
                return Ok(());
            }
            Err(wgpu::SurfaceError::OutOfMemory) => {
                anyhow::bail!("Surface out of memory");
            }
            Err(e) => {
                return Err(anyhow::anyhow!(e));
            }
        };

        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            rpass.set_pipeline(&self.pipeline);
            rpass.set_bind_group(0, &self.bind_group_0, &[]);
            rpass.set_bind_group(1, &self.bind_group_1, &[]);
            rpass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            rpass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);

            // Draw the real belt (instance count determined by assets/real_belt.bin).
            rpass.draw_indexed(0..self.index_count, 0, 0..self.num_bodies);
        }

        self.queue.submit(Some(encoder.finish()));
        frame.present();
        Ok(())
    }
}

struct App {
    window: Option<Arc<Window>>,
    state: Option<State>,
}

impl App {
    fn new() -> Self {
        Self {
            window: None,
            state: None,
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let attrs = Window::default_attributes()
            .with_title("Neural Planetarium")
            .with_inner_size(PhysicalSize::new(1280, 720));

        let window = Arc::new(event_loop.create_window(attrs).unwrap());
        let state = pollster::block_on(State::new(Arc::clone(&window))).unwrap();

        self.window = Some(window);
        self.state = Some(state);

        if let Some(w) = &self.window {
            w.request_redraw();
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        // PASS INPUT TO CAMERA FIRST
        if let Some(state) = &mut self.state {
            if state.camera_controller.process_events(&event) {
                return;
            }
        }

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),

            WindowEvent::Resized(size) => {
                if let Some(state) = &mut self.state {
                    state.resize(size);
                }
            }

            WindowEvent::RedrawRequested => {
                if let Some(state) = &mut self.state {
                    state.update();
                    let _ = state.render();
                }

                if let Some(w) = &self.window {
                    w.request_redraw();
                }
            }

            _ => {}
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: winit::event::DeviceId,
        event: DeviceEvent,
    ) {
        if let Some(state) = &mut self.state {
            if let DeviceEvent::MouseMotion { delta } = event {
                state.camera_controller.process_mouse(delta.0, delta.1);
            }
        }
    }
}

fn main() -> Result<()> {
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::new();
    event_loop.run_app(&mut app)?;

    Ok(())
}
