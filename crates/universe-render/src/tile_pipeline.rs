//! Tile-based Gaussian Splatting rendering pipeline
//!
//! This module implements proper 3DGS rendering with:
//! 1. 3D Gaussian to 2D covariance projection
//! 2. Tile-based depth sorting
//! 3. Front-to-back alpha blending

use universe_raster::{shaders, gpu_types::*, TILE_SIZE};

use bytemuck::{Pod, Zeroable};

/// Blit shader source
const BLIT_SHADER: &str = include_str!("shaders/blit.wgsl");

/// Maximum number of splats we can handle
const MAX_SPLATS: u32 = 1_000_000;
/// Maximum tile keys (splats * avg tiles per splat)
/// Limited to fit within default max_storage_buffer_binding_size (128MB)
const MAX_TILE_KEYS: u32 = MAX_SPLATS * 4; // 4M keys * 16 bytes = 64MB

/// Tile-based rendering pipeline
pub struct TilePipeline {
    // Compute pipelines
    project_pipeline: wgpu::ComputePipeline,
    tile_assign_pipeline: wgpu::ComputePipeline,
    sort_pipeline: wgpu::ComputePipeline,
    tile_ranges_pipeline: wgpu::ComputePipeline,
    raster_pipeline: wgpu::ComputePipeline,

    // Blit pipeline for copying output to surface
    blit_pipeline: wgpu::RenderPipeline,
    blit_layout: wgpu::BindGroupLayout,
    blit_sampler: wgpu::Sampler,

    // Bind group layouts
    project_layout: wgpu::BindGroupLayout,
    tile_assign_layout: wgpu::BindGroupLayout,
    sort_layout: wgpu::BindGroupLayout,
    tile_ranges_layout: wgpu::BindGroupLayout,
    raster_layout: wgpu::BindGroupLayout,

    // Persistent buffers
    splats_2d_buffer: wgpu::Buffer,
    tile_keys_buffer: wgpu::Buffer,
    tile_ranges_buffer: wgpu::Buffer,
    visible_count_buffer: wgpu::Buffer,
    key_count_buffer: wgpu::Buffer,

    // Output texture
    output_texture: wgpu::Texture,
    output_view: wgpu::TextureView,

    // Parameters
    width: u32,
    height: u32,
    num_tiles: u32,
}

/// Sort parameters uniform
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct SortParams {
    num_elements: u32,
    stage: u32,
    substage: u32,
    _pad: u32,
}

/// Tile range params uniform
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct TileRangeParams {
    num_keys: u32,
    num_tiles: u32,
    _pad0: u32,
    _pad1: u32,
}

impl TilePipeline {
    pub fn new(device: &wgpu::Device, width: u32, height: u32, surface_format: wgpu::TextureFormat) -> Self {
        let num_tiles_x = (width + TILE_SIZE - 1) / TILE_SIZE;
        let num_tiles_y = (height + TILE_SIZE - 1) / TILE_SIZE;
        let num_tiles = num_tiles_x * num_tiles_y;

        // Create shader modules
        let project_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Project Shader"),
            source: wgpu::ShaderSource::Wgsl(shaders::PROJECT.into()),
        });

        let tile_assign_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Tile Assign Shader"),
            source: wgpu::ShaderSource::Wgsl(shaders::TILE_ASSIGN.into()),
        });

        let sort_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Sort Shader"),
            source: wgpu::ShaderSource::Wgsl(shaders::SORT.into()),
        });

        let tile_ranges_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Tile Ranges Shader"),
            source: wgpu::ShaderSource::Wgsl(shaders::TILE_RANGES.into()),
        });

        let raster_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Raster Tile Shader"),
            source: wgpu::ShaderSource::Wgsl(shaders::RASTER_TILE.into()),
        });

        // === Project Pipeline ===
        let project_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Project Bind Group Layout"),
            entries: &[
                // Input splats
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Output Splat2D
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Camera uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Visible count (atomic)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let project_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Project Pipeline Layout"),
            bind_group_layouts: &[&project_layout],
            push_constant_ranges: &[],
        });

        let project_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Project Pipeline"),
            layout: Some(&project_pipeline_layout),
            module: &project_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // === Tile Assignment Pipeline ===
        let tile_assign_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Tile Assign Bind Group Layout"),
            entries: &[
                // Splat2D input
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Tile keys output
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Key count (atomic)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Tile params uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let tile_assign_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Tile Assign Pipeline Layout"),
            bind_group_layouts: &[&tile_assign_layout],
            push_constant_ranges: &[],
        });

        let tile_assign_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Tile Assign Pipeline"),
            layout: Some(&tile_assign_pipeline_layout),
            module: &tile_assign_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // === Sort Pipeline ===
        let sort_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Sort Bind Group Layout"),
            entries: &[
                // Keys (read/write)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Sort params
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let sort_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Sort Pipeline Layout"),
            bind_group_layouts: &[&sort_layout],
            push_constant_ranges: &[],
        });

        let sort_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Sort Pipeline"),
            layout: Some(&sort_pipeline_layout),
            module: &sort_shader,
            entry_point: Some("bitonic_sort_step"),
            compilation_options: Default::default(),
            cache: None,
        });

        // === Tile Ranges Pipeline ===
        let tile_ranges_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Tile Ranges Bind Group Layout"),
            entries: &[
                // Sorted keys
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Tile ranges output
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Params
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let tile_ranges_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Tile Ranges Pipeline Layout"),
            bind_group_layouts: &[&tile_ranges_layout],
            push_constant_ranges: &[],
        });

        let tile_ranges_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Tile Ranges Pipeline"),
            layout: Some(&tile_ranges_pipeline_layout),
            module: &tile_ranges_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // === Raster Pipeline ===
        let raster_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Raster Bind Group Layout"),
            entries: &[
                // Splat2D
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Sorted keys
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Tile ranges
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Output texture
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        let raster_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Raster Pipeline Layout"),
            bind_group_layouts: &[&raster_layout],
            push_constant_ranges: &[],
        });

        let raster_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Raster Pipeline"),
            layout: Some(&raster_pipeline_layout),
            module: &raster_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // === Blit Pipeline ===
        let blit_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Blit Shader"),
            source: wgpu::ShaderSource::Wgsl(BLIT_SHADER.into()),
        });

        let blit_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Blit Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let blit_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Blit Pipeline Layout"),
            bind_group_layouts: &[&blit_layout],
            push_constant_ranges: &[],
        });

        let blit_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Blit Pipeline"),
            layout: Some(&blit_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &blit_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &blit_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let blit_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Blit Sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // === Create Buffers ===
        let splats_2d_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Splat2D Buffer"),
            size: (MAX_SPLATS as usize * Splat2D::SIZE) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let tile_keys_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tile Keys Buffer"),
            size: (MAX_TILE_KEYS as usize * TileKey::SIZE) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let tile_ranges_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tile Ranges Buffer"),
            size: (num_tiles as usize * TileRange::SIZE) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let visible_count_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Visible Count Buffer"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let key_count_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Key Count Buffer"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // === Create Output Texture ===
        let output_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Raster Output Texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let output_view = output_texture.create_view(&wgpu::TextureViewDescriptor::default());

        Self {
            project_pipeline,
            tile_assign_pipeline,
            sort_pipeline,
            tile_ranges_pipeline,
            raster_pipeline,
            blit_pipeline,
            blit_layout,
            blit_sampler,
            project_layout,
            tile_assign_layout,
            sort_layout,
            tile_ranges_layout,
            raster_layout,
            splats_2d_buffer,
            tile_keys_buffer,
            tile_ranges_buffer,
            visible_count_buffer,
            key_count_buffer,
            output_texture,
            output_view,
            width,
            height,
            num_tiles,
        }
    }

    /// Resize output texture when window size changes
    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        if width == self.width && height == self.height {
            return;
        }

        self.width = width;
        self.height = height;

        let num_tiles_x = (width + TILE_SIZE - 1) / TILE_SIZE;
        let num_tiles_y = (height + TILE_SIZE - 1) / TILE_SIZE;
        self.num_tiles = num_tiles_x * num_tiles_y;

        // Recreate output texture
        self.output_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Raster Output Texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        self.output_view = self.output_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Recreate tile ranges buffer
        self.tile_ranges_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tile Ranges Buffer"),
            size: (self.num_tiles as usize * TileRange::SIZE) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
    }

    /// Get output texture view for display
    pub fn output_view(&self) -> &wgpu::TextureView {
        &self.output_view
    }

    /// Get output texture for copying
    pub fn output_texture(&self) -> &wgpu::Texture {
        &self.output_texture
    }

    /// Execute the full tile-based rendering pipeline
    ///
    /// # Arguments
    /// * `device` - WGPU device
    /// * `queue` - WGPU queue
    /// * `encoder` - Command encoder to record commands into
    /// * `splat_buffer` - Buffer containing input GpuSplat data
    /// * `num_splats` - Number of splats to render
    /// * `camera_buffer` - Buffer containing RasterCameraUniform
    /// * `tile_params_buffer` - Buffer containing TileParams
    pub fn render(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        splat_buffer: &wgpu::Buffer,
        num_splats: u32,
        camera_buffer: &wgpu::Buffer,
        tile_params_buffer: &wgpu::Buffer,
    ) {
        if num_splats == 0 {
            return;
        }

        let num_tiles_x = (self.width + TILE_SIZE - 1) / TILE_SIZE;
        let num_tiles_y = (self.height + TILE_SIZE - 1) / TILE_SIZE;

        // Clear counters
        queue.write_buffer(&self.visible_count_buffer, 0, &[0u8; 4]);
        queue.write_buffer(&self.key_count_buffer, 0, &[0u8; 4]);

        // Clear tile ranges
        let zero_ranges = vec![0u8; self.num_tiles as usize * TileRange::SIZE];
        queue.write_buffer(&self.tile_ranges_buffer, 0, &zero_ranges);

        // === Pass 1: Project & Cull ===
        let project_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Project Bind Group"),
            layout: &self.project_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: splat_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.splats_2d_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: camera_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.visible_count_buffer.as_entire_binding(),
                },
            ],
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Project Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.project_pipeline);
            pass.set_bind_group(0, &project_bind_group, &[]);
            pass.dispatch_workgroups((num_splats + 255) / 256, 1, 1);
        }

        // === Pass 2: Tile Assignment ===
        let tile_assign_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Tile Assign Bind Group"),
            layout: &self.tile_assign_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.splats_2d_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.tile_keys_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.key_count_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: tile_params_buffer.as_entire_binding(),
                },
            ],
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Tile Assign Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.tile_assign_pipeline);
            pass.set_bind_group(0, &tile_assign_bind_group, &[]);
            pass.dispatch_workgroups((num_splats + 255) / 256, 1, 1);
        }

        // === Pass 3: Bitonic Sort ===
        // For simplicity, we'll sort up to a fixed number of elements
        // In production, you'd read back key_count and dispatch accordingly
        let max_keys = num_splats * 4; // Assume average 4 tiles per splat
        let sort_stages = (max_keys as f32).log2().ceil() as u32;

        for stage in 0..sort_stages {
            for substage in (0..=stage).rev() {
                let sort_params = SortParams {
                    num_elements: max_keys,
                    stage,
                    substage,
                    _pad: 0,
                };
                let sort_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Sort Params"),
                    contents: bytemuck::bytes_of(&sort_params),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

                let sort_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Sort Bind Group"),
                    layout: &self.sort_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: self.tile_keys_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: sort_params_buffer.as_entire_binding(),
                        },
                    ],
                });

                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("Sort Pass"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&self.sort_pipeline);
                    pass.set_bind_group(0, &sort_bind_group, &[]);
                    pass.dispatch_workgroups((max_keys + 255) / 256, 1, 1);
                }
            }
        }

        // === Pass 4: Compute Tile Ranges ===
        let tile_range_params = TileRangeParams {
            num_keys: max_keys,
            num_tiles: self.num_tiles,
            _pad0: 0,
            _pad1: 0,
        };
        let tile_range_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Tile Range Params"),
            contents: bytemuck::bytes_of(&tile_range_params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let tile_ranges_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Tile Ranges Bind Group"),
            layout: &self.tile_ranges_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.tile_keys_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.tile_ranges_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: tile_range_params_buffer.as_entire_binding(),
                },
            ],
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Tile Ranges Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.tile_ranges_pipeline);
            pass.set_bind_group(0, &tile_ranges_bind_group, &[]);
            pass.dispatch_workgroups((max_keys + 255) / 256, 1, 1);
        }

        // === Pass 5: Rasterize Tiles ===
        let raster_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Raster Bind Group"),
            layout: &self.raster_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.splats_2d_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.tile_keys_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.tile_ranges_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&self.output_view),
                },
            ],
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Raster Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.raster_pipeline);
            pass.set_bind_group(0, &raster_bind_group, &[]);
            pass.dispatch_workgroups(num_tiles_x, num_tiles_y, 1);
        }
    }

    /// Get dimensions
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Blit the output texture to the surface
    pub fn blit_to_surface(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        surface_view: &wgpu::TextureView,
    ) {
        let blit_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Blit Bind Group"),
            layout: &self.blit_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.output_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.blit_sampler),
                },
            ],
        });

        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Blit Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: surface_view,
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

        render_pass.set_pipeline(&self.blit_pipeline);
        render_pass.set_bind_group(0, &blit_bind_group, &[]);
        render_pass.draw(0..3, 0..1); // Fullscreen triangle
    }
}

use wgpu::util::DeviceExt;
