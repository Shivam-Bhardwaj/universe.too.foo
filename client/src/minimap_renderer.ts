import minimapComputeSource from './shaders/minimap_compute.wgsl?raw';
import minimapRenderSource from './shaders/minimap_render.wgsl?raw';

export class MinimapRenderer {
    private device: GPUDevice;
    private context: GPUCanvasContext;

    private densityBuffer: GPUBuffer;
    private computeParamsBuffer: GPUBuffer;
    private renderParamsBuffer: GPUBuffer;
    private splatBuffer: GPUBuffer | null = null;
    
    private computePipeline: GPUComputePipeline;
    private renderPipeline: GPURenderPipeline;
    
    private bindGroupCompute: GPUBindGroup | null = null;
    private bindGroupRender: GPUBindGroup | null = null;

    private gridSize = 128; // 128x128 grid

    constructor(device: GPUDevice, canvas: HTMLCanvasElement) {
        this.device = device;
        this.context = canvas.getContext('webgpu') as GPUCanvasContext;

        const presentationFormat = navigator.gpu.getPreferredCanvasFormat();
        this.context.configure({
            device,
            format: presentationFormat,
            alphaMode: 'premultiplied',
        });

        // Create pipelines
        const computeModule = device.createShaderModule({
            label: 'Minimap Compute',
            code: minimapComputeSource,
        });

        const renderModule = device.createShaderModule({
            label: 'Minimap Render',
            code: minimapRenderSource,
        });

        this.computePipeline = device.createComputePipeline({
            layout: 'auto',
            compute: {
                module: computeModule,
                entryPoint: 'main',
            },
        });

        this.renderPipeline = device.createRenderPipeline({
            layout: 'auto',
            vertex: {
                module: renderModule,
                entryPoint: 'vs_main',
            },
            fragment: {
                module: renderModule,
                entryPoint: 'fs_main',
                targets: [{ format: presentationFormat }],
            },
            primitive: {
                topology: 'triangle-list',
            },
        });

        // Initialize density buffer
        this.densityBuffer = device.createBuffer({
            size: this.gridSize * this.gridSize * 4, // u32
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        // Initialize params buffers
        this.computeParamsBuffer = device.createBuffer({
            size: 32, // 1 u32 + 4 f32 + pad
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this.renderParamsBuffer = device.createBuffer({
            size: 16, // 1 u32 + 1 f32 + pad
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
    }

    async update(splats: Float32Array, bounds: { minX: number, minZ: number, sizeX: number, sizeZ: number }, offset: { x: number, z: number }) {
        // Resize splat buffer if needed
        const byteLength = splats.byteLength;
        if (!this.splatBuffer || this.splatBuffer.size < byteLength) {
            this.splatBuffer?.destroy();
            this.splatBuffer = this.device.createBuffer({
                size: Math.max(byteLength, 1024 * 1024), // Min 1MB
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            });
            // Invalidate bind group
            this.bindGroupCompute = null;
        }

        // Upload splats
        // TS types for WebGPU are picky about ArrayBuffer vs ArrayBufferLike; provide a byte view.
        const splatBytes = new Uint8Array(splats.buffer as ArrayBuffer, splats.byteOffset, splats.byteLength);
        this.device.queue.writeBuffer(this.splatBuffer, 0, splatBytes);

        // Upload compute params
        const paramsData = new ArrayBuffer(32);
        const view = new DataView(paramsData);
        view.setUint32(0, this.gridSize, true);
        view.setFloat32(4, bounds.minX, true);
        view.setFloat32(8, bounds.minZ, true);
        view.setFloat32(12, bounds.sizeX, true);
        view.setFloat32(16, bounds.sizeZ, true);
        view.setFloat32(20, offset.x, true);
        view.setFloat32(24, offset.z, true);
        
        this.device.queue.writeBuffer(this.computeParamsBuffer, 0, paramsData);

        // Clear density grid
        this.device.queue.writeBuffer(this.densityBuffer, 0, new Uint32Array(this.gridSize * this.gridSize));

        // Create compute bind group if needed
        if (!this.bindGroupCompute && this.splatBuffer) {
            this.bindGroupCompute = this.device.createBindGroup({
                layout: this.computePipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: this.splatBuffer } },
                    { binding: 1, resource: { buffer: this.densityBuffer } },
                    { binding: 2, resource: { buffer: this.computeParamsBuffer } },
                ],
            });
        }

        // Run compute
        const commandEncoder = this.device.createCommandEncoder();
        const pass = commandEncoder.beginComputePass();
        pass.setPipeline(this.computePipeline);
        if (this.bindGroupCompute) {
            pass.setBindGroup(0, this.bindGroupCompute);
            // splats is a float buffer; each instance is 16 floats (see minimap_compute.wgsl).
            const workgroups = Math.ceil((splats.length / 16) / 64);
            pass.dispatchWorkgroups(workgroups);
        }
        pass.end();
        this.device.queue.submit([commandEncoder.finish()]);
    }

    render() {
        // Update render params (e.g. max density for normalization - could be computed or estimated)
        const renderParamsData = new ArrayBuffer(16);
        const view = new DataView(renderParamsData);
        view.setUint32(0, this.gridSize, true);
        view.setFloat32(4, 100.0, true); // Hardcoded max density for now
        this.device.queue.writeBuffer(this.renderParamsBuffer, 0, renderParamsData);

        if (!this.bindGroupRender) {
             this.bindGroupRender = this.device.createBindGroup({
                layout: this.renderPipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: this.densityBuffer } },
                    { binding: 1, resource: { buffer: this.renderParamsBuffer } },
                ],
            });
        }

        const commandEncoder = this.device.createCommandEncoder();
        const textureView = this.context.getCurrentTexture().createView();

        const pass = commandEncoder.beginRenderPass({
            colorAttachments: [{
                view: textureView,
                clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 0.0 },
                loadOp: 'clear',
                storeOp: 'store',
            }],
        });

        pass.setPipeline(this.renderPipeline);
        if (this.bindGroupRender) {
            pass.setBindGroup(0, this.bindGroupRender);
            pass.draw(3); // Fullscreen triangle
        }
        pass.end();

        this.device.queue.submit([commandEncoder.finish()]);
    }
}

