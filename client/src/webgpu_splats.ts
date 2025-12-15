import splatWgsl from './shaders/splat.wgsl?raw';

export class WebGpuSplatRenderer {
    private device!: GPUDevice;
    private context!: GPUCanvasContext;
    private format!: GPUTextureFormat;

    private pipeline!: GPURenderPipeline;
    private bindGroupLayout!: GPUBindGroupLayout;

    private cameraBuffer!: GPUBuffer;
    private splatBuffer!: GPUBuffer;
    private bindGroup!: GPUBindGroup;

    private depthTexture!: GPUTexture;
    private depthView!: GPUTextureView;

    private width = 1;
    private height = 1;
    private splatCount = 0;

    static async create(canvas: HTMLCanvasElement): Promise<WebGpuSplatRenderer> {
        const r = new WebGpuSplatRenderer();
        await r.init(canvas);
        return r;
    }

    get instanceCount(): number {
        return this.splatCount;
    }

    private async init(canvas: HTMLCanvasElement) {
        if (!('gpu' in navigator)) {
            throw new Error('WebGPU not supported in this browser');
        }

        const adapter = await (navigator as any).gpu.requestAdapter();
        if (!adapter) {
            throw new Error('No WebGPU adapter available');
        }

        this.device = await adapter.requestDevice();

        const ctx = canvas.getContext('webgpu');
        if (!ctx) {
            throw new Error('Failed to acquire webgpu context');
        }
        this.context = ctx;
        this.format = (navigator as any).gpu.getPreferredCanvasFormat();

        // Initial size (caller can resize later)
        this.resize(canvas.width || 1, canvas.height || 1);

        // Shader + pipeline
        const shader = this.device.createShaderModule({ code: splatWgsl });

        this.bindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
                    buffer: { type: 'uniform' },
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.VERTEX,
                    buffer: { type: 'read-only-storage' },
                },
            ],
        });

        const pipelineLayout = this.device.createPipelineLayout({
            bindGroupLayouts: [this.bindGroupLayout],
        });

        this.pipeline = this.device.createRenderPipeline({
            layout: pipelineLayout,
            vertex: {
                module: shader,
                entryPoint: 'vs_main',
            },
            fragment: {
                module: shader,
                entryPoint: 'fs_main',
                targets: [
                    {
                        format: this.format,
                        blend: {
                            color: {
                                srcFactor: 'one',
                                dstFactor: 'one-minus-src-alpha',
                                operation: 'add',
                            },
                            alpha: {
                                srcFactor: 'one',
                                dstFactor: 'one-minus-src-alpha',
                                operation: 'add',
                            },
                        },
                        writeMask: GPUColorWrite.ALL,
                    },
                ],
            },
            primitive: {
                topology: 'triangle-list',
                cullMode: 'none',
            },
            depthStencil: {
                format: 'depth32float',
                depthWriteEnabled: true,
                // Reverse-Z: greater is closer
                depthCompare: 'greater',
            },
        });

        // Buffers
        this.cameraBuffer = this.device.createBuffer({
            // 56 floats = 224 bytes (matches CameraUniform in Rust)
            size: 224,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this.splatBuffer = this.device.createBuffer({
            // Placeholder; replaced by setSplats once data is available
            size: 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        this.bindGroup = this.device.createBindGroup({
            layout: this.bindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.cameraBuffer } },
                { binding: 1, resource: { buffer: this.splatBuffer } },
            ],
        });
    }

    resize(width: number, height: number) {
        this.width = Math.max(1, Math.floor(width));
        this.height = Math.max(1, Math.floor(height));

        this.context.configure({
            device: this.device,
            format: this.format,
            alphaMode: 'opaque',
        });

        this.depthTexture?.destroy();
        this.depthTexture = this.device.createTexture({
            size: { width: this.width, height: this.height, depthOrArrayLayers: 1 },
            format: 'depth32float',
            usage: GPUTextureUsage.RENDER_ATTACHMENT,
        });
        this.depthView = this.depthTexture.createView();
    }

    setSplats(gpuSplats: Float32Array) {
        const bytes = gpuSplats.byteLength;
        const required = Math.max(4, bytes);
        const needsRecreate = !this.splatBuffer || this.splatBuffer.size < required;

        if (needsRecreate) {
            this.splatBuffer?.destroy();
            this.splatBuffer = this.device.createBuffer({
                size: required,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            });

            this.bindGroup = this.device.createBindGroup({
                layout: this.bindGroupLayout,
                entries: [
                    { binding: 0, resource: { buffer: this.cameraBuffer } },
                    { binding: 1, resource: { buffer: this.splatBuffer } },
                ],
            });
        }

        this.device.queue.writeBuffer(this.splatBuffer, 0, gpuSplats.buffer, gpuSplats.byteOffset, gpuSplats.byteLength);
        this.splatCount = Math.floor(gpuSplats.length / 16);
    }

    render(cameraUniform: Float32Array) {
        // cameraUniform must be 56 floats (224 bytes)
        this.device.queue.writeBuffer(this.cameraBuffer, 0, cameraUniform.buffer, cameraUniform.byteOffset, cameraUniform.byteLength);

        const colorView = this.context.getCurrentTexture().createView();
        const encoder = this.device.createCommandEncoder();

        const pass = encoder.beginRenderPass({
            colorAttachments: [
                {
                    view: colorView,
                    clearValue: { r: 0.0, g: 0.0, b: 0.02, a: 1.0 },
                    loadOp: 'clear',
                    storeOp: 'store',
                },
            ],
            depthStencilAttachment: {
                view: this.depthView,
                // Reverse-Z: clear to far (0.0)
                depthClearValue: 0.0,
                depthLoadOp: 'clear',
                depthStoreOp: 'store',
            },
        });

        if (this.splatCount > 0) {
            pass.setPipeline(this.pipeline);
            pass.setBindGroup(0, this.bindGroup);
            pass.draw(6, this.splatCount, 0, 0);
        }

        pass.end();
        this.device.queue.submit([encoder.finish()]);
    }
}



