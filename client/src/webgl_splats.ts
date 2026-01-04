type UniformLocations = {
    view: WebGLUniformLocation;
    proj: WebGLUniformLocation;
    far: WebGLUniformLocation;
    logDepthC: WebGLUniformLocation;
};

export class WebGlSplatRenderer {
    private gl!: WebGL2RenderingContext;
    private program!: WebGLProgram;
    private vao!: WebGLVertexArrayObject;

    private instanceBuffer!: WebGLBuffer;
    private instanceBufferBytes = 0;
    private splatCount = 0;

    private u!: UniformLocations;

    private width = 1;
    private height = 1;

    static create(canvas: HTMLCanvasElement): WebGlSplatRenderer {
        const r = new WebGlSplatRenderer();
        r.init(canvas);
        return r;
    }

    resize(width: number, height: number) {
        this.width = Math.max(1, Math.floor(width));
        this.height = Math.max(1, Math.floor(height));
        this.gl.viewport(0, 0, this.width, this.height);
    }

    setSplats(gpuSplats: Float32Array) {
        const gl = this.gl;
        const bytes = gpuSplats.byteLength;
        const required = Math.max(4, bytes);

        gl.bindBuffer(gl.ARRAY_BUFFER, this.instanceBuffer);
        if (required !== this.instanceBufferBytes) {
            // (Re)allocate
            gl.bufferData(gl.ARRAY_BUFFER, gpuSplats, gl.DYNAMIC_DRAW);
            this.instanceBufferBytes = required;
        } else {
            gl.bufferSubData(gl.ARRAY_BUFFER, 0, gpuSplats);
        }
        gl.bindBuffer(gl.ARRAY_BUFFER, null);

        this.splatCount = Math.floor(gpuSplats.length / 16);
    }

    render(view: Float32Array, proj: Float32Array, far: number, logDepthC: number) {
        const gl = this.gl;

        gl.viewport(0, 0, this.width, this.height);

        gl.enable(gl.DEPTH_TEST);
        gl.depthFunc(gl.GREATER); // reverse-Z
        gl.depthMask(true);
        gl.clearDepth(0.0); // far

        gl.enable(gl.BLEND);
        gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA); // premultiplied alpha

        gl.disable(gl.CULL_FACE);

        gl.clearColor(0.0, 0.0, 0.02, 1.0);
        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

        gl.useProgram(this.program);
        gl.bindVertexArray(this.vao);

        gl.uniformMatrix4fv(this.u.view, false, view);
        gl.uniformMatrix4fv(this.u.proj, false, proj);
        gl.uniform1f(this.u.far, far);
        gl.uniform1f(this.u.logDepthC, logDepthC);

        if (this.splatCount > 0) {
            gl.drawArraysInstanced(gl.TRIANGLES, 0, 6, this.splatCount);
        }

        gl.bindVertexArray(null);
        gl.useProgram(null);
    }

    private init(canvas: HTMLCanvasElement) {
        const gl = canvas.getContext('webgl2', { alpha: false, depth: true });
        if (!gl) {
            throw new Error('WebGL2 not supported in this browser');
        }
        this.gl = gl;

        // Shaders (GLSL ES 3.00)
        const vs = `#version 300 es
precision highp float;

layout(location=0) in vec2 aQuadPos;

// Instance data (stride = 16 floats = 64 bytes)
layout(location=1) in vec3 aPos;
layout(location=2) in vec3 aScale;
layout(location=3) in vec4 aRot;
layout(location=4) in vec3 aColor;
layout(location=5) in float aOpacity;

uniform mat4 uView;
uniform mat4 uProj;
uniform float uFar;
uniform float uLogDepthC;

out vec2 vUv;
out vec3 vColor;
out float vOpacity;

void main() {
    vec4 view_pos = uView * vec4(aPos, 1.0);
    float max_scale = max(aScale.x, max(aScale.y, aScale.z));
    // Screen-space sizing (perspective, clamp angular size).
    // Star splats encode a *visual* radius that grows with distance; clamp keeps them sane.
    float dist = max(length(view_pos.xyz), 1.0);
    float angular_size = max_scale / dist;

    // Nearby objects (planets/spacecraft within ~100 AU): ensure visible and allow bigger cap.
    float nearby_threshold = 1.5e13;  // ~100 AU
    bool is_nearby = dist < nearby_threshold;

    // Min angular size for nearby objects (~5px), max angular size:
    // - stars capped at ~4px
    // - nearby objects capped at ~40px
    float min_angular = is_nearby ? 0.003 : 0.0;
    float max_angular = is_nearby ? 0.02 : 0.002;

    float effective_angular = clamp(angular_size, min_angular, max_angular);
    float screen_scale = effective_angular * dist;

    vec3 expanded_pos = view_pos.xyz + vec3(aQuadPos * screen_scale, 0.0);
    vec4 clip = uProj * vec4(expanded_pos, 1.0);

    // Log depth in [0,1] then convert to OpenGL NDC [-1,1]
    float z = max(1e-6, -expanded_pos.z);
    float denom = log2(uFar * uLogDepthC + 1.0);
    float log_depth = log2(z * uLogDepthC + 1.0) / denom;
    float depth01 = clamp(1.0 - log_depth, 0.0, 1.0);
    float ndcZ = depth01 * 2.0 - 1.0;
    clip.z = ndcZ * clip.w;

    gl_Position = clip;
    vUv = aQuadPos * 0.5 + 0.5;
    vColor = aColor;
    vOpacity = aOpacity;
}
`;

        const fs = `#version 300 es
precision highp float;

in vec2 vUv;
in vec3 vColor;
in float vOpacity;

out vec4 outColor;

void main() {
    vec2 d = vUv - vec2(0.5);
    float dist_sq = dot(d, d);
    if (dist_sq > 0.25) {
        discard;
    }
    // Sharper falloff + exposure boost for star visibility
    float gaussian = exp(-32.0 * dist_sq);
    float alpha = clamp(gaussian * vOpacity * 8.0, 0.0, 1.0);
    outColor = vec4(vColor * alpha, alpha);
}
`;

        const vert = this.compileShader(gl.VERTEX_SHADER, vs);
        const frag = this.compileShader(gl.FRAGMENT_SHADER, fs);
        this.program = this.linkProgram(vert, frag);

        // Uniforms
        const uView = gl.getUniformLocation(this.program, 'uView');
        const uProj = gl.getUniformLocation(this.program, 'uProj');
        const uFar = gl.getUniformLocation(this.program, 'uFar');
        const uLogDepthC = gl.getUniformLocation(this.program, 'uLogDepthC');
        if (!uView || !uProj || !uFar || !uLogDepthC) {
            throw new Error('Failed to resolve WebGL uniforms');
        }
        this.u = { view: uView, proj: uProj, far: uFar, logDepthC: uLogDepthC };

        // VAO
        const vao = gl.createVertexArray();
        if (!vao) throw new Error('Failed to create VAO');
        this.vao = vao;
        gl.bindVertexArray(this.vao);

        // Quad vertices
        const quad = new Float32Array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]);
        const quadBuf = gl.createBuffer();
        if (!quadBuf) throw new Error('Failed to create quad buffer');
        gl.bindBuffer(gl.ARRAY_BUFFER, quadBuf);
        gl.bufferData(gl.ARRAY_BUFFER, quad, gl.STATIC_DRAW);
        gl.enableVertexAttribArray(0);
        gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
        gl.vertexAttribDivisor(0, 0);

        // Instance buffer
        const instBuf = gl.createBuffer();
        if (!instBuf) throw new Error('Failed to create instance buffer');
        this.instanceBuffer = instBuf;
        gl.bindBuffer(gl.ARRAY_BUFFER, this.instanceBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, 4, gl.DYNAMIC_DRAW);
        this.instanceBufferBytes = 4;

        const stride = 16 * 4; // bytes
        // aPos @1: vec3 offset 0
        gl.enableVertexAttribArray(1);
        gl.vertexAttribPointer(1, 3, gl.FLOAT, false, stride, 0);
        gl.vertexAttribDivisor(1, 1);
        // aScale @2: vec3 offset 16 bytes (float4)
        gl.enableVertexAttribArray(2);
        gl.vertexAttribPointer(2, 3, gl.FLOAT, false, stride, 16);
        gl.vertexAttribDivisor(2, 1);
        // aRot @3: vec4 offset 32 bytes (float8)
        gl.enableVertexAttribArray(3);
        gl.vertexAttribPointer(3, 4, gl.FLOAT, false, stride, 32);
        gl.vertexAttribDivisor(3, 1);
        // aColor @4: vec3 offset 48 bytes (float12)
        gl.enableVertexAttribArray(4);
        gl.vertexAttribPointer(4, 3, gl.FLOAT, false, stride, 48);
        gl.vertexAttribDivisor(4, 1);
        // aOpacity @5: float offset 60 bytes (float15)
        gl.enableVertexAttribArray(5);
        gl.vertexAttribPointer(5, 1, gl.FLOAT, false, stride, 60);
        gl.vertexAttribDivisor(5, 1);

        gl.bindBuffer(gl.ARRAY_BUFFER, null);
        gl.bindVertexArray(null);

        // Initial size
        this.resize(canvas.width || 1, canvas.height || 1);
    }

    private compileShader(type: number, source: string): WebGLShader {
        const gl = this.gl;
        const s = gl.createShader(type);
        if (!s) throw new Error('Failed to create shader');
        gl.shaderSource(s, source);
        gl.compileShader(s);
        if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
            const info = gl.getShaderInfoLog(s) || 'unknown';
            gl.deleteShader(s);
            throw new Error(`Shader compile failed: ${info}`);
        }
        return s;
    }

    private linkProgram(vert: WebGLShader, frag: WebGLShader): WebGLProgram {
        const gl = this.gl;
        const p = gl.createProgram();
        if (!p) throw new Error('Failed to create program');
        gl.attachShader(p, vert);
        gl.attachShader(p, frag);
        gl.linkProgram(p);
        if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
            const info = gl.getProgramInfoLog(p) || 'unknown';
            gl.deleteProgram(p);
            throw new Error(`Program link failed: ${info}`);
        }
        gl.deleteShader(vert);
        gl.deleteShader(frag);
        return p;
    }
}



