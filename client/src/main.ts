import { UniverseClient } from './client';
import { InputHandler } from './input';
import { HUD } from './hud';

async function main() {
    console.log('%c🚀 Universe Client Starting...', 'color: #0af; font-size: 16px; font-weight: bold');

    const video = document.getElementById('video') as HTMLCanvasElement;
    const videoContainer = document.getElementById('video-container') as HTMLDivElement;
    const loading = document.getElementById('loading') as HTMLDivElement;
    const clickPrompt = document.getElementById('click-prompt') as HTMLDivElement;
    const status = document.getElementById('status') as HTMLDivElement;
    const buffering = document.getElementById('buffering') as HTMLDivElement;

    // Determine WebSocket base URL
    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsHost = window.location.host;
    const baseUrl = `${wsProtocol}//${wsHost}`;

    console.log('Connecting to:', baseUrl);

    // Registration scaffold: URL param (?registered=1) or localStorage toggle
    const params = new URLSearchParams(window.location.search);
    const registered =
        params.get('registered') === '1' ||
        params.get('registered')?.toLowerCase() === 'true' ||
        window.localStorage.getItem('universe_registered') === '1';

    const supportsWebCodecs = typeof (window as any).VideoDecoder !== 'undefined';

    // Create client
    const client = new UniverseClient(baseUrl, { registered, preferH264: supportsWebCodecs });

    // Create HUD
    const hud = new HUD();

    // Create input handler
    const input = new InputHandler(client);

    // Canvas draw helpers (\"contain\" fit)
    const ctx = video.getContext('2d', { alpha: false })!;
    const dpr = window.devicePixelRatio || 1;

    function resizeCanvas() {
        const rect = videoContainer.getBoundingClientRect();
        const w = Math.max(1, Math.floor(rect.width * dpr));
        const h = Math.max(1, Math.floor(rect.height * dpr));
        if (video.width !== w || video.height !== h) {
            video.width = w;
            video.height = h;
        }
    }

    function drawContain(source: CanvasImageSource, srcW: number, srcH: number) {
        resizeCanvas();
        const cw = video.width;
        const ch = video.height;
        const scale = Math.min(cw / srcW, ch / srcH);
        const dw = Math.floor(srcW * scale);
        const dh = Math.floor(srcH * scale);
        const dx = Math.floor((cw - dw) / 2);
        const dy = Math.floor((ch - dh) / 2);
        ctx.clearRect(0, 0, cw, ch);
        ctx.drawImage(source, dx, dy, dw, dh);
    }

    window.addEventListener('resize', resizeCanvas);
    resizeCanvas();

    // Connection status updates
    client.onStatusChange = (newStatus: string) => {
        status.textContent = `● ${newStatus}`;
        status.className = newStatus.toLowerCase().replace(/\s+/g, '');

        if (newStatus === 'CONNECTED') {
            loading.classList.add('hidden');
            clickPrompt.classList.remove('hidden');
        } else if (newStatus === 'DISCONNECTED') {
            loading.classList.remove('hidden');
            loading.innerHTML = `
                <div class="spinner"></div>
                <div style="color: #f00">Connection Lost</div>
                <div style="font-size: 14px; margin-top: 10px; opacity: 0.7">
                    Attempting to reconnect...
                </div>
            `;
        }
    };

    // State updates from server
    client.onStateUpdate = (state) => {
        hud.update(state);
    };

    // Jump budget updates
    client.onJumpStatusUpdate = (remaining, max, isRegistered) => {
        hud.updateJumpStatus(remaining, max, isRegistered);
    };

    // Buffering overlay (teleport UX)
    client.onBufferingChange = (active: boolean) => {
        if (active) buffering.classList.remove('hidden');
        else buffering.classList.add('hidden');
    };

    // Latency updates
    client.onLatencyUpdate = (latency: number) => {
        hud.updateLatency(latency);
    };

    // Frame received (MJPEG)
    client.onFrameReceived = async (blob: Blob) => {
        // MJPEG fallback: decode JPEG and draw to canvas
        try {
            const bmp = await createImageBitmap(blob);
            drawContain(bmp, bmp.width, bmp.height);
            bmp.close();
        } catch (e) {
            // ignore
        }
    };

    // H.264 (WebCodecs) path
    let decoder: VideoDecoder | null = null;
    let decoderReady = false;

    function annexbToAvcc(data: Uint8Array): Uint8Array {
        // Convert Annex-B start-code format to AVCC length-prefixed NALs.
        const nals: Uint8Array[] = [];
        const len = data.length;

        const find = (from: number): { pos: number; scLen: number } | null => {
            for (let i = from; i + 3 < len; i++) {
                if (data[i] === 0 && data[i + 1] === 0) {
                    if (data[i + 2] === 1) return { pos: i, scLen: 3 };
                    if (i + 4 < len && data[i + 2] === 0 && data[i + 3] === 1) return { pos: i, scLen: 4 };
                }
            }
            return null;
        };

        let cur = find(0);
        while (cur) {
            const next = find(cur.pos + cur.scLen);
            const start = cur.pos + cur.scLen;
            const end = next ? next.pos : len;
            if (end > start) {
                nals.push(data.subarray(start, end));
            }
            cur = next;
        }

        let total = 0;
        for (const nal of nals) total += 4 + nal.length;
        const out = new Uint8Array(total);
        const dv = new DataView(out.buffer);
        let off = 0;
        for (const nal of nals) {
            dv.setUint32(off, nal.length, false); // big-endian
            off += 4;
            out.set(nal, off);
            off += nal.length;
        }
        return out;
    }

    client.onVideoConfig = (codec: string, avcc: Uint8Array) => {
        if (!supportsWebCodecs) return;

        decoderReady = false;
        decoder?.close();

        decoder = new VideoDecoder({
            output: (frame: VideoFrame) => {
                try {
                    const w = frame.displayWidth || frame.codedWidth;
                    const h = frame.displayHeight || frame.codedHeight;
                    drawContain(frame, w, h);
                } finally {
                    frame.close();
                }
            },
            error: (e) => {
                console.error('VideoDecoder error:', e);
            },
        });

        decoder.configure({
            codec,
            description: avcc,
            optimizeForLatency: true,
            hardwareAcceleration: 'prefer-hardware',
        } as VideoDecoderConfig);

        decoderReady = true;
    };

    client.onH264Frame = (annexb: Uint8Array, timestampUs: number, isKey: boolean) => {
        if (!decoder || !decoderReady) return;
        try {
            const avcc = annexbToAvcc(annexb);
            decoder.decode(new EncodedVideoChunk({
                type: isKey ? 'key' : 'delta',
                timestamp: timestampUs,
                data: avcc,
            }));
        } catch (e) {
            // ignore decode errors; decoder may be reconfigured on next keyframe/config
        }
    };

    // Click to start (for pointer lock)
    clickPrompt.addEventListener('click', () => {
        clickPrompt.classList.add('hidden');
        input.requestPointerLock();
    });

    // Also allow clicking video
    video.addEventListener('click', () => {
        if (!input.isLocked()) {
            clickPrompt.classList.remove('hidden');
        }
    });

    // Connect
    try {
        await client.connect();
        console.log('%c✓ Connected to Universe server', 'color: #0f0');
    } catch (error) {
        console.error('Connection failed:', error);
        status.textContent = '● CONNECTION FAILED';
        status.className = 'disconnected';
        loading.innerHTML = `
            <div style="color: #f00">Connection Failed</div>
            <div style="font-size: 14px; margin-top: 10px; opacity: 0.7">
                Please ensure the Universe server is running on port 7878
            </div>
            <div style="font-size: 12px; margin-top: 20px; opacity: 0.5">
                Server command: cargo run -p universe-cli -- serve
            </div>
        `;
    }
}

// Start application
main().catch(console.error);
