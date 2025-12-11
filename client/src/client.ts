export interface ServerState {
    type: 'State';
    epoch_jd: number;
    time_rate: number;
    camera_x: number;
    camera_y: number;
    camera_z: number;
    fps: number;
    clients: number;
}

export interface InputEvent {
    type: string;
    [key: string]: any;
}

export class UniverseClient {
    private streamWs: WebSocket | null = null;
    private controlWs: WebSocket | null = null;
    private pingInterval: number | null = null;
    private bufferingTimer: number | null = null;

    public onStatusChange: ((status: string) => void) | null = null;
    public onStateUpdate: ((state: ServerState) => void) | null = null;
    public onLatencyUpdate: ((latency: number) => void) | null = null;
    public onFrameReceived: ((blob: Blob) => void) | null = null;
    public onJumpStatusUpdate: ((remaining: number, max: number, registered: boolean) => void) | null = null;
    public onBufferingChange: ((active: boolean) => void) | null = null;
    public onVideoConfig: ((codec: string, avcc: Uint8Array) => void) | null = null;
    public onH264Frame: ((annexb: Uint8Array, timestampUs: number, isKey: boolean) => void) | null = null;

    constructor(private wsBaseUrl: string, private opts?: { registered?: boolean; preferH264?: boolean }) {
        // Normalize (avoid trailing slash)
        this.wsBaseUrl = this.wsBaseUrl.replace(/\/+$/, '');
    }

    async connect(): Promise<void> {
        this.onStatusChange?.('CONNECTING');

        // WebSocket URLs
        const streamUrl = this.opts?.preferH264
            ? `${this.wsBaseUrl}/stream?codec=h264`
            : `${this.wsBaseUrl}/stream`;
        const controlUrl = `${this.wsBaseUrl}/control${this.opts?.registered ? '?registered=1' : ''}`;

        // Connect to video stream
        await this.connectStream(streamUrl);

        // Connect to control channel
        await this.connectControl(controlUrl);

        this.onStatusChange?.('CONNECTED');

        // Start ping loop
        this.startPingLoop();
    }

    private connectStream(url: string): Promise<void> {
        return new Promise((resolve, reject) => {
            this.streamWs = new WebSocket(url);
            this.streamWs.binaryType = this.opts?.preferH264 ? 'arraybuffer' : 'blob';

            this.streamWs.onopen = () => {
                console.log('Stream connected');
                resolve();
            };

            this.streamWs.onmessage = (event) => {
                if (typeof event.data === 'string') {
                    // H.264 config message
                    try {
                        const msg = JSON.parse(event.data);
                        if (msg.type === 'VideoConfig' && typeof msg.codec === 'string' && typeof msg.avcc === 'string') {
                            const raw = atob(msg.avcc);
                            const bytes = new Uint8Array(raw.length);
                            for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
                            this.onVideoConfig?.(msg.codec, bytes);
                        }
                    } catch (e) {
                        // Ignore non-JSON messages
                    }
                    return;
                }

                if (event.data instanceof ArrayBuffer) {
                    const dv = new DataView(event.data);
                    if (dv.byteLength < 9) return;
                    const flags = dv.getUint8(0);
                    const ts = Number(dv.getBigUint64(1, true));
                    const annexb = new Uint8Array(event.data, 9);
                    this.onH264Frame?.(annexb, ts, (flags & 1) !== 0);
                    return;
                }

                if (event.data instanceof Blob) {
                    this.onFrameReceived?.(event.data);
                }
            };

            this.streamWs.onerror = (error) => {
                console.error('Stream error:', error);
                reject(error);
            };

            this.streamWs.onclose = () => {
                console.log('Stream closed');
                this.onStatusChange?.('DISCONNECTED');
            };
        });
    }

    private connectControl(url: string): Promise<void> {
        return new Promise((resolve, reject) => {
            this.controlWs = new WebSocket(url);

            this.controlWs.onopen = () => {
                console.log('Control connected');
                resolve();
            };

            this.controlWs.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    this.handleServerMessage(data);
                } catch (error) {
                    console.error('Invalid message:', error);
                }
            };

            this.controlWs.onerror = (error) => {
                console.error('Control error:', error);
                reject(error);
            };

            this.controlWs.onclose = () => {
                console.log('Control closed');
            };
        });
    }

    private handleServerMessage(message: any): void {
        switch (message.type) {
            case 'Pong':
                const now = Date.now();
                const latency = now - message.client_time;
                this.onLatencyUpdate?.(latency);
                break;

            case 'State':
                this.onStateUpdate?.(message as ServerState);
                break;

            case 'JumpStatus':
                this.onJumpStatusUpdate?.(message.remaining, message.max, !!message.registered);
                break;

            case 'Error':
                console.error('Server error:', message.message);
                // Stop any buffering overlay on errors (e.g., jump rejected)
                this.onBufferingChange?.(false);
                break;
        }
    }

    private startPingLoop(): void {
        this.pingInterval = window.setInterval(() => {
            this.sendInput({
                type: 'Ping',
                client_time: Date.now(),
            });
        }, 1000);
    }

    sendInput(event: InputEvent): void {
        if (this.controlWs?.readyState === WebSocket.OPEN) {
            this.controlWs.send(JSON.stringify(event));
        }
    }

    sendMouseMove(dx: number, dy: number): void {
        this.sendInput({ type: 'MouseMove', dx, dy });
    }

    sendKey(code: string, pressed: boolean): void {
        this.sendInput({ type: 'Key', code, pressed });
    }

    setTime(jd: number): void {
        this.sendInput({ type: 'SetTime', jd });
    }

    setTimeRate(rate: number): void {
        this.sendInput({ type: 'SetTimeRate', rate });
    }

    teleport(x: number, y: number, z: number): void {
        // Netflix-like \"buffer\" moment on teleports (client-side UX)
        this.onBufferingChange?.(true);
        if (this.bufferingTimer) {
            clearTimeout(this.bufferingTimer);
        }
        this.bufferingTimer = window.setTimeout(() => {
            this.onBufferingChange?.(false);
            this.bufferingTimer = null;
        }, 700);

        this.sendInput({ type: 'Teleport', x, y, z });
    }

    lookAt(x: number, y: number, z: number): void {
        this.sendInput({ type: 'LookAt', x, y, z });
    }

    disconnect(): void {
        if (this.pingInterval) {
            clearInterval(this.pingInterval);
            this.pingInterval = null;
        }
        if (this.bufferingTimer) {
            clearTimeout(this.bufferingTimer);
            this.bufferingTimer = null;
        }

        this.streamWs?.close();
        this.controlWs?.close();

        this.streamWs = null;
        this.controlWs = null;
    }
}
