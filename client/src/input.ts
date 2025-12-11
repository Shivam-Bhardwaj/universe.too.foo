import { UniverseClient } from './client';

const AU = 1.496e11;

export class InputHandler {
    private keysPressed: Set<string> = new Set();
    private pointerLocked = false;

    constructor(private client: UniverseClient) {
        this.setupEventListeners();
    }

    private setupEventListeners(): void {
        // Keyboard
        document.addEventListener('keydown', (e) => this.onKeyDown(e));
        document.addEventListener('keyup', (e) => this.onKeyUp(e));

        // Mouse movement
        document.addEventListener('mousemove', (e) => this.onMouseMove(e));

        // Pointer lock change
        document.addEventListener('pointerlockchange', () => this.onPointerLockChange());
        document.addEventListener('pointerlockerror', () => this.onPointerLockError());
    }

    requestPointerLock(): void {
        document.body.requestPointerLock();
    }

    isLocked(): boolean {
        return this.pointerLocked;
    }

    private onPointerLockChange(): void {
        this.pointerLocked = document.pointerLockElement === document.body;

        const crosshair = document.getElementById('crosshair');
        if (crosshair) {
            crosshair.className = this.pointerLocked ? 'visible' : '';
        }
    }

    private onPointerLockError(): void {
        console.error('Pointer lock error');
    }

    private onMouseMove(e: MouseEvent): void {
        if (!this.pointerLocked) return;

        // Send mouse delta with sensitivity scaling
        const sensitivity = 0.002;
        this.client.sendMouseMove(e.movementX * sensitivity, e.movementY * sensitivity);
    }

    private onKeyDown(e: KeyboardEvent): void {
        // Prevent default for game keys
        const gameKeys = ['Space', 'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight',
                         'KeyW', 'KeyA', 'KeyS', 'KeyD', 'KeyQ', 'KeyE', 'KeyP'];
        if (gameKeys.includes(e.code)) {
            e.preventDefault();
        }

        // Escape releases pointer lock
        if (e.code === 'Escape') {
            if (this.pointerLocked) {
                document.exitPointerLock();
            }
            return;
        }

        // Don't repeat keys
        if (this.keysPressed.has(e.code)) return;
        this.keysPressed.add(e.code);

        this.client.sendKey(e.code, true);

        // Handle special keys
        this.handleSpecialKey(e.code);
    }

    private onKeyUp(e: KeyboardEvent): void {
        this.keysPressed.delete(e.code);
        this.client.sendKey(e.code, false);
    }

    private handleSpecialKey(code: string): void {
        switch (code) {
            case 'Digit0':
                // Solar system overview
                this.client.teleport(0, 5e11, 0);
                this.client.lookAt(0, 0, 0);
                break;

            case 'Digit1':
                // Earth (approximate position - server has real positions)
                this.client.teleport(AU, 0, 1e8);
                this.client.lookAt(AU, 0, 0);
                break;

            case 'Digit2':
                // Mars
                this.client.teleport(1.52 * AU, 0, 1e8);
                this.client.lookAt(1.52 * AU, 0, 0);
                break;

            case 'Digit3':
                // Jupiter
                this.client.teleport(5.2 * AU, 0, 1e9);
                this.client.lookAt(5.2 * AU, 0, 0);
                break;

            case 'Digit4':
                // Saturn
                this.client.teleport(9.5 * AU, 0, 1e9);
                this.client.lookAt(9.5 * AU, 0, 0);
                break;

            case 'Digit5':
                // Uranus
                this.client.teleport(19.2 * AU, 0, 2e9);
                this.client.lookAt(19.2 * AU, 0, 0);
                break;

            case 'Digit6':
                // Neptune
                this.client.teleport(30 * AU, 0, 2e9);
                this.client.lookAt(30 * AU, 0, 0);
                break;
        }
    }
}
