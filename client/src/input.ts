import { UniverseClient } from './client';
import type { LocalCamera } from './camera';
import type { Vec3d } from './camera';
import { FlightControls, type ControlMode, type JumpState } from './flight_controls';

const AU = 1.496e11;

// Named navigation targets for digit keys
interface NavTarget {
    name: string;
    pos: Vec3d;
    radius_m?: number;
}

const NAV_TARGETS: Record<string, NavTarget> = {
    'Digit0': { name: 'Home', pos: { x: 0, y: 0, z: 1.5e11 } },
    // Approximate semi-major-axis positions (match dataset-mode overlay in main.ts).
    'Digit1': { name: 'Earth', pos: { x: AU, y: 0, z: 0 }, radius_m: 6.371e6 },
    'Digit2': { name: 'Mars', pos: { x: 1.52 * AU, y: 0, z: 0 }, radius_m: 3.3895e6 },
    'Digit3': { name: 'Jupiter', pos: { x: 5.2 * AU, y: 0, z: 0 }, radius_m: 6.9911e7 },
    'Digit4': { name: 'Saturn', pos: { x: 9.5 * AU, y: 0, z: 0 }, radius_m: 5.8232e7 },
    'Digit5': { name: 'Uranus', pos: { x: 19.2 * AU, y: 0, z: 0 }, radius_m: 2.5362e7 },
    'Digit6': { name: 'Neptune', pos: { x: 30 * AU, y: 0, z: 0 }, radius_m: 2.4622e7 },
};

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
            crosshair.classList.toggle('visible', this.pointerLocked);
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

/**
 * Local input handler for client-side WebGPU rendering mode.
 * Uses FlightControls for unified navigation (mouse-steer + scroll-throttle).
 */
export class LocalInputHandler {
    private keysPressed: Set<string> = new Set();
    private virtualKeysPressed: Set<string> = new Set();
    private flightControls: FlightControls;
    private lastKeyDebug: string = '';

    constructor(private camera: LocalCamera) {
        this.flightControls = new FlightControls(camera);
        this.setupEventListeners();
    }

    private setupEventListeners(): void {
        // Keyboard: attach to both document and window for reliability
        document.addEventListener('keydown', (e) => this.onKeyDown(e));
        document.addEventListener('keyup', (e) => this.onKeyUp(e));
        window.addEventListener('keydown', (e) => this.onKeyDown(e));
        window.addEventListener('keyup', (e) => this.onKeyUp(e));
        // NOTE: mouse/touch/wheel events are handled in `main.ts` because they
        // must be bound to the render canvas element.
    }

    getFlightControls(): FlightControls {
        return this.flightControls;
    }

    getMode(): ControlMode {
        return this.flightControls.getMode();
    }

    getLastKeyDebug(): string {
        return this.lastKeyDebug;
    }

    setVirtualKey(code: string, down: boolean): void {
        if (down) this.virtualKeysPressed.add(code);
        else this.virtualKeysPressed.delete(code);
    }

    clearVirtualKeys(): void {
        this.virtualKeysPressed.clear();
    }

    update(dt: number): void {
        // Update flight controls (handles steer + throttle)
        this.flightControls.update(dt);

        // Keyboard fallback: WASD for movement, Space/Shift for up/down
        const keySources: string[] = [];
        for (const k of this.keysPressed) keySources.push(k);
        for (const k of this.virtualKeysPressed) keySources.push(k);

        let forward = 0.0;
        let right = 0.0;
        let up = 0.0;

        for (const key of keySources) {
            switch (key) {
                case 'KeyW':
                    forward += 1.0;
                    break;
                case 'KeyS':
                    forward -= 1.0;
                    break;
                case 'KeyA':
                    right -= 1.0;
                    break;
                case 'KeyD':
                    right += 1.0;
                    break;
                case 'Space':
                    up += 1.0;
                    break;
                case 'ShiftLeft':
                case 'ShiftRight':
                    up -= 1.0;
                    break;
            }
        }

        // Apply keyboard movement if any
        if (forward !== 0 || right !== 0 || up !== 0) {
            this.camera.updateGain(dt, true);
            this.camera.translate(forward, right, up, dt);
        }
    }

    private onKeyDown(e: KeyboardEvent): void {
        const gameKeys = [
            'Space',
            'ArrowUp',
            'ArrowDown',
            'ArrowLeft',
            'ArrowRight',
            'KeyW',
            'KeyA',
            'KeyS',
            'KeyD',
            'KeyQ',
            'KeyE',
            'KeyR',
            'KeyF',
            'BracketLeft',
            'BracketRight',
            'KeyP',
            'Escape',
        ];
        if (gameKeys.includes(e.code)) {
            e.preventDefault();
        }

        // Debug: track last key pressed
        this.lastKeyDebug = e.code || e.key || 'unknown';

        if (e.code === 'Escape') {
            // Cancel jump if active
            if (this.flightControls.isJumping()) {
                this.flightControls.cancelJump();
                return;
            }
            // Exit orbit focus if active
            if (this.flightControls.getMode() === 'orbitFocus') {
                this.flightControls.exitOrbitFocus();
            }
            return;
        }

        if (this.keysPressed.has(e.code)) return;
        this.keysPressed.add(e.code);

        // Instant actions (pressed)
        this.handleSpecialKey(e.code);
    }

    private onKeyUp(e: KeyboardEvent): void {
        this.keysPressed.delete(e.code);
    }

    private handleSpecialKey(code: string): void {
        switch (code) {
            case 'KeyQ':
                this.camera.adjustTravelTime(10);
                break;
            case 'KeyE':
                this.camera.adjustTravelTime(-10);
                break;
            case 'KeyR':
                // Reset movement/throttle and cancel any jump
                this.camera.resetGain();
                this.flightControls.resetThrottle();
                this.flightControls.cancelJump();
                break;
            case 'BracketLeft':
                this.camera.adjustTravelTime(10);
                break;
            case 'BracketRight':
                this.camera.adjustTravelTime(-10);
                break;
            default:
                // Handle digit keys as jump targets
                if (code in NAV_TARGETS) {
                    const target = NAV_TARGETS[code];
                    this.flightControls.startJump(target.pos, target.name, target.radius_m);
                }
                break;
        }
    }

    getJumpState(): JumpState {
        return this.flightControls.getJumpState();
    }

    isJumping(): boolean {
        return this.flightControls.isJumping();
    }
}
