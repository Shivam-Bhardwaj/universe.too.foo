import type { LocalCamera } from './camera';
import type { Vec3d } from './camera';

function clamp(v: number, lo: number, hi: number): number {
    return Math.max(lo, Math.min(hi, v));
}

function vec3Len(v: Vec3d): number {
    return Math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

function vec3Sub(a: Vec3d, b: Vec3d): Vec3d {
    return { x: a.x - b.x, y: a.y - b.y, z: a.z - b.z };
}

interface Spherical {
    r: number;
    theta: number;
    phi: number;
}

function toSpherical(v: Vec3d): Spherical {
    const r = Math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    if (r === 0) return { r: 0, theta: 0, phi: 0 };
    return {
        r,
        theta: Math.atan2(v.x, v.z),
        phi: Math.asin(v.y / r)
    };
}

function toCartesian(s: Spherical): Vec3d {
    const cosPhi = Math.cos(s.phi);
    const sinPhi = Math.sin(s.phi);
    const cosTheta = Math.cos(s.theta);
    const sinTheta = Math.sin(s.theta);
    return {
        x: s.r * cosPhi * sinTheta,
        y: s.r * sinPhi,
        z: s.r * cosPhi * cosTheta
    };
}

function lerpSpherical(a: Spherical, b: Spherical, t: number): Spherical {
    // Linear interpolation of r and phi
    const r = lerp(a.r, b.r, t);
    const phi = lerp(a.phi, b.phi, t);

    // Shortest path interpolation for theta
    let dTheta = b.theta - a.theta;
    while (dTheta > Math.PI) dTheta -= 2 * Math.PI;
    while (dTheta < -Math.PI) dTheta += 2 * Math.PI;

    const theta = a.theta + dTheta * t;

    return { r, theta, phi };
}

function lerp(a: number, b: number, t: number): number {
    return a + (b - a) * t;
}

export type ControlMode = 'flight' | 'orbitFocus';

export type JumpPhase = 'inactive' | 'aligning' | 'accelerating' | 'cruising' | 'easing';

export interface JumpState {
    phase: JumpPhase;
    targetPos: Vec3d;
    targetName: string;
    targetId: string;  // For anchor switching
    startPos: Vec3d;
    startDist: number;
    startTime: number;
    durationS: number;
    maxSpeedMps: number;
    /** Absolute (meter) distance from target center at which easing begins. */
    easeStartDistAbsM: number;
    /** Absolute (meter) distance from target center at which the jump stops. */
    stopDistAbsM: number;
    // Alignment state
    targetYaw: number;
    targetPitch: number;
    startYaw: number;
    startPitch: number;
}

export interface OrbitFocusState {
    active: boolean;
    targetPos: Vec3d;
    radius: number; // distance from target
    azimuth: number; // rotation around target (radians)
    polar: number; // elevation angle (radians, 0 = horizontal)
}

/**
 * FlightControls: unified navigation system
 * - Desktop: mouse-steer + scroll-throttle
 * - Mobile: 1-finger drag steer + pinch throttle
 */
export class FlightControls {
    private mode: ControlMode = 'flight';
    private orbitState: OrbitFocusState = {
        active: false,
        targetPos: { x: 0, y: 0, z: 0 },
        radius: 1e11,
        azimuth: 0,
        polar: 0,
    };

    // Jump/warp state
    private jumpState: JumpState = {
        phase: 'inactive',
        targetPos: { x: 0, y: 0, z: 0 },
        targetName: '',
        targetId: '',
        startPos: { x: 0, y: 0, z: 0 },
        startDist: 0,
        startTime: 0,
        durationS: 0,
        maxSpeedMps: 0,
        easeStartDistAbsM: 0,
        stopDistAbsM: 0,
        targetYaw: 0,
        targetPitch: 0,
        startYaw: 0,
        startPitch: 0,
    };

    // Jump completion callback (for anchor switching in main)
    private onJumpComplete: ((targetId: string, targetName: string, targetPos: Vec3d) => void) | null = null;

    // Saved camera settings (restored after jump)
    private savedTravelTime = 60.0;
    private savedAccelTime = 10.0;

    // Jump tuning
    private readonly JUMP_TRAVEL_TIME = 8.0;   // faster warp speed
    private readonly JUMP_ACCEL_TIME = 3.0;    // faster ramp-up
    private readonly ALIGN_DURATION_S = 0.3;   // time to rotate to target
    private readonly ACCEL_DURATION_S = 2.0;   // time to reach cruise
    private readonly STOP_FRAC_DEFAULT = 0.005;      // 0.5% of start distance (fallback)
    private readonly EASE_FRAC_DEFAULT = 0.25;       // start easing at 25% of start distance (earlier)
    private readonly VIEW_RADIUS_MULT = 30.0;        // stop at ~N radii (farther out)
    private readonly STOP_MIN_M = 1e7;               // never stop closer than 10,000 km

    // Throttle (signed, forward/back)
    private throttle = 0.0;
    private throttleTarget = 0.0;
    private throttleDamping = 0.92; // per-frame decay
    private throttleTargetDecay = 0.995; // gently returns to 0 without input

    // Pinch state (mobile)
    private lastPinchDistance = 0.0;
    private pinchActive = false;

    constructor(private camera: LocalCamera) { }

    getMode(): ControlMode {
        return this.mode;
    }

    getOrbitState(): OrbitFocusState {
        return { ...this.orbitState };
    }

    getJumpState(): JumpState {
        return { ...this.jumpState };
    }

    isJumping(): boolean {
        return this.jumpState.phase !== 'inactive';
    }

    /**
     * Set callback to be invoked when jump completes (for anchor switching).
     */
    setJumpCompleteCallback(callback: ((targetId: string, targetName: string, targetPos: Vec3d) => void) | null) {
        this.onJumpComplete = callback;
    }

    /**
     * Initiate a jump/warp toward a target.
     * The camera auto-orients to face the target, then accelerates forward.
     */
    startJump(targetPos: Vec3d, targetName: string = '', targetId: string = '', targetRadiusM?: number): void {
        // If already jumping, cancel first to restore saved settings before saving again
        if (this.isJumping()) {
            this.cancelJump();
        }

        // Exit orbit focus if active
        if (this.mode === 'orbitFocus') {
            this.exitOrbitFocus();
        }

        const toTarget = vec3Sub(targetPos, this.camera.position);
        const dist = vec3Len(toTarget);
        if (dist < 1e6) {
            // Already very close, no need to jump
            return;
        }

        // Compute a reasonable stop distance.
        // - For generic targets, stop at a small fraction of the start distance.
        // - If we know the target radius, prefer a radius-based viewing distance *when it is closer*
        //   (this prevents huge stop shells for diagonal jumps like start->Earth, while keeping Sun behavior stable).
        const stopByFrac = dist * this.STOP_FRAC_DEFAULT;
        const stopBySurface = targetRadiusM ? targetRadiusM * 1.05 : 0; // never stop inside the body
        const stopByRadius = targetRadiusM ? Math.max(stopBySurface, targetRadiusM * this.VIEW_RADIUS_MULT) : 0;
        const stopPreferred = targetRadiusM ? Math.min(stopByFrac, stopByRadius) : stopByFrac;
        const stopDistAbsM = Math.max(stopPreferred, stopBySurface, this.STOP_MIN_M);

        // If we are already at or inside the stop shell, just orient and do nothing.
        if (dist <= stopDistAbsM) {
            this.camera.lookAt(targetPos);
            return;
        }

        const easeStartDistAbsM = Math.max(dist * this.EASE_FRAC_DEFAULT, stopDistAbsM * 2.0);

        // Save current camera settings
        this.savedTravelTime = this.camera.travel_time_s;
        this.savedAccelTime = this.camera.accel_time_s;

        // Apply warp settings for faster travel
        this.camera.travel_time_s = this.JUMP_TRAVEL_TIME;
        this.camera.accel_time_s = this.JUMP_ACCEL_TIME;

        // Reset speed ramp so warp starts "very slow" even if user was already moving
        this.camera.resetSpeed();

        // Compute distance-aware jump duration (closer = slower, farther = faster but capped)
        const AU = 1.496e11;
        const distAU = dist / AU;
        const t = clamp(Math.log10(distAU + 1) / 6, 0, 1);
        const durationS = lerp(3.0, 10.0, t); // min 3s, max 10s
        const maxSpeedMps = dist / durationS;

        // Compute target angles
        const { yaw, pitch } = this.camera.computeLookAtAngles(targetPos);

        // Initialize jump state with alignment check
        let phase: JumpPhase = 'aligning';

        const startYaw = this.camera.getYaw();
        const startPitch = this.camera.getPitch();

        let dYaw = yaw - startYaw;
        // Normalize to [-PI, PI]
        while (dYaw > Math.PI) dYaw -= 2 * Math.PI;
        while (dYaw < -Math.PI) dYaw += 2 * Math.PI;

        const dPitch = Math.abs(pitch - startPitch);
        const ALIGN_EPS = 0.02; // ~1 degree

        if (Math.abs(dYaw) < ALIGN_EPS && dPitch < ALIGN_EPS) {
            phase = 'accelerating';
            console.log('[Jump] Skipping alignment (delta small)');
        }

        this.jumpState.phase = phase;
        this.jumpState.targetPos = { ...targetPos };
        this.jumpState.targetName = targetName;
        this.jumpState.targetId = targetId;
        this.jumpState.startPos = { ...this.camera.position };
        this.jumpState.startDist = dist;
        this.jumpState.startTime = performance.now() / 1000;
        this.jumpState.durationS = durationS;
        this.jumpState.maxSpeedMps = maxSpeedMps;
        this.jumpState.stopDistAbsM = stopDistAbsM;
        this.jumpState.easeStartDistAbsM = easeStartDistAbsM;

        this.jumpState.targetYaw = yaw;
        this.jumpState.targetPitch = pitch;
        this.jumpState.startYaw = startYaw;
        this.jumpState.startPitch = startPitch;

        // Force immediate lookAt if skipping alignment, to ensure perfect lock
        if (phase === 'accelerating') {
            this.camera.lookAt(targetPos);
        }

        // Set throttle to start ramping up
        this.throttle = 0.0;
        this.throttleTarget = 0.0;
    }

    /**
     * Cancel an active jump and restore normal flight controls.
     */
    cancelJump(): void {
        if (this.jumpState.phase === 'inactive') return;

        this.jumpState.phase = 'inactive';

        // Restore camera settings
        this.camera.travel_time_s = this.savedTravelTime;
        this.camera.accel_time_s = this.savedAccelTime;

        // Reset throttle and speed (but NOT travel_time_s)
        this.throttle = 0;
        this.throttleTarget = 0;
        this.camera.resetSpeed();
    }

    /**
     * Internal: complete the jump (called when we arrive).
     */
    private completeJump(): void {
        // Invoke callback before clearing state (so main can switch anchor)
        if (this.onJumpComplete && this.jumpState.targetId) {
            this.onJumpComplete(this.jumpState.targetId, this.jumpState.targetName, this.jumpState.targetPos);
        }

        this.jumpState.phase = 'inactive';

        // Restore camera settings
        this.camera.travel_time_s = this.savedTravelTime;
        this.camera.accel_time_s = this.savedAccelTime;

        // Stop movement (but NOT travel_time_s)
        this.throttle = 0;
        this.throttleTarget = 0;
        this.camera.resetSpeed();
    }

    /**
     * Enter orbit focus mode around a target position.
     */
    enterOrbitFocus(targetPos: Vec3d): void {
        this.mode = 'orbitFocus';
        this.orbitState.active = true;
        this.orbitState.targetPos = { ...targetPos };

        // Compute initial orbit parameters from current camera position
        const dx = this.camera.position.x - targetPos.x;
        const dy = this.camera.position.y - targetPos.y;
        const dz = this.camera.position.z - targetPos.z;
        const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
        this.orbitState.radius = Math.max(1e8, dist);

        // Compute azimuth and polar from direction
        this.orbitState.azimuth = Math.atan2(dx, -dz);
        this.orbitState.polar = Math.asin(dy / Math.max(1e-6, dist));
    }

    /**
     * Exit orbit focus and return to flight mode.
     */
    exitOrbitFocus(): void {
        this.mode = 'flight';
        this.orbitState.active = false;
    }

    /**
     * Update steer input from mouse movement (desktop).
     * Pass MouseEvent.movementX/Y in CSS pixels.
     */
    updateSteerFromMouse(dx: number, dy: number): void {
        if (this.mode !== 'flight') return;
        if (this.isJumping()) return; // Freeze steering during jump

        // Delta-based steering (no continuous rotation when mouse is stationary).
        this.camera.rotate(dx, dy);
    }

    /**
     * Update steer from drag delta (mobile).
     */
    updateSteerFromDrag(dx: number, dy: number): void {
        if (this.mode !== 'flight') return;
        if (this.isJumping()) return; // Freeze steering during jump
        // Mobile-style drag look: rotate directly by pixel delta.
        // (Desktop uses mouse-deflection steering via updateSteerFromMouse.)
        this.camera.rotate(dx, dy);
    }

    /**
     * Update throttle from scroll wheel (desktop).
     */
    updateThrottleFromWheel(deltaY: number): void {
        if (this.mode !== 'flight') return;

        // Scroll up = forward thrust, scroll down = reverse.
        const throttleDelta = -deltaY * 0.001; // scale factor
        this.throttleTarget = clamp(this.throttleTarget + throttleDelta, -1, 1);
    }

    /**
     * Update throttle from pinch gesture (mobile).
     */
    updateThrottleFromPinch(distance: number, initialDistance: number): void {
        if (this.mode !== 'flight') return;

        if (!this.pinchActive) {
            this.lastPinchDistance = initialDistance;
            this.pinchActive = true;
            return;
        }

        const delta = (distance - this.lastPinchDistance) / Math.max(1, initialDistance);
        const throttleDelta = delta * 0.25; // scale factor (gentler, less chaotic)
        this.throttleTarget = clamp(this.throttleTarget + throttleDelta, -1, 1);
        this.lastPinchDistance = distance;
    }

    /**
     * Reset pinch state (call when pinch ends).
     */
    resetPinch(): void {
        this.pinchActive = false;
        this.lastPinchDistance = 0;
    }

    /**
     * Update orbit focus from drag (rotate azimuth/polar).
     */
    updateOrbitFromDrag(dx: number, dy: number, canvasWidth: number, canvasHeight: number): void {
        if (this.mode !== 'orbitFocus') return;

        const azimuthDelta = (-dx / Math.max(1, canvasWidth)) * Math.PI;
        const polarDelta = (dy / Math.max(1, canvasHeight)) * Math.PI;

        this.orbitState.azimuth += azimuthDelta;
        this.orbitState.polar = clamp(this.orbitState.polar + polarDelta, -Math.PI * 0.45, Math.PI * 0.45);
    }

    /**
     * Update orbit radius from scroll/pinch (zoom).
     */
    updateOrbitRadius(delta: number, isPinch: boolean = false): void {
        if (this.mode !== 'orbitFocus') return;

        const factor = isPinch ? delta : -delta * 0.0001;
        const newRadius = this.orbitState.radius * (1 + factor);
        // Clamp orbit radius to anchor's local bubble
        const maxRadius = this.camera.maxLocalDistanceM;
        this.orbitState.radius = clamp(newRadius, 1e7, maxRadius);
    }

    /**
     * Update camera position and orientation each frame.
     */
    update(dt: number): void {
        // Handle jump state machine (runs in flight mode)
        if (this.jumpState.phase !== 'inactive') {
            this.updateJump(dt);
        }

        if (this.mode === 'flight') {
            const isJumping = this.jumpState.phase !== 'inactive';

            // During jump, throttle and movement are controlled by the jump system
            if (!isJumping) {
                // Smooth throttle and slowly decay target back to zero (reduces "runaway" feel).
                this.throttle = this.throttle * this.throttleDamping + this.throttleTarget * (1 - this.throttleDamping);
                this.throttleTarget *= this.throttleTargetDecay;

                // Apply throttle to movement (magnitude matters)
                const moving = Math.abs(this.throttle) > 1e-4;
                this.camera.updateGain(dt, moving);
                if (moving) {
                    this.camera.translate(this.throttle, 0, 0, dt);
                }
            }

            // NOTE: Steering is applied directly in input event handlers (mouse/touch deltas).
        } else if (this.mode === 'orbitFocus') {
            // Update camera position based on orbit state
            const cosPolar = Math.cos(this.orbitState.polar);
            const sinPolar = Math.sin(this.orbitState.polar);
            const cosAz = Math.cos(this.orbitState.azimuth);
            const sinAz = Math.sin(this.orbitState.azimuth);

            const offsetX = this.orbitState.radius * cosPolar * sinAz;
            const offsetY = this.orbitState.radius * sinPolar;
            const offsetZ = this.orbitState.radius * cosPolar * cosAz;

            this.camera.setPosition({
                x: this.orbitState.targetPos.x + offsetX,
                y: this.orbitState.targetPos.y + offsetY,
                z: this.orbitState.targetPos.z + offsetZ,
            });

            this.camera.lookAt(this.orbitState.targetPos);
        }
    }

    /**
     * Internal: update the jump state machine.
     * Moves directly toward target with locked orientation for monotonic distance decrease.
     */
    private updateJump(dt: number): void {
        const now = performance.now() / 1000;
        const elapsed = now - this.jumpState.startTime;

        // Lock orientation ONLY if not aligning (aligning handles its own rotation)
        if (this.jumpState.phase !== 'aligning') {
            this.camera.lookAt(this.jumpState.targetPos);
        }

        // Compute current distance to target
        const toTarget = vec3Sub(this.jumpState.targetPos, this.camera.position);
        const dist = vec3Len(toTarget);

        const stopDistAbs = this.jumpState.stopDistAbsM;
        const easeStartDistAbs = Math.max(stopDistAbs, this.jumpState.easeStartDistAbsM);
        const remainingToStop = dist - stopDistAbs;

        // If we've reached or passed the stop distance, complete immediately
        if (remainingToStop <= 0) {
            this.completeJump();
            return;
        }

        // Epsilon for "close enough" snap-to-stop (at least 1000 km)
        const eps = Math.max(1e6, stopDistAbs * 0.001);

        // If very close to stop shell, snap to it and complete
        if (remainingToStop <= eps && dist > 1e-6) {
            const dir = {
                x: toTarget.x / dist,
                y: toTarget.y / dist,
                z: toTarget.z / dist,
            };
            // Snap to exactly the stop shell
            this.camera.position.x = this.jumpState.targetPos.x - dir.x * stopDistAbs;
            this.camera.position.y = this.jumpState.targetPos.y - dir.y * stopDistAbs;
            this.camera.position.z = this.jumpState.targetPos.z - dir.z * stopDistAbs;
            this.completeJump();
            return;
        }

        // Compute speed fraction based on phase
        let speedFrac = 0.0;

        switch (this.jumpState.phase) {
            case 'aligning': {
                const t = clamp(elapsed / this.ALIGN_DURATION_S, 0, 1);
                // Ease-in-out for rotation
                const smoothT = t * t * (3 - 2 * t);

                // Interpolate pitch
                const p = lerp(this.jumpState.startPitch, this.jumpState.targetPitch, smoothT);

                // Interpolate yaw (shortest path)
                let dYaw = this.jumpState.targetYaw - this.jumpState.startYaw;
                // Normalize to [-PI, PI]
                dYaw = ((dYaw + Math.PI) % (2 * Math.PI)) - Math.PI; // Standard modulo behavior fix needed?
                // Actually JS % operator can return negative.
                // Proper wrap:
                while (dYaw > Math.PI) dYaw -= 2 * Math.PI;
                while (dYaw < -Math.PI) dYaw += 2 * Math.PI;

                const y = this.jumpState.startYaw + dYaw * smoothT;
                this.camera.setYawPitch(y, p);

                if (elapsed >= this.ALIGN_DURATION_S) {
                    this.jumpState.phase = 'accelerating';
                    this.jumpState.startTime = now; // reset timer for accel phase
                }
                break;
            }

            case 'accelerating': {
                // Ramp speed from 0 to 1 over ACCEL_DURATION_S
                const t = clamp(elapsed / this.ACCEL_DURATION_S, 0, 1);
                speedFrac = t;
                this.throttle = t; // For UI display

                // Transition to cruising after accel time
                if (elapsed >= this.ACCEL_DURATION_S) {
                    this.jumpState.phase = 'cruising';
                }

                // Or if we're already in ease range, skip to easing
                if (dist <= easeStartDistAbs) {
                    this.jumpState.phase = 'easing';
                    this.jumpState.startTime = now; // reset timer for ease phase
                }
                break;
            }

            case 'cruising': {
                // Full speed forward
                speedFrac = 1.0;
                this.throttle = 1.0;

                // Transition to easing when close to target
                if (dist <= easeStartDistAbs) {
                    this.jumpState.phase = 'easing';
                    this.jumpState.startTime = now; // reset timer for ease phase
                }
                break;
            }

            case 'easing': {
                // Gradually reduce speed as we approach target
                // Map dist from [stopDistAbs, easeStartDistAbs] to speed [0, 1]
                const easeRange = Math.max(1e-6, easeStartDistAbs - stopDistAbs);
                const easeProg = clamp((dist - stopDistAbs) / easeRange, 0, 1);

                // Non-linear easing: square the progress to make it taper aggressively
                // easeProg goes from 1.0 (start of ease) -> 0.0 (stop shell)
                // Squaring means we spend more time at lower speeds.
                speedFrac = Math.pow(easeProg, 2.0) * 0.8;
                this.throttle = speedFrac;

                // Gradually restore camera settings
                const restoreFrac = 1 - easeProg;
                this.camera.travel_time_s = this.JUMP_TRAVEL_TIME + restoreFrac * (this.savedTravelTime - this.JUMP_TRAVEL_TIME);
                this.camera.accel_time_s = this.JUMP_ACCEL_TIME + restoreFrac * (this.savedAccelTime - this.JUMP_ACCEL_TIME);
                break;
            }
        }

        // Compute jump speed in m/s
        const jumpSpeedMps = this.jumpState.maxSpeedMps * speedFrac;

        // Calculate movement step in Spherical coordinates (unused legacy step logic removed)
        if (dist > 1e-6) {
            // Compute desired position at new time (t + dt) via spherical interpolation?
            // No, that's complex because we are speed-based, not time-based.
            // But we need to follow the curve.

            // Alternative: Map 'progress' from 0 (Start) to 1 (Target)
            // But we don't have a reliable 'progress' because speed varies.
            // We can approximate:
            // Let's project current position onto the start->target path? No.

            // Simple approach:
            // Track 'progress' explicitly based on distance covered?
            // Or re-calculate position based on a 't' parameter?

            // Let's rely on 'elapsed' and duration? 
            // The jump has a defined duration `durationS`.
            // We can map `elapsed / durationS` to approximate t, but apply the speed profile.

            // Actually, we already integrate position. That works for straight lines.
            // For a curve, integrating velocity vector is hard if the vector changes direction.
            // BETTER: Calculate position strictly as a function of 't' (parameter 0-1) 
            // where 't' advances based on our speed profile.

            // Let's store 'currentT' in JumpState.
            if (typeof (this.jumpState as any).currentT === 'undefined') {
                (this.jumpState as any).currentT = 0;
            }

            // Calculate dt in parameter space.
            // total_dist approx = arc length? Or just linear start-end dist?
            // Let's assume start->end linear dist for speed calc to keep it simple.
            const totalLinearDist = this.jumpState.startDist;

            // speedMps = maxSpeed * speedFrac
            // dt_linear = speedMps * dt
            // dt_param = dt_linear / totalLinearDist

            let dtParam = 0;
            if (totalLinearDist > 0) {
                dtParam = jumpSpeedMps * dt / totalLinearDist;
            }

            let t = (this.jumpState as any).currentT + dtParam;
            if (t > 1) t = 1;
            (this.jumpState as any).currentT = t;

            // Convert start/end to spherical
            const startSph = toSpherical(this.jumpState.startPos);
            const targetSph = toSpherical(this.jumpState.targetPos);

            // Interpolate
            const currentSph = lerpSpherical(startSph, targetSph, t);
            const nextPos = toCartesian(currentSph);

            // Update position
            this.camera.position.x = nextPos.x;
            this.camera.position.y = nextPos.y;
            this.camera.position.z = nextPos.z;

            // Check completion: t=1 or close enough to stop shell
            // Actually, strict t=1 is arrival at center.
            // We need to stop at stopDistAbsM from center.
            // dist = r ? No, distance to center.
            // We can check cartesian dist to target.

            const dx = this.jumpState.targetPos.x - this.camera.position.x;
            const dy = this.jumpState.targetPos.y - this.camera.position.y;
            const dz = this.jumpState.targetPos.z - this.camera.position.z;
            const distRemaining = Math.sqrt(dx * dx + dy * dy + dz * dz);

            if (distRemaining <= stopDistAbs || t >= 1.0) {
                // Snap to stop shell
                // But in spherical, "stop shell" is just r_target +/- stopDist? 
                // No, target might be planetary (small R relative to Sun).
                // Use vector from target center.

                // If t >= 1, we are at target. Back up to stop shell.
                // Vector target -> origin.
                // Actually easier: just use the previous vector logic to snap.
                const toTarget = { x: dx, y: dy, z: dz };
                const d = Math.sqrt(dx * dx + dy * dy + dz * dz);
                if (d > 0) {
                    this.camera.position.x = this.jumpState.targetPos.x - (toTarget.x / d) * stopDistAbs;
                    this.camera.position.y = this.jumpState.targetPos.y - (toTarget.y / d) * stopDistAbs;
                    this.camera.position.z = this.jumpState.targetPos.z - (toTarget.z / d) * stopDistAbs;
                }
                this.completeJump();
            }
        }
    }

    /**
     * Get current throttle value (for UI display).
     */
    getThrottle(): number {
        return this.throttle;
    }

    /**
     * Reset throttle to zero.
     */
    resetThrottle(): void {
        this.throttle = 0.0;
        this.throttleTarget = 0.0;
    }
}







