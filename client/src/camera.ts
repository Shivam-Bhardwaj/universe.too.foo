import {
    determineRegime,
    computeSpeed,
    computeRegimeBoundaryBlend,
    REGIME_SPEEDS,
} from './scale_system';

export interface Vec3d {
    x: number;
    y: number;
    z: number;
}

export interface Quat {
    x: number;
    y: number;
    z: number;
    w: number;
}

function clamp(v: number, lo: number, hi: number): number {
    return Math.max(lo, Math.min(hi, v));
}

function vec3Len(v: Vec3d): number {
    return Math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

function vec3Norm(v: Vec3d): Vec3d {
    const len = vec3Len(v);
    if (len <= 0) return { x: 0, y: 0, z: 0 };
    return { x: v.x / len, y: v.y / len, z: v.z / len };
}

function quatMul(a: Quat, b: Quat): Quat {
    // Hamilton product
    return {
        w: a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
        x: a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        y: a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        z: a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
    };
}

function quatNorm(q: Quat): Quat {
    const len = Math.sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
    if (len <= 0) return { x: 0, y: 0, z: 0, w: 1 };
    const inv = 1.0 / len;
    return { x: q.x * inv, y: q.y * inv, z: q.z * inv, w: q.w * inv };
}

function quatFromAxisAngle(axis: Vec3d, angleRad: number): Quat {
    const a = vec3Norm(axis);
    const half = angleRad * 0.5;
    const s = Math.sin(half);
    return quatNorm({ x: a.x * s, y: a.y * s, z: a.z * s, w: Math.cos(half) });
}

function quatRotateVec(q: Quat, v: Vec3d): Vec3d {
    // v' = q * (v,0) * conj(q)
    const vx = v.x;
    const vy = v.y;
    const vz = v.z;

    const qx = q.x;
    const qy = q.y;
    const qz = q.z;
    const qw = q.w;

    // t = 2 * cross(q.xyz, v)
    const tx = 2 * (qy * vz - qz * vy);
    const ty = 2 * (qz * vx - qx * vz);
    const tz = 2 * (qx * vy - qy * vx);

    // v' = v + qw * t + cross(q.xyz, t)
    return {
        x: vx + qw * tx + (qy * tz - qz * ty),
        y: vy + qw * ty + (qz * tx - qx * tz),
        z: vz + qw * tz + (qx * ty - qy * tx),
    };
}

function vec3Cross(a: Vec3d, b: Vec3d): Vec3d {
    return {
        x: a.y * b.z - a.z * b.y,
        y: a.z * b.x - a.x * b.z,
        z: a.x * b.y - a.y * b.x,
    };
}

function vec3Dot(a: Vec3d, b: Vec3d): number {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

function mat4Mul(a: Float32Array, b: Float32Array): Float32Array {
    // Column-major 4x4 multiply: out = a * b
    const out = new Float32Array(16);
    for (let c = 0; c < 4; c++) {
        for (let r = 0; r < 4; r++) {
            out[c * 4 + r] =
                a[0 * 4 + r] * b[c * 4 + 0] +
                a[1 * 4 + r] * b[c * 4 + 1] +
                a[2 * 4 + r] * b[c * 4 + 2] +
                a[3 * 4 + r] * b[c * 4 + 3];
        }
    }
    return out;
}

export class LocalCamera {
    // World position (meters; JS numbers are f64)
    position: Vec3d = { x: 0, y: 0, z: 1.5e11 };

    // Floating origin (world-space). We keep all GPU positions relative to this origin
    // and keep the camera near the origin for numeric stability.
    origin: Vec3d = { x: 0, y: 0, z: 0 };

    // Orientation (unit quaternion, local -> world). Avoids gimbal lock.
    private orientation: Quat = { x: 0, y: 0, z: 0, w: 1 };

    // Yaw/pitch-only orientation (no roll drift)
    private yawRad = 0;
    private pitchRad = 0;

    // Camera params
    fovY = (60 * Math.PI) / 180;
    near = 1.0;
    far = 1e20;
    logDepthC = 1.0;

    // ---------------------------------------------------------------------
    // Navigation / "gain" tuning (time-based)
    //
    // Desired UX:
    // - Max speed ramps in ~10 seconds of sustained input
    // - At max speed, you can traverse the current distance-to-origin scale in ~1 minute
    // ---------------------------------------------------------------------
    accel_time_s = 10.0;
    travel_time_s = 60.0;

    // Current ramp state (0..1) based on sustained movement input.
    private speed_ramp = 0.0;

    // Current movement speed (m/s), updated each frame by `updateGain`.
    speed = 0.0;
    sensitivity = 0.003;

    forward(): Vec3d {
        // Local forward is -Z
        return vec3Norm(quatRotateVec(this.orientation, { x: 0, y: 0, z: -1 }));
    }

    right(): Vec3d {
        return vec3Norm(quatRotateVec(this.orientation, { x: 1, y: 0, z: 0 }));
    }

    up(): Vec3d {
        return vec3Norm(quatRotateVec(this.orientation, { x: 0, y: 1, z: 0 }));
    }

    rotate(dx: number, dy: number) {
        // Mouse/touch look: yaw around world +Y, pitch around local +X.
        // Convention: dragging/moving down looks down (invert screen Y).
        this.rotateRad(dx * this.sensitivity, -dy * this.sensitivity);
    }

    rotateRad(yawDelta: number, pitchDelta: number) {
        // Apply yaw/pitch deltas (no roll - stable horizon)
        const MAX_PITCH = Math.PI / 2 - 0.001; // Prevent gimbal lock at vertical
        this.yawRad += yawDelta;
        this.pitchRad = clamp(this.pitchRad + pitchDelta, -MAX_PITCH, MAX_PITCH);
        this.rebuildOrientationFromYawPitch();
    }

    private rebuildOrientationFromYawPitch() {
        // Yaw around world +Y, then pitch around local +X
        const qYaw = quatFromAxisAngle({ x: 0, y: 1, z: 0 }, this.yawRad);
        const qPitch = quatFromAxisAngle({ x: 1, y: 0, z: 0 }, this.pitchRad);
        this.orientation = quatNorm(quatMul(qYaw, qPitch));
    }

    translate(forward: number, right: number, up: number, dt: number) {
        const f = this.forward();
        const r = this.right();
        const u = this.up();

        const mx = f.x * forward + r.x * right + u.x * up;
        const my = f.y * forward + r.y * right + u.y * up;
        const mz = f.z * forward + r.z * right + u.z * up;

        // Proposed new position
        const newX = this.position.x + mx * this.speed * dt;
        const newY = this.position.y + my * this.speed * dt;
        const newZ = this.position.z + mz * this.speed * dt;
        const newDist = Math.sqrt(newX * newX + newY * newY + newZ * newZ);

        // Enforce distance limits (min 1 km, max based on heliosphere constraint)
        const MIN_DISTANCE = 1e3;  // 1 km minimum
        const MAX_DISTANCE = 1e25; // ~300 Mpc maximum (computed from heliosphere constraint)

        if (newDist < MIN_DISTANCE || newDist > MAX_DISTANCE) {
            // Clamp to limit sphere
            const targetDist = clamp(newDist, MIN_DISTANCE, MAX_DISTANCE);
            const scale = targetDist / Math.max(1e-10, newDist);
            this.position.x = newX * scale;
            this.position.y = newY * scale;
            this.position.z = newZ * scale;
        } else {
            this.position.x = newX;
            this.position.y = newY;
            this.position.z = newZ;
        }
    }

    /**
     * Update the time-based movement gain with regime-aware speed scaling.
     *
     * - Hold any movement input (WASD / Space / Shift) to ramp up speed
     * - Speed automatically scales based on distance from origin (scale regime)
     * - Smooth transitions across regime boundaries
     */
    updateGain(dt: number, moving: boolean) {
        const dist = vec3Len(this.position);
        const regime = determineRegime(dist);
        const config = REGIME_SPEEDS[regime];

        // Ramp time varies by regime
        const a = Math.max(0.001, config.ramp_time_s);
        if (moving) {
            this.speed_ramp = clamp(this.speed_ramp + dt / a, 0.0, 1.0);
        } else {
            // Reset when you release movement so it doesn't "stick" at huge gain.
            this.speed_ramp = 0.0;
        }

        // Compute speed for current regime
        let regimeSpeed = computeSpeed(regime, this.speed_ramp);

        // Apply smooth blending near regime boundaries
        const boundaryBlend = computeRegimeBoundaryBlend(dist);
        this.speed = regimeSpeed * boundaryBlend;

        // Respect manual travel_time adjustment (Q/E keys) as speed multiplier
        // Default travel_time_s is 60s; Q increases it (slower), E decreases it (faster)
        const travelTimeFactor = 60.0 / Math.max(5.0, this.travel_time_s);
        this.speed *= travelTimeFactor;
    }

    adjustTravelTime(deltaSeconds: number) {
        // Keep within a sensible range so controls remain recoverable.
        this.travel_time_s = clamp(this.travel_time_s + deltaSeconds, 5.0, 600.0);
    }

    resetGain() {
        this.travel_time_s = 60.0;
        this.speed_ramp = 0.0;
        this.speed = 0.0;
    }

    resetSpeed() {
        this.speed_ramp = 0.0;
        this.speed = 0.0;
    }

    setPosition(pos: Vec3d) {
        this.position = { ...pos };
        // Teleports should not preserve any accumulated "gain"
        this.speed_ramp = 0.0;
        this.speed = 0.0;
    }

    getYaw(): number {
        return this.yawRad;
    }

    getPitch(): number {
        return this.pitchRad;
    }

    setYawPitch(yawRad: number, pitchRad: number) {
        const MAX_PITCH = Math.PI / 2 - 0.001;
        this.yawRad = yawRad;
        this.pitchRad = clamp(pitchRad, -MAX_PITCH, MAX_PITCH);
        this.rebuildOrientationFromYawPitch();
    }

    computeLookAtAngles(target: Vec3d): { yaw: number; pitch: number } {
        const dir = vec3Norm({
            x: target.x - this.position.x,
            y: target.y - this.position.y,
            z: target.z - this.position.z,
        });

        // pitch = dir.y.asin()
        // yaw = atan2(-dir.x, -dir.z)
        // (Note: dir.x is negated to match our Left-is-Positive-Yaw convention)
        return {
            pitch: Math.asin(dir.y),
            yaw: Math.atan2(-dir.x, -dir.z),
        };
    }

    lookAt(target: Vec3d) {
        const angles = this.computeLookAtAngles(target);
        this.setYawPitch(angles.yaw, angles.pitch);
    }

    viewMatrix(): Float32Array {
        // Mat4::look_to_rh(eye=camera_rel, dir=forward, up=up)
        // (camera_rel is kept small via floating origin)
        const eye = {
            x: this.position.x - this.origin.x,
            y: this.position.y - this.origin.y,
            z: this.position.z - this.origin.z,
        };
        const f = vec3Norm(this.forward());
        const up = vec3Norm(this.up());
        const s = vec3Norm(vec3Cross(f, up));
        const u = vec3Cross(s, f);

        const tx = -vec3Dot(s, eye);
        const ty = -vec3Dot(u, eye);
        const tz = vec3Dot(f, eye);

        // Column-major
        return new Float32Array([
            s.x, s.y, s.z, 0,
            u.x, u.y, u.z, 0,
            -f.x, -f.y, -f.z, 0,
            tx, ty, tz, 1,
        ]);
    }

    projectionMatrix(aspect: number): Float32Array {
        const f = 1.0 / Math.tan(this.fovY / 2);

        // Reverse-Z infinite projection (matches Rust)
        return new Float32Array([
            f / aspect, 0, 0, 0,
            0, f, 0, 0,
            0, 0, 0, -1,
            0, 0, this.near, 0,
        ]);
    }

    cameraUniform(aspect: number): Float32Array {
        const view = this.viewMatrix();
        const proj = this.projectionMatrix(aspect);
        const viewProj = mat4Mul(proj, view);

        // Rust layout: view(16) + proj(16) + view_proj(16) + position(3)+pad + near+far+fov_y+log_depth_c
        const out = new Float32Array(56);
        out.set(view, 0);
        out.set(proj, 16);
        out.set(viewProj, 32);

        // camera position relative to origin (mostly for debugging; shader doesn't rely on it)
        out[48] = (this.position.x - this.origin.x) as number;
        out[49] = (this.position.y - this.origin.y) as number;
        out[50] = (this.position.z - this.origin.z) as number;
        out[51] = 0; // pad

        out[52] = this.near;
        out[53] = this.far;
        out[54] = this.fovY;
        out[55] = this.logDepthC;
        return out;
    }
}



