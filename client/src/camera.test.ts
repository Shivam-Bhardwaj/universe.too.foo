
import { describe, it, expect, beforeEach } from 'vitest';
import { LocalCamera } from './camera';

describe('LocalCamera', () => {
    let camera: LocalCamera;

    beforeEach(() => {
        camera = new LocalCamera();
        // Reset to default (0,0,0) facing -Z
        camera.position = { x: 0, y: 0, z: 0 };
        camera.setYawPitch(0, 0);
    });

    it('should have initial forward vector as (0, 0, -1)', () => {
        const f = camera.forward();
        expect(f.x).toBeCloseTo(0);
        expect(f.y).toBeCloseTo(0);
        expect(f.z).toBeCloseTo(-1);
    });

    it('should have initial right vector as (1, 0, 0)', () => {
        const r = camera.right();
        expect(r.x).toBeCloseTo(1);
        expect(r.y).toBeCloseTo(0);
        expect(r.z).toBeCloseTo(0);
    });

    it('should have initial up vector as (0, 1, 0)', () => {
        const u = camera.up();
        expect(u.x).toBeCloseTo(0);
        expect(u.y).toBeCloseTo(1);
        expect(u.z).toBeCloseTo(0);
    });

    it('should compute correct yaw for viewing Right (+X)', () => {
        // Target at (10, 0, 0).
        // Since Forward is -Z, Right is +X.
        // To look Right, we rotate -90 degrees (or +270) around Y? 
        // Let's check the convention.
        // Hand rule: Up=+Y. Thumb=+Y. Fingers curl +Z -> +X.
        // Wait, standard RH: X cross Y = Z. 
        // If X=Right, Y=Up, then Z=Back. Forward=-Z.
        // Rotation +90 deg around Y: X -> -Z.
        // So Forward (-Z) -> Right (+X)? No. 
        // Forward (-Z) at +90 -> (-(-Z)) -> Z? 
        // Let's trust the math: 
        // q = axisAngle(Y, 90). Rotate(0,0,-1) by 90 deg around Y.
        // (0,0,-1) -> (-1, 0, 0). (Left).
        // So +90 deg Yaw = Look Left.
        // Therefore, to look Right (+X), we need -90 deg Yaw.

        const target = { x: 10, y: 0, z: 0 };
        const angles = camera.computeLookAtAngles(target);

        // angles.yaw should be -PI/2
        expect(angles.yaw).toBeCloseTo(-Math.PI / 2);
    });

    it('should compute correct yaw for viewing Left (-X)', () => {
        const target = { x: -10, y: 0, z: 0 };
        const angles = camera.computeLookAtAngles(target);

        // angles.yaw should be +PI/2
        expect(angles.yaw).toBeCloseTo(Math.PI / 2);
    });

    it('should compute correct yaw for viewing Back (+Z)', () => {
        const target = { x: 0, y: 0, z: 10 };
        const angles = camera.computeLookAtAngles(target);

        // angles.yaw should be PI (or -PI)
        expect(Math.abs(angles.yaw)).toBeCloseTo(Math.PI);
    });

    it('should lookAt correctly', () => {
        const target = { x: 100, y: 0, z: 0 }; // Right
        camera.lookAt(target);

        const f = camera.forward();
        // Forward should now be (1, 0, 0)
        expect(f.x).toBeCloseTo(1);
        expect(f.y).toBeCloseTo(0);
        expect(f.z).toBeCloseTo(0);
    });
});
