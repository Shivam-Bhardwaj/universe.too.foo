/**
 * Label Renderer for HELIOS Universe Planetarium
 *
 * Renders billboard-style text labels for celestial objects
 * with distance-based filtering and formatting.
 */

import { formatDistance } from './scale_system';

export interface LabeledObject {
    name: string;
    type: string;  // "Planet", "Star", "Galaxy", "Nebula", etc.
    position: Float32Array;  // [x, y, z] in meters (heliocentric)
}

export interface LabelConfig {
    maxDistance: number;        // Maximum distance to show labels (meters)
    fontSize: number;           // Font size in pixels
    fontFamily: string;         // CSS font family
    textColor: string;          // CSS color
    backgroundColor: string;    // CSS background color (with alpha)
    padding: number;            // Padding around text (pixels)
    minScreenDistance: number;  // Minimum pixels between labels to avoid overlap
}

const DEFAULT_CONFIG: LabelConfig = {
    maxDistance: 1e14,          // ~670 AU (show planets + nearby stars)
    fontSize: 14,
    fontFamily: 'monospace',
    textColor: 'rgba(255, 255, 255, 0.95)',
    backgroundColor: 'rgba(0, 0, 0, 0.7)',
    padding: 4,
    minScreenDistance: 50,      // Avoid labels closer than 50px
};

interface ScreenLabel {
    object: LabeledObject;
    distance: number;
    screenX: number;
    screenY: number;
    text: string;
}

export class LabelRenderer {
    private canvas: HTMLCanvasElement;
    private ctx: CanvasRenderingContext2D;
    private config: LabelConfig;

    constructor(canvas: HTMLCanvasElement, config: Partial<LabelConfig> = {}) {
        this.canvas = canvas;
        const ctx = canvas.getContext('2d');
        if (!ctx) {
            throw new Error('Failed to get 2D context for label canvas');
        }
        this.ctx = ctx;
        this.config = { ...DEFAULT_CONFIG, ...config };
    }

    /**
     * Render labels for all objects within range
     */
    render(
        objects: LabeledObject[],
        cameraPos: Float32Array,  // [x, y, z]
        viewMatrix: Float32Array,  // 4x4 view matrix
        projMatrix: Float32Array,  // 4x4 projection matrix
        viewportWidth: number,
        viewportHeight: number
    ): void {
        // Clear canvas
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        // Filter and project objects to screen space
        const screenLabels: ScreenLabel[] = [];

        for (const obj of objects) {
            // Calculate distance from camera
            const dx = obj.position[0] - cameraPos[0];
            const dy = obj.position[1] - cameraPos[1];
            const dz = obj.position[2] - cameraPos[2];
            const distance = Math.sqrt(dx * dx + dy * dy + dz * dz);

            // Filter by max distance
            if (distance > this.config.maxDistance) {
                continue;
            }

            // Project to screen space
            const screenPos = this.projectToScreen(
                obj.position,
                viewMatrix,
                projMatrix,
                viewportWidth,
                viewportHeight
            );

            // Check if on screen
            if (!screenPos || screenPos.z > 1.0 || screenPos.z < 0.0) {
                continue;  // Behind camera or clipped
            }

            // Check if within viewport bounds
            if (
                screenPos.x < 0 || screenPos.x > viewportWidth ||
                screenPos.y < 0 || screenPos.y > viewportHeight
            ) {
                continue;  // Off screen
            }

            // Format label text
            const distanceStr = formatDistance(distance);
            const text = `${obj.name} - ${distanceStr} (${obj.type})`;

            screenLabels.push({
                object: obj,
                distance,
                screenX: screenPos.x,
                screenY: screenPos.y,
                text,
            });
        }

        // Sort by distance (closest first for priority)
        screenLabels.sort((a, b) => a.distance - b.distance);

        // Remove overlapping labels
        const finalLabels = this.cullOverlappingLabels(screenLabels);

        // Render labels
        for (const label of finalLabels) {
            this.drawLabel(label.text, label.screenX, label.screenY);
        }
    }

    /**
     * Project 3D position to screen space using view and projection matrices
     */
    private projectToScreen(
        worldPos: Float32Array,
        viewMatrix: Float32Array,
        projMatrix: Float32Array,
        viewportWidth: number,
        viewportHeight: number
    ): { x: number; y: number; z: number } | null {
        // Transform to view space
        const viewPos = this.transformPoint(worldPos, viewMatrix);

        // Transform to clip space
        const clipPos = this.transformPoint(viewPos, projMatrix);

        // Perspective divide
        if (Math.abs(clipPos[3]) < 1e-6) {
            return null;  // Too close to camera
        }

        const ndcX = clipPos[0] / clipPos[3];
        const ndcY = clipPos[1] / clipPos[3];
        const ndcZ = clipPos[2] / clipPos[3];

        // NDC to screen space
        const screenX = (ndcX * 0.5 + 0.5) * viewportWidth;
        const screenY = (1.0 - (ndcY * 0.5 + 0.5)) * viewportHeight;  // Flip Y

        return { x: screenX, y: screenY, z: ndcZ };
    }

    /**
     * Transform a 3D point by a 4x4 matrix
     */
    private transformPoint(point: Float32Array, matrix: Float32Array): Float32Array {
        const x = point[0];
        const y = point[1];
        const z = point[2];
        const w = 1.0;

        const outX = matrix[0] * x + matrix[4] * y + matrix[8] * z + matrix[12] * w;
        const outY = matrix[1] * x + matrix[5] * y + matrix[9] * z + matrix[13] * w;
        const outZ = matrix[2] * x + matrix[6] * y + matrix[10] * z + matrix[14] * w;
        const outW = matrix[3] * x + matrix[7] * y + matrix[11] * z + matrix[15] * w;

        return new Float32Array([outX, outY, outZ, outW]);
    }

    /**
     * Remove overlapping labels (keep closest)
     */
    private cullOverlappingLabels(labels: ScreenLabel[]): ScreenLabel[] {
        const result: ScreenLabel[] = [];

        for (const label of labels) {
            let overlaps = false;

            for (const existing of result) {
                const dx = label.screenX - existing.screenX;
                const dy = label.screenY - existing.screenY;
                const dist = Math.sqrt(dx * dx + dy * dy);

                if (dist < this.config.minScreenDistance) {
                    overlaps = true;
                    break;
                }
            }

            if (!overlaps) {
                result.push(label);
            }
        }

        return result;
    }

    /**
     * Draw a single label at screen position
     */
    private drawLabel(text: string, x: number, y: number): void {
        const ctx = this.ctx;
        const config = this.config;

        // Set font
        ctx.font = `${config.fontSize}px ${config.fontFamily}`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';

        // Measure text
        const metrics = ctx.measureText(text);
        const textWidth = metrics.width;
        const textHeight = config.fontSize;

        // Background box
        const boxWidth = textWidth + config.padding * 2;
        const boxHeight = textHeight + config.padding * 2;
        const boxX = x - boxWidth / 2;
        const boxY = y - boxHeight / 2;

        // Draw background
        ctx.fillStyle = config.backgroundColor;
        ctx.fillRect(boxX, boxY, boxWidth, boxHeight);

        // Draw border
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
        ctx.lineWidth = 1;
        ctx.strokeRect(boxX, boxY, boxWidth, boxHeight);

        // Draw text
        ctx.fillStyle = config.textColor;
        ctx.fillText(text, x, y);
    }

    /**
     * Update configuration
     */
    setConfig(config: Partial<LabelConfig>): void {
        this.config = { ...this.config, ...config };
    }

    /**
     * Resize label canvas
     */
    resize(width: number, height: number): void {
        this.canvas.width = width;
        this.canvas.height = height;
    }
}
