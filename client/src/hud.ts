import { ServerState } from './client';

export class HUD {
    private timeEl: HTMLElement;
    private rateEl: HTMLElement;
    private posEl: HTMLElement;
    private latencyEl: HTMLElement;
    private fpsEl: HTMLElement;
    private usersEl: HTMLElement;
    private jumpsEl: HTMLElement;
    private datasetEl: HTMLElement;

    constructor() {
        this.timeEl = document.getElementById('hud-time')!;
        this.rateEl = document.getElementById('time-rate')!;
        this.posEl = document.getElementById('position')!;
        this.latencyEl = document.getElementById('latency')!;
        this.fpsEl = document.getElementById('fps')!;
        this.usersEl = document.getElementById('users')!;
        this.jumpsEl = document.getElementById('jumps')!;
        this.datasetEl = document.getElementById('dataset')!;
    }

    update(state: ServerState): void {
        // Convert JD to date string
        const date = this.jdToDate(state.epoch_jd);
        this.timeEl.textContent = date;

        // Time rate
        const rateYearsPerSec = state.time_rate / (86400.0 * 365.25);
        if (Math.abs(rateYearsPerSec) >= 1) {
            this.rateEl.textContent = `${rateYearsPerSec.toFixed(1)} yr/s`;
        } else if (Math.abs(rateYearsPerSec) >= 1/365) {
            this.rateEl.textContent = `${(rateYearsPerSec * 365).toFixed(1)} d/s`;
        } else if (Math.abs(rateYearsPerSec) >= 1/(365 * 24)) {
            this.rateEl.textContent = `${(rateYearsPerSec * 365 * 24).toFixed(1)} hr/s`;
        } else {
            this.rateEl.textContent = `${(rateYearsPerSec * 365 * 24 * 60).toFixed(1)} min/s`;
        }

        // Position (convert to AU)
        const AU = 1.496e11;
        const x = state.camera_x / AU;
        const y = state.camera_y / AU;
        const z = state.camera_z / AU;
        this.posEl.textContent = `${x.toFixed(2)}, ${y.toFixed(2)}, ${z.toFixed(2)} AU`;

        // FPS
        this.fpsEl.textContent = `${state.fps.toFixed(1)}`;

        // Connected users (best-effort)
        this.usersEl.textContent = `${state.clients}`;
    }

    updateJumpStatus(remaining: number, max: number, registered: boolean): void {
        const tier = registered ? 'REG' : 'GUEST';
        this.jumpsEl.textContent = `${remaining}/${max} (${tier})`;
    }

    updateDatasetProgress(loadedCells: number, totalCells: number, loadedSplats: number, totalSplats: number): void {
        const cellsTxt = `${loadedCells}/${totalCells} cells`;
        const splatsTxt = `${loadedSplats.toLocaleString()}/${totalSplats.toLocaleString()} splats`;
        this.datasetEl.textContent = `${cellsTxt} · ${splatsTxt}`;
    }

    updateLatency(ms: number): void {
        this.latencyEl.textContent = `${ms.toFixed(0)} ms`;
    }

    private jdToDate(jd: number): string {
        // Convert Julian Date to Gregorian calendar date
        const z = Math.floor(jd + 0.5);
        const f = jd + 0.5 - z;

        let a: number;
        if (z < 2299161) {
            a = z;
        } else {
            const alpha = Math.floor((z - 1867216.25) / 36524.25);
            a = z + 1 + alpha - Math.floor(alpha / 4);
        }

        const b = a + 1524;
        const c = Math.floor((b - 122.1) / 365.25);
        const d = Math.floor(365.25 * c);
        const e = Math.floor((b - d) / 30.6001);

        const day = b - d - Math.floor(30.6001 * e);
        const month = e < 14 ? e - 1 : e - 13;
        const year = month > 2 ? c - 4716 : c - 4715;

        const fracDay = f * 24;
        const hour = Math.floor(fracDay);
        const fracHour = (fracDay - hour) * 60;
        const minute = Math.floor(fracHour);
        const second = Math.floor((fracHour - minute) * 60);

        const pad = (n: number) => n.toString().padStart(2, '0');

        return `${year}-${pad(month)}-${pad(day)} ${pad(hour)}:${pad(minute)}:${pad(second)} UTC`;
    }
}
