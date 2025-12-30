/**
 * Client-Side Time Controller
 *
 * Manages simulation time for the universe visualization,
 * supporting ±100,000 year range from J2000 epoch.
 */

// J2000 epoch (January 1, 2000, 12:00 TT)
export const J2000_JD = 2451545.0;

// Time limits (Julian Days relative to J2000)
const YEAR_IN_DAYS = 365.25;
const MIN_YEARS = -100000;
const MAX_YEARS = 100000;
export const MIN_JD = J2000_JD + (MIN_YEARS * YEAR_IN_DAYS);
export const MAX_JD = J2000_JD + (MAX_YEARS * YEAR_IN_DAYS);

/**
 * Accuracy mode determines propagation fidelity
 */
export enum TimeAccuracyMode {
    Ephemeris = 0,     // ±1,000 years: Sub-km precision (future: use ANISE/SPICE)
    Keplerian = 1,     // ±10,000 years: Arc-second accuracy with secular rates
    Statistical = 2,   // ±100,000 years: Proper motion + galactic rotation
}

export interface TimeState {
    julianDate: number;      // Current Julian Date
    rate: number;            // Simulation seconds per real second
    isPaused: boolean;
    accuracyMode: TimeAccuracyMode;
}

export class ClientTimeController {
    private state: TimeState;
    private listeners: Array<(state: TimeState) => void> = [];

    constructor(initialJD: number = J2000_JD) {
        this.state = {
            julianDate: this.clampJD(initialJD),
            rate: 0,  // Start paused
            isPaused: true,
            accuracyMode: this.determineAccuracyMode(initialJD),
        };
    }

    /**
     * Update time based on real elapsed time
     */
    tick(realDtSeconds: number): void {
        if (this.state.isPaused || this.state.rate === 0) {
            return;
        }

        const simDtSeconds = realDtSeconds * this.state.rate;
        const deltaDays = simDtSeconds / 86400.0;

        this.setJulianDate(this.state.julianDate + deltaDays);
    }

    /**
     * Set absolute Julian Date
     */
    setJulianDate(jd: number): void {
        this.state.julianDate = this.clampJD(jd);
        this.state.accuracyMode = this.determineAccuracyMode(this.state.julianDate);
        this.notifyListeners();
    }

    /**
     * Set time rate (simulation seconds per real second)
     */
    setRate(rate: number): void {
        this.state.rate = rate;
        this.notifyListeners();
    }

    /**
     * Adjust rate by factor
     */
    multiplyRate(factor: number): void {
        this.state.rate *= factor;
        this.notifyListeners();
    }

    /**
     * Toggle pause state
     */
    togglePause(): void {
        this.state.isPaused = !this.state.isPaused;
        this.notifyListeners();
    }

    setPaused(paused: boolean): void {
        this.state.isPaused = paused;
        this.notifyListeners();
    }

    /**
     * Jump by years
     */
    addYears(years: number): void {
        const deltaDays = years * YEAR_IN_DAYS;
        this.setJulianDate(this.state.julianDate + deltaDays);
    }

    /**
     * Reset to J2000
     */
    resetToJ2000(): void {
        this.setJulianDate(J2000_JD);
        this.setRate(0);
        this.setPaused(true);
    }

    /**
     * Get current state
     */
    getState(): TimeState {
        return { ...this.state };
    }

    getJulianDate(): number {
        return this.state.julianDate;
    }

    getRate(): number {
        return this.state.rate;
    }

    isPaused(): boolean {
        return this.state.isPaused;
    }

    /**
     * Convert JD to years from J2000
     */
    getYearsFromJ2000(): number {
        return (this.state.julianDate - J2000_JD) / YEAR_IN_DAYS;
    }

    /**
     * Subscribe to time changes
     */
    addListener(callback: (state: TimeState) => void): void {
        this.listeners.push(callback);
    }

    private notifyListeners(): void {
        for (const listener of this.listeners) {
            listener(this.state);
        }
    }

    private clampJD(jd: number): number {
        return Math.max(MIN_JD, Math.min(MAX_JD, jd));
    }

    private determineAccuracyMode(jd: number): TimeAccuracyMode {
        const yearsFromJ2000 = Math.abs((jd - J2000_JD) / YEAR_IN_DAYS);

        if (yearsFromJ2000 < 1000) {
            return TimeAccuracyMode.Ephemeris;
        } else if (yearsFromJ2000 < 10000) {
            return TimeAccuracyMode.Keplerian;
        } else {
            return TimeAccuracyMode.Statistical;
        }
    }

    /**
     * Convert Julian Date to Gregorian date string
     */
    toDateString(jd?: number): string {
        const targetJD = jd ?? this.state.julianDate;

        const z = Math.floor(targetJD + 0.5);
        const f = targetJD + 0.5 - z;

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

        // Handle large year ranges
        const yearStr = year >= 10000 || year <= -10000 ? year.toString() : year.toString();

        return `${yearStr}-${pad(month)}-${pad(day)} ${pad(hour)}:${pad(minute)}:${pad(second)} UTC`;
    }

    /**
     * Get human-readable time offset from J2000
     */
    getTimeOffsetString(): string {
        const years = this.getYearsFromJ2000();
        const absYears = Math.abs(years);

        if (absYears < 1) {
            const days = years * 365.25;
            return `${days >= 0 ? '+' : ''}${days.toFixed(1)} days`;
        } else if (absYears < 1000) {
            return `${years >= 0 ? '+' : ''}${years.toFixed(1)} years`;
        } else if (absYears < 10000) {
            const ky = years / 1000;
            return `${ky >= 0 ? '+' : ''}${ky.toFixed(2)} ky`;
        } else {
            const ky = years / 1000;
            return `${ky >= 0 ? '+' : ''}${ky.toFixed(1)} ky`;
        }
    }
}
