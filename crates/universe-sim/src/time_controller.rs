//! Time controller for simulation playback

use hifitime::{Epoch, Duration};

/// Time controller with variable rate playback
pub struct TimeController {
    /// Current simulation time
    current: Epoch,
    /// Minimum allowed time (J2000 - 5000 years)
    min_epoch: Epoch,
    /// Maximum allowed time (J2000 + 5000 years)
    max_epoch: Epoch,
    /// Simulation rate (sim seconds per real second)
    /// 1.0 = realtime, 86400.0 = 1 day per second
    rate: f64,
    /// Is simulation paused?
    paused: bool,
}

impl TimeController {
    pub fn new() -> Self {
        let j2000 = Epoch::from_gregorian_utc(2000, 1, 1, 12, 0, 0, 0);

        Self {
            current: j2000,
            min_epoch: j2000 - Duration::from_days(5000.0 * 365.25),
            max_epoch: j2000 + Duration::from_days(5000.0 * 365.25),
            rate: 86400.0, // Default: 1 day per second
            paused: false,
        }
    }

    /// Create at specific epoch
    pub fn at_epoch(epoch: Epoch) -> Self {
        let mut tc = Self::new();
        tc.set_time(epoch);
        tc
    }

    /// Get current simulation time
    pub fn current(&self) -> Epoch {
        self.current
    }

    /// Set absolute time
    pub fn set_time(&mut self, epoch: Epoch) {
        self.current = epoch.clamp(self.min_epoch, self.max_epoch);
    }

    /// Get current rate
    pub fn rate(&self) -> f64 {
        self.rate
    }

    /// Set simulation rate (sim seconds per real second)
    pub fn set_rate(&mut self, rate: f64) {
        self.rate = rate.clamp(-1e9, 1e9); // Allow reverse time
    }

    /// Set rate as time units per second
    pub fn set_rate_days_per_second(&mut self, days: f64) {
        self.set_rate(days * 86400.0);
    }

    pub fn set_rate_years_per_second(&mut self, years: f64) {
        self.set_rate(years * 365.25 * 86400.0);
    }

    /// Get rate in years per second
    pub fn rate_years_per_second(&self) -> f64 {
        self.rate / (365.25 * 86400.0)
    }

    /// Pause simulation
    pub fn pause(&mut self) {
        self.paused = true;
    }

    /// Resume simulation
    pub fn resume(&mut self) {
        self.paused = false;
    }

    /// Toggle pause state
    pub fn toggle_pause(&mut self) {
        self.paused = !self.paused;
    }

    /// Is paused?
    pub fn is_paused(&self) -> bool {
        self.paused
    }

    /// Advance simulation by real-world delta time
    /// Returns new epoch
    pub fn tick(&mut self, real_dt_seconds: f64) -> Epoch {
        if self.paused {
            return self.current;
        }

        let sim_dt = real_dt_seconds * self.rate;
        self.current = (self.current + Duration::from_seconds(sim_dt))
            .clamp(self.min_epoch, self.max_epoch);

        self.current
    }

    /// Jump forward/backward by duration
    pub fn jump(&mut self, duration: Duration) {
        self.current = (self.current + duration)
            .clamp(self.min_epoch, self.max_epoch);
    }

    /// Jump to preset epochs
    pub fn jump_to_j2000(&mut self) {
        self.current = Epoch::from_gregorian_utc(2000, 1, 1, 12, 0, 0, 0);
    }

    pub fn jump_to_now(&mut self) {
        // Current time (approximate - would need system time)
        self.current = Epoch::from_gregorian_utc(2024, 1, 1, 0, 0, 0, 0);
    }

    /// Get formatted time string
    pub fn format_time(&self) -> String {
        format!("{}", self.current)
    }

    /// Get year (approximate)
    pub fn year(&self) -> f64 {
        let j2000 = Epoch::from_gregorian_utc(2000, 1, 1, 12, 0, 0, 0);
        let days = (self.current - j2000).to_seconds() / 86400.0;
        2000.0 + days / 365.25
    }
}

impl Default for TimeController {
    fn default() -> Self {
        Self::new()
    }
}

/// Preset time rates
pub mod rates {
    /// Real-time
    pub const REALTIME: f64 = 1.0;
    /// 1 minute per second
    pub const MINUTE_PER_SEC: f64 = 60.0;
    /// 1 hour per second
    pub const HOUR_PER_SEC: f64 = 3600.0;
    /// 1 day per second
    pub const DAY_PER_SEC: f64 = 86400.0;
    /// 1 week per second
    pub const WEEK_PER_SEC: f64 = 7.0 * 86400.0;
    /// 1 month per second (~30 days)
    pub const MONTH_PER_SEC: f64 = 30.0 * 86400.0;
    /// 1 year per second
    pub const YEAR_PER_SEC: f64 = 365.25 * 86400.0;
    /// 10 years per second
    pub const DECADE_PER_SEC: f64 = 10.0 * 365.25 * 86400.0;
    /// 100 years per second
    pub const CENTURY_PER_SEC: f64 = 100.0 * 365.25 * 86400.0;
}
