use anyhow::Result;
use clap::{Parser, Subcommand};
use universe_core::grid::HLGGrid;
use universe_core::coordinates::CartesianPosition;
use universe_core::constants::*;
use tracing_subscriber;
use std::path::PathBuf;
use hifitime::Epoch;
use std::str::FromStr;
use serde::{Serialize, Deserialize};
use std::io::Write;

#[derive(Parser)]
#[command(name = "universe")]
#[command(about = "Universe Visualization System")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Test grid coordinate transformation
    GridTest {
        /// Position as "x,y,z" in meters (or with suffix: "1.496e11,0,0" for 1 AU)
        #[arg(short, long)]
        position: String,

        /// Interpret position as AU instead of meters
        #[arg(long, default_value = "false")]
        au: bool,
    },

    /// Show grid configuration and statistics
    GridInfo,

    /// List cells for a distance range
    ListCells {
        /// Inner distance (AU)
        #[arg(long)]
        from_au: f64,

        /// Outer distance (AU)
        #[arg(long)]
        to_au: f64,
    },

    /// Download NASA DE440 ephemeris
    FetchEphemeris {
        #[arg(short, long, default_value = "data/ephemeris")]
        output: PathBuf,
    },

    /// Generate synthetic test stars
    GenerateSynthetic {
        #[arg(short, long, default_value = "100000")]
        count: usize,
        #[arg(short, long, default_value = "data/synthetic.csv")]
        output: PathBuf,
        #[arg(long, default_value = "42")]
        seed: u64,
    },

    /// Ingest star catalog into HLG
    IngestStars {
        #[arg(short, long)]
        input: Vec<PathBuf>,
        #[arg(short, long, default_value = "universe")]
        output: PathBuf,
        #[arg(long)]
        max_mag: Option<f64>,
        #[arg(long)]
        limit: Option<usize>,
    },

    /// Show planet positions at epoch
    Planets {
        #[arg(short, long, default_value = "2000-01-01T12:00:00 UTC")]
        epoch: String,
        #[arg(long, default_value = "data/ephemeris")]
        ephemeris_dir: PathBuf,
    },

    /// Build complete universe
    Build {
        #[arg(long)]
        stars: Vec<PathBuf>,
        #[arg(long)]
        synthetic: Option<usize>,
        /// Keep only stars brighter than this magnitude (smaller = brighter)
        #[arg(long)]
        max_mag: Option<f64>,
        /// Take the top N brightest stars after filtering (useful for quick builds)
        #[arg(long)]
        limit: Option<usize>,
        #[arg(long, default_value = "2000-01-01T12:00:00 UTC")]
        epoch: String,
        #[arg(long, default_value = "data/ephemeris")]
        ephemeris_dir: PathBuf,
        /// If DE440 is missing, download it automatically
        #[arg(long, default_value_t = true)]
        auto_fetch_ephemeris: bool,
        #[arg(short, long, default_value = "universe")]
        output: PathBuf,
    },

    /// Show planet positions from Keplerian propagation
    SimPlanets {
        /// Epoch (ISO format or "now")
        #[arg(short, long, default_value = "2000-01-01T12:00:00 UTC")]
        epoch: String,
    },

    /// Validate orbital propagation against ephemeris
    ValidateOrbits {
        /// Start epoch
        #[arg(long, default_value = "-100y")]
        start: String,
        /// End epoch
        #[arg(long, default_value = "+100y")]
        end: String,
        /// Step size (e.g., "30d", "1y")
        #[arg(long, default_value = "30d")]
        step: String,
        /// Ephemeris directory
        #[arg(long, default_value = "data/ephemeris")]
        ephemeris_dir: PathBuf,
    },

    /// Interactive time simulation (prints positions as time advances)
    TimeSim {
        /// Starting epoch
        #[arg(short, long, default_value = "2000-01-01T12:00:00 UTC")]
        epoch: String,
        /// Time rate (years per second)
        #[arg(long, default_value = "100")]
        rate: f64,
        /// Duration to simulate (seconds of real time)
        #[arg(long, default_value = "10")]
        duration: f64,
    },

    /// Run interactive renderer
    Render {
        /// Universe directory
        #[arg(short, long, default_value = "universe")]
        universe: PathBuf,
    },

    /// Pack cell binaries into a single file for fast loading
    ///
    /// This dramatically speeds up browser dataset mode by avoiding thousands of
    /// tiny HTTP requests (one per cell).
    Pack {
        /// Universe directory (contains index.json + cells/*.bin)
        #[arg(short, long, default_value = "universe")]
        universe: PathBuf,

        /// Output pack file name (written into the universe directory)
        #[arg(long, default_value = "cells.pack.bin")]
        pack_file: String,

        /// Output pack index JSON (written into the universe directory)
        #[arg(long, default_value = "cells.pack.json")]
        pack_index: String,
    },

    /// Run streaming server (browser-based remote viewing)
    Serve {
        /// HTTP server port
        #[arg(short, long, default_value = "7878")]
        port: u16,
        /// Universe dataset directory (contains index.json and cells/)
        #[arg(short, long, default_value = "universe")]
        universe: PathBuf,
        /// Stream width
        #[arg(long, default_value = "1280")]
        width: u32,
        /// Stream height
        #[arg(long, default_value = "720")]
        height: u32,
        /// Target FPS
        #[arg(long, default_value = "30")]
        fps: u32,
    },

    /// Train a single cell
    TrainCell {
        /// Input cell file
        #[arg(short, long)]
        input: PathBuf,
        /// Output cell file
        #[arg(short, long)]
        output: PathBuf,
        /// Training iterations
        #[arg(long, default_value = "500")]
        iterations: usize,
        /// Training backend: wgpu, torch-cuda, torch-cpu
        #[arg(long, default_value = "wgpu")]
        backend: String,
    },

    /// Train all cells in universe
    TrainAll {
        /// Input universe directory
        #[arg(short, long, default_value = "universe")]
        input: PathBuf,
        /// Output directory for trained cells
        #[arg(short, long, default_value = "trained")]
        output: PathBuf,
        /// Training iterations per cell
        #[arg(long, default_value = "500")]
        iterations: usize,
        /// Training backend: wgpu, torch-cuda, torch-cpu
        #[arg(long, default_value = "wgpu")]
        backend: String,
    },

    /// Train only cells near landmarks (selective training)
    TrainLandmarks {
        /// Input universe directory
        #[arg(short, long, default_value = "universe")]
        input: PathBuf,
        /// Output directory for trained cells
        #[arg(short, long, default_value = "trained")]
        output: PathBuf,
        /// Landmarks file
        #[arg(short, long, default_value = "universe/landmarks.json")]
        landmarks: PathBuf,
        /// Expand selection by N neighbor cells (in l/theta/phi directions)
        #[arg(short, long, default_value = "1")]
        neighbors: usize,
        /// Training iterations per cell
        #[arg(long, default_value = "500")]
        iterations: usize,
        /// Training backend: wgpu, torch-cuda, torch-cpu
        #[arg(long, default_value = "wgpu")]
        backend: String,
    },

    /// Generate landmarks.json from dataset (brightest/highest quality stars)
    GenerateLandmarks {
        /// Universe directory (contains index.json)
        #[arg(short, long, default_value = "universe")]
        universe: PathBuf,
        /// Input star catalog (CSV)
        #[arg(short, long)]
        stars: Option<PathBuf>,
        /// Number of landmarks to generate
        #[arg(short, long, default_value = "50")]
        count: usize,
        /// Output file
        #[arg(short, long, default_value = "universe/landmarks.json")]
        output: PathBuf,
        /// Selection method: "brightness" or "quality"
        #[arg(long, default_value = "brightness")]
        method: String,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct CellPackEntry {
    file_name: String,
    offset: u64,
    size: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct CellPackIndex {
    version: u32,
    pack_file: String,
    total_size_bytes: u64,
    cells: Vec<CellPackEntry>,
}

/// Landmark for client navigation (matches TypeScript schema)
#[derive(Clone, Debug, Serialize, Deserialize)]
struct LandmarkPos {
    x: f64,
    y: f64,
    z: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Landmark {
    id: String,
    name: String,
    kind: String,
    pos_meters: LandmarkPos,
    #[serde(skip_serializing_if = "Option::is_none")]
    radius_hint: Option<f64>,
    source: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let cli = Cli::parse();
    let grid = HLGGrid::with_defaults();

    match cli.command {
        Commands::GridTest { position, au } => {
            let coords: Vec<f64> = position
                .split(',')
                .map(|s| s.trim().parse::<f64>())
                .collect::<Result<Vec<_>, _>>()?;

            if coords.len() != 3 {
                anyhow::bail!("Position must be x,y,z");
            }

            let pos = if au {
                CartesianPosition::from_au(coords[0], coords[1], coords[2])
            } else {
                CartesianPosition::new(coords[0], coords[1], coords[2])
            };

            println!("Input Position:");
            println!("  Cartesian: ({:.6e}, {:.6e}, {:.6e}) m", pos.x, pos.y, pos.z);
            println!("  Distance:  {:.6e} m ({:.4} AU)", pos.magnitude(), pos.magnitude() / AU);

            let spherical = pos.to_spherical();
            println!("\nSpherical:");
            println!("  r:     {:.6e} m", spherical.r);
            println!("  θ:     {:.4} rad ({:.2}°)", spherical.theta, spherical.theta.to_degrees());
            println!("  φ:     {:.4} rad ({:.2}°)", spherical.phi, spherical.phi.to_degrees());

            if let Some(cell_id) = grid.cartesian_to_cell(pos) {
                println!("\nCell ID: ({}, {}, {})", cell_id.l, cell_id.theta, cell_id.phi);
                println!("File:    {}", cell_id.file_name());

                let r_inner = grid.shell_inner_radius(cell_id.l);
                let r_outer = grid.shell_outer_radius(cell_id.l);
                println!("\nShell {} Bounds:", cell_id.l);
                println!("  Inner: {:.6e} m ({:.4} AU)", r_inner, r_inner / AU);
                println!("  Outer: {:.6e} m ({:.4} AU)", r_outer, r_outer / AU);

                let bounds = grid.cell_to_bounds(cell_id);
                println!("\nCell AABB:");
                println!("  Min: ({:.6e}, {:.6e}, {:.6e})", bounds.min.x, bounds.min.y, bounds.min.z);
                println!("  Max: ({:.6e}, {:.6e}, {:.6e})", bounds.max.x, bounds.max.y, bounds.max.z);
                println!("  Centroid: ({:.6e}, {:.6e}, {:.6e})", bounds.centroid.x, bounds.centroid.y, bounds.centroid.z);
            } else {
                println!("\nPosition is inside minimum radius (r_min = {:.6e} m)", R_MIN);
            }
        }

        Commands::GridInfo => {
            let config = grid.config();
            println!("Heliocentric Logarithmic Grid Configuration:");
            println!("  r_min:    {:.6e} m ({:.4} AU)", config.r_min, config.r_min / AU);
            println!("  log_base: {}", config.log_base);
            println!("  n_theta:  {} divisions", config.n_theta);
            println!("  n_phi:    {} divisions", config.n_phi);
            println!("  Cells per shell: {}", config.n_theta * config.n_phi);

            println!("\nShell Examples:");
            for l in [0, 5, 10, 20, 30, 40, 50] {
                let r_inner = grid.shell_inner_radius(l);
                let r_outer = grid.shell_outer_radius(l);
                println!("  Shell {:2}: {:.2e} - {:.2e} m ({:.4} - {:.4} AU)",
                    l, r_inner, r_outer, r_inner / AU, r_outer / AU);
            }

            println!("\nReference Distances:");
            println!("  Mercury:        0.39 AU  -> Shell {}", grid.max_shell_for_distance(0.39 * AU));
            println!("  Earth:          1.00 AU  -> Shell {}", grid.max_shell_for_distance(1.0 * AU));
            println!("  Jupiter:        5.20 AU  -> Shell {}", grid.max_shell_for_distance(5.2 * AU));
            println!("  Neptune:       30.07 AU  -> Shell {}", grid.max_shell_for_distance(30.07 * AU));
            println!("  Voyager 1:    ~160 AU    -> Shell {}", grid.max_shell_for_distance(160.0 * AU));
            println!("  Proxima Cen:  ~4.24 ly   -> Shell {}", grid.max_shell_for_distance(4.24 * 9.461e15));
        }

        Commands::ListCells { from_au, to_au } => {
            let from_m = from_au * AU;
            let to_m = to_au * AU;

            let l_min = grid.max_shell_for_distance(from_m);
            let l_max = grid.max_shell_for_distance(to_m);

            println!("Cells from {:.2} AU to {:.2} AU:", from_au, to_au);
            println!("Shells {} to {} (inclusive)", l_min, l_max);

            let cells_per_shell = grid.config().n_theta * grid.config().n_phi;
            let total_cells = (l_max - l_min + 1) * cells_per_shell;

            println!("Total cells: {} ({} shells × {} cells/shell)",
                total_cells, l_max - l_min + 1, cells_per_shell);
        }

        Commands::FetchEphemeris { output } => {
            universe_data::download_de440(&output).await?;
        }

        Commands::GenerateSynthetic { count, output, seed } => {
            let stars = universe_data::generate_synthetic_stars(count, seed);
            std::fs::create_dir_all(output.parent().unwrap())?;
            let mut w = csv::Writer::from_path(&output)?;
            w.write_record(&["source_id","ra","dec","parallax","phot_g_mean_mag","bp_rp"])?;
            for s in &stars {
                w.write_record(&[
                    s.source_id.unwrap_or(0).to_string(),
                    s.ra.to_string(), s.dec.to_string(),
                    s.parallax.to_string(), s.phot_g_mean_mag.to_string(),
                    s.bp_rp.map(|x|x.to_string()).unwrap_or_default(),
                ])?;
            }
            w.flush()?;
            println!("Generated {} stars -> {:?}", count, output);
        }

        Commands::IngestStars { input, output, max_mag, limit } => {
            let paths: Vec<_> = input.iter().map(|p| p.as_path()).collect();
            let mut cat = universe_data::StarCatalog::load_multiple(&paths)?;
            if let Some(m) = max_mag { cat = cat.filter_by_magnitude(m); }
            if let Some(n) = limit { cat = cat.take_brightest(n); }
            let pipe = universe_data::DataPipeline::with_defaults();
            let manifest = pipe.ingest_stars(&cat, &output)?;
            println!("Done: {} cells, {} splats", manifest.cells.len(), manifest.total_splats);
        }

        Commands::Planets { epoch, ephemeris_dir } => {
            let epoch = Epoch::from_str(&epoch)?;
            let eph = universe_data::EphemerisProvider::load_default(&ephemeris_dir)?;
            println!("{:<12} {:>15} {:>15} {:>15} {:>12}", "Body", "X (AU)", "Y (AU)", "Z (AU)", "Dist (AU)");
            for body in universe_data::SolarSystemBody::all() {
                let p = eph.body_position(*body, epoch)?;
                println!("{:<12} {:>15.6} {:>15.6} {:>15.6} {:>12.4}",
                    body.name(), p.x/AU, p.y/AU, p.z/AU, p.magnitude()/AU);
            }
        }

        Commands::Build { stars, synthetic, max_mag, limit, epoch, ephemeris_dir, auto_fetch_ephemeris, output } => {
            std::fs::create_dir_all(&output)?;
            let pipe = universe_data::DataPipeline::with_defaults();
            let epoch = Epoch::from_str(&epoch)?;

            // Ensure ephemeris is present (DE440)
            if auto_fetch_ephemeris {
                universe_data::download_de440(&ephemeris_dir).await?;
            }

            // Stars
            let star_man = if !stars.is_empty() {
                let paths: Vec<_> = stars.iter().map(|p|p.as_path()).collect();
                let mut cat = universe_data::StarCatalog::load_multiple(&paths)?;
                if let Some(m) = max_mag { cat = cat.filter_by_magnitude(m); }
                if let Some(n) = limit { cat = cat.take_brightest(n); }
                Some(pipe.ingest_stars(&cat, &output)?)
            } else if let Some(n) = synthetic {
                let cat = universe_data::StarCatalog::from_records(universe_data::generate_synthetic_stars(n, 42));
                Some(pipe.ingest_stars(&cat, &output)?)
            } else { None };

            // Planets
            let eph = universe_data::EphemerisProvider::load_default(&ephemeris_dir)?;
            let planet_man = pipe.generate_planets(&eph, epoch, &output)?;

            // Landmarks (optional - load if landmarks.json exists in output dir)
            let landmarks_path = output.join("landmarks.json");
            let landmark_man = if landmarks_path.exists() {
                println!("Loading landmarks from {:?}", landmarks_path);
                let landmarks = universe_data::load_landmarks_json(&landmarks_path)?;
                println!("Ingesting {} landmarks", landmarks.len());
                Some(pipe.ingest_landmarks(&landmarks, &output)?)
            } else {
                println!("No landmarks.json found, skipping landmark ingestion");
                None
            };

            // Merge
            let mut all = vec![planet_man];
            if let Some(m) = star_man { all.push(m); }
            if let Some(m) = landmark_man { all.push(m); }
            let final_man = universe_data::merge_manifests(all)?;
            final_man.save(&output.join("index.json"))?;

            println!("Universe built: {} cells, {} splats, {:.1} MB",
                final_man.cells.len(), final_man.total_splats, final_man.total_size_bytes as f64/1e6);
        }

        Commands::SimPlanets { epoch } => {
            let epoch = Epoch::from_str(&epoch)?;
            let system = universe_sim::SolarSystem::at_epoch(epoch);

            println!("Keplerian positions at {}:", epoch);
            println!("{:<12} {:>15} {:>15} {:>15} {:>12}", "Body", "X (AU)", "Y (AU)", "Z (AU)", "Dist (AU)");

            for body in universe_sim::Body::all() {
                let pos = system.body_position(*body);
                println!("{:<12} {:>15.6} {:>15.6} {:>15.6} {:>12.4}",
                    body.name(), pos.x/AU, pos.y/AU, pos.z/AU, pos.magnitude()/AU);
            }
        }

        Commands::ValidateOrbits { start, end, step, ephemeris_dir } => {
            let ephemeris = universe_data::EphemerisProvider::load_default(&ephemeris_dir)?;
            let mut system = universe_sim::SolarSystem::new();

            // Parse relative times like "-100y", "+100y"
            let j2000 = Epoch::from_gregorian_utc(2000, 1, 1, 12, 0, 0, 0);
            let start_epoch = parse_relative_epoch(&start, j2000)?;
            let end_epoch = parse_relative_epoch(&end, j2000)?;
            let step_dur = parse_duration(&step)?;

            println!("Validating from {} to {} with step {}", start_epoch, end_epoch, step);

            let results = universe_sim::validate_range(&mut system, &ephemeris, start_epoch, end_epoch, step_dur)?;
            let summary = universe_sim::summarize_validation(&results);

            println!("\n{:<12} {:>10} {:>15} {:>15} {:>15}", "Body", "Points", "Mean (km)", "Max (km)", "Mean (%)");
            for s in &summary {
                println!("{:<12} {:>10} {:>15.1} {:>15.1} {:>15.4}",
                    s.body.name(), s.num_points, s.mean_error_km, s.max_error_km, s.mean_error_percent);
            }
        }

        Commands::TimeSim { epoch, rate, duration } => {
            let epoch = Epoch::from_str(&epoch)?;
            let mut tc = universe_sim::TimeController::at_epoch(epoch);
            tc.set_rate_years_per_second(rate);

            let mut system = universe_sim::SolarSystem::at_epoch(epoch);

            let steps = (duration * 10.0) as usize; // 10 updates per second
            let dt = 0.1;

            for _ in 0..steps {
                let current = tc.tick(dt);
                system.set_epoch(current);

                let earth = system.body_position(universe_sim::Body::Earth);
                let mars = system.body_position(universe_sim::Body::Mars);

                // Distance Earth-Mars
                let dx = earth.x - mars.x;
                let dy = earth.y - mars.y;
                let dz = earth.z - mars.z;
                let dist_au = (dx*dx + dy*dy + dz*dz).sqrt() / AU;

                println!("Year {:.1}: Earth-Mars distance = {:.3} AU", tc.year(), dist_au);

                std::thread::sleep(std::time::Duration::from_millis(100));
            }
        }

        Commands::Render { universe } => {
            println!("Starting Universe renderer...");
            println!("Controls:");
            println!("  WASD - Move camera");
            println!("  Mouse - Look around (click to capture)");
            println!("  Space/Shift - Up/Down");
            println!("  Q/E - Slow/Fast movement");
            println!("  P - Pause/Resume time");
            println!("  ,/. - Slower/Faster time");
            println!("  1 - Go to Earth");
            println!("  2 - Go to Mars");
            println!("  0 - Solar system overview");
            println!("  Escape - Release mouse / Exit");

            universe_render::run(&universe)?;
        }

        Commands::Pack { universe, pack_file, pack_index } => {
            let manifest_path = universe.join("index.json");
            let manifest = universe_data::CellManifest::load(&manifest_path)?;

            // Prefer `universe/cells/`, fall back to legacy flat layout
            let preferred_cells_dir = universe.join("cells");
            let cells_dir = if preferred_cells_dir.is_dir() {
                preferred_cells_dir
            } else {
                universe.clone()
            };

            let pack_path = universe.join(&pack_file);
            let index_path = universe.join(&pack_index);

            println!("Packing {} cells…", manifest.cells.len());
            println!("  Cells dir: {}", cells_dir.display());
            println!("  Pack file: {}", pack_path.display());
            println!("  Index:     {}", index_path.display());

            let mut out = std::io::BufWriter::new(std::fs::File::create(&pack_path)?);
            let mut offset: u64 = 0;
            let mut entries: Vec<CellPackEntry> = Vec::with_capacity(manifest.cells.len());

            for cell in &manifest.cells {
                let path = cells_dir.join(&cell.file_name);
                let bytes = std::fs::read(&path)?;
                let size = bytes.len() as u64;

                use std::io::Write as _;
                out.write_all(&bytes)?;

                entries.push(CellPackEntry {
                    file_name: cell.file_name.clone(),
                    offset,
                    size,
                });

                offset += size;
            }

            out.flush()?;

            let index = CellPackIndex {
                version: 1,
                pack_file,
                total_size_bytes: offset,
                cells: entries,
            };
            std::fs::write(&index_path, serde_json::to_string_pretty(&index)?)?;

            println!("Done: {:.2} MB packed", offset as f64 / 1e6);
        }

        Commands::Serve { port, universe, width, height, fps } => {
            println!("Starting Universe streaming server...");
            println!("  Port: {}", port);
            println!("  Universe: {}", universe.display());
            println!("  Resolution: {}x{} @ {} FPS", width, height, fps);
            println!("  URL: http://localhost:{}", port);
            println!();
            println!("Open http://localhost:{} in your browser to view the stream", port);

            let config = universe_stream::StreamConfig {
                port,
                width,
                height,
                fps,
                universe,
            };

            universe_stream::run_server(config).await?;
        }

        Commands::TrainCell { input, output, iterations, backend } => {
            let config = universe_train::TrainConfig {
                iterations,
                ..Default::default()
            };

            match backend.as_str() {
                "wgpu" => {
                    use burn_autodiff::Autodiff;
                    use burn_wgpu::{Wgpu, WgpuDevice};

                    type Backend = Autodiff<Wgpu>;
                    let device = WgpuDevice::default();

                    let cell = universe_data::CellData::load(&input)?;
                    let trainer = universe_train::Trainer::<Backend>::new(config, device);
                    let trained = trainer.train_cell(&cell)?;

                    let mut out_cell = universe_data::CellData::new(cell.metadata.id, cell.metadata.bounds);
                    for s in trained {
                        out_cell.add_splat(s);
                    }
                    out_cell.save(&output)?;
                }
                #[cfg(feature = "torch")]
                "torch-cuda" | "torch-cpu" => {
                    use universe_train::torch_backend::{TorchDevice, TorchTrainer};

                    let device = if backend == "torch-cuda" {
                        TorchDevice::cuda_if_available()
                    } else {
                        TorchDevice::Cpu
                    };

                    let cell = universe_data::CellData::load(&input)?;
                    let trainer = TorchTrainer::new(config, device)?;
                    let trained = trainer.train_cell(&cell)?;

                    let mut out_cell = universe_data::CellData::new(cell.metadata.id, cell.metadata.bounds);
                    for s in trained {
                        out_cell.add_splat(s);
                    }
                    out_cell.save(&output)?;
                }
                #[cfg(not(feature = "torch"))]
                "torch-cuda" | "torch-cpu" => {
                    anyhow::bail!("Torch backend not available. Rebuild with --features torch");
                }
                _ => {
                    anyhow::bail!("Unknown backend '{}'. Use: wgpu, torch-cuda, torch-cpu", backend);
                }
            }

            println!("Trained {} -> {}", input.display(), output.display());
        }

        Commands::TrainAll { input, output, iterations, backend } => {
            let config = universe_train::TrainConfig {
                iterations,
                ..Default::default()
            };

            match backend.as_str() {
                "wgpu" => {
                    use burn_autodiff::Autodiff;
                    use burn_wgpu::{Wgpu, WgpuDevice};

                    type Backend = Autodiff<Wgpu>;
                    let device = WgpuDevice::default();

                    universe_train::train_universe::<Backend>(&input, &output, config, device)?;
                }
                #[cfg(feature = "torch")]
                "torch-cuda" | "torch-cpu" => {
                    use universe_train::torch_backend::{train_universe, TorchDevice};

                    let device = if backend == "torch-cuda" {
                        TorchDevice::cuda_if_available()
                    } else {
                        TorchDevice::Cpu
                    };

                    train_universe(&input, &output, config, device)?;
                }
                #[cfg(not(feature = "torch"))]
                "torch-cuda" | "torch-cpu" => {
                    anyhow::bail!("Torch backend not available. Rebuild with --features torch");
                }
                _ => {
                    anyhow::bail!("Unknown backend '{}'. Use: wgpu, torch-cuda, torch-cpu", backend);
                }
            }
        }

        Commands::TrainLandmarks { input, output, landmarks, neighbors, iterations, backend } => {
            use std::collections::HashSet;
            use universe_core::grid::{HLGGrid, CellId};
            use universe_core::coordinates::CartesianPosition;

            println!("Loading landmarks from {:?}", landmarks);
            let landmark_list = universe_data::load_landmarks_json(&landmarks)?;
            println!("Loaded {} landmarks", landmark_list.len());

            // Load manifest to get HLG config and existing cells
            let manifest_path = input.join("index.json");
            let manifest = universe_data::CellManifest::load(&manifest_path)?;
            let grid = HLGGrid::new(manifest.config.clone());

            println!("Universe has {} cells total", manifest.cells.len());

            // Find cells containing landmarks
            let mut selected_cells = HashSet::new();
            for landmark in &landmark_list {
                let pos = CartesianPosition {
                    x: landmark.pos_meters.x,
                    y: landmark.pos_meters.y,
                    z: landmark.pos_meters.z,
                };

                if let Some(cell_id) = grid.cartesian_to_cell(pos) {
                    selected_cells.insert(cell_id);
                }
            }

            println!("Found {} cells containing landmarks", selected_cells.len());

            // Expand selection by neighbors
            if neighbors > 0 {
                let mut expanded = selected_cells.clone();
                for &cell_id in &selected_cells {
                    // Expand in l (radial), theta (azimuth), phi (polar) directions
                    for dl in -(neighbors as i32)..=(neighbors as i32) {
                        for dtheta in -(neighbors as i32)..=(neighbors as i32) {
                            for dphi in -(neighbors as i32)..=(neighbors as i32) {
                                let new_l = (cell_id.l as i32 + dl).max(0) as usize;
                                let new_theta = (cell_id.theta as i32 + dtheta).rem_euclid(manifest.config.n_theta as i32) as usize;
                                let new_phi = (cell_id.phi as i32 + dphi).clamp(0, (manifest.config.n_phi - 1) as i32) as usize;

                                let neighbor = CellId {
                                    l: new_l as u32,
                                    theta: new_theta as u32,
                                    phi: new_phi as u32,
                                };

                                // Only add if cell exists in manifest
                                if manifest.cells.iter().any(|e| e.id == neighbor) {
                                    expanded.insert(neighbor);
                                }
                            }
                        }
                    }
                }
                println!("Expanded to {} cells (neighbors={})", expanded.len(), neighbors);
                selected_cells = expanded;
            }

            // Filter to only cells that exist in the manifest
            let cells_to_train: Vec<_> = manifest.cells.iter()
                .filter(|e| selected_cells.contains(&e.id))
                .cloned()
                .collect();

            println!("Training {} cells (out of {} total)", cells_to_train.len(), manifest.cells.len());

            // Train selected cells
            let config = universe_train::TrainConfig {
                iterations,
                ..Default::default()
            };

            match backend.as_str() {
                "wgpu" => {
                    use burn_autodiff::Autodiff;
                    use burn_wgpu::{Wgpu, WgpuDevice};

                    type Backend = Autodiff<Wgpu>;
                    let device = WgpuDevice::default();

                    universe_train::train_selected_cells::<Backend>(&input, &output, &cells_to_train, config, device)?;
                }
                #[cfg(feature = "torch")]
                "torch-cuda" | "torch-cpu" => {
                    use universe_train::torch_backend::{train_selected_cells, TorchDevice};

                    let device = if backend == "torch-cuda" {
                        TorchDevice::cuda_if_available()
                    } else {
                        TorchDevice::Cpu
                    };

                    train_selected_cells(&input, &output, &cells_to_train, config, device)?;
                }
                #[cfg(not(feature = "torch"))]
                "torch-cuda" | "torch-cpu" => {
                    anyhow::bail!("Torch backend not available. Rebuild with --features torch");
                }
                _ => {
                    anyhow::bail!("Unknown backend '{}'. Use: wgpu, torch-cuda, torch-cpu", backend);
                }
            }

            println!("Landmark-focused training complete!");
        }

        Commands::GenerateLandmarks { universe, stars, count, output, method } => {
            println!("Generating {} landmarks using {} method...", count, method);

            let mut landmarks: Vec<Landmark> = Vec::new();

            // If star catalog is provided, generate landmarks from it
            if let Some(stars_path) = stars {
                let cat = universe_data::StarCatalog::load_gaia_csv(&stars_path)?;

                // Collect stars with valid parallax for sorting
                let mut sorted: Vec<_> = cat.iter()
                    .filter(|r| r.parallax > 0.1)
                    .collect();

                // Sort by brightness (lower magnitude = brighter) or quality
                match method.as_str() {
                    "quality" => {
                        // Quality score: parallax / estimated_error (higher = better)
                        sorted.sort_by(|a, b| {
                            let qa = a.parallax / (a.parallax * 0.1).max(0.01);
                            let qb = b.parallax / (b.parallax * 0.1).max(0.01);
                            qb.partial_cmp(&qa).unwrap_or(std::cmp::Ordering::Equal)
                        });
                    }
                    _ => {
                        // Default: brightness (lower magnitude = brighter)
                        sorted.sort_by(|a, b| {
                            a.phot_g_mean_mag.partial_cmp(&b.phot_g_mean_mag).unwrap_or(std::cmp::Ordering::Equal)
                        });
                    }
                }

                // Take top N
                for (i, record) in sorted.iter().take(count).enumerate() {
                    if let Some(pos) = record.to_cartesian() {
                        let distance_pc = 1000.0 / record.parallax.max(0.1);
                        let dist_ly = distance_pc * 3.26156;

                        landmarks.push(Landmark {
                            id: format!("star-{}", record.source_id.unwrap_or(i as u64)),
                            name: format!("Star {} (Gaia {})", i + 1, record.source_id.unwrap_or(0)),
                            kind: "star".to_string(),
                            pos_meters: LandmarkPos {
                                x: pos.x,
                                y: pos.y,
                                z: pos.z,
                            },
                            radius_hint: None,
                            source: "ml".to_string(),
                            description: Some(format!("Mag {:.1}, {:.1} ly", record.phot_g_mean_mag, dist_ly)),
                        });
                    }
                }
            } else {
                // No star catalog: generate simple landmarks from manifest cells
                let manifest_path = universe.join("index.json");
                if manifest_path.exists() {
                    let manifest = universe_data::CellManifest::load(&manifest_path)?;

                    // Sample cells at different distances
                    let mut cells_by_shell: std::collections::HashMap<u32, Vec<_>> = std::collections::HashMap::new();
                    for cell in &manifest.cells {
                        cells_by_shell.entry(cell.id.l).or_default().push(cell);
                    }

                    // Pick representative cells from different shells
                    let mut shell_keys: Vec<_> = cells_by_shell.keys().copied().collect();
                    shell_keys.sort();

                    for &shell in shell_keys.iter().take(count) {
                        if let Some(cells) = cells_by_shell.get(&shell) {
                            if let Some(cell) = cells.first() {
                                // Compute centroid from cell bounds
                                let bounds = grid.cell_to_bounds(cell.id);
                                let r_inner = grid.shell_inner_radius(shell);
                                let r_outer = grid.shell_outer_radius(shell);
                                let r_mid = (r_inner + r_outer) / 2.0;
                                let dist_ly = r_mid / 9.461e15;

                                landmarks.push(Landmark {
                                    id: format!("shell-{}", shell),
                                    name: format!("Shell {} Region", shell),
                                    kind: "region".to_string(),
                                    pos_meters: LandmarkPos {
                                        x: bounds.centroid.x,
                                        y: bounds.centroid.y,
                                        z: bounds.centroid.z,
                                    },
                                    radius_hint: Some(r_outer - r_inner),
                                    source: "ml".to_string(),
                                    description: Some(format!("{:.1} - {:.1} ly", r_inner / 9.461e15, dist_ly * 2.0 - r_inner / 9.461e15)),
                                });
                            }
                        }
                    }
                }
            }

            // Write output
            std::fs::create_dir_all(output.parent().unwrap())?;
            let json = serde_json::to_string_pretty(&landmarks)?;
            std::fs::write(&output, json)?;

            println!("Generated {} landmarks -> {}", landmarks.len(), output.display());
        }
    }

    Ok(())
}

/// Parse relative epoch like "-100y", "+100y" relative to reference
fn parse_relative_epoch(s: &str, reference: Epoch) -> Result<Epoch> {
    use hifitime::Duration;
    if s.ends_with('y') {
        let years: f64 = s.trim_end_matches('y').parse()?;
        Ok(reference + Duration::from_days(years * 365.25))
    } else {
        Ok(Epoch::from_str(s)?)
    }
}

/// Parse duration like "30d", "1y"
fn parse_duration(s: &str) -> Result<hifitime::Duration> {
    use hifitime::Duration;
    if s.ends_with('d') {
        let days: f64 = s.trim_end_matches('d').parse()?;
        Ok(Duration::from_days(days))
    } else if s.ends_with('y') {
        let years: f64 = s.trim_end_matches('y').parse()?;
        Ok(Duration::from_days(years * 365.25))
    } else {
        Ok(Duration::from_seconds(s.parse()?))
    }
}
