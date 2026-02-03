use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use anyhow::Result;
use universe_core::chebyshev::ChebyshevPropagator;

/// Load spacecraft trajectories from JSON file
pub fn load_spacecraft_json<P: AsRef<Path>>(path: P) -> Result<Vec<ChebyshevPropagator>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let propagators: Vec<ChebyshevPropagator> = serde_json::from_reader(reader)?;
    Ok(propagators)
}


