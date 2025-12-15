pub mod camera;
pub mod entropy_model;
pub mod ground_truth;
pub mod loss;
pub mod model;
pub mod rasterizer;
pub mod trainer;

pub use camera::Camera;
pub use model::GaussianCloud;
pub use trainer::{train_universe, TrainConfig, Trainer};
