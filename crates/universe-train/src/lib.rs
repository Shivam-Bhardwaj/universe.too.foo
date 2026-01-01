pub mod camera;
pub mod entropy_model;
pub mod ground_truth;
pub mod loss;
pub mod model;
pub mod rasterizer;
pub mod trainer;

#[cfg(feature = "torch")]
pub mod torch_backend;

pub use camera::Camera;
pub use model::GaussianCloud;
pub use trainer::{train_universe, train_selected_cells, TrainConfig, Trainer};
