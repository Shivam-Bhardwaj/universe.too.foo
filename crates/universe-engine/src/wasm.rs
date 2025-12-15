//! Phase 2.1: WASM bindings for web viewer

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub struct WasmEngine {
    core: crate::renderer::RendererCore,
    policy: crate::streaming::StreamingPolicy,
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl WasmEngine {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            core: crate::renderer::RendererCore::new(),
            policy: crate::streaming::StreamingPolicy::new(None),
        }
    }

    #[wasm_bindgen]
    pub fn update_camera(&mut self, dt: f64) {
        self.core.update_camera(dt);
    }

    #[wasm_bindgen]
    pub fn get_camera_uniform(&self, aspect: f32) -> Vec<f32> {
        let uniform = self.core.camera_uniform(aspect);
        bytemuck::cast_slice(&[uniform]).to_vec()
    }
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub struct WasmCell {
    centroid_x: f64,
    centroid_y: f64,
    centroid_z: f64,
    splats: Vec<f32>,
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl WasmCell {
    #[wasm_bindgen(getter)]
    pub fn centroid_x(&self) -> f64 {
        self.centroid_x
    }

    #[wasm_bindgen(getter)]
    pub fn centroid_y(&self) -> f64 {
        self.centroid_y
    }

    #[wasm_bindgen(getter)]
    pub fn centroid_z(&self) -> f64 {
        self.centroid_z
    }

    /// Returns splat data as a flat f32 array (14 floats per splat).
    /// Layout matches `universe-data::GaussianSplat` (pos[3], scale[3], rot[4], color[3], opacity[1]).
    pub fn splats(&self) -> Vec<f32> {
        self.splats.clone()
    }
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub async fn fetch_cell(base_url: String, file_name: String) -> Result<WasmCell, JsValue> {
    let cell = crate::http_adapter::WasmHttpFetcher::fetch_cell_file(&base_url, &file_name)
        .await
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let c = cell.metadata.bounds.centroid;
    let splats: Vec<f32> = bytemuck::cast_slice(&cell.splats).to_vec();

    Ok(WasmCell {
        centroid_x: c.x,
        centroid_y: c.y,
        centroid_z: c.z,
        splats,
    })
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn parse_cell(bytes: js_sys::Uint8Array) -> Result<WasmCell, JsValue> {
    let v = bytes.to_vec();
    let mut cursor = std::io::Cursor::new(v);
    let cell = universe_data::CellData::deserialize(&mut cursor)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let c = cell.metadata.bounds.centroid;
    let splats: Vec<f32> = bytemuck::cast_slice(&cell.splats).to_vec();

    Ok(WasmCell {
        centroid_x: c.x,
        centroid_y: c.y,
        centroid_z: c.z,
        splats,
    })
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(start)]
pub fn init() {
    console_error_panic_hook::set_once();
}
