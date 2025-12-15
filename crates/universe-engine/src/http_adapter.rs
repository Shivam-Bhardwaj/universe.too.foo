//! Phase 2.2: Tile fetch adapters
//!
//! Native viewer reads tiles from disk.
//! WASM/web viewer fetches tiles over HTTP (async).

use anyhow::Result;
use universe_core::grid::CellId;
use universe_data::{CellData, CellManifest};

// -----------------------------
// Native (disk) adapter
// -----------------------------

/// Platform-agnostic tile fetcher trait (native only).
///
/// Note: For the web target we use async fetch APIs, so we expose a separate
/// async interface instead of forcing async_trait into the core crate.
#[cfg(not(target_arch = "wasm32"))]
pub trait TileFetcher: Send + Sync {
    fn fetch_manifest(&self, universe_dir: &std::path::Path) -> Result<CellManifest>;
    fn fetch_cell(&self, universe_dir: &std::path::Path, cell_id: CellId) -> Result<CellData>;
}

/// Native file-based tile fetcher (for reference viewer)
#[cfg(not(target_arch = "wasm32"))]
pub struct NativeFileFetcher;

#[cfg(not(target_arch = "wasm32"))]
impl TileFetcher for NativeFileFetcher {
    fn fetch_manifest(&self, universe_dir: &std::path::Path) -> Result<CellManifest> {
        let manifest_path = universe_dir.join("index.json");
        CellManifest::load(&manifest_path)
    }

    fn fetch_cell(&self, universe_dir: &std::path::Path, cell_id: CellId) -> Result<CellData> {
        let file_name = cell_id.file_name();
        let cell_path = universe_dir.join("cells").join(&file_name);
        CellData::load(&cell_path)
    }
}

// -----------------------------
// WASM (HTTP) adapter
// -----------------------------

/// WASM HTTP tile fetcher.
///
/// This is intentionally async because browser fetch is async.
#[cfg(target_arch = "wasm32")]
pub struct WasmHttpFetcher;

#[cfg(target_arch = "wasm32")]
impl WasmHttpFetcher {
    pub async fn fetch_manifest(base_url: &str) -> Result<CellManifest> {
        let base = base_url.trim_end_matches('/');
        let url = format!("{}/index.json", base);

        let text = fetch_text(&url).await?;
        Ok(serde_json::from_str::<CellManifest>(&text)?)
    }

    pub async fn fetch_cell(base_url: &str, cell_id: CellId) -> Result<CellData> {
        let base = base_url.trim_end_matches('/');
        let file_name = cell_id.file_name();
        let url = format!("{}/cells/{}", base, file_name);

        let bytes = fetch_bytes(&url).await?;
        let mut cursor = std::io::Cursor::new(bytes);
        CellData::deserialize(&mut cursor)
    }

    pub async fn fetch_cell_file(base_url: &str, file_name: &str) -> Result<CellData> {
        let base = base_url.trim_end_matches('/');
        let url = format!("{}/cells/{}", base, file_name);

        let bytes = fetch_bytes(&url).await?;
        let mut cursor = std::io::Cursor::new(bytes);
        CellData::deserialize(&mut cursor)
    }
}

#[cfg(target_arch = "wasm32")]
fn js_err(e: wasm_bindgen::JsValue) -> anyhow::Error {
    anyhow::anyhow!("{:?}", e)
}

#[cfg(target_arch = "wasm32")]
async fn fetch_text(url: &str) -> Result<String> {
    use wasm_bindgen::JsCast;
    use wasm_bindgen_futures::JsFuture;

    let win = web_sys::window().ok_or_else(|| anyhow::anyhow!("no window"))?;
    let resp_val = JsFuture::from(win.fetch_with_str(url))
        .await
        .map_err(js_err)?;
    let resp: web_sys::Response = resp_val.dyn_into().map_err(js_err)?;

    if !resp.ok() {
        anyhow::bail!("HTTP {} {}", resp.status(), resp.status_text());
    }

    let text_promise = resp.text().map_err(js_err)?;
    let text_val = JsFuture::from(text_promise).await.map_err(js_err)?;
    text_val
        .as_string()
        .ok_or_else(|| anyhow::anyhow!("response.text() was not a string"))
}

#[cfg(target_arch = "wasm32")]
async fn fetch_bytes(url: &str) -> Result<Vec<u8>> {
    use wasm_bindgen::JsCast;
    use wasm_bindgen_futures::JsFuture;

    let win = web_sys::window().ok_or_else(|| anyhow::anyhow!("no window"))?;
    let resp_val = JsFuture::from(win.fetch_with_str(url))
        .await
        .map_err(js_err)?;
    let resp: web_sys::Response = resp_val.dyn_into().map_err(js_err)?;

    if !resp.ok() {
        anyhow::bail!("HTTP {} {}", resp.status(), resp.status_text());
    }

    let buf_promise = resp.array_buffer().map_err(js_err)?;
    let buf_val = JsFuture::from(buf_promise).await.map_err(js_err)?;
    let u8 = js_sys::Uint8Array::new(&buf_val);
    let mut out = vec![0u8; u8.length() as usize];
    u8.copy_to(&mut out);
    Ok(out)
}



