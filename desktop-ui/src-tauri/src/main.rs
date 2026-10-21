#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use serde_json::{json, Value};
use std::fs;
use std::path::{Path, PathBuf};
use tauri::Manager;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|path| path.parent())
        .expect("src-tauri must live under the repository")
        .to_path_buf()
}

fn model_path(app: &tauri::AppHandle) -> PathBuf {
    if let Ok(path) = std::env::var("VALVEGUARD_MODEL") {
        return PathBuf::from(path);
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let portable = dir.join("model/model.onnx");
            if portable.exists() {
                return portable;
            }
        }
    }
    if let Ok(dir) = app.path().resource_dir() {
        let bundled = dir.join("model/model.onnx");
        if bundled.exists() {
            return bundled;
        }
    }
    let root = repo_root();
    let onnx = root.join("artifacts/valveguard_cnn_v1_onnx/model.onnx");
    if onnx.exists() {
        return onnx;
    }
    root.join("artifacts/valveguard_cnn_v1/model.onnx")
}

#[tauri::command]
async fn score_demo(app: tauri::AppHandle, path: String) -> Result<Value, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let model = model_path(&app);
        if !model.exists() {
            return Err(format!("ONNX model not found: {}", model.display()));
        }
        valveguard_core::app::score_demo(Path::new(&path), &model)
            .map_err(|error| format!("{error:#}"))
    })
    .await
    .map_err(|error| format!("scoring task failed: {error}"))?
}

fn records_path(app: &tauri::AppHandle) -> Result<PathBuf, String> {
    let dir = app
        .path()
        .document_dir()
        .map_err(|error| format!("locate Documents: {error}"))?
        .join("ValveGuard");
    Ok(dir.join("records.json"))
}

#[tauri::command]
fn load_records(app: tauri::AppHandle) -> Result<Value, String> {
    let path = records_path(&app)?;
    if !path.exists() {
        return Ok(json!([]));
    }
    let text = fs::read_to_string(&path)
        .map_err(|error| format!("read {}: {error}", path.display()))?;
    let records: Value =
        serde_json::from_str(&text).map_err(|error| format!("parse {}: {error}", path.display()))?;
    if records.is_array() {
        Ok(records)
    } else {
        Ok(json!([]))
    }
}

#[tauri::command]
fn save_record(app: tauri::AppHandle, record: Value) -> Result<(), String> {
    let path = records_path(&app)?;
    let id = record
        .get("id")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    if id.is_empty() {
        return Err("record has no id".to_string());
    }

    let mut records: Vec<Value> = if path.exists() {
        let text = fs::read_to_string(&path)
            .map_err(|error| format!("read {}: {error}", path.display()))?;
        serde_json::from_str(&text).unwrap_or_default()
    } else {
        Vec::new()
    };

    records.retain(|item| item.get("id").and_then(Value::as_str) != Some(id.as_str()));
    records.insert(0, record);
    records.truncate(50);

    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir)
            .map_err(|error| format!("create {}: {error}", dir.display()))?;
    }
    let text = serde_json::to_string_pretty(&records)
        .map_err(|error| format!("encode records: {error}"))?;
    fs::write(&path, text).map_err(|error| format!("write {}: {error}", path.display()))
}

#[tauri::command]
fn health(app: tauri::AppHandle) -> Result<Value, String> {
    let model = model_path(&app);
    let manifest_path = model
        .parent()
        .ok_or_else(|| "model has no parent directory".to_string())?
        .join("manifest.json");
    let text = std::fs::read_to_string(&manifest_path)
        .map_err(|error| format!("read manifest {}: {error}", manifest_path.display()))?;
    let manifest: Value =
        serde_json::from_str(&text).map_err(|error| format!("parse manifest: {error}"))?;
    Ok(json!({
        "ok": true,
        "model": manifest.get("version").and_then(Value::as_str).unwrap_or("unknown"),
        "threshold": manifest
            .get("threshold")
            .and_then(|value| value.get("value"))
            .and_then(Value::as_f64)
            .unwrap_or(0.0),
        "runtime": "rust-ort",
    }))
}

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![
            health,
            score_demo,
            load_records,
            save_record
        ])
        .run(tauri::generate_context!())
        .expect("error while running ValveGuard");
}
