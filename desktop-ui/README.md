# ValveGuard Desktop UI

Tauri v2 desktop shell for the interactive review app. Rust backend,
WebView2 frontend, no local HTTP server in production.

## Run

From this directory:

```bash
npm install
npm run tauri dev
```

Dev frontend is fixed to `127.0.0.1:47183` with `strictPort`. Production
uses the Tauri custom protocol.

The Rust command `score_demo` calls the shared `valveguard-core` crate:
`awpy` parses the demo, `channels.rs` builds the 29 aim channels, and
`ort` runs `artifacts/valveguard_cnn_v1_onnx/model.onnx`. No Python is
involved.

## Build

```bash
npm run build
npm run tauri build
```

The portable release zip contains `ValveGuard.exe` and a `model/` folder
with the ONNX bundle. The app checks that folder first, then the Tauri
resource directory, then the repo `artifacts/` fallback.

## Layout

```text
index.html
vite.config.js            port 47183
src/main.js               invoke + dialog + drag-drop UI
src/styles.css
src-tauri/
  Cargo.toml
  tauri.conf.json
  capabilities/default.json
  src/main.rs             health, score_demo
  icons/
```
