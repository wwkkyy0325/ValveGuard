# Remote Training Workspace

Status: active (2026-10-20)

## Host

- LAN address `192.168.3.143`, user `administrator`, host
  `DESKTOP-BC3OC2K`.
- OpenSSH server on port 22 with password authentication. The password is not
  stored in this repository. No SSH key was installed.
- Hardware: Intel i9-14900KF (32 threads), 48 GB RAM, NVIDIA RTX 5070 Ti
  (16 GB, compute capability 12.0), drives C/D/E with about 2.9 TB free on E.

## Workspace

Everything for the training box lives under `E:\ValveGuardWork`:

```
E:\ValveGuardWork\
  README.txt                  layout and cleanup note
  python312\                  Python 3.12.10 per-user install
  venv\                       project virtualenv, editable valveguard install
  repo\ValveGuard\            source, docs, artifacts, holdout split
  data\processed\cw256\      90,707 x 256 x 44 window cache
  data\processed\cw256_channels\  43-channel tensors
  data\processed\raw1024\    1024-tick pilot features
  data\processed\mouse_resid_subset\
  data\processed\pro_windows\
  data\processed\5e_windows\
  data\processed\events_cs2cd\  63 raw event features, 20,252 kill windows
  data\processed\events_cs2cd_400\  200+200 match event features
  data\processed\events_cs2cd_full\ 78,202 windows / 701 matches
  data\raw\5e_sample\        three 5E demos
  data\raw\cs2cd_subset\     748 matches (1,496 files, ~43.5 GB)
  downloads\                 reserved for new datasets
  runs\                      experiment outputs
  logs\                      process logs
  tmp\                       installers and scratch
```

Removing `E:\ValveGuardWork` removes the Python install, venv, code, caches,
downloads, and results. Nothing else on the host is needed for the workspace.

## Environment

- Python 3.12.10, pip 26.2.1.
- Core packages match the local venv (numpy 2.5.x, pandas 3.0.6,
  scikit-learn 1.9.1, scipy 1.18.1, pyarrow 26.0.0, demoparser2 0.42.0).
- PyTorch 2.9.0+cu128; `torch.cuda.is_available()` is true on the RTX 5070 Ti.
- PyPI and python.org are reachable directly. `huggingface.co` times out
  directly but works through the local proxy `http://192.168.3.129:7890`
  (the local machine listens on `0.0.0.0:7890`). Set `HTTP_PROXY` and
  `HTTPS_PROXY` for CS2CD downloads; `HF_ENDPOINT=https://hf-mirror.com` is the
  fallback.

## Verified

- Full clean-label baseline on 79,533 windows, 79 features: test player AUC
  0.9430, test kill AUC 0.9006, recall@FPR=1% 0.2626 (single split).
- 5E demo `g161-20261017205155650208834_de_dust2.dem`: 10 players, 0 flagged,
  3.9 s end to end.
- 90,707-window cache and 43-channel tensor cache loaded on the host.
- Raw event pilot on 100+100 matches: 20,252 windows, 63 features; kill AUC
  0.8276 -> 0.8459 (Δ +0.0183, bootstrap CI [0.0106, 0.0258]); player recall
  gain not significant on 1,353 players. See `experiments.md` section 16.
- 200+200 replication: 41,888 windows, 35,814 after the clean-label filter;
  kill AUC 0.8702 -> 0.8904 (Δ +0.0202, CI [0.0145, 0.0258]); player AUC
  +0.0048 (CI [0.0009, 0.0087]); player recall CI still crosses zero. See
  `experiments.md` section 17.
- Full cohort: 700 matches / 67,067 clean windows; kill AUC 0.8794 -> 0.9005,
  player AUC 0.9284 -> 0.9339, recall@FPR1 0.2087 -> 0.2111 (CI crosses zero).
  Q4 FPR 3.14% -> 2.86%; sniper FPR ~5.6%. Fire timing is the best single
  group (recall 0.2157); aim-only lowers recall to 0.1908. See
  `experiments.md` section 18.
- Target acquisition (26 features, 92 event total): all recall 0.2095,
  aim_all 0.1986, Q4 FPR 2.29% for aim_all; primary recall gain still not
  significant. See `experiments.md` section 19.
- GPU 1D CNN on 43 channels: player AUC 0.9467, recall@FPR1 0.2872, kill AUC
  0.9046; vs GBDT 0.9338 / 0.2296 / 0.8497. Recall CI [-0.0108, +0.1119];
  Q4 FPR 3.11% vs 2.59%. See `experiments.md` section 20.
- 3-seed CNN ensemble: player AUC 0.9475, recall 0.2794, Q4 FPR 3.11%;
  Δrecall +0.0535 [-0.0056, +0.1174]. Still not promoted.

## Commands

```powershell
$root = 'E:\ValveGuardWork'
$vpy = "$root\venv\Scripts\python.exe"

# full handcrafted-feature baseline
& $vpy -m valveguard.scripts.run_baseline `
  --cache "$root\data\processed\cw256" `
  --out "$root\runs\baseline_full" --jobs 24

# score a 5E demo
& $vpy -m valveguard.scripts.score_demo `
  --demo "$root\data\raw\5e_sample\g161-20261017205155650208834_de_dust2.dem" `
  --model "$root\repo\ValveGuard\artifacts\valveguard_gbdt_v1.joblib" `
  --out "$root\runs\5e_g161_result.json" --include-kills

# CS2CD raw subset through the local proxy
$env:HTTP_PROXY = 'http://192.168.3.129:7890'
$env:HTTPS_PROXY = 'http://192.168.3.129:7890'
$env:HF_ENDPOINT = 'https://huggingface.co'
& $vpy "$root\repo\ValveGuard\scripts\download_raw_subset.py" `
  --out "$root\data\raw\cs2cd_subset" --n 200 --workers 4

# raw event features and pilot
& $vpy -m valveguard.scripts.build_event_features `
  --cache "$root\data\processed\cw256" `
  --raw "$root\data\raw\cs2cd_subset" `
  --out "$root\data\processed\events_cs2cd" --n-matches 100 --workers 24

& $vpy -m valveguard.scripts.run_event_pilot `
  --index "$root\data\processed\cw256\index.parquet" `
  --features "$root\repo\ValveGuard\reports\baseline_clean\summary_features.parquet" `
  --events "$root\data\processed\events_cs2cd\event_features.parquet" `
  --out "$root\runs\event_pilot"
```
