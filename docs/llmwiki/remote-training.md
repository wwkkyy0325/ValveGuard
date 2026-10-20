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
  data\raw\5e_sample\        three 5E demos
  data\raw\cs2cd_subset\     100 cheater + 100 clean matches
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
- `HF_ENDPOINT=https://hf-mirror.com` is required for CS2CD downloads because
  `huggingface.co` times out from this host. PyPI and python.org are reachable.

## Verified

- Full clean-label baseline on 79,533 windows, 79 features: test player AUC
  0.9430, test kill AUC 0.9006, recall@FPR=1% 0.2626 (single split).
- 5E demo `g161-20261017205155650208834_de_dust2.dem`: 10 players, 0 flagged,
  3.9 s end to end.
- 90,707-window cache and 43-channel tensor cache loaded on the host.

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

# CS2CD raw subset through the mirror
$env:HF_ENDPOINT = 'https://hf-mirror.com'
& $vpy "$root\repo\ValveGuard\scripts\download_raw_subset.py" `
  --out "$root\data\raw\cs2cd_subset" --n 100 --workers 4
```
