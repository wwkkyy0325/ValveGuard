# Model I/O Contract

Status: frozen 2026-10-18 for the Windows application. Version
`valveguard-gbdt-2026-10-18`.

## 1. What the "raw model" is

The raw model is a **tabular scorer**, not a sequence model:

```
59 float features  ->  HistGradientBoostingClassifier  ->  kill risk score
```

The 256x44 window is an *upstream artifact* consumed by the feature
extractor. The Windows application should bundle the extractor and the
scorer together; do not treat the window tensor as the model input.

Pipeline and shapes:

| Stage | Input shape | Output shape | dtype | Owner |
|---|---|---|---|---|
| Demo parse | `.dem` | per-tick table, ~1.3M rows x 15 cols | float64/int | demoparser2 |
| Window build | per-tick table + kill event | `(N_kills, 256, 44)` | float64 | `raw_window` |
| Feature extract | `(256, 44)` | `(59,)` | float64 | `summarize` |
| Scorer | `(N, 59)` | `(N,)` in [0,1] | float64 | GBDT artifact |
| Player aggregate | `(N,)` + player ids | `(P, 6)` table | mixed | app/post-process |

Typical per demo: `N_kills` 100-250, `P` 10 players, runtime ~19 s on CPU.

## 2. Window tensor layout

- Shape: `(256, 44)`, rows are ticks at 64 Hz.
- Row 223 is the kill tick; rows 0-222 are pre-kill (3.48 s); rows 224-255
  are post-kill (0.5 s).
- Columns, in this exact order:

```
attacker_X, attacker_Y, attacker_Z, attacker_vel,
attacker_pitch, attacker_yaw,
attacker_pitch_delta, attacker_yaw_delta,
attacker_pitch_head_delta, attacker_yaw_head_delta,
attacker_flashed, attacker_shot, attacker_kill,
is_kill_headshot, is_kill_through_smoke, is_kill_wallbang, attacker_midair,
attacker_weapon_knife, attacker_weapon_auto_rifle,
attacker_weapon_semi_rifle, attacker_weapon_pistol,
attacker_weapon_grenade, attacker_weapon_smg, attacker_weapon_shotgun,
victim_X, victim_Y, victim_Z, victim_health, victim_noise,
map_dust2, map_mirage, map_inferno, map_train, map_nuke, map_ancient,
map_vertigo, map_anubis, map_office, map_overpass, map_basalt, map_edin,
map_italy, map_thera, map_mills
```

- Position and angle columns are normalized to [0,1] with the per-map bounds
  in `geometry.MAP_BOUNDS`; `summarize` denormalizes internally.
- `attacker_vel` is speed/250. `victim_health` is health/100.
- `attacker_shot` is the positive difference of `shots_fired`.
- `attacker_kill` is 1 only at row 223.
- The published `*_delta` columns are unused by the extractor; they may be
  zero.
- Invariants: exactly 256 rows, exactly 44 columns in this order, one map
  one-hot active, kill row 223, 64 Hz.

## 3. Feature vector (59,)

Order is fixed by the artifact; any change requires a new model version.
Feature groups:

| Group | Count | Examples | Units |
|---|---|---|---|
| Target alignment | 13 | `time_to_target`, `final_err`, `overshoot`, `lock_ratio` | s, deg, ratio |
| Speed / accel / jerk | 20 | `speed_dps_*`, `accel_*`, `jerk_*` | deg/s, internal |
| Distance | 7 | `dist_*`, `dist_change_rate` | world units |
| Trajectory shape | 7 | `path_len`, `straightness`, `curvature_*` | deg, ratio, 1/deg |
| Smoothness / complexity | 5 | `sparc`, `ldlj`, `psd_slope`, `sample_entropy`, `submovements` | dimensionless |
| Target motion | 5 | `victim_speed_*` | units/s |
| Lead-lag | 2 | `lead_lag_ticks`, `lead_lag_corr` | ticks, [-1,1] |

- dtype: float64 (float32 is acceptable; the trees do not need scaling).
- NaN is allowed and handled natively by the GBDT; do not impute.
- No normalization or clipping should be applied by the app.
- Feature values must come from the bundled extractor; hand-built vectors
  are not supported.

## 4. Scorer contract

- Input: `X` with shape `(N, 59)`, float32/float64, columns in artifact order.
- Output: `predict_proba(X)[:, 1]`, shape `(N,)`, in [0,1].
- The score is a **risk score**, not a calibrated probability.
- Batch freely; the scorer is stateless and order-preserving.
- Artifact contains: model, feature names, threshold, prior, k0, version.

## 5. Player aggregation contract

For each player in the demo:

```
n          = number of valid kill windows
mean       = mean(kill scores)
shrunk     = (sum(kill scores) + k0 * prior) / (n + k0)     k0 = 5
flagged    = shrunk >= threshold
reliable   = n >= 3
```

- `prior` and `threshold` come from the artifact.
- `flagged && !reliable` means insufficient evidence; do not action it.
- Players with no valid windows are omitted.

## 6. Output schema (application level)

```json
{
  "demo": "match.dem",
  "map": "de_mirage",
  "model_version": "valveguard-gbdt-2026-10-18",
  "threshold": 0.7916,
  "prior": 0.2901,
  "k0": 5.0,
  "players": [
    {"steamid": "7656...", "name": "player",
     "kills": 18, "mean_score": 0.42, "max_kill_score": 0.81,
     "shrunk_score": 0.40, "flagged": false, "reliable": true}
  ]
}
```

Optional kill-level array when requested:

```json
{"kills": [{"steamid": "...", "kill_tick": 12791, "score": 0.31}]}
```

## 7. Versioning rules

A new version is required if any of these change:

- window length (256) or kill row (223);
- the 44-column order or normalization;
- the 59-feature order or definitions;
- the model family or training label policy;
- the threshold/prior/k0 policy.

The app must compare the artifact's feature list with the extractor's output
and refuse to score on mismatch.

## 8. Windows application boundary

The application owns:

- file selection / demo acquisition;
- calling the parser and extractor;
- displaying the player table, flagged/reliable states, and export;
- batch queue and progress.

The library owns:

- demo parsing (`awpy` in the Rust app, `demoparser2` in Python scripts);
- window building and feature extraction;
- scoring and aggregation.

Recommended packaging (2026-10-20): Tauri v2 with the Rust core and the
ONNX bundle as a resource. No Python is needed at runtime. The Python
CLI remains for artifact export and dataset experiments.

ONNX export was attempted on 2026-10-18 for the GBDT model. `skl2onnx`
1.20 fails to convert this `HistGradientBoostingClassifier` (boolean
attribute in `TreeEnsembleClassifier`), so the GBDT desktop model keeps the
JSON tree format.

The CNN champion now has a separate ONNX bundle: 15 fold models,
per-seed normalization, and sigmoid averaging in one graph. The existing
demo scripts select it automatically via `load_detector`. See
`onnx-distribution.md`.

The chosen portable format is a plain JSON tree ensemble
(`valveguard_gbdt_v1.model.json`, ~931 KB, 300 trees).  A Rust traversal was
verified against sklearn on 200 real feature rows containing 204 NaNs:
**max absolute difference 3.3e-16**.  This makes a pure-Rust Tauri backend
viable without Python at inference time.  The export is produced by

```bash
python -m valveguard.scripts.export_model_json \
  --model artifacts/valveguard_gbdt_v1.joblib \
  --out artifacts/valveguard_gbdt_v1.model.json
```

## 9. Open decisions for the Windows app

1. GUI toolkit and whether the app is a single exe or folder.
2. Whether `--include-kills` output is shown or only exported.
3. Default threshold preset: 1% FPR (current) vs a stricter 0.1% preset.
4. Minimum-kills policy: hide, grey out, or explicitly mark as insufficient.
5. Batch size and parallelism (process count vs memory).
