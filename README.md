# TOMI — Topology Optimization through Machine Intelligence

Reinforcement learning framework for automated LTE antenna electrical tilt and azimuth optimization. Deployed on the Victorian metropolitan network across L07 (700 MHz), L18 (1800 MHz), and L26 (2600 MHz) bands.

**Author:** Chirag Shinde (chirag.m.shinde@gmail.com)

![TOMI Flow](diagrams/Fig_TOMI_Flow.jpg)

---

## What It Does

TOMI learns which antenna tilt and azimuth adjustments improve network quality by treating the problem as a reinforcement learning task over a 1024×1024 spatial grid (~50 m resolution). A deep Q-network proposes changes, a simulated annealing stage refines them locally, and a post-performance monitoring (PPM) pipeline validates every change against real UE measurements before committing.

Formal assessment on SA3 Dandenong: **100 cells (L07)** and **80 cells (L18)** had tilts changed with measurable RSRP improvement and overlap reduction. The worst-case serving signal improved by ~9 dB on L07, and overlapping area percentage dropped consistently across overlap categories.

![Dandenong Low Band Results](diagrams/Fig_Dandenong_LowBand_Results.jpg)

---

## Architecture

![System Architecture](diagrams/Fig_Architecture.svg)

![Modes of Operation](diagrams/Fig_Modes_of_Operation.jpg)

### CNN Architecture

![CNN Architecture](diagrams/Fig_CNN_Architecture.svg)

### Reward Function

![Reward Function](diagrams/Fig_Reward_Function.svg)

### PPM Validation Pipeline

![PPM Pipeline](diagrams/Fig_PPM_Pipeline.svg)

---

## Repository Contents

| File | Lines | Description |
|------|-------|-------------|
| `code/TOMI_Qlearner.py` | 935 | Complete RL system: DQN agent, CNN propagation model, footprint cache, simulated annealing optimizer, PPM validation pipeline |
| `code/TOMI_clutter_prep.py` | 857 | Clutter data pipeline: SRTM terrain, OSM buildings, Sentinel-2 NDVI, normalisation, CNN integration helpers |
| `TOMI_whitepaper.md` | 712 | Full whitepaper (12 sections + 5 appendices) with deployment results and verified references |
| `diagrams/` | 10 JPGs + 4 SVGs | Presentation figures (JPG) and technical architecture diagrams (SVG) |

---

## Key Technical Choices

**Adaptive reward:** Alpha-weighted blend of SINR (urban/interference-limited) and RSRQ (rural/noise-limited) per grid cell, with graduated overlap penalty (onset at 3+ cells) and a coverage-gap protection term that prevents aggressive downtilting.

**Mixed precision:** fp32 weights, fp16 forward pass via autocast + GradScaler, fp64 CuPy grids (arccos precision). Training time on a T4 GPU dropped from ~8–12 hours to ~1.5–3 hours.

**CNN design:** 4-block progressive downsample (32→64→128→128 channels, ~400 px receptive field) with a separate cached clutter encoder. Dueling V(s)+A(s,a) head with no output activation — an earlier ReLU was clipping negative Q-values and preventing the agent from learning that some actions are harmful.

**Operationally constrained actions:** Only electrical tilt (RET-adjustable, 0° to −10°) is in the action space. Mechanical tilt is read-only. 8 discrete actions per antenna: fine/coarse tilt (±0.5°/±2.0°), fine/coarse azimuth (±1°/±5°).

**PPM small-sample statistics:** t-distribution (not Gaussian) posteriors, paired hourly before/after comparison to remove diurnal patterns, adaptive confidence thresholds calibrated to sample size (N≥150 → 85%, N≥40 → 80%) for 2-week test windows.

---

## Quick Start

### Prerequisites

```
pip install torch cupy-cuda12x numpy scipy pandas matplotlib pillow
```

For clutter preparation:

```
pip install rasterio pyrosm shapely sentinelsat
```

### 1. Prepare Clutter Data

```bash
python TOMI_clutter_prep.py \
    --antenna-csv AntennasSA3_tilts.csv \
    --srtm melbourne_srtm.tif \
    --osm-pbf melbourne.osm.pbf \
    --output-dir ./clutter_layers/ \
    --ndvi melbourne_ndvi.tif
```

Outputs: per-layer pickles + combined `clutter_4ch.npy` (terrain, building density, building height, vegetation).

### 2. Run the Optimizer

```python
from TOMI_Qlearner import *

# Load antenna data and clutter
load_antennas("AntennasSA3_tilts.csv")
clutter = np.load("clutter_layers/clutter_4ch.npy")

# Build environment and train
env = build_environment(G_antennas, clutter)
agent = train_dqn(env, episodes=100_000)

# Local refinement
refined = simulated_annealing(agent.best_config, env)

# Validate via PPM
ppm = PPMSinrRsrqTracker()
ppm.run_validation(refined, baseline_kpis, test_window_days=14)
```

### 3. Read the Whitepaper

Full methodology, reward function derivation, CNN architecture details, and deployment results are in [`TOMI_Whitepaper_Brief.md`](TOMI_Whitepaper_Brief.md).

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | NVIDIA T4 (16 GB VRAM) | NVIDIA V100 (16–32 GB) |
| System RAM | 16 GB | 32 GB |
| CUDA | 11.x+ | 12.x |
| Storage | 2 GB (model + clutter) | 10 GB (with raw source data) |

Peak VRAM usage is ~3–4 GB at minibatch size 8. The replay buffer is the primary RAM consumer (~8–12 GB at 50K transitions). CuPy requires a CUDA-capable GPU.

---

## Results at a Glance

### Dandenong L07 (700 MHz) — 100 cells changed

- RSRP distribution shifted right; worst-case serving signal improved by ~9 dB
- Cell dominance remained stable (no new pilot pollution)
- Coverage loss confined to low-traffic peripheral areas (correctly deprioritized by population-weighted reward)

### Dandenong L18 (1800 MHz) — 80 cells changed

- RSRP and RSRQ distributions preserved (no regression)
- Overlapping area percentage reduced consistently across overlap categories

### Overlap Reduction (Metro-wide)

![Overlap Pollution Reduction](diagrams/Fig_Overlap_Pollution_Reduction.jpg)

### Scalability

Three generations of optimization brought runtime from several minutes per SA3 to covering all of Australia within a few hours — a 1,000–10,000× speedup — through a generalized antenna abstraction that eliminates weekly retraining.

---

## Diagrams

All figures are in `diagrams/`. Technical SVG diagrams are new; JPGs are extracted from the deployment presentation:

| Figure | Description |
|--------|-------------|
| `Fig_Architecture.svg` | System architecture: data inputs → clutter prep → DQN agent → SA refinement → PPM → commit/rollback |
| `Fig_CNN_Architecture.svg` | CNN dual-encoder: 4-block env + 3-block clutter → concat → dueling V(s)+A(s,a) head |
| `Fig_Reward_Function.svg` | Adaptive reward: alpha blend (SINR↔RSRQ), graduated overlap penalty, coverage gap protection |
| `Fig_PPM_Pipeline.svg` | PPM validation: paired hourly comparison → t-distribution posterior → adaptive threshold → go/no-go |
| `Fig_TOMI_Flow.jpg` | 4-step operational loop: AI → RPA → PPM → Record update |
| `Fig_Modes_of_Operation.jpg` | Normal mode (monthly, all cells) vs Patch mode (ad-hoc, tagged cells) |
| `Fig_Normal_Run_Pipeline.jpg` | 6-step data pipeline with antenna CSV, call distribution, BigQuery output |
| `Fig_Patch_Run_Pipeline.jpg` | Patch mode pipeline with status tracking and tilt comparison |
| `Fig_Network_State_Engineered.jpg` | Traffic heat map and antenna footprint visualisation on 1024×1024 grid |
| `Fig_Noisy_vs_Clean_Environment.jpg` | Before/after propagation model: noisy artifacts vs clean footprints |
| `Fig_Overlap_Pollution_Reduction.jpg` | Before/after overlap maps showing interference reduction |
| `Fig_Victoria_SA3_Map.jpg` | Victorian SA3 region map |
| `Fig_Dandenong_LowBand_Results.jpg` | L07 RSRP CDF improvement, coverage maps, tilt deltas |
| `Fig_Results_Summary.jpg` | L18 RSRP/RSRQ CDFs and overlapping area reduction chart |

---

## References

- Mnih, V. et al. (2015). Human-level control through deep reinforcement learning. *Nature*, 518(7540).
- van Hasselt, H. et al. (2016). Deep reinforcement learning with double Q-learning. *AAAI*.
- ITU-R P.1812-6. A path-specific propagation model for point-to-area services.
- 3GPP TS 36.214. Physical layer measurements (RSRP, RSRQ, SINR definitions).

---

## License

Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International

Copyright (c) 2026- Chirag Shinde

You are free to:
- Share: copy and redistribute the material in any medium or format
- Adapt: remix, transform, and build upon the material

Under the following terms:
- Attribution: You must provide appropriate credit
- Non-Commercial: You may not use the material for commercial purposes
- ShareAlike: If you remix, you must distribute under the same license

Full license text: https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
