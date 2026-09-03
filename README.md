# Bi-GS-PINN: A Unified Framework with Closed-Form Optimal Weights and Bidirectional Gradient Surgery for Physics-Informed Neural Networks

This repository contains the implementation of **Bi-GS-PINN**, a unified training framework that tackles the **multi-loss balancing problem** in Physics-Informed Neural Networks (PINNs). In PINN training, the PDE residual, boundary, initial-condition (and data, for inverse problems) losses typically conflict with each other and cause optimization difficulty. Bi-GS-PINN combines two complementary mechanisms:

1. **OAW (Optimal Adaptive Weight)** — a closed-form weighting scheme that updates the loss weights every `weight_update_freq` steps based on inverse squared gradient norms, smoothed with an exponential moving average, so that the instantaneous optimization progress of the competing loss terms is balanced.
2. **Bi-GS** (**Bi**directional **G**radient **S**urgery) — operates on the gradients of individual loss terms in two directions:
   - **Angle direction** — PCGrad-style projection removes conflicting gradient components between loss terms.
   - **Magnitude direction** — conditional magnitude equalization rescales each loss term's gradient magnitude when the average magnitude similarity falls below the threshold `gamma = 0.5`.

The framework is benchmarked against four widely used baselines (**standard** uniform weighting, **PCGrad**, **GradNorm**, **MOO-VARI**) across three forward PDEs and two inverse problems, with ablation and noise-robustness studies.

## 🌟 Key Features

- **Bidirectional Gradient Surgery**: Angle projection of conflicting gradients + conditional magnitude equalization (triggered when the average magnitude similarity falls below `gamma=0.5`), applied jointly to the per-loss gradients.
- **OAW Closed-Form Weights**: Optimal weights computed from inverse squared gradient norms every 500 steps with EMA smoothing (`beta=0.9`) — no extra learnable parameters, no hyper-gradient updates.
- **Rich Baselines**: `standard`, `pcgrad`, `gradnorm`, `moo_vari` (NSGA-II Pareto search + VARI adaptive weighting)
- **Multi-Physics Support**:
  - **1D Burgers equation** — forward and inverse problems (reference data from Raissi's `burgers_shock.mat`).
  - **2D Fisher-KPP equation** — forward problem and inverse estimation of the reaction rate `rho`.
  - **4D atmospheric equation system (MMS)** — manufactured solution with forcing terms, four output components `(u, v, w, h)`.
- **Inverse Problems**: Unknown parameters (viscosity `nu`, reaction rate `rho`) are registered as learnable `nn.Parameter`s and recovered from 200 noisy observation points.
- **Two-Stage Optimization**: Adam (`lr=1e-3`) pretraining followed by L-BFGS fine-tuning with strong-Wolfe line search.
- **Ablation & Robustness Studies**: Component ablation (`full` / `no_angle` / `no_mag` / `no_oaw` / `no_ema`) and noise-robustness grid (0% / 1% / 5% Gaussian noise).

---

## 📂 Project Structure

```text
.
├── Bi_GS_正问题/                     # Forward problems (solve the PDE)
│   ├── burgers/                      # 1D Burgers equation
│   │   ├── Forward_burgers.py        # Main script: PINN training for the forward problem
│   │   ├── burgers_shock.mat         # Reference data (Raissi's Burgers dataset)
│   │   ├── plt/                      # Paper-figure plotting scripts
│   │   │   ├── plot_burgers_seed5_compare.py            # Loss/L2 convergence comparison (5 strategies)
│   │   │   ├── plot_burgers_seed5_train_heatmap_v2.py   # Solution heatmap over training
│   │   │   └── plot_burgers_seed5_weights.py            # Loss-weight evolution
│   │   └── results/                  # Training outputs (config, weights, histories, figures)
│   ├── Fisher-KPP/                   # 2D Fisher-KPP equation
│   │   ├── Forward_Fisher-KPP.py     # Main script: PINN training for the forward problem
│   │   ├── plt/
│   │   │   ├── plot_Fisher-KPP_seed4_compare.py         # Loss/L2 convergence comparison
│   │   │   ├── plot_Fisher-KPP_seed4_error_5x3.py       # 5-strategy absolute-error heatmap grid
│   │   │   └── plot_Fisher-KPP_seed4_weights.py         # Loss-weight evolution
│   │   └── results/
│   └── 3DMMS/                        # 4D atmospheric equation system (MMS)
│       ├── Forward_3DMMS.py          # Main script: PINN training for the MMS system
│       ├── ablation_3DMMS_Bi_GS_PINN.py  # Ablation study of Bi-GS components
│       ├── plt/
│       │   ├── plot_3DMMS_seed8_compare.py              # Loss/L2 convergence comparison
│       │   ├── plot_3DMMS_seed8_error_5x4.py            # 5-strategy error grid (4 components)
│       │   └── plot_3DMMS_seed8_weights.py              # Loss-weight evolution
│       └── results/
├── Bi_GS_反问题/                     # Inverse problems (estimate PDE parameters)
│   ├── burgers/                      # Estimate the viscosity nu
│   │   ├── burgers_shock.mat
│   │   ├── inverse_burgers.py        # Main script: parameter estimation for nu
│   │   ├── noise_robustness_burgers.py  # Noise-robustness study (0% / 1% / 5% noise)
│   │   ├── plt/
│   │   │   └── plot_burgers_inverse_seed9_mixed_5x3.py  # Mixed error figures (cloud + slices)
│   │   └── results/
│   └── Fisher-KPP/                   # Estimate the reaction rate rho
│       ├── inverse_Fisher-KPP.py     # Main script: parameter estimation for rho
│       ├── plt/
│       │   └── plot_Fisher-KPP_inverse_seed4_mixed_5x3.py
│       └── results/
├── tree.py                           # Utility: generates the directory-tree snapshot
└── tree.txt                          # Directory-tree snapshot (results/ excluded, depth 4)
```

---

## 🛠️ Installation

### 1. Environment Requirements

Python 3.8+ is required. Install the dependencies:

```bash
pip install -r requirement.txt
```

> **Note**: `requirement.txt` pins the versions verified in the local `pytorch_env` (PyTorch 2.4.1 CPU build). The server image `pytorch24.03-cuda12.4` already ships a CUDA 12.4 PyTorch — switch to the commented `torch==2.4.1+cu124` line only when installing a GPU build elsewhere (the file's comments give the exact `--index-url` commands).

### 2. Verified Environment

The code has been verified in the following environment:

| Item | Value |
|---|---|
| Python | 3.11.5 |
| Conda env | `pytorch_env` (conda 23.7.4) |
| GPU | NVIDIA GeForce RTX 4090 |
| Compute node | `ainode04` |
| Docker image | `10.252.18.69:5000/pytorch/pytorch24.03-cuda12.4-iei:v1.0` (PyTorch 24.03, CUDA 12.4) |

---

## 🚀 Quick Start

**Important**: every experiment script is **self-contained** — it embeds its own `Config` class, `PINN` model, and all training logic, so there is no shared `model.py` / `config.py` and no command-line arguments. All hyperparameters are edited directly in the `Config` class at the top of each script, then the script is run as-is:

```bash
# Forward problems
python Bi_GS_正问题/burgers/Forward_burgers.py
python Bi_GS_正问题/Fisher-KPP/Forward_Fisher-KPP.py
python Bi_GS_正问题/3DMMS/Forward_3DMMS.py
python Bi_GS_正问题/3DMMS/ablation_3DMMS_Bi_GS_PINN.py

# Inverse problems
python Bi_GS_反问题/burgers/inverse_burgers.py
python Bi_GS_反问题/Fisher-KPP/inverse_Fisher-KPP.py
python Bi_GS_反问题/burgers/noise_robustness_burgers.py
```

### Key `Config` Options

| Option | Values | Description |
|---|---|---|
| `DEBUG_MODE` | `True` / `False` | `True`: 10x fewer collocation points and only 500 Adam steps — use for local smoke tests before launching full runs on the server. |
| `INVERSE_MODE` | `True` / `False` | Switches the burgers scripts between forward (`False`) and inverse (`True`) problems. |
| `RUN_STRATEGY` | `"all"`, `"standard"`, `"bi_gs"`, `"pcgrad"`, `"gradnorm"`, `"moo_vari"`, or comma combinations (e.g. `"bi_gs,gradnorm"`) | Which loss-balancing strategies to run. The ablation script instead uses `"full"`, `"no_angle"`, `"no_mag"`, `"no_oaw"`, `"no_ema"`, `"all_ablation"`. |
| `NOISE_LEVEL` | `0.0` / `0.01` / `0.05` | Observation noise level (noise-robustness script only). |
| `seed` | int | Base random seed; each strategy is run over 10 seeds (5 seeds for the ablation / noise studies). |

### Main Experiment Scripts

| Script | Problem | Description |
|---|---|---|
| `Bi_GS_正问题/burgers/Forward_burgers.py` | 1D Burgers, forward | `nu = 0.01/pi`; reference solution interpolated from `burgers_shock.mat`; 20,000 PDE residual / 8,000 boundary / 2,000 initial points; MLP `[2, 64, 64, 64, 1]`. |
| `Bi_GS_正问题/Fisher-KPP/Forward_Fisher-KPP.py` | 2D Fisher-KPP, forward | Analytic traveling-wave solution; `nu = 0.05`, `rho = 20`; domain `[-1,1]^2 x [0, 0.4]`; MLP `[3, 64, 64, 64, 1]`. |
| `Bi_GS_正问题/3DMMS/Forward_3DMMS.py` | 4D MMS atmospheric system, forward | Manufactured solution + forcing terms; 4 outputs `(u, v, w, h)`; domain `[0,1]^4`; MLP `[4, 64, 64, 64, 4]`. |
| `Bi_GS_正问题/3DMMS/ablation_3DMMS_Bi_GS_PINN.py` | Ablation study | Toggles `USE_ANGLE_PROJECTION` / `USE_MAGNITUDE_EQUALIZATION` / `USE_OAW_WEIGHTS` / `OAW_BETA` for the `full` / `no_angle` / `no_mag` / `no_oaw` / `no_ema` variants. |
| `Bi_GS_反问题/burgers/inverse_burgers.py` | Burgers, inverse | Estimates the viscosity `nu` (true value `0.01/pi`) from 200 observation points with 1% Gaussian noise; 4 loss terms (PDE / BC / IC / data). |
| `Bi_GS_反问题/Fisher-KPP/inverse_Fisher-KPP.py` | Fisher-KPP, inverse | Estimates the reaction rate `rho` (true value `20`); same 4-term loss setup. |
| `Bi_GS_反问题/burgers/noise_robustness_burgers.py` | Noise-robustness study | Grid over 3 strategies x noise levels `{0.0, 0.01, 0.05}`; additionally implements the `db_pinn` gradient-statistics strategy (not reported in the paper). |

At the end of each run, the script prints the paper-style summary table: **mean ± std of the relative L2 error (x 1e-3)**, the estimated parameter values (for inverse problems), and the iteration at which the error first drops below `1e-3` (a training diagnostic kept for reference; the paper itself does not report this metric).

---

## 🧭 Training Strategies

| Strategy | Mechanism | Key hyperparameters |
|---|---|---|
| `standard` | Uniform weighted sum of all losses | — |
| `pcgrad` | Angle projection of conflicting gradients (PCGrad) | — |
| `gradnorm` | Gradient-norm balancing with restoring force | `gn_lr=1e-3`, update every 500 steps (after 500-step warm-up), `gn_alpha=1.5` |
| `moo_vari` | NSGA-II Pareto search over weight vectors + VARI adaptive weighting | Population 20, 5 generations, every 1000 steps, history window 100 |
| `bi_gs` | **Bidirectional Gradient Surgery** (angle projection + conditional magnitude equalization) + OAW closed-form weights | `gamma=0.5`, weight update every 500 steps, EMA `beta=0.9` |
| `db_pinn` | Gradient-statistics-based weighting | `db_mm=10` (noise-robustness script only; not reported in the paper) |

All strategies share the same two-stage optimizer: **Adam** (`lr=1e-3`, 10,000–15,000 steps) followed by **L-BFGS** (strong-Wolfe line search, up to 10,000 iterations).

---

## 📊 Outputs & Visualization

### Training Outputs

Each strategy run writes to `results/{strategy}_A{adam_epochs}_L{lbfgs_max_iter}_{timestamp}/seed_{k}/` (inverse problems use the `inverse_` prefix), containing:

| File | Content |
|---|---|
| `config.txt` | Snapshot of the configuration for the run |
| `model_final.pth` | Final model weights |
| `Convergence_Curve.png` / `.pdf` | Loss and L2 error curves (Adam -> L-BFGS phases) |
| `training_history.npz` | Iteration, loss, L2 error, strategy, seed, phase lengths |
| `loss_history.npz` | Per-term losses (`loss_pde`, `loss_bc`, `loss_ic`, and `loss_data` for inverse problems) |
| `weight_history.npz` | Evolution of the loss weights |
| `Burgers_Heatmap.png` / `FisherKPP_Slices.png` | Solution / error heatmaps (per-equation) |

### Plotting Scripts

After a training run has produced its `results/`, the corresponding script in `plt/` regenerates the paper figures (convergence comparison, error-map grids, weight evolution, inverse mixed figures) and writes them to `plt/figures/`:

```bash
python Bi_GS_正问题/burgers/plt/plot_burgers_seed5_compare.py
python Bi_GS_正问题/Fisher-KPP/plt/plot_Fisher-KPP_seed4_error_5x3.py
python Bi_GS_反问题/burgers/plt/plot_burgers_inverse_seed9_mixed_5x3.py
```

---

## 📝 Notes

- **Self-contained scripts**: the 7 main scripts duplicate their own `Config` / `PINN` / training code instead of sharing modules — tune hyperparameters in the `Config` class of the script you are running.
- **Chinese directory names**: the experiment folders are named `Bi_GS_正问题` (forward problems) and `Bi_GS_反问题` (inverse problems). When deploying to Linux servers, keep the directory names intact (or adjust the relative paths) and make sure your terminal supports UTF-8.
- **`tree.py`** is only a small utility that regenerates `tree.txt` (excluding `results/`, max depth 4) — it is unrelated to the experiments.

---

## 📄 License

This project is licensed under the [Apache License 2.0](LICENSE). See the [LICENSE](LICENSE) file for details.
