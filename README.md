# Offline RL for Hybrid PQC-TLS Protocol Selection

Codebase for a research project applying **offline reinforcement learning** to adaptive cryptographic policy selection in hybrid post-quantum TLS handshakes. Five offline RL algorithms are trained on real handshake measurement data and evaluated against a rule-based baseline.

---

## What This Does

At TLS handshake time, an agent selects one of **12 cryptographic configurations** — four policy modes × three NIST security levels — based on observed network and cryptographic timing features. The goal is to minimize latency while respecting security constraints (no classical-only fallback below a minimum level).

**Policy modes:** `REQUIRE_HYBRID` · `PQC_ONLY` · `ALLOW_FALLBACK` · `CLASSICAL_ONLY`  
**NIST security levels:** 1 · 3 · 5  
**State space:** 15 features (RTT, key generation/encapsulation/verification timings, wire overhead)

---

## Algorithms

| Algorithm | Description |
|-----------|-------------|
| **BC** | Behavioral Cloning — supervised imitation baseline |
| **CQL** | Conservative Q-Learning (Kumar et al., 2020) |
| **IQL** | Implicit Q-Learning (Kostrikov et al., 2021) |
| **BCQ** | Batch-Constrained Q-Learning (Fujimoto et al., 2019) |
| **AWAC** | Advantage Weighted Actor-Critic (Nair et al., 2020) |

All models use a shared MLP backbone (2 × 256 hidden units, ReLU) and operate over the discrete 12-action space.

---

## Project Structure

```
hybrid_pqc_tls/           # Main package
├── rl_config.py          # Action space, state space, reward config, hyperparameters
├── rl_models.py          # PyTorch implementations of all 5 algorithms
├── rl_train.py           # Training pipeline
├── rl_offline_dataset.py # Base dataset builder
├── rl_dataset_improved.py      # Improved dataset (epsilon=0.3)
├── rl_dataset_improved_v2.py   # Diverse dataset (epsilon=0.6, min 2% per action)
├── rl_evaluate.py        # Evaluation v1
├── rl_evaluate_v2.py     # Evaluation v2 — RL vs Rule-Based comparison
├── rl_inference.py       # Single-model inference
├── rl_inference_multi.py # Multi-model inference (deployment)
├── rl_env.py             # Gymnasium-compatible environment wrapper
├── generate_paper_figures.py  # Publication figure generation
│
├── config.py             # TLS/crypto configuration
├── policy.py             # Cryptographic policy logic
├── primitives.py         # Crypto primitives
├── protocol.py           # TLS protocol simulation
└── session.py            # Session management

run_rl_pipeline.py        # Main pipeline entry point (place in project root)
run_pipeline_v2.py        # Pipeline v2 (diverse dataset variant)
run_action_masking_eval.py # Inference-time action masking experiment
```

---

## Setup

```bash
pip install torch numpy pandas matplotlib scikit-learn gymnasium cryptography stable-baselines3
```

Python 3.9+ recommended.

---

## Usage

### Run the full pipeline

```bash
# Default (100 epochs)
python run_rl_pipeline.py

# Custom options
python run_rl_pipeline.py --data-path path/to/handshake_raw.csv --epochs 200

# Skip steps if already done
python run_rl_pipeline.py --skip-dataset   # reuse existing dataset
python run_rl_pipeline.py --skip-training  # reuse existing models
```

### Run the diverse-dataset variant

```bash
python run_pipeline_v2.py
```

### Run action masking evaluation (no retraining)

```bash
python run_action_masking_eval.py
```

---

## Data

The pipeline expects raw TLS handshake measurements at:

```
results/eval_grid/handshake_raw.csv
```

The dataset builder samples ~10,000 transitions with configurable exploration epsilon and minimum action coverage, then saves to:

```
results/rl/offline_rl_dataset_v2.npz
```

---

## Outputs

After running, results are written to:

```
results/rl/
├── models/                    # Trained model checkpoints (.pt)
├── evaluation/                # CSV comparison tables
│   ├── action_masking/        # Masked vs unmasked results
│   └── latex/                 # LaTeX-ready tables
└── figures/                   # PNG plots
```

---

## Key Hyperparameters

| Parameter | Default |
|-----------|---------|
| Hidden dims | 256 × 256 |
| Batch size | 256 |
| Learning rate | 3e-4 |
| Discount γ | 0.99 |
| CQL α | 1.0 |
| IQL τ (expectile) | 0.7 |
| BCQ threshold | 0.3 |
| AWAC λ | 1.0 |
| Min acceptable security level | 3 |

---

## Reward Function

The reward is RTT-dependent and encodes the security priority hierarchy:

```
R = base_reward
  − α(RTT) × latency        # RTT-scaled latency penalty
  − β × wire_overhead_KB    # Wire cost penalty
  + γ(RTT) × security_level # RTT-scaled security bonus
  + mode_bonus               # REQUIRE_HYBRID: +3.0, PQC_ONLY: +1.5,
                             # ALLOW_FALLBACK: −1.0, CLASSICAL_ONLY: −15.0
  − violation_penalty        # −5.0 if level < 3
```
