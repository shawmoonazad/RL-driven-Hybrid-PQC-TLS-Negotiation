# Offline RL for Hybrid PQC-TLS Protocol Selection

## Files to Add to Your Project

Copy these files to your `hybrid_pqc_tls/` directory:

| File | Description |
|------|-------------|
| `__init__.py` | **Replace** your existing `__init__.py` |
| `rl_config.py` | Configuration and constants |
| `rl_offline_dataset.py` | Dataset builder with RTT-dependent reward |
| `rl_models.py` | 5 RL algorithms (BC, CQL, IQL, BCQ, AWAC) |
| `rl_train.py` | Training pipeline |
| `rl_evaluate.py` | Evaluation and visualization |
| `rl_inference_multi.py` | Multi-model inference for deployment |
| `run_rl_pipeline.py` | **Place in project root** (not in hybrid_pqc_tls/) |

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run complete pipeline
python run_rl_pipeline.py --epochs 100
```

## Output Locations

After running, results will be in:
- `results/rl/models/` - Trained model checkpoints
- `results/rl/evaluation/` - CSV comparison tables
- `results/rl/figures/` - PNG visualizations
- `results/rl/evaluation/latex/` - LaTeX tables for paper

## RL Algorithms Implemented

1. **BC** - Behavioral Cloning (supervised baseline)
2. **CQL** - Conservative Q-Learning
3. **IQL** - Implicit Q-Learning
4. **BCQ** - Batch-Constrained Q-Learning
5. **AWAC** - Advantage Weighted Actor-Critic
