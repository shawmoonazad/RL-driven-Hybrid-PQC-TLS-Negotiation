#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reward Ablation Study for Hybrid PQC-TLS
=========================================
Experiment 3: Evaluate sensitivity of CQL to individual reward components.

Ablation variants:
  (a) Full reward      — baseline (existing CQL)
  (b) No mode bonus    — hybrid_bonus=0, pqc_bonus=0, penalties=0
  (c) No latency pen.  — alpha_base=0, alpha_rtt_scale=0
  (d) No wire penalty  — beta=0

For each variant:
  1. Rebuild dataset with modified reward config (same states/actions, different R)
  2. Retrain CQL (100 epochs)
  3. Evaluate with action masking (PQC-safety enforced)
  4. Compare against full-reward baseline

Usage:
    python run_ablation_study.py [--csv-path PATH] [--epochs N]

Output (to results/rl/evaluation/ablation/):
    ├── ablation_comparison.csv
    ├── ablation_latex.tex
    ├── ablation_bar_chart.png
    └── ablation_policy_distribution.png
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from typing import Dict, List, Optional
from dataclasses import replace
from collections import defaultdict

# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
for candidate in [SCRIPT_DIR, os.path.join(SCRIPT_DIR, "..")]:
    if os.path.isdir(os.path.join(candidate, "hybrid_pqc_tls")):
        sys.path.insert(0, candidate)
        break

from hybrid_pqc_tls.rl_config import (
    ACTION_LIST,
    ACTION_TO_IDX,
    NUM_ACTIONS,
    STATE_DIM,
    RewardConfig,
    DEFAULT_REWARD_CONFIG,
    PathConfig,
    DEFAULT_PATH_CONFIG,
    TrainingConfig,
    DEFAULT_TRAINING_CONFIG,
    is_pqc_safe_action,
)
from hybrid_pqc_tls.rl_models import create_algorithm, CQL as CQLAlgo
from hybrid_pqc_tls.rl_dataset_improved import ImprovedDatasetBuilder
from hybrid_pqc_tls.rl_offline_dataset import OfflineRLDatasetBuilder
from hybrid_pqc_tls.rl_evaluate import RuleBasedBaseline


# ============================================================================
# Ablation Configurations
# ============================================================================

ABLATION_CONFIGS = {
    "full_reward": {
        "label": "Full Reward",
        "description": "All reward components active (baseline)",
        "reward_config": DEFAULT_REWARD_CONFIG,  # No changes
    },
    "no_mode_bonus": {
        "label": "No Mode Bonus",
        "description": "B(π) = 0 for all modes; no preference for HYBRID over CLASSICAL",
        "reward_config": RewardConfig(
            alpha_base=DEFAULT_REWARD_CONFIG.alpha_base,
            alpha_rtt_scale=DEFAULT_REWARD_CONFIG.alpha_rtt_scale,
            beta=DEFAULT_REWARD_CONFIG.beta,
            gamma=DEFAULT_REWARD_CONFIG.gamma,
            # --- ABLATED: All mode bonuses/penalties zeroed ---
            hybrid_bonus=0.0,
            pqc_bonus=0.0,
            fallback_penalty=0.0,
            classical_penalty=0.0,
            # Keep security floor
            min_acceptable_level=DEFAULT_REWARD_CONFIG.min_acceptable_level,
            level_violation_penalty=DEFAULT_REWARD_CONFIG.level_violation_penalty,
        ),
    },
    "no_latency_penalty": {
        "label": "No Latency Pen.",
        "description": "α = 0; reward ignores handshake latency entirely",
        "reward_config": RewardConfig(
            # --- ABLATED: No latency penalty ---
            alpha_base=0.0,
            alpha_rtt_scale=0.0,
            beta=DEFAULT_REWARD_CONFIG.beta,
            gamma=DEFAULT_REWARD_CONFIG.gamma,
            hybrid_bonus=DEFAULT_REWARD_CONFIG.hybrid_bonus,
            pqc_bonus=DEFAULT_REWARD_CONFIG.pqc_bonus,
            fallback_penalty=DEFAULT_REWARD_CONFIG.fallback_penalty,
            classical_penalty=DEFAULT_REWARD_CONFIG.classical_penalty,
            min_acceptable_level=DEFAULT_REWARD_CONFIG.min_acceptable_level,
            level_violation_penalty=DEFAULT_REWARD_CONFIG.level_violation_penalty,
        ),
    },
    "no_wire_penalty": {
        "label": "No Wire Pen.",
        "description": "β = 0; reward ignores wire overhead",
        "reward_config": RewardConfig(
            alpha_base=DEFAULT_REWARD_CONFIG.alpha_base,
            alpha_rtt_scale=DEFAULT_REWARD_CONFIG.alpha_rtt_scale,
            # --- ABLATED: No wire penalty ---
            beta=0.0,
            gamma=DEFAULT_REWARD_CONFIG.gamma,
            hybrid_bonus=DEFAULT_REWARD_CONFIG.hybrid_bonus,
            pqc_bonus=DEFAULT_REWARD_CONFIG.pqc_bonus,
            fallback_penalty=DEFAULT_REWARD_CONFIG.fallback_penalty,
            classical_penalty=DEFAULT_REWARD_CONFIG.classical_penalty,
            min_acceptable_level=DEFAULT_REWARD_CONFIG.min_acceptable_level,
            level_violation_penalty=DEFAULT_REWARD_CONFIG.level_violation_penalty,
        ),
    },
}


# ============================================================================
# Action Masking (reused from Experiment 1)
# ============================================================================

def build_pqc_safety_mask() -> np.ndarray:
    return np.array([is_pqc_safe_action(i) for i in range(NUM_ACTIONS)])


class MaskedCQL:
    """CQL with PQC-safety action masking at inference."""
    def __init__(self, cql_algorithm, safety_mask: np.ndarray):
        self.algorithm = cql_algorithm
        self.mask_tensor = torch.BoolTensor(safety_mask)
        self.name = "CQL+Mask"

    def select_action(self, state: np.ndarray, deterministic: bool = True) -> int:
        device = self.algorithm.device
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
            self.algorithm.q_net.eval()
            q_values = self.algorithm.q_net(state_t)
            mask_t = self.mask_tensor.to(device)
            q_values[:, ~mask_t] = float('-inf')
            return q_values.argmax(dim=-1).item()


# ============================================================================
# Ablation Pipeline
# ============================================================================

class AblationStudy:
    """
    Runs the full ablation pipeline:
    1. For each reward variant, rebuild dataset + retrain CQL
    2. Evaluate each variant (with action masking)
    3. Generate comparison tables and figures
    """

    def __init__(
        self,
        csv_path: str,
        output_dir: str,
        num_epochs: int = 100,
        path_config: PathConfig = DEFAULT_PATH_CONFIG,
    ):
        self.csv_path = csv_path
        self.output_dir = output_dir
        self.num_epochs = num_epochs
        self.path_config = path_config
        self.safety_mask = build_pqc_safety_mask()

        # Will hold evaluation data from the full-reward dataset for
        # computing latency/wire metrics during evaluation
        self.eval_data = None

        os.makedirs(output_dir, exist_ok=True)

    def _build_dataset_for_variant(
        self,
        variant_name: str,
        reward_config: RewardConfig,
    ) -> Dict[str, np.ndarray]:
        """
        Build an offline RL dataset with a modified reward config.
        States, actions, latencies, wire bytes stay the same;
        only the reward labels change.
        """
        print(f"\n  [Dataset] Building dataset for variant: {variant_name}")

        builder = ImprovedDatasetBuilder(
            reward_config=reward_config,
            behavior_epsilon=0.3,
            prefer_pqc_safe=True,
        )
        data = builder.build(
            csv_path=self.csv_path,
            train_split=0.8,
            seed=42,
            target_size=10000,
        )

        # Save to a variant-specific path
        variant_dir = os.path.join(self.output_dir, "datasets")
        os.makedirs(variant_dir, exist_ok=True)
        variant_path = os.path.join(variant_dir, f"dataset_{variant_name}.npz")
        np.savez_compressed(variant_path, **data)
        print(f"  [Dataset] Saved to: {variant_path}")

        return data

    def _train_cql_on_dataset(
        self,
        data: Dict[str, np.ndarray],
        variant_name: str,
    ) -> CQLAlgo:
        """Train CQL on a specific dataset variant."""
        print(f"\n  [Training] Training CQL for variant: {variant_name}")

        # Prepare training data
        train_idx = data["train_idx"]
        S_train = torch.FloatTensor(data["S"][train_idx])
        A_train = torch.LongTensor(data["A"][train_idx])
        R_train = torch.FloatTensor(data["R"][train_idx])
        S_next_train = torch.FloatTensor(data["S_next"][train_idx])
        D_train = torch.FloatTensor(data["done"][train_idx])

        train_dataset = torch.utils.data.TensorDataset(
            S_train, A_train, R_train, S_next_train, D_train
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=256, shuffle=True, drop_last=True
        )

        # Create and train CQL
        config = TrainingConfig(
            num_epochs=self.num_epochs,
            batch_size=256,
            learning_rate=3e-4,
            eval_freq=10,
        )
        cql = create_algorithm("CQL", config=config, device="cpu")

        for epoch in range(self.num_epochs):
            epoch_losses = []
            for batch in train_loader:
                states, actions, rewards, next_states, dones = batch
                metrics = cql.train_step(states, actions, rewards, next_states, dones)
                epoch_losses.append(metrics["loss"])

            if (epoch + 1) % 20 == 0:
                avg_loss = np.mean(epoch_losses)
                print(f"    Epoch {epoch+1:3d}/{self.num_epochs} | Loss: {avg_loss:.4f}")

        # Save model
        model_dir = os.path.join(self.output_dir, "models")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"cql_{variant_name}.pt")
        cql.save(model_path)
        print(f"  [Training] Model saved: {model_path}")

        return cql

    def _evaluate_variant(
        self,
        cql: CQLAlgo,
        variant_name: str,
        eval_data: Dict[str, np.ndarray],
    ) -> Dict:
        """
        Evaluate a CQL variant with action masking.

        Uses the FULL-REWARD dataset's latency/wire values for evaluation
        so that all variants are compared on the same performance metrics.
        The variant's reward config only affects training; evaluation
        always measures real latency and wire overhead.
        """
        masked = MaskedCQL(cql, self.safety_mask)

        test_idx = eval_data["test_idx"]
        S = eval_data["S"][test_idx]
        RTT = eval_data["rtt"][test_idx]
        A_all = eval_data["A"]
        RTT_all = eval_data["rtt"]
        LAT_all = eval_data["latency"]
        WIRE_all = eval_data["wire"]

        actions = []
        latencies = []
        wire_bytes = []

        for i in range(len(S)):
            state = S[i]
            rtt = RTT[i]

            action = masked.select_action(state)
            actions.append(action)

            # Look up real latency/wire from full-reward dataset
            mask = (A_all == action) & (np.abs(RTT_all - rtt) < 5)
            if mask.sum() == 0:
                mask = (A_all == action)

            if mask.sum() > 0:
                latencies.append(float(LAT_all[mask].mean()))
                wire_bytes.append(float(WIRE_all[mask].mean()))
            else:
                latencies.append(0.0)
                wire_bytes.append(0.0)

        actions = np.array(actions)
        latencies = np.array(latencies)
        wire_arr = np.array(wire_bytes)

        action_counts = np.bincount(actions, minlength=NUM_ACTIONS)
        action_dist = action_counts / len(actions)

        return {
            "name": variant_name,
            "mean_latency_ms": float(latencies.mean()),
            "median_latency_ms": float(np.median(latencies)),
            "p95_latency_ms": float(np.percentile(latencies, 95)),
            "mean_wire_bytes": float(wire_arr.mean()),
            "hybrid_rate": sum(action_dist[i] for i, (p, l) in enumerate(ACTION_LIST) if p == "REQUIRE_HYBRID"),
            "pqc_rate": sum(action_dist[i] for i, (p, l) in enumerate(ACTION_LIST) if p == "PQC_ONLY"),
            "fallback_rate": sum(action_dist[i] for i, (p, l) in enumerate(ACTION_LIST) if p == "ALLOW_FALLBACK"),
            "classical_rate": sum(action_dist[i] for i, (p, l) in enumerate(ACTION_LIST) if p == "CLASSICAL_ONLY"),
            "violation_rate": sum(action_dist[i] for i, (p, l) in enumerate(ACTION_LIST) if l < 3),
        }

    def _evaluate_rule_based(self, eval_data: Dict[str, np.ndarray]) -> Dict:
        """Evaluate rule-based baseline for reference."""
        rule_based = RuleBasedBaseline()

        test_idx = eval_data["test_idx"]
        S_raw = eval_data["S_raw"][test_idx]
        RTT = eval_data["rtt"][test_idx]
        A_all = eval_data["A"]
        RTT_all = eval_data["rtt"]
        LAT_all = eval_data["latency"]
        WIRE_all = eval_data["wire"]

        actions = []
        latencies = []
        wire_bytes = []

        for i in range(len(S_raw)):
            action = rule_based.select_action(S_raw[i])
            actions.append(action)

            mask = (A_all == action) & (np.abs(RTT_all - RTT[i]) < 5)
            if mask.sum() == 0:
                mask = (A_all == action)
            if mask.sum() > 0:
                latencies.append(float(LAT_all[mask].mean()))
                wire_bytes.append(float(WIRE_all[mask].mean()))
            else:
                latencies.append(0.0)
                wire_bytes.append(0.0)

        actions = np.array(actions)
        latencies = np.array(latencies)
        wire_arr = np.array(wire_bytes)

        action_counts = np.bincount(actions, minlength=NUM_ACTIONS)
        action_dist = action_counts / len(actions)

        return {
            "name": "Rule-Based",
            "mean_latency_ms": float(latencies.mean()),
            "median_latency_ms": float(np.median(latencies)),
            "p95_latency_ms": float(np.percentile(latencies, 95)),
            "mean_wire_bytes": float(wire_arr.mean()),
            "hybrid_rate": sum(action_dist[i] for i, (p, l) in enumerate(ACTION_LIST) if p == "REQUIRE_HYBRID"),
            "pqc_rate": sum(action_dist[i] for i, (p, l) in enumerate(ACTION_LIST) if p == "PQC_ONLY"),
            "fallback_rate": sum(action_dist[i] for i, (p, l) in enumerate(ACTION_LIST) if p == "ALLOW_FALLBACK"),
            "classical_rate": sum(action_dist[i] for i, (p, l) in enumerate(ACTION_LIST) if p == "CLASSICAL_ONLY"),
            "violation_rate": sum(action_dist[i] for i, (p, l) in enumerate(ACTION_LIST) if l < 3),
        }

    def run(self) -> Dict[str, Dict]:
        """Run the full ablation study."""
        all_results = {}

        # --- Step 1: Build full-reward dataset for evaluation reference ---
        print("\n" + "=" * 70)
        print("STEP 1: Building evaluation reference (full-reward dataset)")
        print("=" * 70)
        full_data = self._build_dataset_for_variant("full_reward", DEFAULT_REWARD_CONFIG)
        self.eval_data = full_data

        # --- Step 2: Evaluate Rule-Based baseline ---
        print("\n" + "=" * 70)
        print("STEP 2: Evaluating Rule-Based baseline")
        print("=" * 70)
        all_results["Rule-Based"] = self._evaluate_rule_based(full_data)

        # --- Step 3: For each ablation variant, build + train + evaluate ---
        for variant_name, config in ABLATION_CONFIGS.items():
            print("\n" + "=" * 70)
            print(f"VARIANT: {config['label']} ({variant_name})")
            print(f"  {config['description']}")
            print("=" * 70)

            # Build dataset with this reward config
            data = self._build_dataset_for_variant(variant_name, config["reward_config"])

            # Train CQL on this variant
            cql = self._train_cql_on_dataset(data, variant_name)

            # Evaluate using full-reward dataset metrics (fair comparison)
            result = self._evaluate_variant(cql, variant_name, full_data)
            result["label"] = config["label"]
            all_results[variant_name] = result

        return all_results


# ============================================================================
# Results Formatting
# ============================================================================

def format_comparison_table(results: Dict[str, Dict]) -> pd.DataFrame:
    order = ["Rule-Based", "full_reward", "no_mode_bonus", "no_latency_penalty", "no_wire_penalty"]
    rows = []
    for name in order:
        if name not in results:
            continue
        r = results[name]
        label = r.get("label", r["name"])
        rows.append({
            "Configuration": label,
            "Latency (ms)": f"{r['mean_latency_ms']:.1f}",
            "P50 (ms)": f"{r['median_latency_ms']:.1f}",
            "P95 (ms)": f"{r['p95_latency_ms']:.1f}",
            "Wire (B)": f"{r['mean_wire_bytes']:.0f}",
            "Hybrid (%)": f"{r['hybrid_rate']*100:.1f}",
            "Classical (%)": f"{r['classical_rate']*100:.1f}",
            "Viol. (%)": f"{r['violation_rate']*100:.1f}",
        })
    return pd.DataFrame(rows)


def generate_latex_table(results: Dict[str, Dict], output_path: str):
    """Generate LaTeX table for Table 7 in the paper."""
    lines = []
    lines.append(r"\begin{table}[!h]")
    lines.append(r"\caption{Reward Component Ablation (CQL + Mask)}")
    lines.append(r"\label{tab:ablation}")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Configuration} & \textbf{Latency} & \textbf{Wire} & \textbf{Hybrid} & \textbf{Viol.} \\")
    lines.append(r" & (ms) & (B) & (\%) & (\%) \\")
    lines.append(r"\midrule")

    order = ["full_reward", "no_mode_bonus", "no_latency_penalty", "no_wire_penalty"]
    for name in order:
        if name not in results:
            continue
        r = results[name]
        label = r.get("label", name)
        lat = f"{r['mean_latency_ms']:.1f}"
        wire = f"{r['mean_wire_bytes']:.0f}"
        hybrid = f"{r['hybrid_rate']*100:.1f}"
        viol = f"{r['violation_rate']*100:.1f}"
        lines.append(f"{label} & {lat} & {wire} & {hybrid} & {viol} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    latex_str = "\n".join(lines)
    with open(output_path, "w") as f:
        f.write(latex_str)

    print(f"\n[LaTeX] Table saved to: {output_path}")
    print("\n" + latex_str)
    return latex_str


# ============================================================================
# Figure Generation
# ============================================================================

def plot_ablation_bar_chart(results: Dict[str, Dict], output_dir: str) -> str:
    """
    Grouped bar chart comparing ablation variants.
    Two panels: (a) Mean Latency, (b) Wire Overhead
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    variants = ["full_reward", "no_mode_bonus", "no_latency_penalty", "no_wire_penalty"]
    available = [v for v in variants if v in results]
    labels = [results[v].get("label", v) for v in available]
    x = np.arange(len(available))
    width = 0.5

    # Reference line for Rule-Based
    rule = results.get("Rule-Based", {})

    # --- Panel (a): Mean Latency ---
    ax1 = axes[0]
    lats = [results[v]["mean_latency_ms"] for v in available]
    colors = ['#2ca02c' if v == 'full_reward' else '#1f77b4' for v in available]
    bars1 = ax1.bar(x, lats, width, color=colors, alpha=0.85)

    if rule:
        ax1.axhline(y=rule["mean_latency_ms"], color='gray', linestyle='--',
                     linewidth=1.5, label=f'Rule-Based ({rule["mean_latency_ms"]:.1f} ms)')
        ax1.legend(fontsize=10)

    ax1.set_ylabel('Mean Latency (ms)', fontsize=12)
    ax1.set_title('(a) Handshake Latency', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=10, rotation=15, ha='right')

    # Add value labels
    for bar, val in zip(bars1, lats):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Y-axis: narrow to show differences
    all_lats = lats + ([rule["mean_latency_ms"]] if rule else [])
    y_min = min(all_lats) * 0.97
    y_max = max(all_lats) * 1.03
    ax1.set_ylim(y_min, y_max)

    # --- Panel (b): Wire Overhead ---
    ax2 = axes[1]
    wires = [results[v]["mean_wire_bytes"] for v in available]
    bars2 = ax2.bar(x, wires, width, color=colors, alpha=0.85)

    if rule:
        ax2.axhline(y=rule["mean_wire_bytes"], color='gray', linestyle='--',
                     linewidth=1.5, label=f'Rule-Based ({rule["mean_wire_bytes"]:.0f} B)')
        ax2.legend(fontsize=10)

    ax2.set_ylabel('Mean Wire Overhead (B)', fontsize=12)
    ax2.set_title('(b) Wire Overhead', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=10, rotation=15, ha='right')

    for bar, val in zip(bars2, wires):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 5,
                f'{val:.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    all_wires = wires + ([rule["mean_wire_bytes"]] if rule else [])
    ax2.set_ylim(min(all_wires) * 0.97, max(all_wires) * 1.03)

    plt.tight_layout()
    path = os.path.join(output_dir, "ablation_bar_chart.png")
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Figure] Saved: {path}")
    return path


def plot_ablation_policy_distribution(results: Dict[str, Dict], output_dir: str) -> str:
    """Stacked bar chart showing policy mode distribution per ablation variant."""
    fig, ax = plt.subplots(figsize=(10, 5))

    variants = ["full_reward", "no_mode_bonus", "no_latency_penalty", "no_wire_penalty"]
    available = [v for v in variants if v in results]
    labels = [results[v].get("label", v) for v in available]

    hybrid = [results[v]["hybrid_rate"] * 100 for v in available]
    pqc = [results[v]["pqc_rate"] * 100 for v in available]
    fallback = [results[v]["fallback_rate"] * 100 for v in available]
    classical = [results[v]["classical_rate"] * 100 for v in available]

    x = np.arange(len(available))
    width = 0.55

    ax.bar(x, hybrid, width, label='REQUIRE_HYBRID', color='#2ca02c')
    ax.bar(x, pqc, width, bottom=hybrid, label='PQC_ONLY', color='#1f77b4')
    ax.bar(x, fallback, width, bottom=np.array(hybrid) + np.array(pqc),
           label='ALLOW_FALLBACK', color='#ff7f0e')
    ax.bar(x, classical, width,
           bottom=np.array(hybrid) + np.array(pqc) + np.array(fallback),
           label='CLASSICAL_ONLY', color='#d62728')

    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_title('Policy Mode Distribution by Reward Configuration', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_ylim(0, 105)

    plt.tight_layout()
    path = os.path.join(output_dir, "ablation_policy_distribution.png")
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Figure] Saved: {path}")
    return path


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Reward Ablation Study for Hybrid PQC-TLS"
    )
    parser.add_argument(
        "--csv-path", "-c", type=str, default=None,
        help="Path to handshake_raw.csv (default: results/eval_grid/handshake_raw.csv)"
    )
    parser.add_argument(
        "--data-dir", "-d", type=str, default=".",
        help="Project root directory"
    )
    parser.add_argument(
        "--epochs", "-e", type=int, default=100,
        help="Training epochs per variant (default: 100)"
    )
    args = parser.parse_args()

    # --- Configure paths ---
    path_config = PathConfig()
    if args.data_dir != ".":
        path_config.raw_data_csv = os.path.join(args.data_dir, path_config.raw_data_csv)
        path_config.dataset_dir = os.path.join(args.data_dir, path_config.dataset_dir)
        path_config.model_dir = os.path.join(args.data_dir, path_config.model_dir)
        path_config.eval_dir = os.path.join(args.data_dir, path_config.eval_dir)
        path_config.figures_dir = os.path.join(args.data_dir, path_config.figures_dir)

    csv_path = args.csv_path or path_config.raw_data_csv
    output_dir = os.path.join(path_config.eval_dir, "ablation")

    print("=" * 70)
    print("REWARD ABLATION STUDY")
    print("Experiment 3: Sensitivity of CQL to Reward Components")
    print("=" * 70)
    print(f"CSV data:    {csv_path}")
    print(f"Output:      {output_dir}")
    print(f"Epochs:      {args.epochs}")
    print(f"Variants:    {list(ABLATION_CONFIGS.keys())}")

    if not os.path.exists(csv_path):
        print(f"\n[ERROR] CSV file not found: {csv_path}")
        print("Please provide the path to handshake_raw.csv with --csv-path")
        sys.exit(1)

    # --- Run ablation ---
    study = AblationStudy(
        csv_path=csv_path,
        output_dir=output_dir,
        num_epochs=args.epochs,
        path_config=path_config,
    )
    results = study.run()

    # --- Display results ---
    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}")

    df = format_comparison_table(results)
    print("\n" + df.to_string(index=False))

    csv_out = os.path.join(output_dir, "ablation_comparison.csv")
    df.to_csv(csv_out, index=False)
    print(f"\n[CSV] Saved: {csv_out}")

    # --- Generate LaTeX ---
    latex_path = os.path.join(output_dir, "ablation_latex.tex")
    generate_latex_table(results, latex_path)

    # --- Generate figures ---
    print(f"\n{'='*70}")
    print("GENERATING FIGURES")
    print(f"{'='*70}")

    plot_ablation_bar_chart(results, output_dir)
    plot_ablation_policy_distribution(results, output_dir)

    # --- Key findings ---
    print(f"\n{'='*70}")
    print("KEY FINDINGS FOR PAPER")
    print(f"{'='*70}")

    full = results.get("full_reward", {})
    rule = results.get("Rule-Based", {})

    for variant_name in ["no_mode_bonus", "no_latency_penalty", "no_wire_penalty"]:
        if variant_name not in results:
            continue
        r = results[variant_name]
        label = r.get("label", variant_name)
        print(f"\n  {label} vs. Full Reward:")
        if full:
            lat_diff = r["mean_latency_ms"] - full["mean_latency_ms"]
            wire_diff = r["mean_wire_bytes"] - full["mean_wire_bytes"]
            print(f"    Latency:   {r['mean_latency_ms']:.1f} ms ({lat_diff:+.1f} ms)")
            print(f"    Wire:      {r['mean_wire_bytes']:.0f} B ({wire_diff:+.0f} B)")
            print(f"    Hybrid:    {r['hybrid_rate']*100:.1f}% (full: {full['hybrid_rate']*100:.1f}%)")
            print(f"    Classical: {r['classical_rate']*100:.1f}% (full: {full['classical_rate']*100:.1f}%)")
            print(f"    Viol.:     {r['violation_rate']*100:.1f}% (full: {full['violation_rate']*100:.1f}%)")

    print(f"\n{'='*70}")
    print("EXPERIMENT 3 COMPLETE")
    print(f"{'='*70}")
    print(f"\nAll outputs saved to: {output_dir}/")
    for f in sorted(os.listdir(output_dir)):
        fpath = os.path.join(output_dir, f)
        if os.path.isfile(fpath):
            print(f"  - {f}")


if __name__ == "__main__":
    main()
