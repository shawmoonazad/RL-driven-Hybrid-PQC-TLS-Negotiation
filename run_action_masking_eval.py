#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Action Masking Evaluation for Hybrid PQC-TLS
=============================================
Experiment 1: Inference-time action masking to enforce hard PQC-safety
constraints (zero security violations, zero CLASSICAL usage).

This script:
  1. Loads existing trained models (NO retraining required)
  2. Evaluates all 5 RL methods with and without action masking
  3. Generates comparison tables (CSV + LaTeX-ready)
  4. Generates publication-quality figures

Usage:
    python run_action_masking_eval.py [--data-dir PATH]

The script expects:
  - Trained models at: {data_dir}/results/rl/models/{algo}_model.pt
  - Dataset at:        {data_dir}/results/rl/offline_rl_dataset_v2.npz

Output:
  - {data_dir}/results/rl/evaluation/action_masking/
      ├── masked_vs_unmasked_comparison.csv
      ├── masked_results_latex.tex
      ├── masked_vs_unmasked_bar.png
      └── masked_policy_distribution.png
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from copy import deepcopy

# ---------------------------------------------------------------------------
# Ensure the hybrid_pqc_tls package is importable
# ---------------------------------------------------------------------------
# Adjust this path to point to the directory that CONTAINS hybrid_pqc_tls/
# e.g., if your project root is /home/user/hybrid-pqc-tls/ then:
#   sys.path.insert(0, "/home/user/hybrid-pqc-tls")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Try common project layouts
for candidate in [SCRIPT_DIR, os.path.join(SCRIPT_DIR, "..")]:
    if os.path.isdir(os.path.join(candidate, "hybrid_pqc_tls")):
        sys.path.insert(0, candidate)
        break

from hybrid_pqc_tls.rl_config import (
    ACTION_LIST,
    ACTION_TO_IDX,
    NUM_ACTIONS,
    STATE_DIM,
    PathConfig,
    DEFAULT_PATH_CONFIG,
    TrainingConfig,
    DEFAULT_TRAINING_CONFIG,
    is_pqc_safe_action,
    get_action_policy,
    get_action_level,
)
from hybrid_pqc_tls.rl_models import (
    create_algorithm,
    get_available_algorithms,
    OfflineRLAlgorithm,
    CQL as CQLAlgo,
    BCQ as BCQAlgo,
)
from hybrid_pqc_tls.rl_offline_dataset import OfflineRLDatasetBuilder


# ============================================================================
# PQC-Safety Action Mask
# ============================================================================

def build_pqc_safety_mask() -> np.ndarray:
    """
    Build a boolean mask over the 12-action space.
    
    Safe actions satisfy BOTH:
      1. mode != CLASSICAL_ONLY
      2. level >= 3
    
    Returns:
        np.ndarray of shape (NUM_ACTIONS,), True = safe, False = masked out
    """
    mask = np.array([is_pqc_safe_action(i) for i in range(NUM_ACTIONS)])
    
    safe_indices = np.where(mask)[0]
    unsafe_indices = np.where(~mask)[0]
    
    print(f"\n[Action Mask] PQC-Safety Mask:")
    print(f"  Safe actions   ({len(safe_indices)}/12):")
    for idx in safe_indices:
        p, l = ACTION_LIST[idx]
        print(f"    {idx:2d}: {p} L{l}")
    print(f"  Masked actions ({len(unsafe_indices)}/12):")
    for idx in unsafe_indices:
        p, l = ACTION_LIST[idx]
        print(f"    {idx:2d}: {p} L{l}")
    
    return mask


# ============================================================================
# Masked Action Selection Wrapper
# ============================================================================

class MaskedAlgorithm:
    """
    Wraps any OfflineRLAlgorithm and applies an action mask at inference time.
    
    For Q-value-based methods (CQL, BCQ):
        - Compute Q-values for all actions
        - Set Q[unsafe] = -inf
        - Return argmax
    
    For policy/logit-based methods (BC, IQL, AWAC):
        - Compute logits for all actions
        - Set logits[unsafe] = -inf
        - Return argmax
    """
    
    def __init__(self, algorithm: OfflineRLAlgorithm, safety_mask: np.ndarray):
        self.algorithm = algorithm
        self.safety_mask = safety_mask  # True = safe
        self.mask_tensor = torch.BoolTensor(safety_mask)
        self.name = f"{algorithm.name}+Mask"
    
    def select_action(self, state: np.ndarray, deterministic: bool = True) -> int:
        """Select action with PQC-safety mask applied."""
        device = self.algorithm.device
        
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
            
            # --- Q-value-based methods: CQL ---
            if isinstance(self.algorithm, CQLAlgo):
                self.algorithm.q_net.eval()
                q_values = self.algorithm.q_net(state_t)  # (1, NUM_ACTIONS)
                # Mask unsafe actions
                mask_t = self.mask_tensor.to(device)
                q_values[:, ~mask_t] = float('-inf')
                return q_values.argmax(dim=-1).item()
            
            # --- BCQ: has its own internal mask; we add safety mask on top ---
            elif isinstance(self.algorithm, BCQAlgo):
                self.algorithm.q_net.eval()
                self.algorithm.bc_policy.eval()
                
                # BCQ's internal filtering
                probs = self.algorithm.bc_policy.get_probs(state_t)
                max_prob = probs.max()
                bcq_mask = (probs >= self.algorithm.threshold * max_prob).float()
                
                # Q-values with BCQ mask
                q_values = self.algorithm.q_net(state_t)
                masked_q = q_values - 1e8 * (1 - bcq_mask)
                
                # Apply safety mask on top
                mask_t = self.mask_tensor.to(device)
                masked_q[:, ~mask_t] = float('-inf')
                
                return masked_q.argmax(dim=-1).item()
            
            # --- Policy-based methods: BC, IQL, AWAC ---
            else:
                self.algorithm.policy.eval()
                logits = self.algorithm.policy(state_t)  # (1, NUM_ACTIONS)
                # Mask unsafe actions
                mask_t = self.mask_tensor.to(device)
                logits[:, ~mask_t] = float('-inf')
                return logits.argmax(dim=-1).item()


# ============================================================================
# Evaluation Engine
# ============================================================================

class ActionMaskingEvaluator:
    """
    Evaluates all RL methods with and without action masking.
    """
    
    def __init__(self, path_config: PathConfig = DEFAULT_PATH_CONFIG):
        self.path_config = path_config
        self.safety_mask = build_pqc_safety_mask()
        
        # Load dataset
        self.data = None
        self.test_idx = None
        self._load_data()
    
    def _load_data(self):
        """Load the offline RL dataset."""
        builder = OfflineRLDatasetBuilder()
        self.data = builder.load(self.path_config.dataset_path)
        self.test_idx = self.data["test_idx"]
        print(f"[Evaluator] Loaded {len(self.test_idx)} test samples")
    
    def _get_expected_metrics(self, action_idx: int, rtt: float) -> Dict[str, float]:
        """Get expected latency/wire for a given (action, rtt) pair from the dataset."""
        A = self.data["A"]
        RTT = self.data["rtt"]
        LAT = self.data["latency"]
        WIRE = self.data["wire"]
        R = self.data["R"]
        
        mask = (A == action_idx) & (np.abs(RTT - rtt) < 5)
        if mask.sum() == 0:
            mask = (A == action_idx)
        if mask.sum() == 0:
            return {"latency": 0.0, "wire": 0.0, "reward": 0.0}
        
        return {
            "latency": float(LAT[mask].mean()),
            "wire": float(WIRE[mask].mean()),
            "reward": float(R[mask].mean()),
        }
    
    def evaluate_method(self, method, name: str, use_raw_states: bool = False) -> Dict:
        """Evaluate a single method on the test set."""
        if use_raw_states:
            S = self.data["S_raw"][self.test_idx]
        else:
            S = self.data["S"][self.test_idx]
        
        RTT = self.data["rtt"][self.test_idx]
        
        actions = []
        rewards = []
        latencies = []
        wire_bytes = []
        
        for i in range(len(S)):
            state = S[i]
            rtt = RTT[i]
            
            action = method.select_action(state)
            actions.append(action)
            
            metrics = self._get_expected_metrics(action, rtt)
            rewards.append(metrics["reward"])
            latencies.append(metrics["latency"])
            wire_bytes.append(metrics["wire"])
        
        actions = np.array(actions)
        rewards = np.array(rewards)
        latencies = np.array(latencies)
        wire_arr = np.array(wire_bytes)
        
        # Compute aggregate metrics
        action_counts = np.bincount(actions, minlength=NUM_ACTIONS)
        action_dist = action_counts / len(actions)
        
        result = {
            "name": name,
            "mean_reward": float(rewards.mean()),
            "mean_latency_ms": float(latencies.mean()),
            "median_latency_ms": float(np.median(latencies)),
            "p95_latency_ms": float(np.percentile(latencies, 95)),
            "mean_wire_bytes": float(wire_arr.mean()),
            "hybrid_rate": sum(action_dist[i] for i, (p, l) in enumerate(ACTION_LIST) if p == "REQUIRE_HYBRID"),
            "pqc_rate": sum(action_dist[i] for i, (p, l) in enumerate(ACTION_LIST) if p == "PQC_ONLY"),
            "fallback_rate": sum(action_dist[i] for i, (p, l) in enumerate(ACTION_LIST) if p == "ALLOW_FALLBACK"),
            "classical_rate": sum(action_dist[i] for i, (p, l) in enumerate(ACTION_LIST) if p == "CLASSICAL_ONLY"),
            "violation_rate": sum(action_dist[i] for i, (p, l) in enumerate(ACTION_LIST) if l < 3),
            "level_1_rate": sum(action_dist[i] for i, (p, l) in enumerate(ACTION_LIST) if l == 1),
            "level_3_rate": sum(action_dist[i] for i, (p, l) in enumerate(ACTION_LIST) if l == 3),
            "level_5_rate": sum(action_dist[i] for i, (p, l) in enumerate(ACTION_LIST) if l == 5),
        }
        
        # Per-RTT breakdown
        per_rtt = {}
        for rtt_val in sorted(np.unique(RTT)):
            rtt_mask = RTT == rtt_val
            rtt_actions = actions[rtt_mask]
            rtt_latencies = latencies[rtt_mask]
            rtt_wire = wire_arr[rtt_mask]
            rtt_rewards = rewards[rtt_mask]
            
            rtt_action_counts = np.bincount(rtt_actions, minlength=NUM_ACTIONS)
            rtt_action_dist = rtt_action_counts / max(len(rtt_actions), 1)
            
            per_rtt[rtt_val] = {
                "mean_latency_ms": float(rtt_latencies.mean()),
                "median_latency_ms": float(np.median(rtt_latencies)),
                "p95_latency_ms": float(np.percentile(rtt_latencies, 95)),
                "mean_wire_bytes": float(rtt_wire.mean()),
                "mean_reward": float(rtt_rewards.mean()),
                "violation_rate": sum(rtt_action_dist[i] for i, (p, l) in enumerate(ACTION_LIST) if l < 3),
                "classical_rate": sum(rtt_action_dist[i] for i, (p, l) in enumerate(ACTION_LIST) if p == "CLASSICAL_ONLY"),
                "hybrid_rate": sum(rtt_action_dist[i] for i, (p, l) in enumerate(ACTION_LIST) if p == "REQUIRE_HYBRID"),
            }
        
        result["per_rtt"] = per_rtt
        return result
    
    def evaluate_rule_based(self) -> Dict:
        """Evaluate the rule-based baseline."""
        from hybrid_pqc_tls.rl_evaluate import RuleBasedBaseline
        rule_based = RuleBasedBaseline()
        return self.evaluate_method(rule_based, "Rule-Based", use_raw_states=True)
    
    def evaluate_oracle(self) -> Optional[Dict]:
        """Evaluate the oracle baseline."""
        if "oracle" not in self.data:
            print("[Warning] Oracle data not found in dataset. Skipping oracle.")
            return None
        
        from hybrid_pqc_tls.rl_evaluate import OracleBaseline
        oracle_data = self.data["oracle"]
        oracle_mapping = {float(rtt): int(action) for rtt, action in oracle_data}
        oracle = OracleBaseline(oracle_mapping)
        
        # Oracle needs raw states + rtt
        S_raw = self.data["S_raw"][self.test_idx]
        RTT = self.data["rtt"][self.test_idx]
        
        actions, rewards, latencies, wire_bytes = [], [], [], []
        for i in range(len(S_raw)):
            action = oracle.select_action(S_raw[i], RTT[i])
            actions.append(action)
            metrics = self._get_expected_metrics(action, RTT[i])
            rewards.append(metrics["reward"])
            latencies.append(metrics["latency"])
            wire_bytes.append(metrics["wire"])
        
        actions = np.array(actions)
        action_counts = np.bincount(actions, minlength=NUM_ACTIONS)
        action_dist = action_counts / len(actions)
        
        return {
            "name": "Oracle",
            "mean_reward": float(np.mean(rewards)),
            "mean_latency_ms": float(np.mean(latencies)),
            "median_latency_ms": float(np.median(latencies)),
            "p95_latency_ms": float(np.percentile(latencies, 95)),
            "mean_wire_bytes": float(np.mean(wire_bytes)),
            "hybrid_rate": sum(action_dist[i] for i, (p, l) in enumerate(ACTION_LIST) if p == "REQUIRE_HYBRID"),
            "pqc_rate": sum(action_dist[i] for i, (p, l) in enumerate(ACTION_LIST) if p == "PQC_ONLY"),
            "fallback_rate": sum(action_dist[i] for i, (p, l) in enumerate(ACTION_LIST) if p == "ALLOW_FALLBACK"),
            "classical_rate": sum(action_dist[i] for i, (p, l) in enumerate(ACTION_LIST) if p == "CLASSICAL_ONLY"),
            "violation_rate": sum(action_dist[i] for i, (p, l) in enumerate(ACTION_LIST) if l < 3),
        }
    
    def run_full_evaluation(self, algorithms: List[str] = None) -> Dict[str, Dict]:
        """
        Run evaluation for all algorithms, both masked and unmasked.
        
        Returns:
            Dict mapping method_name -> result_dict
        """
        if algorithms is None:
            algorithms = ["BC", "CQL", "IQL", "BCQ", "AWAC"]
        
        all_results = {}
        
        # --- Baselines ---
        print("\n" + "=" * 60)
        print("EVALUATING BASELINES")
        print("=" * 60)
        
        print("\n  Evaluating Rule-Based...")
        all_results["Rule-Based"] = self.evaluate_rule_based()
        
        oracle_result = self.evaluate_oracle()
        if oracle_result:
            all_results["Oracle"] = oracle_result
        
        # --- RL Methods: Unmasked ---
        print("\n" + "=" * 60)
        print("EVALUATING RL METHODS (UNMASKED)")
        print("=" * 60)
        
        loaded_algos = {}
        for algo_name in algorithms:
            model_path = os.path.join(self.path_config.model_dir, f"{algo_name.lower()}_model.pt")
            
            if not os.path.exists(model_path):
                print(f"  [SKIP] {algo_name}: model not found at {model_path}")
                continue
            
            print(f"\n  Evaluating {algo_name} (unmasked)...")
            algorithm = create_algorithm(algo_name, device="cpu")
            algorithm.load(model_path)
            loaded_algos[algo_name] = algorithm
            
            result = self.evaluate_method(algorithm, algo_name)
            all_results[algo_name] = result
        
        # --- RL Methods: Masked ---
        print("\n" + "=" * 60)
        print("EVALUATING RL METHODS (WITH ACTION MASKING)")
        print("=" * 60)
        
        for algo_name, algorithm in loaded_algos.items():
            print(f"\n  Evaluating {algo_name} + Mask...")
            masked = MaskedAlgorithm(algorithm, self.safety_mask)
            result = self.evaluate_method(masked, f"{algo_name}+Mask")
            all_results[f"{algo_name}+Mask"] = result
        
        return all_results


# ============================================================================
# Results Formatting
# ============================================================================

def format_comparison_table(results: Dict[str, Dict]) -> pd.DataFrame:
    """Create a formatted comparison DataFrame."""
    rows = []
    
    # Define display order
    order = ["Oracle", "Rule-Based",
             "CQL", "BC", "IQL", "BCQ", "AWAC",
             "CQL+Mask", "BC+Mask", "IQL+Mask", "BCQ+Mask", "AWAC+Mask"]
    
    for name in order:
        if name not in results:
            continue
        r = results[name]
        rows.append({
            "Method": r["name"],
            "Reward": f"{r['mean_reward']:.3f}",
            "Latency (ms)": f"{r['mean_latency_ms']:.1f}",
            "P50 (ms)": f"{r['median_latency_ms']:.1f}",
            "P95 (ms)": f"{r['p95_latency_ms']:.1f}",
            "Wire (B)": f"{r['mean_wire_bytes']:.0f}",
            "Hybrid (%)": f"{r['hybrid_rate']*100:.1f}",
            "PQC (%)": f"{r.get('pqc_rate', 0)*100:.1f}",
            "Fallback (%)": f"{r.get('fallback_rate', 0)*100:.1f}",
            "Classical (%)": f"{r['classical_rate']*100:.1f}",
            "Viol. (%)": f"{r['violation_rate']*100:.1f}",
        })
    
    return pd.DataFrame(rows)


def generate_latex_table(results: Dict[str, Dict], output_path: str):
    """
    Generate a LaTeX table for the paper (Table 5 in the revised draft).
    This is the 'masked' results table.
    """
    # We only include Rule-Based + masked RL methods
    methods = ["Rule-Based", "CQL+Mask", "BC+Mask", "IQL+Mask", "BCQ+Mask", "AWAC+Mask"]
    
    lines = []
    lines.append(r"\begin{table}[!h]")
    lines.append(r"\caption{Performance with Inference-Time Action Masking}")
    lines.append(r"\label{tab:masked}")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{lccccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Method} & \textbf{Reward} & \textbf{Latency} & \textbf{Wire} & \textbf{Hybrid} & \textbf{Viol.} \\")
    lines.append(r" &  & (ms) & (B) & (\%) & (\%) \\")
    lines.append(r"\midrule")
    
    for name in methods:
        if name not in results:
            continue
        r = results[name]
        
        display_name = name.replace("+Mask", " + Mask").replace("Rule-Based", "Rule-Based")
        reward = f"${r['mean_reward']:.3f}$"
        latency = f"{r['mean_latency_ms']:.1f}"
        wire = f"{r['mean_wire_bytes']:.0f}"
        hybrid = f"{r['hybrid_rate']*100:.1f}"
        viol = f"{r['violation_rate']*100:.1f}"
        
        if name == "Rule-Based":
            lines.append(f"{display_name} & {reward} & {latency} & {wire} & {hybrid} & {viol} \\\\")
            lines.append(r"\midrule")
        else:
            lines.append(f"{display_name} & {reward} & {latency} & {wire} & {hybrid} & {viol} \\\\")
    
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

def plot_masked_vs_unmasked_comparison(
    results: Dict[str, Dict],
    output_dir: str,
) -> str:
    """
    Bar chart comparing key metrics for masked vs unmasked RL methods.
    Two panels: (a) Violation Rate, (b) Mean Latency
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    algos = ["BC", "CQL", "IQL", "BCQ", "AWAC"]
    available_algos = [a for a in algos if a in results and f"{a}+Mask" in results]
    
    if not available_algos:
        print("[Warning] No algorithms available for comparison plot.")
        return ""
    
    x = np.arange(len(available_algos))
    width = 0.35
    
    # --- Panel (a): Violation Rate ---
    ax1 = axes[0]
    viol_unmasked = [results[a]["violation_rate"] * 100 for a in available_algos]
    viol_masked = [results[f"{a}+Mask"]["violation_rate"] * 100 for a in available_algos]
    
    bars1 = ax1.bar(x - width/2, viol_unmasked, width, label='Unmasked', color='#d62728', alpha=0.85)
    bars2 = ax1.bar(x + width/2, viol_masked, width, label='+ Action Mask', color='#2ca02c', alpha=0.85)
    
    ax1.axhline(y=5, color='red', linestyle='--', linewidth=1, alpha=0.6, label='5% budget')
    ax1.set_ylabel('Violation Rate (%)', fontsize=12)
    ax1.set_title('(a) Security Violations', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(available_algos, fontsize=11)
    ax1.legend(fontsize=10)
    ax1.set_ylim(0, max(max(viol_unmasked) * 1.3, 1))
    
    # Add value labels
    for bar, val in zip(bars1, viol_unmasked):
        if val > 0:
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    for bar, val in zip(bars2, viol_masked):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # --- Panel (b): Mean Latency ---
    ax2 = axes[1]
    lat_unmasked = [results[a]["mean_latency_ms"] for a in available_algos]
    lat_masked = [results[f"{a}+Mask"]["mean_latency_ms"] for a in available_algos]
    
    # Add rule-based reference
    if "Rule-Based" in results:
        rule_lat = results["Rule-Based"]["mean_latency_ms"]
        ax2.axhline(y=rule_lat, color='gray', linestyle='--', linewidth=1.5,
                    alpha=0.7, label=f'Rule-Based ({rule_lat:.1f} ms)')
    
    bars3 = ax2.bar(x - width/2, lat_unmasked, width, label='Unmasked', color='#1f77b4', alpha=0.85)
    bars4 = ax2.bar(x + width/2, lat_masked, width, label='+ Action Mask', color='#ff7f0e', alpha=0.85)
    
    ax2.set_ylabel('Mean Latency (ms)', fontsize=12)
    ax2.set_title('(b) Handshake Latency', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(available_algos, fontsize=11)
    ax2.legend(fontsize=10)
    
    # Narrow y-axis to show differences
    all_lats = lat_unmasked + lat_masked
    y_min = min(all_lats) * 0.95
    y_max = max(all_lats + ([rule_lat] if "Rule-Based" in results else [])) * 1.03
    ax2.set_ylim(y_min, y_max)
    
    plt.tight_layout()
    path = os.path.join(output_dir, "masked_vs_unmasked_bar.png")
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Figure] Saved: {path}")
    return path


def plot_masked_policy_distribution(
    results: Dict[str, Dict],
    output_dir: str,
) -> str:
    """
    Stacked bar chart showing policy distribution for masked methods.
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    
    methods_order = ["Oracle", "Rule-Based", "CQL", "CQL+Mask",
                     "BC", "BC+Mask", "IQL", "IQL+Mask",
                     "BCQ", "BCQ+Mask", "AWAC", "AWAC+Mask"]
    methods = [m for m in methods_order if m in results]
    
    if not methods:
        return ""
    
    hybrid = [results[m]["hybrid_rate"] * 100 for m in methods]
    pqc = [results[m].get("pqc_rate", 0) * 100 for m in methods]
    fallback = [results[m].get("fallback_rate", 0) * 100 for m in methods]
    classical = [results[m]["classical_rate"] * 100 for m in methods]
    
    x = np.arange(len(methods))
    width = 0.6
    
    ax.bar(x, hybrid, width, label='REQUIRE_HYBRID', color='#2ca02c')
    ax.bar(x, pqc, width, bottom=hybrid, label='PQC_ONLY', color='#1f77b4')
    ax.bar(x, fallback, width, bottom=np.array(hybrid) + np.array(pqc),
           label='ALLOW_FALLBACK', color='#ff7f0e')
    ax.bar(x, classical, width,
           bottom=np.array(hybrid) + np.array(pqc) + np.array(fallback),
           label='CLASSICAL_ONLY', color='#d62728')
    
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_title('Policy Mode Distribution: Masked vs. Unmasked', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=10)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_ylim(0, 105)
    
    plt.tight_layout()
    path = os.path.join(output_dir, "masked_policy_distribution.png")
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Figure] Saved: {path}")
    return path


def plot_masked_per_rtt_latency(
    results: Dict[str, Dict],
    output_dir: str,
) -> str:
    """
    Line plot: CQL vs CQL+Mask vs Rule-Based across RTT conditions.
    Shows P50 and P95 latency.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    methods_to_plot = {
        "Rule-Based": {"color": "gray", "linestyle": "--", "marker": "s"},
        "CQL": {"color": "#1f77b4", "linestyle": "-", "marker": "o"},
        "CQL+Mask": {"color": "#ff7f0e", "linestyle": "-", "marker": "^"},
    }
    
    available = {k: v for k, v in methods_to_plot.items() if k in results and "per_rtt" in results[k]}
    
    if not available:
        return ""
    
    # Panel (a): Median (P50) latency
    ax1 = axes[0]
    for name, style in available.items():
        per_rtt = results[name]["per_rtt"]
        rtts = sorted(per_rtt.keys())
        p50 = [per_rtt[r]["median_latency_ms"] for r in rtts]
        ax1.plot(rtts, p50, label=name, marker=style["marker"],
                linestyle=style["linestyle"], color=style["color"], linewidth=2, markersize=8)
    
    ax1.set_xlabel("RTT (ms)", fontsize=12)
    ax1.set_ylabel("Median Latency (ms)", fontsize=12)
    ax1.set_title("(a) Median Latency (P50)", fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Panel (b): Tail (P95) latency
    ax2 = axes[1]
    for name, style in available.items():
        per_rtt = results[name]["per_rtt"]
        rtts = sorted(per_rtt.keys())
        p95 = [per_rtt[r]["p95_latency_ms"] for r in rtts]
        ax2.plot(rtts, p95, label=name, marker=style["marker"],
                linestyle=style["linestyle"], color=style["color"], linewidth=2, markersize=8)
    
    ax2.set_xlabel("RTT (ms)", fontsize=12)
    ax2.set_ylabel("P95 Latency (ms)", fontsize=12)
    ax2.set_title("(b) Tail Latency (P95)", fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    path = os.path.join(output_dir, "masked_per_rtt_latency.png")
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Figure] Saved: {path}")
    return path


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Action Masking Evaluation for Hybrid PQC-TLS"
    )
    parser.add_argument(
        "--data-dir", "-d",
        type=str,
        default=".",
        help="Project root directory containing results/rl/ (default: current dir)"
    )
    parser.add_argument(
        "--algorithms", "-a",
        nargs="+",
        default=["BC", "CQL", "IQL", "BCQ", "AWAC"],
        help="RL algorithms to evaluate (default: all 5)"
    )
    args = parser.parse_args()
    
    # --- Configure paths ---
    path_config = PathConfig()
    # If the user specifies a data-dir, prefix all paths
    if args.data_dir != ".":
        path_config.dataset_dir = os.path.join(args.data_dir, path_config.dataset_dir)
        path_config.model_dir = os.path.join(args.data_dir, path_config.model_dir)
        path_config.eval_dir = os.path.join(args.data_dir, path_config.eval_dir)
        path_config.figures_dir = os.path.join(args.data_dir, path_config.figures_dir)
    
    output_dir = os.path.join(path_config.eval_dir, "action_masking")
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70)
    print("ACTION MASKING EVALUATION")
    print("Experiment 1: Inference-Time PQC-Safety Enforcement")
    print("=" * 70)
    print(f"Dataset:    {path_config.dataset_path}")
    print(f"Models:     {path_config.model_dir}")
    print(f"Output:     {output_dir}")
    
    # --- Run evaluation ---
    evaluator = ActionMaskingEvaluator(path_config=path_config)
    results = evaluator.run_full_evaluation(algorithms=args.algorithms)
    
    # --- Display results ---
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    df = format_comparison_table(results)
    print("\n" + df.to_string(index=False))
    
    # Save CSV
    csv_path = os.path.join(output_dir, "masked_vs_unmasked_comparison.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n[CSV] Saved: {csv_path}")
    
    # --- Generate LaTeX table ---
    latex_path = os.path.join(output_dir, "masked_results_latex.tex")
    generate_latex_table(results, latex_path)
    
    # --- Generate figures ---
    print("\n" + "=" * 70)
    print("GENERATING FIGURES")
    print("=" * 70)
    
    plot_masked_vs_unmasked_comparison(results, output_dir)
    plot_masked_policy_distribution(results, output_dir)
    plot_masked_per_rtt_latency(results, output_dir)
    
    # --- Print key findings for paper ---
    print("\n" + "=" * 70)
    print("KEY FINDINGS FOR PAPER (copy into Table 5)")
    print("=" * 70)
    
    for name in ["CQL+Mask", "BC+Mask", "IQL+Mask", "BCQ+Mask", "AWAC+Mask"]:
        if name not in results:
            continue
        r = results[name]
        print(f"\n  {name}:")
        print(f"    Reward:      {r['mean_reward']:.3f}")
        print(f"    Latency:     {r['mean_latency_ms']:.1f} ms")
        print(f"    Wire:        {r['mean_wire_bytes']:.0f} B")
        print(f"    Hybrid:      {r['hybrid_rate']*100:.1f}%")
        print(f"    Classical:   {r['classical_rate']*100:.1f}%")
        print(f"    Violations:  {r['violation_rate']*100:.1f}%")
    
    if "Rule-Based" in results and "CQL+Mask" in results:
        rule = results["Rule-Based"]
        cql_m = results["CQL+Mask"]
        lat_imp = ((rule["mean_latency_ms"] - cql_m["mean_latency_ms"]) / rule["mean_latency_ms"]) * 100
        print(f"\n  CQL+Mask vs Rule-Based:")
        print(f"    Latency improvement: {lat_imp:.1f}%")
        print(f"    Wire improvement:    {((rule['mean_wire_bytes'] - cql_m['mean_wire_bytes']) / rule['mean_wire_bytes']) * 100:.1f}%")
        print(f"    CQL+Mask violations: {cql_m['violation_rate']*100:.1f}% (target: 0.0%)")
    
    print("\n" + "=" * 70)
    print("EXPERIMENT 1 COMPLETE")
    print("=" * 70)
    print(f"\nAll outputs saved to: {output_dir}/")
    print("Files:")
    for f in sorted(os.listdir(output_dir)):
        print(f"  - {f}")


if __name__ == "__main__":
    main()
