#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Packet Loss Evaluation for Hybrid PQC-TLS
==========================================
Experiment 2: Evaluate robustness of learned policies under packet loss.

Approach:
  - Synthetically simulate packet loss on existing handshake measurements
  - TLS 1.3 handshake has ~3 flight segments; each can be lost with
    probability = packet_loss_rate, causing an RTT-scale retransmission delay
  - Evaluate EXISTING trained models (NO retraining) to test generalization
  - Compare CQL, CQL+Mask, and Rule-Based under {0%, 0.5%, 1%, 2%} loss

Packet Loss Model (TLS 1.3):
  n_flights = 3  (ClientHello, ServerHello+Finished, ClientFinished)
  For each flight:
    if Bernoulli(loss_rate) == 1:
      retransmission_delay += RTT + RTO_base
  effective_latency = base_latency + retransmission_delay

  Wire bytes are unchanged (retransmissions resend the same payload;
  the "wire overhead" metric refers to handshake message sizes).

Usage:
    python run_packet_loss_eval.py [--data-dir PATH]

Output (to results/rl/evaluation/packet_loss/):
    ├── packet_loss_comparison.csv
    ├── packet_loss_latex.tex
    ├── packet_loss_latency.png
    ├── packet_loss_cql_vs_rule.png
    └── packet_loss_per_rtt.png
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

# ---------------------------------------------------------------------------
# Ensure the hybrid_pqc_tls package is importable
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
    PathConfig,
    DEFAULT_PATH_CONFIG,
    TrainingConfig,
    DEFAULT_TRAINING_CONFIG,
    is_pqc_safe_action,
)
from hybrid_pqc_tls.rl_models import create_algorithm, CQL as CQLAlgo, BCQ as BCQAlgo
from hybrid_pqc_tls.rl_offline_dataset import OfflineRLDatasetBuilder
from hybrid_pqc_tls.rl_evaluate import RuleBasedBaseline, OracleBaseline

import torch


# ============================================================================
# Packet Loss Simulation Model
# ============================================================================

class PacketLossSimulator:
    """
    Simulates the effect of packet loss on TLS 1.3 handshake latency.
    
    TLS 1.3 full handshake involves ~3 flight segments:
      1. ClientHello
      2. ServerHello + EncryptedExtensions + Certificate + Finished
      3. Client Finished
    
    Each segment can be independently lost with probability `loss_rate`.
    A lost segment triggers a TCP retransmission, adding approximately
    one RTT plus a retransmission timeout (RTO) base overhead.
    
    This is a stochastic model: each call produces a different delay
    based on random sampling. For reproducibility, use a fixed seed.
    """
    
    # Number of flight segments in a TLS 1.3 full handshake
    N_FLIGHTS = 3
    
    # Base retransmission timeout overhead (ms) beyond one RTT.
    # TCP RTO = RTT + 4*RTTVAR; we use 25ms as a reasonable base.
    RTO_BASE_MS = 25.0
    
    def __init__(self, loss_rate: float, seed: Optional[int] = None):
        """
        Args:
            loss_rate: Probability of losing each flight segment (0.0 to 1.0)
            seed: Random seed for reproducibility
        """
        self.loss_rate = loss_rate
        self.rng = np.random.RandomState(seed)
    
    def simulate_latency(
        self,
        base_latency_ms: float,
        rtt_ms: float,
        n_samples: int = 1,
    ) -> np.ndarray:
        """
        Simulate effective latency under packet loss.
        
        Args:
            base_latency_ms: Original handshake latency without loss
            rtt_ms: Round-trip time in ms
            n_samples: Number of Monte Carlo samples
            
        Returns:
            Array of simulated latencies (shape: n_samples,)
        """
        if self.loss_rate <= 0.0:
            return np.full(n_samples, base_latency_ms)
        
        # For each sample, draw the number of lost flights
        # n_lost ~ Binomial(N_FLIGHTS, loss_rate)
        n_lost = self.rng.binomial(self.N_FLIGHTS, self.loss_rate, size=n_samples)
        
        # Each lost flight adds one retransmission delay
        retransmission_delay = n_lost * (rtt_ms + self.RTO_BASE_MS)
        
        effective_latency = base_latency_ms + retransmission_delay
        return effective_latency
    
    def simulate_single(self, base_latency_ms: float, rtt_ms: float) -> float:
        """Simulate a single handshake latency under loss."""
        return float(self.simulate_latency(base_latency_ms, rtt_ms, n_samples=1)[0])


# ============================================================================
# Action Masking (reused from Experiment 1)
# ============================================================================

def build_pqc_safety_mask() -> np.ndarray:
    """Build boolean mask: True = PQC-safe action."""
    return np.array([is_pqc_safe_action(i) for i in range(NUM_ACTIONS)])


class MaskedAlgorithm:
    """Wraps an RL algorithm with PQC-safety action masking at inference."""
    
    def __init__(self, algorithm, safety_mask: np.ndarray):
        self.algorithm = algorithm
        self.safety_mask = safety_mask
        self.mask_tensor = torch.BoolTensor(safety_mask)
        self.name = f"{algorithm.name}+Mask"
    
    def select_action(self, state: np.ndarray, deterministic: bool = True) -> int:
        device = self.algorithm.device
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
            mask_t = self.mask_tensor.to(device)
            
            if isinstance(self.algorithm, CQLAlgo):
                self.algorithm.q_net.eval()
                q_values = self.algorithm.q_net(state_t)
                q_values[:, ~mask_t] = float('-inf')
                return q_values.argmax(dim=-1).item()
            elif isinstance(self.algorithm, BCQAlgo):
                self.algorithm.q_net.eval()
                self.algorithm.bc_policy.eval()
                probs = self.algorithm.bc_policy.get_probs(state_t)
                max_prob = probs.max()
                bcq_mask = (probs >= self.algorithm.threshold * max_prob).float()
                q_values = self.algorithm.q_net(state_t)
                masked_q = q_values - 1e8 * (1 - bcq_mask)
                masked_q[:, ~mask_t] = float('-inf')
                return masked_q.argmax(dim=-1).item()
            else:
                self.algorithm.policy.eval()
                logits = self.algorithm.policy(state_t)
                logits[:, ~mask_t] = float('-inf')
                return logits.argmax(dim=-1).item()


# ============================================================================
# Packet Loss Evaluator
# ============================================================================

class PacketLossEvaluator:
    """
    Evaluates RL methods under varying packet loss conditions.
    
    Uses Monte Carlo simulation: for each test sample, we simulate
    the handshake N_MC times under the given loss rate and average
    the results. This smooths out stochastic variation.
    """
    
    # Number of Monte Carlo samples per test point
    N_MC = 50
    
    def __init__(self, path_config: PathConfig = DEFAULT_PATH_CONFIG):
        self.path_config = path_config
        self.safety_mask = build_pqc_safety_mask()
        
        # Load dataset
        builder = OfflineRLDatasetBuilder()
        self.data = builder.load(path_config.dataset_path)
        self.test_idx = self.data["test_idx"]
        print(f"[Evaluator] Loaded {len(self.test_idx)} test samples")
    
    def _get_base_metrics(self, action_idx: int, rtt: float) -> Dict[str, float]:
        """Get base (no-loss) latency/wire for a given (action, rtt) pair."""
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
    
    def evaluate_method_under_loss(
        self,
        method,
        name: str,
        loss_rate: float,
        seed: int = 42,
        use_raw_states: bool = False,
    ) -> Dict:
        """
        Evaluate a method under a specific packet loss rate.
        
        The method selects actions using the original (clean) states.
        The resulting latency is then augmented with simulated loss.
        This tests whether the policy's decisions remain good under
        network impairment it was not trained on.
        """
        simulator = PacketLossSimulator(loss_rate=loss_rate, seed=seed)
        
        if use_raw_states:
            S = self.data["S_raw"][self.test_idx]
        else:
            S = self.data["S"][self.test_idx]
        
        RTT = self.data["rtt"][self.test_idx]
        
        actions = []
        latencies_mc = []  # Monte Carlo averaged latencies
        wire_bytes = []
        
        for i in range(len(S)):
            state = S[i]
            rtt = RTT[i]
            
            # Action selection (model uses clean state - tests generalization)
            if hasattr(method, 'select_action'):
                if isinstance(method, OracleBaseline):
                    action = method.select_action(state, rtt)
                elif isinstance(method, RuleBasedBaseline):
                    raw_state = self.data["S_raw"][self.test_idx[i]]
                    action = method.select_action(raw_state)
                else:
                    action = method.select_action(state)
            else:
                action = method(state)
            
            actions.append(action)
            
            # Get base metrics (no loss)
            base = self._get_base_metrics(action, rtt)
            wire_bytes.append(base["wire"])
            
            # Simulate latency under loss (Monte Carlo)
            mc_latencies = simulator.simulate_latency(
                base["latency"], rtt, n_samples=self.N_MC
            )
            latencies_mc.append(float(mc_latencies.mean()))
        
        actions = np.array(actions)
        latencies = np.array(latencies_mc)
        wire_arr = np.array(wire_bytes)
        
        # Compute metrics
        action_counts = np.bincount(actions, minlength=NUM_ACTIONS)
        action_dist = action_counts / len(actions)
        
        result = {
            "name": name,
            "loss_rate": loss_rate,
            "mean_latency_ms": float(latencies.mean()),
            "median_latency_ms": float(np.median(latencies)),
            "p95_latency_ms": float(np.percentile(latencies, 95)),
            "mean_wire_bytes": float(wire_arr.mean()),
            "hybrid_rate": sum(action_dist[i] for i, (p, l) in enumerate(ACTION_LIST) if p == "REQUIRE_HYBRID"),
            "classical_rate": sum(action_dist[i] for i, (p, l) in enumerate(ACTION_LIST) if p == "CLASSICAL_ONLY"),
            "violation_rate": sum(action_dist[i] for i, (p, l) in enumerate(ACTION_LIST) if l < 3),
        }
        
        # Per-RTT breakdown
        per_rtt = {}
        for rtt_val in sorted(np.unique(RTT)):
            rtt_mask = RTT == rtt_val
            rtt_latencies = latencies[rtt_mask]
            rtt_actions = actions[rtt_mask]
            rtt_wire = wire_arr[rtt_mask]
            
            rtt_ad = np.bincount(rtt_actions, minlength=NUM_ACTIONS) / max(rtt_mask.sum(), 1)
            
            per_rtt[float(rtt_val)] = {
                "mean_latency_ms": float(rtt_latencies.mean()),
                "median_latency_ms": float(np.median(rtt_latencies)),
                "p95_latency_ms": float(np.percentile(rtt_latencies, 95)),
                "mean_wire_bytes": float(rtt_wire.mean()),
                "violation_rate": sum(rtt_ad[i] for i, (p, l) in enumerate(ACTION_LIST) if l < 3),
                "classical_rate": sum(rtt_ad[i] for i, (p, l) in enumerate(ACTION_LIST) if p == "CLASSICAL_ONLY"),
            }
        
        result["per_rtt"] = per_rtt
        return result
    
    def run_full_evaluation(
        self,
        loss_rates: List[float] = [0.0, 0.005, 0.01, 0.02],
        algorithms: List[str] = None,
    ) -> Dict[str, List[Dict]]:
        """
        Run evaluation across all loss rates.
        
        Returns:
            Dict mapping method_name -> list of result dicts (one per loss rate)
        """
        if algorithms is None:
            algorithms = ["BC", "CQL", "IQL", "BCQ", "AWAC"]
        
        all_results = {}  # method_name -> [result_per_loss_rate]
        
        # --- Load RL models once ---
        loaded = {}
        for algo_name in algorithms:
            model_path = os.path.join(self.path_config.model_dir, f"{algo_name.lower()}_model.pt")
            if not os.path.exists(model_path):
                print(f"  [SKIP] {algo_name}: model not found at {model_path}")
                continue
            algorithm = create_algorithm(algo_name, device="cpu")
            algorithm.load(model_path)
            loaded[algo_name] = algorithm
        
        # --- Load baselines ---
        rule_based = RuleBasedBaseline()
        
        oracle = None
        if "oracle" in self.data:
            oracle_data = self.data["oracle"]
            oracle_mapping = {float(rtt): int(action) for rtt, action in oracle_data}
            oracle = OracleBaseline(oracle_mapping)
        
        # --- Evaluate across loss rates ---
        for loss_rate in loss_rates:
            loss_pct = loss_rate * 100
            print(f"\n{'='*60}")
            print(f"EVALUATING AT PACKET LOSS = {loss_pct:.1f}%")
            print(f"{'='*60}")
            
            seed = 42  # Fixed seed for reproducibility across methods
            
            # Rule-Based
            print(f"  Rule-Based...")
            r = self.evaluate_method_under_loss(
                rule_based, "Rule-Based", loss_rate, seed=seed, use_raw_states=True
            )
            all_results.setdefault("Rule-Based", []).append(r)
            
            # Oracle
            if oracle:
                print(f"  Oracle...")
                r = self.evaluate_method_under_loss(
                    oracle, "Oracle", loss_rate, seed=seed, use_raw_states=True
                )
                all_results.setdefault("Oracle", []).append(r)
            
            # RL methods (unmasked)
            for algo_name, algorithm in loaded.items():
                print(f"  {algo_name}...")
                r = self.evaluate_method_under_loss(
                    algorithm, algo_name, loss_rate, seed=seed
                )
                all_results.setdefault(algo_name, []).append(r)
            
            # CQL + Mask (primary deployment mode)
            if "CQL" in loaded:
                print(f"  CQL+Mask...")
                masked_cql = MaskedAlgorithm(loaded["CQL"], self.safety_mask)
                r = self.evaluate_method_under_loss(
                    masked_cql, "CQL+Mask", loss_rate, seed=seed
                )
                all_results.setdefault("CQL+Mask", []).append(r)
        
        return all_results


# ============================================================================
# Results Formatting
# ============================================================================

def build_comparison_df(all_results: Dict[str, List[Dict]]) -> pd.DataFrame:
    """Build a flat comparison DataFrame."""
    rows = []
    for method, results_list in all_results.items():
        for r in results_list:
            rows.append({
                "Method": r["name"],
                "Loss (%)": f"{r['loss_rate']*100:.1f}",
                "Latency (ms)": f"{r['mean_latency_ms']:.1f}",
                "P50 (ms)": f"{r['median_latency_ms']:.1f}",
                "P95 (ms)": f"{r['p95_latency_ms']:.1f}",
                "Wire (B)": f"{r['mean_wire_bytes']:.0f}",
                "Hybrid (%)": f"{r['hybrid_rate']*100:.1f}",
                "Classical (%)": f"{r['classical_rate']*100:.1f}",
                "Viol. (%)": f"{r['violation_rate']*100:.1f}",
            })
    return pd.DataFrame(rows)


def generate_latex_table(
    all_results: Dict[str, List[Dict]],
    output_path: str,
):
    """
    Generate LaTeX table for the paper (Table 6 in revised draft).
    Shows CQL+Mask vs Rule-Based across loss rates.
    """
    lines = []
    lines.append(r"\begin{table}[!h]")
    lines.append(r"\caption{Performance Under Packet Loss (CQL+Mask vs.\ Rule-Based)}")
    lines.append(r"\label{tab:packet_loss}")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{lcccccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Loss} & \multicolumn{2}{c}{\textbf{Latency (ms)}} & "
                 r"\multicolumn{2}{c}{\textbf{Viol. (\%)}} & "
                 r"\multicolumn{2}{c}{\textbf{$\Delta$ Lat.}} \\")
    lines.append(r"(\%) & Rule & CQL+M & Rule & CQL+M & (ms) & (\%) \\")
    lines.append(r"\midrule")
    
    # Match by loss rate
    rule_by_loss = {r["loss_rate"]: r for r in all_results.get("Rule-Based", [])}
    cql_by_loss = {r["loss_rate"]: r for r in all_results.get("CQL+Mask", [])}
    
    for loss_rate in sorted(rule_by_loss.keys()):
        rule = rule_by_loss[loss_rate]
        cql = cql_by_loss.get(loss_rate)
        
        loss_pct = f"{loss_rate*100:.1f}"
        rule_lat = f"{rule['mean_latency_ms']:.1f}"
        rule_viol = f"{rule['violation_rate']*100:.1f}"
        
        if cql:
            cql_lat = f"{cql['mean_latency_ms']:.1f}"
            cql_viol = f"{cql['violation_rate']*100:.1f}"
            delta_ms = f"{cql['mean_latency_ms'] - rule['mean_latency_ms']:.1f}"
            delta_pct = f"{((cql['mean_latency_ms'] - rule['mean_latency_ms']) / rule['mean_latency_ms']) * 100:.1f}"
            lines.append(f"{loss_pct} & {rule_lat} & {cql_lat} & {rule_viol} & {cql_viol} & {delta_ms} & {delta_pct} \\\\")
        else:
            lines.append(f"{loss_pct} & {rule_lat} & -- & {rule_viol} & -- & -- & -- \\\\")
    
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

def plot_latency_vs_loss(
    all_results: Dict[str, List[Dict]],
    output_dir: str,
) -> str:
    """
    Line plot: Mean latency vs. packet loss rate for key methods.
    Shows that CQL+Mask maintains its advantage across loss conditions.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    plot_methods = {
        "Rule-Based": {"color": "gray", "linestyle": "--", "marker": "s", "linewidth": 2.5},
        "CQL": {"color": "#1f77b4", "linestyle": "-", "marker": "o", "linewidth": 2},
        "CQL+Mask": {"color": "#ff7f0e", "linestyle": "-", "marker": "^", "linewidth": 2.5},
        "Oracle": {"color": "#2ca02c", "linestyle": ":", "marker": "D", "linewidth": 2},
    }
    
    for method_name, style in plot_methods.items():
        if method_name not in all_results:
            continue
        
        results_list = all_results[method_name]
        loss_rates = [r["loss_rate"] * 100 for r in results_list]
        latencies = [r["mean_latency_ms"] for r in results_list]
        
        ax.plot(loss_rates, latencies, label=method_name, **style, markersize=8)
    
    ax.set_xlabel("Packet Loss Rate (%)", fontsize=12)
    ax.set_ylabel("Mean Handshake Latency (ms)", fontsize=12)
    ax.set_title("Handshake Latency Under Packet Loss", fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks([0, 0.5, 1.0, 2.0])
    
    plt.tight_layout()
    path = os.path.join(output_dir, "packet_loss_latency.png")
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Figure] Saved: {path}")
    return path


def plot_cql_vs_rule_per_loss(
    all_results: Dict[str, List[Dict]],
    output_dir: str,
) -> str:
    """
    Grouped bar chart: CQL+Mask vs Rule-Based at each loss rate.
    Two panels: (a) Mean Latency, (b) Latency Improvement (%).
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    rule_results = all_results.get("Rule-Based", [])
    cql_results = all_results.get("CQL+Mask", [])
    
    if not rule_results or not cql_results:
        print("[Warning] Missing Rule-Based or CQL+Mask results for comparison plot.")
        return ""
    
    loss_labels = [f"{r['loss_rate']*100:.1f}%" for r in rule_results]
    x = np.arange(len(loss_labels))
    width = 0.35
    
    # --- Panel (a): Mean Latency ---
    ax1 = axes[0]
    rule_lats = [r["mean_latency_ms"] for r in rule_results]
    cql_lats = [r["mean_latency_ms"] for r in cql_results]
    
    bars1 = ax1.bar(x - width/2, rule_lats, width, label='Rule-Based', color='gray', alpha=0.85)
    bars2 = ax1.bar(x + width/2, cql_lats, width, label='CQL + Mask', color='#ff7f0e', alpha=0.85)
    
    ax1.set_ylabel('Mean Latency (ms)', fontsize=12)
    ax1.set_title('(a) Handshake Latency', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(loss_labels, fontsize=11)
    ax1.set_xlabel('Packet Loss Rate', fontsize=12)
    ax1.legend(fontsize=10)
    
    # Set y-axis to show differences clearly
    all_lats = rule_lats + cql_lats
    y_min = min(all_lats) * 0.93
    y_max = max(all_lats) * 1.05
    ax1.set_ylim(y_min, y_max)
    
    # --- Panel (b): Latency Improvement ---
    ax2 = axes[1]
    improvements = [
        ((rule_lats[i] - cql_lats[i]) / rule_lats[i]) * 100
        for i in range(len(rule_lats))
    ]
    
    colors = ['#2ca02c' if imp > 0 else '#d62728' for imp in improvements]
    bars3 = ax2.bar(x, improvements, 0.5, color=colors, alpha=0.85)
    
    ax2.set_ylabel('Latency Improvement (%)', fontsize=12)
    ax2.set_title('(b) CQL+Mask Improvement vs. Rule-Based', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(loss_labels, fontsize=11)
    ax2.set_xlabel('Packet Loss Rate', fontsize=12)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    for bar, imp in zip(bars3, improvements):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                f'{imp:.1f}%', ha='center',
                va='bottom' if height >= 0 else 'top',
                fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    path = os.path.join(output_dir, "packet_loss_cql_vs_rule.png")
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Figure] Saved: {path}")
    return path


def plot_per_rtt_under_loss(
    all_results: Dict[str, List[Dict]],
    output_dir: str,
) -> str:
    """
    Multi-panel figure: Mean latency per RTT at each loss rate.
    Shows CQL+Mask vs Rule-Based across RTT conditions under loss.
    """
    rule_results = all_results.get("Rule-Based", [])
    cql_results = all_results.get("CQL+Mask", [])
    
    if not rule_results or not cql_results:
        return ""
    
    n_rates = len(rule_results)
    fig, axes = plt.subplots(1, n_rates, figsize=(4 * n_rates, 4.5), sharey=True)
    
    if n_rates == 1:
        axes = [axes]
    
    for idx, (rule_r, cql_r) in enumerate(zip(rule_results, cql_results)):
        ax = axes[idx]
        loss_pct = rule_r["loss_rate"] * 100
        
        rule_per_rtt = rule_r["per_rtt"]
        cql_per_rtt = cql_r["per_rtt"]
        
        rtts = sorted(rule_per_rtt.keys())
        rule_lats = [rule_per_rtt[r]["mean_latency_ms"] for r in rtts]
        cql_lats = [cql_per_rtt[r]["mean_latency_ms"] for r in rtts]
        
        ax.plot(rtts, rule_lats, 's--', color='gray', label='Rule-Based',
                linewidth=2, markersize=7)
        ax.plot(rtts, cql_lats, '^-', color='#ff7f0e', label='CQL+Mask',
                linewidth=2, markersize=7)
        
        ax.set_xlabel("RTT (ms)", fontsize=11)
        if idx == 0:
            ax.set_ylabel("Mean Latency (ms)", fontsize=11)
        ax.set_title(f"Loss = {loss_pct:.1f}%", fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
    
    fig.suptitle("Latency vs. RTT Under Varying Packet Loss",
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    path = os.path.join(output_dir, "packet_loss_per_rtt.png")
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Figure] Saved: {path}")
    return path


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Packet Loss Evaluation for Hybrid PQC-TLS"
    )
    parser.add_argument(
        "--data-dir", "-d", type=str, default=".",
        help="Project root directory containing results/rl/"
    )
    parser.add_argument(
        "--algorithms", "-a", nargs="+",
        default=["BC", "CQL", "IQL", "BCQ", "AWAC"],
        help="RL algorithms to evaluate"
    )
    parser.add_argument(
        "--loss-rates", "-l", nargs="+", type=float,
        default=[0.0, 0.005, 0.01, 0.02],
        help="Packet loss rates to test (default: 0.0 0.005 0.01 0.02)"
    )
    parser.add_argument(
        "--mc-samples", "-m", type=int, default=50,
        help="Monte Carlo samples per test point (default: 50)"
    )
    args = parser.parse_args()
    
    # --- Configure paths ---
    path_config = PathConfig()
    if args.data_dir != ".":
        path_config.dataset_dir = os.path.join(args.data_dir, path_config.dataset_dir)
        path_config.model_dir = os.path.join(args.data_dir, path_config.model_dir)
        path_config.eval_dir = os.path.join(args.data_dir, path_config.eval_dir)
        path_config.figures_dir = os.path.join(args.data_dir, path_config.figures_dir)
    
    output_dir = os.path.join(path_config.eval_dir, "packet_loss")
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70)
    print("PACKET LOSS EVALUATION")
    print("Experiment 2: Robustness Under Network Impairment")
    print("=" * 70)
    print(f"Dataset:      {path_config.dataset_path}")
    print(f"Models:       {path_config.model_dir}")
    print(f"Output:       {output_dir}")
    print(f"Loss rates:   {[f'{r*100:.1f}%' for r in args.loss_rates]}")
    print(f"MC samples:   {args.mc_samples}")
    
    # --- Run evaluation ---
    evaluator = PacketLossEvaluator(path_config=path_config)
    PacketLossEvaluator.N_MC = args.mc_samples
    
    all_results = evaluator.run_full_evaluation(
        loss_rates=args.loss_rates,
        algorithms=args.algorithms,
    )
    
    # --- Display results ---
    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}")
    
    df = build_comparison_df(all_results)
    print("\n" + df.to_string(index=False))
    
    # Save CSV
    csv_path = os.path.join(output_dir, "packet_loss_comparison.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n[CSV] Saved: {csv_path}")
    
    # --- Generate LaTeX ---
    latex_path = os.path.join(output_dir, "packet_loss_latex.tex")
    generate_latex_table(all_results, latex_path)
    
    # --- Generate figures ---
    print(f"\n{'='*70}")
    print("GENERATING FIGURES")
    print(f"{'='*70}")
    
    plot_latency_vs_loss(all_results, output_dir)
    plot_cql_vs_rule_per_loss(all_results, output_dir)
    plot_per_rtt_under_loss(all_results, output_dir)
    
    # --- Key findings summary ---
    print(f"\n{'='*70}")
    print("KEY FINDINGS FOR PAPER")
    print(f"{'='*70}")
    
    rule_by_loss = {r["loss_rate"]: r for r in all_results.get("Rule-Based", [])}
    cql_by_loss = {r["loss_rate"]: r for r in all_results.get("CQL+Mask", [])}
    
    for loss_rate in sorted(rule_by_loss.keys()):
        rule = rule_by_loss[loss_rate]
        cql = cql_by_loss.get(loss_rate)
        if cql:
            improvement = ((rule["mean_latency_ms"] - cql["mean_latency_ms"]) / rule["mean_latency_ms"]) * 100
            print(f"\n  Loss = {loss_rate*100:.1f}%:")
            print(f"    Rule-Based latency:  {rule['mean_latency_ms']:.1f} ms")
            print(f"    CQL+Mask latency:    {cql['mean_latency_ms']:.1f} ms")
            print(f"    Improvement:         {improvement:+.1f}%")
            print(f"    CQL+Mask violations: {cql['violation_rate']*100:.1f}%")
    
    print(f"\n{'='*70}")
    print("EXPERIMENT 2 COMPLETE")
    print(f"{'='*70}")
    print(f"\nAll outputs saved to: {output_dir}/")
    for f in sorted(os.listdir(output_dir)):
        print(f"  - {f}")


if __name__ == "__main__":
    main()
