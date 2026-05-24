# -*- coding: utf-8 -*-
"""
Comprehensive RL Evaluation for Hybrid PQC-TLS
==============================================
Publication-quality evaluation comparing RL algorithms against Rule-Based baseline.

Focus: RL vs Rule-Based comparison with Oracle as upper bound.
NO Fixed_* baselines - clean comparison for paper.

Generates:
- Per-algorithm comparison figures (P50/P95 side-by-side like reference)
- Combined comparison tables
- Per-RTT analysis
- Security and wire overhead metrics
- LaTeX-ready tables for IEEE CNS paper
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
from dataclasses import dataclass, field

from .rl_config import (
    ACTION_LIST,
    ACTION_TO_IDX,
    NUM_ACTIONS,
    STATE_DIM,
    PathConfig,
    DEFAULT_PATH_CONFIG,
    TrainingConfig,
    DEFAULT_TRAINING_CONFIG,
)
from .rl_offline_dataset import OfflineRLDatasetBuilder
from .rl_models import create_algorithm, get_available_algorithms


# ============================================================================
# Rule-Based Baseline with Proper Precedence
# ============================================================================

class RuleBasedPolicy:
    """
    Rule-Based PERFORMANCE_ADAPTIVE policy.
    
    Precedence: REQUIRE_HYBRID > PQC_ONLY > ALLOW_FALLBACK > CLASSICAL_ONLY
    (Never use CLASSICAL_ONLY - no quantum protection)
    
    Logic based on RTT thresholds:
    - Low RTT (< 50ms): REQUIRE_HYBRID L5 (can afford full security)
    - Medium-Low RTT (50-100ms): REQUIRE_HYBRID L3 (balanced)
    - Medium-High RTT (100-150ms): PQC_ONLY L3 (reduce overhead, maintain PQC)
    - High RTT (> 150ms): ALLOW_FALLBACK L3 (performance priority, still PQC-safe)
    
    This policy NEVER selects CLASSICAL_ONLY to maintain post-quantum security.
    """
    
    def __init__(self):
        self.name = "Rule_Based"
    
    def select_action(self, state: np.ndarray, rtt: float = None) -> int:
        """
        Select action based on RTT with Hybrid > PQC > Fallback precedence.
        
        Args:
            state: State vector (may be normalized)
            rtt: RTT in ms (if None, extract from raw state[0])
        """
        if rtt is None:
            rtt = state[0]
        
        # Precedence-based selection (never CLASSICAL_ONLY)
        if rtt < 50:
            # Low RTT: Full hybrid security at L5
            return ACTION_TO_IDX[("REQUIRE_HYBRID", 5)]
        elif rtt < 100:
            # Medium RTT: Hybrid at L3 (balanced)
            return ACTION_TO_IDX[("REQUIRE_HYBRID", 3)]
        elif rtt < 150:
            # Higher RTT: PQC only at L3 (reduce overhead, still PQC-safe)
            return ACTION_TO_IDX[("PQC_ONLY", 3)]
        else:
            # Very High RTT: Fallback for performance (still prefers PQC)
            return ACTION_TO_IDX[("ALLOW_FALLBACK", 3)]


class OraclePolicy:
    """Oracle that selects best action per RTT from precomputed mapping."""
    
    def __init__(self, oracle_mapping: Dict[float, int]):
        self.oracle = oracle_mapping
        self.name = "Oracle"
    
    def select_action(self, state: np.ndarray, rtt: float = None) -> int:
        if rtt is None:
            rtt = state[0]
        # Find closest RTT in oracle
        closest_rtt = min(self.oracle.keys(), key=lambda x: abs(x - rtt))
        return self.oracle.get(closest_rtt, ACTION_TO_IDX[("REQUIRE_HYBRID", 3)])


# ============================================================================
# Evaluation Result Container
# ============================================================================

@dataclass
class EvalMetrics:
    """Evaluation metrics for a method."""
    name: str
    mean_reward: float
    std_reward: float
    mean_latency: float
    median_latency: float
    p95_latency: float
    mean_wire: float
    hybrid_rate: float
    pqc_rate: float
    fallback_rate: float
    classical_rate: float
    violation_rate: float
    level_3_plus_rate: float
    
    # Per-RTT metrics
    per_rtt: Dict[float, Dict[str, float]] = field(default_factory=dict)


# ============================================================================
# Main Evaluator
# ============================================================================

class ComprehensiveEvaluator:
    """
    Comprehensive evaluator focused on RL vs Rule-Based comparison.
    NO Fixed_* baselines - clean comparison for paper.
    """
    
    def __init__(
        self,
        path_config: PathConfig = DEFAULT_PATH_CONFIG,
        training_config: TrainingConfig = DEFAULT_TRAINING_CONFIG,
    ):
        self.path_config = path_config
        self.training_config = training_config
        self.data = None
        self.test_idx = None
        self.results: Dict[str, EvalMetrics] = {}
        
    def load_data(self, data_path: Optional[str] = None) -> None:
        """Load dataset."""
        if data_path is None:
            data_path = self.path_config.dataset_path
        
        builder = OfflineRLDatasetBuilder()
        self.data = builder.load(data_path)
        self.test_idx = self.data["test_idx"]
        print(f"[Evaluator] Loaded {len(self.test_idx)} test samples")
    
    def _get_metrics_for_action(self, action_idx: int, rtt: float) -> Dict[str, float]:
        """Get expected metrics for action at RTT from dataset."""
        A = self.data["A"]
        RTT = self.data["rtt"]
        LAT = self.data["latency"]
        WIRE = self.data["wire"]
        R = self.data["R"]
        
        # Find matching samples (action + similar RTT)
        mask = (A == action_idx) & (np.abs(RTT - rtt) < 10)
        if mask.sum() == 0:
            mask = (A == action_idx)
        
        if mask.sum() == 0:
            return {"latency": 0.0, "wire": 0.0, "reward": 0.0}
        
        return {
            "latency": float(LAT[mask].mean()),
            "wire": float(WIRE[mask].mean()),
            "reward": float(R[mask].mean()),
        }
    
    def evaluate_method(self, method, name: str) -> EvalMetrics:
        """Evaluate a single method."""
        S_raw = self.data["S_raw"][self.test_idx]
        S_norm = self.data["S"][self.test_idx]
        RTT = self.data["rtt"][self.test_idx]
        
        actions = []
        rewards = []
        latencies = []
        wires = []
        
        # Per-RTT storage
        per_rtt_data = defaultdict(lambda: {"actions": [], "rewards": [], "latencies": [], "wires": []})
        
        for i in range(len(S_raw)):
            rtt = RTT[i]
            
            # Select action based on method type
            if hasattr(method, "select_action"):
                if isinstance(method, (RuleBasedPolicy, OraclePolicy)):
                    action = method.select_action(S_raw[i], rtt=rtt)
                else:
                    # RL algorithm uses normalized state
                    action = method.select_action(S_norm[i], deterministic=True)
            
            actions.append(action)
            
            # Get metrics for this action at this RTT
            metrics = self._get_metrics_for_action(action, rtt)
            rewards.append(metrics["reward"])
            latencies.append(metrics["latency"])
            wires.append(metrics["wire"])
            
            # Store per-RTT
            per_rtt_data[rtt]["actions"].append(action)
            per_rtt_data[rtt]["rewards"].append(metrics["reward"])
            per_rtt_data[rtt]["latencies"].append(metrics["latency"])
            per_rtt_data[rtt]["wires"].append(metrics["wire"])
        
        actions = np.array(actions)
        rewards = np.array(rewards)
        latencies = np.array(latencies)
        wires = np.array(wires)
        
        # Compute action distribution
        action_counts = np.bincount(actions, minlength=NUM_ACTIONS)
        action_dist = action_counts / len(actions)
        
        hybrid_rate = sum(action_dist[i] for i, (p, l) in enumerate(ACTION_LIST) if p == "REQUIRE_HYBRID")
        pqc_rate = sum(action_dist[i] for i, (p, l) in enumerate(ACTION_LIST) if p == "PQC_ONLY")
        fallback_rate = sum(action_dist[i] for i, (p, l) in enumerate(ACTION_LIST) if p == "ALLOW_FALLBACK")
        classical_rate = sum(action_dist[i] for i, (p, l) in enumerate(ACTION_LIST) if p == "CLASSICAL_ONLY")
        violation_rate = sum(action_dist[i] for i, (p, l) in enumerate(ACTION_LIST) if l < 3)
        level_3_plus = sum(action_dist[i] for i, (p, l) in enumerate(ACTION_LIST) if l >= 3)
        
        # Compute per-RTT metrics
        per_rtt_metrics = {}
        for rtt, data in per_rtt_data.items():
            per_rtt_metrics[rtt] = {
                "mean_reward": np.mean(data["rewards"]),
                "mean_latency": np.mean(data["latencies"]),
                "median_latency": np.median(data["latencies"]),
                "p95_latency": np.percentile(data["latencies"], 95) if len(data["latencies"]) > 1 else np.mean(data["latencies"]),
                "mean_wire": np.mean(data["wires"]),
                "count": len(data["actions"]),
            }
        
        return EvalMetrics(
            name=name,
            mean_reward=float(rewards.mean()),
            std_reward=float(rewards.std()),
            mean_latency=float(latencies.mean()),
            median_latency=float(np.median(latencies)),
            p95_latency=float(np.percentile(latencies, 95)),
            mean_wire=float(wires.mean()),
            hybrid_rate=hybrid_rate,
            pqc_rate=pqc_rate,
            fallback_rate=fallback_rate,
            classical_rate=classical_rate,
            violation_rate=violation_rate,
            level_3_plus_rate=level_3_plus,
            per_rtt=per_rtt_metrics,
        )
    
    def evaluate_all(self, algorithms: List[str] = None) -> Dict[str, EvalMetrics]:
        """
        Evaluate all RL algorithms vs Rule-Based baseline.
        NO Fixed_* baselines - only RL methods, Rule-Based, and Oracle.
        """
        if self.data is None:
            self.load_data()
        
        if algorithms is None:
            algorithms = get_available_algorithms()
        
        results = {}
        
        # 1. Rule-Based Baseline (the main baseline for comparison)
        print("[Evaluator] Evaluating Rule-Based baseline...")
        rule_based = RuleBasedPolicy()
        results["Rule_Based"] = self.evaluate_method(rule_based, "Rule_Based")
        
        # 2. Oracle (upper bound reference)
        print("[Evaluator] Evaluating Oracle (upper bound)...")
        if "oracle" in self.data:
            oracle_data = self.data["oracle"]
            oracle_mapping = {float(rtt): int(action) for rtt, action in oracle_data}
            oracle = OraclePolicy(oracle_mapping)
            results["Oracle"] = self.evaluate_method(oracle, "Oracle")
        
        # 3. RL Algorithms (the methods we're evaluating)
        print("[Evaluator] Evaluating RL algorithms...")
        for algo_name in algorithms:
            model_path = os.path.join(self.path_config.model_dir, f"{algo_name.lower()}_model.pt")
            
            if not os.path.exists(model_path):
                print(f"  Skipping {algo_name}: model not found at {model_path}")
                continue
            
            print(f"  Evaluating {algo_name}...")
            algorithm = create_algorithm(algo_name, config=self.training_config, device="cpu")
            algorithm.load(model_path)
            results[algo_name] = self.evaluate_method(algorithm, algo_name)
        
        self.results = results
        return results
    
    def generate_comparison_table(self) -> pd.DataFrame:
        """Generate comparison table: RL methods vs Rule-Based."""
        rows = []
        
        # Order: Oracle first (upper bound), then Rule_Based, then RL algorithms sorted by reward
        rl_methods = [k for k in self.results.keys() if k not in ["Oracle", "Rule_Based"]]
        rl_methods_sorted = sorted(rl_methods, key=lambda x: self.results[x].mean_reward, reverse=True)
        
        order = []
        if "Oracle" in self.results:
            order.append("Oracle")
        order.append("Rule_Based")
        order.extend(rl_methods_sorted)
        
        for name in order:
            if name not in self.results:
                continue
            m = self.results[name]
            rows.append({
                "Method": name,
                "Reward": f"{m.mean_reward:.3f}",
                "Latency (ms)": f"{m.mean_latency:.1f}",
                "P50 Latency": f"{m.median_latency:.1f}",
                "P95 Latency": f"{m.p95_latency:.1f}",
                "Wire (B)": f"{m.mean_wire:.0f}",
                "Hybrid %": f"{m.hybrid_rate*100:.1f}",
                "PQC %": f"{m.pqc_rate*100:.1f}",
                "Fallback %": f"{m.fallback_rate*100:.1f}",
                "Classical %": f"{m.classical_rate*100:.1f}",
                "Violation %": f"{m.violation_rate*100:.1f}",
            })
        
        return pd.DataFrame(rows)
    
    def generate_improvement_table(self) -> pd.DataFrame:
        """Generate improvement vs Rule-Based table for RL methods only."""
        if "Rule_Based" not in self.results:
            return pd.DataFrame()
        
        baseline = self.results["Rule_Based"]
        rows = []
        
        # Only RL methods (exclude Oracle and Rule_Based)
        rl_methods = [k for k in self.results.keys() if k not in ["Oracle", "Rule_Based"]]
        
        for name in rl_methods:
            m = self.results[name]
            
            reward_imp = ((m.mean_reward - baseline.mean_reward) / abs(baseline.mean_reward)) * 100
            latency_red = ((baseline.mean_latency - m.mean_latency) / baseline.mean_latency) * 100
            wire_red = ((baseline.mean_wire - m.mean_wire) / baseline.mean_wire) * 100
            
            rows.append({
                "Method": name,
                "Reward Δ (%)": f"{reward_imp:+.1f}",
                "Latency Δ (%)": f"{latency_red:+.1f}",
                "Wire Δ (%)": f"{wire_red:+.1f}",
                "Hybrid %": f"{m.hybrid_rate*100:.1f}",
                "PQC %": f"{m.pqc_rate*100:.1f}",
                "Violation %": f"{m.violation_rate*100:.1f}",
            })
        
        df = pd.DataFrame(rows)
        if len(df) > 0:
            # Sort by reward improvement (descending)
            df["_sort"] = df["Reward Δ (%)"].str.replace("+", "").astype(float)
            df = df.sort_values("_sort", ascending=False).drop("_sort", axis=1)
        return df
    
    def save_results(self, output_dir: str = None) -> None:
        """Save all results to CSV files."""
        if output_dir is None:
            output_dir = self.path_config.eval_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Main comparison table
        df_main = self.generate_comparison_table()
        df_main.to_csv(os.path.join(output_dir, "rl_vs_rule_comparison.csv"), index=False)
        print(f"Saved: {output_dir}/rl_vs_rule_comparison.csv")
        
        # Improvement table
        df_imp = self.generate_improvement_table()
        if len(df_imp) > 0:
            df_imp.to_csv(os.path.join(output_dir, "improvement_vs_rule_based.csv"), index=False)
            print(f"Saved: {output_dir}/improvement_vs_rule_based.csv")
        
        # Per-RTT table
        rows = []
        for name, m in self.results.items():
            if m.per_rtt:
                for rtt, metrics in sorted(m.per_rtt.items()):
                    rows.append({
                        "Method": name,
                        "RTT_ms": rtt,
                        **metrics
                    })
        df_rtt = pd.DataFrame(rows)
        df_rtt.to_csv(os.path.join(output_dir, "per_rtt_metrics.csv"), index=False)
        print(f"Saved: {output_dir}/per_rtt_metrics.csv")
        
        # Print summary
        print("\n" + "="*80)
        print("EVALUATION SUMMARY: RL vs Rule-Based")
        print("="*80)
        print(df_main.to_string(index=False))
        
        if len(df_imp) > 0:
            print("\n" + "-"*80)
            print("IMPROVEMENT vs RULE-BASED:")
            print("-"*80)
            print(df_imp.to_string(index=False))


# ============================================================================
# Publication-Quality Visualization
# ============================================================================

class PublicationVisualizer:
    """
    Generate publication-quality figures for IEEE CNS paper.
    Style: Side-by-side bar charts (P50/P95) like reference image.
    """
    
    def __init__(self, results: Dict[str, EvalMetrics], output_dir: str):
        self.results = results
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Publication style settings (IEEE style)
        plt.rcParams.update({
            'font.size': 11,
            'font.family': 'serif',
            'axes.labelsize': 12,
            'axes.titlesize': 13,
            'legend.fontsize': 10,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'figure.figsize': (10, 4),
            'figure.dpi': 150,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'axes.grid': True,
            'grid.alpha': 0.3,
            'grid.linestyle': '--',
        })
        
        # Color scheme (colorblind-friendly)
        self.colors = {
            'Rule_Based': '#1f77b4',  # Blue
            'BC': '#ff7f0e',          # Orange
            'CQL': '#2ca02c',         # Green
            'IQL': '#d62728',         # Red
            'BCQ': '#9467bd',         # Purple
            'AWAC': '#8c564b',        # Brown
            'Oracle': '#7f7f7f',      # Gray
        }
    
    def _get_rtts(self) -> List[float]:
        """Get sorted RTT values from results."""
        for m in self.results.values():
            if m.per_rtt:
                return sorted(m.per_rtt.keys())
        return [0, 25, 50, 100, 200]
    
    def plot_rl_vs_rule_latency(self, rl_name: str) -> str:
        """
        Generate side-by-side latency comparison: Rule-Based vs specific RL algorithm.
        
        This matches the reference image style:
        - Left panel: Median Latency (P50)
        - Right panel: Tail Latency (P95)
        - Grouped bars per RTT
        """
        if rl_name not in self.results or "Rule_Based" not in self.results:
            return ""
        
        rl = self.results[rl_name]
        rule = self.results["Rule_Based"]
        rtts = self._get_rtts()
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        x = np.arange(len(rtts))
        width = 0.35
        
        # === Left Panel: Median Latency (P50) ===
        ax1 = axes[0]
        rule_p50 = [rule.per_rtt.get(rtt, {}).get("median_latency", 0) for rtt in rtts]
        rl_p50 = [rl.per_rtt.get(rtt, {}).get("median_latency", 0) for rtt in rtts]
        
        bars1 = ax1.bar(x - width/2, rule_p50, width, label='Rule-based', 
                        color=self.colors['Rule_Based'], edgecolor='black', linewidth=0.5)
        bars2 = ax1.bar(x + width/2, rl_p50, width, label=rl_name, 
                        color=self.colors.get(rl_name, '#ff7f0e'), edgecolor='black', linewidth=0.5)
        
        ax1.set_xlabel('RTT (ms)')
        ax1.set_ylabel('Latency (ms)')
        ax1.set_title(f'Median Latency (P50)')
        ax1.set_xticks(x)
        ax1.set_xticklabels([int(r) for r in rtts])
        ax1.legend(loc='upper left')
        ax1.set_ylim(0, max(max(rule_p50), max(rl_p50)) * 1.15)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.annotate(f'{height:.0f}', xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
        for bar in bars2:
            height = bar.get_height()
            ax1.annotate(f'{height:.0f}', xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
        
        # === Right Panel: Tail Latency (P95) ===
        ax2 = axes[1]
        rule_p95 = [rule.per_rtt.get(rtt, {}).get("p95_latency", 0) for rtt in rtts]
        rl_p95 = [rl.per_rtt.get(rtt, {}).get("p95_latency", 0) for rtt in rtts]
        
        bars3 = ax2.bar(x - width/2, rule_p95, width, label='Rule-based', 
                        color=self.colors['Rule_Based'], edgecolor='black', linewidth=0.5)
        bars4 = ax2.bar(x + width/2, rl_p95, width, label=rl_name, 
                        color=self.colors.get(rl_name, '#ff7f0e'), edgecolor='black', linewidth=0.5)
        
        ax2.set_xlabel('RTT (ms)')
        ax2.set_ylabel('Latency (ms)')
        ax2.set_title(f'Tail Latency (P95)')
        ax2.set_xticks(x)
        ax2.set_xticklabels([int(r) for r in rtts])
        ax2.legend(loc='upper left')
        ax2.set_ylim(0, max(max(rule_p95), max(rl_p95)) * 1.15)
        
        # Add value labels on bars
        for bar in bars3:
            height = bar.get_height()
            ax2.annotate(f'{height:.0f}', xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
        for bar in bars4:
            height = bar.get_height()
            ax2.annotate(f'{height:.0f}', xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
        
        # Add overall title
        fig.suptitle(f'{rl_name} vs Rule-based: Latency Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        path = os.path.join(self.output_dir, f"latency_{rl_name}_vs_rule.png")
        plt.savefig(path)
        plt.close()
        return path
    
    def plot_best_rl_vs_rule_latency(self) -> str:
        """
        Generate the main comparison figure for the paper:
        Best RL algorithm vs Rule-Based (P50/P95 side-by-side).
        """
        if "Rule_Based" not in self.results:
            return ""
        
        # Find best RL algorithm by reward
        rl_methods = [k for k in self.results.keys() if k not in ["Rule_Based", "Oracle"]]
        if not rl_methods:
            return ""
        
        best_rl = max(rl_methods, key=lambda x: self.results[x].mean_reward)
        
        # Generate the comparison
        path = self.plot_rl_vs_rule_latency(best_rl)
        
        # Also save as "best_rl_vs_rule.png" for easy reference
        if path:
            import shutil
            best_path = os.path.join(self.output_dir, "best_rl_vs_rule_latency.png")
            shutil.copy(path, best_path)
            print(f"  Created: best_rl_vs_rule_latency.png (Best RL: {best_rl})")
        
        return path
    
    def plot_all_methods_comparison(self) -> str:
        """
        Single figure comparing all RL methods vs Rule-Based.
        Median latency grouped by RTT.
        """
        rtts = self._get_rtts()
        
        # Methods to compare (Rule-Based + RL methods)
        methods = ["Rule_Based"] + [k for k in self.results.keys() if k not in ["Rule_Based", "Oracle"]]
        methods = [m for m in methods if m in self.results]
        
        if len(methods) < 2:
            return ""
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        n_methods = len(methods)
        width = 0.8 / n_methods
        x = np.arange(len(rtts))
        
        for i, method in enumerate(methods):
            m = self.results[method]
            vals = [m.per_rtt.get(rtt, {}).get("median_latency", 0) for rtt in rtts]
            offset = (i - n_methods/2 + 0.5) * width
            
            bars = ax.bar(x + offset, vals, width, label=method, 
                         color=self.colors.get(method, f'C{i}'),
                         edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel('RTT (ms)')
        ax.set_ylabel('Median Latency (ms)')
        ax.set_title('Median Latency Comparison: All Methods')
        ax.set_xticks(x)
        ax.set_xticklabels([int(r) for r in rtts])
        ax.legend(loc='upper left', ncol=2)
        
        plt.tight_layout()
        path = os.path.join(self.output_dir, "latency_all_methods_comparison.png")
        plt.savefig(path)
        plt.close()
        return path
    
    def plot_reward_comparison(self) -> str:
        """Horizontal bar chart comparing mean reward."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Order by reward (descending)
        methods = list(self.results.keys())
        rewards = [self.results[m].mean_reward for m in methods]
        sorted_pairs = sorted(zip(methods, rewards), key=lambda x: x[1], reverse=True)
        methods, rewards = zip(*sorted_pairs)
        
        colors = [self.colors.get(m, '#7f7f7f') for m in methods]
        
        bars = ax.barh(methods, rewards, color=colors, edgecolor='black', linewidth=0.5)
        
        # Add value labels
        for bar, reward in zip(bars, rewards):
            ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                   f'{reward:.2f}', va='center', fontsize=10)
        
        ax.set_xlabel('Mean Reward')
        ax.set_title('Reward Comparison: RL Methods vs Rule-Based')
        ax.axvline(x=0, color='black', linewidth=0.5)
        
        plt.tight_layout()
        path = os.path.join(self.output_dir, "reward_comparison.png")
        plt.savefig(path)
        plt.close()
        return path
    
    def plot_policy_distribution(self) -> str:
        """Stacked bar chart showing policy distribution for each method."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Exclude Oracle from policy distribution
        methods = [k for k in self.results.keys() if k != "Oracle"]
        
        hybrid = [self.results[m].hybrid_rate * 100 for m in methods]
        pqc = [self.results[m].pqc_rate * 100 for m in methods]
        fallback = [self.results[m].fallback_rate * 100 for m in methods]
        classical = [self.results[m].classical_rate * 100 for m in methods]
        
        x = np.arange(len(methods))
        width = 0.6
        
        ax.bar(x, hybrid, width, label='REQUIRE_HYBRID', color='#2ca02c')
        ax.bar(x, pqc, width, bottom=hybrid, label='PQC_ONLY', color='#1f77b4')
        ax.bar(x, fallback, width, bottom=np.array(hybrid)+np.array(pqc), 
               label='ALLOW_FALLBACK', color='#ff7f0e')
        ax.bar(x, classical, width, 
               bottom=np.array(hybrid)+np.array(pqc)+np.array(fallback),
               label='CLASSICAL_ONLY', color='#d62728')
        
        ax.set_ylabel('Percentage (%)')
        ax.set_title('Policy Distribution by Method')
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.legend(loc='upper right')
        ax.set_ylim(0, 105)
        
        plt.tight_layout()
        path = os.path.join(self.output_dir, "policy_distribution.png")
        plt.savefig(path)
        plt.close()
        return path
    
    def plot_improvement_bars(self) -> str:
        """Three-panel figure showing improvement metrics vs Rule-Based."""
        if "Rule_Based" not in self.results:
            return ""
        
        baseline = self.results["Rule_Based"]
        rl_methods = [k for k in self.results.keys() if k not in ["Rule_Based", "Oracle"]]
        
        if not rl_methods:
            return ""
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # === Panel 1: Reward Improvement ===
        ax1 = axes[0]
        improvements = []
        for m in rl_methods:
            imp = ((self.results[m].mean_reward - baseline.mean_reward) / abs(baseline.mean_reward)) * 100
            improvements.append(imp)
        
        colors = ['#2ca02c' if imp > 0 else '#d62728' for imp in improvements]
        bars = ax1.bar(rl_methods, improvements, color=colors, edgecolor='black', linewidth=0.5)
        ax1.axhline(y=0, color='black', linewidth=0.5)
        ax1.set_ylabel('Improvement (%)')
        ax1.set_title('Reward Improvement')
        ax1.set_xticklabels(rl_methods, rotation=45, ha='right')
        
        for bar, imp in zip(bars, improvements):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{imp:+.1f}%', ha='center', va='bottom' if imp > 0 else 'top', 
                    fontsize=10, fontweight='bold')
        
        # === Panel 2: Latency Reduction ===
        ax2 = axes[1]
        reductions = []
        for m in rl_methods:
            red = ((baseline.mean_latency - self.results[m].mean_latency) / baseline.mean_latency) * 100
            reductions.append(red)
        
        colors = ['#2ca02c' if r > 0 else '#d62728' for r in reductions]
        bars = ax2.bar(rl_methods, reductions, color=colors, edgecolor='black', linewidth=0.5)
        ax2.axhline(y=0, color='black', linewidth=0.5)
        ax2.set_ylabel('Reduction (%)')
        ax2.set_title('Latency Reduction')
        ax2.set_xticklabels(rl_methods, rotation=45, ha='right')
        
        for bar, red in zip(bars, reductions):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{red:+.1f}%', ha='center', va='bottom' if red > 0 else 'top', 
                    fontsize=10, fontweight='bold')
        
        # === Panel 3: Wire Overhead Reduction ===
        ax3 = axes[2]
        wire_reds = []
        for m in rl_methods:
            red = ((baseline.mean_wire - self.results[m].mean_wire) / baseline.mean_wire) * 100
            wire_reds.append(red)
        
        colors = ['#2ca02c' if r > 0 else '#d62728' for r in wire_reds]
        bars = ax3.bar(rl_methods, wire_reds, color=colors, edgecolor='black', linewidth=0.5)
        ax3.axhline(y=0, color='black', linewidth=0.5)
        ax3.set_ylabel('Reduction (%)')
        ax3.set_title('Wire Overhead Reduction')
        ax3.set_xticklabels(rl_methods, rotation=45, ha='right')
        
        for bar, red in zip(bars, wire_reds):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{red:+.1f}%', ha='center', va='bottom' if red > 0 else 'top', 
                    fontsize=10, fontweight='bold')
        
        fig.suptitle('RL Methods Improvement vs Rule-Based Baseline', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        path = os.path.join(self.output_dir, "improvement_vs_rule_based.png")
        plt.savefig(path)
        plt.close()
        return path
    
    def plot_security_comparison(self) -> str:
        """Security metrics: violation rate and PQC-safe usage."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        methods = list(self.results.keys())
        
        # === Left: Violation Rate ===
        ax1 = axes[0]
        violations = [self.results[m].violation_rate * 100 for m in methods]
        colors = ['#2ca02c' if v < 5 else '#d62728' for v in violations]
        
        bars = ax1.barh(methods, violations, color=colors, edgecolor='black', linewidth=0.5)
        ax1.axvline(x=5, color='red', linestyle='--', linewidth=1.5, label='5% threshold')
        ax1.set_xlabel('Violation Rate (%)')
        ax1.set_title('Security Violation Rate (Level < 3)')
        ax1.legend(loc='lower right')
        
        # === Right: Classical Usage (should be 0%) ===
        ax2 = axes[1]
        classical = [self.results[m].classical_rate * 100 for m in methods]
        colors = ['#2ca02c' if c == 0 else '#d62728' for c in classical]
        
        bars = ax2.barh(methods, classical, color=colors, edgecolor='black', linewidth=0.5)
        ax2.set_xlabel('Classical Only Rate (%)')
        ax2.set_title('CLASSICAL_ONLY Usage (Should be 0%)')
        ax2.set_xlim(0, max(10, max(classical) * 1.2))
        
        plt.tight_layout()
        path = os.path.join(self.output_dir, "security_comparison.png")
        plt.savefig(path)
        plt.close()
        return path
    
    def plot_latency_vs_rtt_lines(self) -> str:
        """Line plot showing latency trend across RTT values."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        rtts = self._get_rtts()
        
        for name, m in self.results.items():
            if m.per_rtt:
                latencies = [m.per_rtt.get(rtt, {}).get("median_latency", 0) for rtt in rtts]
                ax.plot(rtts, latencies, marker='o', label=name, 
                       color=self.colors.get(name, None), linewidth=2, markersize=8)
        
        ax.set_xlabel('RTT (ms)')
        ax.set_ylabel('Median Latency (ms)')
        ax.set_title('Latency vs RTT: All Methods')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        path = os.path.join(self.output_dir, "latency_vs_rtt_lines.png")
        plt.savefig(path)
        plt.close()
        return path
    
    def generate_all_figures(self) -> List[str]:
        """Generate all publication figures."""
        paths = []
        
        print("\n[Visualizer] Generating publication-quality figures...")
        
        # 1. Best RL vs Rule-Based (main figure for paper)
        path = self.plot_best_rl_vs_rule_latency()
        if path:
            paths.append(path)
        
        # 2. Individual RL vs Rule-Based comparisons
        rl_methods = [k for k in self.results.keys() if k not in ["Rule_Based", "Oracle"]]
        for rl_name in rl_methods:
            path = self.plot_rl_vs_rule_latency(rl_name)
            if path:
                paths.append(path)
                print(f"  Created: latency_{rl_name}_vs_rule.png")
        
        # 3. All methods comparison
        path = self.plot_all_methods_comparison()
        if path:
            paths.append(path)
            print(f"  Created: latency_all_methods_comparison.png")
        
        # 4. Reward comparison
        path = self.plot_reward_comparison()
        paths.append(path)
        print(f"  Created: reward_comparison.png")
        
        # 5. Policy distribution
        path = self.plot_policy_distribution()
        paths.append(path)
        print(f"  Created: policy_distribution.png")
        
        # 6. Improvement bars
        path = self.plot_improvement_bars()
        if path:
            paths.append(path)
            print(f"  Created: improvement_vs_rule_based.png")
        
        # 7. Security comparison
        path = self.plot_security_comparison()
        paths.append(path)
        print(f"  Created: security_comparison.png")
        
        # 8. Latency vs RTT lines
        path = self.plot_latency_vs_rtt_lines()
        paths.append(path)
        print(f"  Created: latency_vs_rtt_lines.png")
        
        return paths


# ============================================================================
# LaTeX Table Generator for IEEE CNS Paper
# ============================================================================

def generate_latex_tables(results: Dict[str, EvalMetrics], output_dir: str) -> None:
    """Generate LaTeX-ready tables for IEEE CNS paper."""
    os.makedirs(output_dir, exist_ok=True)
    
    # === Table 1: Main Comparison Table ===
    latex_main = r"""\begin{table}[htbp]
\centering
\caption{Performance Comparison: RL Methods vs Rule-Based Baseline}
\label{tab:comparison}
\begin{tabular}{lcccccc}
\toprule
\textbf{Method} & \textbf{Reward} & \textbf{Latency} & \textbf{Wire} & \textbf{Hybrid} & \textbf{PQC} & \textbf{Viol.} \\
 & & (ms) & (B) & (\%) & (\%) & (\%) \\
\midrule
"""
    
    # Order: Oracle, Rule_Based, then RL sorted by reward
    rl_methods = [k for k in results.keys() if k not in ["Oracle", "Rule_Based"]]
    rl_sorted = sorted(rl_methods, key=lambda x: results[x].mean_reward, reverse=True)
    
    order = []
    if "Oracle" in results:
        order.append("Oracle")
    if "Rule_Based" in results:
        order.append("Rule_Based")
    order.extend(rl_sorted)
    
    for name in order:
        if name not in results:
            continue
        m = results[name]
        latex_name = name.replace("_", "\\_")
        latex_main += f"{latex_name} & {m.mean_reward:.2f} & {m.mean_latency:.1f} & {m.mean_wire:.0f} & {m.hybrid_rate*100:.1f} & {m.pqc_rate*100:.1f} & {m.violation_rate*100:.1f} \\\\\n"
    
    latex_main += r"""\bottomrule
\end{tabular}
\end{table}
"""
    
    with open(os.path.join(output_dir, "comparison_table.tex"), "w") as f:
        f.write(latex_main)
    print(f"  LaTeX saved: {output_dir}/comparison_table.tex")
    
    # === Table 2: Improvement vs Rule-Based ===
    if "Rule_Based" in results and rl_methods:
        baseline = results["Rule_Based"]
        
        latex_imp = r"""\begin{table}[htbp]
\centering
\caption{RL Methods Improvement vs Rule-Based Baseline}
\label{tab:improvement}
\begin{tabular}{lccc}
\toprule
\textbf{Method} & \textbf{Reward $\Delta$} & \textbf{Latency $\Delta$} & \textbf{Wire $\Delta$} \\
 & (\%) & (\%) & (\%) \\
\midrule
"""
        
        for name in rl_sorted:
            m = results[name]
            reward_imp = ((m.mean_reward - baseline.mean_reward) / abs(baseline.mean_reward)) * 100
            latency_red = ((baseline.mean_latency - m.mean_latency) / baseline.mean_latency) * 100
            wire_red = ((baseline.mean_wire - m.mean_wire) / baseline.mean_wire) * 100
            
            latex_name = name.replace("_", "\\_")
            latex_imp += f"{latex_name} & {reward_imp:+.1f} & {latency_red:+.1f} & {wire_red:+.1f} \\\\\n"
        
        latex_imp += r"""\bottomrule
\end{tabular}
\end{table}
"""
        
        with open(os.path.join(output_dir, "improvement_table.tex"), "w") as f:
            f.write(latex_imp)
        print(f"  LaTeX saved: {output_dir}/improvement_table.tex")


# ============================================================================
# Main Entry Point
# ============================================================================

def run_comprehensive_evaluation(
    algorithms: List[str] = None,
    generate_figures: bool = True,
) -> Dict[str, EvalMetrics]:
    """
    Run comprehensive RL vs Rule-Based evaluation.
    
    Args:
        algorithms: List of RL algorithms to evaluate
        generate_figures: Whether to generate publication figures
        
    Returns:
        Dictionary of evaluation results
    """
    path_config = DEFAULT_PATH_CONFIG
    
    # Evaluate
    evaluator = ComprehensiveEvaluator(path_config=path_config)
    evaluator.load_data()
    results = evaluator.evaluate_all(algorithms=algorithms)
    evaluator.save_results()
    
    # Generate figures
    if generate_figures:
        visualizer = PublicationVisualizer(
            results=results,
            output_dir=path_config.figures_dir,
        )
        visualizer.generate_all_figures()
    
    # Generate LaTeX tables
    latex_dir = os.path.join(path_config.eval_dir, "latex")
    generate_latex_tables(results, latex_dir)
    
    return results


if __name__ == "__main__":
    run_comprehensive_evaluation()
