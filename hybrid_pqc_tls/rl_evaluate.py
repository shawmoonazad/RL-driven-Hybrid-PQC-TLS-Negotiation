# -*- coding: utf-8 -*-
"""
Offline RL Evaluation for Hybrid PQC-TLS
========================================
Comprehensive evaluation comparing RL algorithms against:
1. Rule-based baseline (PERFORMANCE_ADAPTIVE heuristic)
2. Fixed policy baselines (always Hybrid L3, always PQC L3, etc.)
3. Oracle (best possible action per RTT)

Generates:
- Performance comparison tables (CSV)
- Visualization figures (PNG)
- LaTeX-ready tables for paper

Metrics:
- Latency reduction vs baselines
- Wire overhead comparison
- Security violation rates
- Policy distribution analysis
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

from .rl_config import (
    ACTION_LIST,
    ACTION_TO_IDX,
    NUM_ACTIONS,
    STATE_DIM,
    PathConfig,
    DEFAULT_PATH_CONFIG,
    TrainingConfig,
    DEFAULT_TRAINING_CONFIG,
    get_action_policy,
    get_action_level,
    is_classical_action,
)
from .rl_offline_dataset import OfflineRLDatasetBuilder
from .rl_models import create_algorithm, get_available_algorithms


# ============================================================================
# Evaluation Data Structures
# ============================================================================

class EvaluationResult:
    """Container for evaluation results of a single method."""
    
    def __init__(self, name: str):
        self.name = name
        self.actions: np.ndarray = np.array([])
        self.rewards: np.ndarray = np.array([])
        self.latencies: np.ndarray = np.array([])
        self.wire_bytes: np.ndarray = np.array([])
        self.rtts: np.ndarray = np.array([])
        
        # Per-RTT metrics
        self.per_rtt_metrics: Dict[float, Dict[str, float]] = {}
    
    def compute_metrics(self) -> Dict[str, float]:
        """Compute aggregate metrics."""
        metrics = {
            "mean_reward": float(self.rewards.mean()),
            "std_reward": float(self.rewards.std()),
            "mean_latency_ms": float(self.latencies.mean()),
            "median_latency_ms": float(np.median(self.latencies)),
            "p95_latency_ms": float(np.percentile(self.latencies, 95)),
            "mean_wire_bytes": float(self.wire_bytes.mean()),
            "median_wire_bytes": float(np.median(self.wire_bytes)),
        }
        
        # Policy distribution
        action_counts = np.bincount(self.actions, minlength=NUM_ACTIONS)
        action_dist = action_counts / len(self.actions)
        
        metrics["hybrid_rate"] = sum(action_dist[i] for i, (p, l) in enumerate(ACTION_LIST) if p == "REQUIRE_HYBRID")
        metrics["pqc_rate"] = sum(action_dist[i] for i, (p, l) in enumerate(ACTION_LIST) if p == "PQC_ONLY")
        metrics["classical_rate"] = sum(action_dist[i] for i, (p, l) in enumerate(ACTION_LIST) if p == "CLASSICAL_ONLY")
        metrics["fallback_rate"] = sum(action_dist[i] for i, (p, l) in enumerate(ACTION_LIST) if p == "ALLOW_FALLBACK")
        
        # Security metrics
        metrics["violation_rate"] = sum(action_dist[i] for i, (p, l) in enumerate(ACTION_LIST) if l < 3)
        metrics["level_3_plus_rate"] = sum(action_dist[i] for i, (p, l) in enumerate(ACTION_LIST) if l >= 3)
        
        # Level distribution
        metrics["level_1_rate"] = sum(action_dist[i] for i, (p, l) in enumerate(ACTION_LIST) if l == 1)
        metrics["level_3_rate"] = sum(action_dist[i] for i, (p, l) in enumerate(ACTION_LIST) if l == 3)
        metrics["level_5_rate"] = sum(action_dist[i] for i, (p, l) in enumerate(ACTION_LIST) if l == 5)
        
        return metrics


# ============================================================================
# Baseline Implementations
# ============================================================================

class FixedPolicyBaseline:
    """Baseline that always selects a fixed (policy, level) combination."""
    
    def __init__(self, policy_name: str, level: int):
        self.policy_name = policy_name
        self.level = level
        self.action_idx = ACTION_TO_IDX.get((policy_name, level))
        
        if self.action_idx is None:
            raise ValueError(f"Invalid policy/level: {policy_name}, {level}")
    
    def select_action(self, state: np.ndarray) -> int:
        return self.action_idx
    
    @property
    def name(self) -> str:
        return f"Fixed_{self.policy_name}_L{self.level}"


class OracleBaseline:
    """Oracle baseline that selects the best action per RTT from dataset."""
    
    def __init__(self, oracle_mapping: Dict[float, int]):
        self.oracle = oracle_mapping
        self.default_action = list(oracle_mapping.values())[0] if oracle_mapping else 0
    
    def select_action(self, state: np.ndarray, rtt: float) -> int:
        # Find closest RTT in oracle
        closest_rtt = min(self.oracle.keys(), key=lambda x: abs(x - rtt))
        return self.oracle.get(closest_rtt, self.default_action)
    
    @property
    def name(self) -> str:
        return "Oracle"


class RuleBasedBaseline:
    """
    Rule-based PERFORMANCE_ADAPTIVE heuristic.
    Mimics the behavior of PolicyEngine without RL.
    """
    
    def __init__(self):
        pass
    
    def select_action(self, state: np.ndarray) -> int:
        """
        Simple heuristic based on RTT:
        - Low RTT (< 50ms): Can afford full hybrid at L5
        - Medium RTT (50-150ms): Hybrid at L3
        - High RTT (> 150ms): Fallback to PQC L3 for lower overhead
        """
        rtt = state[0]  # RTT is first state dimension (before normalization)
        
        if rtt < 50:
            # Low latency: maximize security
            return ACTION_TO_IDX[("REQUIRE_HYBRID", 5)]
        elif rtt < 100:
            # Medium: balanced
            return ACTION_TO_IDX[("REQUIRE_HYBRID", 3)]
        elif rtt < 150:
            # Higher latency: still secure but more flexible
            return ACTION_TO_IDX[("ALLOW_FALLBACK", 3)]
        else:
            # High latency: prioritize performance, still PQC
            return ACTION_TO_IDX[("PQC_ONLY", 3)]
    
    @property
    def name(self) -> str:
        return "Rule_Based"


# ============================================================================
# Main Evaluator
# ============================================================================

class OfflineRLEvaluator:
    """
    Comprehensive evaluator for offline RL algorithms.
    """
    
    def __init__(
        self,
        path_config: PathConfig = DEFAULT_PATH_CONFIG,
        training_config: TrainingConfig = DEFAULT_TRAINING_CONFIG,
    ):
        self.path_config = path_config
        self.training_config = training_config
        
        # Data
        self.data: Optional[Dict[str, np.ndarray]] = None
        self.test_idx: Optional[np.ndarray] = None
        
        # Results
        self.results: Dict[str, EvaluationResult] = {}
    
    def load_data(self, data_path: Optional[str] = None) -> None:
        """Load dataset for evaluation."""
        if data_path is None:
            data_path = self.path_config.dataset_path
        
        builder = OfflineRLDatasetBuilder()
        self.data = builder.load(data_path)
        self.test_idx = self.data["test_idx"]
        
        print(f"[Evaluator] Loaded {len(self.test_idx)} test samples")
    
    def _get_expected_metrics(self, action_idx: int, rtt: float) -> Dict[str, float]:
        """
        Get expected latency/wire for a given action at a given RTT.
        Uses dataset averages for that (action, RTT) pair.
        """
        # Find matching samples in dataset
        A = self.data["A"]
        RTT = self.data["rtt"]
        LAT = self.data["latency"]
        WIRE = self.data["wire"]
        R = self.data["R"]
        
        # Mask for matching action and similar RTT (±5ms tolerance)
        mask = (A == action_idx) & (np.abs(RTT - rtt) < 5)
        
        if mask.sum() == 0:
            # Fallback: just use action-level averages
            mask = (A == action_idx)
        
        if mask.sum() == 0:
            return {"latency": 0.0, "wire": 0.0, "reward": 0.0}
        
        return {
            "latency": float(LAT[mask].mean()),
            "wire": float(WIRE[mask].mean()),
            "reward": float(R[mask].mean()),
        }
    
    def evaluate_method(
        self,
        method,
        name: str,
        use_raw_states: bool = False,
    ) -> EvaluationResult:
        """
        Evaluate a single method (RL algorithm or baseline).
        
        Args:
            method: Object with select_action(state) method
            name: Method name for results
            use_raw_states: Whether to use raw (unnormalized) states
        """
        result = EvaluationResult(name)
        
        # Get test data
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
            
            # Select action
            if hasattr(method, "select_action"):
                if isinstance(method, OracleBaseline):
                    action = method.select_action(state, rtt)
                elif isinstance(method, RuleBasedBaseline):
                    # Rule-based needs raw RTT
                    raw_state = self.data["S_raw"][self.test_idx[i]]
                    action = method.select_action(raw_state)
                else:
                    action = method.select_action(state)
            else:
                action = method(state)
            
            actions.append(action)
            
            # Get expected metrics for this action
            metrics = self._get_expected_metrics(action, rtt)
            rewards.append(metrics["reward"])
            latencies.append(metrics["latency"])
            wire_bytes.append(metrics["wire"])
        
        result.actions = np.array(actions)
        result.rewards = np.array(rewards)
        result.latencies = np.array(latencies)
        result.wire_bytes = np.array(wire_bytes)
        result.rtts = RTT
        
        # Compute per-RTT metrics
        unique_rtts = np.unique(RTT)
        for rtt in unique_rtts:
            mask = RTT == rtt
            rtt_result = EvaluationResult(f"{name}_rtt{rtt}")
            rtt_result.actions = result.actions[mask]
            rtt_result.rewards = result.rewards[mask]
            rtt_result.latencies = result.latencies[mask]
            rtt_result.wire_bytes = result.wire_bytes[mask]
            result.per_rtt_metrics[rtt] = rtt_result.compute_metrics()
        
        return result
    
    def evaluate_all(
        self,
        algorithms: Optional[List[str]] = None,
        include_baselines: bool = True,
    ) -> Dict[str, EvaluationResult]:
        """
        Evaluate all RL algorithms and baselines.
        """
        if self.data is None:
            self.load_data()
        
        if algorithms is None:
            algorithms = get_available_algorithms()
        
        results = {}
        
        # ===== Evaluate RL Algorithms =====
        print("\n[Evaluator] Evaluating RL Algorithms...")
        for algo_name in algorithms:
            model_path = os.path.join(self.path_config.model_dir, f"{algo_name.lower()}_model.pt")
            
            if not os.path.exists(model_path):
                print(f"  Skipping {algo_name}: model not found at {model_path}")
                continue
            
            print(f"  Evaluating {algo_name}...")
            algorithm = create_algorithm(
                algo_name,
                config=self.training_config,
                device=self.training_config.device,
            )
            algorithm.load(model_path)
            
            result = self.evaluate_method(algorithm, algo_name)
            results[algo_name] = result
        
        # ===== Evaluate Baselines =====
        if include_baselines:
            print("\n[Evaluator] Evaluating Baselines...")
            
            # Rule-based baseline
            print("  Evaluating Rule-Based...")
            rule_based = RuleBasedBaseline()
            results["Rule_Based"] = self.evaluate_method(rule_based, "Rule_Based", use_raw_states=True)
            
            # Fixed policy baselines
            fixed_baselines = [
                ("REQUIRE_HYBRID", 3),
                ("REQUIRE_HYBRID", 5),
                ("PQC_ONLY", 3),
                ("PQC_ONLY", 5),
                ("ALLOW_FALLBACK", 3),
            ]
            
            for policy_name, level in fixed_baselines:
                baseline_name = f"Fixed_{policy_name}_L{level}"
                print(f"  Evaluating {baseline_name}...")
                baseline = FixedPolicyBaseline(policy_name, level)
                results[baseline_name] = self.evaluate_method(baseline, baseline_name)
            
            # Oracle baseline
            if "oracle" in self.data:
                print("  Evaluating Oracle...")
                oracle_data = self.data["oracle"]
                oracle_mapping = {float(rtt): int(action) for rtt, action in oracle_data}
                oracle = OracleBaseline(oracle_mapping)
                results["Oracle"] = self.evaluate_method(oracle, "Oracle", use_raw_states=True)
        
        self.results = results
        return results
    
    def generate_comparison_table(self) -> pd.DataFrame:
        """Generate a comparison table of all methods."""
        rows = []
        
        for name, result in self.results.items():
            metrics = result.compute_metrics()
            row = {"Method": name}
            row.update(metrics)
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Sort by mean reward (descending)
        df = df.sort_values("mean_reward", ascending=False)
        
        return df
    
    def generate_per_rtt_table(self) -> pd.DataFrame:
        """Generate per-RTT comparison table."""
        rows = []
        
        for name, result in self.results.items():
            for rtt, metrics in result.per_rtt_metrics.items():
                row = {
                    "Method": name,
                    "RTT_ms": rtt,
                }
                row.update(metrics)
                rows.append(row)
        
        df = pd.DataFrame(rows)
        return df
    
    def compute_improvement_vs_baseline(
        self,
        baseline_name: str = "Rule_Based",
    ) -> pd.DataFrame:
        """Compute improvement metrics vs a baseline."""
        if baseline_name not in self.results:
            raise ValueError(f"Baseline {baseline_name} not found in results")
        
        baseline_metrics = self.results[baseline_name].compute_metrics()
        
        rows = []
        for name, result in self.results.items():
            if name == baseline_name:
                continue
            
            metrics = result.compute_metrics()
            
            # Compute improvements
            latency_reduction = (baseline_metrics["mean_latency_ms"] - metrics["mean_latency_ms"]) / baseline_metrics["mean_latency_ms"] * 100
            wire_reduction = (baseline_metrics["mean_wire_bytes"] - metrics["mean_wire_bytes"]) / baseline_metrics["mean_wire_bytes"] * 100
            reward_improvement = (metrics["mean_reward"] - baseline_metrics["mean_reward"]) / abs(baseline_metrics["mean_reward"]) * 100
            
            rows.append({
                "Method": name,
                "Latency_Reduction_%": latency_reduction,
                "Wire_Reduction_%": wire_reduction,
                "Reward_Improvement_%": reward_improvement,
                "Violation_Rate": metrics["violation_rate"],
                "Classical_Rate": metrics["classical_rate"],
            })
        
        df = pd.DataFrame(rows)
        df = df.sort_values("Reward_Improvement_%", ascending=False)
        return df
    
    def save_results(self, output_dir: Optional[str] = None) -> None:
        """Save all results to files."""
        if output_dir is None:
            output_dir = self.path_config.eval_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Main comparison table
        df_main = self.generate_comparison_table()
        df_main.to_csv(os.path.join(output_dir, "comparison_table.csv"), index=False)
        print(f"Saved: {output_dir}/comparison_table.csv")
        
        # Per-RTT table
        df_rtt = self.generate_per_rtt_table()
        df_rtt.to_csv(os.path.join(output_dir, "per_rtt_table.csv"), index=False)
        print(f"Saved: {output_dir}/per_rtt_table.csv")
        
        # Improvement table
        if "Rule_Based" in self.results:
            df_improve = self.compute_improvement_vs_baseline("Rule_Based")
            df_improve.to_csv(os.path.join(output_dir, "improvement_vs_rule_based.csv"), index=False)
            print(f"Saved: {output_dir}/improvement_vs_rule_based.csv")
        
        # Print summary
        print("\n" + "="*70)
        print("EVALUATION SUMMARY")
        print("="*70)
        print(df_main.to_string(index=False))


# ============================================================================
# Visualization
# ============================================================================

class ResultVisualizer:
    """Generate publication-quality figures."""
    
    def __init__(self, results: Dict[str, EvaluationResult], output_dir: str):
        self.results = results
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style for publication
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.labelsize'] = 14
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['legend.fontsize'] = 10
        plt.rcParams['figure.figsize'] = (10, 6)
    
    def plot_latency_comparison(self) -> str:
        """Bar plot comparing mean latency across methods."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        methods = []
        latencies = []
        colors = []
        
        # Color scheme
        color_map = {
            "BC": "#1f77b4",
            "CQL": "#ff7f0e",
            "IQL": "#2ca02c",
            "BCQ": "#d62728",
            "AWAC": "#9467bd",
            "Rule_Based": "#8c564b",
            "Oracle": "#e377c2",
        }
        default_color = "#7f7f7f"
        
        for name, result in self.results.items():
            metrics = result.compute_metrics()
            methods.append(name)
            latencies.append(metrics["mean_latency_ms"])
            colors.append(color_map.get(name, default_color))
        
        # Sort by latency
        sorted_idx = np.argsort(latencies)
        methods = [methods[i] for i in sorted_idx]
        latencies = [latencies[i] for i in sorted_idx]
        colors = [colors[i] for i in sorted_idx]
        
        bars = ax.barh(methods, latencies, color=colors)
        ax.set_xlabel("Mean Latency (ms)")
        ax.set_title("Latency Comparison Across Methods")
        
        # Add value labels
        for bar, lat in zip(bars, latencies):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                   f'{lat:.1f}', va='center', fontsize=10)
        
        plt.tight_layout()
        path = os.path.join(self.output_dir, "latency_comparison.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        return path
    
    def plot_reward_comparison(self) -> str:
        """Bar plot comparing mean reward across methods."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        methods = []
        rewards = []
        
        for name, result in self.results.items():
            metrics = result.compute_metrics()
            methods.append(name)
            rewards.append(metrics["mean_reward"])
        
        # Sort by reward (descending)
        sorted_idx = np.argsort(rewards)[::-1]
        methods = [methods[i] for i in sorted_idx]
        rewards = [rewards[i] for i in sorted_idx]
        
        colors = ['#2ca02c' if r > 0 else '#d62728' for r in rewards]
        
        bars = ax.barh(methods, rewards, color=colors)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xlabel("Mean Reward")
        ax.set_title("Reward Comparison Across Methods")
        
        plt.tight_layout()
        path = os.path.join(self.output_dir, "reward_comparison.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        return path
    
    def plot_policy_distribution(self) -> str:
        """Stacked bar plot showing policy distribution for each method."""
        fig, ax = plt.subplots(figsize=(14, 7))
        
        methods = list(self.results.keys())
        
        hybrid_rates = []
        pqc_rates = []
        fallback_rates = []
        classical_rates = []
        
        for name in methods:
            metrics = self.results[name].compute_metrics()
            hybrid_rates.append(metrics["hybrid_rate"] * 100)
            pqc_rates.append(metrics["pqc_rate"] * 100)
            fallback_rates.append(metrics["fallback_rate"] * 100)
            classical_rates.append(metrics["classical_rate"] * 100)
        
        x = np.arange(len(methods))
        width = 0.6
        
        ax.bar(x, hybrid_rates, width, label='REQUIRE_HYBRID', color='#2ca02c')
        ax.bar(x, pqc_rates, width, bottom=hybrid_rates, label='PQC_ONLY', color='#1f77b4')
        ax.bar(x, fallback_rates, width, bottom=np.array(hybrid_rates)+np.array(pqc_rates), 
               label='ALLOW_FALLBACK', color='#ff7f0e')
        ax.bar(x, classical_rates, width, 
               bottom=np.array(hybrid_rates)+np.array(pqc_rates)+np.array(fallback_rates),
               label='CLASSICAL_ONLY', color='#d62728')
        
        ax.set_ylabel('Percentage (%)')
        ax.set_title('Policy Distribution by Method')
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.legend(loc='upper right')
        ax.set_ylim(0, 105)
        
        plt.tight_layout()
        path = os.path.join(self.output_dir, "policy_distribution.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        return path
    
    def plot_latency_per_rtt(self) -> str:
        """Line plot showing latency vs RTT for each method."""
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Filter to main methods
        main_methods = ["BC", "CQL", "IQL", "BCQ", "AWAC", "Rule_Based", "Oracle"]
        
        for name in main_methods:
            if name not in self.results:
                continue
            
            result = self.results[name]
            rtts = sorted(result.per_rtt_metrics.keys())
            latencies = [result.per_rtt_metrics[rtt]["mean_latency_ms"] for rtt in rtts]
            
            ax.plot(rtts, latencies, marker='o', label=name, linewidth=2, markersize=8)
        
        ax.set_xlabel("RTT (ms)")
        ax.set_ylabel("Mean Latency (ms)")
        ax.set_title("Latency vs RTT by Method")
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        path = os.path.join(self.output_dir, "latency_per_rtt.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        return path
    
    def plot_security_metrics(self) -> str:
        """Plot security-related metrics (violation rate, level distribution)."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        methods = list(self.results.keys())
        
        # Left: Violation rate
        ax1 = axes[0]
        violations = [self.results[m].compute_metrics()["violation_rate"] * 100 for m in methods]
        colors = ['#d62728' if v > 5 else '#2ca02c' for v in violations]
        
        bars = ax1.barh(methods, violations, color=colors)
        ax1.set_xlabel("Violation Rate (%)")
        ax1.set_title("Security Level < 3 Usage")
        ax1.axvline(x=5, color='red', linestyle='--', linewidth=1, label='5% threshold')
        
        # Right: Level distribution for RL methods only
        ax2 = axes[1]
        rl_methods = ["BC", "CQL", "IQL", "BCQ", "AWAC"]
        rl_methods = [m for m in rl_methods if m in self.results]
        
        l1_rates = [self.results[m].compute_metrics()["level_1_rate"] * 100 for m in rl_methods]
        l3_rates = [self.results[m].compute_metrics()["level_3_rate"] * 100 for m in rl_methods]
        l5_rates = [self.results[m].compute_metrics()["level_5_rate"] * 100 for m in rl_methods]
        
        x = np.arange(len(rl_methods))
        width = 0.25
        
        ax2.bar(x - width, l1_rates, width, label='Level 1', color='#d62728')
        ax2.bar(x, l3_rates, width, label='Level 3', color='#ff7f0e')
        ax2.bar(x + width, l5_rates, width, label='Level 5', color='#2ca02c')
        
        ax2.set_ylabel('Percentage (%)')
        ax2.set_title('Security Level Distribution (RL Methods)')
        ax2.set_xticks(x)
        ax2.set_xticklabels(rl_methods)
        ax2.legend()
        
        plt.tight_layout()
        path = os.path.join(self.output_dir, "security_metrics.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        return path
    
    def plot_improvement_summary(self) -> str:
        """Plot improvement over rule-based baseline."""
        if "Rule_Based" not in self.results:
            return ""
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        baseline_metrics = self.results["Rule_Based"].compute_metrics()
        
        rl_methods = ["BC", "CQL", "IQL", "BCQ", "AWAC"]
        rl_methods = [m for m in rl_methods if m in self.results]
        
        improvements = []
        for method in rl_methods:
            metrics = self.results[method].compute_metrics()
            # Reward improvement percentage
            imp = (metrics["mean_reward"] - baseline_metrics["mean_reward"]) / abs(baseline_metrics["mean_reward"]) * 100
            improvements.append(imp)
        
        colors = ['#2ca02c' if imp > 0 else '#d62728' for imp in improvements]
        
        bars = ax.bar(rl_methods, improvements, color=colors)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_ylabel("Improvement (%)")
        ax.set_title("Reward Improvement vs Rule-Based Baseline")
        
        # Add value labels
        for bar, imp in zip(bars, improvements):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{imp:.1f}%', ha='center', va='bottom' if height > 0 else 'top',
                   fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        path = os.path.join(self.output_dir, "improvement_summary.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        return path
    
    def generate_all_figures(self) -> List[str]:
        """Generate all figures and return paths."""
        paths = []
        
        print("\n[Visualizer] Generating figures...")
        
        paths.append(self.plot_latency_comparison())
        print(f"  Created: latency_comparison.png")
        
        paths.append(self.plot_reward_comparison())
        print(f"  Created: reward_comparison.png")
        
        paths.append(self.plot_policy_distribution())
        print(f"  Created: policy_distribution.png")
        
        paths.append(self.plot_latency_per_rtt())
        print(f"  Created: latency_per_rtt.png")
        
        paths.append(self.plot_security_metrics())
        print(f"  Created: security_metrics.png")
        
        imp_path = self.plot_improvement_summary()
        if imp_path:
            paths.append(imp_path)
            print(f"  Created: improvement_summary.png")
        
        return paths


# ============================================================================
# Main Entry Point
# ============================================================================

def run_evaluation(
    algorithms: Optional[List[str]] = None,
    include_baselines: bool = True,
    generate_figures: bool = True,
) -> Dict[str, EvaluationResult]:
    """
    Run full evaluation pipeline.
    
    Args:
        algorithms: List of RL algorithms to evaluate
        include_baselines: Whether to include baseline comparisons
        generate_figures: Whether to generate visualization figures
        
    Returns:
        Dictionary of evaluation results
    """
    path_config = DEFAULT_PATH_CONFIG
    
    # Initialize evaluator
    evaluator = OfflineRLEvaluator(path_config=path_config)
    evaluator.load_data()
    
    # Run evaluation
    results = evaluator.evaluate_all(
        algorithms=algorithms,
        include_baselines=include_baselines,
    )
    
    # Save results
    evaluator.save_results()
    
    # Generate figures
    if generate_figures:
        visualizer = ResultVisualizer(
            results=results,
            output_dir=path_config.figures_dir,
        )
        visualizer.generate_all_figures()
    
    return results


if __name__ == "__main__":
    run_evaluation()
