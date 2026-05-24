# -*- coding: utf-8 -*-
"""
Improved Offline RL Dataset Builder for Hybrid PQC-TLS
======================================================
Creates a meaningful offline RL dataset from the evaluation grid by:

1. Computing the "optimal" action per (RTT, context) based on reward
2. Creating a synthetic behavior policy that mostly picks good actions
   but sometimes explores (to have diverse data)
3. Re-sampling the evaluation grid according to this behavior policy

This transforms the uniform evaluation grid into a dataset that
resembles real deployment data where a reasonably good (but imperfect)
policy was used.
"""

import os
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional, List
from sklearn.model_selection import train_test_split
from collections import defaultdict

from .rl_config import (
    ACTION_LIST,
    ACTION_TO_IDX,
    NUM_ACTIONS,
    STATE_DIM,
    RewardConfig,
    DEFAULT_REWARD_CONFIG,
    DEFAULT_PATH_CONFIG,
    PathConfig,
)


class ImprovedDatasetBuilder:
    """
    Creates a meaningful offline RL dataset from evaluation grid data.
    
    The key insight: your evaluation grid has every (action, RTT) pair
    tested equally. We use this to compute which actions are BEST for
    each RTT, then create a synthetic dataset that mimics a "smart but
    imperfect" behavior policy.
    """
    
    def __init__(
        self,
        reward_config: RewardConfig = DEFAULT_REWARD_CONFIG,
        path_config: PathConfig = DEFAULT_PATH_CONFIG,
        behavior_epsilon: float = 0.3,  # 30% exploration
        prefer_pqc_safe: bool = True,   # Bias toward PQC-safe actions
    ):
        self.reward_config = reward_config
        self.path_config = path_config
        self.behavior_epsilon = behavior_epsilon
        self.prefer_pqc_safe = prefer_pqc_safe
        
        # Normalization parameters
        self.state_mean: Optional[np.ndarray] = None
        self.state_std: Optional[np.ndarray] = None
        
        # Computed optimal actions
        self.optimal_actions: Dict[float, int] = {}
        self.action_rewards: Dict[Tuple[float, int], float] = {}
    
    def _compute_reward(
        self,
        row: pd.Series,
        policy_name: str,
        level_int: int,
    ) -> float:
        """
        Compute reward with mode bonuses.
        
        Reward = base
                 - α(rtt) * latency
                 - β * wire_kb
                 + γ(rtt) * security_level
                 + mode_bonus              # NEW: HYBRID/PQC get bonus
                 - level_violation_penalty
        """
        cfg = self.reward_config
        
        rtt_ms = float(row.get("rtt_ms", 0.0))
        latency_ms = float(row.get("total_time_ms", 0.0))
        
        wire_ec = float(row.get("wire_srv_ecdh", 0) or 0) + float(row.get("wire_cli_ecdh", 0) or 0)
        wire_pq = float(row.get("wire_srv_pqc", 0) or 0) + float(row.get("wire_cli_pqc", 0) or 0)
        total_wire_kb = (wire_ec + wire_pq) / 1000.0
        
        alpha = cfg.get_alpha(rtt_ms)
        security_weight = cfg.get_security_weight(rtt_ms)
        
        security_score = level_int / 5.0
        
        # Mode bonus/penalty (KEY CHANGE)
        mode_bonus = cfg.get_mode_bonus(policy_name)
        
        # Level violation penalty
        level_violation = cfg.level_violation_penalty if level_int < cfg.min_acceptable_level else 0.0
        
        reward = (
            1.0                                # Base success reward
            - alpha * latency_ms               # Latency penalty (RTT-dependent)
            - cfg.beta * total_wire_kb         # Wire overhead penalty
            + security_weight * security_score # Security level bonus
            + mode_bonus                       # Policy mode bonus/penalty
            - level_violation                  # Level < 3 penalty
        )
        
        return float(reward)
    
    def _extract_state(self, row: pd.Series) -> np.ndarray:
        """Extract 15-dimensional state vector."""
        def safe_float(col: str) -> float:
            val = row.get(col, 0.0)
            if pd.isna(val):
                return 0.0
            return float(val)
        
        wire_ec = safe_float("wire_srv_ecdh") + safe_float("wire_cli_ecdh")
        wire_pq = safe_float("wire_srv_pqc") + safe_float("wire_cli_pqc")
        
        return np.array([
            safe_float("rtt_ms"),
            float(row.get("trial", 0)),
            float(row.get("using_mock_pqc", 0)),
            float(row.get("strict_pqc", 0)),
            safe_float("kex_keygen_ms"),
            safe_float("kex_encaps_ms"),
            safe_float("kex_decaps_ms"),
            safe_float("kex_hkdf_c_ms"),
            safe_float("kex_hkdf_s_ms"),
            safe_float("auth_sign_c_ms"),
            safe_float("auth_sign_q_ms"),
            safe_float("auth_ver_c_ms"),
            safe_float("auth_ver_q_ms"),
            wire_ec,
            wire_pq,
        ], dtype=np.float32)
    
    def _analyze_evaluation_grid(self, df: pd.DataFrame) -> None:
        """
        Analyze the evaluation grid to find optimal actions per RTT.
        """
        print("\n[Analysis] Computing optimal actions per RTT...")
        
        # Group by RTT and action, compute mean reward
        for rtt in df["rtt_ms"].unique():
            rtt_data = df[df["rtt_ms"] == rtt]
            
            best_action = None
            best_reward = -np.inf
            
            print(f"\n  RTT = {rtt}ms:")
            
            for action_idx, (policy_name, level_int) in enumerate(ACTION_LIST):
                mask = (rtt_data["policy"] == policy_name) & (rtt_data["level_int"] == level_int)
                if mask.sum() == 0:
                    continue
                
                action_data = rtt_data[mask]
                rewards = [
                    self._compute_reward(row, policy_name, level_int)
                    for _, row in action_data.iterrows()
                ]
                mean_reward = np.mean(rewards)
                
                self.action_rewards[(rtt, action_idx)] = mean_reward
                
                # Mark PQC-safe actions
                is_safe = policy_name != "CLASSICAL_ONLY" and level_int >= 3
                safe_marker = "✓" if is_safe else "✗"
                
                if mean_reward > best_reward:
                    best_reward = mean_reward
                    best_action = action_idx
                
                print(f"    {safe_marker} {policy_name:20s} L{level_int}: reward={mean_reward:7.3f}")
            
            self.optimal_actions[rtt] = best_action
            opt_policy, opt_level = ACTION_LIST[best_action]
            print(f"  -> Optimal: {opt_policy} L{opt_level} (reward={best_reward:.3f})")
    
    def _compute_behavior_policy(self, rtt: float) -> np.ndarray:
        """
        Compute behavior policy probabilities for a given RTT.
        
        The behavior policy:
        1. Puts most probability on the optimal action
        2. Explores other actions with probability epsilon
        3. If prefer_pqc_safe, biases exploration toward PQC-safe actions
        """
        probs = np.zeros(NUM_ACTIONS)
        
        optimal = self.optimal_actions.get(rtt, 4)  # Default to ALLOW_FALLBACK L3
        
        # Optimal action gets (1 - epsilon) probability
        probs[optimal] = 1.0 - self.behavior_epsilon
        
        # Distribute epsilon among other actions
        if self.prefer_pqc_safe:
            # Bias toward PQC-safe actions
            pqc_safe_mask = np.array([
                (ACTION_LIST[i][0] != "CLASSICAL_ONLY" and ACTION_LIST[i][1] >= 3)
                for i in range(NUM_ACTIONS)
            ])
            
            # More weight to PQC-safe actions
            explore_weights = np.where(pqc_safe_mask, 3.0, 1.0)
            explore_weights[optimal] = 0.0  # Don't double-count optimal
            explore_weights = explore_weights / explore_weights.sum()
            
            probs += self.behavior_epsilon * explore_weights
        else:
            # Uniform exploration
            remaining = self.behavior_epsilon / (NUM_ACTIONS - 1)
            probs += remaining
            probs[optimal] -= remaining  # Correct for optimal
        
        # Normalize
        probs = probs / probs.sum()
        
        return probs
    
    def _create_behavioral_dataset(
        self,
        df: pd.DataFrame,
        target_size: int = 10000,
    ) -> pd.DataFrame:
        """
        Re-sample the evaluation grid according to the behavior policy.
        
        This creates a dataset that looks like it came from a real
        deployment where a smart (but not perfect) policy was running.
        """
        print(f"\n[Sampling] Creating behavioral dataset (target size: {target_size})...")
        
        sampled_rows = []
        unique_rtts = sorted(df["rtt_ms"].unique())
        
        samples_per_rtt = target_size // len(unique_rtts)
        
        for rtt in unique_rtts:
            rtt_data = df[df["rtt_ms"] == rtt]
            behavior_probs = self._compute_behavior_policy(rtt)
            
            # Sample actions according to behavior policy
            sampled_actions = np.random.choice(
                NUM_ACTIONS,
                size=samples_per_rtt,
                p=behavior_probs,
            )
            
            # For each sampled action, pick a random row with that action
            for action_idx in sampled_actions:
                policy_name, level_int = ACTION_LIST[action_idx]
                
                matching = rtt_data[
                    (rtt_data["policy"] == policy_name) & 
                    (rtt_data["level_int"] == level_int)
                ]
                
                if len(matching) > 0:
                    sampled_row = matching.sample(n=1).iloc[0]
                    sampled_rows.append(sampled_row)
        
        result_df = pd.DataFrame(sampled_rows).reset_index(drop=True)
        
        # Print new action distribution
        print("\n[Sampling] New action distribution:")
        for action_idx, (policy_name, level_int) in enumerate(ACTION_LIST):
            count = ((result_df["policy"] == policy_name) & 
                    (result_df["level_int"] == level_int)).sum()
            pct = 100.0 * count / len(result_df)
            is_safe = policy_name != "CLASSICAL_ONLY" and level_int >= 3
            marker = "✓" if is_safe else "✗"
            print(f"  {marker} {action_idx:2d}: {policy_name:20s} L{level_int} -> {count:4d} ({pct:5.1f}%)")
        
        return result_df
    
    def build(
        self,
        csv_path: Optional[str] = None,
        train_split: float = 0.8,
        seed: int = 42,
        target_size: int = 10000,
    ) -> Dict[str, np.ndarray]:
        """
        Build the improved offline RL dataset.
        """
        np.random.seed(seed)
        
        if csv_path is None:
            csv_path = self.path_config.raw_data_csv
        
        print(f"[Dataset] Loading evaluation grid from: {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"[Dataset] Loaded {len(df)} rows")
        
        # Step 1: Analyze the grid to find optimal actions
        self._analyze_evaluation_grid(df)
        
        # Step 2: Create behavioral dataset by re-sampling
        behavioral_df = self._create_behavioral_dataset(df, target_size=target_size)
        
        # Step 3: Build arrays
        states = []
        actions = []
        rewards = []
        rtts = []
        latencies = []
        wire_bytes = []
        
        for _, row in behavioral_df.iterrows():
            policy_name = str(row["policy"])
            level_int = int(row["level_int"])
            
            action_key = (policy_name, level_int)
            if action_key not in ACTION_TO_IDX:
                continue
            
            state = self._extract_state(row)
            action = ACTION_TO_IDX[action_key]
            reward = self._compute_reward(row, policy_name, level_int)
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            
            rtts.append(float(row.get("rtt_ms", 0)))
            latencies.append(float(row.get("total_time_ms", 0)))
            wire_ec = float(row.get("wire_srv_ecdh", 0) or 0) + float(row.get("wire_cli_ecdh", 0) or 0)
            wire_pq = float(row.get("wire_srv_pqc", 0) or 0) + float(row.get("wire_cli_pqc", 0) or 0)
            wire_bytes.append(wire_ec + wire_pq)
        
        S = np.array(states, dtype=np.float32)
        A = np.array(actions, dtype=np.int64)
        R = np.array(rewards, dtype=np.float32)
        
        # Normalize states
        self.state_mean = np.mean(S, axis=0)
        self.state_std = np.std(S, axis=0)
        self.state_std = np.where(self.state_std < 1e-8, 1.0, self.state_std)
        
        S_norm = (S - self.state_mean) / self.state_std
        S_next_norm = np.vstack([S_norm[1:], S_norm[-1:]])
        D = np.zeros(len(S), dtype=np.float32)
        D[-1] = 1.0
        
        RTT = np.array(rtts, dtype=np.float32)
        LAT = np.array(latencies, dtype=np.float32)
        WIRE = np.array(wire_bytes, dtype=np.float32)
        
        # Train/test split
        indices = np.arange(len(S))
        train_idx, test_idx = train_test_split(
            indices, train_size=train_split, random_state=seed, shuffle=True
        )
        
        print(f"\n[Dataset] Final dataset: {len(S)} transitions")
        print(f"[Dataset] Train: {len(train_idx)}, Test: {len(test_idx)}")
        print(f"[Dataset] Reward stats - Mean: {R.mean():.3f}, Std: {R.std():.3f}")
        
        # Compute optimal actions for oracle
        oracle_data = [(rtt, action) for rtt, action in self.optimal_actions.items()]
        
        return {
            "S": S_norm,
            "A": A,
            "R": R,
            "S_next": S_next_norm,
            "done": D,
            "S_raw": S,
            "rtt": RTT,
            "latency": LAT,
            "wire": WIRE,
            "state_mean": self.state_mean,
            "state_std": self.state_std,
            "train_idx": train_idx,
            "test_idx": test_idx,
            "action_vocab": np.array(ACTION_LIST, dtype=object),
            "oracle": np.array(oracle_data, dtype=object),
            "action_rewards": np.array(list(self.action_rewards.items()), dtype=object),
        }
    
    def save(self, data: Dict[str, np.ndarray], output_path: Optional[str] = None) -> str:
        if output_path is None:
            output_path = self.path_config.dataset_path
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.savez_compressed(output_path, **data)
        print(f"[Dataset] Saved to: {output_path}")
        return output_path


def build_improved_dataset(
    csv_path: str = None,
    output_path: str = None,
    train_split: float = 0.8,
    seed: int = 42,
    target_size: int = 10000,
    behavior_epsilon: float = 0.3,
) -> Dict[str, np.ndarray]:
    """
    Convenience function to build improved dataset.
    """
    builder = ImprovedDatasetBuilder(
        behavior_epsilon=behavior_epsilon,
        prefer_pqc_safe=True,
    )
    data = builder.build(
        csv_path=csv_path,
        train_split=train_split,
        seed=seed,
        target_size=target_size,
    )
    builder.save(data, output_path)
    return data


if __name__ == "__main__":
    build_improved_dataset()
