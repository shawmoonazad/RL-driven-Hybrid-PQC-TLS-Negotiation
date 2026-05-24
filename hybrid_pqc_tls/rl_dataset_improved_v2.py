# -*- coding: utf-8 -*-
"""
Improved Offline RL Dataset Builder v2 for Hybrid PQC-TLS
=========================================================
Key improvements over v1:
1. Higher behavior_epsilon (0.6) for more diverse exploration
2. Reward-proportional exploration (softmax over rewards, not just PQC-safe bias)
3. Minimum action coverage guarantee (at least 2% per action)
4. Safety-aware sampling that still includes violations for analysis

This creates a dataset where:
- ~40% goes to optimal actions (vs 70% in v1)
- ~45% explores other PQC-safe actions proportional to their rewards
- ~15% explores unsafe actions (for safety-performance tradeoff analysis)
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


class ImprovedDatasetBuilderV2:
    """
    Creates a more diverse offline RL dataset for better algorithm differentiation.
    
    Key changes:
    1. behavior_epsilon = 0.6 (was 0.3)
    2. Reward-proportional exploration using softmax
    3. Minimum 2% coverage per action
    4. Keeps unsafe actions for safety-performance analysis
    """
    
    def __init__(
        self,
        reward_config: RewardConfig = DEFAULT_REWARD_CONFIG,
        path_config: PathConfig = DEFAULT_PATH_CONFIG,
        behavior_epsilon: float = 0.6,      # INCREASED from 0.3
        min_action_coverage: float = 0.02,  # At least 2% per action
        exploration_temperature: float = 2.0,  # Softmax temperature for exploration
    ):
        self.reward_config = reward_config
        self.path_config = path_config
        self.behavior_epsilon = behavior_epsilon
        self.min_action_coverage = min_action_coverage
        self.exploration_temperature = exploration_temperature
        
        # Normalization parameters
        self.state_mean: Optional[np.ndarray] = None
        self.state_std: Optional[np.ndarray] = None
        
        # Computed optimal actions and rewards
        self.optimal_actions: Dict[float, int] = {}
        self.action_rewards: Dict[Tuple[float, int], float] = {}
    
    def _compute_reward(
        self,
        row: pd.Series,
        policy_name: str,
        level_int: int,
    ) -> float:
        """Compute reward with RTT-dependent coefficients and mode bonuses."""
        cfg = self.reward_config
        
        rtt_ms = float(row.get("rtt_ms", 0.0))
        latency_ms = float(row.get("total_time_ms", 0.0))
        
        wire_ec = float(row.get("wire_srv_ecdh", 0) or 0) + float(row.get("wire_cli_ecdh", 0) or 0)
        wire_pq = float(row.get("wire_srv_pqc", 0) or 0) + float(row.get("wire_cli_pqc", 0) or 0)
        total_wire_kb = (wire_ec + wire_pq) / 1000.0
        
        alpha = cfg.get_alpha(rtt_ms)
        security_weight = cfg.get_security_weight(rtt_ms)
        security_score = level_int / 5.0
        mode_bonus = cfg.get_mode_bonus(policy_name)
        level_violation = cfg.level_violation_penalty if level_int < cfg.min_acceptable_level else 0.0
        
        reward = (
            1.0
            - alpha * latency_ms
            - cfg.beta * total_wire_kb
            + security_weight * security_score
            + mode_bonus
            - level_violation
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
        """Analyze the evaluation grid to find optimal and suboptimal actions per RTT."""
        print("\n[Analysis] Computing action rewards per RTT...")
        
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
        Compute behavior policy with reward-proportional exploration.
        
        New approach:
        1. Optimal action gets (1 - epsilon) probability
        2. Remaining epsilon is distributed proportional to softmax(rewards/temperature)
        3. Ensure minimum coverage for each action
        """
        probs = np.zeros(NUM_ACTIONS)
        
        optimal = self.optimal_actions.get(rtt, 4)  # Default to ALLOW_FALLBACK L3
        
        # Optimal action gets (1 - epsilon) probability
        probs[optimal] = 1.0 - self.behavior_epsilon
        
        # Get rewards for all actions at this RTT
        rewards = np.array([
            self.action_rewards.get((rtt, i), -100.0)  # Default to very low reward
            for i in range(NUM_ACTIONS)
        ])
        
        # Softmax over rewards for exploration distribution
        # Shift rewards for numerical stability
        shifted_rewards = rewards - rewards.max()
        exp_rewards = np.exp(shifted_rewards / self.exploration_temperature)
        exp_rewards[optimal] = 0.0  # Don't double-count optimal
        
        if exp_rewards.sum() > 0:
            explore_probs = exp_rewards / exp_rewards.sum()
        else:
            # Fallback to uniform if all zeros
            explore_probs = np.ones(NUM_ACTIONS) / (NUM_ACTIONS - 1)
            explore_probs[optimal] = 0.0
        
        probs += self.behavior_epsilon * explore_probs
        
        # Ensure minimum coverage for each action
        for i in range(NUM_ACTIONS):
            if probs[i] < self.min_action_coverage:
                deficit = self.min_action_coverage - probs[i]
                probs[i] = self.min_action_coverage
                # Take from optimal action (which has the most probability)
                probs[optimal] -= deficit
        
        # Ensure optimal action doesn't go negative
        probs[optimal] = max(probs[optimal], self.min_action_coverage)
        
        # Normalize
        probs = probs / probs.sum()
        
        return probs
    
    def _create_behavioral_dataset(
        self,
        df: pd.DataFrame,
        target_size: int = 10000,
    ) -> pd.DataFrame:
        """Re-sample the evaluation grid according to the improved behavior policy."""
        print(f"\n[Sampling] Creating diverse behavioral dataset (target size: {target_size})...")
        print(f"  behavior_epsilon: {self.behavior_epsilon}")
        print(f"  min_action_coverage: {self.min_action_coverage}")
        print(f"  exploration_temperature: {self.exploration_temperature}")
        
        sampled_rows = []
        unique_rtts = sorted(df["rtt_ms"].unique())
        
        samples_per_rtt = target_size // len(unique_rtts)
        
        for rtt in unique_rtts:
            rtt_data = df[df["rtt_ms"] == rtt]
            behavior_probs = self._compute_behavior_policy(rtt)
            
            # Print behavior policy for this RTT
            print(f"\n  RTT={rtt}ms behavior policy:")
            for i, (p, l) in enumerate(ACTION_LIST):
                if behavior_probs[i] > 0.01:
                    marker = "★" if i == self.optimal_actions.get(rtt) else " "
                    print(f"    {marker} {p:20s} L{l}: {behavior_probs[i]*100:5.1f}%")
            
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
        print("\n" + "="*60)
        print("[Sampling] Final action distribution:")
        print("="*60)
        
        safe_count = 0
        unsafe_count = 0
        
        for action_idx, (policy_name, level_int) in enumerate(ACTION_LIST):
            count = ((result_df["policy"] == policy_name) & 
                    (result_df["level_int"] == level_int)).sum()
            pct = 100.0 * count / len(result_df)
            is_safe = policy_name != "CLASSICAL_ONLY" and level_int >= 3
            marker = "✓" if is_safe else "✗"
            
            if is_safe:
                safe_count += count
            else:
                unsafe_count += count
            
            print(f"  {marker} {action_idx:2d}: {policy_name:20s} L{level_int} -> {count:4d} ({pct:5.1f}%)")
        
        print(f"\n  PQC-Safe actions: {safe_count} ({100*safe_count/len(result_df):.1f}%)")
        print(f"  Unsafe actions:   {unsafe_count} ({100*unsafe_count/len(result_df):.1f}%)")
        
        return result_df
    
    def build(
        self,
        csv_path: Optional[str] = None,
        train_split: float = 0.8,
        seed: int = 42,
        target_size: int = 10000,
    ) -> Dict[str, np.ndarray]:
        """Build the improved offline RL dataset."""
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
        
        # Train/test split (stratified by action to ensure all actions in both sets)
        indices = np.arange(len(S))
        train_idx, test_idx = train_test_split(
            indices, train_size=train_split, random_state=seed, shuffle=True,
            stratify=A  # STRATIFIED SPLIT
        )
        
        print(f"\n[Dataset] Final dataset: {len(S)} transitions")
        print(f"[Dataset] Train: {len(train_idx)}, Test: {len(test_idx)}")
        print(f"[Dataset] Reward stats - Mean: {R.mean():.3f}, Std: {R.std():.3f}")
        
        # Verify action coverage in train/test
        train_actions = A[train_idx]
        test_actions = A[test_idx]
        print(f"\n[Dataset] Action coverage verification:")
        print(f"  Train set unique actions: {len(np.unique(train_actions))}/{NUM_ACTIONS}")
        print(f"  Test set unique actions: {len(np.unique(test_actions))}/{NUM_ACTIONS}")
        
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


def build_diverse_dataset(
    csv_path: str = None,
    output_path: str = None,
    train_split: float = 0.8,
    seed: int = 42,
    target_size: int = 10000,
    behavior_epsilon: float = 0.6,
    min_action_coverage: float = 0.02,
    exploration_temperature: float = 2.0,
) -> Dict[str, np.ndarray]:
    """
    Convenience function to build diverse dataset with configurable exploration.
    
    Args:
        csv_path: Path to handshake_raw.csv
        output_path: Output path for dataset
        train_split: Train/test split ratio
        seed: Random seed
        target_size: Target number of samples
        behavior_epsilon: Exploration rate (0.6 = 60% non-optimal)
        min_action_coverage: Minimum proportion per action (0.02 = 2%)
        exploration_temperature: Softmax temperature for exploration weights
    """
    builder = ImprovedDatasetBuilderV2(
        behavior_epsilon=behavior_epsilon,
        min_action_coverage=min_action_coverage,
        exploration_temperature=exploration_temperature,
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
    # Build with diverse exploration
    build_diverse_dataset(
        behavior_epsilon=0.6,
        min_action_coverage=0.02,
        exploration_temperature=2.0,
    )
