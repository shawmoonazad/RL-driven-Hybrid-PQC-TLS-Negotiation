# -*- coding: utf-8 -*-
"""
Offline RL Dataset Builder for Hybrid PQC-TLS
=============================================
Processes handshake_raw.csv into a proper offline RL dataset with:
- 15-dimensional state vectors (z-score normalized)
- RTT-dependent reward function
- Heavy penalty for CLASSICAL_ONLY actions
- Train/test split support

Output: NPZ file with S, A, R, S_next, done, and metadata
"""

import os
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
from sklearn.model_selection import train_test_split

from .rl_config import (
    ACTION_LIST,
    ACTION_TO_IDX,
    STATE_DIM,
    STATE_FEATURES,
    RewardConfig,
    DEFAULT_REWARD_CONFIG,
    DEFAULT_PATH_CONFIG,
    PathConfig,
)


class OfflineRLDatasetBuilder:
    """
    Builds offline RL dataset from handshake evaluation data.
    
    Features:
    - Z-score normalization of states
    - RTT-dependent reward shaping
    - Heavy classical penalty
    - Proper next-state handling
    """
    
    def __init__(
        self,
        reward_config: RewardConfig = DEFAULT_REWARD_CONFIG,
        path_config: PathConfig = DEFAULT_PATH_CONFIG,
    ):
        self.reward_config = reward_config
        self.path_config = path_config
        
        # Normalization parameters (computed from data)
        self.state_mean: Optional[np.ndarray] = None
        self.state_std: Optional[np.ndarray] = None
        
    def _extract_state(self, row: pd.Series) -> np.ndarray:
        """
        Extract 15-dimensional state vector from a dataframe row.
        
        State layout:
        [0]  rtt_ms
        [1]  trial
        [2]  using_mock_pqc
        [3]  strict_pqc
        [4]  kex_keygen_ms
        [5]  kex_encaps_ms
        [6]  kex_decaps_ms
        [7]  kex_hkdf_c_ms
        [8]  kex_hkdf_s_ms
        [9]  auth_sign_c_ms
        [10] auth_sign_q_ms
        [11] auth_ver_c_ms
        [12] auth_ver_q_ms
        [13] total_wire_ec (srv + cli ECDH)
        [14] total_wire_pq (srv + cli PQC)
        """
        def safe_float(col: str) -> float:
            val = row.get(col, 0.0)
            if pd.isna(val):
                return 0.0
            return float(val)
        
        wire_ec = safe_float("wire_srv_ecdh") + safe_float("wire_cli_ecdh")
        wire_pq = safe_float("wire_srv_pqc") + safe_float("wire_cli_pqc")
        
        state = np.array([
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
        
        return state
    
    def _compute_reward(
        self,
        row: pd.Series,
        policy_name: str,
        level_int: int,
    ) -> float:
        """
        Compute RTT-dependent reward with mode bonuses.
        
        Reward = base
                 - α(rtt) * latency
                 - β * wire_kb
                 + γ(rtt) * security_level
                 + mode_bonus              # HYBRID/PQC get bonus, FALLBACK/CLASSICAL get penalty
                 - level_violation_penalty
        """
        cfg = self.reward_config
        
        # Extract metrics
        rtt_ms = float(row.get("rtt_ms", 0.0))
        latency_ms = float(row.get("total_time_ms", 0.0))
        
        wire_ec = float(row.get("wire_srv_ecdh", 0) or 0) + float(row.get("wire_cli_ecdh", 0) or 0)
        wire_pq = float(row.get("wire_srv_pqc", 0) or 0) + float(row.get("wire_cli_pqc", 0) or 0)
        total_wire_kb = (wire_ec + wire_pq) / 1000.0
        
        # RTT-dependent coefficients
        alpha = cfg.get_alpha(rtt_ms)
        security_weight = cfg.get_security_weight(rtt_ms)
        
        # Security bonus (normalized to [0, 1])
        security_score = level_int / 5.0  # L1=0.2, L3=0.6, L5=1.0
        
        # Mode bonus/penalty
        mode_bonus = cfg.get_mode_bonus(policy_name)
        
        # Level violation penalty
        level_violation = cfg.level_violation_penalty if level_int < cfg.min_acceptable_level else 0.0
        
        # Compute reward
        reward = (
            1.0                                # Base success reward
            - alpha * latency_ms               # Latency penalty (RTT-dependent)
            - cfg.beta * total_wire_kb         # Wire overhead penalty
            + security_weight * security_score # Security bonus (RTT-dependent)
            + mode_bonus                       # Policy mode bonus/penalty
            - level_violation                  # Penalty for low security level
        )
        
        return float(reward)
    
    def _compute_normalization_params(self, states: np.ndarray) -> None:
        """Compute z-score normalization parameters."""
        self.state_mean = np.mean(states, axis=0)
        self.state_std = np.std(states, axis=0)
        # Avoid division by zero
        self.state_std = np.where(self.state_std < 1e-8, 1.0, self.state_std)
    
    def _normalize_states(self, states: np.ndarray) -> np.ndarray:
        """Apply z-score normalization to states."""
        if self.state_mean is None or self.state_std is None:
            raise ValueError("Normalization parameters not computed. Call build() first.")
        return (states - self.state_mean) / self.state_std
    
    def build(
        self,
        csv_path: Optional[str] = None,
        train_split: float = 0.8,
        seed: int = 42,
    ) -> Dict[str, np.ndarray]:
        """
        Build the offline RL dataset from CSV.
        
        Args:
            csv_path: Path to handshake_raw.csv
            train_split: Fraction of data for training
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary with train/test splits and metadata
        """
        if csv_path is None:
            csv_path = self.path_config.raw_data_csv
        
        print(f"[Dataset] Loading data from: {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"[Dataset] Loaded {len(df)} rows")
        
        # Sort by trial for temporal consistency
        df = df.sort_values(["policy", "level_int", "rtt_ms", "trial"]).reset_index(drop=True)
        
        # Build arrays
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        
        # Additional metadata for analysis
        rtts = []
        latencies = []
        wire_bytes = []
        
        skipped = 0
        for idx, row in df.iterrows():
            policy_name = str(row["policy"])
            level_int = int(row["level_int"])
            
            # Build action key
            action_key = (policy_name, level_int)
            if action_key not in ACTION_TO_IDX:
                skipped += 1
                continue
            
            # Extract state
            state = self._extract_state(row)
            
            # Get action index
            action = ACTION_TO_IDX[action_key]
            
            # Compute reward
            reward = self._compute_reward(row, policy_name, level_int)
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            
            # Metadata
            rtts.append(float(row.get("rtt_ms", 0)))
            latencies.append(float(row.get("total_time_ms", 0)))
            wire_ec = float(row.get("wire_srv_ecdh", 0) or 0) + float(row.get("wire_cli_ecdh", 0) or 0)
            wire_pq = float(row.get("wire_srv_pqc", 0) or 0) + float(row.get("wire_cli_pqc", 0) or 0)
            wire_bytes.append(wire_ec + wire_pq)
        
        if skipped > 0:
            print(f"[Dataset] Skipped {skipped} rows with unknown actions")
        
        # Convert to arrays
        S = np.array(states, dtype=np.float32)
        A = np.array(actions, dtype=np.int64)
        R = np.array(rewards, dtype=np.float32)
        
        # Compute normalization from ALL data before splitting
        self._compute_normalization_params(S)
        
        # Normalize states
        S_norm = self._normalize_states(S)
        
        # Next states: shift by 1, last state points to itself
        S_next_norm = np.vstack([S_norm[1:], S_norm[-1:]])
        
        # Done flags: all False for continuing task (last one is True)
        D = np.zeros(len(S), dtype=np.float32)
        D[-1] = 1.0
        
        # Metadata arrays
        RTT = np.array(rtts, dtype=np.float32)
        LAT = np.array(latencies, dtype=np.float32)
        WIRE = np.array(wire_bytes, dtype=np.float32)
        
        print(f"[Dataset] Built {len(S)} transitions")
        print(f"[Dataset] State shape: {S_norm.shape}")
        print(f"[Dataset] Action distribution:")
        for i, (policy, level) in enumerate(ACTION_LIST):
            count = np.sum(A == i)
            pct = 100.0 * count / len(A)
            print(f"    {i:2d}: {policy:20s} L{level} -> {count:5d} ({pct:5.1f}%)")
        
        # Train/test split
        indices = np.arange(len(S))
        train_idx, test_idx = train_test_split(
            indices, train_size=train_split, random_state=seed, shuffle=True
        )
        
        print(f"[Dataset] Train: {len(train_idx)}, Test: {len(test_idx)}")
        
        # Reward statistics
        print(f"[Dataset] Reward stats - Mean: {R.mean():.3f}, Std: {R.std():.3f}, "
              f"Min: {R.min():.3f}, Max: {R.max():.3f}")
        
        return {
            # Full dataset (normalized)
            "S": S_norm,
            "A": A,
            "R": R,
            "S_next": S_next_norm,
            "done": D,
            
            # Raw states (unnormalized) for analysis
            "S_raw": S,
            
            # Metadata
            "rtt": RTT,
            "latency": LAT,
            "wire": WIRE,
            
            # Normalization parameters
            "state_mean": self.state_mean,
            "state_std": self.state_std,
            
            # Train/test indices
            "train_idx": train_idx,
            "test_idx": test_idx,
            
            # Action vocabulary
            "action_vocab": np.array(ACTION_LIST, dtype=object),
        }
    
    def save(
        self,
        data: Dict[str, np.ndarray],
        output_path: Optional[str] = None,
    ) -> str:
        """Save dataset to NPZ file."""
        if output_path is None:
            output_path = self.path_config.dataset_path
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.savez_compressed(output_path, **data)
        print(f"[Dataset] Saved to: {output_path}")
        return output_path
    
    def load(self, path: Optional[str] = None) -> Dict[str, np.ndarray]:
        """Load dataset from NPZ file."""
        if path is None:
            path = self.path_config.dataset_path
        
        data = dict(np.load(path, allow_pickle=True))
        
        # Restore normalization parameters
        self.state_mean = data["state_mean"]
        self.state_std = data["state_std"]
        
        print(f"[Dataset] Loaded from: {path}")
        print(f"[Dataset] Total transitions: {len(data['S'])}")
        
        return data


def compute_oracle_actions(data: Dict[str, np.ndarray]) -> Dict[float, int]:
    """
    Compute oracle (best) action for each RTT level based on average reward.
    
    Returns:
        Dict mapping RTT -> best action index
    """
    S = data["S"]
    A = data["A"]
    R = data["R"]
    RTT = data["rtt"]
    
    unique_rtts = np.unique(RTT)
    oracle = {}
    
    print("\n[Oracle] Computing best actions per RTT:")
    for rtt in unique_rtts:
        mask = RTT == rtt
        
        # Find best action by average reward
        best_action = None
        best_reward = -np.inf
        
        for action_idx in range(len(ACTION_LIST)):
            action_mask = (A == action_idx) & mask
            if np.sum(action_mask) > 0:
                mean_reward = R[action_mask].mean()
                if mean_reward > best_reward:
                    best_reward = mean_reward
                    best_action = action_idx
        
        oracle[rtt] = best_action
        policy, level = ACTION_LIST[best_action]
        print(f"  RTT={rtt:3.0f}ms -> Action {best_action}: {policy} L{level} (reward={best_reward:.3f})")
    
    return oracle


def build_and_save_dataset(
    csv_path: str = None,
    output_path: str = None,
    train_split: float = 0.8,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Convenience function to build and save dataset.
    
    Args:
        csv_path: Input CSV path (default: results/eval_grid/handshake_raw.csv)
        output_path: Output NPZ path (default: results/rl/offline_rl_dataset_v2.npz)
        train_split: Train/test split ratio
        seed: Random seed
        
    Returns:
        Dataset dictionary
    """
    builder = OfflineRLDatasetBuilder()
    data = builder.build(csv_path=csv_path, train_split=train_split, seed=seed)
    
    # Compute oracle
    oracle = compute_oracle_actions(data)
    data["oracle"] = np.array(list(oracle.items()), dtype=object)
    
    builder.save(data, output_path)
    return data


if __name__ == "__main__":
    # Build dataset from default paths
    build_and_save_dataset()
