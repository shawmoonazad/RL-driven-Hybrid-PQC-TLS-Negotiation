# -*- coding: utf-8 -*-
"""
RL Configuration for Hybrid PQC-TLS
===================================
Central configuration for offline RL experiments including:
- Action space definition
- State space normalization parameters
- Reward function coefficients
- Training hyperparameters
- Algorithm-specific settings
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
from enum import Enum


# ============================================================================
# Action Space Definition
# ============================================================================

class PolicyType(Enum):
    """Crypto policies available for selection."""
    REQUIRE_HYBRID = "REQUIRE_HYBRID"
    ALLOW_FALLBACK = "ALLOW_FALLBACK"
    PQC_ONLY = "PQC_ONLY"
    CLASSICAL_ONLY = "CLASSICAL_ONLY"


# Ordered list of policies (precedence order for Rule-Based)
# REQUIRE_HYBRID > PQC_ONLY > ALLOW_FALLBACK > CLASSICAL_ONLY
POLICY_ORDER = [
    PolicyType.REQUIRE_HYBRID,
    PolicyType.ALLOW_FALLBACK,
    PolicyType.PQC_ONLY,
    PolicyType.CLASSICAL_ONLY,
]

# Security levels (NIST)
LEVEL_ORDER = [1, 3, 5]

# Build action vocabulary: List of (policy_name, level_int)
def build_action_vocab() -> Tuple[List[Tuple[str, int]], Dict[Tuple[str, int], int]]:
    """
    Creates the action vocabulary mapping.
    
    Returns:
        action_list: List of (policy_name, level_int) tuples
        action_to_idx: Dict mapping (policy_name, level_int) -> action_index
    """
    action_list = []
    for policy in POLICY_ORDER:
        for level in LEVEL_ORDER:
            action_list.append((policy.value, level))
    
    action_to_idx = {action: idx for idx, action in enumerate(action_list)}
    return action_list, action_to_idx


ACTION_LIST, ACTION_TO_IDX = build_action_vocab()
NUM_ACTIONS = len(ACTION_LIST)  # 12 actions total

# Action indices for specific policies (useful for analysis)
HYBRID_ACTIONS = [i for i, (p, l) in enumerate(ACTION_LIST) if p == "REQUIRE_HYBRID"]
PQC_ACTIONS = [i for i, (p, l) in enumerate(ACTION_LIST) if p == "PQC_ONLY"]
CLASSICAL_ACTIONS = [i for i, (p, l) in enumerate(ACTION_LIST) if p == "CLASSICAL_ONLY"]
FALLBACK_ACTIONS = [i for i, (p, l) in enumerate(ACTION_LIST) if p == "ALLOW_FALLBACK"]


# ============================================================================
# State Space Definition
# ============================================================================

STATE_DIM = 15

STATE_FEATURES = [
    "rtt_ms",           # 0: Round-trip time
    "trial",            # 1: Trial index (can be used as time feature)
    "using_mock_pqc",   # 2: Whether mock PQC is active
    "strict_pqc",       # 3: Whether strict PQC mode is enforced
    "kex_keygen_ms",    # 4: Key generation time
    "kex_encaps_ms",    # 5: Encapsulation time
    "kex_decaps_ms",    # 6: Decapsulation time
    "kex_hkdf_c_ms",    # 7: Client HKDF time
    "kex_hkdf_s_ms",    # 8: Server HKDF time
    "auth_sign_c_ms",   # 9: Classical signing time
    "auth_sign_q_ms",   # 10: PQC signing time
    "auth_ver_c_ms",    # 11: Classical verification time
    "auth_ver_q_ms",    # 12: PQC verification time
    "total_wire_ec",    # 13: Total classical wire overhead
    "total_wire_pq",    # 14: Total PQC wire overhead
]


# ============================================================================
# Reward Function Configuration (RTT-Dependent)
# ============================================================================

@dataclass
class RewardConfig:
    """
    RTT-dependent reward configuration.
    
    RESEARCH FOCUS: Prioritize REQUIRE_HYBRID and PQC_ONLY modes
    over ALLOW_FALLBACK, while heavily penalizing CLASSICAL_ONLY.
    
    Security Value Hierarchy:
        1. REQUIRE_HYBRID (best) - dual classical + PQC protection
        2. PQC_ONLY (good) - pure post-quantum security
        3. ALLOW_FALLBACK (uncertain) - may degrade to single path
        4. CLASSICAL_ONLY (bad) - no quantum resistance
    
    Reward = base_reward
             - α(rtt) * latency           # Latency penalty (RTT-dependent)
             - β * wire_kb                 # Wire overhead penalty
             + γ * security_level_bonus    # Higher level = better
             + mode_bonus                  # Policy mode bonus/penalty
             - violation_penalty           # Level < 3 penalty
    """
    # Base coefficients
    alpha_base: float = 0.01        # Base latency penalty coefficient
    alpha_rtt_scale: float = 0.0001 # How much alpha increases per ms of RTT (reduced)
    beta: float = 0.2               # Wire overhead penalty (per KB) - reduced
    gamma: float = 0.5              # Security level bonus scaling
    
    # === MODE BONUSES (KEY FOR RESEARCH) ===
    # Positive = reward, Negative = penalty
    hybrid_bonus: float = 3.0       # Bonus for REQUIRE_HYBRID (dual security)
    pqc_bonus: float = 1.5          # Bonus for PQC_ONLY (post-quantum)
    fallback_penalty: float = 1.0   # Penalty for ALLOW_FALLBACK (uncertain)
    classical_penalty: float = 15.0 # Heavy penalty for CLASSICAL_ONLY
    
    # Security floor enforcement
    min_acceptable_level: int = 3   # Level 3 is the minimum acceptable
    level_violation_penalty: float = 5.0  # Penalty for going below min level
    
    # RTT thresholds for adaptive behavior
    low_rtt_threshold: float = 50.0   # Below this, prioritize security
    high_rtt_threshold: float = 150.0 # Above this, more latency-sensitive
    
    def get_alpha(self, rtt_ms: float) -> float:
        """
        Get RTT-dependent latency penalty coefficient.
        Higher RTT -> Higher alpha -> More latency penalty
        """
        return self.alpha_base + self.alpha_rtt_scale * rtt_ms
    
    def get_security_weight(self, rtt_ms: float) -> float:
        """
        Get RTT-dependent security weight.
        Lower RTT -> Higher weight -> Prioritize security more
        """
        if rtt_ms <= self.low_rtt_threshold:
            return 1.5 * self.gamma  # 50% bonus for security at low RTT
        elif rtt_ms >= self.high_rtt_threshold:
            return 0.8 * self.gamma  # 20% reduction at high RTT
        else:
            # Linear interpolation
            t = (rtt_ms - self.low_rtt_threshold) / (self.high_rtt_threshold - self.low_rtt_threshold)
            return self.gamma * (1.5 - 0.7 * t)
    
    def get_mode_bonus(self, policy_name: str) -> float:
        """
        Get bonus/penalty for a specific policy mode.
        """
        if policy_name == "REQUIRE_HYBRID":
            return self.hybrid_bonus
        elif policy_name == "PQC_ONLY":
            return self.pqc_bonus
        elif policy_name == "ALLOW_FALLBACK":
            return -self.fallback_penalty
        elif policy_name == "CLASSICAL_ONLY":
            return -self.classical_penalty
        return 0.0


# Default reward configuration
DEFAULT_REWARD_CONFIG = RewardConfig()


# ============================================================================
# Training Hyperparameters
# ============================================================================

@dataclass
class TrainingConfig:
    """Hyperparameters for offline RL training."""
    # General
    seed: int = 42
    device: str = "cpu"  # "cuda" if available
    
    # Data
    train_split: float = 0.8
    batch_size: int = 256
    
    # Training
    num_epochs: int = 100
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    
    # Network architecture
    hidden_dims: List[int] = field(default_factory=lambda: [256, 256])
    dropout: float = 0.1
    
    # Algorithm-specific
    # CQL
    cql_alpha: float = 1.0          # CQL regularization strength
    cql_temperature: float = 1.0
    
    # IQL
    iql_tau: float = 0.7            # Expectile for IQL
    iql_beta: float = 3.0           # Inverse temperature for advantage weighting
    
    # BCQ
    bcq_threshold: float = 0.3      # Action filtering threshold
    
    # AWAC
    awac_lambda: float = 1.0        # Advantage weighting temperature
    
    # Target networks
    target_update_freq: int = 100
    tau: float = 0.005              # Soft update coefficient
    
    # Discount factor
    gamma: float = 0.99
    
    # Evaluation
    eval_freq: int = 10             # Evaluate every N epochs
    
    # Logging
    log_dir: str = "results/rl"
    save_models: bool = True


DEFAULT_TRAINING_CONFIG = TrainingConfig()


# ============================================================================
# Baseline Configurations
# ============================================================================

@dataclass
class BaselineConfig:
    """Configuration for baseline comparisons."""
    # Fixed policy baselines (for reference only, not used in main comparison)
    fixed_policies: List[Tuple[str, int]] = field(default_factory=lambda: [
        ("REQUIRE_HYBRID", 3),   # Always Hybrid L3
        ("REQUIRE_HYBRID", 5),   # Always Hybrid L5
        ("PQC_ONLY", 3),         # Always PQC L3
        ("PQC_ONLY", 5),         # Always PQC L5
        ("ALLOW_FALLBACK", 3),   # Fallback L3
    ])
    
    # Oracle computation
    compute_oracle: bool = True
    oracle_per_rtt: bool = True  # Compute oracle separately for each RTT


DEFAULT_BASELINE_CONFIG = BaselineConfig()


# ============================================================================
# Evaluation Metrics
# ============================================================================

EVALUATION_METRICS = [
    "mean_reward",
    "mean_latency_ms",
    "median_latency_ms",
    "p95_latency_ms",
    "mean_wire_bytes",
    "security_violation_rate",  # % of actions with level < 3
    "classical_usage_rate",     # % of CLASSICAL_ONLY actions
    "hybrid_usage_rate",        # % of REQUIRE_HYBRID actions
    "pqc_usage_rate",          # % of PQC_ONLY actions
    "level_distribution",       # Distribution over L1/L3/L5
]


# ============================================================================
# File Paths
# ============================================================================

@dataclass
class PathConfig:
    """File paths for data and models."""
    # Input
    raw_data_csv: str = "results/eval_grid/handshake_raw.csv"
    
    # Processed dataset
    dataset_dir: str = "results/rl"
    dataset_file: str = "offline_rl_dataset_v2.npz"
    
    # Models
    model_dir: str = "results/rl/models"
    
    # Evaluation results
    eval_dir: str = "results/rl/evaluation"
    figures_dir: str = "results/rl/figures"
    
    # Logging (FIXED: was missing in original)
    log_dir: str = "results/rl"
    
    @property
    def dataset_path(self) -> str:
        return f"{self.dataset_dir}/{self.dataset_file}"


DEFAULT_PATH_CONFIG = PathConfig()


# ============================================================================
# Utility Functions
# ============================================================================

def action_to_string(action_idx: int) -> str:
    """Convert action index to human-readable string."""
    policy, level = ACTION_LIST[action_idx]
    return f"{policy}_L{level}"


def is_classical_action(action_idx: int) -> bool:
    """Check if action uses CLASSICAL_ONLY policy."""
    return action_idx in CLASSICAL_ACTIONS


def is_pqc_safe_action(action_idx: int) -> bool:
    """Check if action is PQC-safe (not classical and level >= 3)."""
    policy, level = ACTION_LIST[action_idx]
    return policy != "CLASSICAL_ONLY" and level >= 3


def get_action_level(action_idx: int) -> int:
    """Get security level for an action."""
    return ACTION_LIST[action_idx][1]


def get_action_policy(action_idx: int) -> str:
    """Get policy name for an action."""
    return ACTION_LIST[action_idx][0]


# Print configuration summary when module is imported
if __name__ == "__main__":
    print("=" * 60)
    print("RL Configuration for Hybrid PQC-TLS")
    print("=" * 60)
    print(f"\nAction Space: {NUM_ACTIONS} actions")
    for i, (policy, level) in enumerate(ACTION_LIST):
        marker = "⚠️ " if policy == "CLASSICAL_ONLY" else "✓ "
        print(f"  {i:2d}: {marker}{policy} @ Level {level}")
    
    print(f"\nState Space: {STATE_DIM} dimensions")
    for i, feat in enumerate(STATE_FEATURES):
        print(f"  {i:2d}: {feat}")
    
    print(f"\nReward Configuration:")
    cfg = DEFAULT_REWARD_CONFIG
    print(f"  Alpha (base): {cfg.alpha_base}")
    print(f"  Alpha (RTT scale): {cfg.alpha_rtt_scale}")
    print(f"  Beta (wire): {cfg.beta}")
    print(f"  Gamma (security): {cfg.gamma}")
    print(f"  Hybrid bonus: +{cfg.hybrid_bonus}")
    print(f"  PQC bonus: +{cfg.pqc_bonus}")
    print(f"  Fallback penalty: -{cfg.fallback_penalty}")
    print(f"  Classical penalty: -{cfg.classical_penalty}")
    
    print(f"\nPath Configuration:")
    print(f"  Dataset: {DEFAULT_PATH_CONFIG.dataset_path}")
    print(f"  Models: {DEFAULT_PATH_CONFIG.model_dir}")
    print(f"  Figures: {DEFAULT_PATH_CONFIG.figures_dir}")
    print(f"  Log dir: {DEFAULT_PATH_CONFIG.log_dir}")
