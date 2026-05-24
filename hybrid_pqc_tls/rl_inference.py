# hybrid_pqc_tls/rl_inference.py
# -*- coding: utf-8 -*-
"""
RL Inference Module for Hybrid PQC-TLS
======================================

This module loads the trained PPO policy and exposes a clean
inference interface so that the PolicyEngine can request:

    (CryptoPolicy, NISTSecurityLevel) = rl_policy.select(state)

It also provides a deployment-time state builder that produces a
15-dimensional vector consistent with the *current* RL training
format used in HybridPQCEnv:

    state = [
        rtt_ms,
        prev_latency_ms,
        using_mock_pqc,      # 0/1
        strict_pqc,          # 0/1
        supports_classical,  # 0/1
        supports_pqc,        # 0/1
        0, 0, 0, 0, 0, 0, 0, 0, 0
    ]

This file is inference-only. No training code lives here.
"""

from __future__ import annotations

import os
import numpy as np
from typing import Optional
from stable_baselines3 import PPO

from .config import CryptoPolicy, NISTSecurityLevel


# ---------------------------------------------------------------------
# Paths — resolved relative to project root (…/hybrid-pqc-tls/)
# ---------------------------------------------------------------------

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

DEFAULT_MODEL_PATH = os.path.join(ROOT_DIR, "results", "rl", "safe_ppo_policy.zip")
DEFAULT_DATASET_PATH = os.path.join(ROOT_DIR, "results", "rl", "offline_rl_dataset.npz")

# Must match HybridPQCEnv.STATE_DIM and rl_dataset state layout
STATE_DIM = 15


# ---------------------------------------------------------------------
# HybridPPOPolicy — Inference Wrapper
# ---------------------------------------------------------------------

class HybridPPOPolicy:
    """
    Inference-only PPO agent for Hybrid PQC-TLS.

    Loads:
        - safe_ppo_policy.zip    (SB3 PPO policy)
        - offline_rl_dataset.npz (action_vocab)

    Provides:
        policy, level = self.select(state)
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        dataset_path: Optional[str] = None,
        device: str = "cpu",
    ) -> None:

        # Resolve default paths if not provided
        if model_path is None:
            model_path = DEFAULT_MODEL_PATH
        if dataset_path is None:
            dataset_path = DEFAULT_DATASET_PATH

        model_path = os.path.abspath(model_path)
        dataset_path = os.path.abspath(dataset_path)

        # Sanity checks
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"[RL] PPO model not found: {model_path}")
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"[RL] offline_rl_dataset.npz missing: {dataset_path}")

        # Load PPO agent
        self.model = PPO.load(model_path, device=device)

        # Load action vocabulary: a mapping of
        #    index → (policy_name, level_int)
        data = np.load(dataset_path, allow_pickle=True)
        self.action_vocab = data["action_vocab"]
        self.device = device

        # Validate shape
        if self.action_vocab.ndim != 2 or self.action_vocab.shape[1] != 2:
            raise ValueError(f"[RL] Invalid action_vocab shape: {self.action_vocab.shape}")

    # ------------------------------------------------------------------
    # select(state)
    # ------------------------------------------------------------------
    def select(self, state: np.ndarray) -> tuple[CryptoPolicy, NISTSecurityLevel]:
        """
        Given a 1 × STATE_DIM state vector, return a valid Hybrid PQC-TLS
        policy tuple:

            (CryptoPolicy.X, NISTSecurityLevel.Y)

        This is called by PolicyEngine in PERFORMANCE_ADAPTIVE mode.

        Expected state layout (same as HybridPQCEnv._build_state):

            [ rtt_ms,
              prev_latency_ms,
              using_mock_pqc,
              strict_pqc,
              supports_classical,
              supports_pqc,
              0,0,0,0,0,0,0,0,0 ]
        """
        state = np.asarray(state, dtype=np.float32).reshape(1, -1)

        if state.shape[1] != STATE_DIM:
            raise ValueError(
                f"[RL] Expected state dimension {STATE_DIM}, got {state.shape[1]}"
            )

        # PPO action
        action_idx, _ = self.model.predict(state, deterministic=True)
        action_idx = int(action_idx)

        if action_idx < 0 or action_idx >= len(self.action_vocab):
            raise IndexError(
                f"[RL] Action index {action_idx} out of bounds for vocab size {len(self.action_vocab)}"
            )

        raw_policy_name, raw_level_int = self.action_vocab[action_idx]

        # Normalize policy string
        policy_str = (
            raw_policy_name.decode("utf-8")
            if isinstance(raw_policy_name, (bytes, bytearray))
            else str(raw_policy_name)
        )

        # Convert to CryptoPolicy Enum
        if policy_str in CryptoPolicy.__members__:
            policy = CryptoPolicy[policy_str]
        else:
            # Fallback: allow CryptoPolicy("PERFORMANCE_ADAPTIVE") style
            policy = CryptoPolicy(policy_str)

        # Convert level to NISTSecurityLevel Enum
        raw_level_int = int(raw_level_int)
        level = None
        for lvl in NISTSecurityLevel:
            if lvl.value == raw_level_int:
                level = lvl
                break
        if level is None:
            raise ValueError(f"[RL] Invalid NISTSecurityLevel int: {raw_level_int}")

        return policy, level


# ---------------------------------------------------------------------
# State Builder for Deployment (PRE-handshake)
# ---------------------------------------------------------------------

def build_deployment_state(
    *,
    rtt_ms: float,
    supports_pqc: bool,
    supports_classical: bool,
    strict_pqc: bool = True,
    using_mock_pqc: bool = False,
    prev_latency_ms: float = 0.0,
) -> np.ndarray:
    """
    Build a 15-dimensional deployment state vector that matches the
    exact state layout used during PPO training in HybridPQCEnv.

    Layout:

        0: rtt_ms
        1: prev_latency_ms
        2: using_mock_pqc       (0/1)
        3: strict_pqc           (0/1)
        4: supports_classical   (0/1)
        5: supports_pqc         (0/1)
        6-14: zero (reserved)

    This is a *pre-handshake* state: it uses only observable
    network/capability context and a simple history feature
    (previous handshake latency).
    """

    return np.array(
        [
            float(rtt_ms),                        # 0
            float(prev_latency_ms),               # 1
            1.0 if using_mock_pqc else 0.0,       # 2
            1.0 if strict_pqc else 0.0,           # 3
            1.0 if supports_classical else 0.0,   # 4
            1.0 if supports_pqc else 0.0,         # 5
            0.0,  # 6
            0.0,  # 7
            0.0,  # 8
            0.0,  # 9
            0.0,  # 10
            0.0,  # 11
            0.0,  # 12
            0.0,  # 13
            0.0,  # 14
        ],
        dtype=np.float32,
    )
# End of rl_inference.py