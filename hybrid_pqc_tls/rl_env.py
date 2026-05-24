# hybrid_pqc_tls/rl_env.py
# -*- coding: utf-8 -*-
"""
RL Environment for Hybrid PQC-TLS
=================================
Gymnasium env that wraps the real Hybrid PQC-TLS microbenchmark.

Action:
    discrete index -> (policy_name, level_int) from action_vocab

State:
    15-D vector aligned with offline dataset:
        [rtt, step_idx, using_mock_pqc, strict_pqc,
         kex_keygen, kex_encaps, kex_decaps,
         kex_hkdf_c, kex_hkdf_s,
         auth_sign_c, auth_sign_q,
         auth_ver_c, auth_ver_q,
         total_wire_ec, total_wire_pq]

Reward:
    same shaping as in rl_dataset.py.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import List, Tuple

from .config import (
    NISTSecurityLevel,
    CryptoPolicy,
    Role,
    AlgorithmCapabilities,
    SECURITY_CONFIGS,
    AUTH_CONFIG,
)
from .policy import PolicyEngine
from .session import FullHybridSession


def _get_exact_caps(level: NISTSecurityLevel) -> AlgorithmCapabilities:
    sec_cfg = SECURITY_CONFIGS[level]
    auth_cfg = AUTH_CONFIG[level]
    return AlgorithmCapabilities(
        supported_levels={level},
        supported_classical={sec_cfg.classical_name, auth_cfg.curve},
        supported_pqc={sec_cfg.pqc_algorithm, auth_cfg.dsa},
        supports_hybrid=True,
        max_latency_ms=None,
        max_wire_bytes=None,
    )


def _policy_from_name(name: str) -> CryptoPolicy:
    return CryptoPolicy[name]


def _level_from_int(level_int: int) -> NISTSecurityLevel:
    return NISTSecurityLevel(level_int)


class HybridPQCEnv(gym.Env):
    """
    Environment where each step corresponds to one handshake decision.

    Observations:
        np.float32 vector shape (15,)
    Actions:
        Discrete(num_actions) mapping to (policy_name, level_int).

    No terminal condition: continuing task.
    """

    metadata = {"render_modes": []}

    def __init__(self, action_vocab: np.ndarray):
        super().__init__()

        # action_vocab: array[(policy_name, level_int), ...]
        self.action_vocab: List[Tuple[str, int]] = [
            (str(p), int(l)) for p, l in action_vocab
        ]

        self.action_space = spaces.Discrete(len(self.action_vocab))
        self.observation_space = spaces.Box(
            low=-1e9, high=1e9, shape=(15,), dtype=np.float32
        )

        self.engine = PolicyEngine()
        self._step_count = 0
        self.state = np.zeros(15, dtype=np.float32)

    # ---------------- Core Helpers ---------------- #

    def _sample_rtt(self) -> float:
        """
        Sample RTT in ms. You can tune this distribution to match your
        evaluation grid or real-world traces.
        """
        # simple mixture: mostly 0-100ms, sometimes 100-250ms
        if np.random.rand() < 0.8:
            return float(np.random.uniform(0, 100))
        else:
            return float(np.random.uniform(100, 250))

    @staticmethod
    def _reward(latency_ms: float, total_wire: float, level_int: int) -> tuple[float, float]:
        """
        Reward + violation metric (for logging).
        Mirrors rl_dataset.build_reward().
        """
        alpha, beta, gamma, lambd = 0.02, 0.5, 0.3, 5.0

        viol = 1.0 if level_int < 3 else 0.0
        succ = 1.0  # here: handshake succeeded; if you add failures, set 0 for fail

        reward = (
            -alpha * latency_ms
            - beta * (total_wire / 1000.0)
            + gamma * (level_int / 5.0)
            + succ
            - lambd * viol
        )
        return float(reward), viol

    @staticmethod
    def _build_state(
        rtt_ms: float,
        step_idx: int,
        using_mock_pqc: bool,
        strict_pqc: bool,
        metrics: dict,
    ) -> np.ndarray:
        k = lambda name: float(metrics.get(name, 0.0) or 0.0)

        wire_ec = k("wire_srv_ecdh") + k("wire_cli_ecdh")
        wire_pq = k("wire_srv_pqc") + k("wire_cli_pqc")

        return np.array(
            [
                float(rtt_ms),
                float(step_idx),
                float(using_mock_pqc),
                float(strict_pqc),
                k("kex_keygen_ms"),
                k("kex_encaps_ms"),
                k("kex_decaps_ms"),
                k("kex_hkdf_c_ms"),
                k("kex_hkdf_s_ms"),
                k("auth_sign_c_ms"),
                k("auth_sign_q_ms"),
                k("auth_ver_c_ms"),
                k("auth_ver_q_ms"),
                wire_ec,
                wire_pq,
            ],
            dtype=np.float32,
        )

    # ---------------- Gym API ---------------- #

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
        self.state = np.zeros(15, dtype=np.float32)
        return self.state, {}

    def step(self, action: int):
        self._step_count += 1
        assert self.action_space.contains(action), "Invalid action index"

        policy_name, level_int = self.action_vocab[action]
        policy = _policy_from_name(policy_name)
        level = _level_from_int(level_int)

        # capabilities pinned to this level
        caps = _get_exact_caps(level)
        rtt_ms = self._sample_rtt()

        # run a single handshake
        srv = FullHybridSession(Role.SERVER, policy, caps, self.engine)
        cli = FullHybridSession(Role.CLIENT, policy, caps, self.engine)

        metrics = FullHybridSession.simulate_handshake(cli, srv, rtt_ms=rtt_ms)

        latency_ms = float(metrics.get("total_time_ms", 0.0) or 0.0)
        wire_ec = (metrics.get("wire_srv_ecdh") or 0) + (metrics.get("wire_cli_ecdh") or 0)
        wire_pq = (metrics.get("wire_srv_pqc") or 0) + (metrics.get("wire_cli_pqc") or 0)
        total_wire = float(wire_ec + wire_pq)

        using_mock_pqc = bool(metrics.get("using_mock_pqc", False))
        strict_pqc = bool(metrics.get("strict_pqc", False))

        reward, viol = self._reward(latency_ms, total_wire, level_int)

        self.state = self._build_state(
            rtt_ms=rtt_ms,
            step_idx=self._step_count,
            using_mock_pqc=using_mock_pqc,
            strict_pqc=strict_pqc,
            metrics=metrics,
        )

        info = {
            "latency": latency_ms,
            "wire": total_wire,
            "violation": float(viol),
            "policy": policy_name,
            "level_int": level_int,
        }

        terminated = False  # continuing task
        truncated = False

        return self.state, float(reward), terminated, truncated, info


# Alias for convenience (if you ever used this name)
HybridEnv = HybridPQCEnv
