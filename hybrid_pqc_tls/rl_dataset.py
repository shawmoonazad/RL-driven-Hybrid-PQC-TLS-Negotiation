# hybrid_pqc_tls/rl_dataset.py
# -*- coding: utf-8 -*-
"""
RL Dataset Builder for Hybrid PQC-TLS
=====================================
Takes handshake_raw.csv (from run_evaluation.py) and builds an
offline RL dataset suitable for CQL and PPO pretraining.

Output:
    results/rl/offline_rl_dataset.npz with:
      S        : (N, state_dim) float32
      A        : (N,) int64 action indices
      R        : (N,) float32 rewards
      S_next   : (N, state_dim) float32 next states
      action_vocab : (num_actions, 2) object array: (policy_name, level_int)
"""

import os
import numpy as np
import pandas as pd

from .config import CryptoPolicy


# deterministic action space: (policy, level_int)
POLICY_ORDER = [
    CryptoPolicy.REQUIRE_HYBRID,
    CryptoPolicy.ALLOW_FALLBACK,
    CryptoPolicy.PQC_ONLY,
    CryptoPolicy.CLASSICAL_ONLY,
]
LEVEL_ORDER = [1, 3, 5]


def build_action_vocab():
    """
    Returns:
        action_list: list[(policy_name, level_int)]
        action_index: dict[(policy_name, level_int) -> idx]
    """
    action_list = []
    for pol in POLICY_ORDER:
        for lvl in LEVEL_ORDER:
            action_list.append((pol.name, int(lvl)))

    action_index = {key: i for i, key in enumerate(action_list)}
    return action_list, action_index


def build_state(row: pd.Series) -> np.ndarray:
    """
    State vector (15 dims) mirroring your Colab scaffold:
    [0]  rtt_ms
    [1]  trial index
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
    [13] total_wire_ec (srv+cli ECDH)
    [14] total_wire_pq (srv+cli PQC)
    """
    k = lambda c: float(row.get(c, 0.0)) if pd.notna(row.get(c, np.nan)) else 0.0

    wire_ec = k("wire_srv_ecdh") + k("wire_cli_ecdh")
    wire_pq = k("wire_srv_pqc") + k("wire_cli_pqc")

    return np.array(
        [
            k("rtt_ms"),
            float(row.get("trial", 0.0)),
            float(row.get("using_mock_pqc", 0.0)),
            float(row.get("strict_pqc", 0.0)),
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


def build_reward(row: pd.Series,
                 alpha: float = 0.02,
                 beta: float = 0.5,
                 gamma: float = 0.3,
                 lambd: float = 5.0) -> float:
    """
    Reward combines:
        - latency penalty (alpha)
        - wire penalty (beta, per KB)
        - security bonus for higher level (gamma)
        - violation penalty if level_int < 3 (λ)
    """
    lat = float(row["total_time_ms"])
    wire = float(
        (row.get("wire_srv_ecdh", 0.0) or 0.0)
        + (row.get("wire_cli_ecdh", 0.0) or 0.0)
        + (row.get("wire_srv_pqc", 0.0) or 0.0)
        + (row.get("wire_cli_pqc", 0.0) or 0.0)
    )
    level_int = int(row["level_int"])
    viol = 1.0 if level_int < 3 else 0.0
    succ = 1.0  # all rows are successful handshakes in this dataset

    reward = -alpha * lat - beta * (wire / 1000.0) + gamma * (level_int / 5.0) + succ - lambd * viol
    return float(reward)


def build_offline_dataset(
    csv_path: str = "results/eval_grid/handshake_raw.csv",
    out_path: str = "results/rl/offline_rl_dataset.npz",
) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    df = pd.read_csv(csv_path)
    # sort by trial so that "next" is meaningful
    df = df.sort_values(["trial"]).reset_index(drop=True)

    action_list, action_index = build_action_vocab()

    # --- Build arrays ---
    states = []
    actions = []
    rewards = []

    for _, row in df.iterrows():
        pol_name = str(row["policy"])
        lvl_int = int(row["level_int"])
        key = (pol_name, lvl_int)
        if key not in action_index:
            # skip weird combos if any
            continue

        s = build_state(row)
        r = build_reward(row)
        a = action_index[key]

        states.append(s)
        actions.append(a)
        rewards.append(r)

    S = np.stack(states, axis=0).astype(np.float32)
    A = np.asarray(actions, dtype=np.int64)
    R = np.asarray(rewards, dtype=np.float32)

    # Simple next-state: next row, or self for last
    S_next = np.vstack([S[i + 1] if i + 1 < len(S) else S[i] for i in range(len(S))]).astype(np.float32)

    # action_vocab will be list[(policy_name, level_int)]
    action_vocab = np.array(action_list, dtype=object)

    np.savez_compressed(
        out_path,
        S=S,
        A=A,
        R=R,
        S_next=S_next,
        action_vocab=action_vocab,
    )

    print(f"[rl_dataset] Dataset saved: {out_path}")
    print(f"  S shape: {S.shape}, A shape: {A.shape}, R shape: {R.shape}")
    print(f"  num_actions: {len(action_vocab)}")


if __name__ == "__main__":
    build_offline_dataset()
