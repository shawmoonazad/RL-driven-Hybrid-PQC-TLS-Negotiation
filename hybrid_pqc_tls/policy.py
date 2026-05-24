# -*- coding: utf-8 -*-
"""
Cryptographic Policy Engine
===========================
Implements decision logic for algorithm selection based on:
- Local and peer capabilities (AlgorithmCapabilities)
- Administrative policy (CryptoPolicy)
- Real-time network conditions (NetworkMetrics)

Handles downgrade detection and policy violation reporting.
"""

import time
from collections import deque
from typing import Tuple, Optional, List, Dict

from .config import (
    NISTSecurityLevel,
    HybridMode,
    CryptoPolicy,
    AlgorithmCapabilities,
    NetworkMetrics,
    SECURITY_CONFIGS,
)

# RL inference wrapper (PPO-based policy) and deployment state builder
from .rl_inference import HybridPPOPolicy, build_deployment_state


class PolicyEngine:
    """
    Enforces cryptographic policies and negotiates optimal parameters.
    Maintains a history of negotiation outcomes for telemetry.

    If `use_rl=True`, the PERFORMANCE_ADAPTIVE policy is implemented via
    the trained PPO agent (HybridPPOPolicy). Otherwise, it falls back
    to a hand-crafted heuristic.
    """

    def __init__(
        self,
        use_rl: bool = False,
        rl_model_path: Optional[str] = None,
        rl_dataset_path: Optional[str] = None,
        rl_device: str = "cpu",
    ):
        self.performance_history = deque(maxlen=100)
        self.downgrade_events: List[Dict] = []
        self.policy_violations: List[Dict] = []

        # RL-related configuration
        self.use_rl = use_rl
        self.rl_policy: Optional[HybridPPOPolicy] = None

        if self.use_rl:
            # Instantiate PPO-based policy for PERFORMANCE_ADAPTIVE
            self.rl_policy = HybridPPOPolicy(
                model_path=rl_model_path,
                dataset_path=rl_dataset_path,
                device=rl_device,
            )

    def _record_violation(self, policy: CryptoPolicy, reason: str) -> None:
        """Log a policy violation for audit purposes."""
        self.policy_violations.append(
            {
                "policy": policy.value,
                "reason": reason,
                "timestamp": time.time(),
            }
        )

    def _record_downgrade(
        self,
        from_mode: HybridMode,
        to_mode: HybridMode,
        from_level: NISTSecurityLevel,
        to_level: NISTSecurityLevel,
    ) -> None:
        """Log a downgrade event (fallback) for audit purposes."""
        self.downgrade_events.append(
            {
                "from_mode": from_mode.value,
                "to_mode": to_mode.value,
                "from_level": from_level.name,
                "to_level": to_level.name,
                "timestamp": time.time(),
            }
        )

    def _level_available(self, cfg, client_caps, server_caps, need_cls, need_pqc) -> bool:
        """Check if a specific security level's algorithms are supported by both peers."""
        if need_cls:
            if (
                cfg.classical_name not in client_caps.supported_classical
                or cfg.classical_name not in server_caps.supported_classical
            ):
                return False
        if need_pqc:
            if (
                cfg.pqc_algorithm not in client_caps.supported_pqc
                or cfg.pqc_algorithm not in server_caps.supported_pqc
            ):
                return False
        return True

    def _sorted_levels(self, common_levels: set) -> List[NISTSecurityLevel]:
        """Return levels sorted from highest to lowest security."""
        return sorted(common_levels, key=lambda x: x.value, reverse=True)

    def negotiate(
        self,
        client_caps: AlgorithmCapabilities,
        server_caps: AlgorithmCapabilities,
        policy: CryptoPolicy,
        network_metrics: Optional[NetworkMetrics] = None,
        preferred_level: NISTSecurityLevel = NISTSecurityLevel.LEVEL_3,
        allow_classical_fallback: bool = True,
    ) -> Tuple[HybridMode, NISTSecurityLevel]:
        """
        Main negotiation entry point. Determines the best (HybridMode, SecurityLevel)
        tuple that satisfies policy, capabilities, and network constraints.
        """
        common_levels = client_caps.supported_levels & server_caps.supported_levels
        if not common_levels:
            raise ValueError("Negotiation failed: No common security levels.")

        # Default starting point: preferred level if available, else highest available
        selected_lvl = (
            preferred_level
            if preferred_level in common_levels
            else max(common_levels, key=lambda x: x.value)
        )
        cfg_sel = SECURITY_CONFIGS[selected_lvl]
        both_hybrid = client_caps.supports_hybrid and server_caps.supports_hybrid

        # --- Policy: REQUIRE_HYBRID ---
        if policy == CryptoPolicy.REQUIRE_HYBRID:
            if not both_hybrid:
                self._record_violation(policy, "Remote peer does not support hybrid mode.")
                raise ValueError("Policy REQUIRE_HYBRID failed: Peer lacks hybrid support.")

            # Try selected level first
            if self._level_available(cfg_sel, client_caps, server_caps, True, True):
                return HybridMode.AND_HYBRID, selected_lvl

            # Fallback to other levels if selected isn't fully available
            for lvl in self._sorted_levels(common_levels):
                if self._level_available(SECURITY_CONFIGS[lvl], client_caps, server_caps, True, True):
                    return HybridMode.AND_HYBRID, lvl

            self._record_violation(
                policy, "No common level has both classical and PQC algorithms."
            )
            raise ValueError("Policy REQUIRE_HYBRID failed: No viable algorithms.")

        # --- Policy: PQC_ONLY ---
        if policy == CryptoPolicy.PQC_ONLY:
            for lvl in self._sorted_levels(common_levels):
                if self._level_available(SECURITY_CONFIGS[lvl], client_caps, server_caps, False, True):
                    return HybridMode.PQC_ONLY, lvl
            self._record_violation(policy, "No common PQC algorithms available.")
            raise ValueError("Policy PQC_ONLY failed.")

        # --- Policy: CLASSICAL_ONLY ---
        if policy == CryptoPolicy.CLASSICAL_ONLY:
            if not allow_classical_fallback:
                raise ValueError(
                    "CLASSICAL_ONLY requested but explicitly disallowed by caller."
                )
            for lvl in self._sorted_levels(common_levels):
                if self._level_available(SECURITY_CONFIGS[lvl], client_caps, server_caps, True, False):
                    return HybridMode.CLASSICAL_ONLY, lvl
            self._record_violation(policy, "No common classical algorithms available.")
            raise ValueError("Policy CLASSICAL_ONLY failed.")

        # --- Policy: ALLOW_FALLBACK ---
        if policy == CryptoPolicy.ALLOW_FALLBACK:
            # 1. Try Hybrid at preferred level
            if both_hybrid and self._level_available(
                cfg_sel, client_caps, server_caps, True, True
            ):
                return HybridMode.OR_FALLBACK, selected_lvl

            # 2. Try PQC-only at preferred level
            if self._level_available(cfg_sel, client_caps, server_caps, False, True):
                self._record_downgrade(
                    HybridMode.OR_FALLBACK,
                    HybridMode.PQC_ONLY,
                    selected_lvl,
                    selected_lvl,
                )
                return HybridMode.PQC_ONLY, selected_lvl

            # 3. Search other levels for Hybrid
            for lvl in self._sorted_levels(common_levels):
                if both_hybrid and self._level_available(
                    SECURITY_CONFIGS[lvl], client_caps, server_caps, True, True
                ):
                    return HybridMode.OR_FALLBACK, lvl

            # 4. Last resort: Classical-only at highest possible level
            if allow_classical_fallback:
                for lvl in self._sorted_levels(common_levels):
                    if self._level_available(
                        SECURITY_CONFIGS[lvl], client_caps, server_caps, True, False
                    ):
                        self._record_downgrade(
                            HybridMode.OR_FALLBACK,
                            HybridMode.CLASSICAL_ONLY,
                            selected_lvl,
                            lvl,
                        )
                        return HybridMode.CLASSICAL_ONLY, lvl

            raise ValueError("ALLOW_FALLBACK failed to find any viable mode.")

        # --- Policy: PERFORMANCE_ADAPTIVE ---
        if policy == CryptoPolicy.PERFORMANCE_ADAPTIVE:
            # If we have no network metrics, behave like ALLOW_FALLBACK.
            if not network_metrics:
                return self.negotiate(
                    client_caps,
                    server_caps,
                    CryptoPolicy.ALLOW_FALLBACK,
                    None,
                    preferred_level,
                    allow_classical_fallback,
                )

            # If RL is enabled and a PPO policy is loaded, delegate decision to RL.
            if self.use_rl and self.rl_policy is not None:
                # RTT estimate
                rtt_ms = getattr(network_metrics, "smoothed_rtt_ms", None)
                if rtt_ms is None:
                    rtt_ms = getattr(network_metrics, "rtt_ms", 0.0)
                rtt_ms = float(rtt_ms or 0.0)

                using_mock_pqc = bool(getattr(network_metrics, "using_mock_pqc", False))
                strict_pqc = True  # enforcing PQC baseline at L3+ (conceptual)

                supports_pqc = bool(
                    client_caps.supported_pqc and server_caps.supported_pqc
                )
                supports_classical = bool(
                    client_caps.supported_classical and server_caps.supported_classical
                )

                prev_latency_ms = float(
                    getattr(network_metrics, "last_latency_ms", 0.0) or 0.0
                )

                state = build_deployment_state(
                    rtt_ms=rtt_ms,
                    supports_pqc=supports_pqc,
                    supports_classical=supports_classical,
                    strict_pqc=strict_pqc,
                    using_mock_pqc=using_mock_pqc,
                    prev_latency_ms=prev_latency_ms,
                )

                rl_policy, rl_level = self.rl_policy.select(state)

                # Re-enter negotiation with RL-selected policy/level.
                # rl_policy is never PERFORMANCE_ADAPTIVE, so this will not recurse into RL again.
                return self.negotiate(
                    client_caps,
                    server_caps,
                    rl_policy,
                    network_metrics,
                    preferred_level=rl_level,
                    allow_classical_fallback=allow_classical_fallback,
                )

            # ----------------------------
            # Fallback: original heuristic
            # ----------------------------

            # High packet loss (>5%) or low bandwidth (<5Mbps) -> prioritize small packets
            if (
                network_metrics.packet_loss_rate > 0.05
                or network_metrics.bandwidth_mbps < 5.0
            ):
                # Under extreme duress, prefer Classical if allowed.
                if allow_classical_fallback:
                    lvl = min(common_levels, key=lambda x: x.value)
                    if self._level_available(
                        SECURITY_CONFIGS[lvl], client_caps, server_caps, True, False
                    ):
                        return HybridMode.CLASSICAL_ONLY, lvl

            # High latency (>100ms) -> avoid extra RTTs, prefer OR mode so fastest wins.
            if both_hybrid and self._level_available(
                cfg_sel, client_caps, server_caps, True, True
            ):
                return HybridMode.OR_FALLBACK, selected_lvl

            # Fallback standard behavior
            return self.negotiate(
                client_caps,
                server_caps,
                CryptoPolicy.ALLOW_FALLBACK,
                None,
                preferred_level,
                allow_classical_fallback,
            )

        raise ValueError(f"Unknown policy: {policy}")
