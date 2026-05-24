# -*- coding: utf-8 -*-
"""
Multi-Model RL Inference for Hybrid PQC-TLS
==========================================
Production-ready inference module that supports:
- Multiple trained RL models (BC, CQL, IQL, BCQ, AWAC)
- Runtime model selection
- State normalization
- Action-to-policy conversion

Integrates with PolicyEngine for PERFORMANCE_ADAPTIVE mode.
"""

from __future__ import annotations

import os
import numpy as np
from typing import Optional, Dict, Tuple, List
from enum import Enum

from .config import CryptoPolicy, NISTSecurityLevel
from .rl_config import (
    ACTION_LIST,
    STATE_DIM,
    DEFAULT_PATH_CONFIG,
    PathConfig,
    TrainingConfig,
    DEFAULT_TRAINING_CONFIG,
)
from .rl_models import (
    create_algorithm,
    get_available_algorithms,
    OfflineRLAlgorithm,
)


class RLModelType(Enum):
    """Available RL model types."""
    BC = "BC"
    CQL = "CQL"
    IQL = "IQL"
    BCQ = "BCQ"
    AWAC = "AWAC"


# Default model to use if none specified
DEFAULT_MODEL = RLModelType.CQL


class MultiModelRLPolicy:
    """
    Multi-model RL policy for Hybrid PQC-TLS.
    
    Supports loading and switching between multiple trained RL models
    at runtime. Provides a clean interface for PolicyEngine integration.
    
    Usage:
        policy = MultiModelRLPolicy()
        policy.load_model(RLModelType.CQL)
        crypto_policy, level = policy.select(state)
    """
    
    def __init__(
        self,
        path_config: PathConfig = DEFAULT_PATH_CONFIG,
        training_config: TrainingConfig = DEFAULT_TRAINING_CONFIG,
        default_model: RLModelType = DEFAULT_MODEL,
        device: str = "cpu",
    ):
        self.path_config = path_config
        self.training_config = training_config
        self.device = device
        
        # Loaded models cache
        self._models: Dict[RLModelType, OfflineRLAlgorithm] = {}
        self._current_model_type: Optional[RLModelType] = None
        
        # Normalization parameters
        self._state_mean: Optional[np.ndarray] = None
        self._state_std: Optional[np.ndarray] = None
        self._normalization_loaded = False
        
        # Load default model if available
        self._load_normalization()
        if default_model:
            try:
                self.load_model(default_model)
            except FileNotFoundError:
                pass  # Model not yet trained
    
    def _load_normalization(self) -> None:
        """Load state normalization parameters from dataset."""
        dataset_path = self.path_config.dataset_path
        
        if not os.path.exists(dataset_path):
            print(f"[RL] Warning: Dataset not found at {dataset_path}")
            print("[RL] Normalization parameters not loaded. States will be used as-is.")
            return
        
        try:
            data = np.load(dataset_path, allow_pickle=True)
            self._state_mean = data["state_mean"]
            self._state_std = data["state_std"]
            self._normalization_loaded = True
            print("[RL] Loaded normalization parameters from dataset")
        except Exception as e:
            print(f"[RL] Warning: Could not load normalization: {e}")
    
    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        """Normalize state using z-score normalization."""
        if not self._normalization_loaded:
            return state
        
        return (state - self._state_mean) / self._state_std
    
    def get_model_path(self, model_type: RLModelType) -> str:
        """Get the path to a model checkpoint."""
        return os.path.join(
            self.path_config.model_dir,
            f"{model_type.value.lower()}_model.pt"
        )
    
    def is_model_available(self, model_type: RLModelType) -> bool:
        """Check if a model checkpoint exists."""
        return os.path.exists(self.get_model_path(model_type))
    
    def get_available_models(self) -> List[RLModelType]:
        """Get list of available (trained) models."""
        return [m for m in RLModelType if self.is_model_available(m)]
    
    def load_model(self, model_type: RLModelType) -> None:
        """
        Load a specific RL model.
        
        Args:
            model_type: The model type to load
            
        Raises:
            FileNotFoundError: If model checkpoint doesn't exist
        """
        model_path = self.get_model_path(model_type)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model checkpoint not found: {model_path}\n"
                f"Please train the model first using rl_train.py"
            )
        
        # Check if already loaded
        if model_type in self._models:
            self._current_model_type = model_type
            print(f"[RL] Using cached model: {model_type.value}")
            return
        
        # Create and load model
        algorithm = create_algorithm(
            model_type.value,
            config=self.training_config,
            device=self.device,
        )
        algorithm.load(model_path)
        
        self._models[model_type] = algorithm
        self._current_model_type = model_type
        print(f"[RL] Loaded model: {model_type.value}")
    
    def set_model(self, model_type: RLModelType) -> None:
        """
        Set the active model (loading if necessary).
        
        Args:
            model_type: The model type to activate
        """
        if model_type not in self._models:
            self.load_model(model_type)
        else:
            self._current_model_type = model_type
    
    @property
    def current_model(self) -> Optional[RLModelType]:
        """Get the currently active model type."""
        return self._current_model_type
    
    def select(
        self,
        state: np.ndarray,
        model_type: Optional[RLModelType] = None,
    ) -> Tuple[CryptoPolicy, NISTSecurityLevel]:
        """
        Select a crypto policy and security level for the given state.
        
        Args:
            state: 15-dimensional state vector (raw, will be normalized)
            model_type: Optional model to use (uses current model if None)
            
        Returns:
            Tuple of (CryptoPolicy, NISTSecurityLevel)
            
        Raises:
            RuntimeError: If no model is loaded
        """
        # Determine which model to use
        if model_type is not None:
            if model_type not in self._models:
                self.load_model(model_type)
            model = self._models[model_type]
        elif self._current_model_type is not None:
            model = self._models[self._current_model_type]
        else:
            raise RuntimeError(
                "No RL model loaded. Call load_model() first or specify model_type."
            )
        
        # Normalize state
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        
        if len(state) != STATE_DIM:
            raise ValueError(
                f"Expected state dimension {STATE_DIM}, got {len(state)}"
            )
        
        state_normalized = self._normalize_state(state)
        
        # Get action from model
        action_idx = model.select_action(state_normalized, deterministic=True)
        
        # Convert to policy and level
        return self._action_to_policy(action_idx)
    
    def _action_to_policy(
        self,
        action_idx: int,
    ) -> Tuple[CryptoPolicy, NISTSecurityLevel]:
        """Convert action index to CryptoPolicy and NISTSecurityLevel."""
        if action_idx < 0 or action_idx >= len(ACTION_LIST):
            raise IndexError(
                f"Action index {action_idx} out of range [0, {len(ACTION_LIST)})"
            )
        
        policy_name, level_int = ACTION_LIST[action_idx]
        
        # Convert policy name to enum
        policy = CryptoPolicy[policy_name]
        
        # Convert level int to enum
        level = NISTSecurityLevel(level_int)
        
        return policy, level
    
    def get_action_probabilities(
        self,
        state: np.ndarray,
        model_type: Optional[RLModelType] = None,
    ) -> Dict[str, float]:
        """
        Get action probabilities for a state (if supported by model).
        
        Note: Only BC and policy-based methods (IQL, AWAC) support this.
        Q-learning methods (CQL, BCQ) return deterministic actions.
        """
        # Determine model
        if model_type is not None:
            if model_type not in self._models:
                self.load_model(model_type)
            model = self._models[model_type]
            model_name = model_type.value
        elif self._current_model_type is not None:
            model = self._models[self._current_model_type]
            model_name = self._current_model_type.value
        else:
            raise RuntimeError("No RL model loaded.")
        
        # Normalize state
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        state_normalized = self._normalize_state(state)
        
        # Check if model has a policy network
        if not hasattr(model, "policy"):
            # Q-learning methods - return deterministic
            action = model.select_action(state_normalized, deterministic=True)
            probs = {f"{p}_L{l}": 0.0 for p, l in ACTION_LIST}
            policy_name, level = ACTION_LIST[action]
            probs[f"{policy_name}_L{level}"] = 1.0
            return probs
        
        # Policy-based methods - get actual probabilities
        import torch
        with torch.no_grad():
            state_t = torch.FloatTensor(state_normalized).unsqueeze(0)
            probs_t = model.policy.get_probs(state_t).squeeze(0).numpy()
        
        return {
            f"{policy}_L{level}": float(probs_t[i])
            for i, (policy, level) in enumerate(ACTION_LIST)
        }


# ============================================================================
# State Builder for Deployment
# ============================================================================

def build_deployment_state(
    *,
    rtt_ms: float,
    supports_pqc: bool,
    supports_classical: bool,
    strict_pqc: bool = True,
    using_mock_pqc: bool = False,
    prev_latency_ms: float = 0.0,
    # Optional timing metrics (set to 0 if not available pre-handshake)
    kex_keygen_ms: float = 0.0,
    kex_encaps_ms: float = 0.0,
    kex_decaps_ms: float = 0.0,
    kex_hkdf_c_ms: float = 0.0,
    kex_hkdf_s_ms: float = 0.0,
    auth_sign_c_ms: float = 0.0,
    auth_sign_q_ms: float = 0.0,
    auth_ver_c_ms: float = 0.0,
    auth_ver_q_ms: float = 0.0,
    wire_ec: float = 0.0,
    wire_pq: float = 0.0,
) -> np.ndarray:
    """
    Build a 15-dimensional state vector for RL inference.
    
    This function creates a state vector compatible with the trained
    offline RL models. At deployment time (before a handshake), most
    timing and wire metrics will be 0 or estimated.
    
    State layout:
        [0]  rtt_ms
        [1]  trial/step index (can use prev_latency_ms as proxy)
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
        [13] total_wire_ec
        [14] total_wire_pq
        
    For pre-handshake prediction, use typical values or estimates
    based on the expected security level.
    """
    return np.array([
        float(rtt_ms),
        float(prev_latency_ms),  # Using as trial/history feature
        1.0 if using_mock_pqc else 0.0,
        1.0 if strict_pqc else 0.0,
        float(kex_keygen_ms),
        float(kex_encaps_ms),
        float(kex_decaps_ms),
        float(kex_hkdf_c_ms),
        float(kex_hkdf_s_ms),
        float(auth_sign_c_ms),
        float(auth_sign_q_ms),
        float(auth_ver_c_ms),
        float(auth_ver_q_ms),
        float(wire_ec),
        float(wire_pq),
    ], dtype=np.float32)


def build_simple_state(
    rtt_ms: float,
    using_mock_pqc: bool = False,
    strict_pqc: bool = True,
) -> np.ndarray:
    """
    Build a minimal state vector with just RTT and PQC flags.
    All other features are set to 0.
    
    Use this for simple deployments where detailed metrics aren't available.
    """
    return build_deployment_state(
        rtt_ms=rtt_ms,
        supports_pqc=True,
        supports_classical=True,
        strict_pqc=strict_pqc,
        using_mock_pqc=using_mock_pqc,
    )


# ============================================================================
# Convenience Functions
# ============================================================================

# Global instance for simple usage
_global_policy: Optional[MultiModelRLPolicy] = None


def get_global_policy() -> MultiModelRLPolicy:
    """Get or create the global RL policy instance."""
    global _global_policy
    if _global_policy is None:
        _global_policy = MultiModelRLPolicy()
    return _global_policy


def select_policy(
    rtt_ms: float,
    model_type: RLModelType = DEFAULT_MODEL,
    **kwargs,
) -> Tuple[CryptoPolicy, NISTSecurityLevel]:
    """
    Convenience function to select a crypto policy for given RTT.
    
    Args:
        rtt_ms: Round-trip time in milliseconds
        model_type: RL model to use
        **kwargs: Additional state features (see build_deployment_state)
        
    Returns:
        Tuple of (CryptoPolicy, NISTSecurityLevel)
    """
    policy = get_global_policy()
    
    state = build_deployment_state(
        rtt_ms=rtt_ms,
        supports_pqc=True,
        supports_classical=True,
        **kwargs,
    )
    
    return policy.select(state, model_type=model_type)


# ============================================================================
# Integration with Original rl_inference.py Interface
# ============================================================================

class HybridPPOPolicy:
    """
    Backwards-compatible interface for PolicyEngine integration.
    
    This class wraps MultiModelRLPolicy to provide the same interface
    as the original rl_inference.py HybridPPOPolicy class.
    
    Note: Despite the name, this now supports multiple model types,
    not just PPO. The default is CQL which typically performs best.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,  # Ignored for compatibility
        dataset_path: Optional[str] = None,  # Ignored for compatibility
        device: str = "cpu",
        model_type: RLModelType = DEFAULT_MODEL,
    ):
        self._policy = MultiModelRLPolicy(device=device)
        self._model_type = model_type
        
        # Try to load the specified model
        try:
            self._policy.load_model(model_type)
        except FileNotFoundError:
            # Fallback: try other models
            available = self._policy.get_available_models()
            if available:
                self._model_type = available[0]
                self._policy.load_model(self._model_type)
            else:
                raise FileNotFoundError(
                    "No trained RL models found. Please run training first."
                )
    
    def select(
        self,
        state: np.ndarray,
    ) -> Tuple[CryptoPolicy, NISTSecurityLevel]:
        """
        Select crypto policy for given state.
        
        Compatible with original HybridPPOPolicy.select() interface.
        """
        return self._policy.select(state, model_type=self._model_type)
    
    def set_model(self, model_type: RLModelType) -> None:
        """Set the active model type."""
        self._policy.load_model(model_type)
        self._model_type = model_type


# ============================================================================
# Main (for testing)
# ============================================================================

if __name__ == "__main__":
    print("Multi-Model RL Policy for Hybrid PQC-TLS")
    print("=" * 50)
    
    policy = MultiModelRLPolicy()
    
    print("\nAvailable models:")
    for model_type in RLModelType:
        available = "✓" if policy.is_model_available(model_type) else "✗"
        print(f"  {available} {model_type.value}")
    
    available_models = policy.get_available_models()
    if available_models:
        print(f"\nTesting with model: {available_models[0].value}")
        policy.load_model(available_models[0])
        
        # Test different RTT scenarios
        test_rtts = [0, 25, 50, 100, 200]
        print("\nPolicy selections by RTT:")
        for rtt in test_rtts:
            state = build_simple_state(rtt_ms=rtt)
            crypto_policy, level = policy.select(state)
            print(f"  RTT={rtt:3d}ms -> {crypto_policy.value} @ Level {level.value}")
    else:
        print("\nNo trained models available. Run training first:")
        print("  python -m hybrid_pqc_tls.rl_train")
