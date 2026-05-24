# -*- coding: utf-8 -*-
"""
Hybrid PQC-TLS Library with Offline RL
======================================
Top-level package exposing key classes for easy import.

Core Components:
    - NISTSecurityLevel, CryptoPolicy, Role, HybridMode (config enums)
    - PolicyEngine (policy negotiation)
    - FullHybridSession (handshake orchestration)

RL Components:
    - MultiModelRLPolicy (multi-model inference)
    - RLModelType (available RL models: BC, CQL, IQL, BCQ, AWAC)
    - build_deployment_state (state vector builder)
    - ComprehensiveEvaluator, PublicationVisualizer (v2 evaluation)
"""

# =============================================================================
# Core Components (existing)
# =============================================================================
from .config import (
    NISTSecurityLevel,
    CryptoPolicy,
    Role,
    HybridMode,
    PathMask,
    AlgorithmCapabilities,
    NetworkMetrics,
    SecurityLevelConfig,
    AuthConfig,
    HybridKEXResult,
    ServerAuthKeys,
    AuthBundle,
    SECURITY_CONFIGS,
    AUTH_CONFIG,
)

from .policy import PolicyEngine
from .session import FullHybridSession

# =============================================================================
# RL Components - Configuration
# =============================================================================
from .rl_config import (
    ACTION_LIST,
    ACTION_TO_IDX,
    NUM_ACTIONS,
    STATE_DIM,
    STATE_FEATURES,
    RewardConfig,
    TrainingConfig,
    PathConfig,
    DEFAULT_REWARD_CONFIG,
    DEFAULT_TRAINING_CONFIG,
    DEFAULT_PATH_CONFIG,
)

# =============================================================================
# RL Components - Models
# =============================================================================
from .rl_models import (
    BehavioralCloning,
    CQL,
    IQL,
    BCQ,
    AWAC,
    create_algorithm,
    get_available_algorithms,
)

# =============================================================================
# RL Components - Inference
# =============================================================================
from .rl_inference_multi import (
    RLModelType,
    MultiModelRLPolicy,
    HybridPPOPolicy,  # Backwards-compatible interface
    build_deployment_state,
    build_simple_state,
    select_policy,
)

# =============================================================================
# RL Components - Dataset Building
# =============================================================================
from .rl_offline_dataset import (
    OfflineRLDatasetBuilder,
    build_and_save_dataset,
    compute_oracle_actions,
)

from .rl_dataset_improved import (
    ImprovedDatasetBuilder,
    build_improved_dataset,
)

# =============================================================================
# RL Components - Training
# =============================================================================
from .rl_train import (
    OfflineRLTrainer,
    run_training,
)

# =============================================================================
# RL Components - Evaluation (v2 - RL vs Rule-Based)
# =============================================================================
from .rl_evaluate_v2 import (
    ComprehensiveEvaluator,
    PublicationVisualizer,
    RuleBasedPolicy,
    OraclePolicy,
    EvalMetrics,
    run_comprehensive_evaluation,
    generate_latex_tables,
)

# Legacy v1 evaluation (for backwards compatibility)
from .rl_evaluate import (
    OfflineRLEvaluator,
    ResultVisualizer,
    RuleBasedBaseline,
    FixedPolicyBaseline,
    OracleBaseline,
    run_evaluation,
)

# =============================================================================
# Package Metadata
# =============================================================================
__version__ = "2.1.0"
__author__ = "Shawmoon"

__all__ = [
    # Core - Config
    "NISTSecurityLevel",
    "CryptoPolicy",
    "Role",
    "HybridMode",
    "PathMask",
    "AlgorithmCapabilities",
    "NetworkMetrics",
    "SECURITY_CONFIGS",
    "AUTH_CONFIG",
    
    # Core - Main Classes
    "PolicyEngine",
    "FullHybridSession",
    
    # RL - Config
    "ACTION_LIST",
    "ACTION_TO_IDX",
    "NUM_ACTIONS",
    "STATE_DIM",
    "STATE_FEATURES",
    "RLModelType",
    "RewardConfig",
    "TrainingConfig",
    "PathConfig",
    "DEFAULT_REWARD_CONFIG",
    "DEFAULT_TRAINING_CONFIG",
    "DEFAULT_PATH_CONFIG",
    
    # RL - Models
    "BehavioralCloning",
    "CQL",
    "IQL",
    "BCQ",
    "AWAC",
    "create_algorithm",
    "get_available_algorithms",
    
    # RL - Inference
    "MultiModelRLPolicy",
    "HybridPPOPolicy",
    "build_deployment_state",
    "build_simple_state",
    "select_policy",
    
    # RL - Dataset
    "OfflineRLDatasetBuilder",
    "ImprovedDatasetBuilder",
    "build_and_save_dataset",
    "build_improved_dataset",
    "compute_oracle_actions",
    
    # RL - Training
    "OfflineRLTrainer",
    "run_training",
    
    # RL - Evaluation (v2 - primary)
    "ComprehensiveEvaluator",
    "PublicationVisualizer",
    "RuleBasedPolicy",
    "OraclePolicy",
    "EvalMetrics",
    "run_comprehensive_evaluation",
    "generate_latex_tables",
    
    # RL - Evaluation (v1 - legacy)
    "OfflineRLEvaluator",
    "ResultVisualizer",
    "run_evaluation",
]
