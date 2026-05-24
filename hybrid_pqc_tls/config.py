# -*- coding: utf-8 -*-
"""
Hybrid PQC-TLS Configuration Definitions
========================================
Central repository for all enumerations, dataclasses, and static configuration
maps used throughout the Hybrid PQC-TLS framework.

Conforms to NIST SP 800-56C Rev. 2 and FIPS 203/204 standards.
"""

from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Dict, List, Optional, Set, Any, Tuple

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec

# ============================================================================
# Core Enumerations
# ============================================================================

class NISTSecurityLevel(Enum):
    """NIST PQC security levels mapped to classical integer equivalents."""
    LEVEL_1 = 1  # ~AES-128
    LEVEL_3 = 3  # ~AES-192
    LEVEL_5 = 5  # ~AES-256

class HybridMode(Enum):
    """Operational modes for hybrid key exchange."""
    AND_HYBRID = "and"             # Both paths must succeed
    OR_FALLBACK = "or"             # Prefer hybrid, allow single path fallback
    PQC_ONLY = "pqc"               # Pure post-quantum
    CLASSICAL_ONLY = "classical"   # Pure classical

class PathMask(IntEnum):
    """Bitmask representing active cryptographic paths in a session."""
    NONE = 0
    CLASSICAL = 1
    PQC = 2
    BOTH = 3

class Role(Enum):
    """Protocol role of the endpoint."""
    CLIENT = "client"
    SERVER = "server"

class CryptoPolicy(Enum):
    """High-level administrative policies dictating negotiation behavior."""
    REQUIRE_HYBRID = "require_hybrid"
    ALLOW_FALLBACK = "allow_fallback"
    PQC_ONLY = "pqc_only"
    CLASSICAL_ONLY = "classical_only"
    PERFORMANCE_ADAPTIVE = "performance_adaptive"

# ============================================================================
# Data Structures (Capabilities & Metrics)
# ============================================================================

@dataclass
class AlgorithmCapabilities:
    """Represents the supported cryptographic algorithms of an endpoint."""
    supported_levels: Set[NISTSecurityLevel]
    supported_classical: Set[str]
    supported_pqc: Set[str]
    supports_hybrid: bool
    max_latency_ms: Optional[float] = None
    max_wire_bytes: Optional[int] = None

@dataclass
class NetworkMetrics:
    """Real-time network conditions used by PERFORMANCE_ADAPTIVE policy."""
    rtt_ms: float
    bandwidth_mbps: float
    packet_loss_rate: float
    mtu: int
    recent_latencies: List[float]

@dataclass
class SecurityLevelConfig:
    """Static configuration for a specific NIST security level (KEX)."""
    level: NISTSecurityLevel
    classical_curve: ec.EllipticCurve
    classical_name: str
    classical_bits: int
    classical_secret_size: int
    pqc_algorithm: str
    pqc_nist_level: int
    pqc_bits: int
    hash_class: type
    kdf_output_length: int

@dataclass
class AuthConfig:
    """Static configuration for a specific NIST security level (Auth)."""
    level: NISTSecurityLevel
    curve: str           # ECDSA curve name
    dsa: str             # ML-DSA parameter set name
    hash_name: str       # ECDSA hash algorithm name

@dataclass
class HybridKEXResult:
    """Results of a completed hybrid key exchange."""
    shared_secret: bytes
    classical_shared: Optional[bytes]
    pqc_shared: Optional[bytes]
    mode: HybridMode
    security_level: NISTSecurityLevel
    commit_mask: PathMask
    timing: Dict[str, float]
    wire_sizes: Dict[str, int]
    security_properties: Dict[str, Any]
    telemetry: Dict[str, Any]

@dataclass
class ServerAuthKeys:
    """Container for server's long-term authentication keypairs."""
    ecdsa_sk: object
    ecdsa_pk: object
    mldsa_sk: bytes
    mldsa_pk: bytes

@dataclass
class AuthBundle:
    """The authentication artifacts sent over the wire."""
    level: NISTSecurityLevel
    classical_sig: Optional[bytes]
    pqc_sig: Optional[bytes]
    mask: PathMask
    curve: str
    dsa: str
    ecdsa_pk_bytes: Optional[bytes]
    mldsa_pk_bytes: Optional[bytes]
    not_before: float
    not_after: float

# ============================================================================
# Static Configuration Maps
# ============================================================================

SECURITY_CONFIGS = {
    NISTSecurityLevel.LEVEL_1: SecurityLevelConfig(
        level=NISTSecurityLevel.LEVEL_1,
        classical_curve=ec.SECP256R1(),
        classical_name="P-256",
        classical_bits=128,
        classical_secret_size=32,
        pqc_algorithm="ML-KEM-512",
        pqc_nist_level=1,
        pqc_bits=128,
        hash_class=hashes.SHA256,
        kdf_output_length=32
    ),
    NISTSecurityLevel.LEVEL_3: SecurityLevelConfig(
        level=NISTSecurityLevel.LEVEL_3,
        classical_curve=ec.SECP384R1(),
        classical_name="P-384",
        classical_bits=192,
        classical_secret_size=48,
        pqc_algorithm="ML-KEM-768",
        pqc_nist_level=3,
        pqc_bits=192,
        hash_class=hashes.SHA384,
        kdf_output_length=48
    ),
    NISTSecurityLevel.LEVEL_5: SecurityLevelConfig(
        level=NISTSecurityLevel.LEVEL_5,
        classical_curve=ec.SECP521R1(),
        classical_name="P-521",
        classical_bits=256,
        classical_secret_size=66,
        pqc_algorithm="ML-KEM-1024",
        pqc_nist_level=5,
        pqc_bits=256,
        hash_class=hashes.SHA512,
        kdf_output_length=64
    )
}

AUTH_CONFIG = {
    NISTSecurityLevel.LEVEL_1: AuthConfig(NISTSecurityLevel.LEVEL_1, "P-256", "ML-DSA-44", "SHA-256"),
    NISTSecurityLevel.LEVEL_3: AuthConfig(NISTSecurityLevel.LEVEL_3, "P-384", "ML-DSA-65", "SHA-384"),
    NISTSecurityLevel.LEVEL_5: AuthConfig(NISTSecurityLevel.LEVEL_5, "P-521", "ML-DSA-87", "SHA-512"),
}