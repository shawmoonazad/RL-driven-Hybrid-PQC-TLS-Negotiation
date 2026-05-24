# -*- coding: utf-8 -*-
"""
Hybrid PQC-TLS Protocol State Machines
======================================
Core logic for:
- Hybrid Key Exchange (ECDH + ML-KEM)
- Hybrid Authentication (ECDSA + ML-DSA)
- Hybrid HKDF (Key Derivation)
- TLS 1.3 Mini-Record Layer (AES-GCM)

[MODIFIED]: This version is updated to use the factory pattern
(make_mldsa_adapter) from primitives.py instead of static calls.
"""

import hashlib
import hmac
import time
import os
import struct
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.kdf.hkdf import HKDF, HKDFExpand
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from .config import (
    NISTSecurityLevel, HybridMode, PathMask, Role, CryptoPolicy,
    SECURITY_CONFIGS, AUTH_CONFIG, HybridKEXResult,
    AlgorithmCapabilities, NetworkMetrics, ServerAuthKeys, AuthBundle
)
# *** CHANGE ***: Import the factories and adapters
from .primitives import make_kem_adapter, ECDSAAdapter, make_mldsa_adapter
from .policy import PolicyEngine

# ============================================================================
# Hybrid HKDF Combiner (No changes)
# ============================================================================

class HybridHKDF:
    """
    Domain-separated HKDF for hybrid TLS 1.3 KEX according to NIST SP 800-56C.
    Binds derived secrets to the specific policy, group, and commit mask used.
    """
    LABEL = b"NIST_HYBRID_PQC_TLS_v3"

    @staticmethod
    def derive(
        transcript_hash: bytes,
        named_group: str,
        level: NISTSecurityLevel,
        policy: CryptoPolicy,
        commit_mask: PathMask,
        classical_secret: Optional[bytes],
        pqc_secret: Optional[bytes],
        loser_commitment: Optional[bytes],
        hash_class: type,
        output_length: int
    ) -> bytes:
        # Input validation
        need_cl = bool(commit_mask & PathMask.CLASSICAL)
        need_pq = bool(commit_mask & PathMask.PQC)
        if need_cl and not classical_secret: raise ValueError("Missing classical secret for mask")
        if need_pq and not pqc_secret: raise ValueError("Missing PQC secret for mask")

        # Context binding info
        info = b"|".join([
            HybridHKDF.LABEL,
            named_group.encode("ascii"),
            bytes([level.value & 0xFF]),
            policy.value.encode("ascii"),
            bytes([int(commit_mask) & 0xFF]),
            transcript_hash
        ])

        # Deterministic IKM construction
        parts = []
        if need_cl: parts.append((b"CL", classical_secret))
        if need_pq: parts.append((b"PQ", pqc_secret))
        parts.sort(key=lambda x: x[0])
        ikm = b"".join(tag + sec for tag, sec in parts)

        if loser_commitment:
            ikm += b"LC" + loser_commitment

        hkdf = HKDF(algorithm=hash_class(), length=output_length,
                    salt=transcript_hash, info=info, backend=default_backend())
        return hkdf.derive(ikm)

# ============================================================================
# Hybrid Key Exchange State Machine (No changes)
# ============================================================================

class NISTHybridKeyExchange:
    """
    Manages the ephemeral key exchange lifecycle:
    KeyGen -> Encaps/Decaps -> Secret Derivation
    """
    _LOSER_COMMIT_LABEL = b"tls13/hybrid/loser-commit/v1"

    def __init__(self, security_level: NISTSecurityLevel, mode: HybridMode,
                 role: Role, policy: CryptoPolicy, enable_short_circuit: bool = True):
        self.level = security_level
        self.mode = mode
        self.role = role
        self.policy = policy
        self.enable_short_circuit = enable_short_circuit
        self.config = SECURITY_CONFIGS[security_level]
        self.kem_adapter, self.kem_sizes = make_kem_adapter(security_level)

        self.state = "INIT"
        self.timing_data = {}
        self.wire_sizes = {}
        self.commit_mask = PathMask.NONE
        self.loser_commitment = None

        # Ephemeral keys/secrets
        self.ecdh_priv = None
        self.kyber_priv = None
        self.classical_shared = None
        self.pqc_shared = None

    @classmethod
    def from_negotiation(cls, engine: PolicyEngine, cli_caps, srv_caps, policy, role,
                         net_metrics=None, pref_level=NISTSecurityLevel.LEVEL_3):
        mode, level = engine.negotiate(cli_caps, srv_caps, policy, net_metrics, pref_level)
        return cls(level, mode, role, policy)

    def server_keygen(self) -> Tuple[bytes, bytes]:
        """Step 1 (Server): Generate ephemeral key shares."""
        assert self.role == Role.SERVER and self.state == "INIT"
        t0 = time.perf_counter()

        # Classical ECDH
        ec_pub_bytes = b""
        if self.mode != HybridMode.PQC_ONLY:
            self.ecdh_priv = ec.generate_private_key(self.config.classical_curve, default_backend())
            ec_pub_bytes = self.ecdh_priv.public_key().public_bytes(
                serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint)

        # PQC ML-KEM
        pq_pub_bytes = b""
        if self.mode != HybridMode.CLASSICAL_ONLY:
            pq_pub_bytes, self.kyber_priv = self.kem_adapter.keygen()

        self.wire_sizes.update({'srv_ecdh': len(ec_pub_bytes), 'srv_pqc': len(pq_pub_bytes)})
        self.timing_data['keygen_ms'] = (time.perf_counter() - t0) * 1000.0
        self.state = "WAIT_CLIENT"
        return ec_pub_bytes, pq_pub_bytes

    def client_encaps(self, srv_ec_pub: bytes, srv_pq_pub: bytes) -> Tuple[bytes, bytes, PathMask, Optional[bytes]]:
        """Step 2 (Client): Process server shares, generate own shares/ciphertext."""
        assert self.role == Role.CLIENT and self.state == "INIT"
        t0 = time.perf_counter()
        self.wire_sizes.update({'srv_ecdh': len(srv_ec_pub or b""), 'srv_pqc': len(srv_pq_pub or b"")})

        # 1. Attempt Classical
        ec_success, ec_pub, ec_time = False, b"", 0.0
        if self.mode != HybridMode.PQC_ONLY and srv_ec_pub:
            t_ec = time.perf_counter()
            try:
                self.ecdh_priv = ec.generate_private_key(self.config.classical_curve, default_backend())
                peer = ec.EllipticCurvePublicKey.from_encoded_point(self.config.classical_curve, srv_ec_pub)
                self.classical_shared = self.ecdh_priv.exchange(ec.ECDH(), peer)
                ec_pub = self.ecdh_priv.public_key().public_bytes(
                     serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint)
                ec_success = True
            except Exception: pass
            ec_time = time.perf_counter() - t_ec

        # 2. Attempt PQC
        pq_success, pq_ct, pq_time = False, b"", 0.0
        if self.mode != HybridMode.CLASSICAL_ONLY and srv_pq_pub:
            t_pq = time.perf_counter()
            try:
                self.pqc_shared, pq_ct = self.kem_adapter.encaps(srv_pq_pub)
                pq_success = True
            except Exception: pass
            pq_time = time.perf_counter() - t_pq

        # 3. Determine Mask & Optional Short-Circuit
        mask = PathMask.NONE
        loser_commit = None

        if self.mode == HybridMode.AND_HYBRID:
            if ec_success and pq_success: mask = PathMask.BOTH
            else: raise RuntimeError("AND_HYBRID requires both paths to succeed")
        elif self.mode == HybridMode.PQC_ONLY:
             if pq_success: mask = PathMask.PQC
             else: raise RuntimeError("PQC_ONLY path failed")
        elif self.mode == HybridMode.CLASSICAL_ONLY:
             if ec_success: mask = PathMask.CLASSICAL
             else: raise RuntimeError("CLASSICAL_ONLY path failed")
        elif self.mode == HybridMode.OR_FALLBACK:
            if not self.enable_short_circuit:
                 if ec_success and pq_success: mask = PathMask.BOTH
                 elif pq_success: mask = PathMask.PQC
                 elif ec_success: mask = PathMask.CLASSICAL
            else:
                 if ec_success and (not pq_success or ec_time < pq_time):
                     mask = PathMask.CLASSICAL
                     if pq_success:
                         loser_commit = hashlib.sha256(self._LOSER_COMMIT_LABEL + b"|ct=" + pq_ct).digest()
                         pq_ct = b""
                         self.pqc_shared = None
                 elif pq_success:
                     mask = PathMask.PQC
                     if ec_success:
                          loser_commit = hashlib.sha256(self._LOSER_COMMIT_LABEL + b"|pub=" + ec_pub).digest()
                          ec_pub = b""
                          self.classical_shared = None
                 else: raise RuntimeError("OR_FALLBACK: Both paths failed")

        self.commit_mask = mask
        self.loser_commitment = loser_commit
        self.wire_sizes.update({'cli_ecdh': len(ec_pub), 'cli_pqc': len(pq_ct)})
        self.timing_data['encaps_ms'] = (time.perf_counter() - t0) * 1000.0
        self.state = "DONE"
        return ec_pub, pq_ct, mask, loser_commit

    def server_decaps(self, cli_ec_pub: bytes, cli_pq_ct: bytes,
                      cli_mask: PathMask, loser_commit: Optional[bytes]):
        """Step 3 (Server): Process client shares/ciphertext based on received mask."""
        assert self.role == Role.SERVER and self.state == "WAIT_CLIENT"
        t0 = time.perf_counter()

        if cli_mask & PathMask.CLASSICAL:
            if not cli_ec_pub: raise ValueError("Mask says CLASSICAL but no pubkey received")
            peer = ec.EllipticCurvePublicKey.from_encoded_point(self.config.classical_curve, cli_ec_pub)
            self.classical_shared = self.ecdh_priv.exchange(ec.ECDH(), peer)

        if cli_mask & PathMask.PQC:
            if not cli_pq_ct: raise ValueError("Mask says PQC but no ciphertext received")
            self.pqc_shared = self.kem_adapter.decaps(self.kyber_priv, cli_pq_ct)

        self.commit_mask = cli_mask
        self.loser_commitment = loser_commit
        self.wire_sizes.update({'cli_ecdh': len(cli_ec_pub or b""), 'cli_pqc': len(cli_pq_ct or b"")})
        self.timing_data['decaps_ms'] = (time.perf_counter() - t0) * 1000.0
        self.state = "DONE"

    def finalize_secret(self, transcript_hash: bytes) -> HybridKEXResult:
        """Final Step: Derive the master shared secret."""
        assert self.state == "DONE"
        t0 = time.perf_counter()
        named_group = f"{self.config.classical_name}+{self.config.pqc_algorithm}"

        final_secret = HybridHKDF.derive(
            transcript_hash, named_group, self.level, self.policy,
            self.commit_mask, self.classical_shared, self.pqc_shared,
            self.loser_commitment, self.config.hash_class, self.config.kdf_output_length
        )
        self.timing_data['hkdf_ms'] = (time.perf_counter() - t0) * 1000.0
        return HybridKEXResult(
            shared_secret=final_secret,
            classical_shared=self.classical_shared, pqc_shared=self.pqc_shared,
            mode=self.mode, security_level=self.level, commit_mask=self.commit_mask,
            timing=self.timing_data, wire_sizes=self.wire_sizes,
            security_properties={}, telemetry={}
        )

# ============================================================================
# Hybrid Authentication State Machine (*** MODIFIED ***)
# ============================================================================

class NISTHybridAuthentication:
    """Handles dual-signature generation and verification bound to the KEX transcript."""

    def __init__(self, policy_engine: PolicyEngine, level: NISTSecurityLevel,
                 policy: CryptoPolicy, role: Role):
        self.level = level
        self.policy = policy
        self.role = role
        
        # *** CHANGE ***
        # Load static config and instantiate the ML-DSA adapter immediately.
        # This honors STRICT_PQC and sets the USING_MOCK_PQC flag if needed.
        self.config = AUTH_CONFIG[level]
        self.mldsa_adapter = make_mldsa_adapter(self.config.dsa)
        # We can keep using the static ECDSAAdapter as its code is stable
        self.ecdsa_adapter = ECDSAAdapter 
        
        self.server_keys: Optional[ServerAuthKeys] = None
        self.telemetry = {"timings_ms": {}, "sizes": {}}

    def server_keygen(self) -> ServerAuthKeys:
        """Generate long-term identity keys (Server only)."""
        assert self.role == Role.SERVER
        
        # *** CHANGE ***
        # ECDSA call is unchanged (static)
        esk, epk = self.ecdsa_adapter.generate_keypair(self.config.curve)
        # ML-DSA call now uses the instance, no param needed
        qsk, qpk = self.mldsa_adapter.generate_keypair()
        
        self.server_keys = ServerAuthKeys(esk, epk, qsk, qpk)
        return self.server_keys

    def _bind_context(self, transcript: bytes, mask: PathMask, epk_bytes, qpk_bytes) -> bytes:
        """Create domain negotiation context for signatures."""
        parts = [
            b"AUTH|v1", transcript,
            b"L", str(self.level.value).encode(),
            b"M", bytes([int(mask)]),
            b"EPK", (hashlib.sha256(epk_bytes).digest() if epk_bytes else b"\x00"*32),
            b"QPK", (hashlib.sha256(qpk_bytes).digest() if qpk_bytes else b"\x00"*32)
        ]
        return hashlib.sha256(b"|".join(parts)).digest()

    def server_sign(self, transcript_hash: bytes) -> AuthBundle:
        assert self.role == Role.SERVER and self.server_keys
        mask = PathMask.BOTH
        if self.policy == CryptoPolicy.CLASSICAL_ONLY: mask = PathMask.CLASSICAL
        elif self.policy == CryptoPolicy.PQC_ONLY: mask = PathMask.PQC

        epk_b = self.ecdsa_adapter.pubkey_bytes_compressed(self.server_keys.ecdsa_pk) if (mask & PathMask.CLASSICAL) else None
        qpk_b = self.server_keys.mldsa_pk if (mask & PathMask.PQC) else None
        ctx = self._bind_context(transcript_hash, mask, epk_b, qpk_b)

        t0 = time.perf_counter()
        sig_c = None
        if mask & PathMask.CLASSICAL:
            # *** CHANGE *** (using self.ecdsa_adapter for consistency)
            sig_c = self.ecdsa_adapter.sign(self.server_keys.ecdsa_sk, ctx, self.config.hash_name)
        
        t1 = time.perf_counter()
        sig_q = None
        if mask & PathMask.PQC:
             # *** CHANGE *** (using self.mldsa_adapter instance)
            sig_q = self.mldsa_adapter.sign(self.server_keys.mldsa_sk, ctx)
        t2 = time.perf_counter()

        self.telemetry["timings_ms"].update({"sign_c": (t1-t0)*1e3, "sign_q": (t2-t1)*1e3})
        return AuthBundle(self.level, sig_c, sig_q, mask, self.config.curve, self.config.dsa,
                          epk_b, qpk_b, time.time()-60, time.time()+3600)

    def client_verify(self, bundle: AuthBundle, transcript_hash: bytes) -> bool:
        assert self.role == Role.CLIENT
        if bundle.level != self.level: return False

        ctx = self._bind_context(transcript_hash, bundle.mask, bundle.ecdsa_pk_bytes, bundle.mldsa_pk_bytes)
        t0 = time.perf_counter()
        ok_c = False # Default to fail
        if bundle.mask & PathMask.CLASSICAL:
            try:
                epk = ec.EllipticCurvePublicKey.from_encoded_point(
                    self.ecdsa_adapter.CURVE_MAP[self.config.curve], bundle.ecdsa_pk_bytes)
                # *** CHANGE *** (using self.ecdsa_adapter for consistency)
                ok_c = self.ecdsa_adapter.verify(epk, bundle.classical_sig, ctx, self.config.hash_name)
            except Exception: ok_c = False
        else:
            ok_c = True # Not required, so "passes"
        t1 = time.perf_counter()

        ok_q = False # Default to fail
        if bundle.mask & PathMask.PQC:
            # *** CHANGE *** (using self.mldsa_adapter instance)
            ok_q = self.mldsa_adapter.verify(bundle.mldsa_pk_bytes, bundle.pqc_sig, ctx)
        else:
            ok_q = True # Not required, so "passes"
        t2 = time.perf_counter()

        self.telemetry["timings_ms"].update({"ver_c": (t1-t0)*1e3, "ver_q": (t2-t1)*1e3})
        
        # Policy enforcement
        if self.policy == CryptoPolicy.REQUIRE_HYBRID:
            return ok_c and ok_q
        if self.policy == CryptoPolicy.CLASSICAL_ONLY:
            return ok_c
        if self.policy == CryptoPolicy.PQC_ONLY:
            return ok_q
        # ALLOW_FALLBACK
        return ok_c or ok_q


# ============================================================================
# TLS 1.3 Mini-Record Layer Helpers (No changes)
# ============================================================================

@dataclass
class TrafficKeys:
    key: bytes
    iv: bytes

def derive_tls13_keys(master_secret: bytes, transcript_hash: bytes,
                      hash_name: str = "SHA256", key_len=16, iv_len=12) -> Tuple[TrafficKeys, TrafficKeys]:
    """Simplified TLS 1.3 traffic key derivation from master secret."""
    hash_alg = getattr(hashes, hash_name)()
    def hkdf_expand_label(secret, label, context, length):
        full_label = b"tls13 " + label.encode("ascii")
        info = struct.pack("!H", length) + bytes([len(full_label)]) + full_label + bytes([len(context)]) + context
        return HKDFExpand(algorithm=hash_alg, length=length, info=info).derive(secret)

    # Derive Application Traffic Secrets
    c_secret = hkdf_expand_label(master_secret, "c ap traffic", transcript_hash, hash_alg.digest_size)
    s_secret = hkdf_expand_label(master_secret, "s ap traffic", transcript_hash, hash_alg.digest_size)

    # Derive Key/IV
    ck = TrafficKeys(hkdf_expand_label(c_secret, "key", b"", key_len),
                     hkdf_expand_label(c_secret, "iv", b"", iv_len))
    sk = TrafficKeys(hkdf_expand_label(s_secret, "key", b"", key_len),
                     hkdf_expand_label(s_secret, "iv", b"", iv_len))
    return ck, sk

def tls13_encrypt(tk: TrafficKeys, seq: int, plaintext: bytes, type_byte: int = 23) -> bytes:
    nonce = bytes(a ^ b for a, b in zip(tk.iv, seq.to_bytes(12, "big")))
    aad = bytes([type_byte, 3, 3]) + struct.pack("!H", len(plaintext) + 1 + 16) # rough AAD
    return AESGCM(tk.key).encrypt(nonce, plaintext + bytes([type_byte]), aad)

def tls13_decrypt(tk: TrafficKeys, seq: int, ciphertext: bytes, type_byte: int = 23) -> bytes:
    nonce = bytes(a ^ b for a, b in zip(tk.iv, seq.to_bytes(12, "big")))
    aad = bytes([type_byte, 3, 3]) + struct.pack("!H", len(ciphertext))
    pt = AESGCM(tk.key).decrypt(nonce, ciphertext, aad)
    return pt[:-1] # strip type byte