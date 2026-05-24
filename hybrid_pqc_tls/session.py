# -*- coding: utf-8 -*-
"""
Full Hybrid Session Orchestrator
================================
End-to-end, multi-flight hybrid PQC-TLS-ish handshake with:
- deterministic JSON transcript hashing
- client/server capability negotiation
- GREASE-like dummy groups
- PQC attestation that also carries "using_mock_pqc"
- dual-auth handshake (server signs transcript, client verifies)
- separate send/recv sequence numbers

[MODIFIED]: Re-added simulate_handshake() as an evaluation harness.
It now calls the 5-flight API internally and returns rich telemetry.
"""

import hashlib
import time
import json
import os
from typing import Optional, Dict, Any

from .config import (
    NISTSecurityLevel,
    CryptoPolicy,
    Role,
    AlgorithmCapabilities,
    HybridMode,
    AuthBundle,
    PathMask
)
from .policy import PolicyEngine
from .protocol import (
    NISTHybridKeyExchange,
    NISTHybridAuthentication,
    derive_tls13_keys,
    tls13_encrypt,
    tls13_decrypt,
)
# This import is now required for simulate_handshake
from .primitives import pqc_runtime_status


# ============================================================================
# JSON / transcript helpers
# ============================================================================

def _bytes_to_str(b: Optional[bytes]) -> Optional[str]:
    """Deterministically encode bytes to a string for JSON."""
    if b is None:
        return None
    # latin-1 is 1:1 for 0..255, so reversible
    return b.decode("latin-1")


def _str_to_bytes(s: Optional[str]) -> Optional[bytes]:
    if s is None:
        return None
    return s.encode("latin-1")


def _caps_to_dict(caps: AlgorithmCapabilities) -> dict:
    """Deterministic, JSON-safe view of capabilities."""
    return {
        "supported_levels": sorted([lvl.value for lvl in caps.supported_levels]),
        "supported_classical": sorted(list(caps.supported_classical)),
        "supported_pqc": sorted(list(caps.supported_pqc)),
        "supports_hybrid": bool(caps.supports_hybrid),
        "max_latency_ms": caps.max_latency_ms,
        "max_wire_bytes": caps.max_wire_bytes,
    }


def _auth_bundle_to_dict(bundle) -> dict:
    """Deterministic, JSON-safe view of an AuthBundle."""
    return {
        "level": bundle.level.value,
        "classical_sig": _bytes_to_str(bundle.classical_sig),
        "pqc_sig": _bytes_to_str(bundle.pqc_sig),
        "mask": bundle.mask.value,
        "curve": bundle.curve,
        "dsa": bundle.dsa,
        "ecdsa_pk_bytes": _bytes_to_str(bundle.ecdsa_pk_bytes),
        "mldsa_pk_bytes": _bytes_to_str(bundle.mldsa_pk_bytes),
    }


def _make_grease_groups() -> list[str]:
    """
    Make 2 stable dummy hybrid group identifiers.
    This gets transcript-bound, so stripping them changes the hash.
    """
    seed = os.getenv("HYBRID_TLS_GREASE_SEED", "hybrid-pqc-grease").encode()
    h = hashlib.sha256(seed).digest()
    g1 = f"GREASE-HYB-{h[0]:02x}{h[1]:02x}"
    g2 = f"GREASE-HYB-{h[2]:02x}{h[3]:02x}"
    return [g1, g2]


def _make_pqc_attestation(level: NISTSecurityLevel, mode: str) -> dict:
    """
    Compact description of what the server claims it used.
    Bound via transcript; later you can sign it separately.
    """
    if level == NISTSecurityLevel.LEVEL_1:
        kem, sig = "ML-KEM-512", "ML-DSA-44"
    elif level == NISTSecurityLevel.LEVEL_3:
        kem, sig = "ML-KEM-768", "ML-DSA-65"
    else: # L5
        kem, sig = "ML-KEM-1024", "ML-DSA-87"
        
    return {
        "version": 1,
        "level": int(level.value),
        "mode": mode,
        "kem": kem,
        "sig": sig,
        "pqc_runtime": pqc_runtime_status(),
    }


# ============================================================================
# Main session class
# ============================================================================

class FullHybridSession:
    def __init__(self, role: Role, policy: CryptoPolicy,
                 caps: AlgorithmCapabilities, engine: PolicyEngine):
        self.role = role
        self.policy = policy
        self.caps = caps
        self.engine = engine

        self.kex: Optional[NISTHybridKeyExchange] = None
        self.auth: Optional[NISTHybridAuthentication] = None
        self.traffic_keys = None

        self.transcript = hashlib.sha256()
        self.seq_num = 0         # write counter
        self.read_seq_num = 0    # read counter

        self.handshake_start = 0.0
        self.handshake_end = 0.0

    # ------------------------------------------------------------
    # transcript updater
    # ------------------------------------------------------------
    def _update_transcript(self, obj: dict):
        blob = json.dumps(obj, sort_keys=True).encode()
        self.transcript.update(blob)

    # ============================================================
    # 1) CLIENT: Flight 1
    # ============================================================
    def connect(self, server_caps_hint: AlgorithmCapabilities) -> Dict[str, Any]:
        assert self.role == Role.CLIENT
        self.handshake_start = time.perf_counter()

        # tentative local negotiation
        self.kex = NISTHybridKeyExchange.from_negotiation(
            self.engine, self.caps, server_caps_hint, self.policy, Role.CLIENT
        )

        client_hello = {
            "msg_type": "client_hello",
            "caps": _caps_to_dict(self.caps),
            "policy": self.policy.value,
            "proposed_level": self.kex.level.value,
        }
        self._update_transcript(client_hello)
        return client_hello

    # ============================================================
    # 2) SERVER: Flight 2
    # ============================================================
    def accept(self, client_hello: Dict[str, Any]) -> Dict[str, Any]:
        assert self.role == Role.SERVER
        self.handshake_start = time.perf_counter()

        self._update_transcript(client_hello)

        client_caps_dict = client_hello["caps"]
        client_caps = AlgorithmCapabilities(
            supported_levels={NISTSecurityLevel(lv) for lv in client_caps_dict["supported_levels"]},
            supported_classical=set(client_caps_dict["supported_classical"]),
            supported_pqc=set(client_caps_dict["supported_pqc"]),
            supports_hybrid=client_caps_dict["supports_hybrid"],
            max_latency_ms=client_caps_dict.get("max_latency_ms"),
            max_wire_bytes=client_caps_dict.get("max_wire_bytes"),
        )

        # authoritative negotiation
        self.kex = NISTHybridKeyExchange.from_negotiation(
            self.engine, client_caps, self.caps, self.policy, Role.SERVER
        )
        self.auth = NISTHybridAuthentication(
            self.engine, self.kex.level, self.policy, Role.SERVER
        )

        srv_ec, srv_pq = self.kex.server_keygen()
        grease = _make_grease_groups()
        pqc_att = _make_pqc_attestation(self.kex.level, self.kex.mode.value)

        server_hello = {
            "msg_type": "server_hello",
            "srv_ec": _bytes_to_str(srv_ec),
            "srv_pq": _bytes_to_str(srv_pq),
            "level": self.kex.level.value,
            "mode": self.kex.mode.value,
            "policy": self.policy.value,
            "grease_groups": grease,
            "pqc_attestation": pqc_att,
        }
        self._update_transcript(server_hello)
        return server_hello

    # ============================================================
    # 3) CLIENT: Flight 3
    # ============================================================
    def client_finish(self, server_hello: Dict[str, Any]) -> Dict[str, Any]:
        assert self.role == Role.CLIENT

        # hash exactly what server sent (with GREASE + attestation)
        self._update_transcript(server_hello)

        server_level = NISTSecurityLevel(server_hello["level"])
        server_mode = HybridMode(server_hello["mode"])
        if self.kex.level != server_level or self.kex.mode != server_mode:
            # re-align to server choice
            self.kex = NISTHybridKeyExchange(
                server_level, server_mode, Role.CLIENT, self.policy
            )

        # client must be ready to verify server's final auth
        self.auth = NISTHybridAuthentication(
            self.engine, self.kex.level, self.policy, Role.CLIENT
        )

        ec_pub, pq_ct, mask, loser_commit = self.kex.client_encaps(
            _str_to_bytes(server_hello["srv_ec"]),
            _str_to_bytes(server_hello["srv_pq"]),
        )

        client_kex = {
            "msg_type": "client_kex",
            "ec_pub": _bytes_to_str(ec_pub),
            "pq_ct": _bytes_to_str(pq_ct),
            "mask": mask.value,
            "lc": _bytes_to_str(loser_commit),
        }
        self._update_transcript(client_kex)

        # derive client traffic keys
        th = self.transcript.digest()
        kex_res = self.kex.finalize_secret(th)
        ck, sk = derive_tls13_keys(kex_res.shared_secret, th)
        # client: (write, read)
        self.traffic_keys = (ck, sk)

        return client_kex

    # ============================================================
    # 4) SERVER: Flight 4 (auth)
    # ============================================================
    def server_finish(self, client_kex: Dict[str, Any]) -> Dict[str, Any]:
        assert self.role == Role.SERVER

        self._update_transcript(client_kex)

        self.kex.server_decaps(
            _str_to_bytes(client_kex["ec_pub"]),
            _str_to_bytes(client_kex["pq_ct"]),
            client_kex["mask"],
            _str_to_bytes(client_kex["lc"]),
        )

        th = self.transcript.digest()

        kex_res = self.kex.finalize_secret(th)
        ck, sk = derive_tls13_keys(kex_res.shared_secret, th)
        # server: (write, read)
        self.traffic_keys = (sk, ck)

        _ = self.auth.server_keygen()
        sig_bundle = self.auth.server_sign(th)
        self.handshake_end = time.perf_counter()

        server_auth = {
            "msg_type": "server_auth",
            "auth_bundle": _auth_bundle_to_dict(sig_bundle),
        }
        return server_auth

    # ============================================================
    # 5) CLIENT: verify server auth
    # ============================================================
    def client_verify_server_auth(self, server_auth: Dict[str, Any]) -> bool:
        assert self.role == Role.CLIENT
        assert self.auth is not None, "Call client_finish() first."

        transcript_hash = self.transcript.digest()

        bundle_dict = server_auth["auth_bundle"]

        # reconstruct AuthBundle
        auth_bundle = AuthBundle(
            level=NISTSecurityLevel(bundle_dict["level"]),
            classical_sig=_str_to_bytes(bundle_dict["classical_sig"]),
            pqc_sig=_str_to_bytes(bundle_dict["pqc_sig"]),
            mask=PathMask(bundle_dict["mask"]),
            curve=bundle_dict["curve"],
            dsa=bundle_dict["dsa"],
            ecdsa_pk_bytes=_str_to_bytes(bundle_dict["ecdsa_pk_bytes"]),
            mldsa_pk_bytes=_str_to_bytes(bundle_dict["mldsa_pk_bytes"]),
            not_before=0.0,
            not_after=time.time() * 2,
        )

        ok = self.auth.client_verify(auth_bundle, transcript_hash)
        if not ok:
            raise RuntimeError("CLIENT: server authentication failed")
        self.handshake_end = time.time()
        return True

    # ============================================================
    # 6) Record layer
    # ============================================================
    def send(self, data: bytes) -> bytes:
        assert self.traffic_keys, "Handshake not complete."
        ct = tls13_encrypt(self.traffic_keys[0], self.seq_num, data)
        self.seq_num += 1
        return ct

    def recv(self, ciphertext: bytes) -> bytes:
        assert self.traffic_keys, "Handshake not complete."
        pt = tls13_decrypt(self.traffic_keys[1], self.read_seq_num, ciphertext)
        self.read_seq_num += 1
        return pt

    # ============================================================
    # 7) *** NEW *** Evaluation Harness
    # ============================================================
    @staticmethod
    def simulate_handshake(
        cli: 'FullHybridSession', srv: 'FullHybridSession',
        rtt_ms: float = 0.0
    ) -> Dict[str, Any]:
        """
        Orchestrates a full handshake between two session objects in-memory,
        simulates network RTT, and returns rich telemetry.
        
        This is a 2-RTT model:
        1. ClientHello -> ServerHello (1 RTT)
        2. ClientKEX -> ServerAuth (1 RTT)
        """
        t_start = time.perf_counter()
        half_rtt_s = (rtt_ms / 1000.0) / 2.0

        # --- Flight 1: ClientHello ---
        # (cli.connect() is ~instant, no RTT)
        client_hello = cli.connect(srv.caps) # Use server's real caps as hint
        
        time.sleep(half_rtt_s) # Client -> Server

        # --- Flight 2: ServerHello ---
        server_hello = srv.accept(client_hello)
        
        time.sleep(half_rtt_s) # Server -> Client
        
        # --- Flight 3: ClientKEX ---
        client_kex = cli.client_finish(server_hello)
        
        time.sleep(half_rtt_s) # Client -> Server
        
        # --- Flight 4: ServerAuth ---
        server_auth_flight = srv.server_finish(client_kex)

        time.sleep(half_rtt_s) # Server -> Client

        # --- Flight 5: Client Verifies (no network cost) ---
        cli.client_verify_server_auth(server_auth_flight)
        
        t_end = time.perf_counter()

        # --- Aggregate and return rich telemetry ---
        cli_kex_timings = cli.kex.timing_data or {}
        srv_kex_timings = srv.kex.timing_data or {}
        cli_auth_timings = cli.auth.telemetry.get("timings_ms", {})
        srv_auth_timings = srv.auth.telemetry.get("timings_ms", {})
        pqc_status = server_hello.get("pqc_attestation", {}).get("pqc_runtime", pqc_runtime_status())

        return {
            "total_time_ms": (t_end - t_start) * 1000.0,
            "level": srv.kex.level,
            "mode": srv.kex.mode,
            "commit_mask": srv.kex.commit_mask,
            
            # KEX Timings
            "kex_encaps_ms": cli_kex_timings.get("encaps_ms"),
            "kex_hkdf_c_ms": cli_kex_timings.get("hkdf_ms"),
            "kex_keygen_ms": srv_kex_timings.get("keygen_ms"),
            "kex_decaps_ms": srv_kex_timings.get("decaps_ms"),
            "kex_hkdf_s_ms": srv_kex_timings.get("hkdf_ms"),

            # Auth Timings
            "auth_sign_c_ms": srv_auth_timings.get("sign_c"),
            "auth_sign_q_ms": srv_auth_timings.get("sign_q"),
            "auth_ver_c_ms": cli_auth_timings.get("ver_c"),
            "auth_ver_q_ms": cli_auth_timings.get("ver_q"),

            # Wire Sizes
            "wire_srv_ecdh": srv.kex.wire_sizes.get("srv_ecdh"),
            "wire_srv_pqc": srv.kex.wire_sizes.get("srv_pqc"),
            "wire_cli_ecdh": cli.kex.wire_sizes.get("cli_ecdh"), # From client's perspective
            "wire_cli_pqc": cli.kex.wire_sizes.get("cli_pqc"),

            # PQC Status
            "using_mock_pqc": pqc_status["using_mock_pqc"],
            "strict_pqc": pqc_status["strict_pqc"],
        }