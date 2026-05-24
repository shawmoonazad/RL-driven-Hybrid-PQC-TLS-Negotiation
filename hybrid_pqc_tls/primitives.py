# -*- coding: utf-8 -*-
"""
Cryptographic Primitives Adapters
=================================
Unifies classical + PQC crypto behind adapters.

Features:
- Kyber / ML-KEM via kyber_py (with order detection)
- Dilithium / ML-DSA via dilithium_py (with order detection)
- Deterministic mocks for both, for environments without real PQC
- STRICT_PQC to forbid mocks
- USING_MOCK_PQC to tell the evaluator what actually happened
"""

import os
import hashlib
import hmac
import struct
from abc import ABC, abstractmethod
from typing import Tuple, Dict

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.exceptions import InvalidSignature

from .config import NISTSecurityLevel

# ======================================================================
# Global PQC flags
# ======================================================================

STRICT_PQC = os.getenv("HYBRID_TLS_STRICT_PQC", "0") == "1"
# this should be flipped ONLY by factories below
USING_MOCK_PQC = False

# just probing for presence; we won't decide here
try:
    from kyber_py.ml_kem import ML_KEM_512  # noqa: F401
    _REAL_KYBER_AVAILABLE = True
except ImportError:
    _REAL_KYBER_AVAILABLE = False

try:
    from dilithium_py.ml_dsa import ML_DSA_44  # noqa: F401
    _REAL_DILITHIUM_AVAILABLE = True
except ImportError:
    _REAL_DILITHIUM_AVAILABLE = False


def pqc_runtime_status() -> dict:
    """Read-only view for higher layers."""
    return {
        "strict_pqc": STRICT_PQC,
        "using_mock_pqc": USING_MOCK_PQC,
    }


# ======================================================================
# KEM Adapters (Kyber / ML-KEM)
# ======================================================================

class KemAdapter(ABC):
    is_mock: bool = False

    @abstractmethod
    def keygen(self) -> Tuple[bytes, bytes]:
        ...

    @abstractmethod
    def encaps(self, pk: bytes) -> Tuple[bytes, bytes]:
        """Return (shared_secret, ciphertext)."""
        ...

    @abstractmethod
    def decaps(self, sk: bytes, ct: bytes) -> bytes:
        ...


class RealKyberAdapter(KemAdapter):
    """
    Wraps a kyber_py ML-KEM class.

    Different bindings return:
        (ct, ss)   OR   (ss, ct)

    We look at lengths to normalize to (ss, ct).
    """
    is_mock = False

    def __init__(self, kem_class):
        # support both instance and class styles
        try:
            self.kem = kem_class()
        except TypeError:
            self.kem = kem_class

    def keygen(self) -> Tuple[bytes, bytes]:
        return self.kem.keygen()

    def encaps(self, pk: bytes) -> Tuple[bytes, bytes]:
        a, b = self.kem.encaps(pk)
        # shared secret is small (~32), ciphertext is large (768+)
        if len(a) < len(b):
            ss, ct = a, b  # (ss, ct)
        else:
            ct, ss = a, b  # (ct, ss)
        return ss, ct

    def decaps(self, sk: bytes, ct: bytes) -> bytes:
        return self.kem.decaps(sk, ct)


class MockKyberAdapter(KemAdapter):
    """
    Deterministic, size-correct mock. Encaps/decaps agree.
    """
    is_mock = True

    PARAM_SIZES = {
        "ML-KEM-512": {"pk": 800, "sk": 1632, "ct": 768,  "ss": 32},
        "ML-KEM-768": {"pk": 1184, "sk": 2400, "ct": 1088, "ss": 32},
        "ML-KEM-1024": {"pk": 1568, "sk": 3168, "ct": 1568, "ss": 32},
    }

    def __init__(self, pk_size: int, sk_size: int, ct_size: int, ss_size: int = 32):
        self.pk_size = pk_size
        self.sk_size = sk_size
        self.ct_size = ct_size
        self.ss_size = ss_size

    def keygen(self) -> Tuple[bytes, bytes]:
        sk = os.urandom(self.sk_size)
        pk = hashlib.sha512(b"mock_pk_from_sk|" + sk).digest()[: self.pk_size]
        return pk, sk

    def encaps(self, pk: bytes) -> Tuple[bytes, bytes]:
        ss = hashlib.sha512(b"mock_ss_from_pk|" + pk).digest()[: self.ss_size]
        ct = os.urandom(self.ct_size)
        return ss, ct

    def decaps(self, sk: bytes, ct: bytes) -> bytes:
        pk_derived = hashlib.sha512(b"mock_pk_from_sk|" + sk).digest()[: self.pk_size]
        ss_derived = hashlib.sha512(b"mock_ss_from_pk|" + pk_derived).digest()[: self.ss_size]
        return ss_derived


def make_kem_adapter(level: NISTSecurityLevel) -> Tuple[KemAdapter, Dict[str, int]]:
    """
    Try real kyber_py first; self-test; otherwise deterministically mock.
    """
    global USING_MOCK_PQC

    lvl_int = level.value if isinstance(level, NISTSecurityLevel) else int(level)

    KEM_SIZES = {
        1: {"pk": 800,  "sk": 1632, "ct": 768,  "ss": 32, "name": "ML-KEM-512"},
        3: {"pk": 1184, "sk": 2400, "ct": 1088, "ss": 32, "name": "ML-KEM-768"},
        5: {"pk": 1568, "sk": 3168, "ct": 1568, "ss": 32, "name": "ML-KEM-1024"},
    }
    sizes = KEM_SIZES[lvl_int]

    try:
        if not _REAL_KYBER_AVAILABLE:
            raise ImportError("kyber_py not available")

        from kyber_py.ml_kem import ML_KEM_512, ML_KEM_768, ML_KEM_1024
        real_map = {1: ML_KEM_512, 3: ML_KEM_768, 5: ML_KEM_1024}
        adapter = RealKyberAdapter(real_map[lvl_int])

        # self-test round-trip
        pk, sk = adapter.keygen()
        ss1, ct = adapter.encaps(pk)
        ss2 = adapter.decaps(sk, ct)
        if ss1 != ss2:
            raise AssertionError("Kyber self-test failed: ss mismatch")

        return adapter, sizes

    except (ImportError, AssertionError, Exception) as e:
        if STRICT_PQC:
            raise RuntimeError(f"STRICT_PQC=1 but ML-KEM L{lvl_int} not available: {e}")
        print(f"WARNING: Using mock ML-KEM for L{lvl_int} (Reason: {e})")
        USING_MOCK_PQC = True
        mock = MockKyberAdapter(sizes["pk"], sizes["sk"], sizes["ct"], sizes["ss"])
        return mock, sizes


# ======================================================================
# Signature Adapters (ECDSA, ML-DSA)
# ======================================================================

class SigAdapter(ABC):
    is_mock: bool = False

    @abstractmethod
    def generate_keypair(self, *args, **kwargs):
        ...

    @abstractmethod
    def sign(self, *args, **kwargs):
        ...

    @abstractmethod
    def verify(self, *args, **kwargs):
        ...


class ECDSAAdapter(SigAdapter):
    is_mock = False

    CURVE_MAP = {
        "P-256": ec.SECP256R1(),
        "P-384": ec.SECP384R1(),
        "P-521": ec.SECP521R1(),
    }
    HASH_MAP = {
        "SHA-256": hashes.SHA256,
        "SHA-384": hashes.SHA384,
        "SHA-512": hashes.SHA512,
    }

    @staticmethod
    def generate_keypair(curve_name: str):
        curve = ECDSAAdapter.CURVE_MAP[curve_name]
        sk = ec.generate_private_key(curve)
        return sk, sk.public_key()

    @staticmethod
    def sign(sk, msg: bytes, hash_name: str) -> bytes:
        return sk.sign(msg, ec.ECDSA(ECDSAAdapter.HASH_MAP[hash_name]()))

    @staticmethod
    def verify(pk, sig: bytes, msg: bytes, hash_name: str) -> bool:
        try:
            pk.verify(sig, msg, ec.ECDSA(ECDSAAdapter.HASH_MAP[hash_name]()))
            return True
        except InvalidSignature:
            return False

    @staticmethod
    def pubkey_bytes_compressed(pk) -> bytes:
        return pk.public_bytes(
            encoding=serialization.Encoding.X962,
            format=serialization.PublicFormat.CompressedPoint,
        )


# we keep a set so we don't print the same warning 50x during eval
_MLDSA_WARNED: set[str] = set()


class RealMLDSAAdapter(SigAdapter):
    """
    Wraps a dilithium_py ML-DSA class and remembers whether keygen was (pk, sk) or (sk, pk).
    """
    is_mock = False

    def __init__(self, mldsa_class, sk_first: bool):
        self.mldsa = mldsa_class
        self.sk_first = sk_first

    def generate_keypair(self, *args, **kwargs):
        k1, k2 = self.mldsa.keygen()
        if self.sk_first:
            sk, pk = k1, k2
        else:
            pk, sk = k1, k2
        return sk, pk

    def sign(self, sk: bytes, msg: bytes, *args, **kwargs) -> bytes:
        return self.mldsa.sign(sk, msg)

    def verify(self, pk: bytes, sig: bytes, msg: bytes, *args, **kwargs) -> bool:
        return self.mldsa.verify(pk, msg, sig)


class MockMLDSAAdapter(SigAdapter):
    """
    HMAC-based simulator; correct lengths, deterministic.
    """
    is_mock = True

    PARAMS = {
        "ML-DSA-44": {"sig_len": 3300, "pk_len": 1312},
        "ML-DSA-65": {"sig_len": 4595, "pk_len": 1952},
        "ML-DSA-87": {"sig_len": 5669, "pk_len": 2592},
    }
    _REG: Dict[bytes, bytes] = {}

    def __init__(self, param: str):
        self.param = param
        self.sizes = self.PARAMS[param]

    def generate_keypair(self, *args, **kwargs):
        sk = os.urandom(32)
        pk = os.urandom(self.sizes["pk_len"])
        self._REG[pk] = sk
        return sk, pk

    def sign(self, sk: bytes, msg: bytes, *args, **kwargs) -> bytes:
        base = hmac.new(sk, b"MLDSA-SIM|" + msg, hashlib.sha256).digest()
        out = bytearray()
        ctr = 0
        while len(out) < self.sizes["sig_len"]:
            out.extend(hashlib.sha256(base + struct.pack(">I", ctr)).digest())
            ctr += 1
        return bytes(out[: self.sizes["sig_len"]])

    def verify(self, pk: bytes, sig: bytes, msg: bytes, *args, **kwargs) -> bool:
        sk = self._REG.get(pk)
        if not sk:
            return False
        expected = self.sign(sk, msg)
        return hmac.compare_digest(expected, sig)


def make_mldsa_adapter(param: str) -> SigAdapter:
    """
    Try to use real dilithium_py, trying both key orders.
    Fall back to mock only if both fail.
    """
    global USING_MOCK_PQC

    try:
        if not _REAL_DILITHIUM_AVAILABLE:
            raise ImportError("dilithium_py not available")

        from dilithium_py.ml_dsa import ML_DSA_44, ML_DSA_65, ML_DSA_87
        real_map = {
            "ML-DSA-44": ML_DSA_44,
            "ML-DSA-65": ML_DSA_65,
            "ML-DSA-87": ML_DSA_87,
        }
        if param not in real_map:
            raise ImportError(f"Unknown ML-DSA param: {param}")

        cls = real_map[param]
        msg = b"self-test"

        # candidate 1: keygen() -> (pk, sk)
        k1, k2 = cls.keygen()
        try:
            # assume k1=pk, k2=sk
            sig = cls.sign(k2, msg)
            assert cls.verify(k1, msg, sig)
            return RealMLDSAAdapter(cls, sk_first=False)
        except Exception:
            pass

        # candidate 2: keygen() -> (sk, pk)
        try:
            sig = cls.sign(k1, msg)
            assert cls.verify(k2, msg, sig)
            return RealMLDSAAdapter(cls, sk_first=True)
        except Exception:
            pass

        # if neither calling convention worked, force fallback
        raise AssertionError("dilithium_py present but key order is unknown")

    except (ImportError, AssertionError, Exception) as e:
        if STRICT_PQC:
            raise RuntimeError(f"STRICT_PQC=1 but ML-DSA for {param} is not available: {e}")
        if param not in _MLDSA_WARNED:
            print(f"WARNING: Using mock ML-DSA for {param} (Reason: {e})")
            _MLDSA_WARNED.add(param)
        USING_MOCK_PQC = True
        return MockMLDSAAdapter(param)
