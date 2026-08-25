"""Query and claim embeddings, stored int8 in the same table as everything else.

Two backends, chosen by MUFASA_EMBED_BACKEND:

``hashing``  deterministic character-n-gram hashing. No model file, no download,
             identical on every machine, so CI and the test suite can exercise
             the whole dense path. This is the default.
``onnx``     a real sentence encoder (MiniLM-class, 384-d) run through
             onnxruntime from a local model directory. Used for the shipped
             package; set MUFASA_EMBED_MODEL_DIR to its folder.

Vectors live in ``claim_vec`` keyed by claim_id, so a dense hit *is* a claim row
and there is no second id space to keep in sync — the one property that made a
graph database attractive in the first place, kept without one.

int8 rather than float32: 180k x 384 float32 is 276 MB resident, the same
vectors int8 are 69 MB, and a full scan of 69 MB is a few milliseconds. The
memory ceiling is the scored metric; brute force over int8 is the cheap way to
respect it.
"""

from __future__ import annotations

import hashlib
import os
import struct
from typing import Protocol

import numpy as np

DIM = 384


class Embedder(Protocol):
    dim: int
    name: str

    def encode(self, texts: list[str]) -> np.ndarray: ...


class HashingEmbedder:
    """Character-n-gram hashing with sign-and-bucket projection."""

    name = "hashing-v1"

    def __init__(self, dim: int = DIM, ngram: int = 4) -> None:
        self.dim = dim
        self.ngram = ngram

    def _one(self, text: str) -> np.ndarray:
        from .normalize import norm

        vec = np.zeros(self.dim, dtype=np.float32)
        t = f" {norm(text)} "
        if len(t.strip()) == 0:
            return vec
        grams = [t[i : i + self.ngram] for i in range(max(1, len(t) - self.ngram + 1))]
        grams += [w for w in t.split() if w]
        for g in grams:
            h = hashlib.blake2b(g.encode("utf-8"), digest_size=8).digest()
            (idx,) = struct.unpack("<Q", h)
            sign = 1.0 if (idx >> 63) & 1 else -1.0
            vec[idx % self.dim] += sign
        n = float(np.linalg.norm(vec))
        return vec / n if n else vec

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack([self._one(t) for t in texts])


class OnnxEmbedder:
    """Local ONNX sentence encoder. Never downloads; the directory must exist."""

    name = "onnx"

    def __init__(self, model_dir: str, dim: int = DIM) -> None:
        import onnxruntime  # noqa: PLC0415  (optional dependency)
        from tokenizers import Tokenizer  # noqa: PLC0415

        self.dim = dim
        self.model_dir = model_dir
        self.name = f"onnx:{os.path.basename(model_dir.rstrip('/\\'))}"
        self._tok = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
        self._tok.enable_truncation(max_length=256)
        self._tok.enable_padding()
        self._sess = onnxruntime.InferenceSession(
            os.path.join(model_dir, "model.onnx"),
            providers=["CPUExecutionProvider"],
            sess_options=self._session_options(onnxruntime),
        )

    @staticmethod
    def _session_options(onnxruntime):
        opts = onnxruntime.SessionOptions()
        # Same reason as the llama.cpp thread cap: a laptop running flat out
        # throttles, and a thermal trip costs 10 marks.
        opts.intra_op_num_threads = int(os.getenv("MUFASA_EMBED_THREADS", "2"))
        return opts

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        enc = self._tok.encode_batch(texts)
        ids = np.array([e.ids for e in enc], dtype=np.int64)
        mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
        feeds = {"input_ids": ids, "attention_mask": mask}
        names = {i.name for i in self._sess.get_inputs()}
        if "token_type_ids" in names:
            feeds["token_type_ids"] = np.zeros_like(ids)
        out = self._sess.run(None, {k: v for k, v in feeds.items() if k in names})[0]
        m = mask[..., None].astype(np.float32)
        pooled = (out * m).sum(axis=1) / np.clip(m.sum(axis=1), 1e-9, None)
        n = np.linalg.norm(pooled, axis=1, keepdims=True)
        return (pooled / np.clip(n, 1e-9, None)).astype(np.float32)


def get_embedder(backend: str | None = None, model_dir: str | None = None) -> Embedder:
    backend = (backend or os.getenv("MUFASA_EMBED_BACKEND") or "hashing").lower()
    if backend == "hashing":
        return HashingEmbedder()
    if backend == "onnx":
        model_dir = model_dir or os.getenv("MUFASA_EMBED_MODEL_DIR")
        if not model_dir or not os.path.isdir(model_dir):
            raise FileNotFoundError(
                f"MUFASA_EMBED_BACKEND=onnx needs MUFASA_EMBED_MODEL_DIR to point at a "
                f"local model directory (got {model_dir!r}). Nothing is downloaded at runtime."
            )
        return OnnxEmbedder(model_dir)
    raise ValueError(f"unknown embedding backend {backend!r}; use 'hashing' or 'onnx'")


# -- int8 storage ----------------------------------------------------------


def quantize(vecs: np.ndarray) -> tuple[np.ndarray, float]:
    """Symmetric int8 quantisation of unit-norm rows. Returns (int8, scale)."""
    if vecs.size == 0:
        return vecs.astype(np.int8), 1.0
    scale = float(np.abs(vecs).max()) or 1.0
    q = np.clip(np.rint(vecs / scale * 127.0), -127, 127).astype(np.int8)
    return q, scale


def dequantize(q: np.ndarray, scale: float) -> np.ndarray:
    return q.astype(np.float32) * (scale / 127.0)
