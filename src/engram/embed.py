"""Service-less bge-m3 embedder: onnxruntime on the locally-cached ONNX export.

Decision #1: local ONNX bge-m3, no service, no torch. The ONNX graph already bakes in CLS pooling
(output `sentence_embedding`), so we only tokenize → run → L2-normalize. Model is resolved from the
HF cache with `local_files_only` — engram never triggers a network download.
"""
from __future__ import annotations

import os

import numpy as np

_MAX_LEN = 512  # description + body-head fit easily; keeps CPU inference fast
_BATCH = 32  # chunk encode() so a few-hundred-note index doesn't build one giant padded batch (OOM)


def _resolve() -> tuple[str, str]:
    """Return (onnx_path, tokenizer_path) from the local HF cache, or raise if absent."""
    from huggingface_hub import hf_hub_download

    repo = os.environ.get("ENGRAM_EMBED_REPO", "BAAI/bge-m3")  # read at use, not frozen at import
    onnx = hf_hub_download(repo, "onnx/model.onnx", local_files_only=True)
    hf_hub_download(repo, "onnx/model.onnx_data", local_files_only=True)  # external weights, co-located
    tok = hf_hub_download(repo, "tokenizer.json", local_files_only=True)
    return onnx, tok


def model_available() -> bool:
    try:
        _resolve()
        return True
    except Exception:
        return False


class Embedder:
    """Loaded once per process (heavy). `Embedder()` returns the shared instance."""

    _instance: Embedder | None = None

    def __new__(cls) -> Embedder:
        if cls._instance is None:
            inst = super().__new__(cls)
            inst._load()
            cls._instance = inst
        return cls._instance

    def _load(self) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        onnx_path, tok_path = _resolve()
        self._tok = Tokenizer.from_file(tok_path)
        self._tok.enable_truncation(max_length=_MAX_LEN)
        self._tok.enable_padding()
        self._sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    def encode(self, texts) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.zeros((0, 1024), dtype=np.float32)
        rows = []
        for i in range(0, len(texts), _BATCH):  # chunk: cap padded-batch activations, not one huge matmul
            encs = self._tok.encode_batch(texts[i:i + _BATCH])
            ids = np.array([e.ids for e in encs], dtype=np.int64)
            mask = np.array([e.attention_mask for e in encs], dtype=np.int64)
            rows.append(self._sess.run(["sentence_embedding"], {"input_ids": ids, "attention_mask": mask})[0])
        out = np.concatenate(rows, axis=0)
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return (out / np.clip(norms, 1e-12, None)).astype(np.float32)
