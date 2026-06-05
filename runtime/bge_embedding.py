"""bge_embedding.py — wrapper minimale per bge-m3 ONNX dense embeddings.

API simmetrica a `suprastructure.embedding.onnx_embedding.EmbeddingService`:
  emb = BGEEmbeddingService(model_dir)
  vec = emb.embed_query("testo")      # (1024,) L2-normalized
  mat = emb.embed_texts(["a", "b"])    # (N, 1024)

Note implementative:
- Modello: Xenova/bge-m3 onnx/sentence_transformers_fp16.onnx
- Tokenizer: HF Rust tokenizers (tokenizer.json)
- Output: ultimo hidden state → mean pooling masked → L2 normalize
- Max length: 8192 token (padding/truncation a sliding window inferiore
  per query brevi: usa cap 256 di default, configurabile).

Solo dense (no sparse output gestito qui — useremmo `sentence_transformers.onnx`
per ottenere il vettore dense L2 normalizzato che e' il format standard).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer


class BGEEmbeddingService:
    """bge-m3 dense embeddings via ONNX (fp16 default)."""

    def __init__(self, model_dir: str | None = None,
                 model_file: str = "onnx/sentence_transformers_int8.onnx",
                 max_length: int = 256):
        # ADR 0148 rename-resilient: default derived from install root.
        if model_dir is None:
            import config as _C
            model_dir = str(_C.PATH_ROOT / "models" / "embedding-bge")
        self._dir = Path(model_dir)
        onnx_path = self._dir / model_file
        tok_path = self._dir / "tokenizer.json"
        if not onnx_path.exists():
            raise FileNotFoundError(f"BGE ONNX non trovato: {onnx_path}")
        if not tok_path.exists():
            raise FileNotFoundError(f"BGE tokenizer non trovato: {tok_path}")

        # ORT session (cpu provider su Strix Halo basta).
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._sess = ort.InferenceSession(
            str(onnx_path), sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        # Tokenizer HF Rust.
        self._tok = Tokenizer.from_file(str(tok_path))
        self._tok.enable_padding(pad_id=1, pad_token="<pad>")
        self._tok.enable_truncation(max_length=max_length)

        # Sanity: introspect input/output shapes.
        self._input_names = [i.name for i in self._sess.get_inputs()]
        self._output_names = [o.name for o in self._sess.get_outputs()]

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Ritorna (N, 1024) L2-normalized."""
        if not texts:
            return np.zeros((0, 1024), dtype=np.float32)
        encodings = self._tok.encode_batch(list(texts))
        ids = np.array([e.ids for e in encodings], dtype=np.int64)
        mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
        feed = {"input_ids": ids, "attention_mask": mask}
        # Alcune varianti del modello richiedono token_type_ids; fornisci se serve.
        if "token_type_ids" in self._input_names:
            feed["token_type_ids"] = np.zeros_like(ids)
        out = self._sess.run(None, feed)
        # Output: per sentence_transformers.onnx in genere e' GIA' il vettore
        # dense L2-normalizzato (1024,). Verifichiamo la shape.
        first = out[0]
        if first.ndim == 3:
            # ultimo hidden state (B, T, H) → mean pool con mask + L2 norm
            mask_f = mask.astype(np.float32)[..., None]
            summed = (first * mask_f).sum(axis=1)
            counts = mask_f.sum(axis=1).clip(min=1e-9)
            vec = summed / counts
        else:
            # gia' (B, H)
            vec = first
        # L2 normalize (cosine = dot)
        norms = np.linalg.norm(vec, axis=1, keepdims=True).clip(min=1e-9)
        return (vec / norms).astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_texts([text])[0]


if __name__ == "__main__":
    emb = BGEEmbeddingService()
    print("inputs:", emb._input_names, "outputs:", emb._output_names)
    v = emb.embed_texts(["organico personale scolastico", "che ore sono",
                          "PTOF documento ufficiale"])
    print("shape:", v.shape, "norm[0]:", np.linalg.norm(v[0]))
    q = emb.embed_query("trova organico di diritto")
    sims = v @ q
    for label, s in zip(["organico", "ore", "PTOF"], sims):
        print(f"  {label:10s} cosine={s:+.3f}")
