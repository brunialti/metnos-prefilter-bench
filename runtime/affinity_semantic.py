"""affinity_semantic.py — semantic fallback per affinity match (ADR 15/5/2026).

Hard match (bag-of-words su tag affinity) resta il path primario, veloce
e affidabile per query "ben formate" (95% dei casi). Quando il top-1
hard score e' sotto la soglia minima (query con typo, sinonimi semantici,
cross-lingua, declinazioni irregolari), questo modulo fa fallback su
BGE-M3 embedding + cosine similarity e re-rank i candidati.

Architettura: fast-path
  - normal: hard match → 0 overhead BGE
  - fallback: hard max < soglia → encode query (~15ms) + cosine (~10ms)

Cache:
  - I tag affinity di tutti gli executor sono encodati una volta e
    persistiti su disco. La chiave cache e' sha256 sui (executor, tag)
    ordinati: ricalcolo automatico quando il catalogo cambia.
  - Storage: ~/.cache/metnos/affinity_emb/<key>.npz
  - Re-build lazy al primo uso (~500ms una tantum).

Determinismo §7.9: niente LLM, sola inferenza ONNX deterministica.
"""
from __future__ import annotations

import hashlib
import logging
import os
import threading
from pathlib import Path
from typing import Any, Iterable

import numpy as np


_LOG = logging.getLogger(__name__)

CACHE_DIR = Path.home() / ".cache" / "metnos" / "affinity_emb"

# Soglia hard score sotto la quale attivare il fallback semantico.
# Calibrato via bench_semantic_tuning.py (15/5/2026, corpus 301 query reali):
# - threshold=4 attivazione 1.3% top1 invariato (effetto marginale)
# - threshold=6 attivazione 10.3% top3 +1.0% top5 +0.7%
# - threshold=8 attivazione 14.0% top1 +1.0% top3 +1.0% top5 +0.7% (best)
# - threshold=10 plateau (stesso recall ma 21% activation, latency sprecata)
# Best Pareto = 8: copre query con 1 hard match debole dove la semantica
# puo' aggiungere valore (es. "Sutuazione server" con solo "server" hard).
SEMANTIC_THRESHOLD_DEFAULT = 8

# Peso del semantic score (max cosine 0..1) sommato all'hard score.
# Top cosine BGE per query ben mirate ~0.85; bonus tipico 2-3 punti.
# Bench 15/5 ha mostrato che alpha ∈ [2..6] e' equivalente sul recall
# (la differenziazione avviene nella SELEZIONE post-rerank, non nel
# valore puntuale). Default conservativo a 4.
SEMANTIC_ALPHA_DEFAULT = 4.0


# Singleton lazy embedder. Threadsafe init via lock.
_EMB_LOCK = threading.Lock()
_EMB: Any = None


def _get_embedder():
    """Lazy load BGEEmbeddingService. None se non disponibile (degrade silent)."""
    global _EMB
    if _EMB is not None:
        return _EMB
    with _EMB_LOCK:
        if _EMB is not None:
            return _EMB
        try:
            from bge_embedding import BGEEmbeddingService
            _EMB = BGEEmbeddingService()
        except Exception as e:
            _LOG.info("affinity_semantic: BGE non disponibile (%r); fallback disattivo", e)
            _EMB = False  # sentinel: gia' tentato e fallito
    return _EMB if _EMB is not False else None


def _cache_key(executors: Iterable[Any]) -> str:
    """Hash sha256 di (executor.name, tag) ordinati. Invalida cache quando
    qualunque executor cambia affinity (re-sign manifest, nuovo synth)."""
    h = hashlib.sha256()
    pairs = []
    for e in executors:
        name = getattr(e, "name", "") or ""
        aff = getattr(e, "affinity", None) or []
        for tag in aff:
            pairs.append((name, tag))
    pairs.sort()
    for name, tag in pairs:
        h.update(f"{name}\t{tag}\n".encode("utf-8"))
    return h.hexdigest()[:16]


def _load_cache(key: str) -> dict | None:
    p = CACHE_DIR / f"{key}.npz"
    if not p.exists():
        return None
    try:
        data = np.load(p, allow_pickle=False)
        return {
            "matrix": data["matrix"],
            "executor_names": data["executor_names"].tolist(),
            "tag_idx_to_executor": data["tag_idx_to_executor"].tolist(),
        }
    except Exception as e:
        _LOG.warning("affinity_semantic: cache load fail (%s): %r", p, e)
        return None


def _save_cache(key: str, matrix: np.ndarray,
                executor_names: list[str],
                tag_idx_to_executor: list[int]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    p = CACHE_DIR / f"{key}.npz"
    tmp = p.with_name(p.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez_compressed(
            f,
            matrix=matrix.astype(np.float32, copy=False),
            executor_names=np.array(executor_names),
            tag_idx_to_executor=np.array(tag_idx_to_executor, dtype=np.int32),
        )
    tmp.replace(p)


_CACHE_LOCK = threading.Lock()
_BUILT_CACHE: dict | None = None
_BUILT_CACHE_KEY: str | None = None


def build_or_load_cache(executors: list[Any]) -> dict | None:
    """Build (or load from disk) matrix shape (N_tags, 1024) +  reverse
    index. Cache in memoria una volta caricata; rebuild se chiave cambia.

    Ritorna None se BGE-M3 non disponibile o catalogo senza affinity.
    """
    global _BUILT_CACHE, _BUILT_CACHE_KEY
    key = _cache_key(executors)
    with _CACHE_LOCK:
        if _BUILT_CACHE is not None and _BUILT_CACHE_KEY == key:
            return _BUILT_CACHE
        cached = _load_cache(key)
        if cached is not None:
            _BUILT_CACHE = cached
            _BUILT_CACHE_KEY = key
            return cached

        emb = _get_embedder()
        if emb is None:
            return None
        all_tags: list[str] = []
        tag_idx_to_executor: list[int] = []
        executor_names: list[str] = [getattr(e, "name", "") for e in executors]
        for i, e in enumerate(executors):
            aff = getattr(e, "affinity", None) or []
            for tag in aff:
                if not isinstance(tag, str) or not tag.strip():
                    continue
                all_tags.append(tag)
                tag_idx_to_executor.append(i)
        if not all_tags:
            return None
        matrix = emb.embed_texts(all_tags)  # (N, 1024) L2-normalized
        _save_cache(key, matrix, executor_names, tag_idx_to_executor)
        result = {
            "matrix": matrix,
            "executor_names": executor_names,
            "tag_idx_to_executor": tag_idx_to_executor,
        }
        _BUILT_CACHE = result
        _BUILT_CACHE_KEY = key
        return result


def invalidate_cache() -> None:
    """Forza rebuild al prossimo build_or_load_cache. Usato dai test."""
    global _BUILT_CACHE, _BUILT_CACHE_KEY
    with _CACHE_LOCK:
        _BUILT_CACHE = None
        _BUILT_CACHE_KEY = None


def semantic_max_per_executor(query: str, cache: dict | None
                                 ) -> dict[str, float]:
    """Max cosine per executor verso i suoi tag affinity. Ritorna dict
    {executor_name: float [0..1]}. Vuoto se cache non disponibile."""
    if cache is None or not query.strip():
        return {}
    emb = _get_embedder()
    if emb is None:
        return {}
    try:
        qv = emb.embed_query(query)  # (1024,)
    except Exception as e:
        _LOG.warning("affinity_semantic: embed_query fail: %r", e)
        return {}
    matrix = cache["matrix"]
    scores = matrix @ qv  # (N_tags,) cosine perche' L2-normalized entrambi
    out: dict[str, float] = {}
    names = cache["executor_names"]
    idx_map = cache["tag_idx_to_executor"]
    for i, exec_idx in enumerate(idx_map):
        name = names[exec_idx]
        s = float(scores[i])
        prev = out.get(name)
        if prev is None or s > prev:
            out[name] = s
    return out


def is_enabled() -> bool:
    """Default ON; opt-out via METNOS_SEMANTIC_MATCH=0."""
    return os.environ.get("METNOS_SEMANTIC_MATCH", "1") != "0"


def threshold() -> int:
    try:
        return int(os.environ.get("METNOS_SEMANTIC_THRESHOLD",
                                    str(SEMANTIC_THRESHOLD_DEFAULT)))
    except ValueError:
        return SEMANTIC_THRESHOLD_DEFAULT


def alpha() -> float:
    try:
        return float(os.environ.get("METNOS_SEMANTIC_ALPHA",
                                     str(SEMANTIC_ALPHA_DEFAULT)))
    except ValueError:
        return SEMANTIC_ALPHA_DEFAULT
