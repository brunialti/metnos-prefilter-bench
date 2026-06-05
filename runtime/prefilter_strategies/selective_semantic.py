"""Strategy `selective_semantic`: token-flat + BGE-M3 semantic come
secondary ranker quando la confidence del primario e' bassa.

Estende ADR 0134 dall'affinity-only al prefilter principale. Idea:
- Esegui token-flat normale (rank_adaptive legacy).
- Se confidence < soglia (`METNOS_SELECTIVE_SEM_THRESHOLD`, default 0.5)
  E BGE-M3 disponibile: re-rank top-K con score combinato
  `score = token + alpha * semantic_cosine`.
- Altrimenti: ritorna risultato token-flat invariato.

Cost-aware: zero overhead sulle query "chiare" (confidence alta);
semantic attivo solo dove serve. Backward compat completo.

Determinismo §7.9: la soglia e' parametro, no LLM, BGE-M3 embedding e'
deterministico (modello fissato).
"""
from __future__ import annotations

import os
from typing import Callable


class SelectiveSemanticStrategy:
    name = "selective_semantic"

    def rank(
        self,
        query: str,
        catalog,
        k_min: int = 5,
        k_max: int = 8,
        *,
        llm_call: Callable | None = None,
        prefer_intent: bool = True,
    ) -> tuple[list, dict]:
        # Step 1: token-flat baseline
        from prefilter import _rank_adaptive_legacy
        candidates, route_info = _rank_adaptive_legacy(
            query, catalog, k_min=k_min, k_max=k_max,
            llm_call=llm_call, prefer_intent=prefer_intent,
        )
        conf = (route_info or {}).get("confidence", 1.0)
        try:
            conf = float(conf or 0.0)
        except (TypeError, ValueError):
            conf = 0.0
        threshold = float(os.environ.get(
            "METNOS_SELECTIVE_SEM_THRESHOLD", "0.5"))
        if conf >= threshold:
            return candidates, route_info  # confidence alta: skip semantic
        # Step 2: semantic re-rank top-K
        try:
            from affinity_semantic import (
                is_enabled as _sem_enabled,
                build_or_load_cache as _sem_build,
                semantic_max_per_executor as _sem_max,
                alpha as _sem_alpha,
            )
            if not _sem_enabled():
                return candidates, route_info
            cache = _sem_build(list(catalog))
            if cache is None:
                return candidates, route_info
            semmap = _sem_max(query, cache)
            if not semmap:
                return candidates, route_info
            a = _sem_alpha()
            # Re-rank candidates by combined score. Tie-breaker: original order.
            scored = []
            for i, e in enumerate(candidates):
                sem = semmap.get(e.name, 0.0)
                # Score combinato: 1 / (1 + i) come proxy del rank token,
                # piu' alpha * semantic.
                combined = (1.0 / (i + 1)) + a * 0.01 * sem
                scored.append((combined, e))
            scored.sort(key=lambda p: p[0], reverse=True)
            reranked = [e for _, e in scored]
            info = dict(route_info or {})
            info["reason"] = (info.get("reason", "") or "") + "+semantic"
            info["selective_semantic_active"] = True
            return reranked, info
        except Exception:
            return candidates, route_info  # fallback silente


def make() -> SelectiveSemanticStrategy:
    return SelectiveSemanticStrategy()
