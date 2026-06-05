"""Strategy `selective_semantic_v2`: ottimizzato (17/5/2026 sera).

Migliorie rispetto a `selective_semantic` v1:
1. **Trigger basato su margin top1-top2**, non `route_info.confidence`
   (poco affidabile, vincolato a 1.0 quando intent_extractor scatta).
   Margin piccolo (top1 - top2 < threshold) = scelta incerta → attiva
   semantic re-rank.
2. **Formula score linear combination corretta**:
   `score_final = (1-alpha) * score_token_norm + alpha * score_semantic`
   con score_token_norm normalizzato 0-1 (era 1/(i+1)).
3. **Triggera anche su low top score assoluto** (top score < min_score):
   query con weak match al token-rank, il semantic puo' ripescare.
4. **Numerical cap survivors** per BGE-M3 (max 30 tool re-ranked) per
   evitare di pagare embedding cost su pool grandi.
5. **Cache embedding query** (un solo embed per call).

Determinismo §7.9 (BGE-M3 ONNX e' deterministico). Cost-aware:
embedding solo quando serve.
"""
from __future__ import annotations

import os
from typing import Callable


_DEFAULT_MARGIN_THRESHOLD = 0.15  # se top1-top2_norm < 0.15 → attiva semantic
_DEFAULT_LOW_SCORE_THRESHOLD = 0.3  # se top1_norm < 0.3 → attiva semantic
_DEFAULT_ALPHA = 0.35  # peso semantic nella combinazione
_MAX_SEMANTIC_RERANK = 30


def _normalize_scores(values: list[float]) -> list[float]:
    """Normalizza in [0, 1] con max-min scaling. Se max==min, ritorna 1.0
    per primi e degrade per rank."""
    if not values:
        return []
    vmax, vmin = max(values), min(values)
    if vmax == vmin:
        return [1.0 / (i + 1) for i in range(len(values))]
    return [(v - vmin) / (vmax - vmin) for v in values]


class SelectiveSemanticV2Strategy:
    name = "selective_semantic_v2"

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
        # Step 1: token-flat baseline con piu' candidati per re-rank
        from prefilter import (
            _rank_adaptive_legacy, _filter_dormant, tokenize,
            affinity_score, detect_canonical_verb, detect_canonical_object,
        )
        baseline_candidates, baseline_info = _rank_adaptive_legacy(
            query, catalog, k_min=k_min, k_max=k_max,
            llm_call=llm_call, prefer_intent=prefer_intent,
        )
        if not baseline_candidates:
            return baseline_candidates, baseline_info
        # Step 2: ri-calcola score grezzi per i candidati (deterministic).
        # Il legacy non li espone direttamente, allora ricostruisci via affinity_score.
        catalog_list = _filter_dormant(list(catalog))
        qtokens = tokenize(query)
        cv = detect_canonical_verb(qtokens)
        co = detect_canonical_object(qtokens, query)
        raw_scores = []
        for tool in baseline_candidates:
            s = affinity_score(qtokens, tool,
                                query_canonical_verb=cv,
                                query_canonical_object=co)
            raw_scores.append(float(s))
        normed = _normalize_scores(raw_scores)
        top1 = normed[0] if normed else 0
        top2 = normed[1] if len(normed) > 1 else 0
        margin = top1 - top2
        margin_th = float(os.environ.get(
            "METNOS_SEM_V2_MARGIN", str(_DEFAULT_MARGIN_THRESHOLD)))
        low_score_th = float(os.environ.get(
            "METNOS_SEM_V2_LOW_SCORE", str(_DEFAULT_LOW_SCORE_THRESHOLD)))
        # Skip semantic se confident (margin grande E top1 alto)
        if margin >= margin_th and top1 >= low_score_th:
            return baseline_candidates, baseline_info
        # Step 3: re-rank con semantic
        try:
            from affinity_semantic import (
                is_enabled as _sem_enabled,
                build_or_load_cache as _sem_build,
                semantic_max_per_executor as _sem_max,
            )
            if not _sem_enabled():
                return baseline_candidates, baseline_info
            cache = _sem_build(catalog_list)
            if cache is None:
                return baseline_candidates, baseline_info
            # Limita pool per evitare overhead embedding pesante
            sem_pool = baseline_candidates[:_MAX_SEMANTIC_RERANK]
            semmap = _sem_max(query, cache)
            if not semmap:
                return baseline_candidates, baseline_info
            alpha = float(os.environ.get(
                "METNOS_SEM_V2_ALPHA", str(_DEFAULT_ALPHA)))
            # Score combinato
            combined = []
            for i, tool in enumerate(sem_pool):
                name = getattr(tool, "name", "") or ""
                token_norm = normed[i] if i < len(normed) else 0
                sem_score = semmap.get(name, 0.0)
                # Normalize sem in [0,1] approx (BGE-M3 cosine 0-1 grezzo)
                sem_norm = max(0.0, min(1.0, sem_score))
                final = (1 - alpha) * token_norm + alpha * sem_norm
                combined.append((final, tool))
            combined.sort(key=lambda x: x[0], reverse=True)
            reranked = [t for _, t in combined[:k_max]]
            info = dict(baseline_info or {})
            info["sem_v2_active"] = True
            info["sem_v2_margin"] = round(margin, 3)
            info["sem_v2_top1"] = round(top1, 3)
            info["sem_v2_alpha"] = alpha
            return reranked, info
        except Exception as ex:
            import logging
            logging.getLogger(__name__).debug("sem_v2 fail: %s", ex)
            return baseline_candidates, baseline_info


def make() -> SelectiveSemanticV2Strategy:
    return SelectiveSemanticV2Strategy()
