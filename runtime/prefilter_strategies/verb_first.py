"""Strategy `verb_first`: dispatch deterministico §2.2 verb→tool sub-set,
poi token-rank legacy sui survivors.

Idea (§5.bis.1 della roadmap):
1. Estrai verb canonico dalla query (intent_extractor o fallback tokens).
2. Filtra catalog ai tool con prefisso `<verb>_*` (es. verb="find" →
   find_files, find_messages, find_images_indices, ...).
3. Se sub-set vuoto o << k_min: fallback al catalog completo.
4. Esegui token-flat rank sul sub-set (riduce N→N/23).

Deterministico §7.9. Vocabolario chiuso §2.2 amplifica il segnale.
Compatibile con grammar GBNF (top-K input invariato).

Costo: O(N) per filtering + O(K) per rank. Nessun LLM aggiuntivo.
"""
from __future__ import annotations

import os
from typing import Callable


class VerbFirstStrategy:
    name = "verb_first"

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
        from prefilter import _rank_adaptive_legacy, tokenize, detect_canonical_verbs_all
        # 1. Detect verb canonico
        verbs = []
        if llm_call is not None and prefer_intent:
            try:
                from intent_extractor import extract_intent
                intent = extract_intent(query, llm_call)
                if intent and intent.get("verb"):
                    verbs = [intent["verb"]]
            except Exception:
                pass
        if not verbs:
            qtokens = tokenize(query)
            verbs = detect_canonical_verbs_all(qtokens) or []
        # 2. Filtra catalog
        catalog_list = list(catalog)
        if verbs:
            filtered = [
                e for e in catalog_list
                if any((getattr(e, "name", "") or "").startswith(v + "_")
                       or (getattr(e, "name", "") or "") == v
                       for v in verbs)
            ]
        else:
            filtered = []
        # 3. Fallback se sub-set troppo piccolo
        if len(filtered) < k_min:
            filtered = catalog_list
            filter_reason = "verb_subset_too_small"
        else:
            filter_reason = f"verb_first({','.join(verbs)})"
        # 4. Token-rank sui survivors
        candidates, route_info = _rank_adaptive_legacy(
            query, filtered, k_min=k_min, k_max=k_max,
            llm_call=llm_call, prefer_intent=prefer_intent,
        )
        info = dict(route_info or {})
        info["verb_first_filter"] = filter_reason
        info["verb_first_subset_size"] = len(filtered)
        return candidates, info


def make() -> VerbFirstStrategy:
    return VerbFirstStrategy()
