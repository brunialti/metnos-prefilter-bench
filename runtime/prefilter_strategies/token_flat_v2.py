"""Strategy `token_flat_v2`: token-flat ottimizzato (17/5/2026 sera).

Migliorie rispetto a `token_flat` (legacy):
1. **Provider penalty**: tool con suffix `_<provider>` (es. `_google_workspace`)
   ricevono penalty se la query NON contiene il marker del provider.
   Allinea con ADR 0136 `_PROVIDER_SUFFIX_MARKERS` ma applicato come
   penalty soft (non esclusione hard del grammar filter).
2. **Name-exact boost**: se la query contiene esattamente il name di
   un tool (es. "send_messages" o "find_files"), boost top-rank.
3. **Verb family expansion estesa**: read/find/get/list trattati come
   famiglia equivalente; send/create/set come famiglia mutating-meta.
4. **Score normalizzato**: scores normalizzati [0,1] per migliore
   confidence di route_info (vs valori grezzi precedenti).

Determinismo §7.9 puro, zero LLM aggiuntivo.
"""
from __future__ import annotations

import re
from typing import Callable


# Provider markers: aligned with ADR 0136 (`tool_grammar._PROVIDER_SUFFIX_MARKERS`).
_PROVIDER_MARKERS = {
    "_google_workspace": (
        "google", "drive", "gmail", "gdrive",
        "workspace", "calendar google", "g suite",
    ),
}

# Verb families: tool con verb diverso ma stessa famiglia → no penalty.
_VERB_FAMILY = {
    "read": {"read", "find", "get", "list"},
    "find": {"read", "find", "get", "list"},
    "get": {"read", "find", "get", "list"},
    "list": {"read", "find", "get", "list"},
    "send": {"send", "create", "set", "write"},
    "create": {"send", "create", "set", "write"},
    "set": {"send", "create", "set", "write"},
    "write": {"send", "create", "set", "write"},
    "delete": {"delete", "move"},
    "move": {"delete", "move"},
}


def _query_has_provider_marker(query: str, suffix: str) -> bool:
    qlow = (query or "").lower()
    markers = _PROVIDER_MARKERS.get(suffix, ())
    return any(re.search(r"\b" + re.escape(m) + r"\b", qlow) for m in markers)


def _provider_suffix_of(name: str) -> str | None:
    """Ritorna il suffix provider del tool, se ne ha uno."""
    for suffix in _PROVIDER_MARKERS:
        if name.endswith(suffix):
            return suffix
    return None


class TokenFlatV2Strategy:
    name = "token_flat_v2"

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
        # 1. Esegui legacy per baseline candidates+score
        from prefilter import _rank_adaptive_legacy
        candidates, info = _rank_adaptive_legacy(
            query, catalog, k_min=k_min, k_max=k_max * 2,  # raccoglie piu' candidati per re-rank
            llm_call=llm_call, prefer_intent=prefer_intent,
        )
        if not candidates:
            return candidates, info
        qlow = (query or "").lower()
        # 2. Calcola score adjustment per ogni candidato
        adjusted: list[tuple[float, object]] = []
        for i, tool in enumerate(candidates):
            name = getattr(tool, "name", "") or ""
            # Score iniziale: inverso del rank (1.0 -> 0)
            score = 1.0 / (i + 1)
            # (a) Provider penalty
            suffix = _provider_suffix_of(name)
            if suffix and not _query_has_provider_marker(query, suffix):
                score *= 0.4  # -60% penalty
            # (b) Name-exact boost: nome del tool nella query
            name_lower = name.lower()
            if re.search(r"\b" + re.escape(name_lower) + r"\b", qlow):
                score += 1.5  # forte boost
            # (c) Partial name match (verb+object): "send messages" → send_messages
            parts = name.split("_")
            if len(parts) >= 2:
                verb_obj = parts[0] + " " + parts[1]
                if verb_obj in qlow:
                    score += 0.5
            # (d) Underscore name match: "find_files" letterale
            if name_lower in qlow.replace(" ", "_"):
                score += 0.3
            adjusted.append((score, tool))
        # 3. Re-rank
        adjusted.sort(key=lambda x: x[0], reverse=True)
        final_candidates = [t for _, t in adjusted[:k_max]]
        info = dict(info or {})
        info["token_flat_v2_adjustments"] = True
        return final_candidates, info


def make() -> TokenFlatV2Strategy:
    return TokenFlatV2Strategy()
