"""Strategy `length_adaptive`: dispatch in base a lunghezza query.

Pensiero laterale: le strategy hanno regimi ottimali diversi a seconda
della verbosita' della query.
- Query corte (1-3 token, es. "che ore sono", "trova foto"): il verb
  e' dominante → trie (sfrutta struttura §2.2)
- Query medie (4-8 token, es. "manda email gli appuntamenti di
  domani"): pattern token-rank classico → token_flat
- Query lunghe (>8 token): query verbose con context, BGE-M3 utile
  per match semantico → selective_semantic

Determinismo §7.9: dispatch O(1) sulla lunghezza. Sub-strategy stesse
proprieta'.
"""
from __future__ import annotations

import re
from typing import Callable


def _count_meaningful_tokens(query: str) -> int:
    """Conta token >=2 char (skip stopword brevi)."""
    return len([t for t in re.findall(r"\w+", (query or "").lower())
                 if len(t) >= 2])


class LengthAdaptiveStrategy:
    name = "length_adaptive"

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
        n_tokens = _count_meaningful_tokens(query)
        from prefilter_strategies import select_strategy
        if n_tokens <= 3:
            chosen = "trie"
        elif n_tokens <= 8:
            chosen = "token_flat"
        else:
            chosen = "selective_semantic"
        strat = select_strategy(chosen)
        candidates, info = strat.rank(
            query, catalog, k_min=k_min, k_max=k_max,
            llm_call=llm_call, prefer_intent=prefer_intent,
        )
        info = dict(info or {})
        info["length_adaptive_chose"] = chosen
        info["length_adaptive_tokens"] = n_tokens
        return candidates, info


def make() -> LengthAdaptiveStrategy:
    return LengthAdaptiveStrategy()
