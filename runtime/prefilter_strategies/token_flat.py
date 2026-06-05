"""Strategy `token_flat` (alias `legacy`): wrap del prefilter attuale.

Delega a `prefilter.rank_adaptive` originale per backward compatibility.
Baseline per A/B con altre strategy.

Determinismo §7.9: zero LLM aggiuntivo. Lo stesso `rank_adaptive` puo'
usare LLM intent extractor se `llm_call` e' passato, ma quello e' input
del chiamante, non scelta della strategy.
"""
from __future__ import annotations

from typing import Callable


class TokenFlatStrategy:
    name = "token_flat"

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
        # Import locale per evitare ciclo: prefilter.rank_adaptive importa
        # il package strategies indirettamente.
        from prefilter import _rank_adaptive_legacy
        return _rank_adaptive_legacy(
            query, catalog, k_min=k_min, k_max=k_max,
            llm_call=llm_call, prefer_intent=prefer_intent,
        )


def make() -> TokenFlatStrategy:
    return TokenFlatStrategy()
