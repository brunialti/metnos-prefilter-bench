"""Strategy `rrf_ensemble`: Reciprocal Rank Fusion del top-3 strategy.

Pensiero laterale: invece di scegliere UNA strategia migliore, COMBINA
le top-3 con RRF (Cormack, Clarke, Buettcher 2009). Letteratura IR:
RRF batte tipicamente la migliore strategy singola di 5-15% perche'
le strategy fanno errori non correlati.

Formula: `score(tool) = sum(1/(k + rank_i(tool)) for each strategy i)`
con k=60 (default raccomandato dalla letteratura).

Combinazione corrente: trie (verb+object structure) + token_flat
(BM25-like classic) + selective_semantic (BGE-M3 recovery). Errori
ortogonali → ensemble robusto.

Cost: somma latency delle 3 (12-70ms tipico).
"""
from __future__ import annotations

from typing import Callable

_RRF_K = 60  # parametro raccomandato in letteratura


class RrfEnsembleStrategy:
    name = "rrf_ensemble"

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
        # Esegui le 3 strategy
        from prefilter_strategies import select_strategy
        sub_strategies = ["trie", "token_flat", "selective_semantic"]
        rankings: list[list[str]] = []  # liste di name in ordine di rank
        name_to_tool: dict[str, object] = {}
        for sname in sub_strategies:
            try:
                strat = select_strategy(sname)
                # k_max * 2 per avere piu' candidati su cui fare RRF
                candidates, _info = strat.rank(
                    query, catalog, k_min=k_min, k_max=k_max * 2,
                    llm_call=llm_call, prefer_intent=prefer_intent,
                )
                ranking = []
                for t in candidates:
                    name = getattr(t, "name", "") or ""
                    if not name:
                        continue
                    ranking.append(name)
                    name_to_tool[name] = t
                rankings.append(ranking)
            except Exception:
                pass
        if not rankings:
            from prefilter import _rank_adaptive_legacy
            return _rank_adaptive_legacy(
                query, catalog, k_min=k_min, k_max=k_max,
                llm_call=llm_call, prefer_intent=prefer_intent,
            )
        # RRF aggregation
        rrf_scores: dict[str, float] = {}
        for ranking in rankings:
            for rank, name in enumerate(ranking, start=1):
                rrf_scores[name] = rrf_scores.get(name, 0.0) + 1.0 / (_RRF_K + rank)
        # Sort by RRF score desc
        sorted_names = sorted(rrf_scores.keys(),
                               key=lambda n: rrf_scores[n], reverse=True)
        final_tools = [name_to_tool[n] for n in sorted_names[:k_max]
                        if n in name_to_tool]
        return final_tools, {
            "chosen_k": len(final_tools),
            "confidence": min(1.0, rrf_scores[sorted_names[0]] * 30) if sorted_names else 0,
            "reason": f"rrf_ensemble({len(rankings)} strategies)",
            "rrf_top_score": round(rrf_scores[sorted_names[0]], 4) if sorted_names else 0,
        }


def make() -> RrfEnsembleStrategy:
    return RrfEnsembleStrategy()
