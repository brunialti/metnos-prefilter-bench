"""Strategy `constraint`: filtro constraint-based (SAT-style).

Estrae constraint dall'intent: `{verb_in: {...}, object_in: {...},
qualifier_in?: {...}, exclude_provider?: ...}`. Filtro come intersezione
di set. Risultato deterministico, esplicabile (audit log perfetto).

Pattern §7.9 puro: zero LLM aggiuntivo (riusa intent_extractor se
disponibile, fallback bag-of-words). Compatibile con grammar GBNF.

Best for: query con verbo + oggetto chiari (most "Manda mail a X",
"Trova foto di Y"). Fallback per query vaghe ("cosa puoi fare?").
"""
from __future__ import annotations

from typing import Callable


def _extract_constraints(query: str, llm_call=None,
                          prefer_intent=True) -> dict:
    """Ritorna `{verb, object, qualifier?, provider?}` dalla query.

    Usa intent_extractor (LLM) se disponibile; fallback bag-of-words.
    """
    verb = None
    obj = None
    qualifier = None
    if llm_call is not None and prefer_intent:
        try:
            from intent_extractor import extract_intent
            intent = extract_intent(query, llm_call)
            if intent:
                verb = intent.get("verb")
                obj = intent.get("object")
                qualifier = intent.get("qualifier")
        except Exception:
            pass
    if not verb or not obj:
        from prefilter import detect_canonical_verb, detect_canonical_object, tokenize
        qtokens = tokenize(query)
        if not verb:
            verb = detect_canonical_verb(qtokens)
        if not obj:
            obj = detect_canonical_object(qtokens, query)
    # Provider qualifier: check marker
    provider = None
    qlow = (query or "").lower()
    if any(m in qlow for m in ("google", "gmail", "drive", "workspace", "gcal", "gdrive")):
        provider = "google_workspace"
    elif "outlook" in qlow or "microsoft" in qlow:
        provider = "outlook"
    elif "telegram" in qlow:
        provider = "telegram"
    return {"verb": verb, "object": obj, "qualifier": qualifier,
            "provider": provider}


def _matches(tool, constraints: dict) -> tuple[bool, str]:
    """Verifica se un tool soddisfa i constraint. Ritorna (ok, reason)."""
    name = getattr(tool, "name", "") or ""
    if not name:
        return False, "no_name"
    parts = name.split("_")
    tool_verb = parts[0] if parts else ""
    tool_obj = parts[1] if len(parts) > 1 else ""
    # Verb constraint
    cv = constraints.get("verb")
    if cv and tool_verb != cv:
        # Alias: read accetta find, find accetta read (read-only family)
        READ_FAMILY = {"read", "find", "get", "list"}
        if not (cv in READ_FAMILY and tool_verb in READ_FAMILY):
            return False, f"verb_mismatch({cv}!={tool_verb})"
    # Object constraint
    co = constraints.get("object")
    if co and tool_obj != co:
        return False, f"object_mismatch({co}!={tool_obj})"
    # Provider constraint: se query menziona provider, tool con quel
    # provider preferito; altrimenti tool generico ok.
    cp = constraints.get("provider")
    name_lower = name.lower()
    if cp and cp not in name_lower:
        # Tool generico ammesso comunque (provider esplicito e' preferenza
        # non vincolo: il PLANNER decide finalmente)
        pass
    return True, "ok"


class ConstraintStrategy:
    name = "constraint"

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
        catalog_list = list(catalog)
        if not catalog_list:
            return [], {"chosen_k": 0, "confidence": 0.0, "reason": "empty_catalog"}
        from prefilter import _filter_dormant
        catalog_list = _filter_dormant(catalog_list)
        constraints = _extract_constraints(query, llm_call, prefer_intent)
        # Filtra
        matched = []
        for tool in catalog_list:
            ok, _reason = _matches(tool, constraints)
            if ok:
                matched.append(tool)
        # Fallback se constraint troppo rigido
        if len(matched) < k_min:
            from prefilter import _rank_adaptive_legacy
            candidates, info = _rank_adaptive_legacy(
                query, catalog, k_min=k_min, k_max=k_max,
                llm_call=llm_call, prefer_intent=prefer_intent,
            )
            info = dict(info or {})
            info["constraint_fallback"] = "too_few_matches"
            info["constraint_extracted"] = constraints
            return candidates, info
        # Provider boost: tool con provider matching → prima
        cp = constraints.get("provider")
        if cp:
            def _provider_rank(t):
                return 0 if cp in getattr(t, "name", "").lower() else 1
            matched.sort(key=_provider_rank)
        # Token-rank sui survivors per refine final
        from prefilter import _rank_adaptive_legacy
        candidates, info = _rank_adaptive_legacy(
            query, matched, k_min=k_min, k_max=k_max,
            llm_call=llm_call, prefer_intent=prefer_intent,
        )
        info = dict(info or {})
        info["constraint_extracted"] = constraints
        info["constraint_matched"] = len(matched)
        return candidates, info


def make() -> ConstraintStrategy:
    return ConstraintStrategy()
