"""Strategy `trie_v2`: trie ottimizzato (17/5/2026 sera).

Migliorie rispetto a `trie` v1:
1. **Multi-verb path**: detect TUTTI i verbi canonici della query, lookup
   per ciascuno → UNION dei survivors (es. "fissa appuntamento e mandami
   email" → trie path [set,events] U [send,messages]).
2. **Qualifier-aware navigation**: se la query contiene marker di
   provider (google/gmail/drive/outlook/telegram) o qualifier formato
   (xlsx/csv/html), scendi nel trie a quel livello quando possibile.
3. **Partial-match fallback weighted**: se il path completo non c'e',
   usa i tool del nodo piu' profondo matchato + boost per profondita'
   (deeper = piu' specifico = score piu' alto). NON cade su legacy se
   anche un solo livello e' matchato.
4. **Stem expansion comune**: "appuntamenti"/"eventi"→events,
   "foto"/"immagini"→images, "mail"/"email"→messages (gia' coperto da
   _OBJECT_HINTS, riusato).
5. **Score weighted dal rank legacy**: il legacy sui survivors riceve
   score boost per i tool del trie depth maggiore.

Determinismo §7.9 puro. Backward compat sul fallback finale.
"""
from __future__ import annotations

import re
from typing import Callable


class TrieNode:
    __slots__ = ("children", "tools", "depth")

    def __init__(self, depth: int = 0):
        self.children: dict[str, TrieNode] = {}
        self.tools: list = []  # tool che terminano qui o discendono
        self.depth = depth


_TRIE_CACHE: dict[str, tuple[str, TrieNode]] = {}


_QUALIFIER_MARKERS = {
    "google_workspace": ("google", "gmail", "drive", "gdrive",
                         "workspace", "gcal"),
    "xlsx": ("xlsx", "spreadsheet", "foglio", "sheet"),
    "csv": ("csv", "comma"),
    "text": ("text", "testo", "txt", "md", "markdown"),
    "html": ("html", "htm", "pagina"),
    "ocr": ("ocr", "scan", "scansione"),
    "pdf": ("pdf",),
    "zip": ("zip", "archive", "archivio"),
    "image": ("foto", "immagine", "immagini", "photo", "image"),
}


def _build_trie(catalog_list) -> TrieNode:
    import hashlib
    sig = hashlib.sha256(
        "|".join(sorted(getattr(e, "name", "") for e in catalog_list)).encode()
    ).hexdigest()[:16]
    cached = _TRIE_CACHE.get("current")
    if cached and cached[0] == sig:
        return cached[1]
    root = TrieNode(depth=0)
    for tool in catalog_list:
        name = getattr(tool, "name", "") or ""
        if not name:
            continue
        parts = name.split("_")
        node = root
        for i, p in enumerate(parts):
            if p not in node.children:
                node.children[p] = TrieNode(depth=i + 1)
            node = node.children[p]
            node.tools.append(tool)
    _TRIE_CACHE["current"] = (sig, root)
    return root


def _detect_qualifiers(query: str) -> list[str]:
    """Ritorna qualifier rilevati nella query, in ordine di apparizione."""
    qlow = (query or "").lower()
    detected = []
    for qual, markers in _QUALIFIER_MARKERS.items():
        if any(re.search(r"\b" + re.escape(m) + r"\b", qlow) for m in markers):
            detected.append(qual)
    return detected


def _detect_objects_all(qtokens, query: str | None = None) -> list[str]:
    """Ritorna TUTTI gli oggetti canonici matchati, ordinati per hit count."""
    from prefilter import _OBJECT_HINTS
    scores = {}
    for obj, hints in _OBJECT_HINTS.items():
        hit = sum(1 for h in hints if h in qtokens)
        if hit:
            scores[obj] = hit
    return [obj for obj, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]


def _navigate_path(root: TrieNode, path: list[str]) -> tuple[TrieNode, int]:
    """Naviga il trie. Ritorna (deepest_node, depth_reached)."""
    node = root
    reached = 0
    for p in path:
        child = node.children.get(p)
        if child is None:
            break
        node = child
        reached += 1
    return node, reached


class TrieV2Strategy:
    name = "trie_v2"

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
            return [], {"chosen_k": 0, "confidence": 0.0,
                        "reason": "empty_catalog"}
        from prefilter import (
            _filter_dormant, detect_canonical_verbs_all, tokenize,
        )
        catalog_list = _filter_dormant(catalog_list)
        root = _build_trie(catalog_list)
        qtokens = tokenize(query)
        if not qtokens:
            from prefilter import _rank_adaptive_legacy
            return _rank_adaptive_legacy(
                query, catalog, k_min=k_min, k_max=k_max,
                llm_call=llm_call, prefer_intent=prefer_intent,
            )
        # 1. Multi-verb + multi-object detection
        verbs = detect_canonical_verbs_all(qtokens) or []
        objects = _detect_objects_all(qtokens, query) or []
        qualifiers = _detect_qualifiers(query)
        # 2. Compose path candidates: per ogni verb, prova object + qualifier
        path_candidates: list[list[str]] = []
        if verbs and objects:
            for v in verbs:
                for o in objects:
                    base = [v, o]
                    path_candidates.append(base)
                    # Estendi con qualifier (xlsx, google_workspace, ...)
                    for q in qualifiers:
                        path_candidates.append(base + [q])
        elif verbs:
            for v in verbs:
                path_candidates.append([v])
        elif objects:
            # No verb: prova tutti i verbi sotto oggetto
            for o in objects:
                for v in ("read", "find", "get"):  # default verb read-only
                    path_candidates.append([v, o])
        if not path_candidates:
            from prefilter import _rank_adaptive_legacy
            return _rank_adaptive_legacy(
                query, catalog, k_min=k_min, k_max=k_max,
                llm_call=llm_call, prefer_intent=prefer_intent,
            )
        # 3. Per ogni path naviga il trie, raccogli (depth_reached, tools)
        weighted_survivors: dict[str, tuple[int, object]] = {}  # name -> (max_depth, tool)
        max_depth_seen = 0
        paths_used: list[tuple[list[str], int]] = []
        for path in path_candidates:
            node, depth = _navigate_path(root, path)
            if depth == 0:
                continue
            paths_used.append((path[:depth], depth))
            max_depth_seen = max(max_depth_seen, depth)
            for tool in node.tools:
                name = getattr(tool, "name", "")
                if not name:
                    continue
                prev = weighted_survivors.get(name)
                if prev is None or prev[0] < depth:
                    weighted_survivors[name] = (depth, tool)
        # 4. Fallback se nessun path utile
        if not weighted_survivors or max_depth_seen == 0:
            from prefilter import _rank_adaptive_legacy
            return _rank_adaptive_legacy(
                query, catalog, k_min=k_min, k_max=k_max,
                llm_call=llm_call, prefer_intent=prefer_intent,
            )
        # 5. Tool-rank legacy sui survivors. Boost per depth maggiore:
        #    survivors al depth massimo finiscono prima.
        deepest = [t for name, (d, t) in weighted_survivors.items()
                    if d == max_depth_seen]
        shallower = [t for name, (d, t) in weighted_survivors.items()
                      if d < max_depth_seen]
        # Concat: deepest prima, poi shallower
        ordered = deepest + shallower
        # Fallback se troppo pochi
        if len(ordered) < k_min:
            # Allarga al catalog completo MA mantieni i survivors deepest in testa
            ordered_set = set(getattr(t, "name", "") for t in ordered)
            extras = [t for t in catalog_list
                       if getattr(t, "name", "") not in ordered_set]
            ordered = ordered + extras
        from prefilter import _rank_adaptive_legacy
        candidates, info = _rank_adaptive_legacy(
            query, ordered, k_min=k_min, k_max=k_max,
            llm_call=llm_call, prefer_intent=prefer_intent,
        )
        info = dict(info or {})
        info["trie_v2_max_depth"] = max_depth_seen
        info["trie_v2_paths_used"] = len(paths_used)
        info["trie_v2_survivors"] = len(weighted_survivors)
        info["trie_v2_deepest_n"] = len(deepest)
        return candidates, info


def make() -> TrieV2Strategy:
    return TrieV2Strategy()
