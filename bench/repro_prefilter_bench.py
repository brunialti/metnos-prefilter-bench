#!/usr/bin/env python3
"""Reproducible prefilter-strategy benchmark (PUBLIC, self-contained).

This is the harness behind the "closed-vocab tool filtering" write-up. It is
designed so anyone who clones the repo can reproduce the numbers WITHOUT our
private turn logs: it reads a frozen, PII-scrubbed query snapshot
(``bench/corpus_snapshot.jsonl``) and a frozen snapshot of the real 96-tool
catalog (``bench/catalog_snapshot.json``), then measures, per strategy:

  - Recall@1 / Recall@5  (ground truth = the first tool prod actually called)
  - latency (mean / p95)
  - mean pool tokens (how much prompt the returned pool costs the planner)

Two modes:
  --mode comparison   real catalog only, all strategies + production rules row
  --mode scaling      pad catalog with HARD synthetic distractors to
                      84/250/500/1000 and report the Recall@5 slope

HONESTY NOTES (read these, they are the method caveats):
  * Ground truth is prod behaviour, not an oracle. It measures agreement with
    what the production planner+prefilter already chose, NOT correctness. A few
    GT labels are genuine prod mis-routes. We do not hand-correct them.
  * The snapshot is one single-user assistant's organic traffic (n=234), with
    test/e2e/bench/smoke turns excluded by conversation_id. Absolute recall is
    in-domain; the interesting quantity is the SLOPE vs pool size.
  * Distractors are HARD negatives: recombinations of the SAME closed vocab
    (verb x object x qualifier) carrying real IT+EN affinity tokens, so they
    collide lexically with queries instead of being trivially separable.
  * Deterministic: distractor RNG is seeded (42); no LLM is called
    (prefer_intent=False, llm_call=None).

Usage:
    python3 bench/repro_prefilter_bench.py --mode comparison
    python3 bench/repro_prefilter_bench.py --mode scaling
    python3 bench/repro_prefilter_bench.py --mode both --output /tmp/report.md
"""
from __future__ import annotations

import os
import sys

# Determinism: ranking tie-breaks iterate sets/dicts, so the result depends on
# Python's per-process hash seed. Pin it and re-exec once, BEFORE any other
# import, so every reader gets the identical number (not a ±1pp hash-seed band).
if os.environ.get("PYTHONHASHSEED") != "0":
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable, *sys.argv])

import argparse
import json
import random
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

_ROOT = Path(__file__).resolve().parents[1]
_BENCH = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT / "runtime"))
# Typed rules (10.5 input-coverage / 11.0 schema-field) read this dir; the
# PII-free typing snapshot ships in bench/typing so the prod number reproduces
# without the private e2e tree.
os.environ.setdefault("METNOS_TYPING_DIR", str(_BENCH / "typing"))

SNAPSHOT = _BENCH / "corpus_snapshot.jsonl"
CATALOG = _BENCH / "catalog_snapshot.json"
POOL_SIZES = (84, 250, 500, 1000)
SEED = 42

# Strategies shown in the comparison table (excludes the "legacy" alias).
# Kept in a stable order; the renderer re-sorts by Recall@5.
COMPARISON_STRATEGIES = [
    "token_flat_v2", "token_flat", "cached_token_flat",
    "selective_semantic", "selective_semantic_v2", "length_adaptive",
    "constraint", "hybrid_cascade", "rrf_ensemble", "trie", "trie_v2",
    "verb_first", "fts5", "bloom",
]
# Lighter subset for the scaling sweep (4 pool sizes x N strategies x 234 q).
SCALING_STRATEGIES = [
    "token_flat_v2", "token_flat", "selective_semantic",
    "constraint", "verb_first", "trie", "fts5",
]


def load_snapshot() -> list[dict]:
    if not SNAPSHOT.exists():
        sys.exit(f"missing snapshot: {SNAPSHOT} (run build_prefilter_corpus.py)")
    out = []
    for ln in SNAPSHOT.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if ln:
            out.append(json.loads(ln))
    return out


def load_catalog_snapshot() -> list:
    """Frozen 96-tool catalog snapshot (PII-scrubbed) -> executor-like objects.

    Decoupled from the live signed loader on purpose: the bench reproduces the
    SAME numbers regardless of which executors happen to be installed, and needs
    no signing/runtime state. Fields are exactly what the strategies read.
    """
    if not CATALOG.exists():
        sys.exit(f"missing catalog snapshot: {CATALOG}")
    data = json.loads(CATALOG.read_text(encoding="utf-8"))
    return [SimpleNamespace(**d) for d in data]


def dense_available() -> bool:
    """True iff the BGE-M3 embedder is installed (optional dense baseline).

    Token-flat reproduces with zero non-stdlib deps. The dense rows are the
    baseline a skeptic runs to verify the tie; they need the embedding model
    (see bench/requirements-dense.txt). Absent it, we SKIP them honestly rather
    than show token-flat's silent fallback mislabelled as 'dense'.
    """
    try:
        import affinity_semantic as _as
        if not _as.is_enabled():
            return False
        return bool(_as.semantic_max_per_executor("ping", _as.build_or_load_cache(
            load_catalog_snapshot())))
    except Exception:
        return False


def _object_word_index() -> dict[str, list[str]]:
    """object -> [real IT+EN affinity words] (invert vocab synonym maps)."""
    import vocab
    idx: dict[str, list[str]] = {}
    for src in (vocab._OBJECT_SYNONYMS_IT, vocab._OBJECT_SYNONYMS_EN):
        for word, obj in src.items():
            idx.setdefault(obj, []).append(word)
    return idx


def build_hard_distractors(real_execs: list, n: int) -> list:
    """Generate n HARD-negative distractor tools.

    Each distractor is a verb_object[_qualifier] recombination of the REAL
    closed vocabulary that is NOT an existing tool, carrying real IT+EN
    affinity tokens for its verb and object. This makes distractors collide
    lexically with the query vocabulary (e.g. a query about mail can match
    both real `read_messages` and synthetic `compute_messages`), so token
    matching is genuinely stressed rather than trivially separable.
    """
    import vocab
    if not real_execs or n <= 0:
        return []
    rng = random.Random(SEED)
    verbs = sorted(vocab.ACTIONS)
    objects = sorted(vocab.OBJECTS)
    quals = list(vocab.QUALIFIERS)
    obj_words = _object_word_index()
    real_names = {getattr(e, "name", "") for e in real_execs}
    seen = set(real_names)
    out: list = []
    attempts = 0
    while len(out) < n and attempts < n * 40:
        attempts += 1
        v = rng.choice(verbs)
        o = rng.choice(objects)
        use_q = rng.random() < 0.35
        q = rng.choice(quals) if use_q else None
        name = f"{v}_{o}_{q}" if q else f"{v}_{o}"
        if name in seen:
            continue
        seen.add(name)
        verb_words = []
        vm = vocab.ACTION_MAPPING.get(v, {})
        verb_words = list(vm.get("it", []))[:3] + list(vm.get("en", []))[:3]
        ow = obj_words.get(o, [o])
        affinity = verb_words + ow[:4] + [v, o]
        template = real_execs[len(out) % len(real_execs)]
        out.append(SimpleNamespace(
            name=name,
            description=f"{' '.join(verb_words[:2])} {' '.join(ow[:2])}".strip()
            or f"{v} {o}",
            affinity=affinity,
            args_schema=getattr(template, "args_schema",
                                {"type": "object", "properties": {}}),
            timeout_s=getattr(template, "timeout_s", 30),
            capabilities=getattr(template, "capabilities", []),
            target_kind=getattr(template, "target_kind", ""),
            revertible=getattr(template, "revertible", False),
            critical=getattr(template, "critical", False),
        ))
    return out


def _tool_tokens(tool) -> int:
    name = getattr(tool, "name", "") or ""
    desc = getattr(tool, "description", "") or ""
    if isinstance(desc, dict):
        desc = desc.get("it") or desc.get("en") or next(iter(desc.values()), "")
    args = getattr(tool, "args_schema", {}) or {}
    try:
        args_str = json.dumps(args, ensure_ascii=False)
    except Exception:
        args_str = str(args)
    return (len(name) + len(str(desc)) + len(args_str)) // 4


def run_strategy(sname: str, corpus: list, executors: list,
                 *, k_max: int = 8, rules: bool = False) -> dict:
    from prefilter_strategies import select_strategy
    os.environ["METNOS_PREFILTER_RULES"] = "1" if rules else "0"
    tok = {getattr(e, "name", ""): _tool_tokens(e) for e in executors}
    try:
        strat = select_strategy(sname)
    except Exception as ex:  # noqa: BLE001
        return {"error": str(ex)}
    r5 = r1 = 0
    lat: list[float] = []
    pool_tok: list[int] = []
    n = 0
    for item in corpus:
        t0 = time.perf_counter()
        try:
            cands, _ = strat.rank(item["query"], executors, k_min=5,
                                  k_max=k_max, llm_call=None, prefer_intent=False)
        except Exception:  # noqa: BLE001
            continue
        lat.append((time.perf_counter() - t0) * 1000)
        names = [getattr(e, "name", "") for e in cands]
        gt = item["first_tool"]
        if gt in names[:5]:
            r5 += 1
        if names[:1] == [gt]:
            r1 += 1
        pool_tok.append(sum(tok.get(x, 0) for x in names))
        n += 1
    if not n:
        return {"error": "no queries scored"}
    lat_sorted = sorted(lat)
    return {
        "n": n,
        "recall_5": r5 / n,
        "recall_1": r1 / n,
        "mean_ms": statistics.mean(lat),
        "p95_ms": lat_sorted[int(len(lat_sorted) * 0.95)],
        "mean_pool_tokens": statistics.mean(pool_tok) if pool_tok else 0,
    }


def comparison(corpus, real_execs) -> list[str]:
    rows = []
    for s in COMPARISON_STRATEGIES:
        rows.append((s, run_strategy(s, corpus, real_execs, rules=False)))
    # PRODUCTION config (verified on live PID): METNOS_PREFILTER unset ->
    # _rank_adaptive_legacy (== strategy "token_flat") + METNOS_PREFILTER_RULES=1.
    # This is what a reader reproduces by cloning and running with the real env.
    rows.append(("token_flat+rules (PRODUCTION)",
                 run_strategy("token_flat", corpus, real_execs, rules=True)))
    rows = [(s, r) for s, r in rows if "error" not in r]
    rows.sort(key=lambda kv: kv[1]["recall_5"], reverse=True)
    base = next((r for s, r in rows if s == "token_flat"), None)
    base_tok = base["mean_pool_tokens"] if base else 0
    out = [
        f"### Comparison — real catalog ({len(real_execs)} tools), "
        f"n={len(corpus)} queries, k_max=8",
        "",
        "| Strategy | Recall@5 | Recall@1 | mean ms | p95 ms | pool tok | tok vs token_flat |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for s, r in rows:
        dv = (f"{100*(r['mean_pool_tokens']-base_tok)/base_tok:+.1f}%"
              if base_tok else "—")
        out.append(f"| `{s}` | **{r['recall_5']:.3f}** | {r['recall_1']:.3f} | "
                   f"{r['mean_ms']:.1f} | {r['p95_ms']:.1f} | "
                   f"{r['mean_pool_tokens']:.0f} | {dv} |")
    return out


def scaling(corpus, real_execs) -> list[str]:
    # pre-build the largest pool, slice down for smaller sizes (stable subset)
    max_pool = max(POOL_SIZES)
    pad = build_hard_distractors(real_execs, max_pool - len(real_execs))
    full = real_execs + pad
    pools = {sz: (full[:sz] if sz <= len(full) else full) for sz in POOL_SIZES}
    data: dict[str, dict[int, float]] = {}
    for s in SCALING_STRATEGIES:
        data[s] = {}
        for sz in POOL_SIZES:
            r = run_strategy(s, corpus, pools[sz], rules=False)
            data[s][sz] = r.get("recall_5", float("nan")) if "error" not in r else float("nan")
    out = [
        f"### Scaling — hard-negative distractors, n={len(corpus)} queries",
        "",
        "| Strategy | " + " | ".join(f"pool {sz}" for sz in POOL_SIZES)
        + " | Δ (84→1000) |",
        "|---|" + "---:|" * (len(POOL_SIZES) + 1),
    ]
    order = sorted(data, key=lambda s: data[s][POOL_SIZES[0]], reverse=True)
    for s in order:
        vals = [data[s][sz] for sz in POOL_SIZES]
        delta = (vals[-1] - vals[0]) * 100
        cells = " | ".join(f"{v:.3f}" for v in vals)
        out.append(f"| `{s}` | {cells} | **{delta:+.1f}pp** |")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["comparison", "scaling", "both"],
                    default="both")
    ap.add_argument("--output", default="")
    args = ap.parse_args()

    print("loading frozen corpus + frozen catalog snapshot…", flush=True)
    corpus = load_snapshot()
    real_execs = load_catalog_snapshot()
    if not dense_available():
        note = ("dense baseline SKIPPED — BGE-M3 embedder not installed "
                "(see bench/README.md 'Reproducing the dense baseline')")
        print(f"  NOTE: {note}", flush=True)
        for lst in (COMPARISON_STRATEGIES, SCALING_STRATEGIES):
            lst[:] = [s for s in lst if not s.startswith("selective_semantic")]
    else:
        note = "dense baseline (BGE-M3) INCLUDED"
    print(f"  {len(corpus)} queries · {len(real_execs)} tools (frozen) · {note}",
          flush=True)

    blocks: list[str] = [
        "# Prefilter strategies — reproducible benchmark",
        "",
        f"Generated by `bench/repro_prefilter_bench.py` · seed={SEED} · "
        "deterministic (no LLM).",
        f"Corpus: `bench/corpus_snapshot.jsonl` "
        f"({len(corpus)} organic queries, PII-scrubbed). "
        f"Catalog: `bench/catalog_snapshot.json` ({len(real_execs)} tools, frozen).",
        f"Dense baseline: {note}.",
        "",
    ]
    if args.mode in ("comparison", "both"):
        print("running comparison…", flush=True)
        blocks += comparison(corpus, real_execs) + [""]
    if args.mode in ("scaling", "both"):
        print("running scaling…", flush=True)
        blocks += scaling(corpus, real_execs) + [""]

    report = "\n".join(blocks) + "\n"
    if args.output:
        Path(args.output).write_text(report, encoding="utf-8")
        print(f"\nreport → {args.output}")
    print("\n" + report)


if __name__ == "__main__":
    main()
