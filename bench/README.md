# Prefilter benchmark — reproduce the numbers

Self-contained, deterministic benchmark behind the write-up *"I deleted the vector DB from
my agent's tool selection. Same recall, none of the cost."* No private data, no LLM, no GPU.

```
python3 bench/repro_prefilter_bench.py --mode comparison   # recall/latency table
python3 bench/repro_prefilter_bench.py --mode scaling      # recall vs pool size (84→1000)
```

Requires only a Python 3.10+ interpreter — that is the point: the production tool-selection
path (`token_flat` + typed rules) reproduces with **nothing to install**.

## What ships here (all frozen, all PII-scrubbed)

| File | What |
|---|---|
| `corpus_snapshot.jsonl` | 234 organic single-user queries; ground truth = the tool prod actually called first |
| `catalog_snapshot.json` | frozen snapshot of the real 96-tool production catalog (name, affinity, description, args) |
| `typing/` | 84 tool typing records the two typed rules read (input-coverage, schema-field) |
| `repro_prefilter_bench.py` | the harness; reads the two snapshots + the real strategy code in `runtime/` |

The strategies themselves are the production ones in `runtime/prefilter*.py` — the bench does
not reimplement them, it calls them. The frozen catalog decouples the numbers from whatever
executors you happen to have installed, so the result is identical on any clone.

## Method honesty (read before quoting a number)

- Ground truth is **prod behaviour, not an oracle**: it measures agreement with what the
  production planner already chose, not correctness. A few labels are genuine prod mis-routes;
  they are not hand-corrected.
- The corpus is **one** single-user assistant's organic traffic (test/e2e/bench turns
  excluded). Absolute recall is in-domain; the interesting quantity is the **slope vs pool
  size** and the **tie with the dense baseline**.
- Deterministic: distractor RNG seeded (42), `prefer_intent=False`, `llm_call=None`.

## Reproducing the dense baseline (optional, heavy)

The `selective_semantic*` rows are the BGE-M3 dense baseline — the thing a skeptic runs to
check the "it's a tie" claim. They need the embedding model, which is **not** pip-trivial and
is intentionally not shipped (it would put a ~half-gigabyte model in a repo whose whole point
is that you don't need one). Without it the bench **skips** those rows and says so, rather
than showing token-flat's silent fallback mislabelled as "dense". To run them, provide the
BGE-M3 embedder via a full Metnos install (`install/`).
