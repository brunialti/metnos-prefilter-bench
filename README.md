# metnos-prefilter-bench

Reproducible benchmark for **closed-vocabulary tool selection** in an LLM agent —
the harness behind the write-up *"I deleted the vector DB from my agent's tool
selection. Same recall, none of the cost."*

The claim: when your tools are named as a **closed compositional grammar**
(`verb_object[_qualifier]`), a plain CPU token-matcher plus a few typed rules
selects tools **as well as a BGE-M3 dense baseline** — with no embedding model, no
vector index, no GPU. This repo lets you check that yourself.

## Run it

```bash
git clone https://github.com/brunialti/metnos-prefilter-bench.git
cd metnos-prefilter-bench
python3 bench/repro_prefilter_bench.py --mode comparison   # recall / latency table
python3 bench/repro_prefilter_bench.py --mode scaling      # recall vs pool size, 84→1000
```

Python 3.10+, **no dependencies**, no model download, deterministic (the harness pins
the hash seed and ships a frozen catalog). A clean clone prints:

```
token_flat+rules (PRODUCTION)   R@5 0.786   R@1 0.487   ~10 ms   no model
```

The dense baseline (BGE-M3) is optional and skipped by default — see
[`bench/README.md`](bench/README.md). It never pulls ahead of the lexical path.

## What's in here

| Path | What |
|---|---|
| `bench/` | the harness + frozen, PII-scrubbed corpus (234 queries) + frozen 96-tool catalog + tool typing |
| `runtime/` | the **real** production prefilter and strategies (the bench calls them, doesn't reimplement) |
| `executors/*/manifest.toml` | the 74 tool manifests — the closed-vocabulary catalog itself |

This is a curated, self-contained slice of a larger self-hosted assistant (Metnos);
it is the subset needed to reproduce the benchmark, nothing more.

## Method honesty

Ground truth is production behaviour, not an oracle (agreement with what the live
planner chose, not correctness). The corpus is one single-user assistant's organic
traffic. Read `bench/README.md` before quoting a number.
