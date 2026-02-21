# CactusRoute — Adaptive Edge/Cloud Hybrid Router

> **FunctionGemma 270M (on-device) ↔ Gemini 2.5 Flash (cloud)**
> Multi-signal adaptive routing for function calling

---

## What is this?

A hybrid inference strategy for the [FunctionGemma Hackathon](https://github.com/cactus-compute/functiongemma-hackathon) that dynamically routes tool-calling queries between a 270M on-device model (FunctionGemma via [Cactus](https://github.com/cactus-compute/cactus)) and Gemini 2.5 Flash in the cloud.

Instead of using a fixed confidence threshold (the baseline uses 0.99, routing nearly everything to cloud), CactusRoute uses a **4-signal decision framework** with adaptive thresholds calibrated per difficulty level.

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────┐
│  Pre-flight Difficulty      │  Zero-cost heuristic:
│  Estimation                 │  tool count + multi-intent markers
│  easy / medium / hard       │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  FunctionGemma (On-Device)  │  Always runs first (~50-100ms)
│  270M params, 3000 tok/s    │  force_tools=True, constrained JSON
└─────────────┬───────────────┘
              │
    ┌─────────┴──────────┐
    │  4-Signal Router   │
    │                    │
    │  A. cloud_handoff  │─── first-token entropy catastrophic
    │  B. spike_handoff  │─── entropy spiked mid-generation
    │  C. confidence     │─── below adaptive threshold
    │  D. validation     │─── invalid tools / missing params / intent gap
    │                    │
    └────┬──────────┬────┘
         │          │
    ┌────┴───┐ ┌────┴────────────────┐
    │ Accept │ │ Gemini 2.5 Flash    │
    │ Local  │ │ Cloud Fallback      │
    └────────┘ └─────────────────────┘
```

## Adaptive Thresholds

| Difficulty | Threshold | Rationale |
|------------|-----------|-----------|
| **Easy** (1 tool) | 0.25 | Can't pick wrong tool name → nearly always on-device |
| **Medium** (2-5 tools, 1 action) | 0.45 | Need correct tool selection → moderate bar |
| **Hard** (multi-call) | 0.60 | Multiple actions → higher bar, still try local first |

## Scoring Formula

The hackathon scores submissions as:

$$\text{Score} = \sum_{d} w_d \cdot (0.60 \cdot F1 + 0.15 \cdot \text{TimeScore} + 0.25 \cdot \text{OnDeviceRatio})$$

Difficulty weights: easy=20%, medium=30%, **hard=50%**.

Our strategy maximizes all three components simultaneously — high F1 through validation, low latency through local-first execution, and high on-device ratio through adaptive thresholds.

## Project Structure

```
cactus-hack/
├── README.md                  ← You are here
├── RESEARCH.md                ← 140 learnings from deep research (2.5 + 3.0 Flash)
├── STRATEGY.md                ← Detailed strategy with research findings
│
├── functiongemma-hackathon/   ← Hackathon submission
│   ├── main.py                ← CactusRoute implementation (the submission)
│   ├── benchmark.py           ← Official benchmark (30 cases: 10 easy/10 med/10 hard)
│   ├── submit.py              ← Leaderboard submission script
│   ├── demo.py                ← Rich interactive demo (scenarios/voice/compare)
│   └── tests.py               ← 57 unit tests (runs on any platform, no Cactus needed)
│
├── deep-research-mcp-server/  ← Deep research pipeline (Gemini-powered)
│   ├── src/                   ← TypeScript source
│   └── output/                ← Research outputs (learnings JSON + reports)
│
├── cactus/                    ← Cactus SDK (git submodule)
│   ├── python/                ← Python bindings
│   └── weights/               ← Model weights (downloaded via cactus CLI)
│
└── papers/                    ← Saved research papers
```

## Quick Start

### Run Tests (any platform)
```bash
cd functiongemma-hackathon
python tests.py -v            # 57 tests, 0.006s, no dependencies
```

### Run Benchmark (Mac only — requires Cactus)
```bash
cd functiongemma-hackathon
export GEMINI_API_KEY="your-key"
python benchmark.py
```

### Run Demo (Mac only — requires Cactus)
```bash
python demo.py                # Curated scenarios with dashboard
python demo.py --interactive  # Free-form text input
python demo.py --voice        # Voice-to-action via Whisper
python demo.py --compare      # Baseline vs CactusRoute side-by-side
python demo.py --benchmark    # Full 30-case benchmark run
```

### Submit to Leaderboard
```bash
python submit.py --team "YourTeamName" --location "YourCity"
```

## Key Optimizations Over Baseline

| # | Optimization | Impact |
|---|---|---|
| 1 | **Model singleton** — load once, reuse | Saves ~7-15s across 30 benchmark calls |
| 2 | **Pre-flight difficulty** — tool count + NLP heuristics | Zero-cost routing signal |
| 3 | **Adaptive thresholds** — 0.25 / 0.45 / 0.60 | Maximizes on-device without sacrificing F1 |
| 4 | **5 routing signals** — handoff, spike, confidence, validation, intent coverage | Multi-signal > single threshold |
| 5 | **Type coercion** — string→int based on schema | `"10"` ≠ `10` in F1 comparator |
| 6 | **Output validation** — tool names + required params | Free structural check |
| 7 | **Intent coverage** — count expected vs actual calls | Catches incomplete multi-call output |
| 8 | **`tool_rag_top_k=0`** — use ALL tools | Default=2 misses needed tools |
| 9 | **Dynamic system prompt** — multi-call instruction for hard queries | "Call ALL relevant tools" |
| 10 | **Cloud model fix** — `gemini-2.5-flash` | Baseline's `gemini-2.0-flash` is deprecated (404) |

## Research

Research was conducted using:
- **arxiv MCP** — 30 papers on edge inference, model routing, confidence calibration
- **deep-research MCP** — Two full runs: Gemini 2.5 Flash (76 learnings, 34 URLs, 248s) and Gemini 3.0 Flash Preview (64 learnings, 50 URLs, 158s)
- **bluera-knowledge MCP** — Cactus SDK source analysis (confidence calculation, handoff signals)
- **GitHub MCP** — Competitive landscape (11 PRs, none doing edge/cloud routing)

Key papers: **STEER** (2511.06190), **U-HLM** (2412.12687), **RouteLLM**, **FrugalGPT**, **RouterBench**, **Entropix**

See [RESEARCH.md](RESEARCH.md) for the full synthesis of 140 learnings.

## License

Hackathon project — see [cactus-compute/functiongemma-hackathon](https://github.com/cactus-compute/functiongemma-hackathon) for terms.
