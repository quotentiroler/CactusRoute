# CactusRoute — 7-Layer Adaptive Edge/Cloud Hybrid Router

> **FunctionGemma 270M (on-device) ↔ Gemini 2.5 Flash (cloud)**  
> Schema-driven adaptive routing for function calling — backed by 8 arXiv papers

---

## What is this?

A hybrid inference strategy for the [FunctionGemma Hackathon](https://github.com/cactus-compute/functiongemma-hackathon) that dynamically routes tool-calling queries between a 270M on-device model (FunctionGemma via [Cactus](https://github.com/cactus-compute/cactus)) and Gemini 2.5 Flash in the cloud.

Instead of using a fixed confidence threshold (the baseline uses 0.99, routing nearly everything to cloud), CactusRoute uses a **7-layer schema-driven adaptive framework** with output repair, semantic validation, deterministic extraction, retry with prompt variation, and per-difficulty adaptive thresholds — every technique grounded in peer-reviewed research.

## Architecture

```
User Query
    │
    ▼
┌──────────────────────────────────────────┐
│  Layer 1: Pre-flight Difficulty          │  Zero-cost heuristic: tool count +
│  Estimation  (easy / medium / hard)      │  multi-intent markers ("and", commas)
│  ↳ ODIA (2507.08877)                     │  Backed by: simple/complex routing
└───────────────────┬──────────────────────┘
                    │
                    ▼
┌──────────────────────────────────────────┐
│  FunctionGemma  (On-Device, 270M)        │  Always runs first (~50-100ms)
│  force_tools=True, constrained JSON      │  Speculative local-first approach
│  ↳ TinyAgent (2409.00608)                │  Backed by: SLM ≥ GPT-4-Turbo
│  ↳ Hammer (2410.04587)                   │  Backed by: description-aware calling
└───────────────────┬──────────────────────┘
                    │
     ┌──────────────┴──────────────┐
     │  Layer 2: Handoff Signals   │
     │  cloud_handoff (1st token)  │──→ catastrophic entropy → Layer 7
     │  spike_handoff (mid-gen)    │──→ entropy spike → Layer 7
     │  ↳ STEER (2511.06190)       │
     │  ↳ U-HLM (2412.12687)      │
     └──────────────┬──────────────┘
                    │ (generation succeeded)
                    ▼
     ┌──────────────────────────────┐
     │  Layer 3: Output Repair     │  AM/PM hour correction, negative fix,
     │  repair_output()            │  semantic mismatch fill, type coercion
     │  ↳ Hybrid-Code (2512.23743) │  Backed by: format normalization
     └──────────────┬──────────────┘
                    │
                    ▼
     ┌──────────────────────────────┐
     │  Layer 4: Multi-Gate        │  A. Structural: tool names + required params
     │  Validation                 │  B. Semantic: word-overlap + integer ranges
     │  validate_output()          │  C. Intent coverage: expected vs actual calls
     │  semantic_validate()        │
     │  ↳ PARSE (2510.08623)       │  Backed by: reflection-based guardrails
     │  ↳ Hammer (2410.04587)      │  Backed by: description-aware validation
     │  ↳ ToolRM (2510.26167)      │  Backed by: rule-based scoring
     └──────────────┬──────────────┘
                    │
                    ▼
     ┌──────────────────────────────┐
     │  Layer 5: Adaptive Conf.    │  easy=0.25  medium=0.45  hard=0.60
     │  Thresholds                 │  Dynamic > fixed (bimodal distribution)
     │  ↳ STEER (2511.06190)       │  Backed by: GMM-fitted confidence
     │  ↳ ODIA (2507.08877)        │  Backed by: simple/complex routing
     └──────┬───────────┬──────────┘
            │ PASS      │ FAIL
            ▼           ▼
     ┌──────────┐ ┌────────────────────┐
     │  ACCEPT  │ │  Layer 6: Retry    │  Alternate system prompt
     │  Local   │ │  with Prompt       │  Full re-validation pipeline
     │  Result  │ │  Variation         │
     └──────────┘ │  ↳ PARSE           │  Backed by: 92% error reduction
                  │  ↳ ToolRM          │  Backed by: self-correction +11.4pts
                  └──────┬─────────────┘
                         │ (retry also failed)
                         ▼
                  ┌────────────────────┐
                  │  Layer 7: Determ.  │  Schema-driven regex extraction
                  │  Extraction +     │  from raw user text; segment
                  │  Cloud Fallback   │  decomposition for multi-call
                  │  ↳ Hybrid-Code    │  Backed by: keyword fallback
                  │  ↳ TinyAgent      │  Backed by: Tool RAG patterns
                  └──────┬─────────────┘
                         │ (extraction failed)
                         ▼
                  ┌────────────────────┐
                  │  Gemini 2.5 Flash  │  Cloud fallback (last resort)
                  │  Cloud Endpoint    │
                  └────────────────────┘
```

## The 7 Layers Explained

| Layer | Function | Research Backing |
|-------|----------|-----------------|
| **1. Difficulty estimation** | Classifies query as easy/medium/hard via tool count + NLP markers — zero model inference | ODIA (2507.08877): ByteDance's simple/complex routing handles 60% of traffic with small model |
| **2. Handoff signals** | Cactus `cloud_handoff` (1st token entropy) and `spike_handoff` (mid-generation entropy spike) | STEER (2511.06190): logit confidence is bimodal → clean separation; U-HLM (2412.12687): speculative local-first saves 46% cloud calls |
| **3. Output repair** | AM/PM hour correction, negative value fix, semantic mismatch fill, type coercion | Hybrid-Code (2512.23743): "format normalization" auto-corrects LLM output errors; 0% hallucination rate |
| **4. Multi-gate validation** | Structural (tool names + required params) + semantic (word-overlap + integer range) + intent coverage | PARSE (2510.08623): reflection-based guardrails; Hammer (2410.04587): description-aware validation; ToolRM (2510.26167): rule-based scoring |
| **5. Adaptive thresholds** | Per-difficulty confidence bars: easy=0.25, medium=0.45, hard=0.60 | STEER: dynamic > fixed thresholds with GMM-fitted bimodal distribution; ODIA: difficulty-based routing proven in production |
| **6. Retry with prompt variation** | Second on-device attempt with alternate system prompt; full re-validation | PARSE: 92% error reduction within first retry; ToolRM: self-correction yields +11.4 accuracy points |
| **7. Deterministic extraction** | Schema-driven regex parsing from raw text; segment decomposition for multi-call; cloud fallback as last resort | Hybrid-Code: "reliability through redundancy"; TinyAgent (2409.00608): 1.1B model exceeds GPT-4-Turbo via structured extraction |

## Semantic Role System

The framework uses 11 semantic roles to map tool parameters to extraction strategies:

| Role | Extraction Strategy | Example |
|------|--------------------|---------|
| `ROLE_HOUR` | Time regex: `at 3pm`, `3:00` | `set_alarm(hour=15)` |
| `ROLE_MINUTE` | Time regex: `3:30`, `half past` | `set_alarm(minute=30)` |
| `ROLE_DURATION` | Duration regex: `10 minutes`, `1 hour` | `set_timer(duration=10)` |
| `ROLE_LOCATION` | Location patterns: `in Paris`, `weather for NYC` | `get_weather(location="Paris")` |
| `ROLE_PERSON` | Proper name patterns: `send Bob`, `to Alice` | `send_message(contact="Bob")` |
| `ROLE_MESSAGE` | Message patterns: `saying "hello"`, `"meet me"` | `send_message(message="hello")` |
| `ROLE_TITLE` | Reminder patterns: `to buy milk` | `create_reminder(title="buy milk")` |
| `ROLE_SONG` | Play patterns: `play Bohemian Rhapsody` | `play_music(song="...")` |
| `ROLE_QUERY` | Search patterns: `find contact`, `search for` | `search_contacts(query="...")` |
| `ROLE_TIME_STR` | Full time string: `at 3pm`, `3:00 PM` | `set_alarm(time="3:00 PM")` |
| `ROLE_UNKNOWN` | Cloud fallback — cannot extract deterministically | — |

## Scoring Formula

$$\text{Score} = \sum_{d} w_d \cdot (0.60 \cdot F1 + 0.15 \cdot \text{TimeScore} + 0.25 \cdot \text{OnDeviceRatio})$$

Difficulty weights: easy=20%, medium=30%, **hard=50%**.

Our 7-layer framework maximizes all three components: high F1 through multi-gate validation and repair, low latency through local-first execution, and high on-device ratio through adaptive thresholds + retry + deterministic extraction.

## Project Structure

```
cactus-hack/
├── README.md                  ← You are here
├── RESEARCH.md                ← 83 papers searched, 8 deeply analyzed, 140+ learnings
├── STRATEGY.md                ← Detailed strategy with research findings
│
├── functiongemma-hackathon/   ← Hackathon submission
│   ├── main.py                ← 7-layer adaptive router (~1010 lines)
│   ├── benchmark.py           ← Official benchmark (30 cases: 10 easy/10 med/10 hard)
│   ├── submit.py              ← Leaderboard submission script
│   ├── demo.py                ← Rich interactive demo (4 modes)
│   └── tests.py               ← 121 unit tests, 10 test classes (any platform)
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

### Prerequisites
- [uv](https://docs.astral.sh/uv/) (Python package manager)
- Mac with Cactus SDK for benchmark/demo (tests run anywhere)
- `GEMINI_API_KEY` environment variable

### Setup
```bash
cd functiongemma-hackathon
uv sync
```

### Run Tests (any platform)
```bash
uv run python tests.py -v     # 121 tests, 10 classes, ~0.01s, no Cactus needed
```

### Run Benchmark (Mac only — requires Cactus)
```bash
export GEMINI_API_KEY="your-key"
uv run python benchmark.py
```

### Run Demo (Mac only — requires Cactus)
```bash
uv run python demo.py                # Curated scenarios with dashboard
uv run python demo.py --interactive  # Free-form text input
uv run python demo.py --voice        # Voice-to-action via Whisper
uv run python demo.py --compare      # Baseline vs CactusRoute side-by-side
uv run python demo.py --benchmark    # Full 30-case benchmark run
```

### Submit to Leaderboard
```bash
uv run python submit.py --team "YourTeamName" --location "YourCity"
```

## Key Optimizations Over Baseline

| # | Optimization | Research Backing | Impact |
|---|---|---|---|
| 1 | **Model singleton** — load once, reuse | — | Saves ~7-15s across 30 benchmark calls |
| 2 | **Pre-flight difficulty** — tool count + NLP heuristics | ODIA (ByteDance) | Zero-cost routing signal |
| 3 | **Adaptive thresholds** — 0.25 / 0.45 / 0.60 | STEER, FrugalGPT | Maximizes on-device without sacrificing F1 |
| 4 | **Schema-driven output repair** — AM/PM, negatives, mismatches | Hybrid-Code | Rescues otherwise-rejected local outputs |
| 5 | **Semantic validation** — word overlap + range checks | PARSE, Hammer | Catches hallucinated parameters |
| 6 | **Role-based extraction** — 11 semantic roles mapped to regex | PARSE (ARCHITECT) | Deterministic fallback for on-device |
| 7 | **Retry with prompt variation** — alternate system prompt | PARSE (92% 1st retry), ToolRM (+11.4pts) | Cheap second chance on-device |
| 8 | **Deterministic extraction** — schema-driven text parsing | Hybrid-Code (keyword fallback) | Extracts calls without any LLM |
| 9 | **Intent coverage augmentation** — fills missing calls | TinyAgent (LLMCompiler) | Catches incomplete multi-call output |
| 10 | **Type coercion** — string→int based on schema | ToolRM (argument similarity) | `"10"` ≠ `10` in F1 comparator |
| 11 | **`tool_rag_top_k=0`** — use ALL tools | TinyAgent (Tool RAG) | Default=2 misses needed tools |
| 12 | **Dynamic system prompt** — multi-call instruction for hard queries | — | "Call ALL relevant tools" |
| 13 | **Cloud model fix** — `gemini-2.5-flash` | — | Baseline's `gemini-2.0-flash` is deprecated |

## Test Suite

121 tests across 10 test classes — runs on any platform, no Cactus or API keys needed:

| Test Class | Tests | What it covers |
|-----------|-------|----------------|
| `TestEstimateDifficulty` | 9 | Tool count + multi-intent classification |
| `TestCountExpectedIntents` | 7 | NLP-based intent counting |
| `TestCoerceArgTypes` | 8 | Schema-driven type coercion |
| `TestValidateOutput` | 6 | Structural validation (names + params) |
| `TestInferParamRole` | 12 | Semantic role inference from schema |
| `TestExtractForRole` | 23 | Regex extraction for all 11 roles |
| `TestSemanticValidate` | 6 | Word-overlap + range validation |
| `TestRepairOutput` | 7 | AM/PM, negatives, semantic repair |
| `TestBuildCallsFromText` | 9 | Deterministic extraction pipeline |
| `TestRoutingDecisions` | 21 | End-to-end routing with thresholds |
| + 13 more | 13 | Threshold boundaries, signal priority, benchmark compat |

## Research

### Phase 1: Routing Fundamentals
- **arxiv MCP** — 30 papers on edge inference, model routing, confidence calibration
- **deep-research MCP** — Two full runs: Gemini 2.5 Flash (76 learnings, 34 URLs, 248s) and Gemini 3.0 Flash Preview (64 learnings, 50 URLs, 158s)
- **bluera-knowledge MCP** — Cactus SDK source analysis (confidence calculation, handoff signals)
- **GitHub MCP** — Competitive landscape (158 forks analyzed, 3 implementations read)

### Phase 2: 7-Layer Technique Validation
- **arxiv MCP** — 53 additional papers; 6 deeply analyzed and cited throughout implementation:
  - **PARSE** (2510.08623) — Schema optimization + reflection-based guardrails → validates our `infer_param_role()` + `semantic_validate()`
  - **Hybrid-Code** (2512.23743) — 3-tier neuro-symbolic framework → validates our LLM → extraction → verification pipeline
  - **TinyAgent** (2409.00608) — 1.1B model exceeds GPT-4-Turbo on function calling → validates SLM-first approach
  - **Hammer** (2410.04587) — Function masking for description-aware calling → validates `extract_for_role()` semantic patterns
  - **ODIA** (2507.08877) — Simple/complex query routing, 78% latency reduction → validates `estimate_difficulty()`
  - **ToolRM** (2510.26167) — Tool-use reward modeling with self-correction → validates `repair_output()` + retry mechanism

### Totals
- **83 papers searched**, **8 deeply analyzed**, **140+ learnings extracted**
- Every layer of the 7-layer framework is backed by at least one peer-reviewed paper

See [RESEARCH.md](RESEARCH.md) for the full synthesis.

## License

Hackathon project — see [cactus-compute/functiongemma-hackathon](https://github.com/cactus-compute/functiongemma-hackathon) for terms.
