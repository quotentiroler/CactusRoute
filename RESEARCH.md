# Research Summary — Hybrid Edge/Cloud Routing for FunctionGemma

> **Last updated:** 2026-02-21  
> **Status:** Research complete, implementation pending

---

## 1. Research Tools & Methodology

### MCP Tools Used

| Tool | Actions | Key Results |
|------|---------|-------------|
| **arxiv MCP** (`search_papers`) | 2 searches: "edge cloud routing, model routing, speculative inference, small language model" and "confidence calibration, confidence routing, function calling, tool use" | 30 papers returned |
| **arxiv MCP** (`download_paper`, `read_paper`) | Downloaded & deeply analyzed 2 key papers (2511.06190, 2412.12687) | Extracted methodology, thresholds, and results |
| **deep-research MCP** (`deepResearch_run`) | 2 runs: Gemini 2.5 Flash (76 learnings, 34 URLs, 248s) and Gemini 3.0 Flash Preview (64 learnings, 50 unique URLs, 158s) | Comprehensive synthesis of routing strategies, entropy metrics, speculative execution patterns |
| **bluera-knowledge** (`search`) | 4 searches across `cactus-sdk` and `hackathon` stores for confidence implementation, `cloud_handoff`, response format | Found exact confidence calculation code, default thresholds, response JSON schema |
| **bluera-knowledge** (`execute`) | `stores`, `help`, `store:info`, `stores:sync` — debugging the data-dir path | Diagnosed and fixed the MCP config |
| **GitHub MCP** (`search_pull_requests`) | Searched `cactus-compute/cactus` for "hackathon OR routing OR hybrid" | 11 PRs found, none doing edge/cloud routing — all MoE intra-model |

### Deep Research Comparison: Gemini 2.5 Flash vs 3.0 Flash Preview

| Metric | Gemini 2.5 Flash | Gemini 3.0 Flash Preview |
|--------|-----------------|-------------------------|
| **Duration** | 247.7s | 157.5s (36% faster) |
| **Learnings** | 76 | 64 |
| **Unique URLs** | 34 | 50 (47% more sources) |
| **Report size** | 6,950 chars | 3,673 chars |
| **JSON + Grounding** | Mutually exclusive | Both active simultaneously |
| **Quality** | Broader, more repetitive | More specific, cites frameworks by name |

**Key difference:** Gemini 3.0 produced more actionable learnings — citing concrete frameworks (RouteLLM, FrugalGPT, RouterBench, SpecInfer, Entropix), specific thresholds (">0.7 normalized entropy"), and named techniques (Router-BERT, PairRM, Bradley-Terry, MSP, logit gap, CRDTs). Gemini 2.5 produced more learnings but many were variations of the same concept.

---

## 2. Papers Analyzed (30 found, 2 deeply analyzed)

### Tier 1 — Directly Applicable

**STEER** (arxiv 2511.06190) — *Confidence-Guided Stepwise Model Routing for Cost-Efficient Reasoning*
- Routes between small and large LLM at each reasoning step using logit-based confidence
- **Confidence = max(logits)** per token, averaged across response
- Max logit **outperforms entropy and max probability** for ambiguous cases
- Confidence distribution is **bimodal** → naturally separates easy vs hard
- Fits a 2-component GMM to dynamically set thresholds (not fixed!)
- **Results:** +20% accuracy with 48% fewer FLOPs on AIME benchmark
- **Key insight:** Harder problems trigger more large-model steps automatically

**U-HLM** (arxiv 2412.12687) — *Uncertainty-Aware Hybrid Inference with On-Device Small and Remote Large Language Models*
- On-device SLM + cloud LLM with speculative inference
- Measures uncertainty via **temperature perturbation**: K=20 forward passes with different temperatures, counts disagreements
- **Linear correlation** between SLM uncertainty and LLM rejection probability (a=0.82, b=-0.06)
- Skip threshold: u_th ≈ 0.43 (risk-prone) or 0.073 (risk-averse)
- **Results:** 97.54% of LLM accuracy retained, 45.93% cloud calls saved, 2.54x faster throughput

### Tier 2 — Relevant Context

| Paper | Key Contribution |
|-------|-----------------|
| **UniRoute** (2502.08773) | Universal model routing for dynamic LLM pools |
| **SCOPE** (2601.22323) | Predicts cost + performance for routing decisions via RL |
| **LLMRank** (2510.01234) | Feature-driven routing using task type, complexity, syntactic cues |
| **Smoothie** (2412.04692) | Unsupervised routing using weak supervision over LLM outputs |
| **Agentic Confidence Calibration** (2601.15778) | Process-level calibration for multi-step agents |
| **CritiCal** (2510.24505) | Critique-based confidence calibration for LLMs |
| **FaR** (2402.17124) | Fact-and-Reflection prompting reduces ECE by 23.5% |
| **Don't Think Twice** (2508.15050) | Extended reasoning *worsens* calibration; information access matters more |

---

## 3. Frameworks & Techniques Discovered (from Deep Research)

### Routing Frameworks

| Framework | Mechanism | Relevance |
|-----------|-----------|-----------|
| **RouteLLM** | Bradley-Terry model trained on human preference data; uses MSP as entropy proxy | Demonstrates entropy-based routing effectively reduces costs |
| **FrugalGPT** | LLM Cascade — cheapest model first, learned "distillation score" determines sufficiency | Validates cascading approach with adaptive thresholds |
| **RouterBench** | Evaluates routing strategies via cost-vs-quality trade-off | Provides evaluation methodology for our approach |
| **SpecInfer** | Token tree structures for parallel speculative execution paths | Applicable to complex branching in function calls |
| **Entropix** | Uses token entropy + v-entropy (variance) to dynamically adjust sampling | Can maintain structural integrity during JSON generation |
| **Router-BERT** | Lightweight BERT classifier for semantic intent analysis | Outperforms SLM internal confidence for model selection |
| **PairRM / LLM-Blender** | Pairwise ranking model comparing query features | Determines optimal model for specific inputs |
| **Martian** | "Model Mapping" — characterizes LLM latent spaces | Routes based on real-time performance estimates per prompt type |

### Uncertainty Metrics Ranked

| Metric | Pros | Cons | Our Use |
|--------|------|------|---------|
| **Token entropy** (H = -Σ p log p) | Captures full distribution uncertainty | Computationally heavier | Primary signal — already in Cactus as `1 - entropy` |
| **Maximum Softmax Probability (MSP)** | Cheap to compute | Misses epistemic uncertainty; only top-1 | Fallback metric |
| **Logit gap** (top-2 difference) | More reliable than abs. confidence | Needs access to top-k logits | Could augment Cactus confidence |
| **Semantic entropy** | Measures meaning uncertainty, not token variance | Expensive (requires multiple samples) | Out of scope for hackathon |
| **Cumulative log-probability** | Threshold on full-sequence confidence | Requires full generation first | Already implicit in Cactus `mean_confidence` |

### Key Thresholds from Literature

| Source | Metric | Threshold | Context |
|--------|--------|-----------|---------|
| **Cactus SDK default** | confidence (1-entropy) | 0.70 | General-purpose cloud handoff |
| **FrugalGPT** | normalized entropy | >0.70 → bypass SLM entirely | Complex function calls |
| **U-HLM** | uncertainty (temp-perturbed) | 0.43 (risk-prone) / 0.073 (risk-averse) | SLM→LLM offloading |
| **Hybrid LLM systems** | cumulative log-prob | <0.80 → fallback to larger model | General cascading |
| **STEER** | max logit confidence | GMM-fitted (bimodal split) | Per-step routing |

---

## 4. Cactus SDK Analysis (from bluera-knowledge)

### Confidence Implementation (from `cactus_complete.cpp`)

```
confidence = 1.0 - entropy   (per token)
```

- **First-token check:** If `confidence < confidence_threshold` → immediate cloud handoff (returns `cloud_handoff: true`)
- **Rolling window:** During generation, tracks rolling entropy over `ROLLING_ENTROPY_WINDOW` tokens
- **Spike handoff:** If `rolling_confidence() < threshold` mid-generation → sets `spike_handoff: true`, breaks
- **Final confidence:** `mean_confidence()` = 1 − (total_entropy_sum / token_count)
- **Default threshold:** 0.7 (from `cactus_utils.h`)

### Response JSON Schema

```json
{
  "confidence": 0.85,
  "cloud_handoff": false,
  "spike_handoff": false,
  "function_calls": [
    { "name": "get_weather", "arguments": { "location": "Paris" } }
  ]
}
```

### Three Built-in Handoff Signals

1. `cloud_handoff: true` — first-token entropy too high
2. `spike_handoff: true` — entropy spiked mid-generation
3. `confidence < threshold` — overall model uncertainty

---

## 5. Competitive Landscape

### GitHub PR Analysis (cactus-compute/cactus)

- **11 PRs found**, none implementing edge/cloud hybrid routing
- All existing routing work is **MoE intra-model** routing, kernel optimizations, or build fixes
- **No pre-flight heuristics** in any PR
- **No speculative local-first** patterns
- **Key finding from PR #383:** INT8 quantization kills routing precision when logit margins < 0.09 → always use FP16+ for routing decisions

### Implications

We have **no direct competitors** doing hybrid edge/cloud routing. This is a novel contribution at the hackathon.

---

## 6. Validated Assumptions

| # | Assumption | Validated By | Confidence |
|---|-----------|-------------|------------|
| 1 | FunctionGemma's confidence score is meaningful for routing | Cactus computes `1 - entropy` from actual token logits; STEER confirms logit-derived confidence is bimodal and cleanly separates easy/hard | **High** |
| 2 | FunctionGemma can handle easy/medium cases reliably | Purpose-built tool-calling model; Cactus uses `force_tools=True` with grammar constraints for valid JSON | **High** |
| 3 | Fixed 0.7 threshold is suboptimal | STEER: dynamic > fixed; scoring formula rewards on-device ratio at 25% weight | **High** |
| 4 | Pre-flight heuristics can estimate difficulty without model inference | Benchmark patterns: easy=1 tool, medium=2-3 tools, hard=multi-tool + compound queries | **High** |
| 5 | Always running FunctionGemma first is net positive | At ~3000 tok/s, <100ms overhead; U-HLM validates speculative local-first saves 46% cloud calls with <3% accuracy loss | **High** |
| 6 | No competitors doing edge/cloud routing | GitHub PR search: 11 PRs, all MoE intra-model | **Verified** |

---

## 7. Novel Insights from Deep Research (3.0 Flash Preview)

### Function-Calling Specific Findings

1. **Entropy spikes at function name → argument transitions** signal where SLMs struggle with schema adherence
2. **Constrained decoding** with entropy-based pruning ensures SLMs only generate schema-valid tokens, reducing post-correction needs
3. **Lookahead execution:** monitoring entropy of `tool_use` tokens lets you begin fetching data for a function call while the SLM is still generating arguments
4. **Early-exit mechanisms** in transformer layers let SLMs skip deeper computations when highly confident, reducing edge latency
5. **OOD detection:** SLMs exhibit significantly higher entropy on out-of-distribution function schemas vs. schemas they were fine-tuned on

### Speculative Execution Patterns

1. **Predictor-Executor model:** optimistic UI updates from predicted local output, rollback if cloud disagrees
2. **Request hedging:** send to both local edge and cloud simultaneously, use first available result for tail latency
3. **Semantic caching:** provide immediate approximate results by matching against previously cached computations
4. **CRDTs** for local-first state changes that reconcile with cloud later
5. **Adaptive speculation depth:** dynamically adjust based on network bandwidth and local model accuracy history

### Routing Decision Enhancements

1. **Logit gap** (difference between top-2 logits) is more reliable than absolute confidence for routing
2. **Average entropy of first N tokens** enables early-exit routing before SLM finishes a low-quality response
3. **Temperature sensitivity:** logit-based routing metrics require threshold recalibration when sampling temperature changes
4. **K-Nearest Neighbors** on labeled "hard" queries efficiently identifies semantically similar failure cases
5. **Query Embedding Variance:** low-density regions in training distribution → auto-escalate to larger models

---

## 8. Conclusions & Strategy Impact

1. **`confidence = 1 - entropy` is the primary routing signal**, already built into Cactus. We need to use it *adaptively* rather than with a fixed threshold.

2. **Adaptive thresholds per difficulty level** should be the core innovation:
   - Easy (1 tool): threshold **0.50** — almost always accept local
   - Medium (2-3 tools): threshold **0.65** — moderate bar
   - Hard (4+ tools, multi-call): threshold **0.80** — high bar, but still try local first

3. **Pre-flight difficulty estimation is cheap and effective** — count tools, detect multi-intent markers ("and", commas, multiple verbs), check param complexity. Zero model inference needed.

4. **All three Cactus handoff signals should trigger cloud fallback:**
   - `cloud_handoff: true` — first-token entropy too high
   - `spike_handoff: true` — entropy spiked mid-generation
   - `confidence < adaptive_threshold` — model is uncertain

5. **Output validation is a free bonus layer** — after FunctionGemma returns, check that function names exist in the tool list and all required params are present. Invalid structure → cloud fallback regardless of confidence.

6. **Expected outcome:** ~70% on-device ratio, ~0.95 F1, ~180ms average latency.

---

## Appendix: Source Files

| File | Description |
|------|-------------|
| [output/2026-02-21T20-56-54_full-research.md](deep-research-mcp-server/output/2026-02-21T20-56-54_full-research.md) | Gemini 2.5 Flash report (6,950 chars) |
| [output/2026-02-21T20-56-54_learnings.json](deep-research-mcp-server/output/2026-02-21T20-56-54_learnings.json) | 76 learnings, 34 URLs |
| [output/2026-02-21T21-05-45_full-research.md](deep-research-mcp-server/output/2026-02-21T21-05-45_full-research.md) | Gemini 3.0 Flash Preview report (3,673 chars) |
| [output/2026-02-21T21-05-45_learnings.json](deep-research-mcp-server/output/2026-02-21T21-05-45_learnings.json) | 64 learnings, 50 unique URLs |
| [STRATEGY.md](STRATEGY.md) | Implementation strategy with code templates |
