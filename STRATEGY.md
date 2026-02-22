# FunctionGemma Hackathon Strategy

## The Challenge
Design a hybrid routing strategy that decides when to use **FunctionGemma** (on-device, 270M params, ~3000 tok/s prefill) vs. **Gemini Flash** (cloud). Optimize for:
- Tool-call correctness (F1)
- Speed (latency)
- Edge/cloud ratio (prioritize local)

## Scoring Formula (from benchmark.py)

**Per-difficulty level:**
- F1 score: **60% weight** (accuracy of tool calls)
- Time score: **15% weight** (faster = better, capped at 500ms baseline: `max(0, 1 - avg_time/500)`)
- On-device ratio: **25% weight** (higher local = better)

**Difficulty weights:** easy=20%, medium=30%, hard=50%

**Implication:** Hard multi-tool calls are 50% of total score. We MUST get those right, even if it means using cloud. But keeping them on-device at high F1 is the jackpot.

---

## Research Setup (Done)
- **arxiv MCP** — searched edge inference, model routing, confidence calibration
- **deep-research MCP** — Gemini-powered multi-step research synthesis
- **bluera-knowledge** — indexed Cactus SDK + hackathon repo (2 stores, working)

---

## Key Research Findings

### Paper 1: STEER (arxiv 2511.06190) — Confidence-Guided Stepwise Model Routing
- Routes between small/large model at each **reasoning step** using logit confidence
- **Confidence = max(logits)** per token, averaged across response
- Max logit **outperforms entropy and max probability** for ambiguous cases
- Confidence distribution is **bimodal** → naturally separates easy vs hard
- Uses 2-component GMM to dynamically set threshold (not fixed!)
- **Results:** +20% accuracy with 48% fewer FLOPs on AIME benchmark
- **Key insight:** Harder problems trigger more large-model steps automatically

### Paper 2: U-HLM (arxiv 2412.12687) — Uncertainty-Aware Hybrid Inference
- On-device SLM + cloud LLM with speculative inference
- Measures uncertainty via **temperature perturbation**: run K=20 forward passes with different temperatures, count disagreements
- **Linear correlation** between SLM uncertainty and LLM rejection probability (a=0.82, b=-0.06)
- Skip threshold: u_th ≈ 0.43 (risk-prone) or 0.073 (risk-averse)
- **Results:** 97.54% of LLM accuracy retained, 45.93% cloud calls saved, 2.54x faster throughput

### Cactus SDK Confidence Implementation (from cactus_complete.cpp)
```
confidence = 1.0 - entropy  (per token)
```
- **First-token check:** If `confidence < confidence_threshold` → immediate cloud handoff (returns `cloud_handoff: true`)
- **Rolling window:** During generation, tracks rolling entropy over `ROLLING_ENTROPY_WINDOW` tokens
- **Spike handoff:** If `rolling_confidence() < threshold` mid-generation → sets `spike_handoff: true`, breaks
- **Final confidence:** `mean_confidence()` = 1 - (total_entropy_sum / token_count)
- **Default threshold:** 0.7 (from `cactus_utils.h`)
- The response JSON includes: `confidence`, `cloud_handoff`, `spike_handoff`, `function_calls`

### Competitive Landscape (GitHub PRs)
- No one is doing edge/cloud routing — all existing PRs are MoE intra-model routing
- **No pre-flight heuristics** in any PR
- **No speculative local-first** patterns
- Key finding from PR #383: INT8 quantization kills routing precision when logit margins < 0.09 → always use FP16+ for routing decisions

---

## Strategy: Multi-Signal Adaptive Router

### Phase 1: Pre-flight Heuristics (before any model call)
Estimate difficulty from the raw query + tool definitions:

| Signal | Easy | Medium | Hard |
|--------|------|--------|------|
| Number of tools | 1 | 2-3 | 4+ |
| Required params | 1 | 2 | 3+ |
| Query word count | <10 | 10-20 | 20+ |
| Multi-intent markers | none | none | "and", "also", commas |
| Param ambiguity | exact match | fuzzy | requires inference |

**Pre-flight decision:**
- If heuristic says **easy** → run FunctionGemma, accept with any confidence > 0.5
- If heuristic says **hard** → lower confidence threshold (still try local first)

### Phase 2: Speculative Local-First Execution
1. **Always run FunctionGemma first** (it's fast: ~3000 tok/s prefill)
2. Read the confidence from the response
3. Apply **adaptive threshold** based on pre-flight difficulty estimate:

| Difficulty | Confidence Threshold | Rationale |
|------------|---------------------|-----------|
| Easy (1 tool) | 0.25 | Almost always correct, maximize on-device ratio |
| Medium (2-3 tools) | 0.45 | Need moderate confidence for tool selection |
| Hard (4+ tools, multi-call) | 0.60 | High bar before trusting complex multi-tool output |

### Phase 3: Validation & Fallback
- If confidence >= threshold → **return on-device result** (fast, free)
- If confidence < threshold → **fall back to Gemini Flash** (accurate, costly)
- If `cloud_handoff: true` from Cactus → **always route to cloud** (model itself is uncertain)
- If `spike_handoff: true` → **always route to cloud** (entropy spiked mid-generation)

### Phase 4: Output Validation (bonus)
- Parse the function call JSON from FunctionGemma
- Check: Are all required params present? Are types correct? Is the function name in the tool list?
- If validation fails → route to cloud regardless of confidence

---

## Implementation Plan

### `generate_hybrid()` rewrite:
```python
def generate_hybrid(messages, tools, confidence_threshold=0.99):
    # Phase 1: Pre-flight heuristics
    difficulty = estimate_difficulty(messages, tools)
    threshold = get_adaptive_threshold(difficulty)
    
    # Phase 2: Speculative local execution
    local = generate_cactus(messages, tools)
    
    # Phase 3: Decision
    if local.get("cloud_handoff") or local.get("spike_handoff"):
        return generate_cloud_fallback(local, messages, tools)
    
    if local["confidence"] >= threshold:
        # Phase 4: Validate output structure
        if validate_tool_calls(local["function_calls"], tools):
            local["source"] = "on-device"
            return local
    
    return generate_cloud_fallback(local, messages, tools)
```

### Key helper functions:
- `estimate_difficulty(messages, tools)` → "easy" | "medium" | "hard"
- `get_adaptive_threshold(difficulty)` → float
- `validate_tool_calls(calls, tools)` → bool (check names, required params, types)

---

## Expected Performance

| Difficulty | On-device % | F1 (est.) | Avg Latency |
|------------|------------|-----------|-------------|
| Easy | ~100% | ~1.0 | <100ms |
| Medium | ~80% | ~0.95 | <150ms |
| Hard | ~40% | ~0.90 | ~300ms (mix) |
| **Overall** | **~70%** | **~0.95** | **~180ms** |

## Next Steps
- [x] Search arxiv for papers on edge/cloud routing, model confidence calibration
- [x] Deep research on hybrid inference strategies for small LLMs
- [x] Analyze Cactus SDK to understand how confidence scores are calculated
- [x] Study the benchmark.py scoring formula
- [x] Review existing PRs to see what others are trying
- [x] Fix Bluera Knowledge MCP data-dir path
- [x] Implement `estimate_difficulty()` pre-flight heuristic
- [x] Implement adaptive threshold routing in `generate_hybrid()`
- [x] Add output validation for function call structure
- [x] Add type coercion for integer/number params (critical for F1)
- [x] Add intent coverage check for hard multi-call queries
- [x] Global model singleton (eliminates ~7s load overhead across 30 calls)
- [x] Dynamic system prompt for multi-call queries
- [x] tool_rag_top_k=0 to consider ALL tools (default 2 misses tools)
- [x] Build rich interactive demo (demo.py) with scenarios + dashboard
- [x] Voice-to-action mode via Cactus Whisper integration
- [x] Run benchmark on Mac and iterate on thresholds
- [x] Tune thresholds based on actual confidence distributions
- [x] Submit to leaderboard
- [x] **WON 1ST PLACE** 🏆
