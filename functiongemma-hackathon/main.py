"""
CactusRoute — Multi-Signal Adaptive Hybrid Router
==================================================

Routes function-calling queries between FunctionGemma (270M on-device via Cactus)
and Gemini 2.5 Flash (cloud) using a 4-signal decision framework:

  1. Pre-flight difficulty estimation   (zero-cost heuristic)
  2. Cactus handoff signals             (cloud_handoff / spike_handoff)
  3. Adaptive confidence thresholds     (per-difficulty, research-calibrated)
  4. Output structural validation       (tool names + required params + intent coverage)

Research basis:
  - STEER (arxiv 2511.06190): logit confidence is bimodal, dynamic > fixed thresholds
  - U-HLM (arxiv 2412.12687): speculative local-first saves 46% cloud calls
  - FrugalGPT: cascading with learned sufficiency thresholds
  - RouteLLM: entropy-based routing via Bradley-Terry model
  - Cactus SDK: confidence = 1 - entropy, with cloud_handoff and spike_handoff signals
"""

import sys
sys.path.insert(0, "cactus/python/src")

import json, os, time, re, atexit
from cactus import cactus_init, cactus_complete, cactus_destroy
from google import genai
from google.genai import types

try:
    from cactus import cactus_reset
except ImportError:
    cactus_reset = None


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

FUNCTIONGEMMA_PATH = "cactus/weights/functiongemma-270m-it"
CLOUD_MODEL = os.environ.get("CLOUD_MODEL", "gemini-2.5-flash")

# Adaptive thresholds calibrated from research (STEER bimodal distribution,
# FrugalGPT >0.70 bypass, U-HLM 0.43 risk-prone threshold).
# Lower = more on-device, Higher = more cloud fallback.
THRESHOLDS = {
    "easy":   0.25,   # 1 tool available → can't pick wrong name → very low bar
    "medium": 0.45,   # multiple tools, 1 call → need correct selection
    "hard":   0.60,   # multi-call → higher bar, but still try local first
}

# Internal Cactus confidence threshold — controls when cloud_handoff fires.
# Set low: we want the model to always attempt generation so we can evaluate
# the output ourselves. Only trigger handoff on catastrophic first-token entropy.
CACTUS_CONFIDENCE = 0.15


# ═══════════════════════════════════════════════════════════════════════════════
# Model Singleton — load once, reuse across all benchmark calls
# Eliminates ~200-500ms model-load overhead per call (30 calls = 6-15s saved)
# ═══════════════════════════════════════════════════════════════════════════════

_model = None
_gemini_client = None


def _get_model():
    """Lazy-load FunctionGemma model exactly once."""
    global _model
    if _model is None:
        _model = cactus_init(FUNCTIONGEMMA_PATH)
        atexit.register(_cleanup)
    return _model


def _get_gemini():
    """Lazy-load Gemini API client exactly once."""
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    return _gemini_client


def _cleanup():
    global _model
    if _model is not None:
        cactus_destroy(_model)
        _model = None


# ═══════════════════════════════════════════════════════════════════════════════
# Pre-flight Difficulty Estimation (zero model-inference cost)
#
# Classifies queries as easy/medium/hard using only the raw text + tool count.
# This sets the adaptive confidence threshold before any model runs.
# ═══════════════════════════════════════════════════════════════════════════════

_MULTI_MARKERS = re.compile(
    r'\b(?:and|then|also|plus|after\s+that)\b', re.IGNORECASE
)

_ACTION_VERBS = frozenset({
    "set", "get", "send", "play", "create", "search", "find", "check",
    "remind", "text", "look", "wake", "call", "start", "stop",
    "timer", "alarm", "message", "weather", "music", "contacts",
})


def estimate_difficulty(messages, tools):
    """
    Zero-cost difficulty classifier.

    Rules:
      1. Single tool available → easy (model can't choose wrong tool name)
      2. Multi-intent markers + ≥2 action verbs → hard (multi-call expected)
      3. Multiple comma-separated clauses with verbs → hard
      4. Otherwise → medium (single call from multiple tool options)
    """
    if len(tools) == 1:
        return "easy"

    content = " ".join(
        m["content"] for m in messages if m["role"] == "user"
    ).lower()

    words = set(re.findall(r'\b\w+\b', content))
    verb_count = len(words & _ACTION_VERBS)
    has_multi = bool(_MULTI_MARKERS.search(content))

    # Comma-separated clauses with substance (>8 chars)
    clauses = [c.strip() for c in content.split(",") if len(c.strip()) > 8]
    has_multi_clauses = len(clauses) >= 2

    if (has_multi or has_multi_clauses) and verb_count >= 2:
        return "hard"

    return "medium"


def count_expected_intents(messages):
    """Estimate distinct action count from the user query."""
    content = " ".join(
        m["content"] for m in messages if m["role"] == "user"
    ).lower()

    # Split on conjunctions and substantial comma clauses
    parts = re.split(r'\b(?:and|then)\b|,\s+(?=[a-z])', content)
    words_per_part = [set(re.findall(r'\b\w+\b', p)) for p in parts]
    intents = sum(1 for w in words_per_part if w & _ACTION_VERBS)
    return max(1, intents)


# ═══════════════════════════════════════════════════════════════════════════════
# Type Coercion & Output Validation
#
# Critical for F1: benchmark uses _normalize() which does NOT coerce types.
# "10" (str) ≠ 10 (int) in Python, so integer params must be properly typed.
# ═══════════════════════════════════════════════════════════════════════════════

def coerce_arg_types(function_calls, tools):
    """Coerce argument types to match tool schema definitions."""
    schema_map = {
        t["name"]: t["parameters"].get("properties", {})
        for t in tools
    }
    for call in function_calls:
        props = schema_map.get(call.get("name"), {})
        args = call.get("arguments", {})
        for key, val in list(args.items()):
            if key not in props:
                continue
            expected = props[key].get("type", "string")
            if expected == "integer" and not isinstance(val, int):
                try:
                    args[key] = int(float(val))
                except (ValueError, TypeError):
                    pass
            elif expected == "number" and not isinstance(val, (int, float)):
                try:
                    args[key] = float(val)
                except (ValueError, TypeError):
                    pass
    return function_calls


def validate_output(function_calls, tools):
    """
    Validate structural correctness of function calls.
    Returns (is_valid: bool, reason: str).
    """
    if not function_calls:
        return False, "empty"

    tool_names = {t["name"] for t in tools}
    tool_map = {t["name"]: t for t in tools}

    for call in function_calls:
        name = call.get("name", "")
        if name not in tool_names:
            return False, f"unknown-tool:{name}"
        required = tool_map[name]["parameters"].get("required", [])
        args = call.get("arguments", {})
        missing = [p for p in required if p not in args]
        if missing:
            return False, f"missing-params:{name}:{missing}"

    return True, "ok"


# ═══════════════════════════════════════════════════════════════════════════════
# On-Device Inference (FunctionGemma via Cactus)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_cactus(messages, tools, difficulty="medium"):
    """
    Run function calling on-device via FunctionGemma + Cactus.

    Optimizations over baseline:
      - Global model singleton (no load/destroy per call)
      - KV cache reset between unrelated queries
      - Dynamic system prompt for multi-call queries
      - tool_rag_top_k=0 to consider ALL tools (critical for hard cases)
      - Type coercion on output arguments
    """
    model = _get_model()

    # Clear KV cache between unrelated calls
    if cactus_reset is not None:
        cactus_reset(model)

    # Dynamic system prompt: instruct multi-call for hard queries
    if difficulty == "hard":
        system = (
            "You are a helpful assistant. When the user requests multiple "
            "actions, you MUST call ALL relevant tools — one for each "
            "distinct action requested."
        )
    else:
        system = "You are a helpful assistant that can use tools."

    cactus_tools = [{"type": "function", "function": t} for t in tools]

    raw_str = cactus_complete(
        model,
        [{"role": "system", "content": system}] + messages,
        tools=cactus_tools,
        force_tools=True,
        tool_rag_top_k=0,             # Use ALL tools (default 2 misses tools for hard cases)
        max_tokens=512,                # Room for multi-call output
        confidence_threshold=CACTUS_CONFIDENCE,
        stop_sequences=["<|im_end|>", "<end_of_turn>"],
    )

    try:
        raw = json.loads(raw_str)
    except json.JSONDecodeError:
        return {
            "function_calls": [],
            "total_time_ms": 0,
            "confidence": 0,
            "cloud_handoff": True,
            "spike_handoff": False,
            "success": False,
        }

    calls = raw.get("function_calls", [])
    coerce_arg_types(calls, tools)

    return {
        "function_calls": calls,
        "total_time_ms": raw.get("total_time_ms", 0),
        "confidence": raw.get("confidence", 0),
        "cloud_handoff": raw.get("cloud_handoff", False),
        "spike_handoff": raw.get("spike_handoff", False),
        "success": raw.get("success", True),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Cloud Inference (Gemini API)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_cloud(messages, tools):
    """Run function calling via Gemini Cloud API with type coercion."""
    client = _get_gemini()

    gemini_tools = [
        types.Tool(function_declarations=[
            types.FunctionDeclaration(
                name=t["name"],
                description=t["description"],
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        k: types.Schema(
                            type=v["type"].upper(),
                            description=v.get("description", ""),
                        )
                        for k, v in t["parameters"]["properties"].items()
                    },
                    required=t["parameters"].get("required", []),
                ),
            )
            for t in tools
        ])
    ]

    contents = [m["content"] for m in messages if m["role"] == "user"]
    start = time.time()

    response = client.models.generate_content(
        model=CLOUD_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(tools=gemini_tools),
    )

    elapsed_ms = (time.time() - start) * 1000

    calls = []
    for candidate in response.candidates:
        for part in candidate.content.parts:
            if part.function_call:
                calls.append({
                    "name": part.function_call.name,
                    "arguments": dict(part.function_call.args),
                })

    coerce_arg_types(calls, tools)

    return {"function_calls": calls, "total_time_ms": elapsed_ms}


# ═══════════════════════════════════════════════════════════════════════════════
# Adaptive Hybrid Router — the core innovation
#
# Decision flow:
#   1. Pre-flight → estimate difficulty → set adaptive threshold
#   2. Always run FunctionGemma first (≈50-100ms at 3000 tok/s, nearly free)
#   3. Evaluate 4 routing signals:
#      a. cloud_handoff  — first-token entropy catastrophically high
#      b. spike_handoff  — entropy spiked mid-generation
#      c. confidence     — below adaptive difficulty-based threshold
#      d. validation     — invalid tool names, missing params, or intent gap
#   4. Fall back to cloud only when signals indicate low quality
# ═══════════════════════════════════════════════════════════════════════════════

def generate_hybrid(messages, tools):
    """
    Multi-signal adaptive routing between FunctionGemma (edge) and Gemini (cloud).

    This is the function evaluated by benchmark.py and the leaderboard.
    Do not modify the input/output signature.
    """
    # ── Step 1: Pre-flight assessment (zero cost) ──
    difficulty = estimate_difficulty(messages, tools)
    threshold = THRESHOLDS[difficulty]

    # ── Step 2: Speculative local execution (fast) ──
    local = generate_cactus(messages, tools, difficulty=difficulty)

    # ── Signal A: Explicit handoff from Cactus engine ──
    if local.get("cloud_handoff") or not local.get("success", True):
        return _fallback(messages, tools, local, difficulty, "handoff")

    # ── Signal B: Entropy spike mid-generation ──
    if local.get("spike_handoff"):
        return _fallback(messages, tools, local, difficulty, "spike")

    # ── Signal C: Adaptive confidence threshold ──
    if local["confidence"] < threshold:
        return _fallback(messages, tools, local, difficulty, "low-conf")

    # ── Signal D: Output structural validation ──
    is_valid, reason = validate_output(local["function_calls"], tools)
    if not is_valid:
        return _fallback(messages, tools, local, difficulty, reason)

    # ── Signal E: Intent coverage for hard queries ──
    if difficulty == "hard":
        expected_intents = count_expected_intents(messages)
        actual_calls = len(local["function_calls"])
        if actual_calls < expected_intents:
            return _fallback(
                messages, tools, local, difficulty,
                f"intent-gap:{actual_calls}/{expected_intents}",
            )

    # ── All signals passed — accept local result ──
    local["source"] = "on-device"
    local["difficulty"] = difficulty
    return local


def _fallback(messages, tools, local, difficulty, reason):
    """Execute cloud fallback with graceful error handling."""
    try:
        cloud = generate_cloud(messages, tools)
    except Exception:
        # Cloud failed — return local anyway (partial credit > zero)
        local["source"] = "on-device"
        local["difficulty"] = difficulty
        return local

    cloud["source"] = f"cloud ({reason})"
    cloud["local_confidence"] = local.get("confidence", 0)
    cloud["difficulty"] = difficulty
    cloud["total_time_ms"] += local.get("total_time_ms", 0)
    return cloud


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def print_result(label, result):
    """Pretty-print a generation result."""
    print(f"\n{'═' * 60}")
    print(f"  {label}")
    print(f"{'═' * 60}")
    if "source" in result:
        print(f"  Source:     {result['source']}")
    if "difficulty" in result:
        print(f"  Difficulty: {result['difficulty']}")
    if "confidence" in result:
        conf = result["confidence"]
        bar = "█" * int(conf * 20) + "░" * (20 - int(conf * 20))
        print(f"  Confidence: {conf:.4f}  [{bar}]")
    if "local_confidence" in result:
        print(f"  Local conf: {result['local_confidence']:.4f}")
    print(f"  Latency:   {result['total_time_ms']:.1f}ms")
    print()
    for call in result["function_calls"]:
        args = json.dumps(call.get("arguments", {}), separators=(",", ":"))
        print(f"  → {call['name']}({args})")
    print()


if __name__ == "__main__":
    tools = [
        {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"}
                },
                "required": ["location"],
            },
        },
        {
            "name": "set_alarm",
            "description": "Set an alarm for a given time",
            "parameters": {
                "type": "object",
                "properties": {
                    "hour": {"type": "integer", "description": "Hour"},
                    "minute": {"type": "integer", "description": "Minute"},
                },
                "required": ["hour", "minute"],
            },
        },
        {
            "name": "send_message",
            "description": "Send a message to a contact",
            "parameters": {
                "type": "object",
                "properties": {
                    "recipient": {"type": "string", "description": "Contact name"},
                    "message": {"type": "string", "description": "Message content"},
                },
                "required": ["recipient", "message"],
            },
        },
    ]

    # Easy
    print_result(
        "Easy — Single tool query",
        generate_hybrid(
            [{"role": "user", "content": "What is the weather in San Francisco?"}],
            [tools[0]],
        ),
    )

    # Medium
    print_result(
        "Medium — Pick right tool from 3",
        generate_hybrid(
            [{"role": "user", "content": "Set an alarm for 7:30 AM."}],
            tools,
        ),
    )

    # Hard
    print_result(
        "Hard — Multi-call",
        generate_hybrid(
            [{"role": "user", "content": "Text Alice good morning and check the weather in Paris."}],
            tools,
        ),
    )
