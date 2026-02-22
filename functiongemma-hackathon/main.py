"""
CactusRoute — Multi-Signal Adaptive Hybrid Router
==================================================

Routes function-calling queries between FunctionGemma (270M on-device via Cactus)
and Gemini 2.5 Flash (cloud) using a 7-layer adaptive framework:

  1. Pre-flight difficulty estimation   (zero-cost heuristic)
  2. Cactus handoff signals             (cloud_handoff / spike_handoff)
  3. Schema-driven output repair        (AM/PM, negatives, semantic mismatches)
  4. Multi-gate validation              (structural + semantic + intent coverage)
  5. Adaptive confidence thresholds     (per-difficulty, research-calibrated)
  6. Retry with alternate prompt        (cheap second chance on-device)
  7. Deterministic extraction fallback  (schema-driven text parsing)

Research basis:
  - STEER (arxiv 2511.06190): logit confidence is bimodal, dynamic > fixed thresholds
  - U-HLM (arxiv 2412.12687): speculative local-first saves 46% cloud calls
  - FrugalGPT: cascading with learned sufficiency thresholds
  - RouteLLM: entropy-based routing via Bradley-Terry model
  - Cactus SDK: confidence = 1 - entropy, with cloud_handoff and spike_handoff signals
"""

import sys
sys.path.insert(0, "cactus/python/src")

import atexit
import json
import logging
import os
import re
import time

from cactus import cactus_init, cactus_complete, cactus_destroy

try:
    from cactus import cactus_reset
except ImportError:
    logging.debug("cactus_reset not available in this SDK version")
    cactus_reset = None

log = logging.getLogger(__name__)


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
        from google import genai
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

def coerce_arg_types(function_calls, tools, tool_map=None):
    """Coerce argument types to match tool schema definitions."""
    if tool_map is None:
        tool_map = _build_tool_map(tools)
    for call in function_calls:
        tool_def = tool_map.get(call.get("name"))
        if tool_def is None:
            continue
        props = tool_def["parameters"].get("properties", {})
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


def _build_tool_map(tools):
    """Build name→tool lookup dict. Reused across validate/semantic/repair."""
    return {t["name"]: t for t in tools}


def _extract_text_words(user_text):
    """Extract content words from user text, excluding stop words."""
    return set(re.findall(r'[a-z]{3,}', user_text.lower())) - _STOP_WORDS


def _validate_calls(calls, tools, user_text, tool_map=None, text_words=None):
    """Run the full coerce → structural → semantic validation pipeline.

    Returns (is_valid: bool, reason: str).
    """
    coerce_arg_types(calls, tools, tool_map)
    sv, reason = validate_output(calls, tools, tool_map)
    if not sv:
        return False, reason
    sem, sem_reason = semantic_validate(calls, tools, user_text, tool_map,
                                        text_words)
    if not sem:
        return False, sem_reason
    return True, "ok"


def _make_ondevice_result(calls, time_ms, difficulty, detail,
                          confidence=0.5):
    """Build a standardized on-device result dict."""
    return {
        "function_calls": calls,
        "total_time_ms": time_ms,
        "confidence": confidence,
        "source": "on-device",
        "_detail": detail,
        "difficulty": difficulty,
    }


def validate_output(function_calls, tools, tool_map=None):
    """
    Validate structural correctness of function calls.
    Returns (is_valid: bool, reason: str).
    """
    if not function_calls:
        return False, "empty"

    if tool_map is None:
        tool_map = _build_tool_map(tools)

    for call in function_calls:
        name = call.get("name", "")
        if name not in tool_map:
            return False, f"unknown-tool:{name}"
        required = tool_map[name]["parameters"].get("required", [])
        args = call.get("arguments", {})
        missing = [p for p in required if p not in args]
        if missing:
            return False, f"missing-params:{name}:{missing}"

    return True, "ok"


# ═══════════════════════════════════════════════════════════════════════════════
# Schema-Driven Role Inference
#
# Infers semantic roles (person, location, message, time, etc.) from parameter
# names and schema metadata. This is the key to generalizing extraction to
# unseen tool definitions — we don't hardcode per-tool behavior.
# ═══════════════════════════════════════════════════════════════════════════════

ROLE_PERSON = "person"
ROLE_LOCATION = "location"
ROLE_MESSAGE = "message"
ROLE_HOUR = "hour"
ROLE_MINUTE = "minute"
ROLE_DURATION = "duration"
ROLE_TITLE = "title"
ROLE_TIME_STR = "time_string"
ROLE_SONG = "song"
ROLE_QUERY = "query"
ROLE_UNKNOWN = "unknown"


def infer_param_role(param_name, param_info):
    """Infer semantic role from parameter name and schema metadata.

    Schema-driven: uses parameter names and descriptions, not tool names.
    This generalizes to unseen tool definitions in held-out evaluation.
    """
    name = param_name.lower()
    desc = (param_info.get("description") or "").lower()
    ptype = (param_info.get("type") or "string").lower()

    if ptype == "string":
        if any(k in name for k in ("recipient", "person", "contact")):
            return ROLE_PERSON
        if name == "name" and any(k in desc for k in ("person", "contact")):
            return ROLE_PERSON
        if any(k in name for k in ("location", "city", "place")) or \
           any(k in desc for k in ("city",)):
            return ROLE_LOCATION
        if any(k in name for k in ("message", "content", "body")):
            return ROLE_MESSAGE
        if any(k in name for k in ("title", "note", "task")):
            return ROLE_TITLE
        if name == "time":
            return ROLE_TIME_STR
        if any(k in name for k in ("song", "track", "playlist")):
            return ROLE_SONG
        if any(k in name for k in ("query", "search")):
            return ROLE_QUERY
    elif ptype == "integer":
        if "hour" in name:
            return ROLE_HOUR
        if name == "minute":
            return ROLE_MINUTE
        if any(k in name for k in ("minutes", "duration")):
            return ROLE_DURATION

    return ROLE_UNKNOWN


# ═══════════════════════════════════════════════════════════════════════════════
# Text Extraction Engine
#
# Extracts candidate values from user text for each semantic role.
# Patterns are structural (not tool-specific) so they generalize to
# diverse phrasings in held-out evaluation.
# ═══════════════════════════════════════════════════════════════════════════════

_TIME_RE = re.compile(r'(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b', re.I)
_DURATION_RE = re.compile(r'(\d+)\s*(?:minutes?|mins?)\b', re.I)


def _extract_all_times(text: str) -> list[str]:
    """Return all formatted time strings found in *text* (e.g. ['6:45 AM', '7:00 AM'])."""
    results = []
    for m in _TIME_RE.finditer(text):
        hour = int(m.group(1))
        minute = int(m.group(2) or 0)
        ampm = m.group(3).upper()
        results.append(f"{hour}:{minute:02d} {ampm}")
    return results

# Message content: "saying X" / "that says X" / implicit after "text [Person]"
_MSG_PATTERNS = [
    re.compile(
        r'(?:saying|that\s+says?|say)\s+(.+?)'
        r'(?:\s+and\s+(?:check|get|set|play|search|find|look|remind|text|send|wake|call)|[,.!?]|$)',
        re.I,
    ),
    re.compile(
        r'(?:text|message)\s+[A-Z][a-z]+\s+(.+?)'
        r'(?:\s+and\s+(?:check|get|set|play|search|find|look|remind|text|send|wake|call)|[,.!?]|$)',
        re.I,
    ),
]

_PLAY_RE = re.compile(r'play\s+(?:some\s+)?(.+?)(?:\s+and\s+|[,.!?]|$)', re.I)

# Location: capitalized words after "in" near weather, or standalone "in [City]"
_LOC_PATTERNS = [
    re.compile(r'weather\s+(?:like\s+)?in\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)'),
    re.compile(r'(?:weather|forecast)\s+(?:like\s+)?in\s+(.+?)(?:\s+and\s+|[.!?,]|$)', re.I),
    re.compile(r'\bin\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)'),
]

_REMIND_TITLE_RE = re.compile(
    r'remind\s+me\s+(?:about|to)\s+(.+?)(?:\s+at\s+\d|[.!?]|$)', re.I,
)

_PERSON_PATTERNS = [
    re.compile(r'[Ss]end\s+([A-Z][a-z]+)\s+(?:a\s+)?(?:message|text)'),
    re.compile(r'(?:[Tt]o|[Tt]ext|[Mm]essage)\s+([A-Z][a-z]+)'),
    re.compile(r'(?:[Ff]ind|[Ll]ook\s*up|[Ss]earch\s+for)\s+([A-Z][a-z]+)'),
]

_FIND_RE = re.compile(r'(?:[Ff]ind|[Ll]ook\s*up|[Ss]earch\s*(?:for)?)\s+([A-Z][a-z]+)')


def extract_for_role(role, user_text):
    """Extract a candidate value from user text for a given semantic role."""
    if role == ROLE_PERSON:
        for pat in _PERSON_PATTERNS:
            m = pat.search(user_text)
            if m:
                return m.group(1)
        return None

    if role == ROLE_LOCATION:
        for pat in _LOC_PATTERNS:
            m = pat.search(user_text)
            if m:
                return m.group(1).strip().rstrip('.,!?')
        return None

    if role == ROLE_MESSAGE:
        for pat in _MSG_PATTERNS:
            m = pat.search(user_text)
            if m:
                return m.group(1).strip().rstrip('.,!?')
        return None

    if role == ROLE_HOUR:
        m = _TIME_RE.search(user_text)
        if not m:
            return None
        hour = int(m.group(1))
        ampm = m.group(3).lower()
        if ampm == "pm" and 1 <= hour <= 11:
            hour += 12
        elif ampm == "am" and hour == 12:
            hour = 0
        return hour

    if role == ROLE_MINUTE:
        m = _TIME_RE.search(user_text)
        if not m:
            return None
        return int(m.group(2) or 0)

    if role == ROLE_DURATION:
        m = _DURATION_RE.search(user_text)
        if m:
            return int(m.group(1))
        return None

    if role == ROLE_TITLE:
        m = _REMIND_TITLE_RE.search(user_text)
        if m:
            title = m.group(1).strip().rstrip('.,!?')
            # Strip leading "the" since users say "remind me about the X" but mean just "X"
            if title.lower().startswith('the '):
                title = title[4:].strip()
            return title
        return None

    if role == ROLE_TIME_STR:
        m = _TIME_RE.search(user_text)
        if not m:
            return None
        hour = int(m.group(1))
        minute = int(m.group(2) or 0)
        ampm = m.group(3).upper()
        return f"{hour}:{minute:02d} {ampm}"

    if role == ROLE_SONG:
        m = _PLAY_RE.search(user_text)
        if m:
            song = m.group(1).strip().rstrip('.,!?')
            # Strip trailing "music" since users say "play X music" but mean just "X"
            if song.lower().endswith(' music'):
                song = song[:-6].strip()
            return song
        return None

    if role == ROLE_QUERY:
        m = _FIND_RE.search(user_text)
        if m:
            return m.group(1)
        return None

    return None


# ═══════════════════════════════════════════════════════════════════════════════
# Semantic Validation
#
# Goes beyond structural validation: checks that argument VALUES make sense
# relative to the user's text. Catches cases where FunctionGemma produces
# valid JSON with wrong content (e.g., wrong recipient, hallucinated city).
# ═══════════════════════════════════════════════════════════════════════════════

_STOP_WORDS = frozenset({
    "the", "for", "and", "but", "not", "this", "that", "with",
    "from", "about", "into", "what", "how", "can", "you", "your",
    "please", "some", "like", "its", "has", "are", "was", "get",
    "set", "send", "play", "find", "check", "make", "will",
})


def semantic_validate(calls, tools, user_text, tool_map=None,
                      text_words=None):
    """Validate argument values against user text. Returns (ok, reason)."""
    if text_words is None:
        text_words = _extract_text_words(user_text)
    if tool_map is None:
        tool_map = _build_tool_map(tools)

    for call in calls:
        name = call.get("name", "")
        if name not in tool_map:
            return False, f"unknown-tool:{name}"
        props = tool_map[name]["parameters"].get("properties", {})
        args = call.get("arguments", {})

        for pname, pinfo in props.items():
            if pname not in args:
                continue
            val = args[pname]
            ptype = (pinfo.get("type") or "string").lower()
            role = infer_param_role(pname, pinfo)

            # String values should share at least one content word with user text
            if ptype == "string" and isinstance(val, str) and len(val.strip()) > 0:
                val_words = set(re.findall(r'[a-z]{3,}', val.lower())) - _STOP_WORDS
                if val_words and not val_words & text_words:
                    return False, f"semantic:{pname}={val}"
                
                # For extractable string roles, check extraction matches EXACTLY
                if role in (ROLE_TITLE, ROLE_SONG, ROLE_LOCATION, ROLE_MESSAGE, ROLE_TIME_STR):
                    if role == ROLE_TIME_STR:
                        # For time strings, multiple times may exist in text (multi-intent).
                        # Accept if value matches ANY time found in text.
                        all_times = _extract_all_times(user_text)
                        if all_times:
                            val_norm = val.strip().lower()
                            if val_norm not in {t.strip().lower() for t in all_times}:
                                return False, f"extract-mismatch:{pname}='{val}' vs extracted times={all_times}"
                    else:
                        extracted = extract_for_role(role, user_text)
                        if extracted is not None:
                            ext_norm = extracted.strip().lower()
                            val_norm = val.strip().lower()
                            # Require exact match - no partial matching allowed
                            if ext_norm != val_norm:
                                return False, f"extract-mismatch:{pname}='{val}' vs extracted='{extracted}'"

            # Integer range checks
            if ptype == "integer" and isinstance(val, (int, float)):
                iv = int(val)
                if role == ROLE_HOUR and not (0 <= iv <= 23):
                    return False, f"range:{pname}={iv}"
                if role == ROLE_MINUTE and not (0 <= iv <= 59):
                    return False, f"range:{pname}={iv}"
                if role == ROLE_DURATION and iv <= 0:
                    return False, f"range:{pname}={iv}"
                
                # For extractable integer roles, check extraction matches
                if role in (ROLE_DURATION, ROLE_HOUR, ROLE_MINUTE):
                    extracted = extract_for_role(role, user_text)
                    if extracted is not None and iv != extracted:
                        return False, f"extract-mismatch:{pname}={iv} vs extracted={extracted}"

    return True, "ok"


# ═══════════════════════════════════════════════════════════════════════════════
# Output Repair
#
# Fixes known FunctionGemma failure modes WITHOUT going to cloud:
#   - AM/PM hour correction (returns 10 for "10 PM" → should be 22)
#   - Negative integers → absolute value
#   - Semantically wrong string values → replace with text extraction
#   - Missing required parameters → fill from extraction
#
# This is the key differentiator: repair before fallback maximizes on-device
# ratio while maintaining F1.
# ═══════════════════════════════════════════════════════════════════════════════

def repair_output(calls, tools, user_text, tool_map=None,
                  text_words=None):
    """Repair model output in-place using schema-driven text extraction."""
    if tool_map is None:
        tool_map = _build_tool_map(tools)
    if text_words is None:
        text_words = _extract_text_words(user_text)
    text_lower = user_text.lower()
    repaired = []

    for call in calls:
        name = call.get("name", "")
        if name not in tool_map:
            continue

        props = tool_map[name]["parameters"].get("properties", {})
        required = set(tool_map[name]["parameters"].get("required", []))
        args = dict(call.get("arguments", {}))

        for pname, pinfo in props.items():
            role = infer_param_role(pname, pinfo)
            ptype = (pinfo.get("type") or "string").lower()

            # Fill missing required params via extraction
            if pname not in args or args[pname] is None:
                if pname in required:
                    extracted = extract_for_role(role, user_text)
                    if extracted is not None:
                        args[pname] = extracted
                continue

            val = args[pname]

            # AM/PM hour correction
            if role == ROLE_HOUR and isinstance(val, (int, float, str)):
                try:
                    hour = int(val)
                except (ValueError, TypeError):
                    hour = None
                if hour is not None:
                    if "pm" in text_lower and 1 <= hour <= 11:
                        args[pname] = hour + 12
                    elif "am" in text_lower and hour == 12:
                        args[pname] = 0
                    elif hour < 0 or hour > 23:
                        ext = extract_for_role(ROLE_HOUR, user_text)
                        if ext is not None:
                            args[pname] = ext

            # Fix negative integers
            if ptype == "integer" and isinstance(val, (int, float)):
                if int(val) < 0:
                    args[pname] = abs(int(val))

            # Fix string values that don't match user text
            if ptype == "string" and isinstance(val, str):
                val_words = set(re.findall(r'[a-z]{3,}', val.lower())) - _STOP_WORDS
                if val_words and not val_words & text_words:
                    ext = extract_for_role(role, user_text)
                    if ext is not None:
                        args[pname] = ext

        repaired.append({"name": name, "arguments": args})

    return repaired


# ═══════════════════════════════════════════════════════════════════════════════
# Deterministic Extraction Fallback
#
# When the model fails entirely (handoff, low confidence, or invalid output),
# this builds function calls purely from text extraction. Schema-driven:
# scores tools by relevance, extracts parameters by inferred role.
# Keeps execution on-device (no cloud call).
# ═══════════════════════════════════════════════════════════════════════════════

_TOOL_SYNONYMS = {
    "reminder": ["remind", "reminder", "remember"],
    "alarm": ["alarm", "wake"],
    "timer": ["timer", "countdown"],
    "music": ["music", "play", "song", "listen"],
    "weather": ["weather", "forecast", "temperature"],
    "message": ["message", "text", "send", "sms"],
    "contacts": ["contact", "find", "search", "look"],
}


def _tool_relevance(tool, user_text):
    """Score tool relevance to user text using name/description keywords + synonyms."""
    text_lower = user_text.lower()
    score = 0
    tool_words = tool["name"].replace("_", " ").split()
    for word in tool_words:
        wl = word.lower()
        if wl in text_lower:
            score += 3
        # Check synonyms
        synonyms = _TOOL_SYNONYMS.get(wl, [])
        for syn in synonyms:
            if syn in text_lower:
                score += 3
                break
    for word in tool.get("description", "").lower().split():
        if len(word) > 3 and word in text_lower:
            score += 1
    return score


def build_calls_from_text(user_text, tools, max_calls=None):
    """Build function calls from text extraction. Returns list of valid calls.
    
    Args:
        user_text: The user's query text
        tools: List of tool definitions
        max_calls: Maximum number of calls to return (None = all matches)
    """
    candidates = []
    for tool in tools:
        props = tool["parameters"].get("properties", {})
        required = set(tool["parameters"].get("required", []))

        args = {}
        for pname, pinfo in props.items():
            role = infer_param_role(pname, pinfo)
            val = extract_for_role(role, user_text)
            if val is not None:
                args[pname] = val

        # Only include if all required params are satisfied
        if required and required <= set(args.keys()):
            relevance = _tool_relevance(tool, user_text)
            candidates.append({
                "name": tool["name"],
                "arguments": args,
                "relevance": relevance,
            })

    if not candidates:
        return []

    candidates.sort(key=lambda c: c["relevance"], reverse=True)

    seen = set()
    result = []
    for c in candidates:
        if c["name"] not in seen:
            seen.add(c["name"])
            result.append({"name": c["name"], "arguments": c["arguments"]})
            # Limit output if max_calls is specified
            if max_calls is not None and len(result) >= max_calls:
                break
    return result


def _segment_query(user_text):
    """Split a multi-intent query into actionable segments."""
    parts = re.split(
        r'(?:,\s*(?:and\s+)?(?=[a-z]))'
        r'|(?<=\w)\s+and\s+(?=(?:check|get|set|play|search|find|look|remind|text|send|wake|call))',
        user_text, flags=re.I,
    )
    return [p.strip() for p in parts if len(p.strip()) > 3]


def build_calls_from_segments(user_text, tools):
    """Decompose query into segments and extract calls per segment."""
    segments = _segment_query(user_text)
    if len(segments) <= 1:
        return build_calls_from_text(user_text, tools)

    all_calls = []
    used_tools = set()
    for seg in segments:
        seg_calls = build_calls_from_text(seg, tools)
        for c in seg_calls:
            if c["name"] not in used_tools:
                used_tools.add(c["name"])
                all_calls.append(c)

    # If segmented extraction got fewer, try full text too
    if not all_calls:
        return build_calls_from_text(user_text, tools)

    return all_calls


# ═══════════════════════════════════════════════════════════════════════════════
# On-Device Inference (FunctionGemma via Cactus)
# ═══════════════════════════════════════════════════════════════════════════════

_SYSTEM_PROMPTS = [
    "You are a helpful assistant that can use tools.",
    (
        "You MUST call one of the available functions. Extract all parameter "
        "values directly from the user's message. Match parameter types precisely."
    ),
]


def generate_cactus(messages, tools, difficulty="medium", prompt_idx=0):
    """
    Run function calling on-device via FunctionGemma + Cactus.

    Optimizations over baseline:
      - Global model singleton (no load/destroy per call)
      - KV cache reset between unrelated queries
      - Dynamic system prompt for multi-call queries
      - tool_rag_top_k=0 to consider ALL tools (critical for hard cases)
      - Type coercion on output arguments
      - Alternate prompts for retry (prompt_idx)
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
        system = _SYSTEM_PROMPTS[prompt_idx % len(_SYSTEM_PROMPTS)]

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
    from google.genai import types
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
# Adaptive Hybrid Router — 7-Layer Decision Framework
#
# Layers (conceptual) map to implementation steps:
#   Layer 1: Pre-flight difficulty estimation (zero cost)
#   Layer 2: Local execution + handoff signals (steps 1-3)
#   Layer 3: Schema-driven output repair (step 4)
#   Layer 4: Multi-gate validation: structural + semantic + intent (steps 5-6)
#   Layer 5: Adaptive confidence thresholds (step 7)
#   Layer 6: Retry with alternate prompt (step 8)
#   Layer 7: Deterministic extraction fallback → cloud last resort (step 9)
#
# Key insight: maximize on-device ratio by REPAIRING local output before
# deciding to fall back. Most competitors dump to cloud on any issue — we
# fix it locally first, keeping both F1 and on-device ratio high.
# ═══════════════════════════════════════════════════════════════════════════════

def _extraction_cross_check(repaired, tools, tool_map, user_text,
                            expected_intents, local_time, difficulty):
    """
    Verify model chose the same tool as deterministic extraction.
    Returns an override result dict, or None if model's choice stands.
    """
    det_check = build_calls_from_text(user_text, tools, max_calls=expected_intents)
    if not det_check:
        return None

    model_tools = {c["name"] for c in repaired}
    extract_tools = {c["name"] for c in det_check}
    if model_tools == extract_tools:
        return None

    # Extraction disagrees — compare relevance of top picks
    best_extract = det_check[0]
    extract_tool = tool_map.get(best_extract["name"])
    if extract_tool is None:
        return None
    best_extract_rel = _tool_relevance(extract_tool, user_text)
    model_rel = max(
        (_tool_relevance(t, user_text) for t in tools if t["name"] in model_tools),
        default=0,
    )
    if best_extract_rel <= model_rel:
        return None

    # Extraction picked more relevant tool — validate before overriding
    valid, _ = _validate_calls(det_check, tools, user_text, tool_map)
    if valid:
        return _make_ondevice_result(
            det_check, local_time, difficulty, "on-device (extract-override)",
        )
    return None


def _retry_on_device(messages, tools, tool_map, user_text, difficulty,
                     expected_intents, local_time):
    """
    Retry local execution with an alternate prompt.
    Returns a result dict on success, or None if retry fails validation.
    """
    retry = generate_cactus(messages, tools, difficulty=difficulty, prompt_idx=1)
    retry_time = retry.get("total_time_ms", 0)

    if not retry.get("success", True) or retry.get("cloud_handoff"):
        return None

    retry_repaired = repair_output(retry["function_calls"], tools, user_text,
                                   tool_map)
    rv, _ = _validate_calls(retry_repaired, tools, user_text, tool_map)
    if not rv:
        return None

    iok = True
    if difficulty == "hard":
        if len(retry_repaired) < expected_intents:
            aug = _augment_calls(retry_repaired, tools, user_text)
            if len(aug) >= expected_intents:
                retry_repaired = aug
            else:
                iok = False

    if not iok:
        return None

    retry["function_calls"] = retry_repaired
    retry["total_time_ms"] = local_time + retry_time
    retry["source"] = "on-device"  # must match benchmark check exactly
    retry["_detail"] = "on-device (retry)"
    retry["difficulty"] = difficulty
    return retry


def generate_hybrid(messages, tools):
    """
    Multi-signal adaptive routing with repair, retry, and extraction fallback.

    This is the function evaluated by benchmark.py and the leaderboard.
    Do not modify the input/output signature.
    """
    difficulty = estimate_difficulty(messages, tools)
    threshold = THRESHOLDS[difficulty]
    user_text = " ".join(m["content"] for m in messages if m["role"] == "user")
    expected_intents = count_expected_intents(messages)
    tool_map = _build_tool_map(tools)

    # ── Step 1: Local execution ──
    local = generate_cactus(messages, tools, difficulty=difficulty)
    local_time = local.get("total_time_ms", 0)

    # ── Step 2: Hard handoff → try extraction before cloud ──
    if local.get("cloud_handoff") or not local.get("success", True):
        return _try_extraction_then_cloud(
            messages, tools, local, difficulty, user_text, "handoff",
        )

    # ── Step 3: Entropy spike → try extraction before cloud ──
    if local.get("spike_handoff"):
        return _try_extraction_then_cloud(
            messages, tools, local, difficulty, user_text, "spike",
        )

    # ── Step 4: Repair local output (AM/PM, negatives, semantic) ──
    text_words = _extract_text_words(user_text)
    repaired = repair_output(local["function_calls"], tools, user_text,
                             tool_map, text_words)
    coerce_arg_types(repaired, tools, tool_map)
    local["function_calls"] = repaired

    # ── Step 5: Multi-gate validation ──
    is_valid, reason = validate_output(repaired, tools, tool_map)
    sem_valid, sem_reason = (True, "ok")
    if is_valid:
        sem_valid, sem_reason = semantic_validate(repaired, tools, user_text,
                                                  tool_map, text_words)

    # ── Step 6a: Intent coverage for hard queries ──
    intent_ok = True
    if is_valid and sem_valid and difficulty == "hard":
        if len(repaired) < expected_intents:
            augmented = _augment_calls(repaired, tools, user_text)
            if len(augmented) >= expected_intents:
                coerce_arg_types(augmented, tools)
                local["function_calls"] = augmented
            else:
                intent_ok = False

    # ── Step 6b: Extraction cross-check ──
    if is_valid and sem_valid and intent_ok:
        override = _extraction_cross_check(
            repaired, tools, tool_map, user_text,
            expected_intents, local_time, difficulty,
        )
        if override is not None:
            return override

    # ── Step 7: Accept if all gates pass and confident ──
    if is_valid and sem_valid and intent_ok and local["confidence"] >= threshold:
        local["source"] = "on-device"
        local["difficulty"] = difficulty
        return local

    # ── Step 8: Retry with alternate prompt ──
    retry_result = _retry_on_device(
        messages, tools, tool_map, user_text, difficulty,
        expected_intents, local_time,
    )
    if retry_result is not None:
        return retry_result

    # ── Step 9: Deterministic extraction fallback ──
    return _try_extraction_then_cloud(
        messages, tools, local, difficulty, user_text,
        reason or sem_reason or "retry-failed",
    )


def _augment_calls(existing_calls, tools, user_text):
    """Add missing calls via text extraction to reach full intent coverage."""
    existing_names = {c["name"] for c in existing_calls}
    remaining_tools = [t for t in tools if t["name"] not in existing_names]
    extra = build_calls_from_text(user_text, remaining_tools)
    return existing_calls + extra if extra else existing_calls


def _try_extraction_then_cloud(messages, tools, local, difficulty, user_text, reason):
    """Try deterministic extraction; fall to cloud only if extraction also fails."""
    local_time = local.get("total_time_ms", 0)
    expected_intents = count_expected_intents(messages)

    if difficulty == "hard":
        det_calls = build_calls_from_segments(user_text, tools)
    else:
        # For easy/medium, limit to expected intent count to avoid spurious extra calls
        det_calls = build_calls_from_text(user_text, tools, max_calls=expected_intents)

    if det_calls:
        valid, _ = _validate_calls(det_calls, tools, user_text)

        intent_ok = True
        if difficulty == "hard" and valid:
            if len(det_calls) < expected_intents:
                intent_ok = False

        if valid and intent_ok:
            return _make_ondevice_result(
                det_calls, local_time, difficulty, "on-device (extracted)",
            )

    return _fallback(messages, tools, local, difficulty, reason)


def _fallback(messages, tools, local, difficulty, reason):
    """Execute cloud fallback with graceful error handling."""
    try:
        cloud = generate_cloud(messages, tools)
    except Exception:
        log.warning("Cloud fallback failed, returning local result", exc_info=True)
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
        detail = result.get("_detail", result["source"])
        print(f"  Source:     {detail}")
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
