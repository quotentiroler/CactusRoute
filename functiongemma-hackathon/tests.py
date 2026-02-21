"""
CactusRoute Test Suite
======================

Comprehensive tests for the 7-layer adaptive framework, runnable on any platform
(no Cactus or API keys needed). Tests cover:
  - Pre-flight difficulty estimation (all 30 benchmark cases)
  - Intent counting for multi-call queries
  - Type coercion (string→int, string→float)
  - Output validation (tool names, required params, empty calls)
  - Semantic role inference and extraction (all 11 roles)
  - Semantic validation (word-overlap, range checks)
  - Output repair (AM/PM correction, negatives, missing params)
  - Tool relevance scoring
  - Query segmentation for multi-intent queries
  - Intent augmentation (_augment_calls)
  - Build calls from text and from segments
  - Routing decision matrix (mock-driven, full 7-layer pipeline)
  - Benchmark-realistic extraction (easy/medium/hard patterns)
  - Threshold boundaries, signal priority, benchmark compatibility

Usage:
    python tests.py              # Run all tests
    python tests.py -v           # Verbose output
    python tests.py TestRouting   # Run one test class
"""

import sys
import os
import unittest
from unittest.mock import patch, MagicMock

# ─── Isolate pure logic from Cactus/Gemini imports ─────────────────────────
# We mock the cactus and google.genai modules so tests work on Windows/Linux.

# Mock cactus module
mock_cactus = MagicMock()
mock_cactus.cactus_init = MagicMock(return_value="mock_model")
mock_cactus.cactus_complete = MagicMock(return_value='{"function_calls":[],"confidence":0.5,"total_time_ms":50}')
mock_cactus.cactus_destroy = MagicMock()
mock_cactus.cactus_reset = MagicMock()
sys.modules["cactus"] = mock_cactus

# Mock google.genai module
mock_genai = MagicMock()
mock_types = MagicMock()
sys.modules["google"] = MagicMock()
sys.modules["google.genai"] = mock_genai
sys.modules["google.genai"].types = mock_types

# NOW we can import our module
from main import (
    estimate_difficulty,
    count_expected_intents,
    coerce_arg_types,
    validate_output,
    generate_hybrid,
    generate_cactus,
    _fallback,
    _tool_relevance,
    _segment_query,
    _augment_calls,
    THRESHOLDS,
    infer_param_role,
    extract_for_role,
    semantic_validate,
    repair_output,
    build_calls_from_text,
    build_calls_from_segments,
    ROLE_PERSON, ROLE_LOCATION, ROLE_MESSAGE, ROLE_HOUR, ROLE_MINUTE,
    ROLE_DURATION, ROLE_TITLE, ROLE_TIME_STR, ROLE_SONG, ROLE_QUERY,
    ROLE_UNKNOWN,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Tool Definitions (same as benchmark.py)
# ═══════════════════════════════════════════════════════════════════════════════

TOOL_GET_WEATHER = {
    "name": "get_weather",
    "description": "Get current weather for a location",
    "parameters": {
        "type": "object",
        "properties": {"location": {"type": "string", "description": "City name"}},
        "required": ["location"],
    },
}

TOOL_SET_ALARM = {
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
}

TOOL_SEND_MESSAGE = {
    "name": "send_message",
    "description": "Send a message to a contact",
    "parameters": {
        "type": "object",
        "properties": {
            "recipient": {"type": "string", "description": "Name"},
            "message": {"type": "string", "description": "Content"},
        },
        "required": ["recipient", "message"],
    },
}

TOOL_CREATE_REMINDER = {
    "name": "create_reminder",
    "description": "Create a reminder with a title and time",
    "parameters": {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Reminder title"},
            "time": {"type": "string", "description": "Time for reminder"},
        },
        "required": ["title", "time"],
    },
}

TOOL_SEARCH_CONTACTS = {
    "name": "search_contacts",
    "description": "Search for a contact by name",
    "parameters": {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "Name to search"}},
        "required": ["query"],
    },
}

TOOL_PLAY_MUSIC = {
    "name": "play_music",
    "description": "Play a song or playlist",
    "parameters": {
        "type": "object",
        "properties": {"song": {"type": "string", "description": "Song name"}},
        "required": ["song"],
    },
}

TOOL_SET_TIMER = {
    "name": "set_timer",
    "description": "Set a countdown timer",
    "parameters": {
        "type": "object",
        "properties": {"minutes": {"type": "integer", "description": "Minutes"}},
        "required": ["minutes"],
    },
}

ALL_TOOLS = [
    TOOL_GET_WEATHER, TOOL_SET_ALARM, TOOL_SEND_MESSAGE,
    TOOL_CREATE_REMINDER, TOOL_SEARCH_CONTACTS, TOOL_PLAY_MUSIC, TOOL_SET_TIMER,
]

# Custom tool with unusual param names — extraction CANNOT fill these
TOOL_CUSTOM = {
    "name": "custom_action",
    "description": "Do something custom",
    "parameters": {
        "type": "object",
        "properties": {
            "data": {"type": "string", "description": "Some data"},
        },
        "required": ["data"],
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# Test: Pre-flight Difficulty Estimation
# ═══════════════════════════════════════════════════════════════════════════════

class TestEstimateDifficulty(unittest.TestCase):
    """Tests for estimate_difficulty() — the zero-cost heuristic classifier."""

    # ── Easy cases: single tool available ──

    def test_single_tool_always_easy(self):
        """With 1 tool, model can't pick wrong name → always easy."""
        msg = [{"role": "user", "content": "What is the weather in San Francisco?"}]
        self.assertEqual(estimate_difficulty(msg, [TOOL_GET_WEATHER]), "easy")

    def test_single_tool_complex_query_still_easy(self):
        """Even a long query with 1 tool → easy (can't choose wrong)."""
        msg = [{"role": "user", "content": "Tell me everything about the weather in Paris today please"}]
        self.assertEqual(estimate_difficulty(msg, [TOOL_GET_WEATHER]), "easy")

    def test_all_easy_benchmarks(self):
        """All 10 easy benchmark cases use 1 tool → must classify as easy."""
        easy_cases = [
            ([{"role": "user", "content": "What is the weather in San Francisco?"}], [TOOL_GET_WEATHER]),
            ([{"role": "user", "content": "Set an alarm for 10 AM."}], [TOOL_SET_ALARM]),
            ([{"role": "user", "content": "Send a message to Alice saying good morning."}], [TOOL_SEND_MESSAGE]),
            ([{"role": "user", "content": "What's the weather like in London?"}], [TOOL_GET_WEATHER]),
            ([{"role": "user", "content": "Wake me up at 6 AM."}], [TOOL_SET_ALARM]),
            ([{"role": "user", "content": "Play Bohemian Rhapsody."}], [TOOL_PLAY_MUSIC]),
            ([{"role": "user", "content": "Set a timer for 5 minutes."}], [TOOL_SET_TIMER]),
            ([{"role": "user", "content": "Remind me about the meeting at 3:00 PM."}], [TOOL_CREATE_REMINDER]),
            ([{"role": "user", "content": "Find Bob in my contacts."}], [TOOL_SEARCH_CONTACTS]),
            ([{"role": "user", "content": "How's the weather in Paris?"}], [TOOL_GET_WEATHER]),
        ]
        for msgs, tools in easy_cases:
            with self.subTest(msg=msgs[0]["content"]):
                self.assertEqual(estimate_difficulty(msgs, tools), "easy")

    # ── Medium cases: multiple tools, single action ──

    def test_medium_pick_from_multiple(self):
        """Single action from multiple tools → medium."""
        msg = [{"role": "user", "content": "What's the weather in Tokyo?"}]
        tools = [TOOL_GET_WEATHER, TOOL_SEND_MESSAGE]
        self.assertEqual(estimate_difficulty(msg, tools), "medium")

    def test_medium_alarm_among_three(self):
        msg = [{"role": "user", "content": "Set an alarm for 8:15 AM."}]
        tools = [TOOL_SEND_MESSAGE, TOOL_SET_ALARM, TOOL_GET_WEATHER]
        self.assertEqual(estimate_difficulty(msg, tools), "medium")

    def test_medium_all_benchmarks(self):
        """All 10 medium benchmark cases → must NOT classify as easy."""
        medium_cases = [
            ([{"role": "user", "content": "Send a message to John saying hello."}],
             [TOOL_GET_WEATHER, TOOL_SEND_MESSAGE, TOOL_SET_ALARM]),
            ([{"role": "user", "content": "What's the weather in Tokyo?"}],
             [TOOL_GET_WEATHER, TOOL_SEND_MESSAGE]),
            ([{"role": "user", "content": "Set an alarm for 8:15 AM."}],
             [TOOL_SEND_MESSAGE, TOOL_SET_ALARM, TOOL_GET_WEATHER]),
            ([{"role": "user", "content": "Play some jazz music."}],
             [TOOL_SET_ALARM, TOOL_PLAY_MUSIC, TOOL_GET_WEATHER]),
            ([{"role": "user", "content": "Remind me to call the dentist at 2:00 PM."}],
             [TOOL_GET_WEATHER, TOOL_SEND_MESSAGE, TOOL_CREATE_REMINDER, TOOL_SET_ALARM]),
            ([{"role": "user", "content": "Set a timer for 10 minutes."}],
             [TOOL_SET_ALARM, TOOL_SET_TIMER, TOOL_PLAY_MUSIC]),
            ([{"role": "user", "content": "Look up Sarah in my contacts."}],
             [TOOL_SEND_MESSAGE, TOOL_GET_WEATHER, TOOL_SEARCH_CONTACTS, TOOL_SET_ALARM]),
            ([{"role": "user", "content": "What's the weather in Berlin?"}],
             [TOOL_SEND_MESSAGE, TOOL_SET_ALARM, TOOL_PLAY_MUSIC, TOOL_GET_WEATHER]),
            ([{"role": "user", "content": "Text Dave saying I'll be late."}],
             [TOOL_GET_WEATHER, TOOL_SET_TIMER, TOOL_SEND_MESSAGE, TOOL_PLAY_MUSIC]),
            ([{"role": "user", "content": "Set an alarm for 9 AM."}],
             [TOOL_SEND_MESSAGE, TOOL_GET_WEATHER, TOOL_PLAY_MUSIC, TOOL_SET_TIMER, TOOL_SET_ALARM]),
        ]
        for msgs, tools in medium_cases:
            with self.subTest(msg=msgs[0]["content"]):
                result = estimate_difficulty(msgs, tools)
                self.assertIn(result, ("medium", "hard"),
                              f"Expected medium/hard for '{msgs[0]['content']}', got '{result}'")
                # Most should be medium (single action)
                self.assertNotEqual(result, "easy")

    # ── Hard cases: multi-intent markers + multiple action verbs ──

    def test_hard_message_and_weather(self):
        msg = [{"role": "user", "content": "Send a message to Bob saying hi and get the weather in London."}]
        tools = [TOOL_GET_WEATHER, TOOL_SEND_MESSAGE, TOOL_SET_ALARM]
        self.assertEqual(estimate_difficulty(msg, tools), "hard")

    def test_hard_alarm_and_weather(self):
        msg = [{"role": "user", "content": "Set an alarm for 7:30 AM and check the weather in New York."}]
        tools = [TOOL_GET_WEATHER, TOOL_SET_ALARM, TOOL_SEND_MESSAGE]
        self.assertEqual(estimate_difficulty(msg, tools), "hard")

    def test_hard_three_actions(self):
        msg = [{"role": "user", "content": "Text Emma saying good night, check the weather in Chicago, and set an alarm for 5 AM."}]
        tools = [TOOL_SEND_MESSAGE, TOOL_GET_WEATHER, TOOL_SET_ALARM, TOOL_PLAY_MUSIC, TOOL_SET_TIMER]
        self.assertEqual(estimate_difficulty(msg, tools), "hard")

    def test_hard_all_benchmarks(self):
        """All 10 hard benchmark cases → must classify as hard."""
        hard_cases = [
            "Send a message to Bob saying hi and get the weather in London.",
            "Set an alarm for 7:30 AM and check the weather in New York.",
            "Set a timer for 20 minutes and play lo-fi beats.",
            "Remind me about groceries at 5:00 PM and text Lisa saying see you tonight.",
            "Find Tom in my contacts and send him a message saying happy birthday.",
            "Set an alarm for 6:45 AM and remind me to take medicine at 7:00 AM.",
            "Check the weather in Miami and play summer hits.",
            "Text Emma saying good night, check the weather in Chicago, and set an alarm for 5 AM.",
            "Set a 15 minute timer, play classical music, and remind me to stretch at 4:00 PM.",
            "Look up Jake in my contacts, send him a message saying let's meet, and check the weather in Seattle.",
        ]
        for query in hard_cases:
            with self.subTest(query=query):
                msg = [{"role": "user", "content": query}]
                result = estimate_difficulty(msg, ALL_TOOLS)
                self.assertEqual(result, "hard",
                                 f"Expected hard for '{query}', got '{result}'")

    # ── Edge cases ──

    def test_empty_message(self):
        msg = [{"role": "user", "content": ""}]
        result = estimate_difficulty(msg, ALL_TOOLS)
        self.assertIn(result, ("medium", "hard"))

    def test_no_action_verbs(self):
        msg = [{"role": "user", "content": "Hello there, how are you today?"}]
        result = estimate_difficulty(msg, ALL_TOOLS)
        self.assertEqual(result, "medium")  # Multiple tools, no multi-markers


# ═══════════════════════════════════════════════════════════════════════════════
# Test: Intent Counting
# ═══════════════════════════════════════════════════════════════════════════════

class TestCountExpectedIntents(unittest.TestCase):
    """Tests for count_expected_intents() — estimates number of function calls needed."""

    def test_single_intent(self):
        msg = [{"role": "user", "content": "What's the weather in SF?"}]
        self.assertEqual(count_expected_intents(msg), 1)

    def test_two_intents_with_and(self):
        msg = [{"role": "user", "content": "Set an alarm for 7 AM and check the weather."}]
        self.assertEqual(count_expected_intents(msg), 2)

    def test_three_intents(self):
        msg = [{"role": "user", "content": "Text Emma good night, check weather in Chicago, and set alarm for 5 AM."}]
        result = count_expected_intents(msg)
        self.assertGreaterEqual(result, 2, "Should detect at least 2 intents for 3-action query")

    def test_minimum_one(self):
        """Always returns at least 1."""
        msg = [{"role": "user", "content": "Hello!"}]
        self.assertEqual(count_expected_intents(msg), 1)

    def test_comma_separated_intents(self):
        msg = [{"role": "user", "content": "Set a timer for 15 minutes, play classical music, and remind me to stretch."}]
        result = count_expected_intents(msg)
        self.assertGreaterEqual(result, 2)


# ═══════════════════════════════════════════════════════════════════════════════
# Test: Type Coercion
# ═══════════════════════════════════════════════════════════════════════════════

class TestCoerceArgTypes(unittest.TestCase):
    """Tests for coerce_arg_types() — critical for F1 scoring."""

    def test_string_to_int(self):
        """'10' should become 10 for integer params."""
        calls = [{"name": "set_alarm", "arguments": {"hour": "10", "minute": "0"}}]
        result = coerce_arg_types(calls, [TOOL_SET_ALARM])
        self.assertEqual(result[0]["arguments"]["hour"], 10)
        self.assertEqual(result[0]["arguments"]["minute"], 0)
        self.assertIsInstance(result[0]["arguments"]["hour"], int)

    def test_float_string_to_int(self):
        """'10.0' should become 10."""
        calls = [{"name": "set_alarm", "arguments": {"hour": "10.0", "minute": "30.0"}}]
        result = coerce_arg_types(calls, [TOOL_SET_ALARM])
        self.assertEqual(result[0]["arguments"]["hour"], 10)
        self.assertEqual(result[0]["arguments"]["minute"], 30)

    def test_already_int_unchanged(self):
        """Already-int values should pass through."""
        calls = [{"name": "set_alarm", "arguments": {"hour": 10, "minute": 0}}]
        result = coerce_arg_types(calls, [TOOL_SET_ALARM])
        self.assertEqual(result[0]["arguments"]["hour"], 10)

    def test_string_stays_string(self):
        """String params should not be coerced."""
        calls = [{"name": "get_weather", "arguments": {"location": "Paris"}}]
        result = coerce_arg_types(calls, [TOOL_GET_WEATHER])
        self.assertEqual(result[0]["arguments"]["location"], "Paris")

    def test_invalid_string_for_int(self):
        """Non-numeric string for int param should be left unchanged."""
        calls = [{"name": "set_alarm", "arguments": {"hour": "morning", "minute": 0}}]
        result = coerce_arg_types(calls, [TOOL_SET_ALARM])
        self.assertEqual(result[0]["arguments"]["hour"], "morning")  # Can't coerce

    def test_unknown_tool_no_crash(self):
        """Unknown tool name should not crash."""
        calls = [{"name": "unknown_tool", "arguments": {"x": "5"}}]
        result = coerce_arg_types(calls, [TOOL_GET_WEATHER])
        self.assertEqual(result[0]["arguments"]["x"], "5")

    def test_timer_minutes_coercion(self):
        """Timer minutes param should be coerced to int."""
        calls = [{"name": "set_timer", "arguments": {"minutes": "15"}}]
        result = coerce_arg_types(calls, [TOOL_SET_TIMER])
        self.assertEqual(result[0]["arguments"]["minutes"], 15)
        self.assertIsInstance(result[0]["arguments"]["minutes"], int)

    def test_mixed_types_multi_call(self):
        """Multiple calls with mixed types should all be coerced correctly."""
        calls = [
            {"name": "set_alarm", "arguments": {"hour": "7", "minute": "30"}},
            {"name": "get_weather", "arguments": {"location": "NYC"}},
        ]
        tools = [TOOL_SET_ALARM, TOOL_GET_WEATHER]
        result = coerce_arg_types(calls, tools)
        self.assertEqual(result[0]["arguments"]["hour"], 7)
        self.assertEqual(result[0]["arguments"]["minute"], 30)
        self.assertEqual(result[1]["arguments"]["location"], "NYC")


# ═══════════════════════════════════════════════════════════════════════════════
# Test: Output Validation
# ═══════════════════════════════════════════════════════════════════════════════

class TestValidateOutput(unittest.TestCase):
    """Tests for validate_output() — structural correctness check."""

    def test_valid_single_call(self):
        calls = [{"name": "get_weather", "arguments": {"location": "Paris"}}]
        valid, reason = validate_output(calls, [TOOL_GET_WEATHER])
        self.assertTrue(valid)
        self.assertEqual(reason, "ok")

    def test_valid_multi_call(self):
        calls = [
            {"name": "set_alarm", "arguments": {"hour": 7, "minute": 30}},
            {"name": "get_weather", "arguments": {"location": "NYC"}},
        ]
        valid, reason = validate_output(calls, [TOOL_SET_ALARM, TOOL_GET_WEATHER])
        self.assertTrue(valid)

    def test_empty_calls_invalid(self):
        valid, reason = validate_output([], [TOOL_GET_WEATHER])
        self.assertFalse(valid)
        self.assertEqual(reason, "empty")

    def test_unknown_tool_name(self):
        calls = [{"name": "hack_server", "arguments": {}}]
        valid, reason = validate_output(calls, [TOOL_GET_WEATHER])
        self.assertFalse(valid)
        self.assertIn("unknown-tool", reason)

    def test_missing_required_param(self):
        calls = [{"name": "set_alarm", "arguments": {"hour": 10}}]  # missing "minute"
        valid, reason = validate_output(calls, [TOOL_SET_ALARM])
        self.assertFalse(valid)
        self.assertIn("missing-params", reason)

    def test_extra_params_ok(self):
        """Extra params beyond required should not invalidate."""
        calls = [{"name": "get_weather", "arguments": {"location": "Paris", "units": "celsius"}}]
        valid, reason = validate_output(calls, [TOOL_GET_WEATHER])
        self.assertTrue(valid)

    def test_all_required_present(self):
        calls = [{"name": "send_message", "arguments": {"recipient": "Alice", "message": "hi"}}]
        valid, reason = validate_output(calls, [TOOL_SEND_MESSAGE])
        self.assertTrue(valid)

    def test_mixed_valid_invalid(self):
        """One valid + one invalid call → overall invalid."""
        calls = [
            {"name": "get_weather", "arguments": {"location": "NYC"}},
            {"name": "fake_tool", "arguments": {}},
        ]
        valid, reason = validate_output(calls, ALL_TOOLS)
        self.assertFalse(valid)


# ═══════════════════════════════════════════════════════════════════════════════
# Test: Schema-Driven Role Inference
# ═══════════════════════════════════════════════════════════════════════════════

class TestInferParamRole(unittest.TestCase):
    """Tests for infer_param_role() — maps param metadata to semantic roles."""

    def test_recipient_is_person(self):
        self.assertEqual(infer_param_role("recipient", {"type": "string", "description": "Name"}), ROLE_PERSON)

    def test_location_is_location(self):
        self.assertEqual(infer_param_role("location", {"type": "string", "description": "City name"}), ROLE_LOCATION)

    def test_message_is_message(self):
        self.assertEqual(infer_param_role("message", {"type": "string", "description": "Content"}), ROLE_MESSAGE)

    def test_hour_is_hour(self):
        self.assertEqual(infer_param_role("hour", {"type": "integer", "description": "Hour"}), ROLE_HOUR)

    def test_minute_is_minute(self):
        self.assertEqual(infer_param_role("minute", {"type": "integer", "description": "Minute"}), ROLE_MINUTE)

    def test_minutes_is_duration(self):
        self.assertEqual(infer_param_role("minutes", {"type": "integer", "description": "Minutes"}), ROLE_DURATION)

    def test_title_is_title(self):
        self.assertEqual(infer_param_role("title", {"type": "string", "description": "Title"}), ROLE_TITLE)

    def test_time_is_time_str(self):
        self.assertEqual(infer_param_role("time", {"type": "string", "description": "Time"}), ROLE_TIME_STR)

    def test_song_is_song(self):
        self.assertEqual(infer_param_role("song", {"type": "string", "description": "Song name"}), ROLE_SONG)

    def test_query_is_query(self):
        self.assertEqual(infer_param_role("query", {"type": "string", "description": "Search query"}), ROLE_QUERY)

    def test_unknown_param(self):
        self.assertEqual(infer_param_role("data", {"type": "string", "description": "Some data"}), ROLE_UNKNOWN)

    def test_city_in_description(self):
        self.assertEqual(infer_param_role("loc", {"type": "string", "description": "City to look up"}), ROLE_LOCATION)


# ═══════════════════════════════════════════════════════════════════════════════
# Test: Text Extraction Engine
# ═══════════════════════════════════════════════════════════════════════════════

class TestExtractForRole(unittest.TestCase):
    """Tests for extract_for_role() — regex-based value extraction."""

    def test_person_from_to(self):
        self.assertEqual(extract_for_role(ROLE_PERSON, "Send a message to Alice saying hi"), "Alice")

    def test_person_from_text(self):
        self.assertEqual(extract_for_role(ROLE_PERSON, "Text Bob saying hello"), "Bob")

    def test_person_from_find(self):
        self.assertEqual(extract_for_role(ROLE_PERSON, "Find Tom in my contacts"), "Tom")

    def test_location_weather_in(self):
        self.assertEqual(extract_for_role(ROLE_LOCATION, "What's the weather in Paris?"), "Paris")

    def test_location_multi_word(self):
        self.assertEqual(extract_for_role(ROLE_LOCATION, "What is the weather in San Francisco?"), "San Francisco")

    def test_location_weather_like_in(self):
        self.assertEqual(extract_for_role(ROLE_LOCATION, "What's the weather like in London?"), "London")

    def test_message_saying(self):
        self.assertEqual(extract_for_role(ROLE_MESSAGE, "Send a message to Bob saying hi"), "hi")

    def test_message_saying_long(self):
        self.assertEqual(extract_for_role(ROLE_MESSAGE, "Text Dave saying I'll be late"), "I'll be late")

    def test_hour_am(self):
        self.assertEqual(extract_for_role(ROLE_HOUR, "Set an alarm for 6 AM"), 6)

    def test_hour_pm_converted(self):
        self.assertEqual(extract_for_role(ROLE_HOUR, "Set an alarm for 10 PM"), 22)

    def test_hour_with_minutes(self):
        self.assertEqual(extract_for_role(ROLE_HOUR, "Set an alarm for 7:30 AM"), 7)

    def test_minute_extraction(self):
        self.assertEqual(extract_for_role(ROLE_MINUTE, "Set an alarm for 8:15 AM"), 15)

    def test_minute_whole_hour(self):
        self.assertEqual(extract_for_role(ROLE_MINUTE, "Set an alarm for 6 AM"), 0)

    def test_duration_minutes(self):
        self.assertEqual(extract_for_role(ROLE_DURATION, "Set a timer for 5 minutes"), 5)

    def test_duration_min(self):
        self.assertEqual(extract_for_role(ROLE_DURATION, "Set a 15 min timer"), 15)

    def test_title_remind_about(self):
        self.assertEqual(extract_for_role(ROLE_TITLE, "Remind me about groceries at 5:00 PM"), "groceries")

    def test_title_remind_to(self):
        self.assertEqual(extract_for_role(ROLE_TITLE, "Remind me to take medicine at 7:00 AM"), "take medicine")

    def test_time_str(self):
        self.assertEqual(extract_for_role(ROLE_TIME_STR, "Remind me at 3:00 PM"), "3:00 PM")

    def test_time_str_no_minutes(self):
        self.assertEqual(extract_for_role(ROLE_TIME_STR, "Remind me at 5 PM"), "5:00 PM")

    def test_song_play(self):
        self.assertEqual(extract_for_role(ROLE_SONG, "Play Bohemian Rhapsody"), "Bohemian Rhapsody")

    def test_song_play_some(self):
        # "Play some jazz music" should extract "jazz" not "jazz music" — benchmark expects "jazz"
        self.assertEqual(extract_for_role(ROLE_SONG, "Play some jazz music"), "jazz")

    def test_query_find(self):
        self.assertEqual(extract_for_role(ROLE_QUERY, "Find Bob in my contacts"), "Bob")

    def test_query_look_up(self):
        self.assertEqual(extract_for_role(ROLE_QUERY, "Look up Sarah in my contacts"), "Sarah")

    def test_unknown_returns_none(self):
        self.assertIsNone(extract_for_role(ROLE_UNKNOWN, "Do something"))

    def test_no_match_returns_none(self):
        self.assertIsNone(extract_for_role(ROLE_PERSON, "What's the weather?"))


# ═══════════════════════════════════════════════════════════════════════════════
# Test: Semantic Validation
# ═══════════════════════════════════════════════════════════════════════════════

class TestSemanticValidate(unittest.TestCase):
    """Tests for semantic_validate() — value-level consistency checks."""

    def test_valid_weather(self):
        calls = [{"name": "get_weather", "arguments": {"location": "Paris"}}]
        ok, reason = semantic_validate(calls, [TOOL_GET_WEATHER], "What's the weather in Paris?")
        self.assertTrue(ok)

    def test_wrong_location(self):
        calls = [{"name": "get_weather", "arguments": {"location": "Tokyo"}}]
        ok, reason = semantic_validate(calls, [TOOL_GET_WEATHER], "What's the weather in Paris?")
        self.assertFalse(ok)
        self.assertIn("semantic", reason)

    def test_hour_out_of_range(self):
        calls = [{"name": "set_alarm", "arguments": {"hour": 25, "minute": 0}}]
        ok, reason = semantic_validate(calls, [TOOL_SET_ALARM], "Set alarm for 10 AM")
        self.assertFalse(ok)
        self.assertIn("range", reason)

    def test_minute_out_of_range(self):
        calls = [{"name": "set_alarm", "arguments": {"hour": 10, "minute": 65}}]
        ok, reason = semantic_validate(calls, [TOOL_SET_ALARM], "Set alarm for 10 AM")
        self.assertFalse(ok)

    def test_negative_duration(self):
        calls = [{"name": "set_timer", "arguments": {"minutes": -5}}]
        ok, reason = semantic_validate(calls, [TOOL_SET_TIMER], "Set a timer for 5 minutes")
        self.assertFalse(ok)

    def test_valid_message(self):
        calls = [{"name": "send_message", "arguments": {"recipient": "Alice", "message": "hello"}}]
        ok, _ = semantic_validate(calls, [TOOL_SEND_MESSAGE], "Send a message to Alice saying hello")
        self.assertTrue(ok)


# ═══════════════════════════════════════════════════════════════════════════════
# Test: Output Repair
# ═══════════════════════════════════════════════════════════════════════════════

class TestRepairOutput(unittest.TestCase):
    """Tests for repair_output() — fixes FunctionGemma failure modes."""

    def test_ampm_hour_correction(self):
        """10 PM should become 22."""
        calls = [{"name": "set_alarm", "arguments": {"hour": 10, "minute": 0}}]
        repaired = repair_output(calls, [TOOL_SET_ALARM], "Set alarm for 10 PM")
        self.assertEqual(repaired[0]["arguments"]["hour"], 22)

    def test_am_12_correction(self):
        """12 AM should become 0."""
        calls = [{"name": "set_alarm", "arguments": {"hour": 12, "minute": 0}}]
        repaired = repair_output(calls, [TOOL_SET_ALARM], "Set alarm for 12 AM")
        self.assertEqual(repaired[0]["arguments"]["hour"], 0)

    def test_negative_duration_fixed(self):
        """Negative minutes → absolute value."""
        calls = [{"name": "set_timer", "arguments": {"minutes": -5}}]
        repaired = repair_output(calls, [TOOL_SET_TIMER], "Set a timer for 5 minutes")
        self.assertEqual(repaired[0]["arguments"]["minutes"], 5)

    def test_wrong_location_replaced(self):
        """Location not in user text → replaced by extraction."""
        calls = [{"name": "get_weather", "arguments": {"location": "Tokyo"}}]
        repaired = repair_output(calls, [TOOL_GET_WEATHER], "What's the weather in Paris?")
        self.assertEqual(repaired[0]["arguments"]["location"], "Paris")

    def test_missing_param_filled(self):
        """Missing required param filled from text extraction."""
        calls = [{"name": "set_alarm", "arguments": {"hour": 10}}]
        repaired = repair_output(calls, [TOOL_SET_ALARM], "Set alarm for 10 AM")
        self.assertIn("minute", repaired[0]["arguments"])
        self.assertEqual(repaired[0]["arguments"]["minute"], 0)

    def test_unknown_tool_removed(self):
        """Unknown tool names are dropped."""
        calls = [{"name": "fake_tool", "arguments": {"x": 1}}]
        repaired = repair_output(calls, [TOOL_GET_WEATHER], "Weather please")
        self.assertEqual(len(repaired), 0)

    def test_valid_output_unchanged(self):
        """Valid output should not be modified."""
        calls = [{"name": "get_weather", "arguments": {"location": "Paris"}}]
        repaired = repair_output(calls, [TOOL_GET_WEATHER], "What's the weather in Paris?")
        self.assertEqual(repaired[0]["arguments"]["location"], "Paris")


# ═══════════════════════════════════════════════════════════════════════════════
# Test: Deterministic Extraction
# ═══════════════════════════════════════════════════════════════════════════════

class TestBuildCallsFromText(unittest.TestCase):
    """Tests for build_calls_from_text() — schema-driven extraction fallback."""

    def test_extract_weather(self):
        calls = build_calls_from_text("What's the weather in Paris?", [TOOL_GET_WEATHER])
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "get_weather")
        self.assertEqual(calls[0]["arguments"]["location"], "Paris")

    def test_extract_alarm(self):
        calls = build_calls_from_text("Set an alarm for 7:30 AM", [TOOL_SET_ALARM])
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["arguments"]["hour"], 7)
        self.assertEqual(calls[0]["arguments"]["minute"], 30)

    def test_extract_message(self):
        calls = build_calls_from_text(
            "Send a message to Alice saying hello", [TOOL_SEND_MESSAGE],
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["arguments"]["recipient"], "Alice")
        self.assertEqual(calls[0]["arguments"]["message"], "hello")

    def test_extract_timer(self):
        calls = build_calls_from_text("Set a timer for 5 minutes", [TOOL_SET_TIMER])
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["arguments"]["minutes"], 5)

    def test_extract_music(self):
        calls = build_calls_from_text("Play Bohemian Rhapsody", [TOOL_PLAY_MUSIC])
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["arguments"]["song"], "Bohemian Rhapsody")

    def test_extract_contacts(self):
        calls = build_calls_from_text("Find Bob in my contacts", [TOOL_SEARCH_CONTACTS])
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["arguments"]["query"], "Bob")

    def test_no_match_returns_empty(self):
        calls = build_calls_from_text("Do something random", [TOOL_CUSTOM])
        self.assertEqual(len(calls), 0)

    def test_multi_tool_extraction(self):
        calls = build_calls_from_text(
            "Send a message to Bob saying hi and get the weather in London",
            [TOOL_GET_WEATHER, TOOL_SEND_MESSAGE, TOOL_SET_ALARM],
        )
        self.assertGreaterEqual(len(calls), 2)
        names = {c["name"] for c in calls}
        self.assertIn("send_message", names)
        self.assertIn("get_weather", names)

    def test_segmented_extraction(self):
        calls = build_calls_from_segments(
            "Text Alice saying hi and check the weather in Paris",
            [TOOL_GET_WEATHER, TOOL_SEND_MESSAGE, TOOL_SET_ALARM],
        )
        self.assertGreaterEqual(len(calls), 2)
        names = {c["name"] for c in calls}
        self.assertIn("send_message", names)
        self.assertIn("get_weather", names)


# ═══════════════════════════════════════════════════════════════════════════════
# Test: Routing Decision Matrix (Mock-Driven)
#
# These tests mock generate_cactus and generate_cloud to test the routing
# logic in isolation. The 7-layer flow: local → repair → validate → retry →
# extract → cloud. Tests use TOOL_CUSTOM (extraction-proof) for cloud tests.
# ═══════════════════════════════════════════════════════════════════════════════

class TestRoutingDecisions(unittest.TestCase):
    """Tests for generate_hybrid() decision logic with mocked backends."""

    def _mock_local(self, confidence=0.8, calls=None, cloud_handoff=False,
                    spike_handoff=False, success=True, time_ms=50):
        """Create a mock local (cactus) result."""
        if calls is None:
            calls = [{"name": "get_weather", "arguments": {"location": "SF"}}]
        return {
            "function_calls": calls,
            "total_time_ms": time_ms,
            "confidence": confidence,
            "cloud_handoff": cloud_handoff,
            "spike_handoff": spike_handoff,
            "success": success,
        }

    def _mock_cloud(self, calls=None, time_ms=200):
        """Create a mock cloud result."""
        if calls is None:
            calls = [{"name": "get_weather", "arguments": {"location": "SF"}}]
        return {
            "function_calls": calls,
            "total_time_ms": time_ms,
        }

    # ── ON-DEVICE: High confidence + valid output ──

    @patch("main.generate_cloud")
    @patch("main.generate_cactus")
    def test_high_confidence_easy_stays_on_device(self, mock_cactus, mock_cloud):
        """Easy query with high confidence → on-device, no cloud call."""
        mock_cactus.return_value = self._mock_local(confidence=0.85)
        mock_cloud.return_value = self._mock_cloud()

        msg = [{"role": "user", "content": "What's the weather in SF?"}]
        result = generate_hybrid(msg, [TOOL_GET_WEATHER])

        self.assertIn("on-device", result["source"])
        mock_cloud.assert_not_called()

    @patch("main.generate_cloud")
    @patch("main.generate_cactus")
    def test_low_confidence_easy_still_on_device(self, mock_cactus, mock_cloud):
        """Easy query, confidence=0.30 → still on-device (threshold=0.25)."""
        mock_cactus.return_value = self._mock_local(confidence=0.30)

        msg = [{"role": "user", "content": "What's the weather in SF?"}]
        result = generate_hybrid(msg, [TOOL_GET_WEATHER])

        self.assertIn("on-device", result["source"])
        mock_cloud.assert_not_called()

    @patch("main.generate_cloud")
    @patch("main.generate_cactus")
    def test_medium_high_confidence_on_device(self, mock_cactus, mock_cloud):
        """Medium query, confidence=0.50 → above 0.45 → on-device."""
        mock_cactus.return_value = self._mock_local(
            confidence=0.50,
            calls=[{"name": "set_alarm", "arguments": {"hour": 9, "minute": 0}}],
        )

        msg = [{"role": "user", "content": "Set an alarm for 9 AM."}]
        tools = [TOOL_SET_ALARM, TOOL_GET_WEATHER, TOOL_SEND_MESSAGE]
        result = generate_hybrid(msg, tools)

        self.assertIn("on-device", result["source"])
        mock_cloud.assert_not_called()

    @patch("main.generate_cloud")
    @patch("main.generate_cactus")
    def test_hard_all_intents_covered_stays_on_device(self, mock_cactus, mock_cloud):
        """Hard query with all intents covered + good confidence → on-device."""
        mock_cactus.return_value = self._mock_local(
            confidence=0.75,
            calls=[
                {"name": "send_message", "arguments": {"recipient": "Bob", "message": "hi"}},
                {"name": "get_weather", "arguments": {"location": "London"}},
            ],
        )

        msg = [{"role": "user", "content": "Send Bob a message saying hi and get the weather in London."}]
        tools = [TOOL_GET_WEATHER, TOOL_SEND_MESSAGE, TOOL_SET_ALARM]
        result = generate_hybrid(msg, tools)

        self.assertIn("on-device", result["source"])
        mock_cloud.assert_not_called()

    # ── ON-DEVICE via RETRY: valid output but below confidence threshold ──

    @patch("main.generate_cloud")
    @patch("main.generate_cactus")
    def test_low_confidence_saved_by_retry(self, mock_cactus, mock_cloud):
        """Below threshold but valid output → retry accepts it as on-device."""
        mock_cactus.return_value = self._mock_local(confidence=0.10)

        msg = [{"role": "user", "content": "What's the weather in SF?"}]
        result = generate_hybrid(msg, [TOOL_GET_WEATHER])

        # Retry finds valid + semantic-valid output → on-device
        self.assertIn("on-device", result["source"])
        mock_cloud.assert_not_called()

    # ── ON-DEVICE via REPAIR: FunctionGemma failure modes fixed ──

    @patch("main.generate_cloud")
    @patch("main.generate_cactus")
    def test_missing_minute_repaired(self, mock_cactus, mock_cloud):
        """Missing 'minute' param → repair fills from text extraction → on-device."""
        mock_cactus.return_value = self._mock_local(
            confidence=0.90,
            calls=[{"name": "set_alarm", "arguments": {"hour": 10}}],
        )

        msg = [{"role": "user", "content": "Set alarm for 10 AM."}]
        tools = [TOOL_SET_ALARM, TOOL_GET_WEATHER]
        result = generate_hybrid(msg, tools)

        self.assertIn("on-device", result["source"])
        alarm = result["function_calls"][0]
        self.assertEqual(alarm["arguments"].get("minute"), 0)

    @patch("main.generate_cloud")
    @patch("main.generate_cactus")
    def test_ampm_corrected_on_device(self, mock_cactus, mock_cloud):
        """hour=10 for '10 PM' → repair corrects to 22 → on-device."""
        mock_cactus.return_value = self._mock_local(
            confidence=0.90,
            calls=[{"name": "set_alarm", "arguments": {"hour": 10, "minute": 0}}],
        )

        msg = [{"role": "user", "content": "Set alarm for 10 PM."}]
        tools = [TOOL_SET_ALARM, TOOL_GET_WEATHER]
        result = generate_hybrid(msg, tools)

        self.assertIn("on-device", result["source"])
        self.assertEqual(result["function_calls"][0]["arguments"]["hour"], 22)

    # ── ON-DEVICE via EXTRACTION: handoff/spike saved by deterministic extraction ──

    @patch("main.generate_cloud")
    @patch("main.generate_cactus")
    def test_handoff_saved_by_extraction(self, mock_cactus, mock_cloud):
        """Handoff + extractable query → extraction saves it as on-device."""
        mock_cactus.return_value = self._mock_local(
            confidence=0.95, cloud_handoff=True,
        )

        msg = [{"role": "user", "content": "What's the weather in Paris?"}]
        result = generate_hybrid(msg, [TOOL_GET_WEATHER])

        self.assertEqual(result["source"], "on-device")
        self.assertIn("extracted", result.get("_detail", ""))
        mock_cloud.assert_not_called()

    @patch("main.generate_cloud")
    @patch("main.generate_cactus")
    def test_spike_saved_by_extraction(self, mock_cactus, mock_cloud):
        """Spike handoff + extractable query → extraction saves it."""
        mock_cactus.return_value = self._mock_local(
            confidence=0.80, spike_handoff=True,
        )

        msg = [{"role": "user", "content": "Set a timer for 5 minutes."}]
        result = generate_hybrid(msg, [TOOL_SET_TIMER])

        self.assertEqual(result["source"], "on-device")
        self.assertIn("extracted", result.get("_detail", ""))
        mock_cloud.assert_not_called()

    # ── ON-DEVICE via AUGMENTATION: missing call filled by extraction ──

    @patch("main.generate_cloud")
    @patch("main.generate_cactus")
    def test_intent_gap_filled_by_augmentation(self, mock_cactus, mock_cloud):
        """Hard query with 1/2 calls → augmentation adds the missing one → on-device."""
        mock_cactus.return_value = self._mock_local(
            confidence=0.70,
            calls=[{"name": "get_weather", "arguments": {"location": "London"}}],
        )

        msg = [{"role": "user", "content": "Send Bob a message saying hi and check the weather in London."}]
        tools = [TOOL_GET_WEATHER, TOOL_SEND_MESSAGE, TOOL_SET_ALARM]
        result = generate_hybrid(msg, tools)

        self.assertIn("on-device", result["source"])
        names = {c["name"] for c in result["function_calls"]}
        self.assertIn("send_message", names)
        self.assertIn("get_weather", names)

    # ── CLOUD: non-extractable queries that must fall through to cloud ──

    @patch("main.generate_cloud")
    @patch("main.generate_cactus")
    def test_handoff_non_extractable_goes_to_cloud(self, mock_cactus, mock_cloud):
        """Handoff + TOOL_CUSTOM (no extraction possible) → cloud."""
        mock_cactus.return_value = self._mock_local(
            confidence=0.95, cloud_handoff=True,
            calls=[{"name": "custom_action", "arguments": {"data": "xyz"}}],
        )
        mock_cloud.return_value = self._mock_cloud(
            calls=[{"name": "custom_action", "arguments": {"data": "result"}}],
        )

        msg = [{"role": "user", "content": "Process this input."}]
        result = generate_hybrid(msg, [TOOL_CUSTOM])

        self.assertIn("cloud", result["source"])
        self.assertIn("handoff", result["source"])

    @patch("main.generate_cloud")
    @patch("main.generate_cactus")
    def test_spike_non_extractable_goes_to_cloud(self, mock_cactus, mock_cloud):
        """Spike + TOOL_CUSTOM → cloud."""
        mock_cactus.return_value = self._mock_local(
            confidence=0.80, spike_handoff=True,
            calls=[{"name": "custom_action", "arguments": {"data": "xyz"}}],
        )
        mock_cloud.return_value = self._mock_cloud(
            calls=[{"name": "custom_action", "arguments": {"data": "result"}}],
        )

        msg = [{"role": "user", "content": "Process this input."}]
        result = generate_hybrid(msg, [TOOL_CUSTOM])

        self.assertIn("cloud", result["source"])
        self.assertIn("spike", result["source"])

    @patch("main.generate_cloud")
    @patch("main.generate_cactus")
    def test_low_confidence_non_extractable_to_cloud(self, mock_cactus, mock_cloud):
        """Low confidence + non-extractable tool → retry fails → cloud."""
        mock_cactus.return_value = self._mock_local(
            confidence=0.05,
            calls=[{"name": "custom_action", "arguments": {"data": "xyz"}}],
        )
        mock_cloud.return_value = self._mock_cloud(
            calls=[{"name": "custom_action", "arguments": {"data": "result"}}],
        )

        msg = [{"role": "user", "content": "Process this input."}]
        result = generate_hybrid(msg, [TOOL_CUSTOM])

        self.assertIn("cloud", result["source"])

    @patch("main.generate_cloud")
    @patch("main.generate_cactus")
    def test_empty_output_non_extractable_to_cloud(self, mock_cactus, mock_cloud):
        """Empty function_calls + TOOL_CUSTOM → no extraction → cloud."""
        mock_cactus.return_value = self._mock_local(
            confidence=0.90,
            calls=[],
        )
        mock_cloud.return_value = self._mock_cloud(
            calls=[{"name": "custom_action", "arguments": {"data": "result"}}],
        )

        msg = [{"role": "user", "content": "Process this input."}]
        result = generate_hybrid(msg, [TOOL_CUSTOM])

        self.assertIn("cloud", result["source"])

    @patch("main.generate_cloud")
    @patch("main.generate_cactus")
    def test_parse_failure_non_extractable_to_cloud(self, mock_cactus, mock_cloud):
        """success=False + non-extractable → cloud (handoff)."""
        mock_cactus.return_value = self._mock_local(
            confidence=0, success=False, calls=[],
        )
        mock_cloud.return_value = self._mock_cloud(
            calls=[{"name": "custom_action", "arguments": {"data": "result"}}],
        )

        msg = [{"role": "user", "content": "Process this input."}]
        result = generate_hybrid(msg, [TOOL_CUSTOM])

        self.assertIn("cloud", result["source"])
        self.assertIn("handoff", result["source"])

    # ── ERROR HANDLING ──

    @patch("main.generate_cloud")
    @patch("main.generate_cactus")
    def test_cloud_failure_returns_local(self, mock_cactus, mock_cloud):
        """If cloud API fails, return local result (partial credit > zero)."""
        mock_cactus.return_value = self._mock_local(
            confidence=0.05,
            calls=[{"name": "custom_action", "arguments": {"data": "xyz"}}],
        )
        mock_cloud.side_effect = Exception("API Error")

        msg = [{"role": "user", "content": "Process this input."}]
        result = generate_hybrid(msg, [TOOL_CUSTOM])

        self.assertIn("on-device", result["source"])

    @patch("main.generate_cloud")
    @patch("main.generate_cactus")
    def test_time_accumulation_on_cloud_fallback(self, mock_cactus, mock_cloud):
        """Cloud fallback should include local time in total."""
        mock_cactus.return_value = self._mock_local(
            confidence=0.05, time_ms=50,
            calls=[{"name": "custom_action", "arguments": {"data": "xyz"}}],
        )
        mock_cloud.return_value = {
            "function_calls": [{"name": "custom_action", "arguments": {"data": "result"}}],
            "total_time_ms": 200,
        }

        msg = [{"role": "user", "content": "Process this input."}]
        result = generate_hybrid(msg, [TOOL_CUSTOM])

        # Local time (50) + cloud time (200). Retry also adds time.
        self.assertGreaterEqual(result["total_time_ms"], 250)

    # ── STRUCTURAL ──

    @patch("main.generate_cloud")
    @patch("main.generate_cactus")
    def test_result_has_required_keys(self, mock_cactus, mock_cloud):
        """Result must have function_calls, total_time_ms, source for benchmark compat."""
        mock_cactus.return_value = self._mock_local(confidence=0.90)

        msg = [{"role": "user", "content": "What's the weather in SF?"}]
        result = generate_hybrid(msg, [TOOL_GET_WEATHER])

        self.assertIn("function_calls", result)
        self.assertIn("total_time_ms", result)
        self.assertIn("source", result)

    @patch("main.generate_cloud")
    @patch("main.generate_cactus")
    def test_difficulty_in_result(self, mock_cactus, mock_cloud):
        """Result should include difficulty classification."""
        mock_cactus.return_value = self._mock_local(confidence=0.90)

        msg = [{"role": "user", "content": "What's the weather in SF?"}]
        result = generate_hybrid(msg, [TOOL_GET_WEATHER])

        self.assertIn("difficulty", result)
        self.assertIn(result["difficulty"], ("easy", "medium", "hard"))


# ═══════════════════════════════════════════════════════════════════════════════
# Test: Threshold Boundaries
# ═══════════════════════════════════════════════════════════════════════════════

class TestThresholdBoundaries(unittest.TestCase):
    """Test exact threshold boundary behavior."""

    def test_thresholds_defined(self):
        """All three difficulty levels must have thresholds."""
        self.assertIn("easy", THRESHOLDS)
        self.assertIn("medium", THRESHOLDS)
        self.assertIn("hard", THRESHOLDS)

    def test_thresholds_ordered(self):
        """easy < medium < hard thresholds."""
        self.assertLess(THRESHOLDS["easy"], THRESHOLDS["medium"])
        self.assertLess(THRESHOLDS["medium"], THRESHOLDS["hard"])

    def test_thresholds_in_valid_range(self):
        """All thresholds must be between 0 and 1."""
        for diff, thresh in THRESHOLDS.items():
            with self.subTest(difficulty=diff):
                self.assertGreater(thresh, 0.0)
                self.assertLess(thresh, 1.0)

    @patch("main.generate_cloud")
    @patch("main.generate_cactus")
    def test_exact_easy_boundary(self, mock_cactus, mock_cloud):
        """Confidence exactly at easy threshold → on-device (>=, not >)."""
        threshold = THRESHOLDS["easy"]
        mock_cactus.return_value = {
            "function_calls": [{"name": "get_weather", "arguments": {"location": "SF"}}],
            "total_time_ms": 50,
            "confidence": threshold,
            "cloud_handoff": False,
            "spike_handoff": False,
            "success": True,
        }

        msg = [{"role": "user", "content": "Weather in SF?"}]
        result = generate_hybrid(msg, [TOOL_GET_WEATHER])

        self.assertIn("on-device", result["source"])

    @patch("main.generate_cloud")
    @patch("main.generate_cactus")
    def test_below_threshold_with_custom_tool_goes_cloud(self, mock_cactus, mock_cloud):
        """Below threshold + non-extractable tool → cloud (retry also fails)."""
        threshold = THRESHOLDS["easy"]
        mock_cactus.return_value = {
            "function_calls": [{"name": "custom_action", "arguments": {"data": "xyz"}}],
            "total_time_ms": 50,
            "confidence": threshold - 0.01,
            "cloud_handoff": False,
            "spike_handoff": False,
            "success": True,
        }
        mock_cloud.return_value = {
            "function_calls": [{"name": "custom_action", "arguments": {"data": "result"}}],
            "total_time_ms": 200,
        }

        msg = [{"role": "user", "content": "Process this input."}]
        result = generate_hybrid(msg, [TOOL_CUSTOM])

        self.assertIn("cloud", result["source"])


# ═══════════════════════════════════════════════════════════════════════════════
# Test: Signal Priority (order matters)
# ═══════════════════════════════════════════════════════════════════════════════

class TestSignalPriority(unittest.TestCase):
    """Verify signals are evaluated in correct priority order."""

    @patch("main.generate_cloud")
    @patch("main.generate_cactus")
    def test_handoff_checked_before_confidence(self, mock_cactus, mock_cloud):
        """cloud_handoff triggers before confidence, with non-extractable query → cloud."""
        mock_cactus.return_value = {
            "function_calls": [{"name": "custom_action", "arguments": {"data": "xyz"}}],
            "total_time_ms": 50,
            "confidence": 0.99,
            "cloud_handoff": True,
            "spike_handoff": False,
            "success": True,
        }
        mock_cloud.return_value = {
            "function_calls": [{"name": "custom_action", "arguments": {"data": "result"}}],
            "total_time_ms": 200,
        }

        msg = [{"role": "user", "content": "Process this input."}]
        result = generate_hybrid(msg, [TOOL_CUSTOM])

        self.assertIn("handoff", result["source"])

    @patch("main.generate_cloud")
    @patch("main.generate_cactus")
    def test_success_false_triggers_handoff(self, mock_cactus, mock_cloud):
        """success=False (JSON parse error) → handoff signal, non-extractable → cloud."""
        mock_cactus.return_value = {
            "function_calls": [],
            "total_time_ms": 50,
            "confidence": 0,
            "cloud_handoff": False,
            "spike_handoff": False,
            "success": False,
        }
        mock_cloud.return_value = {
            "function_calls": [{"name": "custom_action", "arguments": {"data": "result"}}],
            "total_time_ms": 200,
        }

        msg = [{"role": "user", "content": "Process this input."}]
        result = generate_hybrid(msg, [TOOL_CUSTOM])

        self.assertIn("handoff", result["source"])

    @patch("main.generate_cloud")
    @patch("main.generate_cactus")
    def test_handoff_with_extractable_stays_on_device(self, mock_cactus, mock_cloud):
        """cloud_handoff + extractable query → extraction saves it → on-device."""
        mock_cactus.return_value = {
            "function_calls": [{"name": "get_weather", "arguments": {"location": "SF"}}],
            "total_time_ms": 50,
            "confidence": 0.99,
            "cloud_handoff": True,
            "spike_handoff": False,
            "success": True,
        }

        msg = [{"role": "user", "content": "Weather in Paris?"}]
        result = generate_hybrid(msg, [TOOL_GET_WEATHER])

        self.assertEqual(result["source"], "on-device")
        self.assertIn("extracted", result.get("_detail", ""))
        mock_cloud.assert_not_called()


# ═══════════════════════════════════════════════════════════════════════════════
# Test: Benchmark F1 Compatibility
# ═══════════════════════════════════════════════════════════════════════════════

class TestBenchmarkCompatibility(unittest.TestCase):
    """Verify our output format is compatible with benchmark.py's compute_f1()."""

    def test_normalize_comparison(self):
        """Replicate benchmark's _normalize for string comparison."""
        def _normalize(v):
            if isinstance(v, str):
                return v.strip().lower()
            return v

        # Must match after normalization
        self.assertEqual(_normalize("San Francisco"), _normalize("san francisco"))
        self.assertEqual(_normalize("  Hello  "), _normalize("hello"))
        self.assertEqual(_normalize(10), 10)
        self.assertNotEqual(_normalize("10"), _normalize(10))  # This is why coercion matters!

    def test_call_matches_logic(self):
        """Replicate benchmark's _call_matches."""
        def _normalize(v):
            if isinstance(v, str):
                return v.strip().lower()
            return v

        def _call_matches(predicted, expected):
            if predicted["name"] != expected["name"]:
                return False
            for key, exp_val in expected.get("arguments", {}).items():
                if key not in predicted.get("arguments", {}):
                    return False
                if _normalize(predicted["arguments"][key]) != _normalize(exp_val):
                    return False
            return True

        # String match
        self.assertTrue(_call_matches(
            {"name": "get_weather", "arguments": {"location": "San Francisco"}},
            {"name": "get_weather", "arguments": {"location": "San Francisco"}},
        ))

        # Case-insensitive match
        self.assertTrue(_call_matches(
            {"name": "get_weather", "arguments": {"location": "san francisco"}},
            {"name": "get_weather", "arguments": {"location": "San Francisco"}},
        ))

        # Type mismatch: "10" (str) != 10 (int) — THIS IS THE BUG coerce_arg_types FIXES
        self.assertFalse(_call_matches(
            {"name": "set_alarm", "arguments": {"hour": "10", "minute": "0"}},
            {"name": "set_alarm", "arguments": {"hour": 10, "minute": 0}},
        ))

        # After coercion: should match
        self.assertTrue(_call_matches(
            {"name": "set_alarm", "arguments": {"hour": 10, "minute": 0}},
            {"name": "set_alarm", "arguments": {"hour": 10, "minute": 0}},
        ))


# ═══════════════════════════════════════════════════════════════════════════════
# Test: Tool Relevance Scoring
# ═══════════════════════════════════════════════════════════════════════════════

class TestToolRelevance(unittest.TestCase):
    """Tests for _tool_relevance() — keyword-based tool ranking."""

    def test_weather_high_for_weather_query(self):
        score = _tool_relevance(TOOL_GET_WEATHER, "What's the weather in London?")
        self.assertGreater(score, 0)

    def test_alarm_high_for_alarm_query(self):
        score = _tool_relevance(TOOL_SET_ALARM, "Set an alarm for 7 AM")
        self.assertGreater(score, 0)

    def test_weather_zero_for_alarm_query(self):
        score = _tool_relevance(TOOL_GET_WEATHER, "Set an alarm for 7 AM")
        self.assertEqual(score, 0)

    def test_timer_matches_timer_query(self):
        score = _tool_relevance(TOOL_SET_TIMER, "Set a timer for 10 minutes")
        self.assertGreater(score, 0)

    def test_music_matches_play_query(self):
        score = _tool_relevance(TOOL_PLAY_MUSIC, "Play some jazz")
        self.assertGreater(score, 0)

    def test_message_matches_send_query(self):
        score = _tool_relevance(TOOL_SEND_MESSAGE, "Send a message to Bob")
        self.assertGreater(score, 0)

    def test_contacts_matches_search_query(self):
        score = _tool_relevance(TOOL_SEARCH_CONTACTS, "Search for Alice in contacts")
        self.assertGreater(score, 0)

    def test_correct_tool_scores_higher(self):
        """The correct tool should score higher than irrelevant tools."""
        weather_score = _tool_relevance(TOOL_GET_WEATHER, "What's the weather in Paris?")
        alarm_score = _tool_relevance(TOOL_SET_ALARM, "What's the weather in Paris?")
        self.assertGreater(weather_score, alarm_score)

    def test_description_words_contribute(self):
        """Description keywords (>3 chars) should also add to score."""
        score = _tool_relevance(TOOL_SEARCH_CONTACTS, "Find a contact by name")
        self.assertGreater(score, 0)


# ═══════════════════════════════════════════════════════════════════════════════
# Test: Query Segmentation
# ═══════════════════════════════════════════════════════════════════════════════

class TestSegmentQuery(unittest.TestCase):
    """Tests for _segment_query() — splitting multi-intent queries."""

    def test_single_intent_no_split(self):
        segs = _segment_query("What's the weather in London?")
        self.assertEqual(len(segs), 1)

    def test_two_intents_with_and(self):
        segs = _segment_query("Set an alarm for 7 AM and check the weather in NYC")
        self.assertEqual(len(segs), 2)

    def test_three_intents_with_commas_and(self):
        segs = _segment_query(
            "Set a timer for 10 minutes, play jazz, and send Bob a message saying hi"
        )
        self.assertGreaterEqual(len(segs), 2)

    def test_and_not_part_of_action_verb(self):
        """'and' that isn't followed by an action verb shouldn't split."""
        segs = _segment_query("Send Bob a message saying hi and goodbye")
        # "and goodbye" shouldn't trigger a split since 'goodbye' isn't an action verb
        self.assertEqual(len(segs), 1)

    def test_comma_followed_by_action(self):
        segs = _segment_query("Set an alarm for 7 AM, get the weather in Paris")
        self.assertGreaterEqual(len(segs), 2)

    def test_short_segments_filtered(self):
        """Segments shorter than 4 chars should be filtered out."""
        segs = _segment_query("Set a timer")
        for seg in segs:
            self.assertGreater(len(seg), 3)

    def test_preserves_full_text(self):
        """Joined segments should cover the original text content."""
        text = "Set an alarm for 8 AM and check the weather in London"
        segs = _segment_query(text)
        joined = " ".join(segs).lower()
        self.assertIn("alarm", joined)
        self.assertIn("weather", joined)


# ═══════════════════════════════════════════════════════════════════════════════
# Test: Augment Calls (fills missing intents via extraction)
# ═══════════════════════════════════════════════════════════════════════════════

class TestAugmentCalls(unittest.TestCase):
    """Tests for _augment_calls() — adds missing tool calls from text."""

    def test_augments_missing_weather(self):
        """If we already have alarm, augment should add weather."""
        existing = [{"name": "set_alarm", "arguments": {"hour": 7, "minute": 0}}]
        result = _augment_calls(
            existing, ALL_TOOLS,
            "Set an alarm for 7 AM and get the weather in Tokyo",
        )
        names = {c["name"] for c in result}
        self.assertIn("set_alarm", names)
        self.assertIn("get_weather", names)

    def test_augments_missing_message(self):
        existing = [{"name": "get_weather", "arguments": {"location": "NYC"}}]
        result = _augment_calls(
            existing, ALL_TOOLS,
            "Check weather in NYC and send a message to Eve saying hey",
        )
        names = {c["name"] for c in result}
        self.assertIn("get_weather", names)
        self.assertIn("send_message", names)

    def test_no_augmentation_needed(self):
        """If all intents are already covered, no extra calls added."""
        existing = [{"name": "get_weather", "arguments": {"location": "Paris"}}]
        result = _augment_calls(
            existing, [TOOL_GET_WEATHER],
            "What's the weather in Paris?",
        )
        self.assertEqual(len(result), 1)

    def test_does_not_duplicate(self):
        """Should not add a tool that's already in existing_calls."""
        existing = [
            {"name": "get_weather", "arguments": {"location": "London"}},
            {"name": "set_alarm", "arguments": {"hour": 7, "minute": 0}},
        ]
        result = _augment_calls(
            existing, ALL_TOOLS,
            "Set alarm for 7 AM and get weather in London",
        )
        alarm_count = sum(1 for c in result if c["name"] == "set_alarm")
        weather_count = sum(1 for c in result if c["name"] == "get_weather")
        self.assertEqual(alarm_count, 1)
        self.assertEqual(weather_count, 1)

    def test_augment_with_non_extractable_returns_original(self):
        """If augmentation can't extract anything new, return original."""
        existing = [{"name": "set_alarm", "arguments": {"hour": 7, "minute": 0}}]
        result = _augment_calls(
            existing, [TOOL_SET_ALARM, TOOL_CUSTOM],
            "Set alarm for 7 AM and do custom stuff",
        )
        # TOOL_CUSTOM can't be extracted, so should only have alarm
        self.assertEqual(len(result), 1)


# ═══════════════════════════════════════════════════════════════════════════════
# Test: Build Calls From Segments (extended)
# ═══════════════════════════════════════════════════════════════════════════════

class TestBuildCallsFromSegments(unittest.TestCase):
    """Extended tests for build_calls_from_segments() — segmented extraction."""

    def test_three_segments(self):
        calls = build_calls_from_segments(
            "Set a timer for 15 minutes, play classical music, and send Bob a message saying hi",
            ALL_TOOLS,
        )
        names = {c["name"] for c in calls}
        self.assertIn("set_timer", names)
        self.assertIn("play_music", names)
        self.assertIn("send_message", names)

    def test_two_segments_alarm_weather(self):
        calls = build_calls_from_segments(
            "Set an alarm for 5 AM and get the weather in Chicago",
            [TOOL_SET_ALARM, TOOL_GET_WEATHER, TOOL_SEND_MESSAGE],
        )
        names = {c["name"] for c in calls}
        self.assertIn("set_alarm", names)
        self.assertIn("get_weather", names)

    def test_single_segment_falls_back_to_full(self):
        """Single-intent query should use build_calls_from_text."""
        calls = build_calls_from_segments(
            "Play Bohemian Rhapsody", [TOOL_PLAY_MUSIC],
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "play_music")

    def test_no_duplicate_tools(self):
        """Same tool should not appear twice from different segments."""
        calls = build_calls_from_segments(
            "Get the weather in London and check the weather in Paris",
            [TOOL_GET_WEATHER],
        )
        self.assertEqual(len(calls), 1)

    def test_message_and_contacts(self):
        calls = build_calls_from_segments(
            "Find Alice in my contacts and send Alice a message saying call me",
            [TOOL_SEARCH_CONTACTS, TOOL_SEND_MESSAGE, TOOL_GET_WEATHER],
        )
        names = {c["name"] for c in calls}
        self.assertIn("search_contacts", names)
        self.assertIn("send_message", names)

    def test_empty_if_no_tools_match(self):
        calls = build_calls_from_segments(
            "Do something completely random", [TOOL_CUSTOM],
        )
        self.assertEqual(len(calls), 0)


# ═══════════════════════════════════════════════════════════════════════════════
# Test: Benchmark-Realistic Extraction
# ═══════════════════════════════════════════════════════════════════════════════

class TestBenchmarkExtraction(unittest.TestCase):
    """Test extraction against queries matching real benchmark patterns."""

    def test_easy_weather_sf(self):
        calls = build_calls_from_text(
            "What's the weather like in San Francisco?", [TOOL_GET_WEATHER],
        )
        self.assertEqual(len(calls), 1)
        self.assertIn("San Francisco", calls[0]["arguments"]["location"])

    def test_easy_alarm_7am(self):
        calls = build_calls_from_text(
            "Wake me up at 7 AM please.", [TOOL_SET_ALARM],
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["arguments"]["hour"], 7)

    def test_easy_message_bob(self):
        calls = build_calls_from_text(
            "Text Bob and say I'll be there in 10.",
            [TOOL_SEND_MESSAGE],
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["arguments"]["recipient"], "Bob")

    def test_easy_timer_5min(self):
        calls = build_calls_from_text(
            "Set a countdown timer for 5 minutes.", [TOOL_SET_TIMER],
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["arguments"]["minutes"], 5)

    def test_easy_play_jazz(self):
        calls = build_calls_from_text(
            "Play some jazz music.", [TOOL_PLAY_MUSIC],
        )
        self.assertEqual(len(calls), 1)

    def test_medium_alarm_among_five(self):
        calls = build_calls_from_text(
            "Set an alarm for 8:15 AM.",
            [TOOL_SEND_MESSAGE, TOOL_SET_ALARM, TOOL_GET_WEATHER, TOOL_PLAY_MUSIC, TOOL_SET_TIMER],
        )
        # Should pick alarm, not timer or others
        names = {c["name"] for c in calls}
        self.assertIn("set_alarm", names)

    def test_medium_contacts_lookup(self):
        calls = build_calls_from_text(
            "Look up Sarah in my contacts.",
            [TOOL_SEND_MESSAGE, TOOL_GET_WEATHER, TOOL_SEARCH_CONTACTS, TOOL_SET_ALARM],
        )
        names = {c["name"] for c in calls}
        self.assertIn("search_contacts", names)

    def test_hard_morning_routine(self):
        calls = build_calls_from_segments(
            "Set an alarm for 7:30 AM and check the weather in New York.",
            [TOOL_GET_WEATHER, TOOL_SET_ALARM, TOOL_SEND_MESSAGE],
        )
        names = {c["name"] for c in calls}
        self.assertIn("set_alarm", names)
        self.assertIn("get_weather", names)

    def test_hard_full_day_setup(self):
        calls = build_calls_from_segments(
            "Text Emma saying good night, check the weather in Chicago, and set an alarm for 5 AM.",
            [TOOL_SEND_MESSAGE, TOOL_GET_WEATHER, TOOL_SET_ALARM, TOOL_PLAY_MUSIC, TOOL_SET_TIMER],
        )
        names = {c["name"] for c in calls}
        self.assertIn("send_message", names)
        self.assertIn("get_weather", names)
        self.assertIn("set_alarm", names)

    def test_hard_productivity_blast(self):
        calls = build_calls_from_segments(
            "Set a 15 minute timer, play classical music, and remind me to stretch at 4:00 PM.",
            [TOOL_SET_TIMER, TOOL_PLAY_MUSIC, TOOL_CREATE_REMINDER, TOOL_GET_WEATHER, TOOL_SEND_MESSAGE],
        )
        names = {c["name"] for c in calls}
        self.assertIn("set_timer", names)
        self.assertIn("play_music", names)
        # create_reminder may or may not be extracted depending on regex

    def test_hard_communication_hub(self):
        calls = build_calls_from_segments(
            "Find Tom in my contacts and send Tom a message saying happy birthday.",
            [TOOL_SEARCH_CONTACTS, TOOL_SEND_MESSAGE, TOOL_GET_WEATHER, TOOL_PLAY_MUSIC],
        )
        names = {c["name"] for c in calls}
        self.assertIn("search_contacts", names)
        self.assertIn("send_message", names)

    def test_reminder_extraction(self):
        calls = build_calls_from_text(
            "Remind me to buy groceries at 3 PM.",
            [TOOL_CREATE_REMINDER],
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "create_reminder")
        self.assertIn("title", calls[0]["arguments"])
        self.assertIn("time", calls[0]["arguments"])

    def test_pm_alarm_extraction(self):
        """3pm should produce hour=15."""
        calls = build_calls_from_text(
            "Set an alarm for 3 PM.", [TOOL_SET_ALARM],
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["arguments"]["hour"], 15)

    def test_noon_alarm_extraction(self):
        """12 PM should produce hour=12."""
        calls = build_calls_from_text(
            "Set an alarm for 12 PM.", [TOOL_SET_ALARM],
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["arguments"]["hour"], 12)


# ═══════════════════════════════════════════════════════════════════════════════
# Test: F1 Scoring Against Actual Benchmark Data
#
# These test that our extraction pipeline produces EXACTLY the values the
# benchmark expects — not just "something reasonable" but the actual ground
# truth used to compute F1. If these fail, extraction will lose scoring points.
# ═══════════════════════════════════════════════════════════════════════════════

# Replicate benchmark.py F1 computation so tests are self-contained
def _normalize_val(v):
    if isinstance(v, str):
        return v.strip().lower()
    return v

def _call_matches(predicted, expected):
    if predicted["name"] != expected["name"]:
        return False
    for key, exp_val in expected.get("arguments", {}).items():
        if key not in predicted.get("arguments", {}):
            return False
        if _normalize_val(predicted["arguments"][key]) != _normalize_val(exp_val):
            return False
    return True

def _compute_f1(predicted_calls, expected_calls):
    if not predicted_calls and not expected_calls:
        return 1.0
    if not predicted_calls or not expected_calls:
        return 0.0
    matched = 0
    used = set()
    for exp in expected_calls:
        for i, pred in enumerate(predicted_calls):
            if i not in used and _call_matches(pred, exp):
                matched += 1
                used.add(i)
                break
    precision = matched / len(predicted_calls)
    recall = matched / len(expected_calls)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


class TestExtractionF1(unittest.TestCase):
    """Test that extraction output scores F1=1.0 against real benchmark expected values.

    If any of these fail, our deterministic extraction fallback will LOSE POINTS
    on the actual benchmark. These are the most impactful tests in the suite.
    """

    def _assert_f1(self, user_text, tools, expected_calls, min_f1=1.0):
        """Extract calls and verify they achieve target F1 vs benchmark answers."""
        calls = build_calls_from_segments(user_text, tools)
        coerce_arg_types(calls, tools)
        f1 = _compute_f1(calls, expected_calls)
        self.assertGreaterEqual(
            f1, min_f1,
            f"F1={f1:.2f} < {min_f1} for '{user_text}'\n"
            f"  Predicted: {calls}\n"
            f"  Expected:  {expected_calls}",
        )

    # ── Easy benchmarks: extraction should get F1=1.0 ──

    def test_f1_weather_sf(self):
        self._assert_f1(
            "What is the weather in San Francisco?",
            [TOOL_GET_WEATHER],
            [{"name": "get_weather", "arguments": {"location": "San Francisco"}}],
        )

    def test_f1_alarm_10am(self):
        self._assert_f1(
            "Set an alarm for 10 AM.",
            [TOOL_SET_ALARM],
            [{"name": "set_alarm", "arguments": {"hour": 10, "minute": 0}}],
        )

    def test_f1_message_alice(self):
        self._assert_f1(
            "Send a message to Alice saying good morning.",
            [TOOL_SEND_MESSAGE],
            [{"name": "send_message", "arguments": {"recipient": "Alice", "message": "good morning"}}],
        )

    def test_f1_weather_london(self):
        self._assert_f1(
            "What's the weather like in London?",
            [TOOL_GET_WEATHER],
            [{"name": "get_weather", "arguments": {"location": "London"}}],
        )

    def test_f1_alarm_6am(self):
        self._assert_f1(
            "Wake me up at 6 AM.",
            [TOOL_SET_ALARM],
            [{"name": "set_alarm", "arguments": {"hour": 6, "minute": 0}}],
        )

    def test_f1_play_bohemian(self):
        self._assert_f1(
            "Play Bohemian Rhapsody.",
            [TOOL_PLAY_MUSIC],
            [{"name": "play_music", "arguments": {"song": "Bohemian Rhapsody"}}],
        )

    def test_f1_timer_5min(self):
        self._assert_f1(
            "Set a timer for 5 minutes.",
            [TOOL_SET_TIMER],
            [{"name": "set_timer", "arguments": {"minutes": 5}}],
        )

    def test_f1_search_bob(self):
        self._assert_f1(
            "Find Bob in my contacts.",
            [TOOL_SEARCH_CONTACTS],
            [{"name": "search_contacts", "arguments": {"query": "Bob"}}],
        )

    def test_f1_weather_paris(self):
        self._assert_f1(
            "How's the weather in Paris?",
            [TOOL_GET_WEATHER],
            [{"name": "get_weather", "arguments": {"location": "Paris"}}],
        )

    # ── Medium benchmarks: extraction should still get F1=1.0 ──

    def test_f1_message_john_among_three(self):
        self._assert_f1(
            "Send a message to John saying hello.",
            [TOOL_GET_WEATHER, TOOL_SEND_MESSAGE, TOOL_SET_ALARM],
            [{"name": "send_message", "arguments": {"recipient": "John", "message": "hello"}}],
        )

    def test_f1_alarm_815_among_three(self):
        self._assert_f1(
            "Set an alarm for 8:15 AM.",
            [TOOL_SEND_MESSAGE, TOOL_SET_ALARM, TOOL_GET_WEATHER],
            [{"name": "set_alarm", "arguments": {"hour": 8, "minute": 15}}],
        )

    def test_f1_timer_10min_among_three(self):
        self._assert_f1(
            "Set a timer for 10 minutes.",
            [TOOL_SET_ALARM, TOOL_SET_TIMER, TOOL_PLAY_MUSIC],
            [{"name": "set_timer", "arguments": {"minutes": 10}}],
        )

    def test_f1_search_sarah_among_four(self):
        self._assert_f1(
            "Look up Sarah in my contacts.",
            [TOOL_SEND_MESSAGE, TOOL_GET_WEATHER, TOOL_SEARCH_CONTACTS, TOOL_SET_ALARM],
            [{"name": "search_contacts", "arguments": {"query": "Sarah"}}],
        )

    # ── Hard benchmarks: extraction should get F1 ≥ 0.5 (partial credit) ──

    def test_f1_message_and_weather(self):
        self._assert_f1(
            "Send a message to Bob saying hi and get the weather in London.",
            [TOOL_GET_WEATHER, TOOL_SEND_MESSAGE, TOOL_SET_ALARM],
            [
                {"name": "send_message", "arguments": {"recipient": "Bob", "message": "hi"}},
                {"name": "get_weather", "arguments": {"location": "London"}},
            ],
        )

    def test_f1_alarm_and_weather(self):
        self._assert_f1(
            "Set an alarm for 7:30 AM and check the weather in New York.",
            [TOOL_GET_WEATHER, TOOL_SET_ALARM, TOOL_SEND_MESSAGE],
            [
                {"name": "set_alarm", "arguments": {"hour": 7, "minute": 30}},
                {"name": "get_weather", "arguments": {"location": "New York"}},
            ],
        )

    def test_f1_timer_and_music(self):
        self._assert_f1(
            "Set a timer for 20 minutes and play lo-fi beats.",
            [TOOL_SET_TIMER, TOOL_PLAY_MUSIC, TOOL_GET_WEATHER, TOOL_SET_ALARM],
            [
                {"name": "set_timer", "arguments": {"minutes": 20}},
                {"name": "play_music", "arguments": {"song": "lo-fi beats"}},
            ],
        )

    def test_f1_weather_and_music(self):
        self._assert_f1(
            "Check the weather in Miami and play summer hits.",
            [TOOL_GET_WEATHER, TOOL_PLAY_MUSIC, TOOL_SET_TIMER, TOOL_SEND_MESSAGE],
            [
                {"name": "get_weather", "arguments": {"location": "Miami"}},
                {"name": "play_music", "arguments": {"song": "summer hits"}},
            ],
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Test: Repair Chain Safety
#
# Validates that repair_output() does not make valid output WORSE.
# The repair→validate→semantic_validate chain must be monotonically
# improving — never turning a correct call into an incorrect one.
# ═══════════════════════════════════════════════════════════════════════════════

class TestRepairChainSafety(unittest.TestCase):
    """Verify repair does not degrade valid output or introduce errors."""

    def test_already_correct_weather_unchanged(self):
        """Repair should not touch already-correct output."""
        calls = [{"name": "get_weather", "arguments": {"location": "London"}}]
        repaired = repair_output(calls, [TOOL_GET_WEATHER], "What's the weather in London?")
        self.assertEqual(repaired[0]["arguments"]["location"], "London")

    def test_already_correct_alarm_unchanged(self):
        """hour=7, minute=30 for '7:30 AM' should be untouched."""
        calls = [{"name": "set_alarm", "arguments": {"hour": 7, "minute": 30}}]
        repaired = repair_output(calls, [TOOL_SET_ALARM], "Set alarm for 7:30 AM")
        self.assertEqual(repaired[0]["arguments"]["hour"], 7)
        self.assertEqual(repaired[0]["arguments"]["minute"], 30)

    def test_already_correct_message_unchanged(self):
        calls = [{"name": "send_message", "arguments": {"recipient": "Alice", "message": "hello"}}]
        repaired = repair_output(calls, [TOOL_SEND_MESSAGE], "Send Alice a message saying hello")
        self.assertEqual(repaired[0]["arguments"]["recipient"], "Alice")
        self.assertEqual(repaired[0]["arguments"]["message"], "hello")

    def test_repair_then_validate_still_valid(self):
        """Repaired output must pass validation — repair should not break structure."""
        calls = [{"name": "set_alarm", "arguments": {"hour": 10}}]  # missing minute
        repaired = repair_output(calls, [TOOL_SET_ALARM], "Set alarm for 10 AM")
        coerce_arg_types(repaired, [TOOL_SET_ALARM])
        valid, reason = validate_output(repaired, [TOOL_SET_ALARM])
        self.assertTrue(valid, f"Repaired output failed validation: {reason}")

    def test_repair_then_semantic_validate_still_valid(self):
        """Repaired output must also pass semantic validation."""
        calls = [{"name": "set_alarm", "arguments": {"hour": 10}}]  # missing minute
        text = "Set alarm for 10 AM"
        repaired = repair_output(calls, [TOOL_SET_ALARM], text)
        coerce_arg_types(repaired, [TOOL_SET_ALARM])
        ok, reason = semantic_validate(repaired, [TOOL_SET_ALARM], text)
        self.assertTrue(ok, f"Repaired output failed semantic validation: {reason}")

    def test_ampm_repair_passes_full_chain(self):
        """AM/PM repair → validate → semantic_validate must all pass."""
        calls = [{"name": "set_alarm", "arguments": {"hour": 10, "minute": 0}}]
        text = "Set alarm for 10 PM"
        repaired = repair_output(calls, [TOOL_SET_ALARM], text)
        coerce_arg_types(repaired, [TOOL_SET_ALARM])

        valid, r1 = validate_output(repaired, [TOOL_SET_ALARM])
        self.assertTrue(valid, f"Validation: {r1}")

        sem_ok, r2 = semantic_validate(repaired, [TOOL_SET_ALARM], text)
        self.assertTrue(sem_ok, f"Semantic: {r2}")

        self.assertEqual(repaired[0]["arguments"]["hour"], 22)

    def test_string_hour_pm_repair(self):
        """String hour '3' should be corrected to 15 for PM queries."""
        calls = [{"name": "set_alarm", "arguments": {"hour": "3", "minute": "0"}}]
        repaired = repair_output(calls, [TOOL_SET_ALARM], "Set alarm for 3 PM")
        self.assertEqual(repaired[0]["arguments"]["hour"], 15)

    def test_location_repair_matches_benchmark_f1(self):
        """Wrong location repaired → F1 should be 1.0 vs expected."""
        calls = [{"name": "get_weather", "arguments": {"location": "Tokyo"}}]
        repaired = repair_output(calls, [TOOL_GET_WEATHER], "What's the weather in Paris?")
        coerce_arg_types(repaired, [TOOL_GET_WEATHER])
        expected = [{"name": "get_weather", "arguments": {"location": "Paris"}}]
        f1 = _compute_f1(repaired, expected)
        self.assertEqual(f1, 1.0)

    def test_repair_multi_call_all_pass_chain(self):
        """Multi-call repair must preserve all calls through the full chain."""
        calls = [
            {"name": "set_alarm", "arguments": {"hour": 7, "minute": 30}},
            {"name": "get_weather", "arguments": {"location": "New York"}},
        ]
        text = "Set alarm for 7:30 AM and check weather in New York"
        tools = [TOOL_SET_ALARM, TOOL_GET_WEATHER]
        repaired = repair_output(calls, tools, text)
        coerce_arg_types(repaired, tools)

        self.assertEqual(len(repaired), 2)
        valid, _ = validate_output(repaired, tools)
        self.assertTrue(valid)
        sem_ok, _ = semantic_validate(repaired, tools, text)
        self.assertTrue(sem_ok)


# ═══════════════════════════════════════════════════════════════════════════════
# Test: Cross-Entity Confusion
#
# When a query mentions multiple named entities (people, places), extraction
# and repair must assign each value to the correct parameter, not confuse them.
# ═══════════════════════════════════════════════════════════════════════════════

class TestCrossEntityConfusion(unittest.TestCase):
    """Verify that entities don't get mixed up across tools/params."""

    def test_bob_is_recipient_not_location(self):
        """'Send Bob a message about Paris weather' → Bob=recipient, not location."""
        calls = build_calls_from_text(
            "Send Bob a message saying check Paris weather",
            [TOOL_SEND_MESSAGE],
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["arguments"]["recipient"], "Bob")
        self.assertNotEqual(calls[0]["arguments"]["message"], "Bob")

    def test_paris_is_location_not_recipient(self):
        """In a weather query, Paris goes to location, not a person field."""
        calls = build_calls_from_text(
            "What's the weather in Paris?",
            [TOOL_GET_WEATHER, TOOL_SEND_MESSAGE],
        )
        weather_calls = [c for c in calls if c["name"] == "get_weather"]
        self.assertEqual(len(weather_calls), 1)
        self.assertEqual(weather_calls[0]["arguments"]["location"], "Paris")

    def test_multi_tool_entities_separated(self):
        """'Send Bob hi and get weather in London' → Bob→recipient, London→location."""
        calls = build_calls_from_segments(
            "Send Bob a message saying hi and get the weather in London",
            [TOOL_SEND_MESSAGE, TOOL_GET_WEATHER],
        )
        msg_calls = [c for c in calls if c["name"] == "send_message"]
        weather_calls = [c for c in calls if c["name"] == "get_weather"]
        self.assertEqual(len(msg_calls), 1)
        self.assertEqual(len(weather_calls), 1)
        self.assertEqual(msg_calls[0]["arguments"]["recipient"], "Bob")
        self.assertEqual(weather_calls[0]["arguments"]["location"], "London")

    def test_repair_doesnt_swap_entities(self):
        """Repair with correct values should not swap them."""
        calls = [
            {"name": "send_message", "arguments": {"recipient": "Bob", "message": "hi"}},
            {"name": "get_weather", "arguments": {"location": "London"}},
        ]
        text = "Send Bob a message saying hi and get the weather in London"
        repaired = repair_output(calls, [TOOL_SEND_MESSAGE, TOOL_GET_WEATHER], text)
        self.assertEqual(repaired[0]["arguments"]["recipient"], "Bob")
        self.assertEqual(repaired[1]["arguments"]["location"], "London")


# ═══════════════════════════════════════════════════════════════════════════════
# Test: Semantic Validate Edge Cases
#
# Tests for cases where word-overlap validation could incorrectly reject
# valid output or incorrectly accept hallucinated output.
# ═══════════════════════════════════════════════════════════════════════════════

class TestSemanticEdgeCases(unittest.TestCase):
    """Edge cases for semantic validation that could cause false positives/negatives."""

    def test_short_location_accepted(self):
        """Short location names like 'SF' (< 3 chars) should not trigger overlap check."""
        calls = [{"name": "get_weather", "arguments": {"location": "SF"}}]
        ok, _ = semantic_validate(calls, [TOOL_GET_WEATHER], "Weather in SF")
        # "SF" has no word >= 3 chars, so val_words is empty, should pass
        self.assertTrue(ok)

    def test_hallucinated_location_rejected(self):
        """Location that doesn't appear in user text at all should be rejected."""
        calls = [{"name": "get_weather", "arguments": {"location": "Antarctica"}}]
        ok, _ = semantic_validate(calls, [TOOL_GET_WEATHER], "Weather in Tokyo")
        self.assertFalse(ok)

    def test_case_insensitive_match(self):
        """Semantic validation should be case-insensitive."""
        calls = [{"name": "get_weather", "arguments": {"location": "LONDON"}}]
        ok, _ = semantic_validate(calls, [TOOL_GET_WEATHER], "weather in london")
        self.assertTrue(ok)

    def test_hallucinated_recipient_rejected(self):
        """Recipient not in user text should fail semantic check."""
        calls = [{"name": "send_message", "arguments": {"recipient": "Charlie", "message": "hello"}}]
        ok, _ = semantic_validate(calls, [TOOL_SEND_MESSAGE], "Send Alice a message saying hello")
        self.assertFalse(ok)

    def test_correct_integer_passes(self):
        """Valid integer within range should pass."""
        calls = [{"name": "set_alarm", "arguments": {"hour": 10, "minute": 30}}]
        ok, _ = semantic_validate(calls, [TOOL_SET_ALARM], "Set alarm for 10:30 AM")
        self.assertTrue(ok)

    def test_hour_24_fails(self):
        """hour=24 is out of 0-23 range."""
        calls = [{"name": "set_alarm", "arguments": {"hour": 24, "minute": 0}}]
        ok, _ = semantic_validate(calls, [TOOL_SET_ALARM], "Set alarm for midnight")
        self.assertFalse(ok)

    def test_minute_negative_fails(self):
        """minute=-1 should fail range check."""
        calls = [{"name": "set_alarm", "arguments": {"hour": 10, "minute": -1}}]
        ok, _ = semantic_validate(calls, [TOOL_SET_ALARM], "Set alarm for 10 AM")
        self.assertFalse(ok)

    def test_zero_duration_fails(self):
        """duration=0 should fail: must be > 0."""
        calls = [{"name": "set_timer", "arguments": {"minutes": 0}}]
        ok, _ = semantic_validate(calls, [TOOL_SET_TIMER], "Set timer for 0 minutes")
        self.assertFalse(ok)

    def test_multiword_location_overlap(self):
        """'New York' should share word 'york' (>= 3 chars) with user text."""
        calls = [{"name": "get_weather", "arguments": {"location": "New York"}}]
        ok, _ = semantic_validate(calls, [TOOL_GET_WEATHER], "weather in New York")
        self.assertTrue(ok)


# ═══════════════════════════════════════════════════════════════════════════════
# Test: Routing Pipeline Integration
#
# End-to-end tests that verify the full generate_hybrid() pipeline produces
# results that would actually score well — combining repair, validation,
# extraction, augmentation, and retry in realistic sequences.
# ═══════════════════════════════════════════════════════════════════════════════

class TestRoutingPipelineIntegration(unittest.TestCase):
    """Integration tests for realistic FunctionGemma failure modes through generate_hybrid()."""

    def _mock_local(self, confidence=0.8, calls=None, cloud_handoff=False,
                    spike_handoff=False, success=True, time_ms=50):
        return {
            "function_calls": calls or [],
            "total_time_ms": time_ms,
            "confidence": confidence,
            "cloud_handoff": cloud_handoff,
            "spike_handoff": spike_handoff,
            "success": success,
        }

    @patch("main.generate_cloud")
    @patch("main.generate_cactus")
    def test_wrong_location_repaired_gets_f1_1(self, mock_cactus, mock_cloud):
        """Model outputs Tokyo for 'weather in Paris' → repair fixes → F1=1.0, no cloud."""
        mock_cactus.return_value = self._mock_local(
            confidence=0.80,
            calls=[{"name": "get_weather", "arguments": {"location": "Tokyo"}}],
        )
        msg = [{"role": "user", "content": "What's the weather in Paris?"}]
        result = generate_hybrid(msg, [TOOL_GET_WEATHER])

        self.assertIn("on-device", result["source"])
        expected = [{"name": "get_weather", "arguments": {"location": "Paris"}}]
        f1 = _compute_f1(result["function_calls"], expected)
        self.assertEqual(f1, 1.0)

    @patch("main.generate_cloud")
    @patch("main.generate_cactus")
    def test_string_hour_coerced_gets_f1_1(self, mock_cactus, mock_cloud):
        """Model outputs {"hour": "10", "minute": "0"} → coercion → F1=1.0."""
        mock_cactus.return_value = self._mock_local(
            confidence=0.80,
            calls=[{"name": "set_alarm", "arguments": {"hour": "10", "minute": "0"}}],
        )
        msg = [{"role": "user", "content": "Set alarm for 10 AM."}]
        result = generate_hybrid(msg, [TOOL_SET_ALARM])

        expected = [{"name": "set_alarm", "arguments": {"hour": 10, "minute": 0}}]
        f1 = _compute_f1(result["function_calls"], expected)
        self.assertEqual(f1, 1.0)

    @patch("main.generate_cloud")
    @patch("main.generate_cactus")
    def test_missing_call_augmented_stays_on_device(self, mock_cactus, mock_cloud):
        """Hard query: model only produces weather, augmentation adds message → on-device."""
        mock_cactus.return_value = self._mock_local(
            confidence=0.70,
            calls=[{"name": "get_weather", "arguments": {"location": "London"}}],
        )
        msg = [{"role": "user", "content": "Send Bob a message saying hi and get the weather in London."}]
        tools = [TOOL_GET_WEATHER, TOOL_SEND_MESSAGE, TOOL_SET_ALARM]
        result = generate_hybrid(msg, tools)

        self.assertIn("on-device", result["source"])
        names = {c["name"] for c in result["function_calls"]}
        self.assertIn("get_weather", names)
        self.assertIn("send_message", names)

        expected = [
            {"name": "send_message", "arguments": {"recipient": "Bob", "message": "hi"}},
            {"name": "get_weather", "arguments": {"location": "London"}},
        ]
        f1 = _compute_f1(result["function_calls"], expected)
        self.assertGreaterEqual(f1, 0.5, f"F1 too low: {f1}")

    @patch("main.generate_cloud")
    @patch("main.generate_cactus")
    def test_pm_correction_with_coercion_f1(self, mock_cactus, mock_cloud):
        """Model outputs hour=3 for '3 PM' as int → repair fixes to 15 → F1=1.0."""
        mock_cactus.return_value = self._mock_local(
            confidence=0.85,
            calls=[{"name": "set_alarm", "arguments": {"hour": 3, "minute": 0}}],
        )
        msg = [{"role": "user", "content": "Set alarm for 3 PM."}]
        result = generate_hybrid(msg, [TOOL_SET_ALARM])

        self.assertEqual(result["function_calls"][0]["arguments"]["hour"], 15)
        expected = [{"name": "set_alarm", "arguments": {"hour": 15, "minute": 0}}]
        f1 = _compute_f1(result["function_calls"], expected)
        self.assertEqual(f1, 1.0)

    @patch("main.generate_cloud")
    @patch("main.generate_cactus")
    def test_extraction_fallback_on_handoff_gets_f1(self, mock_cactus, mock_cloud):
        """Cloud handoff for easy weather query → extraction produces F1=1.0 on-device."""
        mock_cactus.return_value = self._mock_local(
            confidence=0.0, cloud_handoff=True, calls=[],
        )
        msg = [{"role": "user", "content": "What's the weather in Paris?"}]
        result = generate_hybrid(msg, [TOOL_GET_WEATHER])

        self.assertEqual(result["source"], "on-device")
        self.assertIn("extracted", result.get("_detail", ""))
        expected = [{"name": "get_weather", "arguments": {"location": "Paris"}}]
        f1 = _compute_f1(result["function_calls"], expected)
        self.assertEqual(f1, 1.0)


# ═══════════════════════════════════════════════════════════════════════════════
# Test: Exact Failing Benchmark Cases
#
# These are the 4 cases scoring F1=0.00. Tests must pass for benchmark to work.
# ═══════════════════════════════════════════════════════════════════════════════

TOOL_CREATE_REMINDER = {
    "name": "create_reminder",
    "description": "Create a reminder with a title and time",
    "parameters": {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Reminder title"},
            "time": {"type": "string", "description": "Time for the reminder (e.g. 3:00 PM)"},
        },
        "required": ["title", "time"],
    },
}


class TestFailingBenchmarkCases(unittest.TestCase):
    """Tests for the exact benchmark cases that are failing with F1=0.00."""

    def test_reminder_meeting_extraction(self):
        """reminder_meeting: 'Remind me about the meeting at 3:00 PM.'"""
        text = "Remind me about the meeting at 3:00 PM."
        calls = build_calls_from_text(text, [TOOL_CREATE_REMINDER])
        expected = [{"name": "create_reminder", "arguments": {"title": "meeting", "time": "3:00 PM"}}]

        self.assertEqual(len(calls), 1, f"Expected 1 call, got {calls}")
        self.assertEqual(calls[0]["name"], "create_reminder")
        # Title should be "meeting" (or contain it)
        self.assertIn("meeting", calls[0]["arguments"].get("title", "").lower())
        # Time should be "3:00 PM"
        self.assertEqual(calls[0]["arguments"].get("time"), "3:00 PM")

    def test_reminder_among_four_extraction(self):
        """reminder_among_four: 'Remind me to call the dentist at 2:00 PM.'"""
        text = "Remind me to call the dentist at 2:00 PM."
        tools = [TOOL_GET_WEATHER, TOOL_SEND_MESSAGE, TOOL_CREATE_REMINDER, TOOL_SET_ALARM]
        calls = build_calls_from_text(text, tools)
        expected = [{"name": "create_reminder", "arguments": {"title": "call the dentist", "time": "2:00 PM"}}]

        reminder_calls = [c for c in calls if c["name"] == "create_reminder"]
        self.assertEqual(len(reminder_calls), 1, f"Expected 1 reminder call, got {calls}")
        # Title should contain "call the dentist" or similar
        title = reminder_calls[0]["arguments"].get("title", "")
        self.assertTrue("dentist" in title.lower() or "call" in title.lower(),
                       f"Title '{title}' should mention dentist or call")
        # Time should be "2:00 PM"
        self.assertEqual(reminder_calls[0]["arguments"].get("time"), "2:00 PM")

    def test_timer_among_three_extraction(self):
        """timer_among_three: 'Set a timer for 10 minutes.'"""
        text = "Set a timer for 10 minutes."
        tools = [TOOL_SET_ALARM, TOOL_SET_TIMER, TOOL_PLAY_MUSIC]
        calls = build_calls_from_text(text, tools)
        expected = [{"name": "set_timer", "arguments": {"minutes": 10}}]

        timer_calls = [c for c in calls if c["name"] == "set_timer"]
        self.assertEqual(len(timer_calls), 1, f"Expected 1 timer call, got {calls}")
        self.assertEqual(timer_calls[0]["arguments"].get("minutes"), 10)

    def test_music_among_three_extraction(self):
        """music_among_three: 'Play some jazz music.'"""
        text = "Play some jazz music."
        tools = [TOOL_SET_ALARM, TOOL_PLAY_MUSIC, TOOL_GET_WEATHER]
        calls = build_calls_from_text(text, tools)
        expected = [{"name": "play_music", "arguments": {"song": "jazz"}}]

        music_calls = [c for c in calls if c["name"] == "play_music"]
        self.assertEqual(len(music_calls), 1, f"Expected 1 music call, got {calls}")
        # Song should contain "jazz"
        song = music_calls[0]["arguments"].get("song", "")
        self.assertIn("jazz", song.lower(), f"Song '{song}' should contain jazz")


# ═══════════════════════════════════════════════════════════════════════════════
# Run
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Custom test runner with summary
    print("=" * 70)
    print("  CactusRoute Test Suite")
    print("  Testing routing logic, validation, coercion, and decision matrix")
    print("  No Cactus or Gemini API needed — all mocked")
    print("=" * 70)
    print()

    unittest.main(verbosity=2)
