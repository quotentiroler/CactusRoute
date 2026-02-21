"""
CactusRoute Test Suite
======================

Comprehensive tests for all routing logic, runnable on any platform (no Cactus needed).
Tests cover:
  - Pre-flight difficulty estimation (all 30 benchmark cases)
  - Intent counting for multi-call queries
  - Type coercion (string→int, string→float)
  - Output validation (tool names, required params, empty calls)
  - Routing decision matrix (mock-driven, all 5 signals)
  - Edge cases and regressions

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
    THRESHOLDS,
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
# Test: Routing Decision Matrix (Mock-Driven)
#
# These tests mock generate_cactus and generate_cloud to test the routing
# logic in isolation. Each test verifies a specific signal triggers correctly.
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
    def test_very_low_confidence_easy_falls_to_cloud(self, mock_cactus, mock_cloud):
        """Easy query, confidence=0.10 → below 0.25 threshold → cloud."""
        mock_cactus.return_value = self._mock_local(confidence=0.10)
        mock_cloud.return_value = self._mock_cloud()

        msg = [{"role": "user", "content": "What's the weather in SF?"}]
        result = generate_hybrid(msg, [TOOL_GET_WEATHER])

        self.assertIn("cloud", result["source"])
        mock_cloud.assert_called_once()

    @patch("main.generate_cloud")
    @patch("main.generate_cactus")
    def test_medium_needs_higher_confidence(self, mock_cactus, mock_cloud):
        """Medium query, confidence=0.30 → below 0.45 threshold → cloud."""
        mock_cactus.return_value = self._mock_local(confidence=0.30)
        mock_cloud.return_value = self._mock_cloud()

        msg = [{"role": "user", "content": "Set an alarm for 9 AM."}]
        tools = [TOOL_SET_ALARM, TOOL_GET_WEATHER, TOOL_SEND_MESSAGE]
        result = generate_hybrid(msg, tools)

        self.assertIn("cloud", result["source"])

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
    def test_cloud_handoff_signal(self, mock_cactus, mock_cloud):
        """cloud_handoff=True → always route to cloud regardless of confidence."""
        mock_cactus.return_value = self._mock_local(
            confidence=0.95, cloud_handoff=True
        )
        mock_cloud.return_value = self._mock_cloud()

        msg = [{"role": "user", "content": "What's the weather in SF?"}]
        result = generate_hybrid(msg, [TOOL_GET_WEATHER])

        self.assertIn("cloud", result["source"])
        self.assertIn("handoff", result["source"])

    @patch("main.generate_cloud")
    @patch("main.generate_cactus")
    def test_spike_handoff_signal(self, mock_cactus, mock_cloud):
        """spike_handoff=True → route to cloud."""
        mock_cactus.return_value = self._mock_local(
            confidence=0.80, spike_handoff=True
        )
        mock_cloud.return_value = self._mock_cloud()

        msg = [{"role": "user", "content": "What's the weather in SF?"}]
        result = generate_hybrid(msg, [TOOL_GET_WEATHER])

        self.assertIn("cloud", result["source"])
        self.assertIn("spike", result["source"])

    @patch("main.generate_cloud")
    @patch("main.generate_cactus")
    def test_empty_output_triggers_cloud(self, mock_cactus, mock_cloud):
        """Empty function_calls → validation fails → cloud fallback."""
        mock_cactus.return_value = self._mock_local(confidence=0.90, calls=[])
        mock_cloud.return_value = self._mock_cloud()

        msg = [{"role": "user", "content": "What's the weather in SF?"}]
        result = generate_hybrid(msg, [TOOL_GET_WEATHER])

        self.assertIn("cloud", result["source"])

    @patch("main.generate_cloud")
    @patch("main.generate_cactus")
    def test_unknown_tool_triggers_cloud(self, mock_cactus, mock_cloud):
        """Invalid tool name in output → validation fails → cloud."""
        mock_cactus.return_value = self._mock_local(
            confidence=0.90,
            calls=[{"name": "nonexistent_tool", "arguments": {}}],
        )
        mock_cloud.return_value = self._mock_cloud()

        msg = [{"role": "user", "content": "What's the weather in SF?"}]
        result = generate_hybrid(msg, [TOOL_GET_WEATHER])

        self.assertIn("cloud", result["source"])
        self.assertIn("unknown-tool", result["source"])

    @patch("main.generate_cloud")
    @patch("main.generate_cactus")
    def test_missing_params_triggers_cloud(self, mock_cactus, mock_cloud):
        """Missing required params → validation fails → cloud."""
        mock_cactus.return_value = self._mock_local(
            confidence=0.90,
            calls=[{"name": "set_alarm", "arguments": {"hour": 10}}],  # missing minute
        )
        mock_cloud.return_value = self._mock_cloud(
            calls=[{"name": "set_alarm", "arguments": {"hour": 10, "minute": 0}}]
        )

        msg = [{"role": "user", "content": "Set alarm for 10 AM."}]
        tools = [TOOL_SET_ALARM, TOOL_GET_WEATHER]
        result = generate_hybrid(msg, tools)

        self.assertIn("cloud", result["source"])
        self.assertIn("missing-params", result["source"])

    @patch("main.generate_cloud")
    @patch("main.generate_cactus")
    def test_intent_gap_triggers_cloud(self, mock_cactus, mock_cloud):
        """Hard query expects 2 calls but only 1 produced → cloud fallback."""
        mock_cactus.return_value = self._mock_local(
            confidence=0.70,
            calls=[{"name": "get_weather", "arguments": {"location": "London"}}],
            # Missing the send_message call
        )
        mock_cloud.return_value = self._mock_cloud(
            calls=[
                {"name": "send_message", "arguments": {"recipient": "Bob", "message": "hi"}},
                {"name": "get_weather", "arguments": {"location": "London"}},
            ]
        )

        msg = [{"role": "user", "content": "Send Bob a message saying hi and check the weather in London."}]
        tools = [TOOL_GET_WEATHER, TOOL_SEND_MESSAGE, TOOL_SET_ALARM]
        result = generate_hybrid(msg, tools)

        self.assertIn("cloud", result["source"])
        self.assertIn("intent-gap", result["source"])

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

    @patch("main.generate_cloud")
    @patch("main.generate_cactus")
    def test_cloud_failure_returns_local(self, mock_cactus, mock_cloud):
        """If cloud API fails, return local result (partial credit > zero)."""
        mock_cactus.return_value = self._mock_local(confidence=0.10)
        mock_cloud.side_effect = Exception("API Error")

        msg = [{"role": "user", "content": "What's the weather?"}]
        result = generate_hybrid(msg, [TOOL_GET_WEATHER])

        # Should gracefully fall back to local
        self.assertIn("on-device", result["source"])

    @patch("main.generate_cloud")
    @patch("main.generate_cactus")
    def test_time_accumulation_on_fallback(self, mock_cactus, mock_cloud):
        """Cloud fallback should include local time in total."""
        mock_cactus.return_value = self._mock_local(confidence=0.10, time_ms=50)
        mock_cloud.return_value = self._mock_cloud(time_ms=200)

        msg = [{"role": "user", "content": "What's the weather?"}]
        result = generate_hybrid(msg, [TOOL_GET_WEATHER])

        self.assertEqual(result["total_time_ms"], 250)  # 50 + 200

    @patch("main.generate_cloud")
    @patch("main.generate_cactus")
    def test_result_has_required_keys(self, mock_cactus, mock_cloud):
        """Result must have function_calls, total_time_ms, source for benchmark compat."""
        mock_cactus.return_value = self._mock_local(confidence=0.90)

        msg = [{"role": "user", "content": "What's the weather?"}]
        result = generate_hybrid(msg, [TOOL_GET_WEATHER])

        self.assertIn("function_calls", result)
        self.assertIn("total_time_ms", result)
        self.assertIn("source", result)


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
            "confidence": threshold,  # Exactly at boundary
            "cloud_handoff": False,
            "spike_handoff": False,
            "success": True,
        }

        msg = [{"role": "user", "content": "Weather in SF?"}]
        result = generate_hybrid(msg, [TOOL_GET_WEATHER])

        self.assertIn("on-device", result["source"])

    @patch("main.generate_cloud")
    @patch("main.generate_cactus")
    def test_just_below_easy_boundary(self, mock_cactus, mock_cloud):
        """Confidence just below easy threshold → cloud."""
        threshold = THRESHOLDS["easy"]
        mock_cactus.return_value = {
            "function_calls": [{"name": "get_weather", "arguments": {"location": "SF"}}],
            "total_time_ms": 50,
            "confidence": threshold - 0.01,
            "cloud_handoff": False,
            "spike_handoff": False,
            "success": True,
        }
        mock_cloud.return_value = {
            "function_calls": [{"name": "get_weather", "arguments": {"location": "SF"}}],
            "total_time_ms": 200,
        }

        msg = [{"role": "user", "content": "Weather in SF?"}]
        result = generate_hybrid(msg, [TOOL_GET_WEATHER])

        self.assertIn("cloud", result["source"])


# ═══════════════════════════════════════════════════════════════════════════════
# Test: Signal Priority (order matters)
# ═══════════════════════════════════════════════════════════════════════════════

class TestSignalPriority(unittest.TestCase):
    """Verify signals are evaluated in correct priority order."""

    @patch("main.generate_cloud")
    @patch("main.generate_cactus")
    def test_handoff_checked_before_confidence(self, mock_cactus, mock_cloud):
        """cloud_handoff should trigger before confidence check, even with high conf."""
        mock_cactus.return_value = {
            "function_calls": [{"name": "get_weather", "arguments": {"location": "SF"}}],
            "total_time_ms": 50,
            "confidence": 0.99,
            "cloud_handoff": True,  # This should win
            "spike_handoff": False,
            "success": True,
        }
        mock_cloud.return_value = {
            "function_calls": [{"name": "get_weather", "arguments": {"location": "SF"}}],
            "total_time_ms": 200,
        }

        msg = [{"role": "user", "content": "Weather?"}]
        result = generate_hybrid(msg, [TOOL_GET_WEATHER])

        self.assertIn("handoff", result["source"])

    @patch("main.generate_cloud")
    @patch("main.generate_cactus")
    def test_success_false_triggers_handoff(self, mock_cactus, mock_cloud):
        """success=False (JSON parse error) → handoff signal."""
        mock_cactus.return_value = {
            "function_calls": [],
            "total_time_ms": 50,
            "confidence": 0,
            "cloud_handoff": False,
            "spike_handoff": False,
            "success": False,
        }
        mock_cloud.return_value = {
            "function_calls": [{"name": "get_weather", "arguments": {"location": "SF"}}],
            "total_time_ms": 200,
        }

        msg = [{"role": "user", "content": "Weather?"}]
        result = generate_hybrid(msg, [TOOL_GET_WEATHER])

        self.assertIn("handoff", result["source"])


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
