#!/usr/bin/env python3
"""
CactusRoute Demo — Adaptive Edge/Cloud Voice-to-Action Assistant
================================================================

A sophisticated demo showcasing the 7-layer schema-driven adaptive framework
that dynamically routes function calls between FunctionGemma (270M on-device)
and Gemini 2.5 Flash (cloud).

Features:
  - Curated real-world scenarios with mock function execution
  - Real-time routing decision visualization
  - Confidence bar rendering
  - Performance dashboard with per-difficulty breakdown
  - Interactive text mode for free-form queries
  - Voice-to-action mode via Cactus Whisper integration

Usage:
    python demo.py                       # Run curated scenarios
    python demo.py --interactive         # Interactive text mode
    python demo.py --voice               # Voice-to-action mode
    python demo.py --compare             # Side-by-side baseline vs ours
    python demo.py --benchmark           # Run full benchmark with details
"""

import sys, os, json, time, argparse
sys.path.insert(0, "cactus/python/src")

from main import (
    generate_hybrid, generate_cactus,
    estimate_difficulty, count_expected_intents,
    THRESHOLDS, _cleanup,
)

# ─── Rich Console (graceful fallback) ───────────────────────────────────────

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich import box
    from rich.rule import Rule
    from rich.align import Align
    console = Console()
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    class _FallbackConsole:
        def print(self, *args, **kwargs):
            text = " ".join(str(a) for a in args)
            # Strip rich markup
            import re
            text = re.sub(r'\[/?[^\]]*\]', '', text)
            print(text)
        def rule(self, title="", **kw):
            print(f"\n{'─' * 20} {title} {'─' * 20}")
    console = _FallbackConsole()


# ─── Tool Definitions ──────────────────────────────────────────────────────

TOOL_GET_WEATHER = {
    "name": "get_weather",
    "description": "Get current weather for a location",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {"type": "string", "description": "City name"}
        },
        "required": ["location"],
    },
}

TOOL_SET_ALARM = {
    "name": "set_alarm",
    "description": "Set an alarm for a given time",
    "parameters": {
        "type": "object",
        "properties": {
            "hour": {"type": "integer", "description": "Hour to set the alarm for"},
            "minute": {"type": "integer", "description": "Minute to set the alarm for"},
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
            "recipient": {"type": "string", "description": "Name of the person"},
            "message": {"type": "string", "description": "The message content"},
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
            "time": {"type": "string", "description": "Time for the reminder"},
        },
        "required": ["title", "time"],
    },
}

TOOL_SEARCH_CONTACTS = {
    "name": "search_contacts",
    "description": "Search for a contact by name",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Name to search for"},
        },
        "required": ["query"],
    },
}

TOOL_PLAY_MUSIC = {
    "name": "play_music",
    "description": "Play a song or playlist",
    "parameters": {
        "type": "object",
        "properties": {
            "song": {"type": "string", "description": "Song or playlist name"},
        },
        "required": ["song"],
    },
}

TOOL_SET_TIMER = {
    "name": "set_timer",
    "description": "Set a countdown timer",
    "parameters": {
        "type": "object",
        "properties": {
            "minutes": {"type": "integer", "description": "Number of minutes"},
        },
        "required": ["minutes"],
    },
}

ALL_TOOLS = [
    TOOL_GET_WEATHER, TOOL_SET_ALARM, TOOL_SEND_MESSAGE,
    TOOL_CREATE_REMINDER, TOOL_SEARCH_CONTACTS, TOOL_PLAY_MUSIC, TOOL_SET_TIMER,
]


# ─── Mock Function Executors ───────────────────────────────────────────────

MOCK_EXECUTORS = {
    "get_weather": lambda a: (
        f"☀️  {a.get('location', '?')}: 72°F, Sunny, Wind 8mph, Humidity 45%"
    ),
    "set_alarm": lambda a: (
        f"⏰  Alarm set for {a.get('hour', '?')}:{str(a.get('minute', 0)).zfill(2)} AM — "
        f"repeating: off"
    ),
    "send_message": lambda a: (
        f"✉️  Message delivered to {a.get('recipient', '?')}: "
        f"\"{a.get('message', '')}\""
    ),
    "create_reminder": lambda a: (
        f"📋  Reminder scheduled: \"{a.get('title', '?')}\" → {a.get('time', '?')}"
    ),
    "search_contacts": lambda a: (
        f"👤  Found: {a.get('query', '?')} — (555) 867-5309, "
        f"{a.get('query', '').lower()}@email.com"
    ),
    "play_music": lambda a: (
        f"🎵  Now playing: {a.get('song', '?')} — via Apple Music"
    ),
    "set_timer": lambda a: (
        f"⏱️  Timer started: {a.get('minutes', '?')} minutes remaining"
    ),
}


# ─── Demo Scenarios ────────────────────────────────────────────────────────

SCENARIOS = [
    {
        "name": "Quick Weather Check",
        "description": "Single-tool query — should stay 100% on-device",
        "messages": [{"role": "user", "content": "What's the weather in San Francisco?"}],
        "tools": [TOOL_GET_WEATHER],
        "expected_difficulty": "easy",
    },
    {
        "name": "Play Music",
        "description": "Single action with one tool — trivial for FunctionGemma",
        "messages": [{"role": "user", "content": "Play Bohemian Rhapsody."}],
        "tools": [TOOL_PLAY_MUSIC],
        "expected_difficulty": "easy",
    },
    {
        "name": "Smart Tool Selection",
        "description": "Must pick set_alarm from 5 options — tests tool discrimination",
        "messages": [{"role": "user", "content": "Set an alarm for 8:15 AM."}],
        "tools": [TOOL_SEND_MESSAGE, TOOL_SET_ALARM, TOOL_GET_WEATHER, TOOL_PLAY_MUSIC, TOOL_SET_TIMER],
        "expected_difficulty": "medium",
    },
    {
        "name": "Contact Lookup",
        "description": "Search contacts from 4 tool options — medium difficulty",
        "messages": [{"role": "user", "content": "Look up Sarah in my contacts."}],
        "tools": [TOOL_SEND_MESSAGE, TOOL_GET_WEATHER, TOOL_SEARCH_CONTACTS, TOOL_SET_ALARM],
        "expected_difficulty": "medium",
    },
    {
        "name": "Morning Routine",
        "description": "Multi-tool: alarm + weather — tests adaptive routing on hard",
        "messages": [{"role": "user", "content": "Set an alarm for 7:30 AM and check the weather in New York."}],
        "tools": [TOOL_GET_WEATHER, TOOL_SET_ALARM, TOOL_SEND_MESSAGE],
        "expected_difficulty": "hard",
    },
    {
        "name": "Communication Hub",
        "description": "Contact search + message — tests intent coverage validation",
        "messages": [{"role": "user", "content": "Find Tom in my contacts and send him a message saying happy birthday."}],
        "tools": [TOOL_SEARCH_CONTACTS, TOOL_SEND_MESSAGE, TOOL_GET_WEATHER, TOOL_PLAY_MUSIC],
        "expected_difficulty": "hard",
    },
    {
        "name": "Productivity Blast",
        "description": "3 tools at once — maximum difficulty, tests full pipeline",
        "messages": [{"role": "user", "content": "Set a 15 minute timer, play classical music, and remind me to stretch at 4:00 PM."}],
        "tools": [TOOL_SET_TIMER, TOOL_PLAY_MUSIC, TOOL_CREATE_REMINDER, TOOL_GET_WEATHER, TOOL_SEND_MESSAGE],
        "expected_difficulty": "hard",
    },
    {
        "name": "Full Day Setup",
        "description": "3 actions across 5 tools — stress test for 270M model",
        "messages": [{"role": "user", "content": "Text Emma saying good night, check the weather in Chicago, and set an alarm for 5 AM."}],
        "tools": [TOOL_SEND_MESSAGE, TOOL_GET_WEATHER, TOOL_SET_ALARM, TOOL_PLAY_MUSIC, TOOL_SET_TIMER],
        "expected_difficulty": "hard",
    },
]


# ─── Rendering Helpers ─────────────────────────────────────────────────────

def confidence_bar(value, width=30):
    """Render a confidence bar with color coding."""
    filled = int(value * width)
    empty = width - filled
    if value >= 0.7:
        color = "green"
    elif value >= 0.4:
        color = "yellow"
    else:
        color = "red"

    if HAS_RICH:
        bar = f"[{color}]{'█' * filled}[/{color}][dim]{'░' * empty}[/dim]"
        return f"{bar} {value:.3f}"
    else:
        return f"{'█' * filled}{'░' * empty} {value:.3f}"


def source_badge(source):
    """Render a colored source badge."""
    if "on-device" in source:
        if HAS_RICH:
            return "[bold green]⚡ ON-DEVICE[/bold green]"
        return "⚡ ON-DEVICE"
    else:
        if HAS_RICH:
            return f"[bold blue]☁️  CLOUD[/bold blue] [dim]({source.split('(')[1] if '(' in source else ''}[/dim]"
        return f"☁️  CLOUD ({source})"


def difficulty_badge(difficulty):
    """Render a colored difficulty badge."""
    colors = {"easy": "green", "medium": "yellow", "hard": "red"}
    icons = {"easy": "●", "medium": "◆", "hard": "▲"}
    color = colors.get(difficulty, "white")
    icon = icons.get(difficulty, "?")
    if HAS_RICH:
        return f"[{color}]{icon} {difficulty.upper()}[/{color}]"
    return f"{icon} {difficulty.upper()}"


# ─── Scenario Runner ──────────────────────────────────────────────────────

def run_scenario(scenario, index, total):
    """Run a single demo scenario with full visualization."""
    name = scenario["name"]
    desc = scenario["description"]
    messages = scenario["messages"]
    tools = scenario["tools"]
    user_query = messages[0]["content"]

    # Pre-flight
    difficulty = estimate_difficulty(messages, tools)
    threshold = THRESHOLDS[difficulty]
    expected_intents = count_expected_intents(messages)

    console.print()
    if HAS_RICH:
        # Scenario header
        header = Text()
        header.append(f"Scenario {index}/{total}: ", style="dim")
        header.append(name, style="bold white")
        header.append(f"  —  {desc}", style="dim italic")

        console.print(Panel(
            header,
            border_style="cyan",
            padding=(0, 2),
        ))
    else:
        console.print(f"\n{'━' * 70}")
        console.print(f"  Scenario {index}/{total}: {name}  —  {desc}")
        console.print(f"{'━' * 70}")

    # User query
    console.print(f"\n  [bold]User:[/bold] \"{user_query}\"" if HAS_RICH
                  else f"\n  User: \"{user_query}\"")

    # Pre-flight analysis
    console.print(f"\n  {'[dim]Pre-flight Analysis:[/dim]' if HAS_RICH else 'Pre-flight Analysis:'}")
    console.print(f"    Difficulty:  {difficulty_badge(difficulty)}  "
                  f"({'[dim]' if HAS_RICH else ''}{len(tools)} tools, "
                  f"{expected_intents} intent{'s' if expected_intents > 1 else ''}"
                  f"{'[/dim]' if HAS_RICH else ''})")
    console.print(f"    Threshold:   {threshold:.2f}")

    # Run hybrid
    start = time.time()
    result = generate_hybrid(messages, tools)
    wall_time = (time.time() - start) * 1000

    # Routing decision
    source = result.get("source", "unknown")
    confidence = result.get("confidence", result.get("local_confidence", 0))

    console.print(f"\n  {'[dim]Routing Decision:[/dim]' if HAS_RICH else 'Routing Decision:'}")
    console.print(f"    Confidence:  {confidence_bar(confidence)}")
    console.print(f"    Decision:    {source_badge(source)}")
    console.print(f"    Latency:     {result['total_time_ms']:.0f}ms "
                  f"({'[dim]' if HAS_RICH else ''}wall: {wall_time:.0f}ms"
                  f"{'[/dim]' if HAS_RICH else ''})")

    # Function calls + mock execution
    calls = result.get("function_calls", [])
    if calls:
        console.print(f"\n  {'[dim]Function Calls & Execution:[/dim]' if HAS_RICH else 'Function Calls & Execution:'}")
        for call in calls:
            fname = call["name"]
            args = call.get("arguments", {})
            args_str = ", ".join(f"{k}={json.dumps(v)}" for k, v in args.items())

            if HAS_RICH:
                console.print(f"    [cyan]→ {fname}[/cyan]({args_str})")
            else:
                console.print(f"    → {fname}({args_str})")

            # Execute mock function
            executor = MOCK_EXECUTORS.get(fname)
            if executor:
                console.print(f"      {executor(args)}")
    else:
        console.print(f"\n  {'[yellow]' if HAS_RICH else ''}⚠  No function calls produced"
                      f"{'[/yellow]' if HAS_RICH else ''}")

    return {
        "name": name,
        "difficulty": difficulty,
        "source": source,
        "confidence": confidence,
        "total_time_ms": result["total_time_ms"],
        "wall_time_ms": wall_time,
        "num_calls": len(calls),
        "on_device": "on-device" in source,
    }


# ─── Dashboard ─────────────────────────────────────────────────────────────

def render_dashboard(results):
    """Render a comprehensive performance dashboard."""
    console.print()

    if HAS_RICH:
        console.print(Rule("Performance Dashboard", style="bold cyan"))

        # Per-scenario table
        table = Table(
            box=box.ROUNDED,
            title="Scenario Results",
            title_style="bold",
            show_lines=True,
        )
        table.add_column("#", style="dim", width=3)
        table.add_column("Scenario", style="bold", min_width=20)
        table.add_column("Difficulty", justify="center")
        table.add_column("Confidence", justify="center")
        table.add_column("Routing", justify="center")
        table.add_column("Latency", justify="right")
        table.add_column("Calls", justify="center")

        for i, r in enumerate(results, 1):
            conf = r["confidence"]
            conf_color = "green" if conf >= 0.7 else "yellow" if conf >= 0.4 else "red"
            source_text = "[green]⚡ Edge[/green]" if r["on_device"] else "[blue]☁️ Cloud[/blue]"

            table.add_row(
                str(i),
                r["name"],
                difficulty_badge(r["difficulty"]),
                f"[{conf_color}]{conf:.3f}[/{conf_color}]",
                source_text,
                f"{r['total_time_ms']:.0f}ms",
                str(r["num_calls"]),
            )

        console.print(table)

        # Summary stats
        console.print()
        total = len(results)
        on_device = sum(1 for r in results if r["on_device"])
        avg_time = sum(r["total_time_ms"] for r in results) / total
        avg_conf = sum(r["confidence"] for r in results) / total

        summary = Table(box=box.SIMPLE, show_header=False, padding=(0, 3))
        summary.add_column("Metric", style="dim")
        summary.add_column("Value", style="bold")

        summary.add_row("On-Device Ratio", f"[bold green]{on_device}/{total} ({100*on_device/total:.0f}%)[/bold green]")
        summary.add_row("Avg Latency", f"{avg_time:.0f}ms")
        summary.add_row("Avg Confidence", f"{avg_conf:.3f}")

        # Per-difficulty breakdown
        for diff in ["easy", "medium", "hard"]:
            group = [r for r in results if r["difficulty"] == diff]
            if not group:
                continue
            grp_on = sum(1 for r in group if r["on_device"])
            grp_time = sum(r["total_time_ms"] for r in group) / len(group)
            summary.add_row(
                f"  {diff.capitalize()}",
                f"{grp_on}/{len(group)} on-device, {grp_time:.0f}ms avg",
            )

        console.print(Panel(summary, title="[bold]Summary[/bold]", border_style="green"))

    else:
        # Plain text fallback
        console.print(f"\n{'═' * 60}")
        console.print("  PERFORMANCE DASHBOARD")
        console.print(f"{'═' * 60}")

        total = len(results)
        on_device = sum(1 for r in results if r["on_device"])
        avg_time = sum(r["total_time_ms"] for r in results) / total

        for i, r in enumerate(results, 1):
            src = "EDGE" if r["on_device"] else "CLOUD"
            console.print(
                f"  {i}. {r['name']:<25} {r['difficulty']:<8} "
                f"conf={r['confidence']:.3f}  {src:<6} {r['total_time_ms']:.0f}ms"
            )

        console.print(f"\n  On-Device: {on_device}/{total} ({100*on_device/total:.0f}%)")
        console.print(f"  Avg Latency: {avg_time:.0f}ms")


# ─── Interactive Mode ──────────────────────────────────────────────────────

def interactive_mode():
    """Interactive text input mode for free-form queries."""
    console.print()
    if HAS_RICH:
        console.print(Panel(
            "[bold]Interactive Mode[/bold]\n"
            "Type natural language commands. Available tools: weather, alarm, "
            "message, reminder, contacts, music, timer.\n"
            "Type [bold cyan]quit[/bold cyan] to exit.",
            border_style="cyan",
        ))
    else:
        console.print("Interactive Mode — type commands, 'quit' to exit")
        console.print("Tools: weather, alarm, message, reminder, contacts, music, timer")

    while True:
        try:
            query = input("\n  You > ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not query or query.lower() in ("quit", "exit", "q"):
            break

        messages = [{"role": "user", "content": query}]

        run_scenario({
            "name": "Interactive",
            "description": "User query",
            "messages": messages,
            "tools": ALL_TOOLS,
            "expected_difficulty": "?",
        }, index="→", total="∞")

    console.print("\n  Goodbye! 👋\n")


# ─── Voice Mode ────────────────────────────────────────────────────────────

def voice_mode():
    """Voice-to-action mode using Cactus Whisper integration."""
    try:
        from cactus import cactus_init, cactus_transcribe, cactus_destroy
    except ImportError:
        console.print("[red]Error: cactus module not available for voice mode[/red]"
                      if HAS_RICH else "Error: cactus module not available")
        return

    whisper_path = "cactus/weights/whisper-small"
    if not os.path.exists(whisper_path):
        console.print(
            f"{'[red]' if HAS_RICH else ''}Whisper model not found at {whisper_path}. "
            f"Run: cactus download openai/whisper-small"
            f"{'[/red]' if HAS_RICH else ''}"
        )
        return

    console.print()
    if HAS_RICH:
        console.print(Panel(
            "[bold]Voice-to-Action Mode[/bold]\n"
            "Place WAV files in the current directory, or record via:\n"
            "  [cyan]arecord -f S16_LE -r 16000 -c 1 -d 5 input.wav[/cyan]\n"
            "Type the path to a WAV file, or [bold cyan]quit[/bold cyan] to exit.",
            border_style="magenta",
        ))

    whisper = cactus_init(whisper_path)
    prompt = "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>"

    try:
        while True:
            try:
                wav_path = input("\n  WAV path > ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if not wav_path or wav_path.lower() in ("quit", "exit", "q"):
                break

            if not os.path.exists(wav_path):
                console.print(f"  File not found: {wav_path}")
                continue

            # Transcribe
            console.print(f"  {'[dim]' if HAS_RICH else ''}Transcribing...{'[/dim]' if HAS_RICH else ''}")
            start = time.time()
            transcript_raw = cactus_transcribe(whisper, wav_path, prompt=prompt)
            transcribe_ms = (time.time() - start) * 1000

            try:
                transcript = json.loads(transcript_raw)["response"]
            except (json.JSONDecodeError, KeyError):
                transcript = str(transcript_raw)

            console.print(f"  {'[bold]' if HAS_RICH else ''}Heard:{' [/bold]' if HAS_RICH else ''} \"{transcript}\"  "
                          f"({'[dim]' if HAS_RICH else ''}{transcribe_ms:.0f}ms{'[/dim]' if HAS_RICH else ''})")

            # Route through hybrid
            messages = [{"role": "user", "content": transcript}]
            run_scenario({
                "name": "Voice Command",
                "description": f"Transcribed in {transcribe_ms:.0f}ms",
                "messages": messages,
                "tools": ALL_TOOLS,
                "expected_difficulty": "?",
            }, index="🎤", total="∞")

    finally:
        cactus_destroy(whisper)

    console.print("\n  Voice mode ended. 👋\n")


# ─── Comparison Mode ──────────────────────────────────────────────────────

def comparison_mode():
    """Side-by-side comparison: baseline (threshold=0.99) vs our adaptive approach."""
    console.print()
    if HAS_RICH:
        console.print(Rule("Baseline vs CactusRoute Comparison", style="bold magenta"))
    else:
        console.print("\n=== Baseline vs CactusRoute Comparison ===")

    # Test cases covering easy/medium/hard
    test_cases = [
        {
            "label": "Easy: Weather",
            "messages": [{"role": "user", "content": "What's the weather in London?"}],
            "tools": [TOOL_GET_WEATHER],
        },
        {
            "label": "Medium: Pick tool",
            "messages": [{"role": "user", "content": "Set an alarm for 9 AM."}],
            "tools": [TOOL_SEND_MESSAGE, TOOL_GET_WEATHER, TOOL_PLAY_MUSIC, TOOL_SET_TIMER, TOOL_SET_ALARM],
        },
        {
            "label": "Hard: Multi-call",
            "messages": [{"role": "user", "content": "Send a message to Bob saying hi and get the weather in London."}],
            "tools": [TOOL_GET_WEATHER, TOOL_SEND_MESSAGE, TOOL_SET_ALARM],
        },
    ]

    if HAS_RICH:
        table = Table(box=box.ROUNDED, show_lines=True, title="Routing Decisions")
        table.add_column("Query", min_width=30)
        table.add_column("Baseline\n(threshold=0.99)", justify="center", style="red")
        table.add_column("CactusRoute\n(adaptive)", justify="center", style="green")
    else:
        console.print(f"\n  {'Query':<35} {'Baseline':<15} {'CactusRoute':<15}")
        console.print(f"  {'-'*35} {'-'*15} {'-'*15}")

    for tc in test_cases:
        msgs = tc["messages"]
        tools = tc["tools"]
        query = msgs[0]["content"]

        # Our approach
        ours = generate_hybrid(msgs, tools)
        ours_source = "⚡ Edge" if "on-device" in ours.get("source", "") else "☁️ Cloud"
        ours_time = f"{ours['total_time_ms']:.0f}ms"

        # Baseline: with 0.99 threshold, almost always cloud
        local = generate_cactus(msgs, tools)
        baseline_source = "⚡ Edge" if local.get("confidence", 0) >= 0.99 else "☁️ Cloud"
        baseline_note = f"conf={local.get('confidence', 0):.2f}"

        if HAS_RICH:
            table.add_row(
                f"[bold]{tc['label']}[/bold]\n[dim]{query}[/dim]",
                f"{baseline_source}\n[dim]{baseline_note}[/dim]",
                f"{ours_source}\n[dim]{ours_time}[/dim]",
            )
        else:
            console.print(f"  {tc['label']:<35} {baseline_source:<15} {ours_source:<15}")

    if HAS_RICH:
        console.print()
        console.print(table)
        console.print()
        console.print(Panel(
            "[bold]Key Insight:[/bold] The baseline uses threshold=0.99, sending almost "
            "everything to cloud.\nCactusRoute uses a 7-layer pipeline: adaptive thresholds "
            "(0.25/0.45/0.60), schema-driven output repair,\nmulti-gate validation "
            "(structural + semantic), retry with prompt variation, and deterministic\n"
            "extraction — maximizing on-device ratio while maintaining high F1.",
            border_style="green",
        ))


# ─── Benchmark Mode ──────────────────────────────────────────────────────

def benchmark_mode():
    """Run the full benchmark with our implementation."""
    console.print()
    if HAS_RICH:
        console.print(Rule("Full Benchmark Run", style="bold yellow"))
        console.print("[dim]Running all 30 benchmark cases with CactusRoute...[/dim]\n")

    # Import and run benchmark
    from benchmark import run_benchmark
    run_benchmark()


# ─── Banner ─────────────────────────────────────────────────────────────────

def print_banner():
    """Print the demo banner."""
    banner = """
   ╔═══════════════════════════════════════════════════════════════╗
   ║                                                               ║
   ║   🌵  CactusRoute — 7-Layer Adaptive Hybrid Router            ║
   ║                                                               ║
   ║   FunctionGemma 270M (on-device)  ↔  Gemini 2.5 Flash (cloud)║
   ║                                                               ║
   ║   7-Layer Schema-Driven Pipeline:                             ║
   ║     1. Pre-flight difficulty estimation                       ║
   ║     2. Cactus handoff signals (entropy-based)                 ║
   ║     3. Schema-driven output repair (AM/PM, types)             ║
   ║     4. Multi-gate validation (structural + semantic)          ║
   ║     5. Adaptive confidence thresholds (0.25/0.45/0.60)       ║
   ║     6. Retry with alternate prompt                            ║
   ║     7. Deterministic extraction + cloud fallback              ║
   ║                                                               ║
   ╚═══════════════════════════════════════════════════════════════╝
"""
    if HAS_RICH:
        console.print(Panel(
            Align.center(Text.from_ansi(banner.strip())),
            border_style="bright_green",
            padding=0,
        ))
    else:
        print(banner)

    # Show threshold table
    if HAS_RICH:
        t = Table(box=box.SIMPLE, show_header=True, title="Adaptive Thresholds")
        t.add_column("Difficulty", style="bold")
        t.add_column("Threshold")
        t.add_column("Strategy")
        t.add_row("[green]Easy[/green]", "0.25", "1 tool → repair + validate → nearly always on-device")
        t.add_row("[yellow]Medium[/yellow]", "0.45", "Multi-tool → repair + validate + retry if needed")
        t.add_row("[red]Hard[/red]", "0.60", "Multi-call → full 7-layer pipeline, extraction fallback")
        console.print(t)


# ─── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="CactusRoute Demo — Adaptive Edge/Cloud Voice-to-Action Assistant"
    )
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Interactive text input mode")
    parser.add_argument("--voice", "-v", action="store_true",
                        help="Voice-to-action mode (requires Whisper model)")
    parser.add_argument("--compare", "-c", action="store_true",
                        help="Side-by-side baseline vs CactusRoute comparison")
    parser.add_argument("--benchmark", "-b", action="store_true",
                        help="Run full benchmark with detailed output")
    parser.add_argument("--no-banner", action="store_true",
                        help="Skip the banner")
    args = parser.parse_args()

    if not args.no_banner:
        print_banner()

    try:
        if args.interactive:
            interactive_mode()
        elif args.voice:
            voice_mode()
        elif args.compare:
            comparison_mode()
        elif args.benchmark:
            benchmark_mode()
        else:
            # Default: run curated scenarios
            results = []
            total = len(SCENARIOS)

            for i, scenario in enumerate(SCENARIOS, 1):
                r = run_scenario(scenario, i, total)
                results.append(r)

            render_dashboard(results)

    except KeyboardInterrupt:
        console.print("\n\n  Interrupted. 👋\n")
    finally:
        _cleanup()


if __name__ == "__main__":
    main()
