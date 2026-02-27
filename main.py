"""
main.py
AI Travel Planner Crew - Main Entry Point
Multi-agent system: CrewAI + Groq + Serper Dev API
"""

import os
import re
import sys
import time
import math
import json
import logging
from datetime import date, datetime
from pathlib import Path
from dotenv import load_dotenv
from crewai import Crew, Process

load_dotenv()

# Force CrewAI SQLite storage to a writable project-local path.
# This prevents readonly DB errors in constrained environments.
os.environ.setdefault("XDG_DATA_HOME", str((Path.cwd() / ".crewai_data").resolve()))

# ── Logging Setup ──────────────────────────────────────────────
logs_dir = Path("logs")
logs_dir.mkdir(exist_ok=True)

log_filename = logs_dir / f"travel_planner_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("TravelPlannerMain")

REQUIRED_SECTION_ORDER = [
    "Executive Summary",
    "Destination Overview",
    "Budget Breakdown",
    "Day-wise Itinerary",
    "Validation Summary",
    "Risk Factors",
    "Recommendations",
    "Assumptions Made",
]

SECTION_ALIASES = {
    "executive summary": "Executive Summary",
    "summary": "Executive Summary",
    "destination overview": "Destination Overview",
    "destination": "Destination Overview",
    "budget breakdown": "Budget Breakdown",
    "budget": "Budget Breakdown",
    "day wise itinerary": "Day-wise Itinerary",
    "day-wise itinerary": "Day-wise Itinerary",
    "daywise itinerary": "Day-wise Itinerary",
    "itinerary": "Day-wise Itinerary",
    "validation summary": "Validation Summary",
    "validation": "Validation Summary",
    "risk factors": "Risk Factors",
    "risks": "Risk Factors",
    "recommendations": "Recommendations",
    "assumptions made": "Assumptions Made",
    "assumptions": "Assumptions Made",
}


def _safe_float(value: str) -> float | None:
    cleaned = (value or "").strip().replace(",", "")
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _format_money(amount: float, currency: str) -> str:
    return f"{amount:,.2f} {currency}"


def _format_int(value: int) -> str:
    return f"{int(value):,}"


def _empty_llm_usage_totals() -> dict[str, int]:
    return {
        "successful_requests": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cached_prompt_tokens": 0,
    }


def _merge_llm_usage_totals(totals: dict[str, int], usage: dict | None) -> None:
    if not usage:
        return
    for key in ("successful_requests", "prompt_tokens", "completion_tokens", "total_tokens", "cached_prompt_tokens"):
        raw_value = usage.get(key, 0) if isinstance(usage, dict) else getattr(usage, key, 0)
        try:
            totals[key] += int(raw_value or 0)
        except (TypeError, ValueError):
            continue


def _build_usage_summary(llm_totals: dict[str, int], serper_tool) -> dict[str, float | int]:
    llm_api_calls = int(llm_totals.get("successful_requests", 0))
    prompt_tokens = int(llm_totals.get("prompt_tokens", 0))
    completion_tokens = int(llm_totals.get("completion_tokens", 0))
    total_tokens = int(llm_totals.get("total_tokens", 0))
    cached_prompt_tokens = int(llm_totals.get("cached_prompt_tokens", 0))

    serper_api_calls = int(getattr(serper_tool, "api_requests", 0))
    serper_cache_hits = int(getattr(serper_tool, "cache_hits", 0))
    serper_invocations = int(getattr(serper_tool, "tool_invocations", 0))
    total_api_calls = llm_api_calls + serper_api_calls

    avg_tokens_per_llm_call = (total_tokens / llm_api_calls) if llm_api_calls > 0 else 0.0
    avg_tokens_per_api_call = (total_tokens / total_api_calls) if total_api_calls > 0 else 0.0

    return {
        "llm_api_calls": llm_api_calls,
        "serper_api_calls": serper_api_calls,
        "total_api_calls": total_api_calls,
        "serper_tool_invocations": serper_invocations,
        "serper_cache_hits": serper_cache_hits,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cached_prompt_tokens": cached_prompt_tokens,
        "total_tokens": total_tokens,
        "avg_tokens_per_llm_call": round(avg_tokens_per_llm_call, 2),
        "avg_tokens_per_api_call": round(avg_tokens_per_api_call, 2),
    }


def _usage_markdown_table(usage_summary: dict[str, float | int]) -> str:
    rows = [
        ("LLM API Calls", _format_int(int(usage_summary["llm_api_calls"]))),
        ("Serper API Calls", _format_int(int(usage_summary["serper_api_calls"]))),
        ("Total API Calls", _format_int(int(usage_summary["total_api_calls"]))),
        ("Prompt Tokens", _format_int(int(usage_summary["prompt_tokens"]))),
        ("Completion Tokens", _format_int(int(usage_summary["completion_tokens"]))),
        ("Cached Prompt Tokens", _format_int(int(usage_summary["cached_prompt_tokens"]))),
        ("Total Tokens", _format_int(int(usage_summary["total_tokens"]))),
        ("Avg Tokens / LLM API Call", f"{float(usage_summary['avg_tokens_per_llm_call']):,.2f}"),
        ("Avg Tokens / API Call (Overall)", f"{float(usage_summary['avg_tokens_per_api_call']):,.2f}"),
    ]

    lines = ["## API Usage Summary", "", "| Metric | Value |", "|---|---:|"]
    for label, value in rows:
        lines.append(f"| {label} | {value} |")
    return "\n".join(lines)


def _print_usage_summary(usage_summary: dict[str, float | int]) -> None:
    print("\n📊 API Usage Summary")
    print("| Metric | Value |")
    print("|---|---:|")
    print(f"| LLM API Calls | {_format_int(int(usage_summary['llm_api_calls']))} |")
    print(f"| Serper API Calls | {_format_int(int(usage_summary['serper_api_calls']))} |")
    print(f"| Total API Calls | {_format_int(int(usage_summary['total_api_calls']))} |")
    print(f"| Prompt Tokens | {_format_int(int(usage_summary['prompt_tokens']))} |")
    print(f"| Completion Tokens | {_format_int(int(usage_summary['completion_tokens']))} |")
    print(f"| Cached Prompt Tokens | {_format_int(int(usage_summary['cached_prompt_tokens']))} |")
    print(f"| Total Tokens | {_format_int(int(usage_summary['total_tokens']))} |")
    print(f"| Avg Tokens / LLM API Call | {float(usage_summary['avg_tokens_per_llm_call']):,.2f} |")
    print(f"| Avg Tokens / API Call (Overall) | {float(usage_summary['avg_tokens_per_api_call']):,.2f} |")


def _clean_line(line: str) -> str:
    return re.sub(r"\s+", " ", (line or "").strip())


def _normalize_markdown(raw_text: str) -> str:
    lines = raw_text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    normalized: list[str] = []
    prev_blank = False

    for line in lines:
        cleaned = line.rstrip()
        if re.match(r"^\s*[-*]\s*$", cleaned):
            # Remove dangling bullets like "-" that break rendering.
            continue
        if not cleaned.strip():
            if not prev_blank:
                normalized.append("")
            prev_blank = True
            continue
        prev_blank = False
        normalized.append(cleaned)

    return "\n".join(normalized).strip()


def _canonical_section_title(raw_title: str) -> str | None:
    key = re.sub(r"[^a-z0-9\s-]", "", raw_title.lower()).strip()
    key = key.replace("daywise", "day wise")
    return SECTION_ALIASES.get(key)


def _extract_sections(raw_text: str) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {name: [] for name in REQUIRED_SECTION_ORDER}
    current: str | None = None

    for line in raw_text.splitlines():
        header_match = re.match(r"^\s*##\s+(.+?)\s*$", line)
        if header_match:
            current = _canonical_section_title(header_match.group(1))
            continue
        if current:
            sections[current].append(line)

    return sections


def _compact_text_block(lines: list[str], fallback: str, max_lines: int = 4) -> str:
    cleaned = [_clean_line(line) for line in lines if _clean_line(line)]
    if not cleaned:
        return fallback

    bullet_lines = []
    for line in cleaned:
        if line.startswith(("-", "*")):
            bullet_lines.append("- " + line.lstrip("-* ").strip())
    if bullet_lines:
        return "\n".join(bullet_lines[:max_lines])

    merged = " ".join(cleaned)
    merged = re.sub(r"\s+", " ", merged).strip()
    if len(merged) > 420:
        merged = merged[:417].rstrip() + "..."
    return merged


def _extract_planned_spend_from_budget(lines: list[str]) -> float | None:
    for line in lines:
        table_match = re.search(
            r"\|\s*\**total\**\s*\|\s*\$?\s*([0-9][0-9,]*(?:\.[0-9]+)?)",
            line,
            flags=re.IGNORECASE,
        )
        if table_match:
            parsed = _safe_float(table_match.group(1))
            if parsed is not None and parsed > 0:
                return parsed

    blob = " ".join(lines)
    inline_match = re.search(
        r"total[^0-9]{0,12}\$?\s*([0-9][0-9,]*(?:\.[0-9]+)?)",
        blob,
        flags=re.IGNORECASE,
    )
    if inline_match:
        parsed = _safe_float(inline_match.group(1))
        if parsed is not None and parsed > 0:
            return parsed
    return None


def _build_budget_breakdown(lines: list[str], user_input: dict) -> str:
    total_budget = float(user_input.get("budget", 0.0) or 0.0)
    currency = user_input.get("currency", "USD")

    original_planned = _extract_planned_spend_from_budget(lines)
    if original_planned is None:
        original_planned = round(total_budget * 0.90, 2)

    min_target = round(total_budget * 0.85, 2)
    planned_spend = min(total_budget, max(min_target, original_planned))
    remaining = max(0.0, total_budget - planned_spend)

    categories = {
        "Accommodation": round(planned_spend * 0.42, 2),
        "Food": round(planned_spend * 0.22, 2),
        "Transportation": round(planned_spend * 0.16, 2),
        "Activities": round(planned_spend * 0.12, 2),
        "Buffer": round(planned_spend * 0.08, 2),
    }
    category_total = sum(categories.values())
    categories["Buffer"] = round(categories["Buffer"] + (planned_spend - category_total), 2)

    section_lines = [
        "### Budget Allocation",
        "| Category | Amount |",
        "|---|---:|",
    ]
    for category, amount in categories.items():
        section_lines.append(f"| {category} | {_format_money(amount, currency)} |")
    section_lines.append(f"| **Planned Spend** | **{_format_money(planned_spend, currency)}** |")
    section_lines.append(f"| **Remaining Budget** | **{_format_money(remaining, currency)}** |")
    section_lines.append("")
    section_lines.append(
        f"- Spend utilization target: {((planned_spend / total_budget) * 100):.1f}% of total budget."
        if total_budget > 0
        else "- Spend utilization target: N/A."
    )

    if original_planned < total_budget * 0.75:
        upgrade_pool = max(0.0, planned_spend - original_planned)
        section_lines.extend(
            [
                "",
                "### Suggested Upgrades (Within Budget)",
                f"- Better stay quality/location: {_format_money(round(upgrade_pool * 0.45, 2), currency)}",
                f"- Extra paid attractions/day trips: {_format_money(round(upgrade_pool * 0.35, 2), currency)}",
                f"- Halal dining comfort margin: {_format_money(round(upgrade_pool * 0.20, 2), currency)}",
            ]
        )

    return "\n".join(section_lines).strip()


def _extract_day_blocks(lines: list[str]) -> dict[int, dict[str, list[str] | str]]:
    day_blocks: dict[int, dict[str, list[str] | str]] = {}
    current_day: int | None = None

    for raw_line in lines:
        line = raw_line.strip()
        day_match = re.match(r"^(?:[-*]\s*)?Day\s+(\d+)\s*[:\-]?\s*(.*)$", line, flags=re.IGNORECASE)
        if day_match:
            current_day = int(day_match.group(1))
            suffix = day_match.group(2).strip()
            title = f"Day {current_day}" + (f": {suffix}" if suffix else "")
            day_blocks[current_day] = {"title": title, "lines": []}
            continue
        if current_day is not None:
            day_blocks[current_day]["lines"].append(line)

    return day_blocks


def _extract_slot_content(lines: list[str], slot: str) -> str | None:
    direct_pattern = re.compile(rf"^(?:[-*]\s*)?{slot}\s*[:\-]\s*(.+)$", flags=re.IGNORECASE)
    loose_pattern = re.compile(rf"\b{slot}\b", flags=re.IGNORECASE)

    for line in lines:
        direct = direct_pattern.match(line.strip())
        if direct and direct.group(1).strip():
            return direct.group(1).strip().rstrip(".")

    for line in lines:
        if loose_pattern.search(line):
            cleaned = re.sub(r"^(?:[-*]\s*)?", "", line).strip()
            cleaned = re.sub(rf"^{slot}\s*[:\-]?\s*", "", cleaned, flags=re.IGNORECASE).strip()
            if cleaned:
                return cleaned.rstrip(".")

    return None


def _fallback_slot(slot: str, day: int, duration_days: int, destination: str, preferences: str) -> str:
    pref_text = preferences if preferences else "local highlights"
    if day == 1:
        defaults = {
            "Morning": f"Arrive and check in near central {destination}",
            "Afternoon": f"Easy orientation walk and nearby landmark visit in {destination}",
            "Evening": f"Halal-friendly dinner and early rest ({pref_text})",
        }
        return defaults[slot]
    if day == duration_days:
        defaults = {
            "Morning": f"Light activity near hotel in {destination}",
            "Afternoon": "Pack, souvenir stop, and transfer prep",
            "Evening": "Departure transfer or final local meal",
        }
        return defaults[slot]

    defaults = {
        "Morning": "Primary attraction visit",
        "Afternoon": "Nearby secondary stop and lunch",
        "Evening": "Local neighborhood exploration and dinner",
    }
    return defaults[slot]


def _build_day_itinerary(day_blocks: dict[int, dict[str, list[str] | str]], user_input: dict) -> tuple[str, list[int]]:
    duration_days = int(user_input.get("duration_days", 5) or 5)
    destination = user_input.get("destination", "Destination")
    preferences = user_input.get("preferences", "general sightseeing")
    currency = user_input.get("currency", "USD")
    total_budget = float(user_input.get("budget", 0.0) or 0.0)
    base_day_cost = (total_budget * 0.90 / max(duration_days, 1)) if total_budget > 0 else 0

    output_lines: list[str] = []
    autofilled_days: list[int] = []

    for day in range(1, duration_days + 1):
        label = "Arrival & Check-in" if day == 1 else "Departure & Wrap-up" if day == duration_days else "Explore"
        output_lines.append(f"### Day {day}: {label}")

        block = day_blocks.get(day, {"lines": []})
        lines = [line for line in block.get("lines", []) if isinstance(line, str)]

        missing_any = False
        for slot in ("Morning", "Afternoon", "Evening"):
            content = _extract_slot_content(lines, slot)
            if not content:
                missing_any = True
                content = _fallback_slot(slot, day, duration_days, destination, preferences)
            output_lines.append(f"- {slot}: {content}.")

        cost_line = None
        for line in lines:
            if re.search(r"cost|estimated|\$|usd|jpy|¥", line, flags=re.IGNORECASE):
                cost_line = re.sub(r"^(?:[-*]\s*)?", "", line).strip().rstrip(".")
                break

        if not cost_line:
            factor = 0.8 if day == 1 else 0.75 if day == duration_days else 1.0
            estimated = round(base_day_cost * factor, 2) if base_day_cost > 0 else 0.0
            cost_line = f"Estimated cost: ~{estimated:.0f} {currency}"
            missing_any = True

        if missing_any:
            autofilled_days.append(day)
            cost_line = f"{cost_line}. Assumption note: auto-filled short plan."
        output_lines.append(f"- {cost_line}")
        output_lines.append("")

    return "\n".join(output_lines).strip(), autofilled_days


def compile_report(raw_result: str, user_input: dict) -> str:
    """Deterministically compile a complete, clean report with all days present."""
    normalized = _normalize_markdown(raw_result or "")
    sections = _extract_sections(normalized)
    all_lines = normalized.splitlines()

    destination = user_input.get("destination", "Destination")
    travel_dates = user_input.get("travel_dates", "Dates not provided")
    duration_days = int(user_input.get("duration_days", 5) or 5)
    budget = float(user_input.get("budget", 0.0) or 0.0)
    currency = user_input.get("currency", "USD")
    preferences = user_input.get("preferences", "general sightseeing")

    exec_summary = _compact_text_block(
        sections["Executive Summary"],
        (
            f"{destination} trip plan for {travel_dates} ({duration_days} days), "
            f"balanced for {preferences} within {_format_money(budget, currency)}."
        ),
    )
    destination_overview = _compact_text_block(
        sections["Destination Overview"],
        f"{destination} offers a mix of key attractions, local culture, and practical travel access.",
        max_lines=5,
    )

    budget_breakdown = _build_budget_breakdown(sections["Budget Breakdown"], user_input)

    itinerary_source = sections["Day-wise Itinerary"]
    if not any(re.match(r"^(?:[-*]\s*)?Day\s+\d+", line.strip(), flags=re.IGNORECASE) for line in itinerary_source):
        itinerary_source = all_lines
    day_blocks = _extract_day_blocks(itinerary_source)
    itinerary_text, autofilled_days = _build_day_itinerary(day_blocks, user_input)

    validation_summary = _compact_text_block(
        sections["Validation Summary"],
        "- Budget pacing, day sequencing, and travel pacing were checked for feasibility.",
        max_lines=4,
    )
    risk_factors = _compact_text_block(
        sections["Risk Factors"],
        "- Possible weather swings.\n- Attraction timing changes.\n- Local transport timing variability.",
        max_lines=4,
    )
    recommendations = _compact_text_block(
        sections["Recommendations"],
        "- Reserve key attractions early.\n- Keep one flexible backup option per day.\n- Confirm halal options each day.",
        max_lines=4,
    )
    assumptions_default = [
        "- Day structure is intentionally short (Morning/Afternoon/Evening + cost) to reduce token usage.",
        "- Budget utilization is tuned to 85-100% of the declared budget.",
    ]
    if autofilled_days:
        assumptions_default.append(
            f"- Auto-filled days to ensure full coverage: {', '.join(str(d) for d in sorted(set(autofilled_days)))}."
        )
    assumptions = _compact_text_block(sections["Assumptions Made"], "\n".join(assumptions_default), max_lines=6)

    compiled = [
        f"# Travel Plan: {destination}",
        "",
        "## Executive Summary",
        exec_summary,
        "",
        "## Destination Overview",
        destination_overview,
        "",
        "## Budget Breakdown",
        budget_breakdown,
        "",
        "## Day-wise Itinerary",
        itinerary_text,
        "",
        "## Validation Summary",
        validation_summary,
        "",
        "## Risk Factors",
        risk_factors,
        "",
        "## Recommendations",
        recommendations,
        "",
        "## Assumptions Made",
        assumptions,
    ]

    return _normalize_markdown("\n".join(compiled))


def _collect_keys(primary_key: str, pool_key: str, indexed_prefix: str) -> list[str]:
    """Collect API keys from single, pooled, and indexed env vars."""
    keys: list[str] = []

    def _add_key(raw_value: str):
        key = (raw_value or "").strip()
        lower = key.lower()
        if lower in {
            "your_groq_api_key_here",
            "your_key_here",
        }:
            return
        if key and key not in keys:
            keys.append(key)

    _add_key(os.getenv(primary_key, ""))

    raw_pool = os.getenv(pool_key, "").strip()
    if raw_pool:
        for part in raw_pool.split(","):
            _add_key(part)

    for idx in range(1, 11):
        _add_key(os.getenv(f"{indexed_prefix}_{idx}", ""))

    return keys


def get_groq_api_keys() -> list[str]:
    """Collect Groq API keys from single, pooled, and indexed env vars."""
    return _collect_keys("GROQ_API_KEY", "GROQ_API_KEYS", "GROQ_API_KEY")


def validate_env():
    missing = []
    if not os.getenv("SERPER_API_KEY"):
        missing.append("SERPER_API_KEY")
    groq_keys = get_groq_api_keys()
    if not groq_keys:
        missing.append("GROQ_API_KEY (or GROQ_API_KEYS)")
    if missing:
        logger.error(f"Missing environment variables: {missing}")
        print("\n❌ ERROR: Missing API keys!")
        print("Please create a .env file with:")
        for key in missing:
            print(f"  {key}=your_key_here")
        print("\nSee .env.example for reference.")
        sys.exit(1)
    logger.info(f"[ENV] Groq key pool ready ✓ | keys={len(groq_keys)}")
    logger.info("[ENV] All required API keys found ✓")


def calculate_duration(travel_dates: str) -> int:
    """Calculate trip duration from common date-range formats.

    Supported formats:
      - YYYY-MM-DD / YYYY-MM-DD
      - Month DD-DD, YYYY          (e.g. March 20-25, 2026)
      - Month DD - Month DD, YYYY  (e.g. December 28 - January 3, 2026)

    Cross-year ranges (e.g. December 28 - January 3) are handled by
    checking if the end month is earlier than the start month — in that
    case the end date is placed in the following year automatically.

    Returns 0 if the date string cannot be confidently parsed.
    """
    normalized = re.sub(r"\s+to\s+", " - ", travel_dates, flags=re.IGNORECASE).strip()

    def _duration(start: date, end: date) -> int:
        """Return inclusive day count, or 0 if range is invalid."""
        if end < start:
            return 0
        days = (end - start).days + 1
        return days if 1 <= days <= 60 else 0

    # ── Format 1: YYYY-MM-DD - YYYY-MM-DD ─────────────────────
    iso_dates = re.findall(r"\b\d{4}-\d{1,2}-\d{1,2}\b", normalized)
    if len(iso_dates) >= 2:
        try:
            start = datetime.strptime(iso_dates[0], "%Y-%m-%d").date()
            end   = datetime.strptime(iso_dates[1], "%Y-%m-%d").date()
            days  = _duration(start, end)
            if days:
                return days
        except ValueError:
            pass

    # ── Format 2: Month DD[-Month DD], YYYY ───────────────────
    month_pattern = re.search(
        r"(?P<m1>[A-Za-z]+)\s+(?P<d1>\d{1,2})\s*[-–]\s*"
        r"(?:(?P<m2>[A-Za-z]+)\s+)?(?P<d2>\d{1,2})(?:,\s*(?P<y>\d{4}))?",
        normalized,
    )
    if month_pattern:
        def _parse_month(name: str) -> int | None:
            for fmt in ("%B", "%b"):
                try:
                    return datetime.strptime(name, fmt).month
                except ValueError:
                    continue
            return None

        month_1 = _parse_month(month_pattern.group("m1"))
        month_2_text = month_pattern.group("m2") or month_pattern.group("m1")
        month_2 = _parse_month(month_2_text)

        if month_1 and month_2:
            year  = int(month_pattern.group("y") or datetime.now().year)
            day_1 = int(month_pattern.group("d1"))
            day_2 = int(month_pattern.group("d2"))
            try:
                start = date(year, month_1, day_1)

                # Cross-year range: end month is earlier than start month
                # e.g. December 28 - January 3 → end is in year+1
                end_year = year + 1 if month_2 < month_1 else year
                end = date(end_year, month_2, day_2)

                days = _duration(start, end)
                if days:
                    return days
            except ValueError:
                pass

    # Could not parse confidently
    return 0


def get_user_input() -> dict:
    print("\n" + "="*60)
    print("  🌍  AI TRAVEL PLANNER CREW  🌍")
    print("  Powered by CrewAI + Groq + Serper")
    print("="*60 + "\n")

    destination = input("📍 Destination (e.g., Tokyo, Japan): ").strip()
    if not destination:
        destination = "Tokyo, Japan"

    travel_dates = input("📅 Travel Dates (e.g., March 15-22, 2025): ").strip()
    if not travel_dates:
        travel_dates = "March 15-22, 2025"

    # Duration is derived automatically from dates.
    # Keep only 5 user inputs: destination, dates, budget, currency, preferences.
    duration_days = calculate_duration(travel_dates)
    if duration_days <= 0:
        duration_days = 5
        print("   Could not parse dates; using default duration: 5 days")

    try:
        budget = float(input("💰 Total budget (numeric, e.g., 2000): ").strip())
    except ValueError:
        budget = 2000
        print("   Using default: $2000")
    if budget <= 0:
        budget = 2000
        print("   Budget must be positive; using default: $2000")

    currency = input("💱 Currency (e.g., USD, EUR, GBP) [default: USD]: ").strip().upper()
    if not currency:
        currency = "USD"

    preferences = input("🎯 Preferences (optional, e.g., history, food, adventure): ").strip()
    if not preferences:
        preferences = "general sightseeing"

    user_input = {
        "destination": destination,
        "travel_dates": travel_dates,
        "duration_days": duration_days,
        "budget": budget,
        "currency": currency,
        "preferences": preferences
    }

    print("\n📋 Trip Summary:")
    print(f"   Destination  : {destination}")
    print(f"   Dates        : {travel_dates}")
    print(f"   Duration     : {duration_days} days")
    print(f"   Budget       : {budget} {currency}")
    print(f"   Preferences  : {preferences}")
    print()
    return user_input


def save_output(result: str, user_input: dict, usage_summary: dict[str, float | int] | None = None):
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(exist_ok=True)

    dest_safe = re.sub(r"[^A-Za-z0-9._-]+", "_", user_input["destination"]).strip("._")
    if not dest_safe:
        dest_safe = "destination"
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    md_path    = outputs_dir / f"travel_plan_{dest_safe}_{timestamp}.md"
    json_path  = outputs_dir / f"travel_plan_{dest_safe}_{timestamp}.json"

    # ── Build full Markdown output ─────────────────────────────
    budget_value = float(user_input.get("budget", 0.0) or 0.0)
    currency = user_input.get("currency", "USD")
    usage_block = _usage_markdown_table(usage_summary) if usage_summary else ""
    md_content = f"""# AI Travel Plan: {user_input['destination']}

| Field | Details |
|---|---|
| Generated | {datetime.now().strftime('%B %d, %Y %H:%M')} |
| Destination | {user_input['destination']} |
| Dates | {user_input['travel_dates']} |
| Duration | {user_input['duration_days']} days |
| Budget | {_format_money(budget_value, currency)} |
| Preferences | {user_input.get('preferences', 'N/A')} |

---

{result}

{usage_block}

---
*Generated by AI Travel Planner Crew — CrewAI + Groq + Serper*
"""

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    # Also write to a fixed output.md for easy access
    fixed_md_path = outputs_dir / "output.md"
    with open(fixed_md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "generated_at": datetime.now().isoformat(),
            "user_input": user_input,
            "usage_summary": usage_summary or {},
            "output_file": str(md_path),
            "fixed_output_file": str(fixed_md_path),
            "log_file": str(log_filename)
        }, f, indent=2)

    logger.info(f"[Output] Saved to: {md_path}")
    logger.info(f"[Output] Also saved to: {fixed_md_path}")
    return md_path, fixed_md_path

def parse_int_env(name: str, default: int) -> int:
    """Parse integer env var with safe fallback."""
    value = os.getenv(name, "").strip()
    if not value:
        return default
    try:
        parsed = int(value)
        return parsed if parsed > 0 else default
    except ValueError:
        return default


def get_model_candidates() -> list[str]:
    """Return ordered, deduplicated Groq model candidates."""
    primary = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile").strip()
    fallback_raw = os.getenv("GROQ_MODEL_FALLBACKS", "")
    fallback_models = [m.strip() for m in fallback_raw.split(",") if m.strip()]

    models: list[str] = []
    for model in [primary] + fallback_models:
        if model and model not in models:
            models.append(model)
    return models


def is_retryable_model_error(exc: Exception) -> bool:
    """Identify transient/provider errors worth retrying."""
    if is_model_decommissioned_error(exc):
        return False
    message = str(exc).lower()
    retry_markers = [
        "tool_use_failed",
        "failed_generation",
        "failed to call a function",
        "rate_limit_exceeded",
        "429",
        "rate limit",
        "none or empty response",
        "invalid response from llm call - none or empty",
        "timeout",
        "timed out",
        "connection",
        "api_connection_error",
        "temporarily unavailable",
        "service unavailable",
        "502",
        "503",
        "504",
    ]
    return any(marker in message for marker in retry_markers)


def is_rate_limit_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "rate_limit_exceeded" in message or "rate limit" in message or "429" in message


def is_auth_error(exc: Exception) -> bool:
    message = str(exc).lower()
    markers = [
        "invalid api key",
        "incorrect api key",
        "authentication",
        "unauthorized",
        "permission denied",
        "insufficient credit",
        "insufficient credits",
        "quota exceeded",
        "payment required",
        "billing",
        "402",
        "401",
        "403",
    ]
    return any(marker in message for marker in markers)


def is_model_decommissioned_error(exc: Exception) -> bool:
    """Detect hard model lifecycle errors that should not be retried."""
    message = str(exc).lower()
    return (
        "model_decommissioned" in message
        or "decommissioned and is no longer supported" in message
    )


def extract_retry_after_seconds(exc: Exception) -> int | None:
    """Extract provider wait hint.

    Handles both formats:
      - 'try again in 13.6s'
      - 'try again in 15m50.4s'
    """
    match = re.search(
        r"try again in\s*(?:([0-9]+)m)?([0-9]+(?:\.[0-9]+)?)s",
        str(exc),
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    try:
        minutes = int(match.group(1) or 0)
        seconds = math.ceil(float(match.group(2)))
        return minutes * 60 + seconds
    except ValueError:
        return None


def get_retry_per_model() -> int:
    return parse_int_env("GROQ_RETRY_PER_MODEL", 1)


def compute_retry_wait_seconds(exc: Exception, attempt: int) -> int:
    """Compute retry delay with provider hint + bounded exponential backoff."""
    base = parse_int_env("GROQ_RETRY_BACKOFF_BASE_SEC", 3)
    cap = parse_int_env("GROQ_RETRY_BACKOFF_MAX_SEC", 45)
    exp_wait = min(cap, base * (2 ** max(0, attempt - 1)))
    hinted   = extract_retry_after_seconds(exc)
    if hinted is not None:
        # Add 5s safety cushion so we don't retry exactly on the boundary
        return min(cap * 20, max(exp_wait, hinted + 5))
    return exp_wait


def _wait_with_countdown(seconds: int, reason: str):
    """Block with a live per-second countdown printed to the terminal."""
    logger.warning(f"[RUN] {reason} — waiting {seconds}s")
    print(f"\n⏳ {reason}")
    for remaining in range(seconds, 0, -1):
        minutes, secs = divmod(remaining, 60)
        if minutes > 0:
            print(f"   Retrying in {minutes}m {secs:02d}s...   ", end="\r", flush=True)
        else:
            print(f"   Retrying in {secs}s...              ", end="\r", flush=True)
        time.sleep(1)
    print(" " * 50, end="\r")
    print("   ✅ Wait complete — retrying now...")


def run_crew_once(
    model_name: str,
    api_key: str,
    serper_tool,
    calculator_tool,
    budget_summary,
    user_input: dict,
    max_rpm: int,
):
    """Execute one full crew run with a specific Groq model/key."""
    logger.info(f"[INIT] Initializing Groq LLM: {model_name}")
    from agents import get_llm
    llm = get_llm(model_name, api_key=api_key)
    logger.info("[INIT] LLM ready ✓")

    logger.info("[INIT] Creating agents...")
    from agents import create_agents
    researcher, budget_planner, itinerary_designer, validator = create_agents(
        serper_tool, calculator_tool, budget_summary, llm
    )
    logger.info("[INIT] Agents ready ✓")

    logger.info("[INIT] Creating tasks...")
    from tasks import create_tasks
    tasks = create_tasks(
        researcher, budget_planner, itinerary_designer, validator, user_input
    )
    logger.info(f"[INIT] {len(tasks)} tasks ready ✓")

    crew = Crew(
        agents=[researcher, budget_planner, itinerary_designer, validator],
        tasks=tasks,
        process=Process.sequential,
        verbose=True,
        memory=False,
        max_rpm=max_rpm,
    )

    start_time = datetime.now()
    try:
        result = crew.kickoff()
        duration = (datetime.now() - start_time).seconds
        logger.info(f"[CREW] Completed in {duration}s")
        usage_summary = llm.get_token_usage_summary().model_dump()
        return result, duration, usage_summary
    except Exception as e:
        try:
            setattr(e, "_llm_usage", llm.get_token_usage_summary().model_dump())
        except Exception:
            pass
        raise


def main():
    logger.info("="*60)
    logger.info("  AI TRAVEL PLANNER CREW - STARTING")
    logger.info("="*60)

    try:
        validate_env()
        user_input = get_user_input()
        logger.info(f"[Input] {json.dumps(user_input, indent=2)}")

        logger.info("[INIT] Initializing tools...")
        from tools import SerperSearchTool, BudgetCalculatorTool, BudgetSummaryTool
        serper_tool     = SerperSearchTool()
        calculator_tool = BudgetCalculatorTool()
        budget_summary  = BudgetSummaryTool()
        logger.info("[INIT] Tools ready ✓")

        max_rpm = parse_int_env("CREWAI_MAX_RPM", 2)
        model_candidates = get_model_candidates()
        api_keys = get_groq_api_keys()
        retry_per_model = get_retry_per_model()
        logger.info(
            f"[INIT] Model candidates: {model_candidates} | retries per model: {retry_per_model} | key pool: {len(api_keys)}"
        )

        print("\n🚀 Starting AI Travel Planner Crew...")
        print("   This may take 3-5 minutes. Watch the agents work!\n")
        print("="*60)

        result     = None
        duration   = 0
        last_error: Exception | None = None
        llm_usage_totals = _empty_llm_usage_totals()

        for model_index, model_name in enumerate(model_candidates, start=1):
            skip_model = False
            for key_index, api_key in enumerate(api_keys, start=1):
                rotate_key = False
                logger.info(
                    f"[RUN] Trying model {model_index}/{len(model_candidates)}: {model_name} "
                    f"| api key {key_index}/{len(api_keys)}"
                )
                for attempt in range(1, retry_per_model + 1):
                    try:
                        result, duration, run_llm_usage = run_crew_once(
                            model_name=model_name,
                            api_key=api_key,
                            serper_tool=serper_tool,
                            calculator_tool=calculator_tool,
                            budget_summary=budget_summary,
                            user_input=user_input,
                            max_rpm=max_rpm,
                        )
                        _merge_llm_usage_totals(llm_usage_totals, run_llm_usage)
                        break

                    except Exception as e:
                        _merge_llm_usage_totals(llm_usage_totals, getattr(e, "_llm_usage", None))
                        last_error     = e
                        retryable      = is_retryable_model_error(e)
                        decommissioned = is_model_decommissioned_error(e)
                        rate_limited   = is_rate_limit_error(e)
                        auth_error     = is_auth_error(e)
                        hint           = extract_retry_after_seconds(e)
                        wait_seconds   = compute_retry_wait_seconds(e, attempt)

                        logger.warning(
                            f"[RUN] {model_name} failed "
                            f"(attempt {attempt}/{retry_per_model}, key {key_index}/{len(api_keys)}): {e}"
                        )

                        if decommissioned:
                            logger.warning(f"[RUN] Model '{model_name}' is decommissioned; skipping.")
                            skip_model = True
                            break

                        if auth_error:
                            logger.warning(
                                f"[RUN] Groq key {key_index}/{len(api_keys)} rejected; rotating key."
                            )
                            rotate_key = True
                            break

                        if rate_limited and key_index < len(api_keys):
                            logger.warning(
                                f"[RUN] Rate limit hit on Groq key {key_index}/{len(api_keys)}; rotating key."
                            )
                            rotate_key = True
                            break

                        if rate_limited:
                            tpd_wait = (hint + 5) if hint else wait_seconds
                            _wait_with_countdown(
                                tpd_wait,
                                f"Rate limit on '{model_name}' — provider says wait {hint}s"
                                if hint else
                                f"Rate limit on '{model_name}' — waiting to retry"
                            )
                            if attempt < retry_per_model:
                                continue
                            break

                        if retryable and attempt < retry_per_model:
                            _wait_with_countdown(
                                wait_seconds,
                                f"Transient error on '{model_name}' — retrying"
                            )
                            continue

                        break

                if result is not None:
                    break
                if skip_model:
                    break
                if rotate_key:
                    continue

            if result is not None:
                break

            logger.warning(f"[RUN] Falling back from model '{model_name}' to next model")

        if result is None:
            raise last_error or RuntimeError(
                "Crew execution failed for all configured models"
            )

        raw_result_text = str(result)
        result_text = compile_report(raw_result_text, user_input)
        usage_summary = _build_usage_summary(llm_usage_totals, serper_tool)
        output_path, fixed_path = save_output(result_text, user_input, usage_summary=usage_summary)

        print("\n" + "="*60)
        print("✅ TRAVEL PLAN GENERATED SUCCESSFULLY!")
        print("="*60)
        print(f"\n📄 Saved to      : {output_path}")
        print(f"📄 Fixed output  : {fixed_path}")
        print(f"📋 Log file      : {log_filename}")
        print(f"⏱️  Duration      : {duration} seconds")
        _print_usage_summary(usage_summary)
        print("\n" + "="*60)
        print(result_text)
        logger.info(f"[USAGE] {json.dumps(usage_summary)}")
        return result_text

    except KeyboardInterrupt:
        logger.warning("[CREW] Interrupted by user")
        print("\n⚠️  Cancelled.")
        sys.exit(0)

    except Exception as e:
        logger.error(f"[ERROR] {str(e)}", exc_info=True)
        print(f"\n❌ Error: {str(e)}")
        print(f"📋 Check log: {log_filename}")
        error_text  = str(e).upper()
        retry_after = extract_retry_after_seconds(e)

        if "TOOL_USE_FAILED" in error_text or "FAILED_GENERATION" in error_text:
            print("💡 Tip: Model tool-calling failed after retries and fallbacks.")
            print("   Set GROQ_MODEL and fallback vars in .env.")
        elif "MODEL_DECOMMISSIONED" in error_text or "DECOMMISSIONED" in error_text:
            print("💡 Tip: Configured Groq model is unavailable.")
            print("   Update GROQ_MODEL/GROQ_MODEL_FALLBACKS in .env.")
        elif "RATE_LIMIT" in error_text or "RATE LIMIT" in error_text:
            if retry_after:
                print(f"💡 Tip: Rate limit hit — wait about {retry_after}s and retry.")
            else:
                print("💡 Tip: Rate limit hit — wait 30-60s and retry.")
            print("   Reduce CREWAI_MAX_RPM and keep prompts concise.")
        elif "NONE OR EMPTY RESPONSE" in error_text:
            print("💡 Tip: Model returned empty output. Retry; if repeated, lower RPM.")
        elif "SERPER_API_KEY" in error_text or "[SERPERTOOL ERROR]" in error_text:
            print("💡 Tip: Check your SERPER_API_KEY in .env")
        elif "GROQ" in error_text or "LITELLM" in error_text:
            print("💡 Tip: Check GROQ_API_KEY / GROQ_API_KEYS in .env")
        sys.exit(1)


if __name__ == "__main__":
    main()
