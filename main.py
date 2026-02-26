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


def get_groq_api_keys() -> list[str]:
    """Collect Groq API keys from env with deduplication.

    Supported env formats:
      - GROQ_API_KEY
      - GROQ_API_KEYS=key1,key2,key3
      - GROQ_API_KEY_1, GROQ_API_KEY_2, ... (optional)
    """
    keys: list[str] = []

    def _add_key(raw_value: str):
        key = (raw_value or "").strip()
        if key and key not in keys:
            keys.append(key)

    _add_key(os.getenv("GROQ_API_KEY", ""))

    raw_pool = os.getenv("GROQ_API_KEYS", "").strip()
    if raw_pool:
        for part in raw_pool.split(","):
            _add_key(part)

    for idx in range(1, 11):
        _add_key(os.getenv(f"GROQ_API_KEY_{idx}", ""))

    return keys


def validate_env():
    missing = []
    if not os.getenv("SERPER_API_KEY"):
        missing.append("SERPER_API_KEY")
    if not get_groq_api_keys():
        missing.append("GROQ_API_KEY (or GROQ_API_KEYS)")
    if missing:
        logger.error(f"Missing environment variables: {missing}")
        print("\n❌ ERROR: Missing API keys!")
        print("Please create a .env file with:")
        for key in missing:
            print(f"  {key}=your_key_here")
        print("\nSee .env.example for reference.")
        sys.exit(1)
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


def save_output(result: str, user_input: dict):
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(exist_ok=True)

    dest_safe = re.sub(r"[^A-Za-z0-9._-]+", "_", user_input["destination"]).strip("._")
    if not dest_safe:
        dest_safe = "destination"
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    md_path    = outputs_dir / f"travel_plan_{dest_safe}_{timestamp}.md"
    json_path  = outputs_dir / f"travel_plan_{dest_safe}_{timestamp}.json"

    # ── Build full Markdown output ─────────────────────────────
    md_content = f"""# AI Travel Plan: {user_input['destination']}

| Field      | Details                                      |
|------------|----------------------------------------------|
| Generated  | {datetime.now().strftime('%B %d, %Y %H:%M')} |
| Destination| {user_input['destination']}                  |
| Dates      | {user_input['travel_dates']}                 |
| Duration   | {user_input['duration_days']} days           |
| Budget     | {user_input['budget']} {user_input['currency']} |
| Preferences| {user_input.get('preferences', 'N/A')}       |

---

{str(result)}

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
    """Return ordered, deduplicated model candidates for fallback."""
    primary = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile").strip()
    fallback_raw = os.getenv(
        "GROQ_MODEL_FALLBACKS",
        "",
    )
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


def compute_retry_wait_seconds(exc: Exception, attempt: int) -> int:
    """Compute retry delay with provider hint + bounded exponential backoff."""
    base     = parse_int_env("GROQ_RETRY_BACKOFF_BASE_SEC", 3)
    cap      = parse_int_env("GROQ_RETRY_BACKOFF_MAX_SEC", 45)
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
    """Execute one full crew run with a specific Groq model."""
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
    result     = crew.kickoff()
    duration   = (datetime.now() - start_time).seconds
    logger.info(f"[CREW] Completed in {duration}s")
    return result, duration


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

        max_rpm          = parse_int_env("CREWAI_MAX_RPM", 3)
        retry_per_model  = parse_int_env("GROQ_RETRY_PER_MODEL", 1)
        model_candidates = get_model_candidates()
        api_keys         = get_groq_api_keys()
        logger.info(
            f"[INIT] Model candidates: {model_candidates} "
            f"| retries per model: {retry_per_model} "
            f"| api key pool: {len(api_keys)}"
        )

        print("\n🚀 Starting AI Travel Planner Crew...")
        print("   This may take 3-5 minutes. Watch the agents work!\n")
        print("="*60)

        result     = None
        duration   = 0
        last_error: Exception | None = None

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
                        result, duration = run_crew_once(
                            model_name=model_name,
                            api_key=api_key,
                            serper_tool=serper_tool,
                            calculator_tool=calculator_tool,
                            budget_summary=budget_summary,
                            user_input=user_input,
                            max_rpm=max_rpm,
                        )
                        break

                    except Exception as e:
                        last_error     = e
                        retryable      = is_retryable_model_error(e)
                        decommissioned = is_model_decommissioned_error(e)
                        rate_limited   = is_rate_limit_error(e)
                        auth_error     = is_auth_error(e)
                        hint           = extract_retry_after_seconds(e)
                        wait_seconds   = compute_retry_wait_seconds(e, attempt)

                        logger.warning(
                            f"[RUN] Model '{model_name}' failed "
                            f"(attempt {attempt}/{retry_per_model}, key {key_index}/{len(api_keys)}): {e}"
                        )

                        # Hard stop — model is gone, no point retrying
                        if decommissioned:
                            logger.warning(
                                f"[RUN] Model '{model_name}' is decommissioned; skipping."
                            )
                            skip_model = True
                            break

                        # Invalid/unauthorized key -> rotate immediately
                        if auth_error:
                            logger.warning(
                                f"[RUN] API key {key_index}/{len(api_keys)} rejected; rotating key."
                            )
                            rotate_key = True
                            break

                        # Rate-limited key -> rotate immediately if a next key exists
                        if rate_limited and key_index < len(api_keys):
                            logger.warning(
                                f"[RUN] Rate limit hit on key {key_index}/{len(api_keys)}; rotating key."
                            )
                            rotate_key = True
                            break

                        if rate_limited:
                            # Last key in pool: cool down, then retry (if attempts remain).
                            tpd_wait = (hint + 5) if hint else wait_seconds
                            _wait_with_countdown(
                                tpd_wait,
                                f"Rate limit on '{model_name}' "
                                f"— provider says wait {hint}s"
                                if hint else
                                f"Rate limit on '{model_name}' — waiting to retry"
                            )
                            if attempt < retry_per_model:
                                continue
                            break

                        # Transient error (tool_use_failed, empty response, etc.)
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
            logger.warning(
                f"[RUN] Falling back from model '{model_name}' to next candidate"
            )

        if result is None:
            raise last_error or RuntimeError(
                "Crew execution failed for all configured models"
            )

        result_text = str(result)
        output_path, fixed_path = save_output(result_text, user_input)

        print("\n" + "="*60)
        print("✅ TRAVEL PLAN GENERATED SUCCESSFULLY!")
        print("="*60)
        print(f"\n📄 Saved to      : {output_path}")
        print(f"📄 Fixed output  : {fixed_path}")
        print(f"📋 Log file      : {log_filename}")
        print(f"⏱️  Duration      : {duration} seconds")
        print("\n" + "="*60)
        print(result_text)
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
            print("   Set GROQ_MODEL / GROQ_MODEL_FALLBACKS in .env.")
        elif "MODEL_DECOMMISSIONED" in error_text or "DECOMMISSIONED" in error_text:
            print("💡 Tip: One configured Groq model is unavailable.")
            print("   Update GROQ_MODEL / GROQ_MODEL_FALLBACKS in .env.")
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
