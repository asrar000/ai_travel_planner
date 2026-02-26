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


def validate_env():
    missing = []
    if not os.getenv("SERPER_API_KEY"):
        missing.append("SERPER_API_KEY")
    if not os.getenv("GROQ_API_KEY"):
        missing.append("GROQ_API_KEY")
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

    Returns 0 if the date string cannot be confidently parsed, which
    triggers a manual duration prompt in get_user_input().
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

    # Could not parse confidently — caller will prompt manually
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

    # Auto-calculate duration from travel dates (fallback to manual if needed)
    duration_days = calculate_duration(travel_dates)
    if duration_days <= 0:
        try:
            duration_days = int(input("⏱️  Duration in days (e.g., 7): ").strip())
        except ValueError:
            duration_days = 7
            print("   Could not parse dates; using default duration: 7 days")
    if duration_days <= 0:
        duration_days = 7
        print("   Duration must be positive; using default: 7 days")

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

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# AI Travel Plan: {user_input['destination']}\n")
        f.write(f"**Generated:** {datetime.now().strftime('%B %d, %Y %H:%M')}\n")
        f.write(f"**Budget:** {user_input['budget']} {user_input['currency']}\n")
        f.write(f"**Duration:** {user_input['duration_days']} days\n")
        f.write(f"**Dates:** {user_input['travel_dates']}\n\n---\n\n")
        f.write(str(result))

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "generated_at": datetime.now().isoformat(),
            "user_input": user_input,
            "output_file": str(md_path),
            "log_file": str(log_filename)
        }, f, indent=2)

    logger.info(f"[Output] Saved to: {md_path}")
    return md_path


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
        "llama-3.1-8b-instant",
    )
    fallback_models = [m.strip() for m in fallback_raw.split(",") if m.strip()]

    models: list[str] = []
    for model in [primary] + fallback_models:
        if model and model not in models:
            models.append(model)
    return models


def is_retryable_model_error(exc: Exception) -> bool:
    """Identify transient/provider errors worth retrying."""
    message = str(exc).lower()
    if is_model_decommissioned_error(exc):
        return False

    retry_markers = [
        "tool_use_failed",
        "failed_generation",
        "failed to call a function",
        "rate_limit_exceeded",
        "429",
        "rate limit",
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


def is_model_decommissioned_error(exc: Exception) -> bool:
    """Detect hard model lifecycle errors that should not be retried."""
    message = str(exc).lower()
    return (
        "model_decommissioned" in message
        or "decommissioned and is no longer supported" in message
    )


def extract_retry_after_seconds(exc: Exception) -> int | None:
    """Extract provider wait hint from messages like: 'try again in 13.6s'."""
    match = re.search(r"try again in\s*([0-9]+(?:\.[0-9]+)?)s", str(exc), flags=re.IGNORECASE)
    if not match:
        return None
    try:
        return max(1, math.ceil(float(match.group(1))))
    except ValueError:
        return None


def compute_retry_wait_seconds(exc: Exception, attempt: int) -> int:
    """Compute retry delay with provider hint + bounded exponential backoff."""
    base = parse_int_env("GROQ_RETRY_BACKOFF_BASE_SEC", 3)
    cap = parse_int_env("GROQ_RETRY_BACKOFF_MAX_SEC", 45)
    exp_wait = min(cap, base * (2 ** max(0, attempt - 1)))
    hinted = extract_retry_after_seconds(exc)
    if hinted is not None:
        # Add 1s safety cushion so we don't retry exactly on the boundary.
        return min(cap, max(exp_wait, hinted + 1))
    return exp_wait


def run_crew_once(
    model_name: str,
    serper_tool,
    calculator_tool,
    budget_summary,
    user_input: dict,
    max_rpm: int,
):
    """Execute one full crew run with a specific Groq model."""
    logger.info(f"[INIT] Initializing Groq LLM: {model_name}")
    from agents import get_llm

    llm = get_llm(model_name)
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
    result = crew.kickoff()
    duration = (datetime.now() - start_time).seconds
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

        max_rpm = parse_int_env("CREWAI_MAX_RPM", 10)
        retry_per_model = parse_int_env("GROQ_RETRY_PER_MODEL", 2)
        model_candidates = get_model_candidates()
        logger.info(
            f"[INIT] Model candidates: {model_candidates} | retries per model: {retry_per_model}"
        )

        print("\n🚀 Starting AI Travel Planner Crew...")
        print("   This may take 3-5 minutes. Watch the agents work!\n")
        print("="*60)

        result = None
        duration = 0
        last_error: Exception | None = None

        for model_index, model_name in enumerate(model_candidates, start=1):
            logger.info(f"[RUN] Trying model {model_index}/{len(model_candidates)}: {model_name}")
            for attempt in range(1, retry_per_model + 1):
                try:
                    result, duration = run_crew_once(
                        model_name=model_name,
                        serper_tool=serper_tool,
                        calculator_tool=calculator_tool,
                        budget_summary=budget_summary,
                        user_input=user_input,
                        max_rpm=max_rpm,
                    )
                    break
                except Exception as e:
                    last_error = e
                    retryable = is_retryable_model_error(e)
                    decommissioned = is_model_decommissioned_error(e)
                    logger.warning(
                        f"[RUN] Model '{model_name}' failed (attempt {attempt}/{retry_per_model}): {e}"
                    )
                    if decommissioned:
                        logger.warning(
                            f"[RUN] Model '{model_name}' is decommissioned; skipping further retries for this model."
                        )
                        break
                    if retryable and attempt < retry_per_model:
                        wait_seconds = compute_retry_wait_seconds(e, attempt)
                        logger.warning(
                            f"[RUN] Retrying model '{model_name}' in {wait_seconds}s due to transient provider error"
                        )
                        time.sleep(wait_seconds)
                        continue
                    break
            if result is not None:
                break
            logger.warning(f"[RUN] Falling back from model '{model_name}' to next candidate")

        if result is None:
            raise last_error or RuntimeError("Crew execution failed for all configured models")

        result_text = str(result)
        output_path = save_output(result_text, user_input)

        print("\n" + "="*60)
        print("✅ TRAVEL PLAN GENERATED SUCCESSFULLY!")
        print("="*60)
        print(f"\n📄 Saved to   : {output_path}")
        print(f"📋 Log file   : {log_filename}")
        print(f"⏱️  Duration   : {duration} seconds")
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
        error_text = str(e).upper()
        retry_after = extract_retry_after_seconds(e)
        if "TOOL_USE_FAILED" in error_text or "FAILED_GENERATION" in error_text:
            print("💡 Tip: Groq tool-calling failed after retries and fallbacks.")
            print("   Set GROQ_MODEL / GROQ_MODEL_FALLBACKS in .env to alternative models.")
        elif "MODEL_DECOMMISSIONED" in error_text or "DECOMMISSIONED" in error_text:
            print("💡 Tip: One configured Groq model is decommissioned.")
            print("   Update GROQ_MODEL / GROQ_MODEL_FALLBACKS in .env to currently supported models.")
        elif "RATE_LIMIT" in error_text or "RATE LIMIT" in error_text:
            if retry_after:
                print(f"💡 Tip: Rate limit hit — wait about {retry_after}s and retry.")
            else:
                print("💡 Tip: Rate limit hit — wait 30-60s and retry.")
        elif "SERPER_API_KEY" in error_text or "[SERPERTOOL ERROR]" in error_text:
            print("💡 Tip: Check your SERPER_API_KEY in .env")
        elif "GROQ" in error_text or "LITELLM" in error_text:
            print("💡 Tip: Check your GROQ_API_KEY in .env")
        sys.exit(1)


if __name__ == "__main__":
    main()
