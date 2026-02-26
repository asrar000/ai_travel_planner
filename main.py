"""
main.py
AI Travel Planner Crew - Main Entry Point
Multi-agent system: CrewAI + Groq + Serper Dev API
"""

import os
import re
import sys
import json
import logging
from datetime import date, datetime
from pathlib import Path
from dotenv import load_dotenv
from crewai import Crew, Process

load_dotenv()

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

        logger.info("[INIT] Initializing Groq LLM...")
        from agents import get_llm
        llm = get_llm(os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"))
        logger.info("[INIT] LLM ready ✓")

        logger.info("[INIT] Creating agents...")
        from agents import create_agents
        researcher, budget_planner, itinerary_designer, validator = \
            create_agents(serper_tool, calculator_tool, budget_summary, llm)
        logger.info("[INIT] Agents ready ✓")

        logger.info("[INIT] Creating tasks...")
        from tasks import create_tasks
        tasks = create_tasks(
            researcher, budget_planner,
            itinerary_designer, validator,
            user_input
        )
        logger.info(f"[INIT] {len(tasks)} tasks ready ✓")

        try:
            max_rpm = int(os.getenv("CREWAI_MAX_RPM", "10"))
        except ValueError:
            max_rpm = 10
            logger.warning("[INIT] Invalid CREWAI_MAX_RPM, falling back to 10")

        crew = Crew(
            agents=[researcher, budget_planner, itinerary_designer, validator],
            tasks=tasks,
            process=Process.sequential,
            verbose=True,
            memory=False,
            max_rpm=max_rpm
        )

        print("\n🚀 Starting AI Travel Planner Crew...")
        print("   This may take 3-5 minutes. Watch the agents work!\n")
        print("="*60)

        start_time = datetime.now()
        result     = crew.kickoff()
        duration   = (datetime.now() - start_time).seconds

        logger.info(f"[CREW] Completed in {duration}s")

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
        if "SERPER" in str(e).upper():
            print("💡 Tip: Check your SERPER_API_KEY in .env")
        elif "GROQ" in str(e).upper():
            print("💡 Tip: Check your GROQ_API_KEY in .env")
        elif "rate" in str(e).lower():
            print("💡 Tip: Rate limit hit — wait 60s and retry.")
        sys.exit(1)


if __name__ == "__main__":
    main()
