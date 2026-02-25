"""
main.py
AI Travel Planner Crew - Main Entry Point
Multi-agent system: CrewAI + Gemini Free API + Serper Dev API
"""

import os
import sys
import json
import logging
from datetime import datetime
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
    if not os.getenv("GEMINI_API_KEY"):
        missing.append("GEMINI_API_KEY")
    if missing:
        logger.error(f"Missing environment variables: {missing}")
        print("\n❌ ERROR: Missing API keys!")
        print("Please create a .env file with:")
        for key in missing:
            print(f"  {key}=your_key_here")
        print("\nSee .env.example for reference.")
        sys.exit(1)
    logger.info("[ENV] All required API keys found ✓")


def get_user_input() -> dict:
    print("\n" + "="*60)
    print("  🌍  AI TRAVEL PLANNER CREW  🌍")
    print("  Powered by CrewAI + Gemini + Serper")
    print("="*60 + "\n")

    destination = input("📍 Destination (e.g., Tokyo, Japan): ").strip() or "Tokyo, Japan"
    travel_dates = input("📅 Travel Dates (e.g., March 15-22, 2025): ").strip() or "March 15-22, 2025"

    try:
        duration_days = int(input("⏱️  Duration in days (e.g., 7): ").strip())
    except ValueError:
        duration_days = 7
        print("   Using default: 7 days")

    try:
        budget = float(input("💰 Total budget (numeric, e.g., 2000): ").strip())
    except ValueError:
        budget = 2000
        print("   Using default: $2000")

    currency = input("💱 Currency (e.g., USD, EUR, GBP) [default: USD]: ").strip().upper() or "USD"
    preferences = input("🎯 Preferences (e.g., history, food, adventure): ").strip() or "culture, food, sightseeing"

    user_input = {
        "destination": destination,
        "travel_dates": travel_dates,
        "duration_days": duration_days,
        "budget": budget,
        "currency": currency,
        "preferences": preferences
    }

    print("\n📋 Trip Summary:")
    for k, v in user_input.items():
        print(f"   {k:<15} : {v}")
    print()
    return user_input


def save_output(result: str, user_input: dict):
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(exist_ok=True)

    dest_safe  = user_input["destination"].replace(" ", "_").replace(",", "")
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

    validate_env()
    user_input = get_user_input()
    logger.info(f"[Input] {json.dumps(user_input, indent=2)}")

    try:
        logger.info("[INIT] Initializing tools...")
        from tools import SerperSearchTool, BudgetCalculatorTool, BudgetSummaryTool
        serper_tool       = SerperSearchTool()
        calculator_tool   = BudgetCalculatorTool()
        budget_summary    = BudgetSummaryTool()
        logger.info("[INIT] Tools ready ✓")

        logger.info("[INIT] Initializing Gemini LLM...")
        from agents import get_llm
        llm = get_llm(os.getenv("GEMINI_MODEL", "gemini-1.5-flash"))
        logger.info("[INIT] LLM ready ✓")

        logger.info("[INIT] Creating agents...")
        from agents import create_agents
        researcher, budget_planner, itinerary_designer, validator = \
            create_agents(serper_tool, calculator_tool, budget_summary, llm)
        logger.info("[INIT] Agents ready ✓")

        logger.info("[INIT] Creating tasks...")
        from tasks import create_tasks
        tasks = create_tasks(researcher, budget_planner, itinerary_designer, validator, user_input)
        logger.info(f"[INIT] {len(tasks)} tasks ready ✓")

        crew = Crew(
            agents=[researcher, budget_planner, itinerary_designer, validator],
            tasks=tasks,
            process=Process.sequential,
            verbose=True,
            memory=False,
            max_rpm=10
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
        elif "GEMINI" in str(e).upper() or "GOOGLE" in str(e).upper():
            print("💡 Tip: Check your GEMINI_API_KEY in .env")
        elif "rate" in str(e).lower():
            print("💡 Tip: Rate limit hit — wait 60s and retry.")
        sys.exit(1)


if __name__ == "__main__":
    main()
