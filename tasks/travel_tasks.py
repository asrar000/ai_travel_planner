"""
tasks/travel_tasks.py
Defines all tasks assigned to the travel planning agents.
"""

import logging
from datetime import datetime
from crewai import Task

logger = logging.getLogger(__name__)


def create_tasks(
    destination_researcher,
    budget_planner,
    itinerary_designer,
    validation_agent,
    user_input: dict
) -> list:
    """Create all tasks with proper context and delegation flow."""

    destination   = user_input["destination"]
    travel_dates  = user_input["travel_dates"]
    budget        = user_input["budget"]
    currency      = user_input.get("currency", "USD")
    preferences   = user_input.get("preferences", "No specific preferences")
    duration_days = user_input.get("duration_days", 5)
    current_year = datetime.now().year

    logger.info(f"[Tasks] Creating tasks for: {destination} | {travel_dates} | {budget} {currency}")

    # ── TASK 1: Destination Research ──────────────────────────
    task_research = Task(
        description=f"""
        Research the travel destination: **{destination}**
        Travel Dates: {travel_dates}
        Traveler Preferences: {preferences}

        Using the SerperSearch tool, research and compile:
        1. Destination Overview: description, location, why visit
        2. Top Attractions (8-10): names, descriptions, visit duration
        3. Local Culture & Customs: etiquette, dress code
        4. Best Neighborhoods: where to stay, eat, explore
        5. Weather & Climate during travel dates
        6. Practical Info: visa, currency, language, emergency contacts
        7. Safety: current travel advisories, safe/unsafe areas
        8. Hidden Gems: lesser-known worthwhile spots
        9. Food Scene: must-try dishes, restaurant areas

        MANDATORY: Use SerperSearch for EACH query:
        - "{destination} top tourist attractions {travel_dates}"
        - "{destination} travel tips culture customs"
        - "{destination} neighborhoods where to stay"
        - "{destination} travel advisory safety {current_year}"
        - "{destination} local food must try restaurants"

        Format output in clear Markdown with headers.
        Note any data quality issues found.
        """,
        agent=destination_researcher,
        expected_output=f"""
        A comprehensive Markdown research report for {destination} containing:
        - Destination overview (2-3 paragraphs)
        - Numbered list of 8-10 top attractions with descriptions
        - Cultural notes and practical travel information
        - Neighborhood guide
        - Weather information for {travel_dates}
        - Safety and advisory notes
        - Food recommendations
        - At least 3 hidden gems
        All information sourced from actual Serper web searches.
        """
    )

    # ── TASK 2: Budget Planning ────────────────────────────────
    task_budget = Task(
        description=f"""
        Create a detailed budget breakdown for the trip to **{destination}**.
        Total Budget: {budget} {currency}
        Trip Duration: {duration_days} days
        Travel Dates: {travel_dates}
        Preferences: {preferences}

        Steps:
        1. Use SerperSearch to find REAL current prices:
           - "{destination} hotel accommodation prices {travel_dates}"
           - "{destination} average food cost per day tourist"
           - "{destination} transportation costs local transport"
           - "{destination} tourist activities entry fees prices"
           - "flight to {destination} approximate cost"

        2. Use BudgetCalculator tool to compute:
           - Accommodation = nightly rate × {duration_days} nights
           - Food = daily budget × {duration_days} days
           - Transport = local transport total
           - Activities = sum of entry fees + tours

        3. Use BudgetSummary tool to generate the final summary table.
           Include currency explicitly, e.g. "currency:{currency}, accommodation:..., food:..."

        4. Assess feasibility:
           - Is {budget} {currency} realistic for {duration_days} days?
           - Budget tier? (Budget / Mid-range / Luxury)
           - Adjustments if budget is tight?

        Categories: Accommodation, Food, Local Transport,
        Activities & Entry Fees, Miscellaneous Buffer (10%).

        IMPORTANT: Do NOT fabricate prices. Use Serper results only.
        State all assumptions clearly.
        """,
        agent=budget_planner,
        expected_output=f"""
        A detailed budget plan in Markdown containing:
        - Budget feasibility assessment for {budget} {currency}
        - Itemized daily costs per category
        - BudgetSummary tool output (table)
        - Price sources from Serper searches
        - Budget tier classification
        - Money-saving tips for {destination}
        - Risk flags if budget is insufficient
        All calculations verified using BudgetCalculator.
        """
    )

    # ── TASK 3: Itinerary Design ───────────────────────────────
    task_itinerary = Task(
        description=f"""
        Design a complete day-by-day itinerary for **{destination}**.
        Duration: {duration_days} days
        Travel Dates: {travel_dates}
        Budget: {budget} {currency} total
        Preferences: {preferences}

        Use SerperSearch to verify:
        - "{destination} attraction opening hours"
        - "{destination} best day trips from city"
        - "{destination} walking tour self guided map"

        For EACH of the {duration_days} days provide:
        - Day theme/focus (e.g., "Day 1: Historical Downtown")
        - Morning (8AM-12PM): 2-3 activities with times
        - Afternoon (12PM-6PM): 2-3 activities with times
        - Evening (6PM-10PM): Dinner + evening activity
        - Estimated cost for the day
        - Travel tips for that day

        Rules:
        - No time conflicts between activities
        - Geographically logical routing
        - Respect opening hours
        - Match budget tier
        - Day 1 accounts for arrival/check-in
        - Last day accounts for check-out/departure

        Create exactly {duration_days} days of itinerary.
        """,
        agent=itinerary_designer,
        expected_output=f"""
        A complete {duration_days}-day itinerary in Markdown with:
        - Day-by-day schedule (Morning / Afternoon / Evening)
        - Each activity with duration and cost
        - Restaurant recommendations for each meal
        - Daily cost estimate aligned with budget plan
        - Transportation instructions between locations
        - Practical tips per day
        - Total itinerary cost summary
        Itinerary must be realistic, geographically logical, conflict-free.
        """
    )

    # ── TASK 4: Validation ─────────────────────────────────────
    task_validation = Task(
        description=f"""
        Validate and finalize the complete travel plan for **{destination}**.
        Budget: {budget} {currency} | Duration: {duration_days} days

        1. Budget Validation:
           - Use BudgetCalculator: itinerary daily costs × {duration_days} = total?
           - Does total ≤ {budget} {currency}?
           - Flag any budget overruns with specific line items.

        2. Itinerary Validation:
           - Check for time conflicts
           - Verify geographic feasibility
           - Confirm opening hours respected
           - Flag over/under-packed days

        3. Data Consistency:
           - Destination info matches itinerary locations
           - Accommodation area matches itinerary start points
           - Budget tier matches selected activities

        4. Risk Assessment:
           - Weather risks during {travel_dates}
           - Budget risks (categories at risk of overspend)
           - Logistical risks
           - Information reliability issues

        5. Final Output Assembly — compile COMPLETE plan as:
           # Travel Plan: {destination}
           ## Executive Summary
           ## Destination Overview
           ## Budget Breakdown (use BudgetSummary tool)
           ## Day-wise Itinerary (all {duration_days} days)
           ## Validation Summary
           ## Risk Factors
           ## Recommendations
           ## Assumptions Made
        """,
        agent=validation_agent,
        expected_output=f"""
        A complete validated travel plan in structured Markdown:

        # Travel Plan: {destination}
        ## Executive Summary
        ## Destination Overview
        ## Budget Breakdown (table)
        ## Day-wise Itinerary ({duration_days} days)
        ## Validation Summary
           - Budget Status: PASS/FAIL with details
           - Itinerary Status: PASS/FAIL with details
           - Data Quality assessment
        ## Risk Factors (with severity levels)
        ## Recommendations & Tips
        ## Assumptions Made

        Self-contained and ready to share with the traveler.
        """,
        context=[task_research, task_budget, task_itinerary]
    )

    logger.info("[Tasks] All 4 tasks created successfully")
    return [task_research, task_budget, task_itinerary, task_validation]
