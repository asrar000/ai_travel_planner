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
        Research **{destination}** for dates **{travel_dates}** and preferences **{preferences}**.

        Use SerperSearch for each query exactly once (keep to 3 total searches):
        - "{destination} top tourist attractions {travel_dates}"
        - "{destination} neighborhoods where to stay"
        - "{destination} travel advisory safety {current_year}"

        Return concise Markdown with:
        1. Destination overview
        2. Top attractions (5-7 only)
        3. Best neighborhoods (stay/eat)
        4. Weather for travel dates
        5. Practical + safety notes
        6. Budget/halal-friendly quick tips

        Keep total output short, with source links and data quality caveats.
        """,
        agent=destination_researcher,
        expected_output=f"""
        Compact Markdown research report for {destination}, max ~320 words.
        Include links and data-quality notes from Serper results.
        """,
        context=[]
    )

    # ── TASK 2: Budget Planning ────────────────────────────────
    task_budget = Task(
        description=f"""
        Create budget plan for **{destination}** ({duration_days} days, {budget} {currency}).
        Dates: {travel_dates}. Preferences: {preferences}.

        Use SerperSearch for only 2 queries:
        - "{destination} hotel accommodation prices {travel_dates}"
        - "{destination} average food, transportation and activity costs tourist"

        Use BudgetCalculator for arithmetic and BudgetSummary for final table.
        Categories: accommodation, food, transport, activities, 10% buffer.
        Include feasibility assessment and assumptions.
        Do not fabricate prices. Keep output concise.
        """,
        agent=budget_planner,
        expected_output=f"""
        Compact Markdown budget plan, max ~260 words, with category totals,
        daily estimate, BudgetSummary table, feasibility status, and links.
        """,
        context=[]
    )

    # ── TASK 3: Itinerary Design ───────────────────────────────
    task_itinerary = Task(
        description=f"""
        Design a practical {duration_days}-day itinerary for **{destination}**.
        Dates: {travel_dates}. Budget: {budget} {currency}. Preferences: {preferences}.

        Use SerperSearch for 1 query:
        - "{destination} attraction opening hours and best areas"

        For each day include 1 morning, 1 afternoon, 1 evening item,
        estimated cost, and one tip.
        Ensure no time conflicts, realistic routing, and budget alignment.
        Day 1 includes arrival/check-in. Final day includes departure.
        """,
        agent=itinerary_designer,
        expected_output=f"""
        Compact itinerary for exactly {duration_days} days, max ~320 words total,
        with daily schedule, costs, transport notes, and total estimate.
        """,
        context=[]
    )

    # ── TASK 4: Validation ─────────────────────────────────────
    task_validation = Task(
        description=f"""
        Validate and finalize the full plan for **{destination}**.
        Budget: {budget} {currency}. Duration: {duration_days} days.

        Verify:
        1. Budget consistency (calculator-checked)
        2. Itinerary feasibility and timing
        3. Data consistency across research/budget/itinerary
        4. Risks (weather, budget, logistics, data confidence)

        Compile final Markdown:
        # Travel Plan: {destination}
        ## Executive Summary
        ## Destination Overview
        ## Budget Breakdown
        ## Day-wise Itinerary
        ## Validation Summary
        ## Risk Factors
        ## Recommendations
        ## Assumptions Made

        Keep final response short (about 450 words max).
        """,
        agent=validation_agent,
        expected_output=f"""
        Final concise Markdown travel plan with budget table, full itinerary,
        validation status, top risks, recommendations, and assumptions.
        """,
        context=[task_research, task_budget, task_itinerary]
    )

    logger.info("[Tasks] All 4 tasks created successfully")
    return [task_research, task_budget, task_itinerary, task_validation]
