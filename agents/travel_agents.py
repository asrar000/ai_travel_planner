"""
agents/travel_agents.py
Defines all 4 required agents for the AI Travel Planner.
"""

import os
import logging
from crewai import Agent, LLM


logger = logging.getLogger(__name__)


class GroqToolStableLLM(LLM):
    """CrewAI LLM wrapper that disables native function-calling for Groq.

    CrewAI native function-calling with some Groq models can intermittently
    produce `tool_use_failed`. Returning False here forces CrewAI's text-based
    tool loop, which is significantly more stable for this project.
    """

    def supports_function_calling(self) -> bool:
        return False


def get_llm(model_name: str = "llama-3.3-70b-versatile"):
    """Initialize Groq LLM with stable (non-native) tool usage mode."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables!")

    os.environ["GROQ_API_KEY"] = api_key

    llm = GroqToolStableLLM(
        model=f"groq/{model_name}",
        temperature=0.2,
    )
    logger.info(f"[LLM] Initialized Groq model (stable tools mode): {model_name}")
    return llm


def create_agents(serper_tool, calculator_tool, budget_summary_tool, llm):
    """Factory function to create all 4 required agents."""

    destination_researcher = Agent(
        role="Destination Research Specialist",
        goal=(
            "Research the travel destination thoroughly using Serper web search. "
            "Find top attractions, local culture, best time to visit, must-see places, "
            "neighborhoods, safety tips, and current travel advisories."
        ),
        backstory=(
            "You are an expert travel researcher with 15+ years of experience exploring "
            "destinations worldwide. You ALWAYS use Serper web search to get the most "
            "current and accurate destination information — never relying on memory alone. "
            "You validate all information and flag inconsistencies."
        ),
        tools=[serper_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=5
    )

    budget_planner = Agent(
        role="Travel Budget Planning Expert",
        goal=(
            "Create a detailed, realistic budget breakdown covering accommodation, food, "
            "transport, and activities. Use Serper to research actual current prices. "
            "Use the calculator for precise cost computations. Never fabricate prices."
        ),
        backstory=(
            "You are a certified financial travel consultant. You research real market prices "
            "using web search before estimating anything. You use the calculator for all "
            "arithmetic and clearly state assumptions. You flag unrealistic budgets."
        ),
        tools=[serper_tool, calculator_tool, budget_summary_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=6
    )

    itinerary_designer = Agent(
        role="Professional Itinerary Design Specialist",
        goal=(
            "Design a detailed, realistic day-by-day itinerary that fits within the budget, "
            "respects travel distances, opening hours, and traveler preferences. "
            "Each day must have a clear schedule with no time conflicts."
        ),
        backstory=(
            "You are a master itinerary planner who groups nearby attractions to minimize "
            "travel time. You use Serper to verify opening hours and distances. "
            "You create practical, enjoyable plans that don't over-pack each day."
        ),
        tools=[serper_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=5
    )

    validation_agent = Agent(
        role="Travel Plan Quality Assurance Validator",
        goal=(
            "Critically validate the entire travel plan for consistency, feasibility, "
            "and budget alignment. Identify conflicts, unrealistic expectations, "
            "and missing information. Produce a clear validation summary."
        ),
        backstory=(
            "You are a meticulous travel plan auditor. You cross-reference destination "
            "research, budget breakdown, and itinerary to ensure everything aligns. "
            "You use the calculator to verify budget totals and provide risk assessments."
        ),
        tools=[calculator_tool, budget_summary_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=4
    )

    logger.info("[Agents] All 4 agents created successfully")
    return destination_researcher, budget_planner, itinerary_designer, validation_agent
