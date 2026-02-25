# 🌍 AI Travel Planner Crew

Multi-agent AI travel planning system using **CrewAI** + **Serper Dev API** + **Google Gemini Free API**.

## Architecture

```
User Input
    │
    ▼
┌──────────────────────────────────────┐
│         CREW MANAGER                 │
│     (CrewAI Sequential Process)      │
└──────────────────────────────────────┘
    │
    ▼ Task 1
┌──────────────────────┐
│ Destination          │  Tools: SerperSearch
│ Researcher Agent     │  → Attractions, culture, safety, food
└──────────────────────┘
    │
    ▼ Task 2
┌──────────────────────┐
│ Budget Planner       │  Tools: SerperSearch + Calculator + BudgetSummary
│ Agent                │  → Real prices, cost breakdown, feasibility
└──────────────────────┘
    │
    ▼ Task 3
┌──────────────────────┐
│ Itinerary Designer   │  Tools: SerperSearch
│ Agent                │  → Day-wise plan, times, routes
└──────────────────────┘
    │
    ▼ Task 4 (receives context from Tasks 1,2,3)
┌──────────────────────┐
│ Validation Agent     │  Tools: Calculator + BudgetSummary
│                      │  → Cross-validate, risk assessment
└──────────────────────┘
    │
    ▼
Final Structured Output (Markdown + JSON saved to outputs/)
```

## Project Structure

```
ai_travel_planner/
├── main.py
├── requirements.txt
├── .env.example
├── example_input_output.json
├── README.md
├── agents/
│   ├── __init__.py
│   └── travel_agents.py
├── tasks/
│   ├── __init__.py
│   └── travel_tasks.py
├── tools/
│   ├── __init__.py
│   ├── serper_tool.py
│   └── calculator_tool.py
├── outputs/
└── logs/
```

## Quick Start (Ubuntu, Non-Root)

```bash
# 1. Create & activate virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your API keys
cp .env.example .env
nano .env

# 4. Run!
python3 main.py
```

## Get Free API Keys

| API | URL | Free Tier |
|-----|-----|-----------|
| Serper Dev | https://serper.dev | 2,500 searches/month |
| Google Gemini | https://aistudio.google.com/app/apikey | Free |

## Agents Summary

| Agent | Tools |
|-------|-------|
| Destination Researcher | SerperSearch |
| Budget Planner | SerperSearch, BudgetCalculator, BudgetSummary |
| Itinerary Designer | SerperSearch |
| Validation Agent | BudgetCalculator, BudgetSummary |

## Rules Compliance

- No hardcoded travel data
- No fake cost estimates
- Serper mandatory and enforced
- 4 agents implemented
- Structured Markdown + JSON output
- Full execution logging to logs/
- Failure handling for all API errors

## Troubleshooting

| Error | Fix |
|-------|-----|
| `SERPER_API_KEY not found` | Check `.env` file |
| `GEMINI_API_KEY not found` | Check `.env` file |
| `Rate limit exceeded` | Wait 60s, reduce `max_rpm` in main.py |
| `ModuleNotFoundError` | Run `source venv/bin/activate` first |
