# 🌍 AI Travel Planner Crew

Multi-agent AI travel planning system using **CrewAI** + **Serper Dev API** + **Groq API**.

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

## Get API Keys

| API | URL | Free Tier |
|-----|-----|-----------|
| Serper Dev | https://serper.dev | 2,500 searches/month |
| Groq | https://console.groq.com/keys | Provider-managed limits |

Input flow uses 5 prompts only: destination, travel dates, budget, currency, preferences.
Trip duration is auto-derived from dates (default 5 days if parsing fails).

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
| `GROQ_API_KEY not found` | Set `GROQ_API_KEY` or `GROQ_API_KEYS` in `.env` |
| `tool_use_failed` / `failed_generation` | App retries then falls back; tune `GROQ_MODEL` and `GROQ_MODEL_FALLBACKS` |
| `model_decommissioned` | Remove deprecated model from `GROQ_MODEL_FALLBACKS` in `.env` |
| `Rate limit exceeded` | App rotates to the next key from `GROQ_API_KEYS`; if all keys are limited, it backs off |
| `Invalid response from LLM call - None or empty` | Retry once; if repeated, lower `CREWAI_MAX_RPM` and keep prompts/tool output compact |
| `ModuleNotFoundError` | Run `source venv/bin/activate` first |

Recommended rate-limit settings in `.env`:
- `GROQ_MODEL=llama-3.3-70b-versatile`
- `GROQ_MODEL_FALLBACKS=llama-3.1-8b-instant`
- `GROQ_API_KEYS=gsk_key_1,gsk_key_2,gsk_key_3`
- `GROQ_RETRY_PER_MODEL=1`
- `CREWAI_MAX_RPM=2`
- `SERPER_RESULTS_LIMIT=2`
- `SERPER_SNIPPET_MAX_CHARS=120`
- `SERPER_INCLUDE_SNIPPET=false`
- `GROQ_MAX_TOKENS=280`
- `GROQ_TEMPERATURE=0.0`
