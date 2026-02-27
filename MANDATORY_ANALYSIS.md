# Mandatory Analysis Section

## 1)Why Multi-Agent Architecture?
- Multi-agent architecture provides separation of responsibility
- Specialized Agents are better than a general agent
- Increases maintainability by providing an easy-to-separate point of concern
- Clear failure isolation

## 2)What if Serper returns incorrect data?
- Treating the search as probabilistics not a guaranteed
- Cross-checking critical facts using multiple sources
- Preference is given to Official/Government sources for high impact Facts
- When uncertainty occurs, clearly state it in the output
- For low confidence search results, defaults to conservative assumption

## 3)What if the budget is unrealistic?
- To sanitize the budget, do a minimum viable trip cost
- When underfunded, clearly flag it
- Realistic adjustment when possible, such as reduce number of trip days, Cheaper stays, avoiding luxury foods, etc
- Prioritize the essential cost first

## 4)Hallucination Risks
- Keep the output always grounded based on the search facts
- Flaging assumptions when data is missing
- Using Deterministic post-processing, which ensures completeness without extra LLM calls
- Any output from LLM always maintains a certain degree of Halluncination so the travel planning should always be treated as guidance

## 5)Token Usage Considerations
- Keep task prompts concise and structured
- Limiting search results from Serper
- Using low temperature and tightening the max token limits
- Avoiding extra LLM calls by using deterministic compilation
- Keeping a count of Retries and RPM  to minimize LLM calls

## 6) Scalability
- Caching repeated search queries
- Parallelizing an independent AI agent task
- Adding Queue/Job Scheduler for concurrent users
- Maintaining metrics such as external api calls, quotas, rate limits, and network variability, etc
- Using multiple LLM models and a merging agent that merges the tone properly
