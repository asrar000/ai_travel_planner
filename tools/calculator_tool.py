"""
tools/calculator_tool.py
Custom calculator tool for budget reasoning and cost estimation.
"""

import re
import logging
from crewai.tools import BaseTool

logger = logging.getLogger(__name__)


class BudgetCalculatorTool(BaseTool):
    name: str = "BudgetCalculator"
    description: str = (
        "A calculator for computing travel budget breakdowns. "
        "Provide a mathematical expression or budget calculation request. "
        "Examples: '3 * 80', '200 + 150 + 300 + 100', '50 * 7'. "
        "Can compute per-day averages and budget totals."
    )

    def _run(self, expression: str) -> str:
        logger.info(f"[Calculator] Computing: {expression}")
        try:
            math_expr = re.sub(r'[$€£¥₹]', '', expression.strip())
            math_match = re.search(r'[\d\s\+\-\*\/\.\(\)]+', math_expr)
            if math_match:
                pure_math = math_match.group().strip()
                if pure_math and any(c.isdigit() for c in pure_math):
                    result = eval(pure_math, {"__builtins__": {}}, {})
                    logger.info(f"[Calculator] Result: {result}")
                    return (
                        f"Calculation: {pure_math}\n"
                        f"Result: {result:.2f}\n"
                        f"(Original input: {expression})"
                    )
            return f"Could not extract a numeric expression from: '{expression}'"
        except ZeroDivisionError:
            return "Error: Division by zero."
        except Exception as e:
            return f"Calculation Error: {str(e)}"


class BudgetSummaryTool(BaseTool):
    name: str = "BudgetSummary"
    description: str = (
        "Generate a complete budget summary from individual cost components. "
        "Provide costs as comma-separated key:value pairs. "
        "Example: 'accommodation:600, food:350, transport:200, activities:150'"
    )

    def _run(self, costs_input: str) -> str:
        logger.info(f"[BudgetSummary] Processing: {costs_input}")
        try:
            import re
            components = {}
            pairs = re.findall(r'(\w[\w\s]*?)\s*:\s*([\d\.]+)', costs_input)
            for key, value in pairs:
                components[key.strip()] = float(value)

            if not components:
                return f"Could not parse budget components from: '{costs_input}'"

            total = sum(components.values())
            lines = ["=" * 40, "BUDGET SUMMARY", "=" * 40]
            for category, amount in components.items():
                pct = (amount / total * 100) if total > 0 else 0
                lines.append(f"  {category.title():<20} ${amount:>8.2f}  ({pct:.1f}%)")
            lines.append("-" * 40)
            lines.append(f"  {'TOTAL':<20} ${total:>8.2f}  (100%)")
            lines.append("=" * 40)
            return "\n".join(lines)
        except Exception as e:
            return f"Budget summary error: {str(e)}"
