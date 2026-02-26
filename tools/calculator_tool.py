"""
tools/calculator_tool.py
Custom calculator tool for budget reasoning and cost estimation.
"""

import ast
import logging
import operator
import re
from crewai.tools import BaseTool

logger = logging.getLogger(__name__)

ALLOWED_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}
ALLOWED_UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def safe_eval_math(expression: str) -> float:
    """Safely evaluate a basic arithmetic expression."""
    tree = ast.parse(expression, mode="eval")

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Num):  # Python <3.8 compatibility
            return float(node.n)
        if isinstance(node, ast.BinOp) and type(node.op) in ALLOWED_BINARY_OPERATORS:
            left = _eval(node.left)
            right = _eval(node.right)
            if isinstance(node.op, ast.Div) and right == 0:
                raise ZeroDivisionError
            return ALLOWED_BINARY_OPERATORS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in ALLOWED_UNARY_OPERATORS:
            return ALLOWED_UNARY_OPERATORS[type(node.op)](_eval(node.operand))
        raise ValueError("Unsupported expression. Use numbers and + - * / with parentheses.")

    return float(_eval(tree))


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
            math_expr = re.sub(r"[$€£¥₹,]", "", expression.strip())
            math_match = re.search(r"[\d\s\+\-\*\/\.\(\)]+", math_expr)
            if math_match:
                pure_math = math_match.group().strip()
                if pure_math and any(c.isdigit() for c in pure_math):
                    if len(pure_math) > 120:
                        return "Expression too long. Keep it under 120 characters."
                    result = safe_eval_math(pure_math)
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
        "Optional currency key is supported. "
        "Example: 'currency:USD, accommodation:600, food:350, transport:200, activities:150'"
    )

    def _run(self, costs_input: str) -> str:
        logger.info(f"[BudgetSummary] Processing: {costs_input}")
        try:
            components = {}
            currency_symbol = "$"

            for pair in costs_input.split(","):
                if ":" not in pair:
                    continue
                raw_key, raw_value = pair.split(":", 1)
                key = raw_key.strip()
                value = raw_value.strip()
                if not key:
                    continue

                if key.lower() in {"currency", "currency_symbol", "symbol"}:
                    currency_symbol = value or "$"
                    continue

                normalized_value = value.replace(",", "")
                if re.fullmatch(r"-?\d+(?:\.\d+)?", normalized_value):
                    components[key] = float(normalized_value)

            if not components:
                return f"Could not parse budget components from: '{costs_input}'"

            total = sum(components.values())
            lines = ["=" * 40, "BUDGET SUMMARY", "=" * 40]
            for category, amount in components.items():
                pct = (amount / total * 100) if total > 0 else 0
                lines.append(
                    f"  {category.title():<20} {currency_symbol}{amount:>8.2f}  ({pct:.1f}%)"
                )
            lines.append("-" * 40)
            lines.append(f"  {'TOTAL':<20} {currency_symbol}{total:>8.2f}  (100%)")
            lines.append("=" * 40)
            return "\n".join(lines)
        except Exception as e:
            return f"Budget summary error: {str(e)}"
