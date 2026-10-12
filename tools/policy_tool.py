import json
from pathlib import Path


# Find policy.json in the project root
BASE_DIR = Path(__file__).resolve().parent.parent
POLICY_FILE = BASE_DIR / "policy.json"


def load_policies() -> dict:
    """Load expense limits from policy.json."""

    with open(POLICY_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def check_policy(category: str, amount: float) -> dict:
    """
    Check whether an expense is within the allowed policy limit.
    """

    policies = load_policies()

    category = category.strip().title()

    # Check whether category exists
    if category not in policies:
        return {
            "status": "Unknown Category",
            "category": category,
            "amount": amount,
            "limit": None,
            "message": "No policy found for this category."
        }

    limit = policies[category]

    if amount <= limit:
        return {
            "status": "Valid",
            "category": category,
            "amount": amount,
            "limit": limit,
            "message": "Expense is within the allowed limit."
        }

    excess = amount - limit

    return {
        "status": "Violation",
        "category": category,
        "amount": amount,
        "limit": limit,
        "excess": excess,
        "message": f"Expense exceeds the allowed limit by ₹{excess:.2f}."
    }