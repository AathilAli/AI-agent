
import json
import os

from dotenv import load_dotenv
from openai import OpenAI

from tools.currency_tool import convert_currency
from tools.policy_tool import check_policy


load_dotenv()

API_KEY = os.getenv("DEEPSEEK_API_KEY")

if not API_KEY:
    raise ValueError("DEEPSEEK_API_KEY is missing from .env")

client = OpenAI(
    api_key=API_KEY,
    base_url="https://api.deepseek.com",
)

MODEL = "deepseek-flash"
MAX_TOOL_ROUNDS = 4

ALLOWED_CATEGORIES = {
    "Food", "Hotel", "Taxi", "Flight", "Other"
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "convert_currency",
            "description": (
                "Convert a foreign receipt amount to INR. "
                "Use this before policy validation when the "
                "receipt currency is not INR."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "amount": {"type": "number"},
                    "from_currency": {"type": "string"},
                    "to_currency": {"type": "string"},
                },
                "required": [
                    "amount",
                    "from_currency",
                    "to_currency",
                ],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_policy",
            "description": (
                "Check the company's expense policy using the "
                "verified expense amount in INR. For foreign "
                "currency receipts, call only after conversion."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": [
                            "Food", "Hotel", "Taxi",
                            "Flight", "Other",
                        ],
                    },
                    "amount_inr": {"type": "number"},
                },
                "required": ["category", "amount_inr"],
            },
        },
    },
]


def _parse_json(text):
    """Parse a JSON object from the model's final answer."""
    if not text:
        return {}

    text = text.strip()

    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    # Also tolerate a response surrounded by Markdown fences.
    start = text.find("{")
    end = text.rfind("}")

    if start >= 0 and end > start:
        try:
            data = json.loads(text[start:end + 1])
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass

    return {}


def run_receipt_agent(receipt: dict) -> dict:
    """
    Use DeepSeek to reason about an OCR result, choose tools,
    interpret tool results, and return the expense decision.

    Python executes the tools. The model does not execute code.
    """

    initial_amount = receipt.get("amount")
    initial_currency = receipt.get("currency")

    if initial_currency:
        initial_currency = str(initial_currency).strip().upper()

    state = {
        "conversion": None,
        "policy": None,
        "errors": [],
    }

    def execute_convert(args):
        """Validate and execute currency conversion."""

        try:
            amount = float(args["amount"])
            source = str(args["from_currency"]).upper().strip()
            target = str(args["to_currency"]).upper().strip()

            if amount < 0:
                raise ValueError("Amount cannot be negative.")

            if target != "INR":
                raise ValueError("Target currency must be INR.")

            # Avoid converting a different total from the OCR result.
            if initial_amount is not None:
                if abs(amount - float(initial_amount)) > 0.02:
                    raise ValueError(
                        "Requested amount does not match the "
                        "extracted receipt total. Manual review needed."
                    )

            if initial_currency and source != initial_currency:
                raise ValueError(
                    "Requested source currency does not match "
                    "the detected receipt currency."
                )

            result = convert_currency(
                amount=amount,
                from_currency=source,
                to_currency=target,
            )

            state["conversion"] = result

            return result

        except Exception as exc:
            state["errors"].append(
                f"Currency conversion: {exc}"
            )
            return {"error": str(exc)}

    def execute_policy(args):
        """Validate an expense using the verified INR amount."""

        try:
            category = str(args["category"]).strip().title()

            if category not in ALLOWED_CATEGORIES:
                raise ValueError("Invalid expense category.")

            # For foreign currency, conversion must succeed first.
            if initial_currency != "INR":
                conversion = state["conversion"]

                if (
                    not conversion
                    or "converted_amount" not in conversion
                ):
                    raise ValueError(
                        "Convert the foreign expense to INR "
                        "successfully before checking policy."
                    )

                # Use the actual conversion result, not a number
                # independently supplied by the language model.
                amount_inr = float(
                    conversion["converted_amount"]
                )

            else:
                if initial_amount is None:
                    raise ValueError(
                        "Receipt amount is missing."
                    )

                amount_inr = float(initial_amount)

            if amount_inr < 0:
                raise ValueError("Amount cannot be negative.")

            result = check_policy(
                category=category,
                amount=amount_inr,
            )

            state["policy"] = result
            state["policy_category"] = category

            return result

        except Exception as exc:
            state["errors"].append(
                f"Policy validation: {exc}"
            )
            return {"error": str(exc)}

    tool_functions = {
        "convert_currency": execute_convert,
        "check_policy": execute_policy,
    }

    # The LLM receives the OCR output and decides which tool to call.
    messages = [
        {
            "role": "system",
            "content": (
                "You are the reasoning agent in an expense claim "
                "system. Analyze the OCR fields and raw receipt text. "
                "Do not invent missing information. Identify the "
                "merchant, date, final payable amount, currency and "
                "one category: Food, Hotel, Taxi, Flight, Other. "
                "If currency is foreign, select convert_currency "
                "first. After successful conversion, select "
                "check_policy. If currency is INR, check policy "
                "directly. Never claim policy compliance without "
                "the actual policy tool result. If information is "
                "uncertain, explain that manual review is required. "
                "After tools finish, return only a JSON object with "
                "merchant, date, amount, currency, category, "
                "needs_review and summary. Do not output private "
                "internal reasoning."
            ),
        },
        {
            "role": "user",
            "content": (
                "Analyze this receipt and use tools as required:\n"
                + json.dumps(
                    receipt,
                    ensure_ascii=False,
                    default=str,
                )
            ),
        },
    ]

    final_data = {}

    try:
        for _ in range(MAX_TOOL_ROUNDS):
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                max_tokens=800,
                extra_body={
                    "thinking": {"type": "disabled"}
                },
            )

            message = response.choices[0].message

            if not message.tool_calls:
                final_data = _parse_json(message.content)
                break

            # Preserve the assistant message containing tool calls.
            messages.append({
                "role": "assistant",
                "content": message.content,
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.function.name,
                            "arguments": call.function.arguments,
                        },
                    }
                    for call in message.tool_calls
                ],
            })

            for call in message.tool_calls:
                name = call.function.name

                try:
                    args = json.loads(call.function.arguments)

                    if name not in tool_functions:
                        raise ValueError(
                            f"Unknown tool requested: {name}"
                        )

                    result = tool_functions[name](args)

                except Exception as exc:
                    result = {"error": str(exc)}
                    state["errors"].append(str(exc))

                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(
                        result,
                        ensure_ascii=False,
                        default=str,
                    ),
                })

        else:
            state["errors"].append(
                "Agent reached the maximum number of tool rounds."
            )

    except Exception as exc:
        state["errors"].append(
            f"DeepSeek API error: {exc}"
        )

    # The policy tool is the authority for the actual policy result.
    policy = state["policy"]
    conversion = state["conversion"]

    amount_inr = None

    if initial_currency == "INR" and initial_amount is not None:
        amount_inr = float(initial_amount)
    elif conversion and "converted_amount" in conversion:
        amount_inr = conversion["converted_amount"]

    missing_required_data = (
        initial_amount is None
        or not initial_currency
        or amount_inr is None
        or policy is None
    )

    needs_review = (
        missing_required_data
        or bool(state["errors"])
        or bool(final_data.get("needs_review", False))
    )

    if not final_data:
        needs_review = True

    return {
        "merchant": (
            final_data.get("merchant")
            or receipt.get("merchant")
            or ""
        ),
        "date": (
            final_data.get("date")
            or receipt.get("date")
            or ""
        ),
        "amount": (
            final_data.get("amount")
            if final_data.get("amount") is not None
            else initial_amount
        ),
        "currency": (
            final_data.get("currency")
            or initial_currency
            or ""
        ),
        "category": (
            state.get("policy_category")
            or final_data.get("category")
            or receipt.get("category")
            or ""
        ),
        "amount_inr": amount_inr,
        "policy_status": (
            policy.get("status", "Needs Review")
            if policy
            else "Needs Review"
        ),
        "policy_limit_inr": (
            policy.get("limit", "")
            if policy
            else ""
        ),
        "policy_message": (
            policy.get("message", "")
            if policy
            else "Policy validation did not complete."
        ),
        "agent_summary": final_data.get("summary", ""),
        "needs_review": needs_review,
        "error": "; ".join(state["errors"]),
    }
