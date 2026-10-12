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

SUPPORTED_CURRENCIES = {"INR", "EUR", "USD", "AED"}
FOREIGN_CURRENCIES = {"EUR", "USD", "AED"}
ALLOWED_CATEGORIES = {"Food", "Hotel", "Taxi", "Flight", "Other"}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "convert_currency",
            "description": (
                "Convert a foreign-currency receipt total into INR. "
                "Do not use this tool for INR receipts. For EUR, USD, "
                "or AED, call this before policy validation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "amount": {
                        "type": "number",
                        "description": "The original final receipt total",
                    },
                    "from_currency": {
                        "type": "string",
                        "enum": ["EUR", "USD", "AED", "INR"],
                    },
                    "to_currency": {
                        "type": "string",
                        "enum": ["INR"],
                    },
                },
                "required": ["amount", "from_currency", "to_currency"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_policy",
            "description": (
                "Check the company's policy using the expense amount in INR. "
                "For INR receipts, use the original amount directly. For EUR, "
                "USD, or AED receipts, call this only after a successful "
                "currency conversion."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": ["Food", "Hotel", "Taxi", "Flight", "Other"],
                    },
                    "amount_inr": {"type": "number"},
                },
                "required": ["category", "amount_inr"],
            },
        },
    },
]


def _parse_json(text):
    """Parse a JSON object returned by the model."""
    if not text:
        return {}

    text = text.strip()

    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    # Tolerate a response wrapped in Markdown code fences.
    start = text.find("{")
    end = text.rfind("}")

    if start >= 0 and end > start:
        try:
            data = json.loads(text[start : end + 1])
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass

    return {}


def run_receipt_agent(receipt: dict) -> dict:
    """
    Use DeepSeek to interpret an OCR result, choose tools, and evaluate results.

    DeepSeek chooses which tool to call. Python executes the tools and applies
    safety checks. The original amount and currency are preserved separately
    from the INR-converted amount.
    """
    initial_amount = receipt.get("amount")
    initial_currency = receipt.get("currency")

    if initial_currency:
        initial_currency = str(initial_currency).strip().upper()

    # A currency inferred by the model is not treated as verified evidence.
    state = {
        "conversion": None,
        "policy": None,
        "policy_category": None,
        "effective_currency": initial_currency,
        "errors": [],
    }

    def execute_convert(args):
        """Validate and execute a currency conversion, with an INR no-op guard."""
        try:
            amount = float(args["amount"])
            source = str(args["from_currency"]).strip().upper()
            target = str(args["to_currency"]).strip().upper()

            if amount < 0:
                raise ValueError("Amount cannot be negative.")
            if source not in SUPPORTED_CURRENCIES:
                raise ValueError(f"Unsupported currency: {source}")
            if target != "INR":
                raise ValueError("Target currency must be INR.")

            # Never allow the LLM to convert a different total than the OCR total.
            if initial_amount is not None:
                if abs(amount - float(initial_amount)) > 0.02:
                    raise ValueError(
                        "Requested amount does not match the extracted receipt "
                        "total. Manual review required."
                    )

            # If OCR already identified an INR receipt, never call the external
            # currency API, even if the LLM accidentally supplies another source
            # currency in its tool arguments. Python enforces this safety rule.
            if initial_currency == "INR":
                result = {
                    "original_amount": amount,
                    "from_currency": "INR",
                    "to_currency": "INR",
                    "exchange_rate": 1.0,
                    "converted_amount": amount,
                    "message": (
                        "Receipt is already in INR; external currency conversion "
                        "was skipped."
                    ),
                }
                state["effective_currency"] = "INR"
                state["conversion"] = result
                print(
                    "[AGENT TOOL] INR detected; skipping external currency conversion."
                )
                return result

            # If OCR detected a non-INR currency, the model cannot override it.
            if initial_currency and source != initial_currency:
                raise ValueError(
                    "Requested source currency does not match the detected "
                    "receipt currency."
                )

            # If currency was not detected by OCR but the LLM identified INR,
            # treat this tool call as a no-op; policy checking still uses the
            # extracted original amount and the detected/inferred currency.
            if source == "INR":
                result = {
                    "original_amount": amount,
                    "from_currency": "INR",
                    "to_currency": "INR",
                    "exchange_rate": 1.0,
                    "converted_amount": amount,
                    "message": (
                        "No conversion is needed for an INR receipt."
                    ),
                }
                state["effective_currency"] = "INR"
                state["conversion"] = result
                return result

            result = convert_currency(
                amount=amount,
                from_currency=source,
                to_currency=target,
            )

            state["effective_currency"] = source
            state["conversion"] = result
            return result

        except Exception as exc:
            state["errors"].append(f"Currency conversion: {exc}")
            return {"error": str(exc)}

    def execute_policy(args):
            """Validate policy using verified INR amount, not model-supplied math."""
            try:
                category = str(args["category"]).strip().title()
                if category not in ALLOWED_CATEGORIES:
                    raise ValueError(f"Invalid expense category: {category}")

                currency = state["effective_currency"] or initial_currency

                if currency == "INR":
                    if initial_amount is None:
                        raise ValueError("Receipt amount is missing.")
                    amount_inr = float(initial_amount)

                elif currency in FOREIGN_CURRENCIES:
                    conversion = state["conversion"]
                    if not conversion or "converted_amount" not in conversion:
                        # Give the LLM a recoverable tool result so it can call
                        # conversion first, rather than recording a final failure.
                        return {
                            "error": (
                                "Convert the foreign expense to INR successfully "
                                "before checking policy."
                            )
                        }
                    amount_inr = float(conversion["converted_amount"])

                else:
                    raise ValueError(
                        "Receipt currency could not be verified. Manual review required."
                    )

                if amount_inr < 0:
                    raise ValueError("Amount cannot be negative.")

                result = check_policy(category=category, amount=amount_inr)
                state["policy"] = result
                state["policy_category"] = category
                return result

            except Exception as exc:
                state["errors"].append(f"Policy validation: {exc}")
                return {"error": str(exc)}

    tool_functions = {
        "convert_currency": execute_convert,
        "check_policy": execute_policy,
    }

    messages = [    
        {
            "role": "system",
            "content": (
                "You are the reasoning agent for an employee expense-claim system. "
                "Analyze the OCR fields and raw receipt text. Identify merchant, "
                "date, final payable total, original currency, and one category: "
                "Food, Hotel, Taxi, Flight, or Other. Do not invent missing data. "
                "Supported currencies are INR, EUR, USD, and AED only. "
                "Choose tools based on the receipt: if the verified currency is "
                "INR, call check_policy directly with the original amount. Do not "
                "call convert_currency for an INR receipt. If the verified currency "
                "is EUR, USD, or AED, call convert_currency first with target INR, "
                "then call check_policy only after conversion succeeds. Never claim "
                "a policy result without the policy tool's actual result. Do not "
                "change the original amount or original currency. If a required "
                "field is missing, a tool fails, or currency is uncertain, flag the "
                "receipt for manual review. After tool results, return only a JSON "
                "object with merchant, date, amount, currency, category, "
                "needs_review, and summary. Do not include private internal reasoning."
            ),
        },
        {
            "role": "user",
            "content": (
                "Analyze this receipt and use the appropriate tools:\n"
                + json.dumps(receipt, ensure_ascii=False, default=str)
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
                extra_body={"thinking": {"type": "disabled"}},
            )

            message = response.choices[0].message

            if not message.tool_calls:
                final_data = _parse_json(message.content)
                break

            # Add the assistant's tool-call message to conversation history.
            messages.append(
                {
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
                }
            )

            # Execute each requested tool in Python and return its result.
            for call in message.tool_calls:
                try:
                    name = call.function.name
                    args = json.loads(call.function.arguments)

                    if name not in tool_functions:
                        raise ValueError(f"Unknown tool requested: {name}")

                    tool_result = tool_functions[name](args)

                except Exception as exc:
                    tool_result = {"error": str(exc)}
                    state["errors"].append(str(exc))

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": json.dumps(
                            tool_result,
                            ensure_ascii=False,
                            default=str,
                        ),
                    }
                )
        else:
            state["errors"].append(
                "Agent reached the maximum number of tool rounds."
            )

    except Exception as exc:
        state["errors"].append(f"DeepSeek API error: {exc}")

    # The policy tool is authoritative for the policy result.
    policy = state["policy"]
    conversion = state["conversion"]
    effective_currency = state["effective_currency"] or initial_currency

    amount_inr = None
    if effective_currency == "INR" and initial_amount is not None:
        amount_inr = float(initial_amount)
    elif conversion and "converted_amount" in conversion:
        amount_inr = float(conversion["converted_amount"])

    missing_required_data = (
        initial_amount is None
        or effective_currency not in SUPPORTED_CURRENCIES
        or amount_inr is None
        or policy is None
    )

    policy_status = policy.get("status", "Needs Review") if policy else "Needs Review"
    needs_review = (
        missing_required_data
        or bool(state["errors"])
        or bool(final_data.get("needs_review", False))
        or policy_status == "Violation"
        or not final_data
    )

    # Preserve the amount extracted from the receipt. Do not let the model
    # replace it with the converted INR amount in the final response.
    original_amount = initial_amount
    original_currency = effective_currency or final_data.get("currency") or ""

    return {
        "merchant": final_data.get("merchant") or receipt.get("merchant") or "",
        "date": final_data.get("date") or receipt.get("date") or "",
        "amount": original_amount,
        "currency": original_currency,
        "category": (
            state["policy_category"]
            or final_data.get("category")
            or receipt.get("category")
            or ""
        ),
        "amount_inr": amount_inr,
        "policy_status": policy_status,
        "policy_limit_inr": policy.get("limit", "") if policy else "",
        "policy_message": (
            policy.get("message", "")
            if policy
            else "Policy validation did not complete."
        ),
        "agent_summary": final_data.get("summary", ""),
        "needs_review": needs_review,
        "error": "; ".join(state["errors"]),
    }
