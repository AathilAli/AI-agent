import requests
from decimal import Decimal


def convert_currency(
    amount: float,
    from_currency: str,
    to_currency: str = "INR"
) -> dict:
    """
    Convert an amount from one currency to another
    using the Frankfurter exchange-rate API.
    """

    from_currency = from_currency.upper()
    to_currency = to_currency.upper()

    # No conversion needed
    if from_currency == to_currency:
        return {
            "original_amount": amount,
            "from_currency": from_currency,
            "to_currency": to_currency,
            "exchange_rate": 1,
            "converted_amount": amount
        }

    url = (
        f"https://api.frankfurter.dev/v2/"
        f"rate/{from_currency}/{to_currency}"
    )

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        data = response.json()

        rate = Decimal(str(data["rate"]))
        original_amount = Decimal(str(amount))
        converted_amount = original_amount * rate

        return {
            "original_amount": float(original_amount),
            "from_currency": from_currency,
            "to_currency": to_currency,
            "exchange_rate": float(rate),
            "converted_amount": round(float(converted_amount), 2),
            "rate_date": data["date"]
        }

    except requests.RequestException as e:
        raise RuntimeError(
            f"Currency API request failed: {e}"
        )

    except KeyError:
        raise RuntimeError(
            "Unexpected response received from currency API."
        )