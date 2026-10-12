
import re

from tools.ocr_tool import extract_receipt as extract_ocr_receipt


def extract_receipt(image_path: str) -> dict:
    """
    Extract structured receipt data using local PaddleOCR.

    Supported currencies:
        INR - Indian Rupee
        EUR - Euro
        USD - US Dollar
        AED - UAE Dirham

    No LLM API call is made in this function.
    """

    # Extract text and structured fields using PaddleOCR.
    ocr_data = extract_ocr_receipt(image_path)

    raw_text = ocr_data.get("raw_text") or ""
    text_upper = raw_text.upper()

    result = {
        "merchant": ocr_data.get("vendor"),
        "date": ocr_data.get("date"),
        "amount": ocr_data.get("total"),
        "currency": None,
        "category": None,
        "bill_no": ocr_data.get("bill_no"),
        "subtotal": ocr_data.get("subtotal"),
        "tax": ocr_data.get("tax"),
        "items": ocr_data.get("items", []),
        "raw_text": raw_text,
    }

    # --------------------------------------------------
    # 1. Detect the four supported currencies
    # --------------------------------------------------

    currency_patterns = [
        ("AED", r"\bAED\b|\bDIRHAMS?\b"),
        ("EUR", r"\bEUR\b|\bEUROS?\b|€"),
        ("USD", r"\bUSD\b|\bUS\s+DOLLARS?\b|\$"),
        (
            "INR",
            r"\bINR\b|₹|\bRUPEES?\b|\bRS\.?\s*\d",
        ),
    ]

    def detect_currency(text: str):
        """Return a supported currency code or None."""

        for currency_code, pattern in currency_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return currency_code

        return None

    # Prefer currency appearing on a final-total line.
    total_lines = re.findall(
        r"^.*\b(?:GRAND\s+TOTAL|TOTAL\s+DUE|TOTAL)\b.*$",
        raw_text,
        flags=re.IGNORECASE | re.MULTILINE,
    )

    for total_line in total_lines:
        detected_currency = detect_currency(total_line)

        if detected_currency:
            result["currency"] = detected_currency
            break

    # If the total line has no recognizable currency,
    # inspect the complete receipt.
    if result["currency"] is None:
        result["currency"] = detect_currency(raw_text)
        
    # Infer INR when the currency symbol is missing but
    # India-specific receipt indicators are present.
    if result["currency"] is None:
        has_indian_gstin = re.search(
            r"\b\d{2}[A-Z]{5}\d{4}[A-Z][0-9A-Z]Z[0-9A-Z]\b",
            text_upper,
        )

        has_upi_payment = re.search(
            r"\bUPI\b",
            text_upper,
        )

        if has_indian_gstin or has_upi_payment:
            result["currency"] = "INR"


    # --------------------------------------------------
    # 2. Detect the expense category
    # --------------------------------------------------

    merchant = str(result.get("merchant") or "").upper()
    combined_text = f"{merchant}\n{text_upper}"

    # Hotel expenses
    if any(word in combined_text for word in [
        "HOTEL",
        "LODGING",
        "ACCOMMODATION",
        "ROOM STAY",
        "ROOM CHARGE",
        "GUEST NAME",
        "CHECK-IN",
        "CHECK-OUT",
    ]):
        category = "Hotel"

    # Flight expenses
    elif any(word in combined_text for word in [
        "AIRLINE",
        "FLIGHT",
        "BOARDING PASS",
        "AIRWAYS",
        "ELECTRONIC TICKET",
        "BASE FARE",
        "PASSENGER SERVICE CHARGE",
    ]):
        category = "Flight"

    # Taxi and ride expenses
    elif any(word in combined_text for word in [
        "TAXI",
        "CAB FARE",
        "UBER",
        "LYFT",
        "OLA",
        "RIDE RECEIPT",
        "RIDE FARE",
    ]):
        category = "Taxi"

    # Food and restaurant expenses
    elif any(word in combined_text for word in [
        "RESTAURANT",
        "FOOD",
        "CAFE",
        "CAFÉ",
        "COFFEE",
        "STARBUCKS",
        "MUFFIN",
        "BIRYANI",
        "MEAL",
        "BREAKFAST",
        "LUNCH",
        "DINNER",
        "BARBEQUE NATION",
        "THE SPICE HOUSE",
        "PRET A MANGER",
        "SUKIYA",
    ]):
        category = "Food"

    # Other expenses use the existing Other policy category.
    else:
        category = "Other"

    result["category"] = category

    return result
