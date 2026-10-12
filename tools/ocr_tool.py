import os
import re

# Set these before importing PaddleOCR.
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

from paddleocr import PaddleOCR

AMOUNT = r"\d[\d,]*\.\d{2}"

# Reuse the OCR model instead of initializing it for every receipt.
_ocr = None




def get_ocr():
    global _ocr

    if _ocr is None:
        _ocr = PaddleOCR(
            text_detection_model_name="PP-OCRv5_mobile_det",
            text_recognition_model_name="en_PP-OCRv5_mobile_rec",
            enable_mkldnn=False,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )

    return _ocr


def run_ocr(image_path):
    """Extract detected text and its position from a receipt."""
    result = get_ocr().predict(image_path)[0]

    boxes = []

    for text, score, poly in zip(
        result["rec_texts"],
        result["rec_scores"],
        result["rec_polys"],
    ):
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]

        boxes.append({
            "text": text.strip(),
            "score": float(score),
            "x": min(xs),
            "y": (min(ys) + max(ys)) / 2,
            "h": max(ys) - min(ys),
        })

    return boxes


def group_into_lines(boxes):
    """Group detected text into reading-order lines."""
    if not boxes:
        return []

    boxes = sorted(boxes, key=lambda b: b["y"])
    heights = sorted(b["h"] for b in boxes)
    tolerance = 0.6 * heights[len(heights) // 2]

    lines = []
    current = [boxes[0]]

    for box in boxes[1:]:
        if abs(box["y"] - current[-1]["y"]) <= tolerance:
            current.append(box)
        else:
            lines.append(current)
            current = [box]

    lines.append(current)

    return [
        " ".join(box["text"] for box in sorted(line, key=lambda b: b["x"]))
        for line in lines
    ]


def amount_in(line):
    """Extract the last monetary amount from a line."""
    found = re.findall(AMOUNT, line)
    return float(found[-1].replace(",", "")) if found else None



def parse_receipt(lines):
    """Convert OCR lines into structured receipt data."""

    data = {
        "vendor": lines[0] if lines else None,
        "date": None,
        "time": None,
        "bill_no": None,
        "items": [],
        "subtotal": None,
        "tax": None,
        "total": None,
    }

    in_items = False

    for line in lines:
        # Correct common OCR confusion: 0ct -> Oct.
        line = re.sub(
            r"\b0(?=ct\b)",
            "O",
            line,
            flags=re.IGNORECASE,
        )

        low = line.lower()

        # Date: 14 Oct 2024, 14/10/2024, or 14-10-2024.
        match = re.search(
            r"(\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4}"
            r"|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
            line,
            re.IGNORECASE,
        )

        if match and not data["date"]:
            data["date"] = match.group(1)

        # Time: 10:15, 10:15 AM, or 10:15 PM.
        match = re.search(
            r"\b(\d{1,2}:\d{2}\s*(?:AM|PM)?)\b",
            line,
            re.IGNORECASE,
        )

        if match and not data["time"]:
            data["time"] = match.group(1).strip()

        # Receipt or bill number.
        match = re.search(
            r"(?:bill\s*(?:no|number)|receipt\s*(?:no|number))"
            r"\s*[:#.]?\s*([A-Za-z0-9-]+)",
            line,
            re.IGNORECASE,
        )

        if match:
            data["bill_no"] = match.group(1)

        # Start reading item rows.
        if "item" in low and "qty" in low:
            in_items = True
            continue

        # Subtotal ends the item section.
        if "subtotal" in low or "sub total" in low:
            in_items = False
            data["subtotal"] = amount_in(line)
            continue

        # Extract item name, quantity, and price.
        if in_items:
            match = re.match(
                rf"^(.+?)\s+(\d+)\s+({AMOUNT})$",
                line,
                re.IGNORECASE,
            )

            if match:
                data["items"].append({
                    "name": match.group(1).strip(),
                    "qty": int(match.group(2)),
                    "price": float(
                        match.group(3).replace(",", "")
                    ),
                })

            continue

        # Tax amount.
        if "gst" in low or "tax" in low or "vat" in low:
            data["tax"] = amount_in(line)

        # Final total.
        elif "total" in low:
            data["total"] = amount_in(line)

    return data


def extract_receipt(image_path):
    """Return structured data extracted from a receipt image."""
    boxes = run_ocr(image_path)
    lines = group_into_lines(boxes)
    receipt = parse_receipt(lines)

    # Keep the raw OCR lines for debugging and later interpretation.
    receipt["raw_text"] = "\n".join(lines)

    return receipt


def extract_text(image_path):
    """Return OCR text for compatibility with text-based callers."""
    return extract_receipt(image_path)["raw_text"]
