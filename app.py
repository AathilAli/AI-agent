import csv
import time
from pathlib import Path

import streamlit as st

from agent.expense_agent import run_receipt_agent
from tools.receipt_tool import extract_receipt


# --------------------------------------------------
# 1. Streamlit configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Expense Claim AI Agent",
    page_icon="🧾",
    layout="wide",
)

st.title("🧾 Expense Claim AI Agent")
st.caption(
    "Process employee receipts, convert currencies, validate policies, "
    "and generate individual CSV reports."
)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# Initialize session state before creating the widgets.
if "input_folder" not in st.session_state:
    st.session_state["input_folder"] = ""

if "output_folder" not in st.session_state:
    st.session_state["output_folder"] = ""


# --------------------------------------------------
# 2. Folder selection
# --------------------------------------------------
def choose_folder():
    """Open a native Windows folder-selection dialog."""
    import tkinter as tk
    from tkinter import filedialog

    root = None
    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selected = filedialog.askdirectory(title="Select a folder")
        return selected or ""
    finally:
        if root is not None:
            root.destroy()


def browse_input_folder():
    """Callback for selecting the employee input folder."""
    selected = choose_folder()
    if selected:
        st.session_state["input_folder"] = selected


def browse_output_folder():
    """Callback for selecting the CSV output folder."""
    selected = choose_folder()
    if selected:
        st.session_state["output_folder"] = selected


# --------------------------------------------------
# 3. Process one receipt
# --------------------------------------------------
def process_receipt(image_path: Path, employee_name: str) -> dict:
    """Extract a receipt, run the DeepSeek agent, and validate expenses."""
    row = {
        "employee": employee_name,
        "receipt_file": image_path.name,
        "merchant": "",
        "date": "",
        "original_amount": "",
        "currency": "",
        "amount_inr": "",
        "category": "",
        "policy_status": "Needs Review",
        "policy_limit_inr": "",
        "policy_message": "",
        "needs_review": True,
        "agent_summary": "",
        "error": "",
    }

    try:
        # Step 1: Extract receipt fields locally with PaddleOCR.
        ocr_start = time.perf_counter()
        receipt = extract_receipt(str(image_path))
        print(f"[TIMING] OCR: {time.perf_counter() - ocr_start:.2f} seconds")

        # Preserve the source receipt values as the baseline.
        row["merchant"] = receipt.get("merchant") or ""
        row["date"] = receipt.get("date") or ""
        original_amount = receipt.get("amount")
        row["original_amount"] = (
            original_amount if original_amount is not None else ""
        )

        ocr_currency = receipt.get("currency")
        if ocr_currency:
            row["currency"] = str(ocr_currency).strip().upper()

        row["category"] = receipt.get("category") or ""

        # Step 2: Ask DeepSeek to reason about the receipt and choose tools.
        agent_start = time.perf_counter()
        result = run_receipt_agent(receipt)
        print(
            f"[TIMING] DeepSeek agent: "
            f"{time.perf_counter() - agent_start:.2f} seconds"
        )

        # Step 3: Use the agent's interpretation where appropriate.
        row["merchant"] = result.get("merchant") or row["merchant"]
        row["date"] = result.get("date") or row["date"]
        row["category"] = result.get("category") or row["category"]

        # Prefer the currency explicitly detected from the receipt.
        # Use the agent's currency only when the OCR adapter found none.
        if not row["currency"]:
            inferred_currency = result.get("currency")
            if inferred_currency:
                row["currency"] = str(inferred_currency).strip().upper()

        # Keep original_amount in the source currency. Never replace it
        # with the converted INR amount returned by the agent.
        row["original_amount"] = (
            original_amount if original_amount is not None else ""
        )

        amount_inr = result.get("amount_inr")
        if amount_inr is not None:
            row["amount_inr"] = amount_inr

        row["policy_status"] = result.get("policy_status") or "Needs Review"
        row["policy_limit_inr"] = result.get("policy_limit_inr") or ""
        row["policy_message"] = result.get("policy_message") or ""
        row["agent_summary"] = result.get("agent_summary") or ""
        row["needs_review"] = bool(result.get("needs_review", True))
        row["error"] = result.get("error") or ""

        # Missing required data means a manual review is necessary.
        if (
            original_amount is None
            or not row["currency"]
            or not row["category"]
            or row["amount_inr"] == ""
            or row["policy_status"] == "Needs Review"
        ):
            row["needs_review"] = True

        # A policy violation is a finance-review outcome, not a technical
        # processing error. Keep the message explicit for the reviewer.
        if row["policy_status"] == "Violation":
            row["needs_review"] = True
            if not row["error"]:
                row["error"] = "Policy violation; finance review required."

        elif row["needs_review"] and not row["error"]:
            row["error"] = (
                "Manual review required; verify receipt data and agent results."
            )

    except Exception as exc:
        row["policy_status"] = "Needs Review"
        row["needs_review"] = True
        row["error"] = str(exc)
        row["agent_summary"] = (
            "Receipt processing failed; manual review required."
        )

    return row


# --------------------------------------------------
# 4. Generate one CSV per employee
# --------------------------------------------------
CSV_FIELDS = [
    "employee",
    "receipt_file",
    "merchant",
    "date",
    "original_amount",
    "currency",
    "amount_inr",
    "category",
    "policy_status",
    "policy_limit_inr",
    "policy_message",
    "needs_review",
    "agent_summary",
    "error",
]


def write_employee_csv(output_root: Path, employee_name: str, rows: list) -> Path:
    """Write all receipt rows for one employee into one CSV report."""
    output_file = output_root / f"{employee_name}.csv"

    with output_file.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=CSV_FIELDS,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)

    return output_file


# --------------------------------------------------
# 5. Folder selection UI
# --------------------------------------------------
st.subheader("1. Select folders")

col1, col2 = st.columns(2)

with col1:
    st.text_input(
        "Employee input folder",
        key="input_folder",
        placeholder=r"C:\employee",
        help="Contains one subfolder per employee.",
    )
    st.button(
        "📂 Browse input folder",
        key="browse_input_button",
        on_click=browse_input_folder,
        width="stretch",
    )

with col2:
    st.text_input(
        "CSV output folder",
        key="output_folder",
        placeholder=r"C:\outputpath",
        help="CSV reports will be saved here.",
    )
    st.button(
        "📂 Browse output folder",
        key="browse_output_button",
        on_click=browse_output_folder,
        width="stretch",
    )

st.info(
    "Input example: C:\\employee\\person1\\receipt1.jpg. "
    "Each employee must have their own subfolder."
)

st.divider()


# --------------------------------------------------
# 6. Process all employee folders
# --------------------------------------------------
st.subheader("2. Process receipts")

if st.button(
    "🚀 Process receipts and generate CSV",
    type="primary",
    width="stretch",
):
    input_text = st.session_state["input_folder"].strip()
    output_text = st.session_state["output_folder"].strip()

    if not input_text:
        st.error("Select the employee input folder.")
        st.stop()

    if not output_text:
        st.error("Select the CSV output folder.")
        st.stop()

    input_root = Path(input_text).expanduser()
    if not input_root.is_dir():
        st.error("The selected input folder does not exist.")
        st.stop()

    output_root = Path(output_text).expanduser()

    # Prevent writing reports inside the input folder tree.
    input_resolved = input_root.resolve()
    output_resolved = output_root.resolve()
    if (
        input_resolved == output_resolved
        or input_resolved in output_resolved.parents
    ):
        st.error(
            "Choose an output folder outside the input folder. "
            "This prevents reports from mixing with employee receipts."
        )
        st.stop()

    try:
        output_root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        st.error(f"Cannot create or access output folder: {exc}")
        st.stop()

    if not output_root.is_dir():
        st.error("The output path is not a directory.")
        st.stop()

    # Each immediate subfolder represents one employee.
    employee_folders = sorted(
        (folder for folder in input_root.iterdir() if folder.is_dir()),
        key=lambda folder: folder.name.lower(),
    )

    if not employee_folders:
        st.warning("No employee subfolders were found in the input folder.")
        st.stop()

    # Gather supported images for each employee.
    employee_images = {}
    for employee_folder in employee_folders:
        images = sorted(
            (
                file
                for file in employee_folder.iterdir()
                if file.is_file()
                and file.suffix.lower() in SUPPORTED_EXTENSIONS
            ),
            key=lambda file: file.name.lower(),
        )
        if images:
            employee_images[employee_folder.name] = images

    total_receipts = sum(len(images) for images in employee_images.values())
    if total_receipts == 0:
        st.warning(
            "No supported receipt images were found. "
            "Use JPG, JPEG, PNG, or WEBP files."
        )
        st.stop()

    results = []
    csv_files = []
    errors = []
    processed = 0

    progress = st.progress(0)
    status = st.empty()

    for employee_name, images in employee_images.items():
        employee_rows = []

        for image_path in images:
            status.write(f"Processing {employee_name}: {image_path.name}")
            row = process_receipt(image_path, employee_name)
            employee_rows.append(row)
            results.append(row)

            if row["error"]:
                errors.append(
                    f"{employee_name}/{image_path.name}: {row['error']}"
                )

            processed += 1
            progress.progress(processed / total_receipts)

        # Only write a CSV if this employee has supported receipt images.
        try:
            csv_file = write_employee_csv(
                output_root=output_root,
                employee_name=employee_name,
                rows=employee_rows,
            )
            csv_files.append(csv_file)
        except OSError as exc:
            errors.append(f"Could not write {employee_name}.csv: {exc}")

    status.empty()
    st.session_state["report_rows"] = results

    st.success(f"Finished processing {processed} receipt(s).")

    # --------------------------------------------------
    # 7. Display results
    # --------------------------------------------------
    st.subheader("3. Processing results")
    st.dataframe(results, width="stretch", hide_index=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Receipts processed", processed)
    col2.metric("Employee CSV files", len(csv_files))
    col3.metric(
        "Receipts needing review",
        sum(1 for row in results if row["needs_review"]),
    )

    # --------------------------------------------------
    # 8. Display and download generated CSV files
    # --------------------------------------------------
    st.subheader("4. Generated CSV files")

    if csv_files:
        st.write(f"Output location: `{output_root}`")
        for csv_file in csv_files:
            st.write(f"✅ `{csv_file.name}`")
            try:
                csv_data = csv_file.read_bytes()
                st.download_button(
                    label=f"⬇️ Download {csv_file.name}",
                    data=csv_data,
                    file_name=csv_file.name,
                    mime="text/csv",
                    key=f"download_{csv_file.name}",
                )
            except OSError as exc:
                st.warning(f"Could not read {csv_file.name}: {exc}")
    else:
        st.warning("No employee CSV files were generated.")

    if errors:
        st.warning(
            "Some receipts need review or some CSV files could not be written."
        )
        with st.expander("View errors"):
            for error in errors:
                st.write(f"- {error}")
