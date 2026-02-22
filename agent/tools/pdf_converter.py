import re
import os
from pathlib import Path

import pandas as pd
import pdfplumber


# ---------------------------------------------------------------------------
# Tax return CSV template — edit these defaults as needed
# ---------------------------------------------------------------------------
TAX_TEMPLATE = {
    "Personal Information": {
        "Full Name": "",
        "SSN (last 4 digits)": "",
        "Filing Status": "",  # Single / MFS / MFJ / HOH / QSS
        "Tax Year": "",
    },
    "Income": {
        "W-2 Wages": 0.0,
        "Self-Employment / Freelance Income": 0.0,
        "Interest Income (1099-INT)": 0.0,
        "Dividend Income (1099-DIV)": 0.0,
        "Capital Gains / Losses": 0.0,
        "Other Income": 0.0,
    },
    "Deductions": {
        "Deduction Type": "",  # Standard or Itemized
        "Mortgage Interest": 0.0,
        "Charitable Contributions": 0.0,
        "Medical Expenses": 0.0,
        "State & Local Taxes (SALT)": 0.0,
        "Other Deductions": 0.0,
    },
    "Tax Credits": {
        "Child Tax Credit": 0.0,
        "Education Credit": 0.0,
        "EV / Energy Credit": 0.0,
        "Other Credits": 0.0,
    },
    "Summary": {
        "Gross Income": 0.0,
        "Total Deductions": 0.0,
        "Taxable Income": 0.0,
        "Estimated Tax Owed": 0.0,
        "Taxes Already Paid (W-2 withholding)": 0.0,
        "Refund / Amount Due": 0.0,
    },
}


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract all text from a PDF file using pdfplumber."""
    text_parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    return "\n".join(text_parts)


def _find_value(text: str, *patterns: str) -> str:
    """Search for the first regex pattern match in text, return the capture group."""
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return ""


def _find_amount(text: str, *patterns: str) -> float:
    """Search for a dollar amount near a keyword, return as float."""
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            raw = match.group(1).replace(",", "").replace("$", "").strip()
            try:
                return float(raw)
            except ValueError:
                pass
    return 0.0


def parse_pdf_into_template(pdf_text: str) -> dict:
    """
    Best-effort extraction of tax fields from raw PDF text.
    Unrecognised fields are left at their defaults (blank / 0.0).
    The LLM node will describe what was and wasn't found.
    """
    data = {section: dict(fields) for section, fields in TAX_TEMPLATE.items()}

    t = pdf_text  # shorthand

    # --- Personal Information ---
    data["Personal Information"]["Full Name"] = _find_value(
        t, r"name[:\s]+([A-Za-z ,'-]+)", r"taxpayer[:\s]+([A-Za-z ,'-]+)"
    )
    data["Personal Information"]["SSN (last 4 digits)"] = _find_value(
        t, r"ssn.*?(\d{4})\b", r"social security.*?(\d{4})\b"
    )
    data["Personal Information"]["Filing Status"] = _find_value(
        t,
        r"filing status[:\s]+(\w[\w /]+)",
        r"(single|married filing jointly|married filing separately|head of household)",
    )
    data["Personal Information"]["Tax Year"] = _find_value(
        t, r"tax year[:\s]+(\d{4})", r"\b(20\d{2})\b"
    )

    # --- Income ---
    data["Income"]["W-2 Wages"] = _find_amount(
        t, r"wages.*?\$?([\d,]+\.?\d*)", r"w-?2.*?\$?([\d,]+\.?\d*)"
    )
    data["Income"]["Self-Employment / Freelance Income"] = _find_amount(
        t,
        r"self.employ.*?\$?([\d,]+\.?\d*)",
        r"freelance.*?\$?([\d,]+\.?\d*)",
        r"1099.*?\$?([\d,]+\.?\d*)",
    )
    data["Income"]["Interest Income (1099-INT)"] = _find_amount(
        t, r"interest income.*?\$?([\d,]+\.?\d*)", r"1099-int.*?\$?([\d,]+\.?\d*)"
    )
    data["Income"]["Dividend Income (1099-DIV)"] = _find_amount(
        t, r"dividend.*?\$?([\d,]+\.?\d*)", r"1099-div.*?\$?([\d,]+\.?\d*)"
    )
    data["Income"]["Capital Gains / Losses"] = _find_amount(
        t, r"capital gain.*?\$?([\d,]+\.?\d*)"
    )

    # --- Deductions ---
    data["Deductions"]["Deduction Type"] = _find_value(
        t, r"(standard|itemized) deduction"
    )
    data["Deductions"]["Mortgage Interest"] = _find_amount(
        t, r"mortgage interest.*?\$?([\d,]+\.?\d*)"
    )
    data["Deductions"]["Charitable Contributions"] = _find_amount(
        t, r"charit.*?\$?([\d,]+\.?\d*)"
    )
    data["Deductions"]["Medical Expenses"] = _find_amount(
        t, r"medical.*?\$?([\d,]+\.?\d*)"
    )
    data["Deductions"]["State & Local Taxes (SALT)"] = _find_amount(
        t, r"salt.*?\$?([\d,]+\.?\d*)", r"state.*?local.*?tax.*?\$?([\d,]+\.?\d*)"
    )

    # --- Tax Credits ---
    data["Tax Credits"]["Child Tax Credit"] = _find_amount(
        t, r"child tax credit.*?\$?([\d,]+\.?\d*)"
    )
    data["Tax Credits"]["Education Credit"] = _find_amount(
        t, r"education credit.*?\$?([\d,]+\.?\d*)"
    )
    data["Tax Credits"]["EV / Energy Credit"] = _find_amount(
        t, r"(?:ev|electric vehicle|energy) credit.*?\$?([\d,]+\.?\d*)"
    )

    # --- Summary ---
    data["Summary"]["Gross Income"] = _find_amount(
        t, r"gross income.*?\$?([\d,]+\.?\d*)", r"total income.*?\$?([\d,]+\.?\d*)"
    )
    data["Summary"]["Taxable Income"] = _find_amount(
        t, r"taxable income.*?\$?([\d,]+\.?\d*)"
    )
    data["Summary"]["Estimated Tax Owed"] = _find_amount(
        t, r"tax owed.*?\$?([\d,]+\.?\d*)", r"total tax.*?\$?([\d,]+\.?\d*)"
    )
    data["Summary"]["Taxes Already Paid (W-2 withholding)"] = _find_amount(
        t,
        r"withhold.*?\$?([\d,]+\.?\d*)",
        r"federal.*?withheld.*?\$?([\d,]+\.?\d*)",
    )
    data["Summary"]["Refund / Amount Due"] = _find_amount(
        t, r"refund.*?\$?([\d,]+\.?\d*)", r"amount due.*?\$?([\d,]+\.?\d*)"
    )

    return data


def template_to_csv(data: dict, output_path: str) -> str:
    """Convert the nested template dict into a tidy two-column CSV."""
    rows = []
    for section, fields in data.items():
        rows.append({"Section": f"=== {section} ===", "Field": "", "Value": ""})
        for field, value in fields.items():
            rows.append({"Section": section, "Field": field, "Value": value})

    df = pd.DataFrame(rows)[["Section", "Field", "Value"]]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def convert_pdf_to_csv(pdf_path: str, output_dir: str = "output") -> tuple[str, str]:
    """
    Full pipeline: PDF → extracted text → filled template → CSV file.
    Returns (csv_path, summary_for_llm).
    """
    pdf_text = extract_text_from_pdf(pdf_path)

    if not pdf_text.strip():
        raise ValueError("No text could be extracted from the PDF. It may be image-based (scanned). Try an OCR tool first.")

    data = parse_pdf_into_template(pdf_text)

    stem = Path(pdf_path).stem
    csv_path = str(Path(output_dir) / f"{stem}_tax_return.csv")
    template_to_csv(data, csv_path)

    # Build a human-readable summary for the LLM to relay back
    filled = []
    empty = []
    for section, fields in data.items():
        for field, value in fields.items():
            if value and value != 0.0:
                filled.append(f"  {section} > {field}: {value}")
            else:
                empty.append(f"  {section} > {field}")

    summary = (
        f"PDF processed: {Path(pdf_path).name}\n"
        f"CSV saved to: {csv_path}\n\n"
        f"Fields extracted ({len(filled)}):\n" + "\n".join(filled or ["  (none)"]) + "\n\n"
        f"Fields left blank ({len(empty)}) — fill manually:\n" + "\n".join(empty or ["  (none)"])
    )
    return csv_path, summary
