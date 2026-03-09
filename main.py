"""
Extract tables from CAM122025.pdf (Cambodia Tourism Statistics Report, December 2025)
and write each table to a separate CSV file in the output/ directory.
"""

import os
import re

import pandas as pd
import pdfplumber

PDF_PATH = os.path.join(os.path.dirname(__file__), "CAM122025.pdf")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _numeric_token(tok: str) -> bool:
    """Return True if tok looks like a number (integer or decimal, optional sign/commas)."""
    return bool(re.match(r'^-?[\d,]+(\.\d+)?$', tok))


def _split_country_row(line: str):
    """
    Split a country-level data row into (name, [8 numeric values]).
    The row format is: <name>  <2024>  <holiday>  <business>  <others>  <total>  <female>  <share%>  <change%>
    Returns (name_str, list_of_8_tokens) or None if the line cannot be parsed.
    """
    tokens = line.split()
    if len(tokens) < 3:
        return None
    # Walk from the right collecting numeric tokens until we have 8
    count = 0
    split_idx = len(tokens)
    for i in range(len(tokens) - 1, -1, -1):
        if _numeric_token(tokens[i]):
            count += 1
            split_idx = i
            if count == 8:
                break
        else:
            break
    if count < 8:
        return None
    name = " ".join(tokens[:split_idx]).strip()
    data = tokens[split_idx: split_idx + 8]
    if not name:
        return None
    return name, data


def _write_csv(df: pd.DataFrame, filename: str) -> None:
    """Write DataFrame to output/<filename> as UTF-8 CSV (no BOM)."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(path, index=False, encoding="utf-8")
    print(f"{filename}: {len(df)} rows x {len(df.columns)} columns")


def _page_text(pdf, page_index: int) -> str:
    return pdf.pages[page_index].extract_text() or ""


def _lines(text: str):
    return [ln.strip() for ln in text.splitlines()]


# ---------------------------------------------------------------------------
# Table 1 – Tourism Highlights (page 2)
# ---------------------------------------------------------------------------

def extract_tourism_highlights(pdf) -> pd.DataFrame:
    text = _page_text(pdf, 1)
    rows = []

    # Context for the 2020 anomaly:
    # PDF text order: "Jan - Mar: …" line, then the "2020 …" year row, then "Apr - Dec: …" line.
    # We store Jan-Mar values, defer the 2020 row, then complete it when Apr-Dec values arrive.
    saved_jan_mar_stay = saved_jan_mar_occ = ""
    deferred_row = None  # holds the partial 2020 row

    def _flush_deferred(apr_stay: str, apr_occ: str) -> None:
        nonlocal deferred_row
        if deferred_row is not None:
            deferred_row["Average Length of Stays (Days)"] = (
                f"Jan-Mar:{saved_jan_mar_stay} / Apr-Dec:{apr_stay}"
            )
            deferred_row["Hotel Occupancy (%)"] = (
                f"Jan-Mar:{saved_jan_mar_occ} / Apr-Dec:{apr_occ}"
            )
            rows.append(deferred_row)
            deferred_row = None

    # We need access to saved_jan_mar_* inside the nested function, so keep them at
    # function scope but update via a mutable container trick
    jm = {"stay": "", "occ": ""}

    for line in _lines(text):
        # "Jan - Mar:" supplementary line (appears BEFORE the 2020 year row)
        if line.startswith("Jan - Mar:"):
            nums = re.findall(r"[\d.]+", line)
            if len(nums) >= 2:
                jm["stay"], jm["occ"] = nums[0], nums[1]
            continue

        # "Apr - Dec:" supplementary line (appears AFTER the 2020 year row)
        if line.startswith("Apr - Dec:"):
            nums = re.findall(r"[\d.]+", line)
            apr_stay = nums[0] if len(nums) >= 1 else ""
            apr_occ = nums[1] if len(nums) >= 2 else ""
            if deferred_row is not None:
                deferred_row["Average Length of Stays (Days)"] = (
                    f"Jan-Mar:{jm['stay']} / Apr-Dec:{apr_stay}"
                )
                deferred_row["Hotel Occupancy (%)"] = (
                    f"Jan-Mar:{jm['occ']} / Apr-Dec:{apr_occ}"
                )
                rows.append(deferred_row)
                deferred_row = None
                jm["stay"] = jm["occ"] = ""
            continue

        # Year rows: start with 1993–2024
        m = re.match(r"^(19\d{2}|20[012]\d)\s+(.*)", line)
        if not m:
            continue

        year = m.group(1)
        rest = m.group(2)
        tokens = rest.split()

        if jm["stay"]:
            # 2020 special row – defer until we read the Apr-Dec line
            # The year row contains: number, change, receipts, share
            # (stays/occupancy will be filled when Apr-Dec line arrives)
            deferred_row = {
                "Years": year,
                "Int'l Tourist Arrivals Number": tokens[0] if len(tokens) > 0 else "",
                "Change (%)": tokens[1] if len(tokens) > 1 else "",
                "Average Length of Stays (Days)": "",  # filled when Apr-Dec line arrives
                "Hotel Occupancy (%)": "",              # filled when Apr-Dec line arrives
                "Int'l Tourism Receipts (Million USD)": tokens[2] if len(tokens) > 2 else "",
                "Tourism Sector Share to the GDP (%)": tokens[3] if len(tokens) > 3 else "",
            }
        else:
            # Normal rows have 5 or 6 tokens after the year
            rows.append({
                "Years": year,
                "Int'l Tourist Arrivals Number": tokens[0] if len(tokens) > 0 else "",
                "Change (%)": tokens[1] if len(tokens) > 1 else "",
                "Average Length of Stays (Days)": tokens[2] if len(tokens) > 2 else "",
                "Hotel Occupancy (%)": tokens[3] if len(tokens) > 3 else "",
                "Int'l Tourism Receipts (Million USD)": tokens[4] if len(tokens) > 4 else "",
                "Tourism Sector Share to the GDP (%)": tokens[5] if len(tokens) > 5 else "",
            })

    # Safety flush (should not be needed with correct PDF)
    if deferred_row is not None:
        rows.append(deferred_row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Table 2 – International Tourist Arrivals (monthly/quarterly, page 2)
# ---------------------------------------------------------------------------

def extract_international_arrivals(pdf) -> pd.DataFrame:
    text = _page_text(pdf, 1)
    rows = []

    # Month/quarter labels that begin each data row
    labels = {"Q1", "Q2", "Q3", "Q4", "Jan", "Feb", "Mar", "Apr",
              "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec", "Total"}

    for line in _lines(text):
        tokens = line.split()
        if not tokens or tokens[0] not in labels:
            continue
        if len(tokens) < 12:
            continue
        rows.append({
            "Month/Quarter": tokens[0],
            "2019": tokens[1],
            "2021": tokens[2],
            "2022": tokens[3],
            "2023": tokens[4],
            "2024": tokens[5],
            "2025*": tokens[6],
            "Change (%) 22/21": tokens[7],
            "Change (%) 23/22": tokens[8],
            "Change (%) 24/23": tokens[9],
            "Change (%) 25*/19": tokens[10],
            "Change (%) 2025*/24": tokens[11],
        })
        # Stop after the "Total" summary row
        if tokens[0] == "Total":
            break

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tables 3–5 – Annual Inbound / Outbound / Internal Tourism (page 3)
# Tables 7–9 – December Inbound / Outbound / Internal Tourism (page 6)
# ---------------------------------------------------------------------------

def _extract_inbound(text: str) -> pd.DataFrame:
    """
    Parse the 'International tourist arrivals' section from a page 3 / page 6 text block.
    Columns: Inbound tourism, 2019, 2023, 2024, 2025*, Share (%) 2025*, Change (%) 24/23,
             Change (%) 25*/19, Change (%) 2025*/24
    """
    # Row labels in order
    row_labels = [
        "Air",
        "Phnom Penh Int'l Airport (PNH)",
        "Siem Reap Angkor Int'l Airport (SAI)",
        "Sihanouk Int'l Airport (KOS)",
        "Land and Waterways",
        "Land",
        "Waterways",
        "Total",
    ]
    # Build a lookup: first distinctive token(s) → full label
    row_keys = {
        "Air": "Air",
        "Phnom": "Phnom Penh Int'l Airport (PNH)",
        "Siem": "Siem Reap Angkor Int'l Airport (SAI)",
        "Sihanouk": "Sihanouk Int'l Airport (KOS)",
        "Land": None,   # disambiguate below
        "Waterways": "Waterways",
        "Total": "Total",
    }

    rows_dict: dict[str, list] = {}
    for line in _lines(text):
        tokens = line.split()
        if not tokens:
            continue
        first = tokens[0]
        label = None
        if first == "Air":
            label = "Air"
        elif first == "Phnom":
            label = "Phnom Penh Int'l Airport (PNH)"
        elif first == "Siem":
            label = "Siem Reap Angkor Int'l Airport (SAI)"
        elif first == "Sihanouk":
            label = "Sihanouk Int'l Airport (KOS)"
        elif first == "Land" and len(tokens) > 1 and tokens[1] == "and":
            label = "Land and Waterways"
        elif first == "Land" and (len(tokens) < 2 or tokens[1] != "and"):
            label = "Land"
        elif first == "Waterways":
            label = "Waterways"
        elif first == "Total":
            label = "Total"
        else:
            continue

        # Collect numeric tokens from the line
        nums = [t for t in tokens[1:] if _numeric_token(t)]
        if len(nums) >= 8:
            rows_dict[label] = nums[:8]

    columns = [
        "Inbound tourism",
        "2019", "2023", "2024", "2025*",
        "Share (%) 2025*",
        "Change (%) 24/23", "Change (%) 25*/19", "Change (%) 2025*/24",
    ]
    records = []
    for lbl in row_labels:
        if lbl in rows_dict:
            records.append([lbl] + rows_dict[lbl])
    return pd.DataFrame(records, columns=columns)


def _extract_outbound(text: str) -> pd.DataFrame:
    """
    Parse the 'Outbound tourism' section.
    Columns: Outbound tourism, 2019, 2023, 2024, 2025*, Change (%) 24/23, 25*/19, 2025*/24
    """
    row_labels = [
        "Cambodia Outbound Tourists",
        "International Tourists Departure",
    ]
    rows_dict: dict[str, list] = {}
    for line in _lines(text):
        tokens = line.split()
        if not tokens:
            continue
        if tokens[0] == "Cambodia" and len(tokens) > 1 and tokens[1] == "Outbound":
            label = "Cambodia Outbound Tourists"
        elif tokens[0] == "International" and len(tokens) > 1 and tokens[1] == "Tourists":
            label = "International Tourists Departure"
        else:
            continue
        nums = [t for t in tokens if _numeric_token(t)]
        if len(nums) >= 7:
            rows_dict[label] = nums[:7]

    columns = [
        "Outbound tourism",
        "2019", "2023", "2024", "2025*",
        "Change (%) 24/23", "Change (%) 25*/19", "Change (%) 2025*/24",
    ]
    records = []
    for lbl in row_labels:
        if lbl in rows_dict:
            records.append([lbl] + rows_dict[lbl])
    return pd.DataFrame(records, columns=columns)


def _extract_internal(text: str) -> pd.DataFrame:
    """
    Parse the 'Domestic and foreign visitors' arrivals to regions' section.
    Columns: Region, KHM prev, FOR prev, KHM curr, FOR curr, Change(%) KHM, Change(%) FOR
    """
    region_labels = [
        "Phnom Penh",
        "Siem Reap Angkor",
        "Coastal Zone",
        "Preah Sihanouk",
        "Eco-tourism Zone",
        "Others",
        "Total",
    ]
    rows_dict: dict[str, list] = {}
    for line in _lines(text):
        tokens = line.split()
        if not tokens:
            continue
        label = None
        if tokens[0] == "Phnom" and len(tokens) > 1 and tokens[1] == "Penh":
            label = "Phnom Penh"
        elif tokens[0] == "Siem" and len(tokens) > 1 and tokens[1] == "Reap":
            label = "Siem Reap Angkor"
        elif tokens[0] == "Coastal":
            label = "Coastal Zone"
        elif tokens[0] == "Preah":
            label = "Preah Sihanouk"
        elif tokens[0] == "Eco-tourism":
            label = "Eco-tourism Zone"
        elif tokens[0] == "Others":
            label = "Others"
        elif tokens[0] == "Total":
            label = "Total"
        else:
            continue
        nums = [t for t in tokens if _numeric_token(t)]
        if len(nums) >= 6:
            rows_dict[label] = nums[:6]

    columns = [
        "Region",
        "Prev Year KHM (Cambodian visitors)",
        "Prev Year FOR (Foreign visitors)",
        "Curr Year KHM (Cambodian visitors)",
        "Curr Year FOR (Foreign visitors)",
        "Change (%) KHM",
        "Change (%) FOR",
    ]
    records = []
    for lbl in region_labels:
        if lbl in rows_dict:
            records.append([lbl] + rows_dict[lbl])
    return pd.DataFrame(records, columns=columns)


def extract_annual_inbound(pdf) -> pd.DataFrame:
    return _extract_inbound(_page_text(pdf, 2))


def extract_annual_outbound(pdf) -> pd.DataFrame:
    return _extract_outbound(_page_text(pdf, 2))


def extract_annual_internal(pdf) -> pd.DataFrame:
    return _extract_internal(_page_text(pdf, 2))


def extract_dec_inbound(pdf) -> pd.DataFrame:
    return _extract_inbound(_page_text(pdf, 5))


def extract_dec_outbound(pdf) -> pd.DataFrame:
    return _extract_outbound(_page_text(pdf, 5))


def extract_dec_internal(pdf) -> pd.DataFrame:
    return _extract_internal(_page_text(pdf, 5))


# ---------------------------------------------------------------------------
# Tables 6 & 10 – Arrivals by Country (annual: pages 4–5; Dec: pages 7–8)
# ---------------------------------------------------------------------------

_COUNTRY_TABLE_HEADERS = {
    "Country of Residence",
    "2024",
    "Grand Total",
    "Regions",
    "Holiday",
    "Business",
    "Others",
    "Female",
    "(by all means",
    "International Tourist",
}

_COUNTRY_TABLE_COL_NAMES = [
    "Country / Region",
    "2024 Total",
    "2025* Holiday",
    "2025* Business",
    "2025* Others",
    "2025* Total",
    "2025* Female",
    "Share (%) 2025*",
    "Change (%) 2025*/24",
]


def _extract_country_table(pages_text: list[str]) -> pd.DataFrame:
    """
    Parse the multi-page arrivals-by-country table.
    pages_text: list of text strings for the pages containing the table.
    """
    rows = []
    pending_name = None
    pending_nums: list[str] = []

    skip_prefixes = {
        "International Tourist",
        "(by all",
        "Regions",
        "Country of",
        "Holiday",
        "Source:",
    }

    def flush_pending():
        nonlocal pending_name, pending_nums
        if pending_name and len(pending_nums) == 8:
            rows.append([pending_name] + pending_nums)
        pending_name = None
        pending_nums = []

    for text in pages_text:
        for line in _lines(text):
            if not line:
                continue
            # Skip header / footer lines
            first_word = line.split()[0] if line.split() else ""
            if any(line.startswith(p) for p in skip_prefixes):
                continue
            # Skip the repeated "Grand Total" header line (appears at top of continuation pages)
            # but we DO want the Grand Total data row itself
            tokens = line.split()

            result = _split_country_row(line)
            if result:
                flush_pending()
                name, nums = result
                rows.append([name] + nums)
            else:
                # Could be a continuation: either name with too few nums, or nums continuation
                all_numeric = all(_numeric_token(t) for t in tokens)
                if all_numeric and pending_name:
                    # These are continuation numbers for the previous name
                    pending_nums.extend(tokens)
                    if len(pending_nums) >= 8:
                        rows.append([pending_name] + pending_nums[:8])
                        pending_name = None
                        pending_nums = []
                elif not all_numeric and tokens:
                    # Could be a partial row (name + some numbers, wraps to next line)
                    nums_in_line = [t for t in tokens if _numeric_token(t)]
                    non_nums = [t for t in tokens if not _numeric_token(t)]
                    if non_nums and len(nums_in_line) < 8:
                        flush_pending()
                        pending_name = " ".join(non_nums + nums_in_line) if not nums_in_line else " ".join(non_nums)
                        pending_nums = nums_in_line
                    else:
                        flush_pending()

    flush_pending()

    # Deduplicate Grand Total rows (repeated on continuation pages)
    seen_grand_total = False
    unique_rows = []
    for row in rows:
        if row[0] == "Grand Total":
            if seen_grand_total:
                continue
            seen_grand_total = True
        unique_rows.append(row)

    return pd.DataFrame(unique_rows, columns=_COUNTRY_TABLE_COL_NAMES)


def extract_arrivals_by_country_annual(pdf) -> pd.DataFrame:
    texts = [_page_text(pdf, 3), _page_text(pdf, 4)]
    return _extract_country_table(texts)


def extract_arrivals_by_country_dec(pdf) -> pd.DataFrame:
    texts = [_page_text(pdf, 6), _page_text(pdf, 7)]
    return _extract_country_table(texts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with pdfplumber.open(PDF_PATH) as pdf:
        tables = [
            ("01_tourism_highlights.csv",
             extract_tourism_highlights(pdf)),
            ("02_international_tourist_arrivals.csv",
             extract_international_arrivals(pdf)),
            ("03_annual_inbound_tourism.csv",
             extract_annual_inbound(pdf)),
            ("04_annual_outbound_tourism.csv",
             extract_annual_outbound(pdf)),
            ("05_annual_internal_tourism.csv",
             extract_annual_internal(pdf)),
            ("06_arrivals_by_country_annual.csv",
             extract_arrivals_by_country_annual(pdf)),
            ("07_dec_inbound_tourism.csv",
             extract_dec_inbound(pdf)),
            ("08_dec_outbound_tourism.csv",
             extract_dec_outbound(pdf)),
            ("09_dec_internal_tourism.csv",
             extract_dec_internal(pdf)),
            ("10_arrivals_by_country_december.csv",
             extract_arrivals_by_country_dec(pdf)),
        ]

    for filename, df in tables:
        _write_csv(df, filename)


if __name__ == "__main__":
    main()
