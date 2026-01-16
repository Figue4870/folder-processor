import os
import re
import json
import csv
import argparse
import logging
import warnings
from datetime import datetime
from datetime import datetime, timezone

import pandas as pd


# ----------------------------
# Helpers: header normalization
# ----------------------------
def normalize_header(name):
    s = str(name).strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^\w_]", "", s)  # keep letters, digits, underscore
    s = re.sub(r"_+", "_", s)
    s = s.strip("_")
    if s == "":
        s = "col"
    return s


def dedupe_headers(headers):
    seen = {}
    out = []
    for h in headers:
        base = h
        if base not in seen:
            seen[base] = 0
            out.append(base)
        else:
            seen[base] += 1
            out.append(f"{base}_{seen[base]}")
    return out


# ----------------------------
# Helpers: delimiter + encoding
# ----------------------------
def detect_delimiter(sample_text):
    # Try csv.Sniffer first
    try:
        dialect = csv.Sniffer().sniff(sample_text, delimiters=[",", ";", "\t", "|"])
        if dialect.delimiter in [",", ";", "\t", "|"]:
            return dialect.delimiter
    except Exception:
        pass

    # Fallback: count delimiters on first non-empty line
    lines = [ln for ln in sample_text.splitlines() if ln.strip()][:5]
    if not lines:
        return ","
    line = lines[0]
    counts = {d: line.count(d) for d in [",", ";", "\t", "|"]}
    best = max(counts, key=counts.get)
    return best if counts[best] > 0 else ","


def read_text_sample(path, max_bytes=8192):
    # Try utf-8, fallback latin-1
    with open(path, "rb") as f:
        raw = f.read(max_bytes)
    for enc in ["utf-8", "utf-8-sig", "latin-1"]:
        try:
            return raw.decode(enc), enc
        except Exception:
            continue
    # Last resort: decode with errors
    return raw.decode("utf-8", errors="replace"), "utf-8(errors=replace)"


# ----------------------------
# Helpers: value normalization
# ----------------------------
_CURRENCY_RE = re.compile(r"[^\d,\.\-\(\)]+")

def normalize_number_str(x):
    """
    Normalize numeric strings:
    - removes currency and spaces
    - handles thousands separators and decimal comma
    - returns a string with '.' as decimal separator
    """
    if x is None:
        return None
    s = str(x).strip()
    if s == "":
        return ""

    # parentheses negative: (123) -> -123
    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg = True
        s = s[1:-1].strip()

    # remove currency symbols/letters
    s = _CURRENCY_RE.sub("", s)
    s = s.replace(" ", "")

    # If it has both '.' and ',', decide decimal by last occurrence
    if "." in s and "," in s:
        if s.rfind(",") > s.rfind("."):
            # comma decimal: 1.234,56 -> 1234.56
            s = s.replace(".", "")
            s = s.replace(",", ".")
        else:
            # dot decimal: 1,234.56 -> 1234.56
            s = s.replace(",", "")
    else:
        # Only comma: could be decimal comma OR thousands
        if "," in s and "." not in s:
            # heuristic: if last group has 1-2 digits, treat as decimal comma
            parts = s.split(",")
            if len(parts[-1]) in [1, 2]:
                s = s.replace(",", ".")
            else:
                s = s.replace(",", "")

        # Only dot: could be decimal dot OR thousands
        if "." in s and "," not in s:
            parts = s.split(".")
            if len(parts[-1]) in [1, 2]:
                # decimal dot, keep
                pass
            else:
                # likely thousands, remove dots
                s = s.replace(".", "")

    s = s.strip()
    if s == "" or s == "-" or s == ".":
        return ""

    # Validate numeric-ish
    if not re.fullmatch(r"-?\d+(\.\d+)?", s):
        return str(x).strip()  # return original trimmed if not cleanly numeric

    if neg and not s.startswith("-"):
        s = "-" + s
    return s


def looks_numeric_series(series):
    # sample up to 50 non-empty values
    vals = [v for v in series.astype(str).tolist() if v.strip() != ""]
    if not vals:
        return False
    sample = vals[:50]
    ok = 0
    for v in sample:
        nv = normalize_number_str(v)
        if re.fullmatch(r"-?\d+(\.\d+)?", nv):
            ok += 1
    return ok / max(1, len(sample)) >= 0.85


def should_parse_date(colname):
    c = colname.lower()
    # Spanish + English common hints
    return any(k in c for k in ["date", "fecha", "time", "timestamp", "created", "updated", "birth"])


def normalize_date_series(series):
    s = series.astype(str).map(lambda x: x.strip())
    empty_mask = s.eq("")

    candidate = s.where(~empty_mask, pd.NA)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)

        dt_dayfirst = pd.to_datetime(candidate, errors="coerce", dayfirst=True)
        dt_monthfirst = pd.to_datetime(candidate, errors="coerce", dayfirst=False)

    valid_dayfirst = int(dt_dayfirst.notna().sum())
    valid_monthfirst = int(dt_monthfirst.notna().sum())

    dt = dt_dayfirst if valid_dayfirst >= valid_monthfirst else dt_monthfirst

    out = dt.dt.strftime("%Y-%m-%d")
    out = out.where(~dt.isna(), "")
    out = out.where(~empty_mask, "")
    return out.astype(str)




# ----------------------------
# Core processing
# ----------------------------
def process_file(path, out_dir, dry_run=False):
    base = os.path.basename(path)
    name_no_ext = os.path.splitext(base)[0]
    cleaned_path = os.path.join(out_dir, f"cleaned_{name_no_ext}.csv")

    sample_text, enc = read_text_sample(path)
    delim = detect_delimiter(sample_text)

    result = {
        "file": base,
        "path": path,
        "encoding_guess": enc,
        "delimiter": repr(delim),
        "status": "pending",
        "error": None,
        "rows": 0,
        "cols": 0,
        "columns_original": [],
        "columns_normalized": [],
        "nulls_by_column": {},
        "normalized_numeric_columns": [],
        "normalized_date_columns": [],
        "output_cleaned": cleaned_path,
    }

    if dry_run:
        result["status"] = "dry-run"
        return result

    try:
        # Read everything as string to avoid surprises; handle empty as empty
        df = pd.read_csv(
            path,
            sep=delim,
            dtype=str,
            encoding=None,  # let pandas try
            keep_default_na=False,
            na_filter=False,
            engine="python"
        )

        result["columns_original"] = [str(c) for c in df.columns]

        # Normalize headers
        norm_headers = [normalize_header(c) for c in df.columns]
        norm_headers = dedupe_headers(norm_headers)
        df.columns = norm_headers
        result["columns_normalized"] = norm_headers

        # Trim all cells (strings)
        for c in df.columns:
            df[c] = df[c].astype(str).map(lambda x: x.strip())

        # Normalize dates and numbers heuristically
        date_cols = [c for c in df.columns if should_parse_date(c)]
        for c in date_cols:
            df[c] = normalize_date_series(df[c])
        result["normalized_date_columns"] = date_cols

        numeric_cols = []
        for c in df.columns:
            if c in date_cols:
                continue
            if looks_numeric_series(df[c]):
                numeric_cols.append(c)
                df[c] = df[c].map(normalize_number_str)
        result["normalized_numeric_columns"] = numeric_cols

        # Null/empty counts
        result["rows"] = int(df.shape[0])
        result["cols"] = int(df.shape[1])
        nulls = {}
        for c in df.columns:
            nulls[c] = int((df[c].astype(str).str.strip() == "").sum())
        result["nulls_by_column"] = nulls

        # Write cleaned
        os.makedirs(out_dir, exist_ok=True)
        df.to_csv(cleaned_path, index=False)

        result["status"] = "ok"
        return result

    except Exception as e:
        result["status"] = "failed"
        result["error"] = str(e)
        return result


def build_report_md(run_summary):
    lines = []
    lines.append(f"# Folder Processor Report\n")
    lines.append(f"- Run timestamp: **{run_summary['run_timestamp']}**")
    lines.append(f"- Input folder: `{run_summary['input_folder']}`")
    lines.append(f"- Output folder: `{run_summary['output_folder']}`")
    lines.append(f"- Dry-run: **{run_summary['dry_run']}**")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("| File | Status | Rows | Cols | Empty cells (total) | Output | Error |")
    lines.append("|---|---:|---:|---:|---:|---|---|")

    for f in run_summary["files"]:
        empty_total = 0
        if isinstance(f.get("nulls_by_column"), dict):
            empty_total = sum(int(v) for v in f["nulls_by_column"].values())

        lines.append(
            f"| {f['file']} | {f['status']} | {f.get('rows', 0)} | {f.get('cols', 0)} | {empty_total} | "
            f"`{f.get('output_cleaned', '')}` | {'' if not f.get('error') else f.get('error')} |"
        )

    lines.append("")
    lines.append("## Notes")
    lines.append("- Headers are normalized (trim/lowercase/spaces→underscore).")
    lines.append("- Date-like columns (by name) are converted to ISO `YYYY-MM-DD` when possible.")
    lines.append("- Numeric-like columns are normalized (currency removed, decimal comma handled).")
    lines.append("- If a file fails, the run continues with the rest.")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Folder Processor: clean CSVs + summary.json + report.md")
    parser.add_argument("--input", required=True, help="Input folder containing CSV files")
    parser.add_argument("--output", default="output", help="Output folder (default: output)")
    parser.add_argument("--dry-run", action="store_true", help="Only show what would happen (no outputs written)")
    args = parser.parse_args()

    input_dir = args.input
    out_dir = args.output
    dry_run = bool(args.dry_run)

    os.makedirs(out_dir, exist_ok=True)

    # Logging
    log_path = os.path.join(out_dir, "process.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()]
    )
    logger = logging.getLogger("folder_processor")

    if not os.path.isdir(input_dir):
        raise SystemExit(f"Input folder not found: {input_dir}")

    csv_files = []
    for fn in os.listdir(input_dir):
        if fn.lower().endswith(".csv"):
            csv_files.append(os.path.join(input_dir, fn))
    csv_files.sort()

    logger.info("Found %d CSV file(s) in %s", len(csv_files), input_dir)

    files_summary = []
    for path in csv_files:
        base = os.path.basename(path)
        logger.info("Processing: %s", base)
        info = process_file(path, out_dir, dry_run=dry_run)

        if info["status"] == "ok":
            logger.info("OK: %s -> %s", base, info["output_cleaned"])
            if info["normalized_date_columns"]:
                logger.info("Date columns: %s", ", ".join(info["normalized_date_columns"]))
            if info["normalized_numeric_columns"]:
                logger.info("Numeric columns: %s", ", ".join(info["normalized_numeric_columns"]))
        elif info["status"] == "dry-run":
            logger.info("DRY-RUN: would write %s", info["output_cleaned"])
        else:
            logger.error("FAILED: %s | %s", base, info.get("error"))

        files_summary.append(info)

    run_summary = {
        "run_timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "input_folder": os.path.abspath(input_dir),
        "output_folder": os.path.abspath(out_dir),
        "dry_run": dry_run,
        "files_total": len(files_summary),
        "files_ok": sum(1 for f in files_summary if f["status"] == "ok"),
        "files_failed": sum(1 for f in files_summary if f["status"] == "failed"),
        "files": files_summary,
    }

    # Always write summary/report unless dry-run (you can change this if you want)
    if not dry_run:
        summary_path = os.path.join(out_dir, "summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(run_summary, f, ensure_ascii=False, indent=2)
        logger.info("Wrote: %s", summary_path)

        report_path = os.path.join(out_dir, "report.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(build_report_md(run_summary))
        logger.info("Wrote: %s", report_path)
    else:
        logger.info("Dry-run enabled: no files written (besides log).")

    logger.info("Done.")


if __name__ == "__main__":
    main()
