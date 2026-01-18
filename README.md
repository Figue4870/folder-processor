# Folder Processor (CSV Cleaner)
“Drag-and-drop a folder of CSVs → get cleaned files + summary + report.”

A small “folder processor” that ingests all CSV files in a folder, cleans them, and produces:
- `output/cleaned_*.csv`
- `output/summary.json`
- `output/report.md`
- `output/process.log`

## Features
- Works with arbitrary CSV filenames (no fixed schema required).
- Auto-detects delimiter: `, ; \t |`
- Normalizes headers (trim + lowercase + spaces to `_`)
- Trims text cells
- Normalizes date-like columns to ISO `YYYY-MM-DD` (best-effort)
- Normalizes numeric-like columns (currency symbols, thousands separators, decimal commas)
- Clear logs and per-file error isolation
- `--dry-run` mode (shows what would be done without writing outputs)

## Setup
```bash
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt

```


## Run

### Dry-run:

python folder_processor.py --input sample_data --output output --dry-run

### Real run:

python folder_processor.py --input sample_data --output output

## Notes

This repo includes sample_data/ with messy CSV examples to reproduce typical real-world issues.

## Example output
![Folder Processor Report](assets/folder_processor_report.png)


