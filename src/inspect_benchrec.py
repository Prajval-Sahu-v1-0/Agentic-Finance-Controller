"""Read-only structural inspection for a downloaded BenchRec dataset.

Run from the project root::

    python -m src.inspect_benchrec

Use ``--raw-dir`` when the dataset was extracted somewhere else.  The script
never transforms, moves, or writes any dataset file.
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

try:  # CSV inspection intentionally works without optional dataframe packages.
    import pandas as pd
except ModuleNotFoundError:  # pragma: no cover - depends on local environment
    pd = None


PROJECT_ROOT = Path(__file__).parent.parent
EXPECTED_RAW_DIR = PROJECT_ROOT / "data" / "external" / "benchrec" / "raw"
FALLBACK_RAW_DIR = PROJECT_ROOT / "data" / "external" / "raw"
SAMPLE_ROWS = 5

_FORMAT_NAMES = {
    ".csv": "CSV",
    ".tsv": "TSV",
    ".json": "JSON",
    ".jsonl": "JSON Lines",
    ".ndjson": "JSON Lines",
    ".parquet": "Parquet",
    ".pq": "Parquet",
    ".xlsx": "Excel workbook",
    ".xls": "Excel workbook",
    ".sqlite": "SQLite database",
    ".db": "SQLite database",
}

_ID_HINTS = ("transaction_id", "transactionid", "txn_id", "trans_id", "record_id", "entry_id", "id")
_AMOUNT_HINTS = ("amount", "amt", "value", "balance", "debit", "credit")
_TIME_HINTS = ("timestamp", "datetime", "date_time", "transaction_date", "date", "time")
_REFERENCE_HINTS = ("reference", "ref", "description", "narration", "memo", "details", "remark")


@dataclass
class TableInspection:
    name: str
    rows: int
    columns: list[str]
    dtypes: dict[str, str]
    sample: list[dict[str, Any]]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read-only BenchRec dataset inspector")
    parser.add_argument("--raw-dir", type=Path, default=None, help="Dataset directory to inspect")
    parser.add_argument("--sample-rows", type=int, default=SAMPLE_ROWS, metavar="N")
    return parser


def discover_raw_dir(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    if EXPECTED_RAW_DIR.exists():
        return EXPECTED_RAW_DIR
    return FALLBACK_RAW_DIR


def format_name(path: Path) -> str:
    return _FORMAT_NAMES.get(path.suffix.lower(), f"Unknown ({path.suffix or 'no extension'})")


def _require_pandas() -> Any:
    if pd is None:
        raise RuntimeError(
            "This format requires pandas (and possibly an engine such as pyarrow or openpyxl). "
            "CSV and TSV inspection do not require pandas."
        )
    return pd


def _read_csv(path: Path, sample_rows: int) -> list[TableInspection]:
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    with path.open("r", encoding="utf-8-sig", newline="", errors="replace") as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        columns = next(reader, [])
        sample_values = [row for _, row in zip(range(sample_rows), reader)]
        row_count = sum(1 for _ in reader) + len(sample_values)
    sample = [dict(zip(columns, row)) for row in sample_values]
    return [TableInspection(path.name, row_count, columns, {column: "string (raw CSV)" for column in columns}, sample)]


def _read_json(path: Path, sample_rows: int) -> list[TableInspection]:
    pandas = _require_pandas()
    try:
        frame = pandas.read_json(path, lines=path.suffix.lower() in {".jsonl", ".ndjson"})
    except ValueError:
        # Some JSON exports are a single object containing named table arrays.
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            tables = []
            for name, value in payload.items():
                if isinstance(value, list):
                    tables.append(_frame_inspection(pandas.DataFrame(value), f"{path.name}:{name}", sample_rows))
            if tables:
                return tables
        raise
    return [_frame_inspection(frame, path.name, sample_rows)]


def _read_excel(path: Path, sample_rows: int) -> list[TableInspection]:
    pandas = _require_pandas()
    workbook = pandas.ExcelFile(path)
    return [
        _frame_inspection(pandas.read_excel(path, sheet_name=sheet), f"{path.name}:{sheet}", sample_rows)
        for sheet in workbook.sheet_names
    ]


def _read_sqlite(path: Path, sample_rows: int) -> list[TableInspection]:
    pandas = _require_pandas()
    connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    try:
        tables = pandas.read_sql_query(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name", connection
        )["name"].tolist()
        inspections = []
        for table in tables:
            frame = pandas.read_sql_query(f'SELECT * FROM "{table}"', connection)
            inspections.append(_frame_inspection(frame, f"{path.name}:{table}", sample_rows))
        return inspections
    finally:
        connection.close()


def _frame_inspection(frame: pd.DataFrame, name: str, sample_rows: int) -> TableInspection:
    sample = json.loads(frame.head(sample_rows).to_json(orient="records", date_format="iso"))
    return TableInspection(
        name=name,
        rows=len(frame),
        columns=[str(column) for column in frame.columns],
        dtypes={str(column): str(dtype) for column, dtype in frame.dtypes.items()},
        sample=sample,
    )


def inspect_file(path: Path, sample_rows: int) -> list[TableInspection]:
    readers: dict[str, Callable[[Path, int], list[TableInspection]]] = {
        ".csv": _read_csv,
        ".tsv": _read_csv,
        ".json": _read_json,
        ".jsonl": _read_json,
        ".ndjson": _read_json,
        ".parquet": lambda value, count: [_frame_inspection(_require_pandas().read_parquet(value), value.name, count)],
        ".pq": lambda value, count: [_frame_inspection(_require_pandas().read_parquet(value), value.name, count)],
        ".xlsx": _read_excel,
        ".xls": _read_excel,
        ".sqlite": _read_sqlite,
        ".db": _read_sqlite,
    }
    reader = readers.get(path.suffix.lower())
    if reader is None:
        return []
    return reader(path, sample_rows)


def _matching_columns(columns: list[str], hints: tuple[str, ...]) -> list[str]:
    matches = []
    for column in columns:
        normalised = column.lower().replace(" ", "_").replace("-", "_")
        if any(hint == normalised or hint in normalised for hint in hints):
            matches.append(column)
    return matches


def _role_for(path: Path, table: TableInspection) -> str:
    name = f"{path.name} {' '.join(table.columns)}".lower()
    if any(token in name for token in ("solution", "label", "ground_truth", "groundtruth", "match")):
        return "Ground-truth / submission candidate"
    if any(token in name for token in ("type_a", "type a", "ledger", "internal")):
        return "Type A candidate (internal ledger)"
    if any(token in name for token in ("type_b", "type b", "statement", "external", "bank")):
        return "Type B candidate (external statement)"
    if "train" in name:
        return "Training data candidate (inspect source/type column)"
    if "eval" in name or "test" in name:
        return "Evaluation data candidate (inspect source/type column)"
    return "Role not determined from file/schema"


def _label_shape(columns: list[str], sample: list[dict[str, Any]]) -> str:
    lowered = [column.lower() for column in columns]
    id_columns = _matching_columns(columns, _ID_HINTS)
    has_pair_columns = len(id_columns) >= 2 or any("type_a" in value or "type_b" in value for value in lowered)
    has_group_values = any(
        isinstance(value, list)
        for row in sample for value in row.values()
    )
    if has_group_values:
        return "Possible grouped labels: at least one sample value is a list."
    if has_pair_columns:
        return "Possible pair labels: schema contains two or more ID/source columns."
    return "Label shape not conclusive from schema/sample."


def print_report(raw_dir: Path, files: list[Path], sample_rows: int) -> None:
    print("BENCHREC READ-ONLY INSPECTION")
    print("=" * 72)
    print(f"Raw directory: {raw_dir}")
    print("No dataset files are modified by this script.\n")

    role_candidates: list[tuple[str, str]] = []
    label_candidates: list[tuple[str, str]] = []
    for path in files:
        print(f"FILE: {path.relative_to(raw_dir)}")
        print(f"Format: {format_name(path)} | Size: {path.stat().st_size:,} bytes")
        try:
            tables = inspect_file(path, sample_rows)
        except Exception as exc:
            print(f"Inspection error: {type(exc).__name__}: {exc}\n")
            continue
        if not tables:
            print("No built-in reader for this format.\n")
            continue
        for table in tables:
            role = _role_for(path, table)
            print(f"Table: {table.name}")
            print(f"Rows: {table.rows:,}")
            print("Columns / schema:")
            for column in table.columns:
                print(f"  - {column}: {table.dtypes.get(column, 'unknown')}")
            print("Likely semantic fields:")
            print(f"  transaction ID: {_matching_columns(table.columns, _ID_HINTS) or 'not identified'}")
            print(f"  amount: {_matching_columns(table.columns, _AMOUNT_HINTS) or 'not identified'}")
            print(f"  timestamp: {_matching_columns(table.columns, _TIME_HINTS) or 'not identified'}")
            print(f"  reference/description: {_matching_columns(table.columns, _REFERENCE_HINTS) or 'not identified'}")
            print(f"BenchRec role: {role}")
            print(f"Sample ({min(sample_rows, len(table.sample))} rows):")
            for row in table.sample:
                print(f"  {json.dumps(row, ensure_ascii=False, default=str)}")
            print()
            role_candidates.append((table.name, role))
            if "Ground-truth" in role:
                label_candidates.append((table.name, _label_shape(table.columns, table.sample)))

    print("SUMMARY")
    print("- Type A (internal ledger) / Type B (external statement):")
    for name, role in role_candidates:
        print(f"  - {name}: {role}")
    print("- Ground-truth match-label candidates:")
    if label_candidates:
        for name, shape in label_candidates:
            print(f"  - {name}: {shape}")
    else:
        print("  - None identified by filename/schema heuristics.")


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    raw_dir = discover_raw_dir(args.raw_dir)
    if not raw_dir.is_dir():
        print(f"Raw directory not found: {raw_dir}")
        print(f"Expected default: {EXPECTED_RAW_DIR}")
        return
    files = sorted(path for path in raw_dir.rglob("*") if path.is_file())
    if not files:
        print(f"No files found in: {raw_dir}")
        return
    print_report(raw_dir, files, args.sample_rows)


if __name__ == "__main__":
    main()
