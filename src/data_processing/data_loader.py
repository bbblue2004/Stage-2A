"""Read the field CSV and build its compact working extract."""

import csv
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FULL_CSV_PATH = ROOT / "data" / "raw" / "radio_sites.csv"
COMPACT_CSV_PATH = ROOT / "data" / "processed" / "radio_sites_10x7.csv"
CSV_PATH = COMPACT_CSV_PATH if COMPACT_CSV_PATH.is_file() else FULL_CSV_PATH
OUTPUT_DIR = ROOT / "figures" / "diagnostics"
DEFAULT_ANTENNA_ID = "00000001U6"

Record = tuple[datetime, str, float, float]


def make_output_path(filename: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR / filename


def _columns(fieldnames: list[str] | None) -> tuple[str, str, str, str]:
    names = fieldnames or []

    def find(*tokens: str) -> str:
        match = next(
            (name for token in tokens for name in names if token in name.lower()),
            None,
        )
        if not match:
            raise ValueError(f"Missing CSV column among {tokens}")
        return match

    return (
        find("heure", "date"),
        find("nidt"),
        find("dl_volume", "pdcp", "gbytes"),
        find("power", "consumption"),
    )


def _record(row: dict, columns: tuple[str, str, str, str]) -> Record | None:
    date_col, id_col, traffic_col, power_col = columns
    try:
        return (
            datetime.fromisoformat(row[date_col].strip()),
            row[id_col].strip(),
            float(row[traffic_col].replace(",", ".")),
            float(row[power_col].replace(",", ".")),
        )
    except (KeyError, TypeError, ValueError):
        return None


def iter_records(csv_path: Path = CSV_PATH):
    """Yield valid field records while reading the CSV only once."""
    with open(csv_path, newline="", encoding="utf-8", errors="replace") as file:
        reader = csv.DictReader(file, delimiter=";")
        columns = _columns(reader.fieldnames)
        for raw_row in reader:
            record = _record(raw_row, columns)
            if record is not None:
                yield record


def first_antenna_ids(count: int, csv_path: Path = CSV_PATH) -> list[str]:
    if count <= 0:
        raise ValueError("count must be positive")

    ids: list[str] = []
    with open(csv_path, newline="", encoding="utf-8", errors="replace") as file:
        reader = csv.DictReader(file, delimiter=";")
        _, id_col, _, _ = _columns(reader.fieldnames)
        for row in reader:
            antenna_id = (row.get(id_col) or "").strip()
            if antenna_id and antenna_id not in ids:
                ids.append(antenna_id)
            if len(ids) == count:
                return ids
    raise ValueError(f"Expected {count} antennas, found {len(ids)}")


def first_antenna_id(csv_path: Path = CSV_PATH) -> str:
    return first_antenna_ids(1, csv_path)[0]


def extract_antenna_time_series(
    antenna_id: str,
    csv_path: Path = CSV_PATH,
) -> list[tuple[datetime, float, float]]:
    rows: list[tuple[datetime, float, float]] = []
    with open(csv_path, newline="", encoding="utf-8", errors="replace") as file:
        reader = csv.DictReader(file, delimiter=";")
        columns = _columns(reader.fieldnames)
        for raw_row in reader:
            record = _record(raw_row, columns)
            if record and record[1] == antenna_id:
                rows.append((record[0], record[2], record[3]))
    if not rows:
        raise ValueError(f"No data for antenna {antenna_id}")
    return sorted(rows)


def create_compact_csv(
    source_csv: Path = FULL_CSV_PATH,
    output_csv: Path = COMPACT_CSV_PATH,
    num_antennas: int = 10,
    num_days: int = 7,
) -> Path:
    """Keep one row per hour for the first antennas and their first days."""
    antenna_ids = first_antenna_ids(num_antennas, source_csv)
    samples: dict[str, dict[tuple, Record]] = {i: {} for i in antenna_ids}

    with open(source_csv, newline="", encoding="utf-8", errors="replace") as file:
        reader = csv.DictReader(file, delimiter=";")
        columns = _columns(reader.fieldnames)
        for raw_row in reader:
            record = _record(raw_row, columns)
            if record and record[1] in samples:
                samples[record[1]].setdefault((record[0].date(), record[0].hour), record)

    output_rows: list[Record] = []
    expected = 24 * num_days
    for antenna_id, antenna_samples in samples.items():
        days = sorted({day for day, _ in antenna_samples})[:num_days]
        rows = sorted(
            record
            for (day, _), record in antenna_samples.items()
            if day in days
        )
        if len(rows) != expected:
            raise ValueError(f"{antenna_id}: expected {expected} rows, found {len(rows)}")
        output_rows.extend(rows)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, delimiter=";")
        writer.writerow(
            (
                "HEURE(PSDATE)",
                "SYS.NIDT",
                "DL_VOLUME_PDCP_GBYTES",
                "AVERAGE_POWER_CONSUMPTION_(W)",
            )
        )
        writer.writerows(
            (date.isoformat(sep=" "), antenna_id, traffic, power)
            for date, antenna_id, traffic, power in output_rows
        )
    return output_csv
