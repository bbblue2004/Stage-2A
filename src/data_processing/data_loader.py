"""CSV loading for radio_sites.csv."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = PROJECT_ROOT / "data" / "raw" / "radio_sites.csv"
OUTPUT_DIR = PROJECT_ROOT / "figures" / "data_figures"
DEFAULT_ANTENNA_ID = "00000001U6"


def make_output_path(filename: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR / filename


def _find_column(fieldnames: list[str] | None, keywords: list[str]) -> str | None:
    for keyword in keywords:
        for name in fieldnames or []:
            if keyword.lower() in name.lower():
                return name
    return None


def _parse_datetime(value: str) -> datetime:
    value = value.strip()
    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d %H",
        "%Y/%m/%d %H:%M:%S",
        "%d/%m/%Y %H:%M:%S",
        "%Y%m%d%H%M%S",
        "%Y%m%d%H%M",
        "%Y-%m-%d",
        "%Y/%m/%d",
    ):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    raise ValueError(f"Unsupported date format: {value!r}")


def is_weekday(dt: datetime) -> bool:
    return dt.weekday() < 5


def detect_columns(fieldnames: list[str] | None) -> dict[str, str | None]:
    return {
        "heure": _find_column(fieldnames, ["heure", "date"]),
        "antenna_id": _find_column(fieldnames, ["nidt", "sys.nidt"]),
        "traffic": _find_column(fieldnames, ["dl_volume", "pdcp", "gbytes"]),
        "power": _find_column(fieldnames, ["power", "consumption"]),
    }


def _parse_row(row: dict, columns: dict[str, str | None], fields: tuple[str, ...]) -> dict | None:
    try:
        parsed: dict = {}
        for field in fields:
            col = columns["heure"] if field == "datetime" else columns.get(field)
            if not col:
                return None
            text = (row.get(col) or "").strip()
            if not text:
                return None
            if field == "datetime":
                parsed["datetime"] = _parse_datetime(text)
            elif field in {"traffic", "power"}:
                parsed[field] = float(text.replace(",", "."))
            else:
                parsed[field] = text
        return parsed
    except ValueError:
        return None


def extract_antenna_time_series(
    antenna_id: str,
    csv_path: Path = CSV_PATH,
) -> list[tuple[datetime, float, float]]:
    """Return (datetime, traffic, power) tuples for one antenna, sorted by time."""
    with open(csv_path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, delimiter=";")
        columns = detect_columns(reader.fieldnames)
        required = ("heure", "antenna_id", "traffic", "power")
        if any(not columns[k] for k in required):
            raise SystemExit(f"Missing CSV columns. Found: {reader.fieldnames}")

        rows = []
        for row in reader:
            parsed = _parse_row(row, columns, ("antenna_id", "datetime", "traffic", "power"))
            if parsed and parsed["antenna_id"] == antenna_id:
                rows.append((parsed["datetime"], parsed["traffic"], parsed["power"]))

    if not rows:
        raise SystemExit(f"No valid rows found for antenna {antenna_id}")
    rows.sort(key=lambda item: item[0])
    return rows


def compute_rho_from_traffic(traffic_values: list[float]) -> list[float]:
    if not traffic_values:
        return []
    peak = max(traffic_values)
    return [0.0 if peak <= 0 else v / peak for v in traffic_values]


def extract_ids_data(
    ids: list[str] | None = None,
    n: int = 1,
    include_power: bool = False,
    csv_path: Path = CSV_PATH,
) -> tuple[list[str], dict]:
    """Extract traffic (and optionally power) time series for one or more antennas."""
    with open(csv_path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, delimiter=";")
        columns = detect_columns(reader.fieldnames)
        fields = ["antenna_id", "datetime", "traffic"]
        if include_power:
            fields.append("power")

        if ids is None:
            ids = []
            for row in reader:
                parsed = _parse_row(row, columns, ("antenna_id",))
                if parsed and parsed["antenna_id"] not in ids:
                    ids.append(parsed["antenna_id"])
                if len(ids) >= n:
                    break
            f.seek(0)
            reader = csv.DictReader(f, delimiter=";")

        data: dict = {
            nid: {"traffic": [], "datetime": [], **({"power": []} if include_power else {})}
            for nid in ids
        }
        for row in reader:
            parsed = _parse_row(row, columns, tuple(fields))
            if not parsed or parsed["antenna_id"] not in data:
                continue
            nid = parsed["antenna_id"]
            data[nid]["datetime"].append(parsed["datetime"])
            data[nid]["traffic"].append(parsed["traffic"])
            if include_power:
                data[nid]["power"].append(parsed["power"])

    ids = [nid for nid in ids if data[nid]["traffic"]]
    if not ids:
        raise SystemExit("No valid data found for the selected IDs")
    return ids, data
