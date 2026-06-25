"""
Centralized data loading and processing utilities for CSV data from radio_sites.csv.
Eliminates code duplication across data processing modules.
"""
import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = PROJECT_ROOT / "data" / "raw" / "radio_sites.csv"
OUTPUT_DIR = PROJECT_ROOT / "figures" / "data_figures"


# ============================================================================
# Utility Functions
# ============================================================================

def make_output_path(filename: str) -> Path:
    """Create output directory and return path for a figure."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR / filename


def find_column(fieldnames, keywords):
    """Find a column name by keywords (case-insensitive)."""
    for keyword in keywords:
        for name in fieldnames:
            if keyword.lower() in name.lower():
                return name
    return None


def parse_datetime(value):
    """Parse datetime from common formats."""
    value = value.strip()
    if not value:
        raise ValueError('Empty date value')

    formats = [
        '%Y-%m-%d %H:%M:%S',
        '%Y-%m-%dT%H:%M:%S',
        '%Y-%m-%d %H:%M',
        '%Y-%m-%d %H',
        '%Y/%m/%d %H:%M:%S',
        '%d/%m/%Y %H:%M:%S',
        '%Y%m%d%H%M%S',
        '%Y%m%d%H%M',
        '%Y-%m-%d',
        '%Y/%m/%d',
    ]

    for fmt in formats:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue

    raise ValueError(f'Unsupported date format: {value!r}')


def hour_key(dt):
    """Convert datetime to hourly key."""
    return dt.replace(minute=0, second=0, microsecond=0)


def is_weekday(dt):
    """Check if datetime is a weekday (Mon-Fri)."""
    return dt.weekday() < 5


# ============================================================================
# CSV Reading and Field Detection
# ============================================================================

def get_fieldnames(csv_path=CSV_PATH):
    """Get column names from CSV."""
    with open(csv_path, newline='', encoding='utf-8', errors='replace') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=';')
        return reader.fieldnames or []


def read_csv_data(csv_path=CSV_PATH):
    """Read all data from CSV into memory."""
    with open(csv_path, newline='', encoding='utf-8', errors='replace') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=';')
        return list(reader), reader.fieldnames or []


def detect_columns(fieldnames):
    """Detect standard column names in CSV."""
    return {
        'heure': find_column(fieldnames, ['heure', 'date']),
        'antenna_id': find_column(fieldnames, ['nidt', 'sys.nidt']),
        'traffic': find_column(fieldnames, ['dl_volume', 'pdcp', 'gbytes']),
        'power': find_column(fieldnames, ['power', 'consumption']),
    }


def validate_columns(columns, required=None):
    """Validate that required columns are detected."""
    if required is None:
        required = ['heure', 'antenna_id', 'traffic', 'power']
    
    missing = [col for col in required if not columns.get(col)]
    if missing:
        raise SystemExit(f'Missing required columns: {missing}')


# ============================================================================
# Data Extraction Functions
# ============================================================================

def extract_antenna_time_series(antenna_id, csv_path=CSV_PATH):
    """
    Return (datetime, traffic, power) tuples for a given antenna.
    """
    fieldnames = get_fieldnames(csv_path)
    columns = detect_columns(fieldnames)
    validate_columns(columns, ['heure', 'antenna_id', 'traffic', 'power'])

    rows = []
    with open(csv_path, newline='', encoding='utf-8', errors='replace') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=';')
        for row in reader:
            nidt = (row.get(columns['antenna_id']) or '').strip()
            if nidt != antenna_id:
                continue

            heure_value = (row.get(columns['heure']) or '').strip()
            traffic_value = (row.get(columns['traffic']) or '').strip()
            power_value = (row.get(columns['power']) or '').strip()
            if not heure_value or not traffic_value or not power_value:
                continue

            try:
                dt = parse_datetime(heure_value)
                traffic = float(traffic_value.replace(',', '.'))
                power = float(power_value.replace(',', '.'))
            except ValueError:
                continue

            rows.append((dt, traffic, power))

    if not rows:
        raise SystemExit(f'No valid rows found for antenna {antenna_id}')

    rows.sort(key=lambda item: item[0])
    return rows


def extract_ids_data(ids=None, n=1, include_power=False, csv_path=CSV_PATH):
    """
    Extract traffic and optionally power data for requested IDs.
    If ids is None, extracts data for first n distinct IDs.
    Returns (list of IDs, data dict).
    """
    fieldnames = get_fieldnames(csv_path)
    columns = detect_columns(fieldnames)
    
    required = ['heure', 'antenna_id', 'traffic']
    if include_power:
        required.append('power')
    validate_columns(columns, required)

    if ids is None:
        ids = []
        with open(csv_path, newline='', encoding='utf-8', errors='replace') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=';')
            for row in reader:
                nidt = (row.get(columns['antenna_id']) or '').strip()
                if nidt and nidt not in ids:
                    ids.append(nidt)
                if len(ids) >= n:
                    break

    if not ids:
        raise SystemExit('No IDs selected for processing')

    print(f'Processing IDs: {ids}')

    if include_power:
        data = {nidt: {'traffic': [], 'power': [], 'datetime': []} for nidt in ids}
    else:
        data = {nidt: {'traffic': [], 'datetime': []} for nidt in ids}

    with open(csv_path, newline='', encoding='utf-8', errors='replace') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=';')
        for row in reader:
            nidt = (row.get(columns['antenna_id']) or '').strip()
            if nidt not in data:
                continue

            heure_value = (row.get(columns['heure']) or '').strip()
            traffic_str = (row.get(columns['traffic']) or '').strip()

            if not heure_value or not traffic_str:
                continue

            try:
                dt = parse_datetime(heure_value)
                traffic = float(traffic_str.replace(',', '.'))
            except ValueError:
                continue

            data[nidt]['datetime'].append(dt)
            data[nidt]['traffic'].append(traffic)

            if include_power:
                power_str = (row.get(columns['power']) or '').strip()
                if not power_str:
                    continue
                try:
                    power = float(power_str.replace(',', '.'))
                    data[nidt]['power'].append(power)
                except ValueError:
                    continue

    # Remove IDs with no valid data
    for nidt in list(data):
        if not data[nidt]['traffic']:
            print(f'Warning: no valid data found for ID {nidt}, removing')
            data.pop(nidt)

    if not data:
        raise SystemExit('No valid data found for the selected IDs')

    return list(data.keys()), data


def extract_antenna_power_data(antenna_id, csv_path=CSV_PATH):
    """Extract hourly power consumption data for a specific antenna ID."""
    fieldnames = get_fieldnames(csv_path)
    columns = detect_columns(fieldnames)
    validate_columns(columns, ['heure', 'antenna_id', 'power'])

    rows_by_hour = defaultdict(list)

    with open(csv_path, newline='', encoding='utf-8', errors='replace') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=';')
        for row in reader:
            nidt = (row.get(columns['antenna_id']) or '').strip()
            if nidt != antenna_id:
                continue

            heure_value = (row.get(columns['heure']) or '').strip()
            power_value = (row.get(columns['power']) or '').strip()
            if not heure_value or not power_value:
                continue

            try:
                dt = parse_datetime(heure_value)
                power = float(power_value.replace(',', '.'))
            except ValueError:
                continue

            rows_by_hour[hour_key(dt)].append(power)

    if not rows_by_hour:
        raise SystemExit(f'No valid power data found for antenna {antenna_id}')

    return rows_by_hour


# ============================================================================
# Data Processing Functions
# ============================================================================

def compute_rho_from_traffic(traffic_values):
    """Return normalized rho values (0-1) for a traffic series."""
    if not traffic_values:
        return []

    max_traffic = max(traffic_values)
    if max_traffic <= 0:
        return [0.0 for _ in traffic_values]

    return [float(value) / max_traffic for value in traffic_values]


def select_weekdays(rows, date_col, day_start=20, day_end=24, max_days=5):
    """Select weekdays within a given day-of-month range."""
    days = []
    seen = set()
    for row in rows:
        raw = (row.get(date_col) or '').strip()
        if not raw:
            continue
        try:
            dt = parse_datetime(raw)
        except ValueError:
            continue
        if not (day_start <= dt.day <= day_end):
            continue
        if not is_weekday(dt):
            continue
        day = dt.date()
        if day not in seen:
            seen.add(day)
            days.append(day)
            if len(days) >= max_days:
                break
    return days


def count_distinct_ids(rows, id_col):
    """Count distinct antenna IDs in data."""
    return len({(row.get(id_col) or '').strip() for row in rows if (row.get(id_col) or '').strip()})


def aggregate_by_hour(series_list, datetime_key=None):
    """Aggregate a list of values by hour."""
    hourly = defaultdict(list)
    
    if datetime_key and hasattr(series_list[0] if series_list else None, 'hour'):
        # Series of datetime objects
        for dt in series_list:
            hourly[dt.hour].append(dt)
    else:
        # Simple values
        for value in series_list:
            hourly[len(hourly)].append(value)
    
    return hourly
