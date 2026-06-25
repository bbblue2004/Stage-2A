"""
Centralized data loading and processing utilities for CSV data from radio_sites.csv.
Eliminates code duplication across data processing modules.
"""
import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path

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


def row_text(row, column_name):
    """Return stripped cell text from a CSV row."""
    return (row.get(column_name) or '').strip()


def parse_float_value(text):
    """Parse a numeric CSV cell, accepting comma decimals."""
    return float(text.replace(',', '.'))


def parse_row_values(row, column_map):
    """
    Parse selected fields from one CSV row.

    column_map: logical field name -> CSV column name.
    Supported logical names:
    - datetime: parsed with parse_datetime
    - traffic, power, value: parsed as float
    - other names: kept as stripped strings

    Returns a dict of parsed values, or None if any field is missing/invalid.
    """
    texts = {}
    for logical_name, csv_column in column_map.items():
        if not csv_column:
            return None
        text = row_text(row, csv_column)
        if not text:
            return None
        texts[logical_name] = text

    try:
        parsed = {}
        for logical_name, text in texts.items():
            if logical_name == 'datetime':
                parsed['datetime'] = parse_datetime(text)
            elif logical_name in {'traffic', 'power', 'value'}:
                parsed[logical_name] = parse_float_value(text)
            else:
                parsed[logical_name] = text
        return parsed
    except ValueError:
        return None


def parse_radio_row(row, columns, include):
    """Parse one radio_sites row using detected column names."""
    column_map = {}
    for field in include:
        if field == 'datetime':
            column_map['datetime'] = columns['heure']
        else:
            column_map[field] = columns[field]
    return parse_row_values(row, column_map)


def iter_csv_rows(csv_path=CSV_PATH):
    """Yield rows from radio_sites.csv."""
    with open(csv_path, newline='', encoding='utf-8', errors='replace') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=';')
        yield from reader


def prepare_columns(csv_path=CSV_PATH, required=None):
    """Detect and optionally validate standard CSV columns."""
    columns = detect_columns(get_fieldnames(csv_path))
    if required is not None:
        validate_columns(columns, required)
    return columns


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
    rows = list(iter_csv_rows(csv_path))
    fieldnames = get_fieldnames(csv_path)
    return rows, fieldnames


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
    columns = prepare_columns(
        csv_path,
        required=['heure', 'antenna_id', 'traffic', 'power'],
    )
    include = ('antenna_id', 'datetime', 'traffic', 'power')

    rows = []
    for row in iter_csv_rows(csv_path):
        parsed = parse_radio_row(row, columns, include)
        if not parsed or parsed['antenna_id'] != antenna_id:
            continue
        rows.append((parsed['datetime'], parsed['traffic'], parsed['power']))

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
    required = ['heure', 'antenna_id', 'traffic']
    if include_power:
        required.append('power')
    columns = prepare_columns(csv_path, required=required)

    if ids is None:
        ids = []
        for row in iter_csv_rows(csv_path):
            parsed = parse_radio_row(row, columns, ('antenna_id',))
            if not parsed:
                continue
            nidt = parsed['antenna_id']
            if nidt not in ids:
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

    include = ['antenna_id', 'datetime', 'traffic']
    if include_power:
        include.append('power')

    for row in iter_csv_rows(csv_path):
        parsed = parse_radio_row(row, columns, include)
        if not parsed:
            continue

        nidt = parsed['antenna_id']
        if nidt not in data:
            continue

        data[nidt]['datetime'].append(parsed['datetime'])
        data[nidt]['traffic'].append(parsed['traffic'])
        if include_power:
            data[nidt]['power'].append(parsed['power'])

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
    columns = prepare_columns(csv_path, required=['heure', 'antenna_id', 'power'])
    rows_by_hour = defaultdict(list)

    for row in iter_csv_rows(csv_path):
        parsed = parse_radio_row(row, columns, ('antenna_id', 'datetime', 'power'))
        if not parsed or parsed['antenna_id'] != antenna_id:
            continue
        rows_by_hour[hour_key(parsed['datetime'])].append(parsed['power'])

    if not rows_by_hour:
        raise SystemExit(f'No valid power data found for antenna {antenna_id}')

    return rows_by_hour


# ============================================================================
# Data Processing Functions
# ============================================================================

def normalize_traffic(value, max_traffic):
    """Normalize one traffic value to rho in [0, 1] using a known maximum."""
    if max_traffic <= 0:
        return 0.0
    return float(value) / max_traffic


def compute_rho_from_traffic(traffic_values):
    """Return normalized rho values (0-1) for a traffic series."""
    if not traffic_values:
        return []

    max_traffic = max(traffic_values)
    return [normalize_traffic(value, max_traffic) for value in traffic_values]


def select_weekdays(rows, date_col, day_start=20, day_end=24, max_days=5):
    """Select weekdays within a given day-of-month range."""
    days = []
    seen = set()
    for row in rows:
        parsed = parse_row_values(row, {'datetime': date_col})
        if not parsed:
            continue
        dt = parsed['datetime']
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
    ids = set()
    for row in rows:
        parsed = parse_row_values(row, {'antenna_id': id_col})
        if parsed:
            ids.add(parsed['antenna_id'])
    return len(ids)


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
