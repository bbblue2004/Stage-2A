"""
Data processing module - consolidates CSV reading, parsing, and data extraction.
Main entry point for all data processing operations.
"""

from .data_loader import (
    # Utility functions
    make_output_path,
    find_column,
    parse_datetime,
    hour_key,
    is_weekday,
    
    # CSV reading
    get_fieldnames,
    read_csv_data,
    detect_columns,
    validate_columns,
    
    # Data extraction
    extract_antenna_time_series,
    extract_ids_data,
    extract_antenna_power_data,
    
    # Processing
    compute_rho_from_traffic,
    select_weekdays,
    count_distinct_ids,
    
    # Constants
    CSV_PATH,
    OUTPUT_DIR,
)

from .antenna_metrics import compute_antenna_metrics
from .avg_daily_rho import compute_hourly_rho
from .power_vs_rho import get_energy_traffic_rho_for_id
from .rho_weekly import plot_traffic_time_series

__all__ = [
    # data_loader
    'make_output_path',
    'find_column',
    'parse_datetime',
    'hour_key',
    'is_weekday',
    'get_fieldnames',
    'read_csv_data',
    'detect_columns',
    'validate_columns',
    'extract_antenna_time_series',
    'extract_ids_data',
    'extract_antenna_power_data',
    'compute_rho_from_traffic',
    'select_weekdays',
    'count_distinct_ids',
    'CSV_PATH',
    'OUTPUT_DIR',
    
    # antenna_metrics
    'compute_antenna_metrics',
    
    # avg_daily_rho
    'compute_hourly_rho',
    
    # power_vs_rho
    'get_energy_traffic_rho_for_id',
    
    # rho_weekly
    'plot_traffic_time_series',
]
