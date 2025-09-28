from config import *
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from math import radians, cos, sin, asin, sqrt
from typing import List, Tuple, Dict
import re
from scipy import interpolate


def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate walking distance between two points in meters"""
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * asin(sqrt(a)) * 6371000


def calculate_walking_time(distance_meters: float) -> float:
    """Calculate walking time in minutes"""
    return distance_meters / WALKING_SPEED


def parse_showtime(showtime_str: str) -> Tuple[datetime, datetime]:
    """
    Parse showtime string to start and end datetime
    Expected format: "2025-09-28T12:30:00+08:00"
    """
    try:
        # Remove timezone info for simplicity
        clean_time = showtime_str.split('+')[0].split('T')
        date_part = clean_time[0]
        time_part = clean_time[1]

        start_time = datetime.strptime(f"{date_part} {time_part}", "%Y-%m-%d %H:%M:%S")
        end_time = start_time + timedelta(minutes=DEFAULT_SHOW_DURATION)

        return start_time, end_time
    except:
        # Fallback to current date with extracted time
        try:
            time_match = re.search(r'(\d{2}):(\d{2})', showtime_str)
            if time_match:
                hour, minute = map(int, time_match.groups())
                today = datetime.now().replace(hour=hour, minute=minute, second=0, microsecond=0)
                return today, today + timedelta(minutes=DEFAULT_SHOW_DURATION)
        except:
            pass

        return None, None


def get_available_showtimes(shows_df: pd.DataFrame, start_time: datetime, end_time: datetime) -> Dict:
    """
    Get all available showtimes within the planning window
    Returns: {show_name: [(start_time, end_time), ...]}
    """
    available_shows = {}

    for _, show in shows_df.iterrows():
        show_name = show['name']
        available_shows[show_name] = []

        if 'showtimes' in show and show['showtimes']:
            for showtime in show['showtimes']:
                if 'startTime' in showtime:
                    show_start, show_end = parse_showtime(showtime['startTime'])

                    if (show_start and show_end and
                            start_time <= show_start <= end_time):
                        available_shows[show_name].append((show_start, show_end))

        # Sort by start time
        available_shows[show_name].sort(key=lambda x: x[0])

    return available_shows


def predict_wait_time_from_timestamp_data(timestamp_data: List[Tuple[datetime, float]], target_time: datetime) -> float:
    """
    Predict wait time based on timestamp data using interpolation
    timestamp_data: [(datetime, wait_time), ...]
    target_time: datetime to predict for
    """
    if not timestamp_data:
        return 0.0

    # Filter to same day and sort by time
    target_date = target_time.date()
    same_day_data = []

    for dt, wait_time in timestamp_data:
        if dt.date() == target_date:
            same_day_data.append((dt, wait_time))

    if not same_day_data:
        # If no data for target date, use all data but adjust time
        same_day_data = []
        for dt, wait_time in timestamp_data:
            # Create new datetime with target date but original time
            adjusted_dt = datetime.combine(target_date, dt.time())
            same_day_data.append((adjusted_dt, wait_time))

    if not same_day_data:
        return 0.0

    # Sort by time
    same_day_data.sort(key=lambda x: x[0])

    # Convert to minutes since midnight for interpolation
    time_points = []
    wait_values = []

    for dt, wait_time in same_day_data:
        minutes_since_midnight = dt.hour * 60 + dt.minute
        time_points.append(minutes_since_midnight)
        wait_values.append(wait_time)

    target_minutes = target_time.hour * 60 + target_time.minute

    # Handle edge cases
    if len(time_points) == 1:
        return wait_values[0]

    if target_minutes <= min(time_points):
        # Before first data point - use first value
        return wait_values[0]
    elif target_minutes >= max(time_points):
        # After last data point - use last value
        return wait_values[-1]
    else:
        # Interpolate between data points
        try:
            f = interpolate.interp1d(time_points, wait_values, kind='linear',
                                     bounds_error=False, fill_value='extrapolate')
            predicted_wait = float(f(target_minutes))
            return max(0, predicted_wait)  # Ensure non-negative
        except:
            # Fallback to nearest neighbor
            nearest_idx = min(range(len(time_points)),
                              key=lambda i: abs(time_points[i] - target_minutes))
            return wait_values[nearest_idx]


def calculate_satisfaction_scores(attractions_df: pd.DataFrame, shows_df: pd.DataFrame) -> Tuple[
    pd.DataFrame, pd.DataFrame]:
    """
    Calculate satisfaction scores
    For attractions: satisfaction score = yesterday's average wait time (lower is better for optimization)
    For shows: satisfaction score = overall average wait * 2 (high value items)
    """
    attractions_df = attractions_df.copy()
    shows_df = shows_df.copy()

    if not attractions_df.empty:
        # For attractions: satisfaction score = yesterday's average wait time
        attractions_df['satisfaction_score'] = attractions_df['yesterday_avg_wait']

    if not shows_df.empty:
        # Shows get high satisfaction score (2x average attraction wait)
        attractions_avg = attractions_df['yesterday_avg_wait'].mean() if not attractions_df.empty else 20
        shows_df['satisfaction_score'] = attractions_avg * SATISFACTION_SCORING['show_multiplier']

    return attractions_df, shows_df