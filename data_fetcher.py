from config import *
import requests
import json
import re
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import time
import streamlit as st


def fetch_data(url):
    """Generic data fetcher with error handling"""
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return {}


def parse_timestamp(timestamp_str):
    """Parse various timestamp formats"""
    formats = ["%m/%d/%y %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%m/%d/%Y %H:%M:%S"]
    for fmt in formats:
        try:
            return datetime.strptime(timestamp_str.split('+')[0], fmt)
        except ValueError:
            continue
    return None


def scrape_weekday_wait_times(attraction_id, attraction_name, progress_placeholder=None):
    """Scrape wait times from same weekday (today - 7 days)"""
    target_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    url = f"{QUEUE_TIMES_BASE_URL}/parks/{QUEUE_TIMES_PARK_ID}/rides/{attraction_id}?given_date={target_date}"

    if progress_placeholder:
        progress_placeholder.text(f"🔄 {attraction_name} ({target_date})...")

    try:
        response = requests.get(url)
        content = response.text

        # Extract chart data
        match = re.search(r'"name":"Reported by park".*?"data":(\[\[.*?\]\])', content, re.DOTALL)
        if not match:
            return [], 0.0

        raw_data = json.loads(match.group(1))
        timestamp_data = []

        for point in raw_data:
            if len(point) == 2:
                parsed_time = parse_timestamp(point[0])
                if parsed_time:
                    timestamp_data.append((parsed_time, float(point[1])))

        if timestamp_data:
            avg_wait = np.mean([wait for _, wait in timestamp_data])
            if progress_placeholder:
                progress_placeholder.text(f"✅ {attraction_name}: {avg_wait:.1f}min")
            time.sleep(0.3)
            return timestamp_data, avg_wait

    except Exception as e:
        if progress_placeholder:
            progress_placeholder.text(f"❌ {attraction_name}: Error")

    return [], 0.0


def is_cache_valid(cache_timestamp, cache_duration_hours=1):
    """Check if cached data is still valid"""
    if not cache_timestamp:
        return False
    return datetime.now() - cache_timestamp < timedelta(hours=cache_duration_hours)


def get_cached_wait_times(progress_container=None):
    """Get wait times with 1-hour caching"""
    # Check if we have valid cached data
    if (hasattr(st.session_state, 'weekday_wait_data') and
            hasattr(st.session_state, 'weekday_data_timestamp') and
            is_cache_valid(st.session_state.weekday_data_timestamp)):

        if progress_container:
            progress_placeholder = progress_container.empty()
            progress_placeholder.text("✅ Using cached wait time data")
            time.sleep(0.5)  # Brief display
            progress_placeholder.empty()

        print("📋 Using cached weekday wait time data")
        return st.session_state.weekday_wait_data

    # Fetch fresh data
    print("🔄 Fetching fresh weekday wait time data...")

    if progress_container:
        progress_placeholder = progress_container.empty()
    else:
        progress_placeholder = None

    # Get attraction ID mapping
    queue_data = fetch_data(f"{QUEUE_TIMES_BASE_URL}/parks/{QUEUE_TIMES_PARK_ID}/queue_times.json")
    id_mapping = {ride['name']: ride['id'] for ride in queue_data.get('rides', [])}

    weekday_data = {}
    successful_fetches = 0

    for attraction_name, attraction_id in id_mapping.items():
        timestamp_data, avg_wait = scrape_weekday_wait_times(attraction_id, attraction_name, progress_placeholder)

        has_valid_data = len(timestamp_data) >= 10 and avg_wait > 0

        weekday_data[attraction_name] = {
            'timestamp_data': timestamp_data,
            'avg_wait': avg_wait,
            'has_data': has_valid_data,
            'queue_times_id': attraction_id
        }

        if has_valid_data:
            successful_fetches += 1

    # Cache the data with timestamp
    st.session_state.weekday_wait_data = weekday_data
    st.session_state.weekday_data_timestamp = datetime.now()

    if progress_placeholder:
        progress_placeholder.text(f"✅ Cached: {successful_fetches}/{len(id_mapping)} attractions")

    print(f"✅ Fetched and cached data for {successful_fetches}/{len(id_mapping)} attractions")
    return weekday_data


def get_park_hours_for_date(date_str=None):
    """Get park opening and closing hours"""
    if not date_str:
        date_str = datetime.now().strftime('%Y-%m-%d')

    schedule = fetch_data(f"{THEMEPARKS_BASE_URL}/{HKDL_DESTINATION_ID}/schedule")

    if 'parks' in schedule and schedule['parks']:
        for day in schedule['parks'][0].get('schedule', []):
            if day['date'] == date_str:
                opening = day['openingTime'].split('T')[1].split('+')[0][:5]
                closing = day['closingTime'].split('T')[1].split('+')[0][:5]
                return opening, closing
    return "10:00", "20:30"


def get_all_attraction_data(progress_container=None):
    """Main data loading function with smart caching"""
    print("📊 Loading attraction and show data...")

    # Check if we have valid cached attraction/show data (separate from wait times)
    if (hasattr(st.session_state, 'attractions_data') and
            hasattr(st.session_state, 'shows_data') and
            hasattr(st.session_state, 'attraction_data_timestamp') and
            is_cache_valid(st.session_state.attraction_data_timestamp, cache_duration_hours=1) and
            st.session_state.attractions_data is not None):

        # Still get fresh wait times (they have their own cache)
        weekday_data = get_cached_wait_times(progress_container)

        if progress_container:
            progress_placeholder = progress_container.empty()
            progress_placeholder.text("✅ Using cached attraction/show data")
            time.sleep(0.5)
            progress_placeholder.empty()

        print("📋 Using cached attraction/show data")
        return st.session_state.attractions_data, st.session_state.shows_data, weekday_data

    # Fetch fresh attraction and show data
    print("🔄 Fetching fresh attraction and show data...")

    # Fetch all data
    live_data = fetch_data(f"{THEMEPARKS_BASE_URL}/{HKDL_DESTINATION_ID}/live")
    location_data = fetch_data(f"{THEMEPARKS_BASE_URL}/{HKDL_DESTINATION_ID}/children")

    # Get wait time data (with its own caching)
    weekday_data = get_cached_wait_times(progress_container)

    # Create mappings
    location_lookup = {item['name']: item['location'] for item in location_data.get('children', []) if
                       'location' in item}

    # Calculate overall average for shows
    valid_waits = [data['avg_wait'] for data in weekday_data.values() if data['has_data']]
    overall_avg = sum(valid_waits) / len(valid_waits) if valid_waits else 20.0

    # Process attractions and shows
    attractions, shows = [], []

    for item in live_data.get('liveData', []):
        name, entity_type, status = item.get('name', ''), item.get('entityType'), item.get('status')

        # Skip if not operational or excluded
        if status != 'OPERATING' or any(kw in name.lower() for kw in EXCLUDE_KEYWORDS):
            continue

        location = location_lookup.get(name, {'latitude': 0, 'longitude': 0})
        if location['latitude'] == 0:
            continue

        base_data = {
            'name': name, 'id': item['id'], 'status': status, 'entity_type': entity_type,
            'latitude': location['latitude'], 'longitude': location['longitude']
        }

        if entity_type == 'ATTRACTION' and name in weekday_data and weekday_data[name]['has_data']:
            weekday_info = weekday_data[name]
            current_wait = 0
            if 'queue' in item and 'STANDBY' in item['queue']:
                current_wait = item['queue']['STANDBY'].get('waitTime', 0)

            attractions.append({
                **base_data,
                'current_wait_time': current_wait,
                'yesterday_avg_wait': weekday_info['avg_wait'],
                'timestamp_data': weekday_info['timestamp_data'],
                'satisfaction_score': weekday_info['avg_wait']
            })

        elif entity_type == 'SHOW' and 'showtimes' in item and item['showtimes']:
            shows.append({
                **base_data,
                'showtimes': item['showtimes'],
                'satisfaction_score': overall_avg * SATISFACTION_SCORING['show_multiplier'],
                'show_duration': DEFAULT_SHOW_DURATION
            })

    # Cache the attraction and show data
    attractions_df = pd.DataFrame(attractions)
    shows_df = pd.DataFrame(shows)

    st.session_state.attractions_data = attractions_df
    st.session_state.shows_data = shows_df
    st.session_state.attraction_data_timestamp = datetime.now()

    if progress_container:
        progress_placeholder = progress_container.empty()
        progress_placeholder.text(f"✅ Cached: {len(attractions_df)} attractions, {len(shows_df)} shows")
        time.sleep(0.5)
        progress_placeholder.empty()

    print(f"✅ Fetched and cached {len(attractions_df)} attractions and {len(shows_df)} shows")
    return attractions_df, shows_df, weekday_data