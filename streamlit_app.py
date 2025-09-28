import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
from datetime import datetime, timedelta

from config import *
from data_fetcher import get_all_attraction_data, get_park_hours_for_date, is_cache_valid
from data_processor import calculate_satisfaction_scores
from optimizer import CompactMonteCarloOptimizer

st.set_page_config(page_title="HKDL Smart Trip Planner", layout="wide")

# Initialize session state
for key in ['selected_attractions', 'selected_shows', 'high_priority_attractions', 'high_priority_shows']:
    if key not in st.session_state:
        st.session_state[key] = []


def show_cache_status():
    """Display cache status in sidebar"""
    with st.sidebar:
        st.subheader("📊 Cache Status")

        # Wait time data cache
        if (hasattr(st.session_state, 'weekday_data_timestamp') and
                st.session_state.weekday_data_timestamp):

            time_since = datetime.now() - st.session_state.weekday_data_timestamp
            minutes_ago = int(time_since.total_seconds() / 60)

            if is_cache_valid(st.session_state.weekday_data_timestamp):
                st.success(f"✅ Wait times: Fresh ({minutes_ago}m ago)")
            else:
                st.warning(f"⚠️ Wait times: Stale ({minutes_ago}m ago)")
        else:
            st.error("❌ Wait times: Not cached")

        # Attraction/show data cache
        if (hasattr(st.session_state, 'attraction_data_timestamp') and
                st.session_state.attraction_data_timestamp):

            time_since = datetime.now() - st.session_state.attraction_data_timestamp
            minutes_ago = int(time_since.total_seconds() / 60)

            if is_cache_valid(st.session_state.attraction_data_timestamp):
                st.success(f"✅ Attractions: Fresh ({minutes_ago}m ago)")
            else:
                st.warning(f"⚠️ Attractions: Stale ({minutes_ago}m ago)")
        else:
            st.error("❌ Attractions: Not cached")

        # Manual refresh button
        if st.button("🔄 Force Refresh Data"):
            # Clear all cache
            for key in ['weekday_wait_data', 'weekday_data_timestamp',
                        'attractions_data', 'shows_data', 'attraction_data_timestamp']:
                if hasattr(st.session_state, key):
                    delattr(st.session_state, key)
            st.rerun()


@st.cache_data(ttl=3600)
def load_data():
    return get_all_attraction_data()


def correct_coordinates(lat, lon):
    return lat + COORDINATE_CORRECTION['lat_offset'], lon + COORDINATE_CORRECTION['lon_offset']


def create_map(attractions_df, shows_df, sequence=None):
    if attractions_df.empty and shows_df.empty:
        return None

    m = folium.Map(location=[22.3129, 114.0413], zoom_start=17, tiles='OpenStreetMap')

    # Add markers
    for _, row in attractions_df.iterrows():
        color = 'darkred' if row['name'] in st.session_state.high_priority_attractions else \
            'red' if row['name'] in st.session_state.selected_attractions else 'blue'
        lat, lon = correct_coordinates(row['latitude'], row['longitude'])

        folium.Marker([lat, lon], popup=f"🎢 {row['name']}<br>Avg: {row['yesterday_avg_wait']:.1f}min",
                      tooltip=f"🎢 {row['name']}", icon=folium.Icon(color=color, icon='star')).add_to(m)

    for _, row in shows_df.iterrows():
        color = 'darkred' if row['name'] in st.session_state.high_priority_shows else \
            'red' if row['name'] in st.session_state.selected_shows else 'green'
        lat, lon = correct_coordinates(row['latitude'], row['longitude'])

        folium.Marker([lat, lon], popup=f"🎭 {row['name']}<br>Shows: {len(row['showtimes'])}",
                      tooltip=f"🎭 {row['name']}", icon=folium.Icon(color=color, icon='play')).add_to(m)

    # Add route
    if sequence:
        coords = []
        for item in sequence:
            if item['type'] in ['attraction', 'show']:
                df = attractions_df if item['type'] == 'attraction' else shows_df
                item_data = df[df['name'] == item['name']]
                if not item_data.empty:
                    lat, lon = item_data.iloc[0]['latitude'], item_data.iloc[0]['longitude']
                    coords.append(correct_coordinates(lat, lon))

        if coords:
            folium.PolyLine(coords, color='purple', weight=4, opacity=0.8).add_to(m)
            for i, coord in enumerate(coords, 1):
                folium.Marker(coord, icon=folium.DivIcon(
                    html=f'<div style="color: white; background-color: purple; border-radius: 50%; width: 25px; height: 25px; text-align: center; line-height: 25px; font-weight: bold;">{i}</div>',
                    icon_size=(25, 25))).add_to(m)

    return m


def format_time(dt):
    return dt.strftime('%H:%M') if isinstance(dt, datetime) else str(dt)


def format_duration(minutes):
    if minutes >= 60:
        hours, mins = int(minutes // 60), int(minutes % 60)
        return f"{hours}h {mins}m" if mins > 0 else f"{hours}h"
    return f"{int(minutes)}m"


def validate_meal_time(meal_type, start_time, end_time):
    """Check if meal is possible given start/end times"""
    start_hour = start_time.hour + start_time.minute / 60
    end_hour = end_time.hour + end_time.minute / 60

    if meal_type == 'lunch':
        return start_hour <= 12.0
    else:  # dinner
        return end_hour >= 19.0


def main():
    st.title("🏰 Hong Kong Disneyland Smart Trip Planner")
    st.markdown("**Plan your perfect day using Monte Carlo optimization with same weekday wait times!**")

    # Show cache status
    show_cache_status()

    # Load data with caching
    progress_container = st.container()

    # Always try to load data (caching is handled internally)
    attractions_df, shows_df, yesterday_data = get_all_attraction_data(progress_container)

    if not attractions_df.empty or not shows_df.empty:
        # Calculate satisfaction scores (only if data changed)
        if not hasattr(st.session_state, 'scores_calculated') or not st.session_state.scores_calculated:
            attractions_df, shows_df = calculate_satisfaction_scores(attractions_df, shows_df)
            st.session_state.scores_calculated = True

        # Initialize selections if empty
        if not st.session_state.selected_attractions:
            st.session_state.selected_attractions = attractions_df['name'].tolist()
        if not st.session_state.selected_shows:
            st.session_state.selected_shows = shows_df['name'].tolist()

        # Show data freshness info
        if hasattr(st.session_state, 'weekday_data_timestamp'):
            time_since = datetime.now() - st.session_state.weekday_data_timestamp
            minutes_ago = int(time_since.total_seconds() / 60)
            cache_status = "🟢 Fresh" if is_cache_valid(st.session_state.weekday_data_timestamp) else "🟡 Stale"
            st.info(
                f"📊 Data loaded: {len(attractions_df)} attractions, {len(shows_df)} shows | Wait times: {cache_status} ({minutes_ago}m ago)")

        progress_container.empty()
    else:
        st.error("❌ Could not load data. Please check connection and try refresh.")
        progress_container.empty()
        return

    # Layout
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("🗺️ Park Map")
        folium_static(create_map(attractions_df, shows_df), width=700, height=500)

    with col2:
        st.subheader("⚙️ Trip Settings")

        today = datetime.now().strftime('%Y-%m-%d')
        park_opening, park_closing = get_park_hours_for_date(today)

        start_time = st.time_input("Start Time", value=datetime.strptime(park_opening, "%H:%M").time(),
                                   help=f"Park: {park_opening}-{park_closing}")
        end_time = st.time_input("End Time", value=datetime.strptime(park_closing, "%H:%M").time())

        if start_time >= end_time:
            st.error("End time must be after start time")
            return

        st.subheader("🍽️ Meals")
        include_lunch = st.checkbox("Lunch (90min, 12:00-14:00)", value=False,
                                    disabled=not validate_meal_time('lunch', start_time, end_time))
        include_dinner = st.checkbox("Dinner (90min, 17:30-20:00)", value=False,
                                     disabled=not validate_meal_time('dinner', start_time, end_time))

        iterations = st.number_input("Monte Carlo Iterations", min_value=100, max_value=5000, value=1000, step=100)

    # Selection
    st.subheader("🎯 Select Experiences")

    tab1, tab2 = st.tabs(["🎢 Attractions", "🎭 Shows"])

    with tab1:
        col_a1, col_a2, col_a3 = st.columns([3, 1, 1])
        with col_a1:
            st.markdown("*⭐ High Priority = 4x score boost*")
        with col_a2:
            if st.button("🔲 All", key="all_attr"):
                st.session_state.selected_attractions = attractions_df['name'].tolist()
                st.rerun()
        with col_a3:
            if st.button("☐ None", key="none_attr"):
                st.session_state.selected_attractions = []
                st.session_state.high_priority_attractions = []
                st.rerun()

        for _, row in attractions_df.sort_values('satisfaction_score', ascending=False).iterrows():
            col1, col2 = st.columns([1, 4])

            with col1:
                high_priority = st.checkbox("⭐", value=row['name'] in st.session_state.high_priority_attractions,
                                            key=f"hp_attr_{row['name']}", help="High Priority",
                                            disabled=row['name'] not in st.session_state.selected_attractions)
            with col2:
                selected = st.checkbox(f"**{row['name']}** ({row['yesterday_avg_wait']:.0f}min)",
                                       value=row['name'] in st.session_state.selected_attractions,
                                       key=f"sel_attr_{row['name']}")

            # Update state
            if selected and row['name'] not in st.session_state.selected_attractions:
                st.session_state.selected_attractions.append(row['name'])
            elif not selected and row['name'] in st.session_state.selected_attractions:
                st.session_state.selected_attractions.remove(row['name'])
                if row['name'] in st.session_state.high_priority_attractions:
                    st.session_state.high_priority_attractions.remove(row['name'])

            if high_priority and row['name'] not in st.session_state.high_priority_attractions:
                st.session_state.high_priority_attractions.append(row['name'])
            elif not high_priority and row['name'] in st.session_state.high_priority_attractions:
                st.session_state.high_priority_attractions.remove(row['name'])

    with tab2:
        col_s1, col_s2, col_s3 = st.columns([3, 1, 1])
        with col_s1:
            st.markdown("*Shows have 3x satisfaction multiplier*")
        with col_s2:
            if st.button("🔲 All", key="all_shows"):
                st.session_state.selected_shows = shows_df['name'].tolist()
                st.rerun()
        with col_s3:
            if st.button("☐ None", key="none_shows"):
                st.session_state.selected_shows = []
                st.session_state.high_priority_shows = []
                st.rerun()

        for _, row in shows_df.sort_values('satisfaction_score', ascending=False).iterrows():
            if not row['showtimes']:
                continue

            col1, col2 = st.columns([1, 4])

            with col1:
                high_priority = st.checkbox("⭐", value=row['name'] in st.session_state.high_priority_shows,
                                            key=f"hp_show_{row['name']}", help="High Priority",
                                            disabled=row['name'] not in st.session_state.selected_shows)
            with col2:
                selected = st.checkbox(f"**{row['name']}** ({len(row['showtimes'])} times)",
                                       value=row['name'] in st.session_state.selected_shows,
                                       key=f"sel_show_{row['name']}")

            # Update state (same logic as attractions)
            if selected and row['name'] not in st.session_state.selected_shows:
                st.session_state.selected_shows.append(row['name'])
            elif not selected and row['name'] in st.session_state.selected_shows:
                st.session_state.selected_shows.remove(row['name'])
                if row['name'] in st.session_state.high_priority_shows:
                    st.session_state.high_priority_shows.remove(row['name'])

            if high_priority and row['name'] not in st.session_state.high_priority_shows:
                st.session_state.high_priority_shows.append(row['name'])
            elif not high_priority and row['name'] in st.session_state.high_priority_shows:
                st.session_state.high_priority_shows.remove(row['name'])

    # Optimization
    st.subheader("🚀 Optimize Your Trip")
    total_selections = len(st.session_state.selected_attractions) + len(st.session_state.selected_shows)
    high_priority_count = len(st.session_state.high_priority_attractions) + len(st.session_state.high_priority_shows)

    col1, col2 = st.columns([3, 1])
    with col1:
        st.info(
            f"📊 Selected: {len(st.session_state.selected_attractions)} attractions, {len(st.session_state.selected_shows)} shows | ⭐ High Priority: {high_priority_count}")

    with col2:
        if st.button("🎲 Optimize!", type="primary", disabled=total_selections < 1):
            if total_selections == 0:
                st.warning("⚠️ Select at least one item.")
            else:
                progress_container = st.container()
                optimizer = CompactMonteCarloOptimizer(attractions_df, shows_df)

                start_datetime = datetime.combine(datetime.today(), start_time)
                end_datetime = datetime.combine(datetime.today(), end_time)

                sequence, stats = optimizer.optimize_trip_monte_carlo(
                    st.session_state.selected_attractions.copy(), st.session_state.selected_shows.copy(),
                    st.session_state.high_priority_attractions.copy(), st.session_state.high_priority_shows.copy(),
                    start_datetime, end_datetime, include_lunch, include_dinner, iterations=iterations,
                    progress_container=progress_container
                )

                if sequence:
                    validity = "✅ Valid" if stats['best_valid'] else "⚠️ Partial"
                    st.success(f"🎉 Optimized! {validity}")

                    # Results
                    st.subheader("📊 Results")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Before", format_duration(stats['baseline_wait_time']))
                    with col2:
                        st.metric("After", format_duration(stats['optimized_wait_time']),
                                  delta=f"-{format_duration(stats['wait_time_reduction'])}")
                    with col3:
                        st.metric("Saved", format_duration(stats['wait_time_reduction']),
                                  delta=f"{stats['reduction_percentage']:.1f}%")
                    with col4:
                        st.metric("Iterations", stats['iterations_completed'])

                    # Itinerary
                    st.subheader("📅 Your Optimized Itinerary")
                    for i, item in enumerate(sequence, 1):
                        icon = {"attraction": "🎢", "show": "🎭", "meal": "🍽️"}[item['type']]
                        priority = " ⭐" if item.get('is_high_priority') else ""

                        with st.expander(
                                f"{i}. {icon} {item['name']}{priority} (Arrive: {format_time(item['arrival_time'])})"):
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.write(f"🚶 Travel: {format_duration(item['travel_time'])}")
                                st.write(f"⏳ Wait: {format_duration(item['wait_time'])}")
                                st.write(f"🎯 Duration: {format_duration(item['experience_duration'])}")
                            with col_b:
                                st.write(f"🕐 Start: {format_time(item['start_time'])}")
                                st.write(f"🕑 Finish: {format_time(item['departure_time'])}")
                                if item['type'] == 'attraction':
                                    base_score = item['satisfaction_score']
                                    if item.get('is_high_priority'):
                                        base_score /= SATISFACTION_SCORING['high_priority_multiplier']
                                    st.write(f"⭐ Same Weekday Avg: {base_score:.1f}min")

                    # Summary and map
                    summary = optimizer.get_summary_stats(sequence)
                    st.subheader("📊 Summary")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Attractions",
                                  f"{summary['visited_attractions']}/{len(st.session_state.selected_attractions)}")
                    with col2:
                        st.metric("Shows", f"{summary['watched_shows']}/{len(st.session_state.selected_shows)}")
                    with col3:
                        st.metric("Meals", summary['meals_scheduled'])

                    st.subheader("🗺️ Optimized Route")
                    folium_static(create_map(attractions_df, shows_df, sequence), width=700, height=400)
                else:
                    st.error("❌ No valid optimization found.")

                progress_container.empty()


if __name__ == "__main__":
    main()