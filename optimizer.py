from config import *
import pandas as pd
from datetime import datetime, timedelta
import random
import numpy as np
from collections import defaultdict
from data_processor import calculate_distance, calculate_walking_time, get_available_showtimes, \
    predict_wait_time_from_timestamp_data


class CompactMonteCarloOptimizer:
    def __init__(self, attractions_df, shows_df):
        self.attractions = attractions_df
        self.shows = shows_df
        self.available_showtimes = {}
        self.successful_operations = defaultdict(int)

    def calculate_baseline_wait_time(self, sequence):
        """Calculate baseline wait time for items in sequence"""
        return sum(item['satisfaction_score'] / (
            SATISFACTION_SCORING['high_priority_multiplier'] if item.get('is_high_priority') else 1)
                   for item in sequence if item['type'] == 'attraction')

    def optimize_trip_monte_carlo(self, selected_attractions, selected_shows, high_priority_attractions,
                                  high_priority_shows, start_time, end_time, include_lunch=False,
                                  include_dinner=False, start_location=(22.316063, 114.056212),
                                  iterations=1000, progress_container=None):
        """Main optimization function"""

        progress_placeholder = progress_container.empty() if progress_container else None
        print(f"🎲 Starting optimization with {iterations} iterations...")

        # Setup
        self.available_showtimes = get_available_showtimes(self.shows, start_time, end_time)
        selected_showtimes = {show: [self.available_showtimes[show][0]]
                              for show in selected_shows if
                              show in self.available_showtimes and self.available_showtimes[show]}

        self.item_pool = self._create_item_pool(selected_attractions, selected_showtimes,
                                                high_priority_attractions, high_priority_shows, include_lunch,
                                                include_dinner)

        # Optimization loop
        current_sequence_items = []
        best_sequence, best_score, best_valid = [], -float('inf'), False
        valid_sequences = no_improvement_count = 0

        for i in range(iterations):
            if no_improvement_count >= MONTE_CARLO_SETTINGS['early_stop_threshold']:
                break

            if i % MONTE_CARLO_SETTINGS['progress_update_interval'] == 0:
                if progress_placeholder:
                    items_count = len([item for item in best_sequence if item['type'] in ['attraction', 'show']])
                    progress_placeholder.text(
                        f"🔄 Iteration {i}/{iterations} | Valid: {valid_sequences} | Best: {items_count} items")

            # Apply operation and evaluate
            operation = self._choose_operation()
            new_sequence_items = self._apply_operation(current_sequence_items, operation)

            current_scheduled = self._schedule_sequence(current_sequence_items, start_time, end_time, start_location)
            new_scheduled = self._schedule_sequence(new_sequence_items, start_time, end_time, start_location)

            current_valid = self._check_constraints(current_scheduled, start_time, end_time, include_lunch,
                                                    include_dinner)
            new_valid = self._check_constraints(new_scheduled, start_time, end_time, include_lunch, include_dinner)

            current_score = self._calculate_score(current_scheduled) if current_scheduled else -float('inf')
            new_score = self._calculate_score(new_scheduled) if new_scheduled else -float('inf')

            # Decision logic
            should_accept = (not current_valid and new_valid) or \
                            (current_valid == new_valid and new_score > current_score)

            if should_accept:
                self.successful_operations[operation] += 1
                current_sequence_items, current_scheduled = new_sequence_items, new_scheduled
                current_valid, current_score = new_valid, new_score
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            if current_valid:
                valid_sequences += 1

            if (current_valid and current_score > best_score) or (not best_valid and current_valid):
                best_sequence, best_score, best_valid = current_scheduled.copy(), current_score, current_valid
                no_improvement_count = 0

        # Calculate stats
        baseline_wait = self.calculate_baseline_wait_time(best_sequence)
        optimized_wait = sum(item['wait_time'] for item in best_sequence if item['type'] == 'attraction')

        return best_sequence, {
            'baseline_wait_time': baseline_wait, 'optimized_wait_time': optimized_wait,
            'wait_time_reduction': baseline_wait - optimized_wait,
            'reduction_percentage': (
                        (baseline_wait - optimized_wait) / baseline_wait * 100) if baseline_wait > 0 else 0,
            'valid_sequences_found': valid_sequences, 'best_score': best_score, 'best_valid': best_valid,
            'iterations_completed': i + 1,
            'early_stopped': no_improvement_count >= MONTE_CARLO_SETTINGS['early_stop_threshold']
        }

    def _create_item_pool(self, attractions, showtimes, hp_attractions, hp_shows, lunch, dinner):
        """Create pool of all possible items"""
        pool = []

        # Add attractions
        for attraction in attractions:
            pool.append({'name': attraction, 'type': 'attraction', 'is_high_priority': attraction in hp_attractions})

        # Add shows
        for show_name, times in showtimes.items():
            if times:
                start_time, end_time = times[0]
                pool.append({'name': show_name, 'type': 'show', 'start_time': start_time, 'end_time': end_time,
                             'is_high_priority': show_name in hp_shows, 'unique_id': f"{show_name}_show"})

        # Add meals
        if lunch:
            pool.append({'type': 'meal', 'meal_type': 'lunch', 'name': 'Lunch'})
        if dinner:
            pool.append({'type': 'meal', 'meal_type': 'dinner', 'name': 'Dinner'})

        return pool

    def _choose_operation(self):
        """Choose operation based on success history"""
        operations = ['add', 'swap', 'exchange', 'add_before_show']

        if sum(self.successful_operations.values()) > 10:
            weights = [max(0.1, self.successful_operations[op] / sum(self.successful_operations.values())) for op in
                       operations]
        else:
            weights = [1, 1, 1, 2]  # Favor add_before_show initially

        return np.random.choice(operations, p=[w / sum(weights) for w in weights])

    def _apply_operation(self, sequence, operation):
        """Apply operation to sequence"""
        new_sequence = sequence.copy()
        available_items = self._get_available_items(new_sequence)

        if operation == 'add' and len(new_sequence) < len(self.item_pool) and available_items:
            # Prefer meals, then high priority
            priority_items = [i for i in available_items if i['type'] == 'meal'] or \
                             [i for i in available_items if i.get('is_high_priority')] or available_items
            new_sequence.append(random.choice(priority_items).copy())

        elif operation == 'add_before_show':
            shows = [i for i in new_sequence if i['type'] in ['show', 'meal']]
            attractions = [i for i in available_items if i['type'] == 'attraction']
            if shows and attractions:
                target_show = random.choice(shows)
                new_attraction = random.choice(attractions).copy()
                for i, item in enumerate(new_sequence):
                    if item['type'] in ['show', 'meal'] and item['name'] == target_show['name']:
                        new_sequence.insert(i, new_attraction)
                        break

        elif operation == 'swap' and len(new_sequence) >= 2:
            non_meal_indices = [i for i, item in enumerate(new_sequence) if item['type'] != 'meal']
            if len(non_meal_indices) >= 2:
                idx1, idx2 = random.sample(non_meal_indices, 2)
                new_sequence[idx1], new_sequence[idx2] = new_sequence[idx2], new_sequence[idx1]

        elif operation == 'exchange' and new_sequence and available_items:
            non_meal_indices = [i for i, item in enumerate(new_sequence) if item['type'] != 'meal']
            if non_meal_indices:
                remove_idx = random.choice(non_meal_indices)
                new_sequence.pop(remove_idx)
                hp_items = [i for i in available_items if i.get('is_high_priority')] or available_items
                new_sequence.insert(remove_idx, random.choice(hp_items).copy())

        return new_sequence

    def _schedule_sequence(self, items, start_time, end_time, start_location):
        """Schedule sequence of items"""
        if not items:
            return []

        scheduled = []
        current_time, current_lat, current_lon = start_time, *start_location

        # Pre-schedule meals
        meal_slots = {}
        for item in items:
            if item['type'] == 'meal':
                config = MEAL_SETTINGS[item['meal_type']]
                earliest, latest = config['earliest_start'], config['latest_start']
                if item['meal_type'] == 'dinner':
                    latest = min(latest, (end_time.hour + end_time.minute / 60) - (config['duration'] / 60))
                if latest > earliest:
                    optimal_hour = (earliest + latest) / 2
                    meal_start = start_time.replace(hour=int(optimal_hour), minute=int((optimal_hour % 1) * 60),
                                                    second=0, microsecond=0)
                    meal_slots[item['meal_type']] = {
                        'start_time': meal_start, 'end_time': meal_start + timedelta(minutes=config['duration']),
                        'duration': config['duration']
                    }

        # Schedule other items
        for item in items:
            if item['type'] == 'meal':
                continue

            # Check for pending meals
            for meal_type, slot in list(meal_slots.items()):
                if current_time <= slot['start_time'] <= current_time + timedelta(hours=2):
                    scheduled.append({
                        'name': meal_type.title(), 'type': 'meal', 'arrival_time': slot['start_time'],
                        'start_time': slot['start_time'], 'wait_time': 0, 'experience_duration': slot['duration'],
                        'departure_time': slot['end_time'], 'travel_time': 0,
                        'latitude': current_lat, 'longitude': current_lon, 'satisfaction_score': 50
                    })
                    current_time = slot['end_time']
                    del meal_slots[meal_type]
                    break

            # Schedule item
            scheduled_item = self._schedule_single_item(item, current_time, current_lat, current_lon, end_time)
            if scheduled_item:
                scheduled.append(scheduled_item)
                current_time = scheduled_item['departure_time']
                current_lat, current_lon = scheduled_item['latitude'], scheduled_item['longitude']
                if current_time >= end_time:
                    break

        # Add remaining meals
        for meal_type, slot in meal_slots.items():
            if slot['end_time'] <= end_time:
                scheduled.append({
                    'name': meal_type.title(), 'type': 'meal', 'arrival_time': slot['start_time'],
                    'start_time': slot['start_time'], 'wait_time': 0, 'experience_duration': slot['duration'],
                    'departure_time': slot['end_time'], 'travel_time': 0,
                    'latitude': current_lat, 'longitude': current_lon, 'satisfaction_score': 50
                })

        return scheduled

    def _schedule_single_item(self, item, current_time, current_lat, current_lon, end_time):
        """Schedule a single item"""
        if item['type'] == 'attraction':
            data = self.attractions[self.attractions['name'] == item['name']]
            if data.empty:
                return None
            row = data.iloc[0]

            travel_distance = calculate_distance(current_lat, current_lon, row['latitude'], row['longitude'])
            travel_time = calculate_walking_time(travel_distance)
            arrival_time = current_time + timedelta(minutes=travel_time)

            predicted_wait = predict_wait_time_from_timestamp_data(row['timestamp_data'], arrival_time)
            departure_time = arrival_time + timedelta(minutes=predicted_wait + DEFAULT_RIDE_DURATION)

            if departure_time > end_time:
                return None

            satisfaction_score = row['satisfaction_score']
            if item.get('is_high_priority'):
                satisfaction_score *= SATISFACTION_SCORING['high_priority_multiplier']

            return {
                'name': item['name'], 'type': 'attraction', 'arrival_time': arrival_time, 'start_time': arrival_time,
                'wait_time': predicted_wait, 'experience_duration': DEFAULT_RIDE_DURATION,
                'departure_time': departure_time,
                'travel_time': travel_time, 'latitude': row['latitude'], 'longitude': row['longitude'],
                'satisfaction_score': satisfaction_score, 'is_high_priority': item.get('is_high_priority', False)
            }

        else:  # show
            data = self.shows[self.shows['name'] == item['name']]
            if data.empty:
                return None
            row = data.iloc[0]

            travel_distance = calculate_distance(current_lat, current_lon, row['latitude'], row['longitude'])
            travel_time = calculate_walking_time(travel_distance)
            arrival_time = current_time + timedelta(minutes=travel_time)

            show_start, show_end = item['start_time'], item['end_time']
            latest_arrival = show_start - timedelta(minutes=DEFAULT_SHOW_BUFFER)

            if arrival_time > latest_arrival or show_end > end_time:
                return None

            actual_wait = max(0, (show_start - arrival_time).total_seconds() / 60)
            satisfaction_score = row['satisfaction_score']
            if item.get('is_high_priority'):
                satisfaction_score *= SATISFACTION_SCORING['high_priority_multiplier']

            return {
                'name': item['name'], 'type': 'show', 'arrival_time': arrival_time, 'start_time': show_start,
                'wait_time': actual_wait, 'experience_duration': DEFAULT_SHOW_DURATION, 'departure_time': show_end,
                'travel_time': travel_time, 'latitude': row['latitude'], 'longitude': row['longitude'],
                'satisfaction_score': satisfaction_score, 'is_high_priority': item.get('is_high_priority', False)
            }

    def _check_constraints(self, sequence, start_time, end_time, include_lunch, include_dinner):
        """Check if sequence meets constraints"""
        if not sequence:
            return not include_lunch and not include_dinner

        # Time constraints
        for item in sequence:
            if item['arrival_time'] < start_time or item['departure_time'] > end_time:
                return False

        # Meal constraints
        if include_lunch and not any(item['type'] == 'meal' and 'lunch' in item['name'].lower() for item in sequence):
            return False
        if include_dinner and not any(item['type'] == 'meal' and 'dinner' in item['name'].lower() for item in sequence):
            return False

        return True

    def _get_available_items(self, sequence):
        """Get items that can be added"""
        current_names = {item['name'] for item in sequence if item['type'] == 'attraction'}
        current_meals = {item['meal_type'] for item in sequence if item['type'] == 'meal'}
        current_shows = {item['name'] for item in sequence if item['type'] == 'show'}

        return [item for item in self.item_pool if
                (item['type'] == 'attraction' and item['name'] not in current_names) or
                (item['type'] == 'show' and item['name'] not in current_shows) or
                (item['type'] == 'meal' and item['meal_type'] not in current_meals)]

    def _calculate_score(self, sequence):
        """Calculate sequence score"""
        if not sequence:
            return -float('inf')

        total_wait = sum(item['wait_time'] for item in sequence)
        total_travel = sum(item['travel_time'] for item in sequence)

        attraction_score = sum(100 - (
            item['satisfaction_score'] / SATISFACTION_SCORING['high_priority_multiplier'] if item.get(
                'is_high_priority') else item['satisfaction_score'])
                               * (SATISFACTION_SCORING['high_priority_multiplier'] if item.get(
            'is_high_priority') else 1)
                               for item in sequence if item['type'] == 'attraction')

        show_score = sum(item['satisfaction_score'] for item in sequence if item['type'] in ['show', 'meal'])

        return attraction_score + show_score + len(
            sequence) * 10 - total_wait * 0.1 - total_travel * 0.05 - total_wait * 0.001

    def get_summary_stats(self, sequence):
        """Get summary statistics"""
        attractions = [item['name'] for item in sequence if item['type'] == 'attraction']
        shows = [item['name'] for item in sequence if item['type'] == 'show']
        meals = [item['name'] for item in sequence if item['type'] == 'meal']

        return {
            'visited_attractions': len(attractions), 'watched_shows': len(shows), 'meals_scheduled': len(meals),
            'visited_attraction_names': attractions, 'watched_show_names': shows, 'meal_names': meals
        }


# Compatibility aliases
MonteCarloTripOptimizer = CompactMonteCarloOptimizer
WorkingMonteCarloOptimizer = CompactMonteCarloOptimizer