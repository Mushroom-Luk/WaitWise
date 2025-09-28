# API Configuration
THEMEPARKS_BASE_URL = "https://api.themeparks.wiki/v1/entity"
HKDL_DESTINATION_ID = "abcfffe7-01f2-4f92-ae61-5093346f5a68"
QUEUE_TIMES_BASE_URL = "https://queue-times.com"
QUEUE_TIMES_PARK_ID = 31

# Exclusions and Settings
EXCLUDE_KEYWORDS = ['meet', 'character encounter', 'photo', 'restaurant', 'shop', 'atmosphere entertainment']
SATISFACTION_SCORING = {'show_multiplier': 3.0, 'high_priority_multiplier': 4.0}
WALKING_SPEED = 80  # meters per minute
DEFAULT_RIDE_DURATION = 15
DEFAULT_SHOW_DURATION = 30
DEFAULT_SHOW_BUFFER = 10

# Meal and Optimization Settings
MEAL_SETTINGS = {
    'lunch': {'duration': 90, 'earliest_start': 12, 'latest_start': 14},
    'dinner': {'duration': 90, 'earliest_start': 17.5, 'latest_start': 20}
}
COORDINATE_CORRECTION = {'lat_offset': -0.003001285, 'lon_offset': -0.011673842}
MONTE_CARLO_SETTINGS = {'early_stop_threshold': 200, 'progress_update_interval': 50}