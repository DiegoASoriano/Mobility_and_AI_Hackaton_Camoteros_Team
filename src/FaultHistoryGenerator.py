import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# 1. SIMULATION CONFIGURATION
# These variables control the dataset scale, geography, and fault frequency.

NUM_YEARS = 10                  # Number of years of historical data to generate.
LINE_LENGTH_KM = 22.6           # Simulated total line length in kilometers.
NUM_STATIONS = 21               # Number of stations on the metro line.

# KEY VALUE 1: Daily Fault Frequency (λ)
LAMBDA_DAILY_FAULTS = 4.5 

# Probability of a non-fault day (1 - 0.85 = 0.15 clean day probability)
DAILY_FAULT_PROBABILITY = 0.85 

# LOGISTICS KEY VARIABLES
NUM_SHIFTS = 3                  # Defines the number of maintenance shifts (e.g., 2 or 3).
OPERATING_HOURS_START = 6       # Operation start time (e.g., 6 AM)
OPERATING_HOURS_END = 23        # Operation end time (e.g., 11 PM)

# KEY VALUE 2: Fault Type Probability (Weighting)
# Defines the relative probability (weight) of each error code. 
FAULT_TYPE_PROBABILITY = {
    'E-401 (Electrical)': 45,  # Most probable
    'M-205 (Mechanical)': 30,  
    'S-101 (Signaling)': 15, 
    'H-300 (Hydraulic)': 10    # Least probable
}

# Reference Coordinates for CDMX (simulated track)
LAT_START = 19.45
LON_START = -99.15
LAT_END = 19.35
LON_END = -99.08

# 2. DATA PREPARATION, FUNCTIONS, AND SHIFT LOGIC

# Extract labels and weights for random.choices()
error_codes = list(FAULT_TYPE_PROBABILITY.keys())
weights = list(FAULT_TYPE_PROBABILITY.values())

# Station coordinate generation
station_names = [f'Station_{i+1}' for i in range(NUM_STATIONS)]
lats = np.linspace(LAT_START, LAT_END, NUM_STATIONS)
lons = np.linspace(LON_START, LON_END, NUM_STATIONS)
station_coords = {name: (lats[i], lons[i]) for i, name in enumerate(station_names)}


def get_shift_boundaries(num_shifts, start_hour, end_hour):
    """Calculates the hour limits for each shift based on the number of shifts."""
    total_hours = end_hour - start_hour + 1 
    hours_per_shift = total_hours // num_shifts
    
    boundaries = {}
    current_start = start_hour
    
    for i in range(num_shifts):
        shift_name = f'Shift_{i+1}'
        current_end = current_start + hours_per_shift - 1
        
        # The last shift covers the remaining hours until closing (23h)
        if i == num_shifts - 1:
            current_end = end_hour
            
        boundaries[shift_name] = (current_start, current_end)
        current_start = current_end + 1
        
    return boundaries

SHIFT_BOUNDARIES = get_shift_boundaries(NUM_SHIFTS, OPERATING_HOURS_START, OPERATING_HOURS_END)

def get_shift_from_hour(hour):
    """Assigns the shift name (Shift_1, Shift_2, etc.) based on the hour."""
    for shift, (start, end) in SHIFT_BOUNDARIES.items():
        if start <= hour <= end:
            return shift
    # For any time outside of operation
    return "OUT_OF_SERVICE" 

# 3. FAULT HISTORY GENERATION (10 YEARS)
# Iterates day by day to simulate random incident occurrence.

start_date = datetime.strptime('2015-01-01', '%Y-%m-%d')
end_date = start_date + timedelta(days=365 * NUM_YEARS + 2)
date_list = pd.date_range(start_date, end_date).tolist()

all_faults = []
fault_id = 10000

print(f"Generating {NUM_YEARS} years of data with {NUM_SHIFTS} shifts.")
print(f"Calculated Shift Boundaries: {SHIFT_BOUNDARIES}")

for current_date in date_list:
    
    num_faults_today = 0
    
    # Daily Decision: Will this day have faults?
    if random.random() <= DAILY_FAULT_PROBABILITY:
        
        # Frequency: Use Poisson for the number of faults (randomness without fixed limits)
        num_faults_today = np.random.poisson(LAMBDA_DAILY_FAULTS) 
        if num_faults_today == 0:
            num_faults_today = 1 
            
    # Loop to generate details for each fault (if num_faults_today > 0)
    if num_faults_today > 0:
        for _ in range(num_faults_today):
            
            # 1. Fault type assignment (E-401, M-205, etc.) based on weights
            error_code = random.choices(error_codes, weights=weights, k=1)[0]
            
            # 2. Fault location (coordinates)
            base_location = random.choice(station_names)
            base_lat, base_lon = station_coords[base_location]
            fault_lat = round(base_lat + random.uniform(-0.001, 0.001), 6)
            fault_lon = round(base_lon + random.uniform(-0.001, 0.001), 6)
            
            # 3. Fault time (random within operating hours)
            fault_time = current_date + timedelta(hours=random.randint(OPERATING_HOURS_START, OPERATING_HOURS_END), minutes=random.randint(0, 59))
            
            # 4. Shift assignment based on fault time
            shift_at_fault = get_shift_from_hour(fault_time.hour)
            
            fault_id += 1
            all_faults.append({
                'fault_id': fault_id,
                'timestamp': fault_time.strftime('%Y-%m-%d %H:%M:%S'),
                'day_of_week': fault_time.strftime('%A'),
                'hour_of_day': fault_time.hour,
                'shift_at_fault': shift_at_fault, # New field: The shift when the fault occurred
                'base_location': base_location,
                'fault_latitude': fault_lat,
                'fault_longitude': fault_lon,
                'error_code': error_code, 
                # Simulates the CURRENT response time (the target to improve)
                'response_time_minutes': random.randint(25, 45) 
            })
    
    # LOGIC: NO INCIDENT DAY
    else:
        # Adds a row for a day without faults. Numeric values are None.
        all_faults.append({
            'fault_id': None,
            'timestamp': current_date.strftime('%Y-%m-%d 00:00:00'),
            'day_of_week': current_date.strftime('%A'),
            'hour_of_day': None,
            'shift_at_fault': "NO_INCIDENT", # Shift marker
            'base_location': "No Incident Day",
            'fault_latitude': None,
            'fault_longitude': None,
            'error_code': "NO_INCIDENT",
            'response_time_minutes': None 
        })

df_fault_history = pd.DataFrame(all_faults)

# 4. EXPORT

file_name = 'data/fault_history_10years_with_shifts.csv'
df_fault_history.to_csv(file_name, index=False)

print(f"\n Fault History successfully generated (includes the fault shift).")
print(f"Saved to: {file_name}")