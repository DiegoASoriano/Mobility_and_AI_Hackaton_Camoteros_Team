import pandas as pd
import numpy as np
import random
from math import radians, sin, cos, sqrt, atan2
from sklearn.ensemble import RandomForestRegressor 
from sklearn.preprocessing import LabelEncoder
# METRICS LIBRARIES
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import os # <--- ADDED: Necessary to locate files from the models folder

# ==============================================================================
# 1. MATHEMATICS (Real Distance Calculation)
# ==============================================================================
def calculate_haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1; dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

# ==============================================================================
# 2. AI TRAINING AND DIAGNOSTICS
# ==============================================================================
def train_and_evaluate_model(df_history):
    print(" STARTING AI TRAINING AND DIAGNOSTICS...")
    
    # Data Preparation
    df_clean = df_history[df_history['response_time_minutes'].notnull()].copy()
    le_day = LabelEncoder()
    df_clean['day_code'] = le_day.fit_transform(df_clean['day_of_week'])
    
    # Feature Engineering
    lat_center = 19.4326
    lon_center = -99.1332
    df_clean['simulated_distance_km'] = df_clean.apply(
        lambda row: calculate_haversine_distance(lat_center, lon_center, row['fault_latitude'], row['fault_longitude']), axis=1
    )
    traffic_noise = np.random.uniform(0.9, 1.3, len(df_clean))
    df_clean['corrected_time'] = (df_clean['simulated_distance_km'] / 30 * 60) * traffic_noise
    
    features = ['simulated_distance_km', 'hour_of_day', 'day_code', 'fault_latitude', 'fault_longitude']
    X = df_clean[features]
    y = df_clean['corrected_time']
    
    # --- AI EXAM (Train/Test Split) ---
    # Split 80% for training and keep 20% for validation (Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model Configuration (Define trees here)
    n_trees = 100
    model = RandomForestRegressor(n_estimators=n_trees, random_state=42)
    
    # Train only with TRAIN set
    model.fit(X_train, y_train)
    
    # Make predictions on TEST set (Data the AI has never seen)
    predictions = model.predict(X_test)
    
    # --- METRICS CALCULATION ---
    mae = mean_absolute_error(y_test, predictions) # Average error in minutes
    r2 = r2_score(y_test, predictions) # Global precision (0 to 1)
    
    print(f"\n MODEL PERFORMANCE REPORT (Diagnostics):")
    print(f"   ---------------------------------------------")
    print(f"   > Decision Trees: {n_trees}")
    print(f"   > Global Precision (R²): {round(r2 * 100, 2)}%")
    print(f"   > Margin of Error (MAE): +/- {round(mae, 2)} minutes")
    
    if mae < 5.0:
        print("    STATUS: OPTIMAL (Error is very low, 100 trees are sufficient).")
    else:
        print("    STATUS: REVIEW (Consider increasing trees or improving data).")
    print(f"   ---------------------------------------------\n")
    
    # Re-train with ALL dataset for production use
    model.fit(X, y)
    
    return model, le_day

# ==============================================================================
# 3. INDIVIDUAL PREDICTION
# ==============================================================================
def predict_eta_with_ai(model, le_day, lat_tech, lon_tech, lat_fault, lon_fault, hour, day_text):
    real_distance = calculate_haversine_distance(lat_tech, lon_tech, lat_fault, lon_fault)
    try:
        day_code = le_day.transform([day_text])[0] 
    except:
        day_code = 0 
        
    input_data = pd.DataFrame({
        'simulated_distance_km': [real_distance],
        'hour_of_day': [hour],
        'day_code': [day_code],
        'fault_latitude': [lat_fault],
        'fault_longitude': [lon_fault]
    })
    
    predicted_time = model.predict(input_data)[0]
    if predicted_time < 2.0: predicted_time = 2.5 + random.uniform(0, 1)
    return predicted_time, real_distance

# ==============================================================================
# 4. DISPATCH LOGIC
# ==============================================================================
def intelligent_dispatch_ml(model, le_day, df_technicians, current_fault):
    fault_shift = current_fault['shift_at_fault']
    
    available_technicians = df_technicians[
        (df_technicians['active_shift'] == fault_shift) & 
        (df_technicians['availability'] == 'Available')
    ].copy()
    
    if available_technicians.empty:
        available_technicians = df_technicians.sample(2) # Fallback demo

    candidates_results = []
    
    for index, tech in available_technicians.iterrows():
        eta_ai, dist_km = predict_eta_with_ai(
            model, le_day,
            tech['initial_latitude'], tech['initial_longitude'],
            current_fault['fault_latitude'], current_fault['fault_longitude'],
            current_fault['hour_of_day'], current_fault['day_of_week']
        )
        candidates_results.append({
            'Technician': tech['name'],
            'ETA (min)': eta_ai,
            'Distance (km)': dist_km
        })
    
    df_results = pd.DataFrame(candidates_results)
    df_sorted = df_results.sort_values(by='ETA (min)')
    
    print(f" CANDIDATE ANALYSIS - SHIFT: {fault_shift}")
    print(f"{'OPERATOR':<15} | {'DISTANCE':<12} | {'PREDICTED ETA':<15}")
    print(f"{'-'*15} | {'-'*12} | {'-'*15}")
    
    for index, row in df_sorted.iterrows():
        name = row['Technician']
        dist = f"{row['Distance (km)']:.2f} km"
        time = f"{row['ETA (min)']:.1f} min"
        marker = " WIN" if index == df_sorted.index[0] else "   "
        print(f"{name:<15} | {dist:<12} | {time:<10} {marker}")
    print(f"{'='*60}")
    
    best_option = df_sorted.iloc[0]
    bureaucracy_time = 25
    time_saved = bureaucracy_time - best_option['ETA (min)']
    
    return {
        "Winner": best_option['Technician'],
        "ETA": round(best_option['ETA (min)'], 1),
        "Savings": round(time_saved, 1)
    }

# ==============================================================================
# 5. EXECUTION
# ==============================================================================
if __name__ == "__main__":
    try:
        # --- PATH FIX START (For 'models' folder) ---
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(current_dir, '..', 'data')
        
        df_technicians = pd.read_csv(os.path.join(data_dir, 'technician_inventory_dynamic.csv'))
        df_faults = pd.read_csv(os.path.join(data_dir, 'fault_history_10years_with_shifts.csv'))
        # --- PATH FIX END ---
        
    except FileNotFoundError:
        print(" ERROR: Missing CSV files. Please check the 'data' folder.")
        exit()

    # TRAINING AND EVALUATION
    ai_model, day_encoder = train_and_evaluate_model(df_faults)

    # SIMULATION
    test_fault = df_faults[df_faults['error_code'] != 'NO_INCIDENT'].sample(1).iloc[0].to_dict()
    
    print(f"\n INCOMING INCIDENT:")
    print(f"   Fault: {test_fault['error_code']} @ {test_fault['base_location']}")
    
    result = intelligent_dispatch_ml(ai_model, day_encoder, df_technicians, test_fault)

    print(f"\n FINAL RESULT:")
    print(f"   Operator: {result['Winner']} | Arrival in: {result['ETA']} min")
    print(f"   TIME SAVED: {result['Savings']} min ")