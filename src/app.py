import streamlit as st
import pandas as pd
import numpy as np
import random
from math import radians, sin, cos, sqrt, atan2
from sklearn.ensemble import RandomForestRegressor 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import time
import os

# ==============================================================================
# CONFIGURACIÓN DE RUTA ABSOLUTA (CORRECCIÓN IMPLEMENTADA)
# ==============================================================================
# El script se encuentra en 'src/app.py'. Usamos os para subir al directorio raíz y entrar en 'data'.
# __file__ obtiene la ruta de este script, os.path.dirname sube a 'src', y '..' sube a 'Camoteros_hackaton'.
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
DATA_PATH = os.path.join(BASE_DIR, '..', 'data') 

# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="Siemens AI Dispatch",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #FAFAFA; }
    div[data-testid="stMetric"] { background-color: #262730; border: 1px solid #4e4e4e; padding: 15px; border-radius: 5px; }
    h1, h2, h3 { color: #00ffcc !important; font-family: 'Courier New', monospace; }
    .stButton>button { color: black; background-color: #00ffcc; border: none; font-weight: bold; width: 100%; }
    .alert-green { background-color: #1b4a2e; border-left: 5px solid #00ffcc; color: #ccffeb; padding: 10px; border-radius: 5px;}
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 1. MATHEMATICS
# ==============================================================================
def calculate_haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1; dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

# ==============================================================================
# 2. TRAINING (OPTIMIZED WITH CACHE)
# ==============================================================================
#  THIS LINE IS THE SPEED MAGIC 
@st.cache_resource(show_spinner="Loading and Training Siemens AI...")
def load_and_train_model():
    
    # Load data inside cached function
    try:
        # CONSTRUCCIÓN DE RUTA ABSOLUTA
        FAULT_FILE = os.path.join(DATA_PATH, 'fault_history_10years_with_shifts.csv')
        df_faults = pd.read_csv(FAULT_FILE)
    except:
        return None, None, 0, 0, 0, None

    # Data Preparation
    df_clean = df_faults[df_faults['response_time_minutes'].notnull()].copy()
    le_day = LabelEncoder()
    df_clean['day_code'] = le_day.fit_transform(df_clean['day_of_week'])
    
    lat_center = 19.4326; lon_center = -99.1332
    df_clean['simulated_distance_km'] = df_clean.apply(
        lambda row: calculate_haversine_distance(lat_center, lon_center, row['fault_latitude'], row['fault_longitude']), axis=1
    )
    traffic_noise = np.random.uniform(0.9, 1.3, len(df_clean))
    df_clean['corrected_time'] = (df_clean['simulated_distance_km'] / 30 * 60) * traffic_noise
    
    features = ['simulated_distance_km', 'hour_of_day', 'day_code', 'fault_latitude', 'fault_longitude']
    X = df_clean[features]
    y = df_clean['corrected_time']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    n_trees = 100
    model = RandomForestRegressor(n_estimators=n_trees, random_state=42)
    model.fit(X_train, y_train) # This is where it takes time, but now only once.
    
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    model.fit(X, y) # Final Re-training
    
    return model, le_day, mae, r2, n_trees, df_faults

# ==============================================================================
# 3. PREDICTION
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
# 4. DISPATCH
# ==============================================================================
def intelligent_dispatch_ml(model, le_day, df_technicians, current_fault):
    fault_shift = current_fault['shift_at_fault']
    
    available_technicians = df_technicians[
        (df_technicians['active_shift'] == fault_shift) & 
        (df_technicians['availability'] == 'Available')
    ].copy()
    
    if available_technicians.empty:
        available_technicians = df_technicians.sample(2)

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
            'ETA (min)': round(eta_ai, 1),
            'Distance (km)': round(dist_km, 2)
        })
    
    df_results = pd.DataFrame(candidates_results)
    df_sorted = df_results.sort_values(by='ETA (min)')
    
    best_option = df_sorted.iloc[0]
    bureaucracy_time = 25
    time_saved = bureaucracy_time - best_option['ETA (min)']
    
    return {
        "Winner": best_option['Technician'],
        "ETA": round(best_option['ETA (min)'], 1),
        "Savings": round(time_saved, 1),
        "Table": df_sorted
    }

# ==============================================================================
# USER INTERFACE
# ==============================================================================

st.title("SIEMENS | Intelligent Dispatch System")
st.markdown("### Operations Control Center - AI Powered")

# --- INITIAL LOAD (ONLY ONCE) ---
# Calling cached function HERE, outside the button.
model_ai, day_encoder, mae, r2, n_trees, df_faults = load_and_train_model()

if model_ai is None:
    st.error("CRITICAL ERROR: CSV files missing. Run data generators first.")
    st.stop()

# Load technicians
try:
    # CONSTRUCCIÓN DE RUTA ABSOLUTA
    TECH_FILE = os.path.join(DATA_PATH, 'technician_inventory_dynamic.csv')
    df_technicians = pd.read_csv(TECH_FILE)
except:
    st.error("Technician inventory missing.")
    st.stop()

# --- SHOW MODEL METRICS ALWAYS ---
with st.expander("View AI Model Diagnostics (Trained & Ready)", expanded=True):
    m1, m2, m3 = st.columns(3)
    m1.metric("Precision (R²)", f"{round(r2 * 100, 2)}%")
    m2.metric("Error Margin", f"+/- {round(mae, 2)} min")
    m3.metric("Status", "OPTIMAL" if mae < 5 else "REVIEW")

# --- EXECUTION BUTTON ---
if st.button("EXECUTE REAL-TIME INCIDENT SIMULATION"):
    
    # WE DO NOT TRAIN HERE ANYMORE, JUST USE THE LOADED AI
    
    # 1. INCIDENT SIMULATION
    test_fault = df_faults[df_faults['error_code'] != 'NO_INCIDENT'].sample(1).iloc[0].to_dict()
    
    st.markdown("#### INCOMING INCIDENT ALERT")
    c1, c2, c3, c4 = st.columns(4)
    c1.error(f"**Fault:** {test_fault['error_code']}")
    c2.warning(f"**Location:** {test_fault['base_location']}")
    c3.info(f"**Time:** {test_fault['hour_of_day']}:00")
    c4.info(f"**Shift:** {test_fault['shift_at_fault']}")

    # 2. DISPATCH EXECUTION (ONLY THING CALCULATED ON CLICK)
    result = intelligent_dispatch_ml(model_ai, day_encoder, df_technicians, test_fault)

    # 3. RESULTS
    st.markdown("---")
    col_left, col_right = st.columns([2, 3])
    
    with col_left:
        st.markdown("### AI DECISION")
        st.success(f"**Assigned:** {result['Winner']}")
        st.info(f"**Arrival:** {result['ETA']} min")
        st.metric("TIME SAVED", f"{result['Savings']} min", delta="Optimized")

    with col_right:
        st.markdown(f"#### Comparative Analysis ({test_fault['shift_at_fault']})")
        st.dataframe(
            result['Table'].style.highlight_min(axis=0, subset=['ETA (min)'], color='#1b4a2e'),
            use_container_width=True,
            hide_index=True
        )