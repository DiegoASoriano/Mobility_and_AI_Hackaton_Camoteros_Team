import streamlit as st
import pandas as pd
import numpy as np
import random
import pydeck as pdk
from math import radians, sin, cos, sqrt, atan2
from sklearn.ensemble import RandomForestRegressor 
from sklearn.preprocessing import LabelEncoder
import time

# ==============================================================================
# CONFIGURACIÓN DE PÁGINA MÓVIL
# ==============================================================================
st.set_page_config(
    page_title="Siemens FieldOps",
    page_icon="🔒",
    initial_sidebar_state="collapsed"
)

# CSS PARA ESTILO DE APP NATIVA (LOGIN + DARK MODE)
st.markdown("""
<style>
    .stApp { background-color: #000000; color: #FAFAFA; }
    
    /* Estilo del Login */
    .login-box {
        background-color: #1e1e1e;
        padding: 30px;
        border-radius: 20px;
        border: 1px solid #333;
        text-align: center;
        margin-top: 50px;
        box-shadow: 0 10px 30px rgba(0, 255, 204, 0.1);
    }
    
    /* Tarjetas de la App */
    .mobile-card {
        background-color: #1e1e1e;
        border-radius: 15px;
        padding: 20px;
        margin-bottom: 20px;
        border: 1px solid #333;
    }
    
    /* Botones Grandes */
    .stButton>button { 
        width: 100%; 
        border-radius: 25px; 
        height: 55px; 
        font-size: 18px !important;
        font-weight: bold;
        background-color: #00ffcc; 
        color: black; 
        border: none;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #00ccaa;
        transform: scale(1.02);
    }
    
    /* Input Fields */
    .stSelectbox > div > div {
        background-color: #2d2d2d;
        color: white;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 1. LÓGICA DE NEGOCIO (IA + DATOS)
# ==============================================================================
def calculate_haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    a = sin((lat2-lat1)/2)**2 + cos(lat1) * cos(lat2) * sin((lon2-lon1)/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

@st.cache_resource
def load_system_data():
    try:
        # CARGA DINÁMICA: Lee el archivo cada vez que se inicia para actualizar la lista
        df_faults = pd.read_csv('fault_history_10years_with_shifts.csv')
        df_techs = pd.read_csv('technician_inventory_dynamic.csv')
    except:
        return None, None, None, None

    # Entrenamiento Ligero para la App
    df_clean = df_faults[df_faults['response_time_minutes'].notnull()].copy()
    le_day = LabelEncoder()
    df_clean['day_code'] = le_day.fit_transform(df_clean['day_of_week'])
    
    lat_c, lon_c = 19.4326, -99.1332
    df_clean['sim_dist'] = df_clean.apply(lambda x: calculate_haversine_distance(lat_c, lon_c, x['fault_latitude'], x['fault_longitude']), axis=1)
    df_clean['target_time'] = (df_clean['sim_dist'] / 30 * 60) * np.random.uniform(0.9, 1.3, len(df_clean))
    
    features = ['sim_dist', 'hour_of_day', 'day_code', 'fault_latitude', 'fault_longitude']
    model = RandomForestRegressor(n_estimators=50, random_state=42) 
    model.fit(df_clean[features], df_clean['target_time'])
    
    return model, le_day, df_faults, df_techs

# ==============================================================================
# 2. GESTIÓN DE ESTADO (SESIÓN)
# ==============================================================================
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user_info' not in st.session_state:
    st.session_state.user_info = None
if 'job_active' not in st.session_state:
    st.session_state.job_active = False

# Cargar cerebro
model, le_day, df_faults, df_techs = load_system_data()

if model is None:
    st.error("⚠️ SERVER ERROR: Database connection failed.")
    st.stop()

# ==============================================================================
# 3. VISTA 1: LOGIN SCREEN (PANTALLA DE INICIO)
# ==============================================================================
if not st.session_state.logged_in:
    
    # Espacio superior
    st.write("")
    st.write("")
    
    # Logo
    c1, c2, c3 = st.columns([1,2,1])
    with c2:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/5/5f/Siemens-logo.svg/2560px-Siemens-logo.svg.png")
    
    st.markdown("""
    <div class='login-box'>
        <h2 style='color: #00ffcc; margin-bottom: 0;'>FIELD OPERATIONS</h2>
        <p style='color: #888; font-size: 14px;'>Secure Access Portal</p>
        <hr style='border-color: #333;'>
    </div>
    """, unsafe_allow_html=True)
    
    # --- FORMULARIO DE LOGIN DINÁMICO ---
    with st.form("login_form"):
        st.write("#### 🆔 Identify Yourself")
        
        # LISTA DINÁMICA: Se llena directo del CSV
        # Creamos una lista bonita tipo "ID: 100 - Technician_1"
        tech_options = df_techs.apply(lambda x: f"{x['technician_id']} - {x['name']} ({x['active_shift']})", axis=1).tolist()
        
        selected_login = st.selectbox("Select Technician ID", tech_options)
        
        # Password simulada (siempre correcta para el hackathon)
        password = st.text_input("Access PIN", type="password", value="1234")
        
        submit_btn = st.form_submit_button("🔓 LOGIN TO SYSTEM")
        
        if submit_btn:
            # Extraer el ID del string seleccionado
            tech_id_selected = int(selected_login.split(" - ")[0])
            
            # Guardar info del usuario en la sesión
            user_data = df_techs[df_techs['technician_id'] == tech_id_selected].iloc[0]
            st.session_state.user_info = user_data
            st.session_state.logged_in = True
            
            with st.spinner("Authenticating..."):
                time.sleep(1)
            st.rerun()

# ==============================================================================
# 4. VISTA 2: APLICACIÓN PRINCIPAL (SOLO SI ESTÁ LOGUEADO)
# ==============================================================================
else:
    # Datos del usuario logueado
    my_data = st.session_state.user_info
    
    # --- HEADER DE LA APP ---
    col_a, col_b = st.columns([3, 1])
    with col_a:
        st.markdown(f"### Hi, {my_data['name'].split('_')[1]}") # Muestra solo el nombre corto
        st.caption(f"ID: {my_data['technician_id']} | {my_data['active_shift']}")
    with col_b:
        if st.button("LOGOUT", key="logout_btn"):
            st.session_state.logged_in = False
            st.rerun()

    st.markdown("---")

    # --- VISTA 2A: ESPERANDO TRABAJO ---
    if not st.session_state.job_active:
        st.markdown(f"""
        <div class="mobile-card" style="text-align: center; border-top: 5px solid #00ffcc;">
            <h1 style="font-size: 50px;">🟢</h1>
            <h3>ONLINE</h3>
            <p>Connected to AI Dispatch Grid</p>
            <p style="color: #888; font-size: 12px;">GPS Location: {my_data['initial_latitude']}, {my_data['initial_longitude']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Botón para simular la llegada de una notificación
        st.markdown("### 📡 Actions")
        if st.button("🔄 CHECK FOR ASSIGNMENTS"):
            with st.spinner("Syncing with Control Center..."):
                time.sleep(1.5)
                
                # Generar falla aleatoria
                new_fault = df_faults[df_faults['error_code'] != 'NO_INCIDENT'].sample(1).iloc[0]
                
                # Calcular ETA Personalizado
                dist_km = calculate_haversine_distance(
                    my_data['initial_latitude'], my_data['initial_longitude'],
                    new_fault['fault_latitude'], new_fault['fault_longitude']
                )
                
                try: day_c = le_day.transform([new_fault['day_of_week']])[0]
                except: day_c = 0
                
                input_data = [[dist_km, new_fault['hour_of_day'], day_c, new_fault['fault_latitude'], new_fault['fault_longitude']]]
                my_eta = model.predict(input_data)[0]
                
                # Guardar estado
                st.session_state.current_job = new_fault
                st.session_state.my_eta = round(my_eta, 1)
                st.session_state.dist = round(dist_km, 2)
                st.session_state.job_active = True
                
                st.toast("⚠️ NEW PRIORITY INCIDENT RECEIVED!", icon="🚨")
                time.sleep(0.5)
                st.rerun()

    # --- VISTA 2B: TRABAJO ACTIVO (NAVEGACIÓN) ---
    else:
        job = st.session_state.current_job
        
        # TARJETA DE ALERTA ROJA
        st.markdown(f"""
        <div class="mobile-card" style="border-left: 8px solid #ff4b4b; background-color: #2a1a1a;">
            <h4 style="color: #ff4b4b; margin:0;">🚨 PRIORITY DISPATCH</h4>
            <h1 style="font-size: 32px; margin: 10px 0;">{job['error_code']}</h1>
            <p>📍 <b>Target:</b> {job['base_location']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # TARJETA DE LOGÍSTICA IA
        st.markdown(f"""
        <div class="mobile-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div style="text-align: center;">
                    <span style="font-size: 12px; color: #888;">AI ESTIMATE</span>
                    <h2 style="color: #00ffcc; margin: 0;">{st.session_state.my_eta} min</h2>
                </div>
                <div style="font-size: 20px; color: #444;">|</div>
                <div style="text-align: center;">
                    <span style="font-size: 12px; color: #888;">DISTANCE</span>
                    <h2 style="color: white; margin: 0;">{st.session_state.dist} km</h2>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # MAPA DE NAVEGACIÓN (PYDECK)
        st.write("**🗺️ Navigation Route**")
        
        tech_loc = [my_data['initial_longitude'], my_data['initial_latitude']]
        fault_loc = [job['fault_longitude'], job['fault_latitude']]
        mid_lat = (my_data['initial_latitude'] + job['fault_latitude']) / 2
        mid_lon = (my_data['initial_longitude'] + job['fault_longitude']) / 2

        layer_line = pdk.Layer(
            "LineLayer",
            data=[{"source": tech_loc, "target": fault_loc}],
            get_source_position="source", get_target_position="target",
            get_color=[0, 255, 204], get_width=6,
        )
        layer_points = pdk.Layer(
            "ScatterplotLayer",
            data=[
                {"pos": tech_loc, "color": [0, 255, 204, 255], "radius": 120},
                {"pos": fault_loc, "color": [255, 75, 75, 255], "radius": 120},
            ],
            get_position="pos", get_color="color", get_radius="radius",
        )

        view_state = pdk.ViewState(latitude=mid_lat, longitude=mid_lon, zoom=12.5, pitch=0)
        st.pydeck_chart(pdk.Deck(layers=[layer_line, layer_points], initial_view_state=view_state, tooltip={"text": "Route"}))
        
        st.write("")
        
        # BOTONES DE ACCIÓN
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🚀 GO"):
                st.toast("Opening Maps...", icon="🗺️")
        with c2:
            if st.button("📞 CALL"):
                st.toast("Calling HQ...", icon="📞")
                
        st.write("")
        if st.button("✅ COMPLETE JOB"):
            st.session_state.job_active = False
            st.balloons() # Efecto de celebración al terminar
            st.success("Report sent successfully!")
            time.sleep(2)
            st.rerun()