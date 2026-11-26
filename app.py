import os
import sys
import datetime
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium

# ==========================================
# 1. PAGE CONFIGURATION & THEME STYLING
# ==========================================
st.set_page_config(
    page_title="Uber Fare Estimator | NYC",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for authentic Uber styling:
# High-contrast typography, crisp borders, and full light/dark readability.
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }

    /* Container Spacing */
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2.5rem;
        max-width: 1280px;
    }

    /* Uber Navbar Header */
    .uber-navbar {
        background-color: #000000;
        color: #FFFFFF;
        padding: 18px 28px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 24px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.12);
    }
    .uber-logo {
        font-size: 32px;
        font-weight: 800;
        letter-spacing: -1px;
        color: #FFFFFF;
        line-height: 1;
    }
    .uber-subtitle {
        font-size: 12px;
        color: #CCCCCC;
        font-weight: 600;
        letter-spacing: 0.5px;
        margin-top: 4px;
    }
    .uber-nav-pills {
        display: flex;
        gap: 10px;
    }
    .uber-pill {
        background: #222222;
        color: #FFFFFF;
        font-size: 13px;
        font-weight: 600;
        padding: 6px 14px;
        border-radius: 20px;
        border: 1px solid #444444;
    }
    .uber-pill.active {
        background: #FFFFFF;
        color: #000000;
        border-color: #FFFFFF;
    }

    /* Fare Hero Card */
    .fare-hero {
        background-color: #000000;
        color: #FFFFFF;
        border-radius: 14px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.18);
        border: 1px solid #222222;
    }
    .fare-hero-label {
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        color: #E0E0E0;
        font-weight: 700;
    }
    .fare-hero-price {
        font-size: 48px;
        font-weight: 800;
        color: #FFFFFF;
        letter-spacing: -1px;
        line-height: 1.1;
        margin: 4px 0 8px 0;
    }
    .fare-hero-route {
        font-size: 15px;
        color: #FFFFFF;
        font-weight: 600;
    }
    .fare-pills-row {
        margin-top: 14px;
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
    }
    .fare-stat-pill {
        background: #1C1C1E;
        color: #FFFFFF;
        border: 1px solid #3A3A3C;
        padding: 6px 12px;
        border-radius: 8px;
        font-size: 12px;
        font-weight: 600;
    }

    /* Ride Options Tier Cards */
    .ride-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 14px 18px;
        border: 1px solid #E5E7EB;
        border-radius: 12px;
        margin-bottom: 10px;
        background-color: #FFFFFF;
        transition: all 0.2s ease;
    }
    .ride-row:hover {
        border-color: #000000;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.06);
    }
    .ride-row.active-tier {
        border: 2px solid #000000;
        background-color: #F9FAFB;
    }
    .ride-info-left {
        display: flex;
        align-items: center;
        gap: 14px;
    }
    .ride-emoji {
        font-size: 28px;
    }
    .ride-name {
        font-size: 16px;
        font-weight: 700;
        color: #111827;
    }
    .ride-desc {
        font-size: 13px;
        color: #4B5563;
        font-weight: 500;
        margin-top: 2px;
    }
    .ride-price-right {
        text-align: right;
    }
    .ride-price-val {
        font-size: 20px;
        font-weight: 800;
        color: #111827;
    }
    .ride-time-est {
        font-size: 12px;
        color: #2563EB;
        font-weight: 600;
        margin-top: 2px;
    }

    /* Primary Uber Buttons */
    div.stButton > button:first-child {
        background-color: #000000 !important;
        color: #FFFFFF !important;
        font-size: 16px !important;
        font-weight: 700 !important;
        border-radius: 10px !important;
        border: none !important;
        padding: 14px 28px !important;
        width: 100% !important;
        box-shadow: 0 4px 14px rgba(0,0,0,0.18) !important;
        transition: all 0.2s ease !important;
    }
    div.stButton > button:first-child:hover {
        background-color: #262626 !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(0,0,0,0.25) !important;
    }

    /* Section Subheadings */
    .section-label {
        font-size: 14px;
        font-weight: 700;
        margin-top: 14px;
        margin-bottom: 4px;
        color: inherit;
        display: flex;
        align-items: center;
        gap: 6px;
    }

    /* Instruction Callout */
    .map-helper {
        background: #F0FDF4;
        border: 1px solid #BBF7D0;
        color: #166534;
        padding: 8px 14px;
        border-radius: 8px;
        font-size: 12px;
        font-weight: 600;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# 2. MODEL LOADING & CACHING
# ==========================================
@st.cache_resource
def load_trained_model():
    """
    Search and load the serialized scikit-learn model from candidate directories.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(current_dir, 'models', 'best_model.joblib'),
        os.path.join(current_dir, 'models', 'uber_fare_model.joblib'),
        os.path.join(current_dir, 'best_model.joblib'),
        os.path.join(current_dir, 'uber_fare_model.joblib'),
        os.path.join(os.path.dirname(current_dir), 'models', 'best_model.joblib'),
        os.path.join(os.path.dirname(current_dir), 'uber-fare-predictor', 'models', 'best_model.joblib'),
        'models/best_model.joblib',
        'best_model.joblib'
    ]

    for path in candidates:
        if os.path.exists(path):
            try:
                model = joblib.load(path)
                return model, path
            except Exception:
                continue
    return None, None


# ==========================================
# 3. ROUTE PRESETS & HAVERSINE FORMULA
# ==========================================
PRESETS = {
    "🗽 Times Square ➔ JFK International Airport": {
        "pickup_name": "Times Square, Manhattan",
        "dropoff_name": "JFK Airport Terminal 4, Queens",
        "p_lon": -73.9855,
        "p_lat": 40.7580,
        "d_lon": -73.7781,
        "d_lat": 40.6413,
        "default_passengers": 2
    },
    "🏙️ Wall Street (Financial District) ➔ Central Park": {
        "pickup_name": "Wall St & Broad St, Financial District",
        "dropoff_name": "Central Park South, Midtown",
        "p_lon": -74.0090,
        "p_lat": 40.7060,
        "d_lon": -73.9680,
        "d_lat": 40.7850,
        "default_passengers": 1
    },
    "✈️ LaGuardia Airport (LGA) ➔ Grand Central Terminal": {
        "pickup_name": "LaGuardia Airport (LGA), Queens",
        "dropoff_name": "Grand Central Terminal, Midtown",
        "p_lon": -73.8740,
        "p_lat": 40.7769,
        "d_lon": -73.9772,
        "d_lat": 40.7527,
        "default_passengers": 1
    },
    "🌉 Brooklyn Bridge (DUMBO) ➔ SoHo": {
        "pickup_name": "Brooklyn Bridge Park, DUMBO",
        "dropoff_name": "Broadway & Prince St, SoHo",
        "p_lon": -73.9934,
        "p_lat": 40.7033,
        "d_lon": -73.9997,
        "d_lat": 40.7233,
        "default_passengers": 2
    },
    "🏛️ Empire State Building ➔ Metropolitan Museum of Art": {
        "pickup_name": "Empire State Building, Midtown",
        "dropoff_name": "The Metropolitan Museum of Art (The Met)",
        "p_lon": -73.9857,
        "p_lat": 40.7484,
        "d_lon": -73.9632,
        "d_lat": 40.7794,
        "default_passengers": 1
    },
    "⚾ Penn Station ➔ Yankee Stadium (The Bronx)": {
        "pickup_name": "Penn Station, Manhattan",
        "dropoff_name": "Yankee Stadium, The Bronx",
        "p_lon": -73.9935,
        "p_lat": 40.7505,
        "d_lon": -73.9262,
        "d_lat": 40.8296,
        "default_passengers": 3
    },
    "📍 Custom / Map Click Selection": {
        "pickup_name": "Selected Pickup Location",
        "dropoff_name": "Selected Dropoff Location",
        "p_lon": -73.9851,
        "p_lat": 40.7488,
        "d_lon": -73.9626,
        "d_lat": 40.7726,
        "default_passengers": 1
    }
}


def calculate_haversine(lat1, lon1, lat2, lon2):
    """
    Computes great-circle distance between two GPS coordinates in kilometers.
    """
    lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
    lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0)**2
    a = np.clip(a, 0.0, 1.0)
    return 6371.0 * 2.0 * np.arcsin(np.sqrt(a))


# ==========================================
# 4. SESSION STATE INITIALIZATION & MAP CLICK HANDLING
# ==========================================
default_preset = "🗽 Times Square ➔ JFK International Airport"
p_init = PRESETS[default_preset]

if "pickup_lat" not in st.session_state:
    st.session_state["pickup_lat"] = float(p_init["p_lat"])
if "pickup_lon" not in st.session_state:
    st.session_state["pickup_lon"] = float(p_init["p_lon"])
if "pickup_label" not in st.session_state:
    st.session_state["pickup_label"] = str(p_init["pickup_name"])

if "dropoff_lat" not in st.session_state:
    st.session_state["dropoff_lat"] = float(p_init["d_lat"])
if "dropoff_lon" not in st.session_state:
    st.session_state["dropoff_lon"] = float(p_init["d_lon"])
if "dropoff_label" not in st.session_state:
    st.session_state["dropoff_label"] = str(p_init["dropoff_name"])

if "passenger_count" not in st.session_state:
    st.session_state["passenger_count"] = int(p_init["default_passengers"])

if "preset_choice" not in st.session_state:
    st.session_state["preset_choice"] = default_preset

if "last_handled_click" not in st.session_state:
    st.session_state["last_handled_click"] = None

if "map_click_mode" not in st.session_state:
    st.session_state["map_click_mode"] = "📍 Pickup Location (Green)"

# Check if a map click occurred from st_folium in previous rerun
if "uber_map" in st.session_state and st.session_state["uber_map"]:
    click_data = st.session_state["uber_map"].get("last_clicked")
    if click_data is not None and click_data != st.session_state["last_handled_click"]:
        st.session_state["last_handled_click"] = click_data
        c_lat = round(float(click_data["lat"]), 4)
        c_lon = round(float(click_data["lng"]), 4)

        if "Pickup" in st.session_state["map_click_mode"]:
            st.session_state["pickup_lat"] = c_lat
            st.session_state["pickup_lon"] = c_lon
            st.session_state["pickup_label"] = f"Map Pin ({c_lat:.4f}, {c_lon:.4f})"
            st.session_state["preset_choice"] = "📍 Custom / Map Click Selection"
            st.toast(f"📍 Pickup coordinates set to ({c_lat:.4f}, {c_lon:.4f})", icon="🟢")
        else:
            st.session_state["dropoff_lat"] = c_lat
            st.session_state["dropoff_lon"] = c_lon
            st.session_state["dropoff_label"] = f"Map Pin ({c_lat:.4f}, {c_lon:.4f})"
            st.session_state["preset_choice"] = "📍 Custom / Map Click Selection"
            st.toast(f"🏁 Dropoff coordinates set to ({c_lat:.4f}, {c_lon:.4f})", icon="🔴")


# Callback when user selects a different preset dropdown option
def handle_preset_change():
    chosen = st.session_state["preset_choice"]
    if chosen in PRESETS and chosen != "📍 Custom / Map Click Selection":
        data = PRESETS[chosen]
        st.session_state["pickup_lat"] = float(data["p_lat"])
        st.session_state["pickup_lon"] = float(data["p_lon"])
        st.session_state["pickup_label"] = str(data["pickup_name"])
        st.session_state["dropoff_lat"] = float(data["d_lat"])
        st.session_state["dropoff_lon"] = float(data["d_lon"])
        st.session_state["dropoff_label"] = str(data["dropoff_name"])
        st.session_state["passenger_count"] = int(data["default_passengers"])


# ==========================================
# 5. UBER NAVBAR
# ==========================================
st.markdown("""
<div class="uber-navbar">
    <div>
        <div class="uber-logo">Uber</div>
        <div class="uber-subtitle">FARE ESTIMATOR & PREDICTION ENGINE &bull; NYC METRO</div>
    </div>
    <div class="uber-nav-pills">
        <div class="uber-pill active">🚕 Ride</div>
        <div class="uber-pill">🚗 Drive</div>
        <div class="uber-pill">💼 Business</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Load Scikit-Learn Model
model, model_path = load_trained_model()
if model is None:
    st.error("⚠️ Model file not found. Please ensure `best_model.joblib` exists in the `models/` directory.")
    st.stop()


# ==========================================
# 6. TWO-COLUMN LAYOUT: INPUTS & OUTPUTS
# ==========================================
col_inputs, col_results = st.columns([1.05, 1.35], gap="large")

# ------------------------------------------
# LEFT COLUMN: USER INPUT CONTROLS
# ------------------------------------------
with col_inputs:
    with st.container(border=True):
        st.markdown("### 🚖 Trip Details")

        # Preset Route Selector
        st.selectbox(
            "⚡ Quick Select NYC Popular Route",
            options=list(PRESETS.keys()),
            key="preset_choice",
            on_change=handle_preset_change,
            help="Select a popular NYC route or click directly on the map to choose coordinates."
        )

        st.divider()

        # Pickup Coordinates & Address
        st.markdown('<div class="section-label">📍 Pickup Location</div>', unsafe_allow_html=True)
        st.text_input(
            "Pickup Address / Label",
            key="pickup_label",
            placeholder="e.g. Times Square, Manhattan"
        )
        c1, c2 = st.columns(2)
        with c1:
            st.number_input(
                "Pickup Latitude",
                min_value=40.0,
                max_value=42.0,
                step=0.001,
                format="%.4f",
                key="pickup_lat"
            )
        with c2:
            st.number_input(
                "Pickup Longitude",
                min_value=-75.0,
                max_value=-72.0,
                step=0.001,
                format="%.4f",
                key="pickup_lon"
            )

        st.divider()

        # Dropoff Coordinates & Address
        st.markdown('<div class="section-label">🏁 Dropoff Location</div>', unsafe_allow_html=True)
        st.text_input(
            "Dropoff Address / Label",
            key="dropoff_label",
            placeholder="e.g. JFK Airport Terminal 4"
        )
        c3, c4 = st.columns(2)
        with c3:
            st.number_input(
                "Dropoff Latitude",
                min_value=40.0,
                max_value=42.0,
                step=0.001,
                format="%.4f",
                key="dropoff_lat"
            )
        with c4:
            st.number_input(
                "Dropoff Longitude",
                min_value=-75.0,
                max_value=-72.0,
                step=0.001,
                format="%.4f",
                key="dropoff_lon"
            )

        st.divider()

        # Date, Time & Passenger Count
        st.markdown('<div class="section-label">🕒 Schedule & Riders</div>', unsafe_allow_html=True)
        c5, c6 = st.columns(2)
        now = datetime.datetime.now()
        with c5:
            ride_date = st.date_input("Ride Date", value=now.date(), key="ride_date_picker")
        with c6:
            ride_time = st.time_input("Ride Time", value=now.time(), key="ride_time_picker")

        st.slider(
            "👥 Passenger Count",
            min_value=1,
            max_value=6,
            key="passenger_count",
            help="Number of passengers traveling."
        )

        st.button("⚡ Recalculate Fare", use_container_width=True)


# ------------------------------------------
# FEATURE EXTRACTION & INFERENCE
# ------------------------------------------
cur_p_lat = float(st.session_state["pickup_lat"])
cur_p_lon = float(st.session_state["pickup_lon"])
cur_d_lat = float(st.session_state["dropoff_lat"])
cur_d_lon = float(st.session_state["dropoff_lon"])

# Distance calculation
dist_km = calculate_haversine(cur_p_lat, cur_p_lon, cur_d_lat, cur_d_lon)
dist_miles = dist_km * 0.621371

# Temporal extraction
combined_datetime = datetime.datetime.combine(ride_date, ride_time)
hour = combined_datetime.hour
day_of_week = combined_datetime.weekday()  # Monday = 0, Sunday = 6
month = combined_datetime.month
year = combined_datetime.year
day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
day_name = day_names[day_of_week]

# Estimated drive time
speed_kmh = 16.0 if (7 <= hour <= 9 or 17 <= hour <= 20) else 22.0
est_minutes = max(int(np.round((dist_km / speed_kmh) * 60)), 3)

# Build features DataFrame matching model training schema
features_df = pd.DataFrame([{
    'distance_km': float(dist_km),
    'pickup_longitude': float(cur_p_lon),
    'pickup_latitude': float(cur_p_lat),
    'dropoff_longitude': float(cur_d_lon),
    'dropoff_latitude': float(cur_d_lat),
    'passenger_count': int(st.session_state["passenger_count"]),
    'hour': int(hour),
    'day_of_week': int(day_of_week),
    'month': int(month),
    'year': int(year)
}])

# Scikit-Learn Prediction
raw_fare = float(model.predict(features_df)[0])
base_uberx_fare = max(raw_fare, 2.50)  # NYC regulatory floor

# Uber Product Tier Multipliers
uber_comfort_fare = base_uberx_fare * 1.25
uber_xl_fare = base_uberx_fare * 1.50
uber_black_fare = base_uberx_fare * 2.10

# Traffic & Peak Analysis
is_rush_hour = (7 <= hour <= 9) or (17 <= hour <= 20)
is_night = (hour >= 23) or (hour <= 4)
traffic_tag = "🔥 Peak Rush Hour" if is_rush_hour else ("🌙 Late Night" if is_night else "🟢 Normal Traffic")


# ------------------------------------------
# RIGHT COLUMN: MAP & UBER RIDE TIERS
# ------------------------------------------
with col_results:
    # 1. Fare Hero Card
    st.markdown(f"""
    <div class="fare-hero">
        <div class="fare-hero-label">Estimated Fare &bull; UberX</div>
        <div class="fare-hero-price">${base_uberx_fare:.2f}</div>
        <div class="fare-hero-route">
            {st.session_state["pickup_label"]} ➔ {st.session_state["dropoff_label"]}
        </div>
        <div class="fare-pills-row">
            <span class="fare-stat-pill">📏 {dist_km:.2f} km ({dist_miles:.2f} mi)</span>
            <span class="fare-stat-pill">⏱️ ~{est_minutes} min drive</span>
            <span class="fare-stat-pill">{traffic_tag}</span>
            <span class="fare-stat-pill">📅 {day_name}, {combined_datetime.strftime('%b %d, %I:%M %p')}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 2. Interactive Map Container with Click-to-Select
    with st.container(border=True):
        st.markdown("### 🗺️ Interactive NYC Map")
        st.markdown(
            '<div class="map-helper">💡 <strong>Interactive Coordinate Picker:</strong> '
            'Click anywhere on the map to set the selected coordinates directly!</div>',
            unsafe_allow_html=True
        )

        # Radio button to choose what a map click updates
        st.radio(
            "Map Click Target:",
            options=["📍 Pickup Location (Green)", "🏁 Dropoff Location (Red)"],
            key="map_click_mode",
            horizontal=True,
            help="Choose whether clicking the map sets the Pickup or Dropoff point."
        )

        # Build Folium Interactive Map
        mid_lat = (cur_p_lat + cur_d_lat) / 2.0
        mid_lon = (cur_p_lon + cur_d_lon) / 2.0

        folium_map = folium.Map(
            location=[mid_lat, mid_lon],
            zoom_start=12,
            tiles="OpenStreetMap"
        )

        # Pickup Marker (Green Pin)
        folium.Marker(
            location=[cur_p_lat, cur_p_lon],
            popup=f"📍 Pickup: {st.session_state['pickup_label']}",
            tooltip="Pickup Location",
            icon=folium.Icon(color="green", icon="play")
        ).add_to(folium_map)

        # Dropoff Marker (Red Pin)
        folium.Marker(
            location=[cur_d_lat, cur_d_lon],
            popup=f"🏁 Dropoff: {st.session_state['dropoff_label']}",
            tooltip="Dropoff Location",
            icon=folium.Icon(color="red", icon="flag")
        ).add_to(folium_map)

        # Route connecting line
        folium.PolyLine(
            locations=[[cur_p_lat, cur_p_lon], [cur_d_lat, cur_d_lon]],
            color="#000000",
            weight=4,
            opacity=0.85,
            dash_array="6, 8"
        ).add_to(folium_map)

        # Dynamic auto-fit bounds
        if abs(cur_p_lat - cur_d_lat) > 0.005 or abs(cur_p_lon - cur_d_lon) > 0.005:
            folium_map.fit_bounds(
                [[min(cur_p_lat, cur_d_lat), min(cur_p_lon, cur_d_lon)],
                 [max(cur_p_lat, cur_d_lat), max(cur_p_lon, cur_d_lon)]],
                padding=[35, 35]
            )

        # Render Folium Map in Streamlit
        st_folium(
            folium_map,
            key="uber_map",
            height=370,
            use_container_width=True,
            returned_objects=["last_clicked"]
        )

    # 3. Uber Available Ride Options
    with st.container(border=True):
        st.markdown("### 🚗 Available Ride Options")

        st.markdown(f"""
        <!-- UberX Tier -->
        <div class="ride-row active-tier">
            <div class="ride-info-left">
                <div class="ride-emoji">🚗</div>
                <div>
                    <div class="ride-name">UberX <span style="font-size: 11px; background: #000; color: #FFF; padding: 2px 7px; border-radius: 4px; margin-left: 6px;">Recommended</span></div>
                    <div class="ride-desc">Affordable, everyday rides &bull; 1-4 seats &bull; Base model</div>
                </div>
            </div>
            <div class="ride-price-right">
                <div class="ride-price-val">${base_uberx_fare:.2f}</div>
                <div class="ride-time-est">~{est_minutes} min away</div>
            </div>
        </div>

        <!-- Uber Comfort Tier -->
        <div class="ride-row">
            <div class="ride-info-left">
                <div class="ride-emoji">🚙</div>
                <div>
                    <div class="ride-name">Uber Comfort</div>
                    <div class="ride-desc">Newer cars with extra legroom & top-rated drivers &bull; 4 seats</div>
                </div>
            </div>
            <div class="ride-price-right">
                <div class="ride-price-val">${uber_comfort_fare:.2f}</div>
                <div class="ride-time-est">~{est_minutes + 2} min away</div>
            </div>
        </div>

        <!-- UberXL Tier -->
        <div class="ride-row">
            <div class="ride-info-left">
                <div class="ride-emoji">🚐</div>
                <div>
                    <div class="ride-name">UberXL</div>
                    <div class="ride-desc">Spacious SUVs and vans for groups up to 6 riders & luggage</div>
                </div>
            </div>
            <div class="ride-price-right">
                <div class="ride-price-val">${uber_xl_fare:.2f}</div>
                <div class="ride-time-est">~{est_minutes + 4} min away</div>
            </div>
        </div>

        <!-- Uber Black Tier -->
        <div class="ride-row">
            <div class="ride-info-left">
                <div class="ride-emoji">🖤</div>
                <div>
                    <div class="ride-name">Uber Black</div>
                    <div class="ride-desc">Premium luxury rides with professional chauffeurs</div>
                </div>
            </div>
            <div class="ride-price-right">
                <div class="ride-price-val">${uber_black_fare:.2f}</div>
                <div class="ride-time-est">~{est_minutes + 1} min away</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Model Transparency Callout
        st.info(
            f"⚡ **Model Intelligence:** Powered by `HistGradientBoostingRegressor` trained on 195,096 NYC Uber trips.\n\n"
            f"**Validation Metrics:** Test R²: **0.8402** | RMSE: **$3.88** | MAE: **$1.82** | Model File: `{os.path.basename(model_path)}`"
        )


# ==========================================
# 7. DIRECT PYTHON CLI RUNNER
# ==========================================
if __name__ == "__main__":
    from streamlit.runtime import exists
    if not exists():
        import subprocess
        print("Starting Uber Fare Predictor web app via Streamlit...")
        subprocess.run([sys.executable, "-m", "streamlit", "run", __file__])
