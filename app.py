import time
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime

st.set_page_config(
    page_title="Predictive Maintenance – Rectifier",
    page_icon="🛠️",
    layout="wide"
)

# --- INIT STATE ---
if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame(columns=["ts","equipment_id","temperature_c","vibration_rms","current_a","voltage_v","fan_rpm"])
if "alarms" not in st.session_state:
    st.session_state.alarms = []
if "running" not in st.session_state:
    st.session_state.running = False
if "faults" not in st.session_state:
    st.session_state.faults = {"cooling":False, "fan":False, "voltage":False}

# --- HEADER ---
colL, colR = st.columns([1,1])
with colL:
    st.markdown("### 🛠️ Predictive Maintenance – Gleichrichter")
    equipment_id = st.text_input("Equipment-ID", value="RECT-0001")
with colR:
    st.markdown("#### Live-Status")
    status_placeholder = st.empty()

st.markdown("---")

# --- Settings ---
tab_overview, tab_live, tab_alerts, tab_settings = st.tabs(["Overview", "Live Charts", "Alerts", "Settings"])

# --- Thresholds ---
THRESHOLDS = {
    "temperature_c": {"warn": 60, "alert": 70},
    "vibration_rms": {"warn": 0.6, "alert": 0.8},
    "current_a":     {"warn": 150, "alert": 180},
    "voltage_v":     {"warn": 580, "alert": 620},
    "fan_rpm":       {"warn": 2600, "alert": 2000},  # WARN, wenn unter 2600; ALERT, wenn unter 2000
}

# --- Fault Injection Flags ---
with tab_settings:
    st.subheader("Fault Injection")
    st.caption("Simuliere Fehler wie in SAP PdM")
    col1,col2,col3 = st.columns(3)
    with col1:
        st.session_state.faults["cooling"] = st.checkbox("Cooling Degradation (Temp steigt)")
    with col2:
        st.session_state.faults["fan"] = st.checkbox("Fan Wear (RPM sinkt)")
    with col3:
        st.session_state.faults["voltage"] = st.checkbox("Voltage Spikes")

    st.subheader("Simulation Control")
    if not st.session_state.running:
        if st.button("▶️ Start Simulation"):
            st.session_state.running = True
            st.experimental_rerun()
    else:
        if st.button("⏹ Stop Simulation"):
            st.session_state.running = False

# --- Simulator ---
def generate_sample(t: int):
    base = {
        "temperature_c": 45.0,
        "vibration_rms": 0.35,
        "current_a": 120.0,
        "voltage_v": 540.0,
        "fan_rpm": 3200.0,
    }
    # Faults
    if st.session_state.faults["cooling"]:
        base["temperature_c"] += 0.01 * t  # drift
    if st.session_state.faults["fan"]:
        base["fan_rpm"] -= 0.5 * t
    if st.session_state.faults["voltage"]:
        base["voltage_v"] += 20*np.sin(t/3.0)

    # Noise
    base["temperature_c"] += np.random.uniform(-0.2,0.2)
    base["vibration_rms"] += np.random.uniform(-0.02,0.02)
    base["current_a"]     += np.random.uniform(-2,2)
    base["voltage_v"]     += np.random.uniform(-1.5,1.5)
    base["fan_rpm"]       += np.random.uniform(-30,30)

    return base

def check_thresholds(vals, ts):
    for k,v in THRESHOLDS.items():
        val = vals[k]
        if k=="fan_rpm":  # Sonderfall: Untergrenze
            if val < v["alert"]:
                st.session_state.alarms.append({"ts":ts,"level":"ALERT","message":f"{k} too low: {val:.1f}"})
            elif val < v["warn"]:
                st.session_state.alarms.append({"ts":ts,"level":"WARN","message":f"{k} low: {val:.1f}"})
        else:
            if val > v["alert"]:
                st.session_state.alarms.append({"ts":ts,"level":"ALERT","message":f"{k} too high: {val:.1f}"})
            elif val > v["warn"]:
                st.session_state.alarms.append({"ts":ts,"level":"WARN","message":f"{k} high: {val:.1f}"})

# --- Live Loop ---
if st.session_state.running:
    placeholder = st.empty()
    t = len(st.session_state.df)
    vals = generate_sample(t)
    ts = datetime.now().strftime("%H:%M:%S")
    row = {"ts":ts, "equipment_id":equipment_id, **vals}
    st.session_state.df = pd.concat([st.session_state.df, pd.DataFrame([row])], ignore_index=True)
    check_thresholds(vals, ts)
    status_placeholder.success(f"RUNNING – Last sample @ {ts}")
    time.sleep(1)
    st.experimental_rerun()
else:
    status_placeholder.warning("Simulation gestoppt")

# --- Overview Tab ---
with tab_overview:
    st.subheader("Gesamtzustand")
    if len(st.session_state.df):
        latest = st.session_state.df.iloc[-1]
        kpi1,kpi2,kpi3,kpi4,kpi5 = st.columns(5)
        kpi1.metric("Temperatur °C", f"{latest['temperature_c']:.1f}")
        kpi2.metric("Vibration RMS", f"{latest['vibration_rms']:.2f}")
        kpi3.metric("Strom A", f"{latest['current_a']:.1f}")
        kpi4.metric("Spannung V", f"{latest['voltage_v']:.1f}")
        kpi5.metric("Lüfter RPM", f"{latest['fan_rpm']:.0f}")
    else:
        st.info("Noch keine Daten.")

# --- Live Charts Tab ---
with tab_live:
    if len(st.session_state.df):
        st.line_chart(st.session_state.df.set_index("ts")[["temperature_c","vibration_rms","current_a","voltage_v","fan_rpm"]])
    else:
        st.info("Noch keine Daten.")

# --- Alerts Tab ---
with tab_alerts:
    if st.session_state.alarms:
        for a in reversed(st.session_state.alarms[-20:]):
            if a["level"]=="ALERT":
                st.error(f"[{a['ts']}] {a['message']}")
            else:
                st.warning(f"[{a['ts']}] {a['message']}")
    else:
        st.info("Keine Alarme.")
