import time
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime
from sklearn.ensemble import IsolationForest

st.set_page_config(page_title="Predictive Maintenance – Rectifier", page_icon="🛠️", layout="wide")

# ------------ STATE ------------
if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame(columns=["ts","equipment_id","temperature_c","vibration_rms","current_a","voltage_v","fan_rpm"])
if "alarms" not in st.session_state:
    st.session_state.alarms = []
if "running" not in st.session_state:
    st.session_state.running = False
if "faults" not in st.session_state:
    st.session_state.faults = {"cooling":False, "fan":False, "voltage":False}

# ------------ CONSTANTS ------------
THRESHOLDS = {
    "temperature_c": {"warn": 60, "alert": 70},
    "vibration_rms": {"warn": 0.6, "alert": 0.8},
    "current_a":     {"warn": 150, "alert": 180},
    "voltage_v":     {"warn": 580, "alert": 620},
    "fan_rpm":       {"warn": 2600, "alert": 2000},  # unter
}
METRICS = ["temperature_c","vibration_rms","current_a","voltage_v","fan_rpm"]

# ------------ UI: HEADER ------------
colL, colR = st.columns([1,1])
with colL:
    st.markdown("### 🛠️ Predictive Maintenance – Gleichrichter")
    equipment_id = st.text_input("Equipment-ID", value="RECT-0001")
with colR:
    st.markdown("#### Live-Status")
    status_placeholder = st.empty()

st.markdown("---")

# ------------ TABS ------------
tab_overview, tab_live, tab_alerts, tab_settings = st.tabs(["Overview", "Live Charts", "Alerts", "Settings"])

# ------------ SETTINGS ------------
with tab_settings:
    st.subheader("Fault Injection")
    col1,col2,col3 = st.columns(3)
    with col1: st.session_state.faults["cooling"] = st.checkbox("Cooling Degradation (Temp ↑)")
    with col2: st.session_state.faults["fan"]     = st.checkbox("Fan Wear (RPM ↓)")
    with col3: st.session_state.faults["voltage"] = st.checkbox("Voltage Spikes")

    st.subheader("Schwellwerte (SAP-like)")
    g1,g2,g3,g4,g5 = st.columns(5)
    THRESHOLDS["temperature_c"]["warn"] = g1.number_input("Temp WARN",  value=THRESHOLDS["temperature_c"]["warn"])
    THRESHOLDS["temperature_c"]["alert"]= g1.number_input("Temp ALERT", value=THRESHOLDS["temperature_c"]["alert"])
    THRESHOLDS["vibration_rms"]["warn"] = g2.number_input("Vib WARN",   value=THRESHOLDS["vibration_rms"]["warn"], step=0.01, format="%.2f")
    THRESHOLDS["vibration_rms"]["alert"]= g2.number_input("Vib ALERT",  value=THRESHOLDS["vibration_rms"]["alert"], step=0.01, format="%.2f")
    THRESHOLDS["current_a"]["warn"]     = g3.number_input("I WARN",     value=THRESHOLDS["current_a"]["warn"])
    THRESHOLDS["current_a"]["alert"]    = g3.number_input("I ALERT",    value=THRESHOLDS["current_a"]["alert"])
    THRESHOLDS["voltage_v"]["warn"]     = g4.number_input("U WARN",     value=THRESHOLDS["voltage_v"]["warn"])
    THRESHOLDS["voltage_v"]["alert"]    = g4.number_input("U ALERT",    value=THRESHOLDS["voltage_v"]["alert"])
    THRESHOLDS["fan_rpm"]["warn"]       = g5.number_input("RPM WARN (unter)",  value=THRESHOLDS["fan_rpm"]["warn"])
    THRESHOLDS["fan_rpm"]["alert"]      = g5.number_input("RPM ALERT (unter)", value=THRESHOLDS["fan_rpm"]["alert"])

    st.subheader("KI-Anomalie (IsolationForest)")
    c1,c2,c3 = st.columns(3)
    window = c1.slider("Fenstergröße (Punkte)", 200, 2000, 600, 50)
    contamination = c2.slider("Kontamination (expected outliers)", 0.001, 0.1, 0.02, 0.001)
    ml_alert_thresh = c3.slider("ML-Alert-Schwelle (0-1)", 0.1, 0.9, 0.8, 0.05, help="Score ≥ Schwelle ⇒ ALERT, sonst WARN/OK")
    st.caption("Hinweis: IsolationForest nutzt die letzten N Punkte und bewertet den neuesten Messpunkt.")

    st.subheader("Simulation Control")
    if not st.session_state.running:
        if st.button("▶️ Start Simulation", use_container_width=True):
            st.session_state.running = True
            st.experimental_rerun()
    else:
        if st.button("⏹ Stop Simulation", use_container_width=True):
            st.session_state.running = False

# ------------ SIMULATOR ------------
def generate_sample(t: int):
    base = {"temperature_c":45.0, "vibration_rms":0.35, "current_a":120.0, "voltage_v":540.0, "fan_rpm":3200.0}
    # Faults
    if st.session_state.faults["cooling"]: base["temperature_c"] += 0.01 * t
    if st.session_state.faults["fan"]:     base["fan_rpm"]       -= 0.5 * t
    if st.session_state.faults["voltage"]: base["voltage_v"]     += 20*np.sin(t/3.0)
    # Noise
    base["temperature_c"] += np.random.uniform(-0.2,0.2)
    base["vibration_rms"] += np.random.uniform(-0.02,0.02)
    base["current_a"]     += np.random.uniform(-2,2)
    base["voltage_v"]     += np.random.uniform(-1.5,1.5)
    base["fan_rpm"]       += np.random.uniform(-30,30)
    return base

def push_alarm(ts, level, msg):
    st.session_state.alarms.append({"ts":ts,"level":level,"message":msg})

def check_thresholds(vals, ts):
    for k,v in THRESHOLDS.items():
        val = float(vals[k])
        if k=="fan_rpm":
            if val < v["alert"]: push_alarm(ts,"ALERT", f"{k} too low: {val:.1f}")
            elif val < v["warn"]: push_alarm(ts,"WARN", f"{k} low: {val:.1f}")
        else:
            if val > v["alert"]: push_alarm(ts,"ALERT", f"{k} too high: {val:.1f}")
            elif val > v["warn"]: push_alarm(ts,"WARN", f"{k} high: {val:.1f}")

def ml_anomaly(df: pd.DataFrame, window: int, contamination: float):
    """Fit IF auf letzten window-1 Punkten, score für den letzten Punkt zurückgeben (0..1)."""
    if len(df) < window: return None, None
    data = df.iloc[-window:].copy()
    # Features skaliert (z-Score je Spalte)
    X = data[METRICS].astype(float).to_numpy()
    mu = X.mean(axis=0); sigma = X.std(axis=0); sigma[sigma==0] = 1e-6
    Z = (X - mu) / sigma
    # Train auf alle bis auf letzten, score den letzten
    Z_train, z_last = Z[:-1], Z[-1].reshape(1,-1)
    model = IsolationForest(contamination=contamination, random_state=42)
    model.fit(Z_train)
    # decision_function: größere Werte = weniger abnormal; wir invertieren + normalisieren zu 0..1
    raw = -model.decision_function(z_last)[0]
    # einfache Min/Max-Norm: gegen Trainingsscores skaliert
    tr_raw = -model.decision_function(Z_train)
    lo, hi = float(tr_raw.min()), float(tr_raw.max()) + 1e-9
    score = (raw - lo) / (hi - lo)
    return float(score), {"mu":mu.tolist(), "sigma":sigma.tolist()}

def overall_level(th_levels, ml_score, ml_thresh):
    order = {"OK":0,"WARN":1,"ALERT":2}
    level = "OK"
    # thresholds
    for lv in th_levels:
        if order[lv] > order[level]: level = lv
    # ml
    if ml_score is not None:
        if ml_score >= ml_thresh: level = "ALERT"
        elif ml_score >= (ml_thresh*0.7) and order["WARN"] > order[level]: level = "WARN"
    return level

# ------------ LIVE LOOP ------------
if st.session_state.running:
    t = len(st.session_state.df)
    vals = generate_sample(t)
    ts = datetime.now().strftime("%H:%M:%S")
    row = {"ts":ts, "equipment_id":equipment_id, **vals}
    st.session_state.df = pd.concat([st.session_state.df, pd.DataFrame([row])], ignore_index=True)

    # Rule-based
    before_count = len(st.session_state.alarms)
    check_thresholds(vals, ts)
    new_threshold_alarms = len(st.session_state.alarms) - before_count

    # ML-based
    score, _ = ml_anomaly(st.session_state.df, window=window, contamination=contamination)
    if score is not None and score >= ml_alert_thresh:
        push_alarm(ts, "ALERT", f"ML anomaly score={score:.2f}")
    elif score is not None and score >= ml_alert_thresh*0.7:
        push_alarm(ts, "WARN", f"ML anomaly score={score:.2f}")

    status_placeholder.success(f"RUNNING – Last sample @ {ts}")
    time.sleep(1)
    st.experimental_rerun()
else:
    status_placeholder.warning("Simulation gestoppt")

# ------------ OVERVIEW ------------
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

        # Health Badge
        # sammle Schwellen-Level der letzten Probe
        th_levels = []
        for k,v in THRESHOLDS.items():
            val = float(latest[k])
            if k=="fan_rpm":
                th_levels.append("ALERT" if val < v["alert"] else "WARN" if val < v["warn"] else "OK")
            else:
                th_levels.append("ALERT" if val > v["alert"] else "WARN" if val > v["warn"] else "OK")
        # ML-Score letzte Probe
        score, _ = ml_anomaly(st.session_state.df, window=window, contamination=contamination)
        lvl = overall_level(th_levels, score, ml_alert_thresh)
        colA, colB = st.columns([1,3])
        with colA:
            color = {"OK":"✅ OK","WARN":"🟠 WARN","ALERT":"🔴 ALERT"}[lvl]
            st.markdown(f"**Health:** {color}")
        with colB:
            st.caption(f"ML-Score: {score:.2f}" if score is not None else "ML-Score: – (zu wenig Daten)")

    else:
        st.info("Noch keine Daten.")

# ------------ LIVE CHARTS ------------
with tab_live:
    if len(st.session_state.df):
        st.line_chart(st.session_state.df.set_index("ts")[METRICS])
    else:
        st.info("Noch keine Daten.")

# ------------ ALERTS ------------
with tab_alerts:
    if st.session_state.alarms:
        for a in reversed(st.session_state.alarms[-50:]):
            (st.error if a["level"]=="ALERT" else st.warning)(f"[{a['ts']}] {a['message']}")
    else:
        st.info("Keine Alarme.")
