import time
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime
from sklearn.ensemble import IsolationForest

st.set_page_config(page_title="Predictive Maintenance – Rectifier", page_icon="🛠️", layout="wide")

# ================= CSS =================
st.markdown("""
<style>
/* Dark SAP Style */
body, .stApp {
    background-color: #1a1d24;
    color: #eaf2ff;
}
section.main > div {max-width: 1400px;}

/* Tabellen */
thead, thead * { color: #ffffff !important; font-weight: bold; }
tbody, tbody * { color: #eaf2ff !important; }

/* KPI-Kacheln */
[data-testid="stMetricValue"] { color: #eaf2ff !important; }
[data-testid="stMetricLabel"] { color: #b9c7d9 !important; }

/* Slider Farben */
.stSlider > div[data-baseweb="slider"] { color: #ff4b4b; }

/* Text allgemein */
html, body, [class*="css"] { color: #eaf2ff !important; }
[data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] * { color: #eaf2ff !important; }

/* Labels */
label, .stSlider label, .stNumberInput label, .stTextInput label, 
.stSelectbox label, .stMultiSelect label, .stCheckbox label {
  color: #eaf2ff !important;
}

/* Captions */
small, .stCaption { color: #b9c7d9 !important; }
</style>
""", unsafe_allow_html=True)

# ================= EQUIPMENT =================
EQUIPMENTS = {
    "10109812-01": {
        "name": "Gleichrichter XD1",
        "location": "Schaltschrank 1 – Galvanik Halle (Schüttgutbereich)",
    },
    "10109812-02": {
        "name": "Gleichrichter XD2",
        "location": "Schaltschrank 2 – Galvanik Halle (Schüttgutbereich)",
    },
}

# ================= STATE =================
if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame(
        columns=["ts","equipment_id","temperature_c","vibration_rms","current_a","voltage_v","fan_rpm"]
    )
if "alarms" not in st.session_state:
    st.session_state.alarms = []
if "running" not in st.session_state:
    st.session_state.running = False
if "faults" not in st.session_state:
    st.session_state.faults = {"cooling": False, "fan": False, "voltage": False}
if "eq_num" not in st.session_state:
    st.session_state.eq_num = "10109812-01"

# Soll- & Schwellwerte
THRESHOLDS = {
    "temperature_c": {"warn": 60.0,  "alert": 70.0},   # °C
    "vibration_rms": {"warn": 0.60,  "alert": 0.80},   # RMS
    "current_a":     {"warn": 150.0, "alert": 180.0},  # A
    "voltage_v":     {"warn": 578.0, "alert": 621.0},  # V
    "fan_rpm":       {"warn": 2600.0,"alert": 2000.0}, # Untergrenze
}
METRICS = ["temperature_c","vibration_rms","current_a","voltage_v","fan_rpm"]

# ================= HEADER =================
colL, colR = st.columns([1,1])
with colL:
    st.markdown("### 🛠️ Predictive Maintenance – Gleichrichter")

    eq_num = st.selectbox("EQ-Nummer", options=list(EQUIPMENTS.keys()), index=list(EQUIPMENTS.keys()).index(st.session_state.eq_num))
    st.session_state.eq_num = eq_num
    st.markdown(f"**Equipment:** {EQUIPMENTS[eq_num]['name']}")
    st.markdown(f"**Standort:** {EQUIPMENTS[eq_num]['location']}")

with colR:
    st.markdown("#### Live-Status")
    status_placeholder = st.empty()

st.markdown("---")

# ================= TABS =================
tab_overview, tab_live, tab_alerts, tab_settings, tab_misc = st.tabs(
    ["Overview", "Live Charts", "Alerts", "Settings", "Sonstiges"]
)

# ================= SETTINGS =================
with tab_settings:
    st.subheader("Simulation Control")
    cstart, cdel = st.columns([2,1])
    with cstart:
        if not st.session_state.running:
            if st.button("▶️ Start Simulation", use_container_width=True):
                st.session_state.running = True
                st.experimental_rerun()
        else:
            if st.button("⏹ Stop Simulation", use_container_width=True):
                st.session_state.running = False
    with cdel:
        if st.button("🗑️ Daten löschen"):
            st.session_state.df = pd.DataFrame(columns=st.session_state.df.columns)
            st.session_state.alarms = []
            st.success("Daten & Alarme gelöscht.")

    st.markdown("---")
    st.subheader("Schwellwerte")
    r1, r2 = st.columns([3,3])
    with r1:
        st.number_input("Temperatur WARN (°C)", value=THRESHOLDS["temperature_c"]["warn"], step=1.0)
        st.number_input("Temperatur ALERT (°C)", value=THRESHOLDS["temperature_c"]["alert"], step=1.0)
        st.number_input("Vibration WARN (RMS)", value=THRESHOLDS["vibration_rms"]["warn"], step=0.01, format="%.2f")
        st.number_input("Vibration ALERT (RMS)", value=THRESHOLDS["vibration_rms"]["alert"], step=0.01, format="%.2f")
    with r2:
        st.number_input("Strom WARN (A)", value=THRESHOLDS["current_a"]["warn"], step=1.0)
        st.number_input("Strom ALERT (A)", value=THRESHOLDS["current_a"]["alert"], step=1.0)
        st.number_input("Spannung WARN (V)", value=THRESHOLDS["voltage_v"]["warn"], step=1.0)
        st.number_input("Spannung ALERT (V)", value=THRESHOLDS["voltage_v"]["alert"], step=1.0)
        st.number_input("Lüfter WARN (RPM, Untergrenze)", value=THRESHOLDS["fan_rpm"]["warn"], step=50.0)
        st.number_input("Lüfter ALERT (RPM, Untergrenze)", value=THRESHOLDS["fan_rpm"]["alert"], step=50.0)

    st.info("""
    **Legende Soll-/Schwellwerte:**  
    - Temperatur: **SOLL ~45 °C → WARN ab 60 °C, ALERT ab 70 °C**  
    - Strom: **SOLL ~120 A → WARN ab 150 A, ALERT ab 180 A**  
    - Spannung: **SOLL ~540 V → WARN ab 578 V, ALERT ab 621 V**  
    - Lüfter (Untergrenze): **SOLL ~3200 RPM → WARN <2600, ALERT <2000**  
    """)

# ================= SIMULATOR FUNCS =================
def generate_sample(t: int):
    base = {"temperature_c":45.0, "vibration_rms":0.35, "current_a":120.0, "voltage_v":540.0, "fan_rpm":3200.0}
    if st.session_state.faults.get("cooling"): base["temperature_c"] += 0.01 * t
    if st.session_state.faults.get("fan"): base["fan_rpm"] -= 0.5 * t
    if st.session_state.faults.get("voltage"): base["voltage_v"] += 20 * np.sin(t/3.0)
    base["temperature_c"] += np.random.uniform(-0.2,0.2)
    base["vibration_rms"] += np.random.uniform(-0.02,0.02)
    base["current_a"] += np.random.uniform(-2,2)
    base["voltage_v"] += np.random.uniform(-1.5,1.5)
    base["fan_rpm"] += np.random.uniform(-30,30)
    return base

def push_alarm(ts, level, msg):
    st.session_state.alarms.append({"ts": ts, "level": level, "message": msg})

def check_thresholds(vals, ts):
    for k, v in THRESHOLDS.items():
        val = float(vals[k])
        if k == "fan_rpm":
            if val < v["alert"]: push_alarm(ts, "ALERT", f"{k} zu niedrig: {val:.0f} RPM")
            elif val < v["warn"]: push_alarm(ts, "WARN", f"{k} niedrig: {val:.0f} RPM")
        else:
            if val > v["alert"]: push_alarm(ts, "ALERT", f"{k} zu hoch: {val:.1f}")
            elif val > v["warn"]: push_alarm(ts, "WARN", f"{k} hoch: {val:.1f}")

def ml_anomaly(df, window, contamination):
    if len(df) < window: return None, None
    data = df.iloc[-window:].copy()
    X = data[METRICS].astype(float).to_numpy()
    mu = X.mean(axis=0); sigma = X.std(axis=0); sigma[sigma == 0] = 1e-6
    Z = (X - mu) / sigma
    Z_train, z_last = Z[:-1], Z[-1].reshape(1,-1)
    model = IsolationForest(contamination=contamination, random_state=42)
    model.fit(Z_train)
    raw_last = -model.decision_function(z_last)[0]
    raw_train = -model.decision_function(Z_train)
    lo, hi = float(raw_train.min()), float(raw_train.max())+1e-9
    score = (raw_last - lo) / (hi - lo)
    return float(score), {}

# ================= LIVE LOOP =================
if st.session_state.running:
    t = len(st.session_state.df)
    vals = generate_sample(t)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = {"ts": ts, "equipment_id": st.session_state.eq_num, **vals}
    st.session_state.df = pd.concat([st.session_state.df, pd.DataFrame([row])], ignore_index=True)
    check_thresholds(vals, ts)
    score, _ = ml_anomaly(st.session_state.df, 600, 0.02)
    if score is not None and score >= 0.8: push_alarm(ts, "ALERT", f"ML anomaly score={score:.2f}")
    elif score is not None and score >= 0.56: push_alarm(ts, "WARN", f"ML anomaly score={score:.2f}")
    status_placeholder.success(f"RUNNING – Last sample @ {ts}")
    time.sleep(1); st.experimental_rerun()
else:
    status_placeholder.warning("Simulation gestoppt")

# ================= OVERVIEW =================
with tab_overview:
    st.subheader("Gesamtzustand")
    if len(st.session_state.df):
        latest = st.session_state.df.iloc[-1]
        kpi1,kpi2,kpi3,kpi4,kpi5 = st.columns(5)
        kpi1.metric("Temperatur (°C)", f"{latest['temperature_c']:.1f}")
        kpi2.metric("Vibration (RMS)", f"{latest['vibration_rms']:.2f}")
        kpi3.metric("Strom (A)", f"{latest['current_a']:.1f}")
        kpi4.metric("Spannung (V)", f"{latest['voltage_v']:.1f}")
        kpi5.metric("Lüfter (RPM)", f"{latest['fan_rpm']:.0f}")
        data_csv = st.session_state.df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Export Messdaten (CSV)", data=data_csv, file_name=f"timeseries_{st.session_state.eq_num}.csv", mime="text/csv")
    else: st.info("Noch keine Daten.")

# ================= LIVE CHARTS =================
with tab_live:
    if len(st.session_state.df): st.line_chart(st.session_state.df.set_index("ts")[METRICS])
    else: st.info("Noch keine Daten.")

# ================= ALERTS =================
with tab_alerts:
    st.subheader("Alarm-Feed")
    if st.session_state.alarms:
        df_alerts = pd.DataFrame(st.session_state.alarms)
        alerts_csv = df_alerts.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Export Alerts (CSV)", data=alerts_csv, file_name=f"alerts_{st.session_state.eq_num}.csv", mime="text/csv")
        for a in reversed(st.session_state.alarms[-200:]):
            (st.error if a["level"]=="ALERT" else st.warning)(f"[{a['ts']}] {a['message']}")
    else: st.info("Keine Alarme.")

# ================= MISC =================
with tab_misc:
    st.subheader("Beispiele (Vorzeigen)")
    st.markdown("**IsolationForest Prinzip – Normal vs. Anomalie**")
    st.image("isoforest_example.png", caption="Normale Punkte vs. Ausreißer (rot = Anomalie)")
    st.markdown("**Beispiel-Export (eine Liste – Vorschau)**")
    sample = pd.DataFrame([
        {"ts":"2025-10-07 21:51:32","equipment_id":"10109812-01","level":"WARN","message":"Spannung hoch: 602.3 V","temperature_c":45.2,"vibration_rms":0.36,"current_a":121,"voltage_v":602.3,"fan_rpm":3180},
        {"ts":"2025-10-07 21:52:15","equipment_id":"10109812-01","level":"ALERT","message":"ML anomaly score=0.86","temperature_c":45.2,"vibration_rms":0.36,"current_a":121,"voltage_v":541.2,"fan_rpm":3180}
    ])
    st.dataframe(sample, use_container_width=True, hide_index=True)
    st.caption("⚡ Hier sieht man: Zeile 1 = Warnung wegen zu hoher Spannung. Zeile 2 = KI-Alarm durch IsolationForest.")
