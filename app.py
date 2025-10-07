import time
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime
from sklearn.ensemble import IsolationForest

st.set_page_config(page_title="Predictive Maintenance – Rectifier", page_icon="🛠️", layout="wide")

# ---------------- STATE ----------------
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

# ---------------- DEFAULT THRESHOLDS ----------------
THRESHOLDS = {
    "temperature_c": {"warn": 60.0,  "alert": 70.0},   # °C
    "vibration_rms": {"warn": 0.60,  "alert": 0.80},   # RMS
    "current_a":     {"warn": 150.0, "alert": 180.0},  # A
    "voltage_v":     {"warn": 580.0, "alert": 620.0},  # V
    "fan_rpm":       {"warn": 2600.0,"alert": 2000.0}, # RPM (Untergrenze)
}
METRICS = ["temperature_c","vibration_rms","current_a","voltage_v","fan_rpm"]

# ---------------- HEADER ----------------
colL, colR = st.columns([1,1])
with colL:
    st.markdown("### 🛠️ Predictive Maintenance – Gleichrichter")
    equipment_id = st.text_input("Equipment-ID", value="RECT-0001")
with colR:
    st.markdown("#### Live-Status")
    status_placeholder = st.empty()

st.markdown("---")

# ---------------- TABS ----------------
tab_overview, tab_live, tab_alerts, tab_settings = st.tabs(["Overview", "Live Charts", "Alerts", "Settings"])

# ---------------- SETTINGS ----------------
with tab_settings:
    st.subheader("Fault Injection (während des Laufs umschaltbar)")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.session_state.faults["cooling"] = st.checkbox("Cooling Degradation — Temperatur steigt")
    with c2:
        st.session_state.faults["fan"] = st.checkbox("Fan Wear — Lüfter RPM sinkt")
    with c3:
        st.session_state.faults["voltage"] = st.checkbox("Voltage Spikes — sporadische Spannungsspitzen")

    st.markdown("---")
    st.subheader("Schwellwerte")
    r1, r2 = st.columns([3,3])
    with r1:
        t_warn  = st.number_input("Temperatur WARN (°C)", value=THRESHOLDS["temperature_c"]["warn"], step=1.0, key="t_warn")
        t_alert = st.number_input("Temperatur ALERT (°C)", value=THRESHOLDS["temperature_c"]["alert"], step=1.0, key="t_alert")
        vib_warn  = st.number_input("Vibration WARN (RMS)", value=THRESHOLDS["vibration_rms"]["warn"], step=0.01, format="%.2f", key="vib_warn")
        vib_alert = st.number_input("Vibration ALERT (RMS)", value=THRESHOLDS["vibration_rms"]["alert"], step=0.01, format="%.2f", key="vib_alert")
    with r2:
        i_warn  = st.number_input("Strom WARN (A)", value=THRESHOLDS["current_a"]["warn"], step=1.0, key="i_warn")
        i_alert = st.number_input("Strom ALERT (A)", value=THRESHOLDS["current_a"]["alert"], step=1.0, key="i_alert")
        u_warn  = st.number_input("Spannung WARN (V)", value=THRESHOLDS["voltage_v"]["warn"], step=1.0, key="u_warn")
        u_alert = st.number_input("Spannung ALERT (V)", value=THRESHOLDS["voltage_v"]["alert"], step=1.0, key="u_alert")

    st.markdown("---")
    st.subheader("Lüfter (RPM) – Untergrenzen")
    rpm_warn  = st.number_input("RPM WARN (unter)", value=THRESHOLDS["fan_rpm"]["warn"], step=10.0, key="rpm_warn")
    rpm_alert = st.number_input("RPM ALERT (unter)", value=THRESHOLDS["fan_rpm"]["alert"], step=10.0, key="rpm_alert")

    # Update Thresholds
    THRESHOLDS["temperature_c"]["warn"], THRESHOLDS["temperature_c"]["alert"] = float(t_warn), float(t_alert)
    THRESHOLDS["vibration_rms"]["warn"], THRESHOLDS["vibration_rms"]["alert"] = float(vib_warn), float(vib_alert)
    THRESHOLDS["current_a"]["warn"],     THRESHOLDS["current_a"]["alert"]     = float(i_warn),  float(i_alert)
    THRESHOLDS["voltage_v"]["warn"],     THRESHOLDS["voltage_v"]["alert"]     = float(u_warn),  float(u_alert)
    THRESHOLDS["fan_rpm"]["warn"],       THRESHOLDS["fan_rpm"]["alert"]       = float(rpm_warn),float(rpm_alert)

    st.markdown("---")
    st.subheader("KI-Anomalie (IsolationForest)")
    st.write(
        """
        **Was macht die KI?**  
        Sie betrachtet immer die **letzten X Messwerte** (Fenstergröße) und merkt sich daraus das **normale Verhalten**.  
        Den neuesten Wert vergleicht sie mit diesem Muster: passt er **nicht**, gilt er als **Anomalie**.

        - **Fenstergröße:** Anzahl der letzten Messwerte, die als Referenz dienen.  
          → Beispiel: Fenstergröße = 600 = die letzten 600 Messwerte.  
        - **Kontamination:** Erwarteter Anteil an Ausreißern im Normalbetrieb (z. B. 0.02 = 2 %).  
        - **ML-Alert-Schwelle:** Score 0..1 – ab welchem Wert die KI Alarm gibt (niedrig = empfindlich, hoch = tolerant).

        👉 **Merksatz:** Die KI merkt sich das normale Verhalten des Gleichrichters.  
        Wenn ein neuer Wert nicht ins Muster passt, meldet sie eine Anomalie – auch ohne feste Grenzwertverletzung.
        """
    )

    # 👉 Dein gewünschter Zusatzblock (Schritt für Schritt)
    st.markdown("### 🧠 IsolationForest – Schritt für Schritt")
    st.markdown("""
    1. **Sie nimmt die letzten X Messpunkte** (Fenstergröße, z. B. 600).  
       → Jeder Messpunkt hat **Temperatur, Strom, Spannung, Vibration, Lüfter**.

    2. **Sie baut aus diesen Punkten ein Modell:**  
       → Viele **Entscheidungsbäume („Forest“)** teilen die Punkte in Gruppen auf.

    3. **Sie bewertet den neuesten Punkt (den letzten deiner Zeitreihe):**  
       → **Passt** er ins Muster der 599 davor → **normal**.  
       → **Lässt** er sich extrem schnell **isolieren** → **ungewöhnlich = Anomalie**.

    4. **Was macht sie dann damit?**  
       → Sie gibt einen **Anomalie-Score** *(0 = normal, 1 = sehr ungewöhnlich).*  
       → **Liegt der Score über deiner eingestellten Schwelle**, erzeugt die App **automatisch eine Alarmmeldung** (Tab **Alerts**) und die **Health-Badge** geht auf **WARN/ALERT**.
    """)

    c1, c2, c3 = st.columns(3)
    window = c1.slider("Fenstergröße (Punkte)", 200, 2000, 600, 50, key="ml_window")
    contamination = c2.slider("Kontamination (erwartete Ausreißer)", 0.001, 0.10, 0.02, 0.001, key="ml_cont")
    ml_alert_thresh = c3.slider("ML-Alert-Schwelle (0–1)", 0.10, 0.90, 0.80, 0.05, key="ml_thresh")

    st.markdown("---")
    st.subheader("Simulation Control")
    if not st.session_state.running:
        if st.button("▶️ Start Simulation", use_container_width=True):
            st.session_state.running = True
            st.experimental_rerun()
    else:
        if st.button("⏹ Stop Simulation", use_container_width=True):
            st.session_state.running = False

# ---------------- SIMULATOR ----------------
def generate_sample(t: int):
    base = {"temperature_c":45.0, "vibration_rms":0.35, "current_a":120.0, "voltage_v":540.0, "fan_rpm":3200.0}
    if st.session_state.faults.get("cooling"):
        base["temperature_c"] += 0.01 * t
    if st.session_state.faults.get("fan"):
        base["fan_rpm"] -= 0.5 * t
    if st.session_state.faults.get("voltage"):
        base["voltage_v"] += 20 * np.sin(t / 3.0)
    base["temperature_c"] += np.random.uniform(-0.2, 0.2)
    base["vibration_rms"] += np.random.uniform(-0.02, 0.02)
    base["current_a"]     += np.random.uniform(-2, 2)
    base["voltage_v"]     += np.random.uniform(-1.5, 1.5)
    base["fan_rpm"]       += np.random.uniform(-30, 30)
    return base

def push_alarm(ts, level, msg):
    st.session_state.alarms.append({"ts": ts, "level": level, "message": msg})

def check_thresholds(vals, ts):
    for k, v in THRESHOLDS.items():
        val = float(vals[k])
        if k == "fan_rpm":
            if val < v["alert"]:
                push_alarm(ts, "ALERT", f"{k} zu niedrig: {val:.1f} RPM")
            elif val < v["warn"]:
                push_alarm(ts, "WARN", f"{k} niedrig: {val:.1f} RPM")
        else:
            if val > v["alert"]:
                push_alarm(ts, "ALERT", f"{k} zu hoch: {val:.1f}")
            elif val > v["warn"]:
                push_alarm(ts, "WARN", f"{k} hoch: {val:.1f}")

def ml_anomaly(df: pd.DataFrame, window: int, contamination: float):
    if len(df) < window:
        return None, None
    data = df.iloc[-window:].copy()
    X = data[METRICS].astype(float).to_numpy()
    mu = X.mean(axis=0); sigma = X.std(axis=0); sigma[sigma == 0] = 1e-6
    Z = (X - mu) / sigma
    Z_train, z_last = Z[:-1], Z[-1].reshape(1, -1)
    model = IsolationForest(contamination=contamination, random_state=42)
    model.fit(Z_train)
    raw_last = -model.decision_function(z_last)[0]
    raw_train = -model.decision_function(Z_train)
    lo, hi = float(raw_train.min()), float(raw_train.max()) + 1e-9
    score = (raw_last - lo) / (hi - lo)
    return float(score), {"mu": mu.tolist(), "sigma": sigma.tolist()}

def overall_level(th_levels, ml_score, ml_thresh):
    order = {"OK": 0, "WARN": 1, "ALERT": 2}
    level = "OK"
    for lv in th_levels:
        if order[lv] > order[level]:
            level = lv
    if ml_score is not None:
        if ml_score >= ml_thresh:
            level = "ALERT"
        elif ml_score >= (ml_thresh * 0.7) and order["WARN"] > order[level]:
            level = "WARN"
    return level

# ---------------- LIVE LOOP ----------------
if st.session_state.running:
    t = len(st.session_state.df)
    vals = generate_sample(t)
    ts = datetime.now().strftime("%H:%M:%S")
    row = {"ts": ts, "equipment_id": equipment_id, **vals}
    st.session_state.df = pd.concat([st.session_state.df, pd.DataFrame([row])], ignore_index=True)

    check_thresholds(vals, ts)
    score, _ = ml_anomaly(st.session_state.df, window=window, contamination=contamination)
    if score is not None:
        if score >= ml_alert_thresh:
            push_alarm(ts, "ALERT", f"ML anomaly score={score:.2f}")
        elif score >= ml_alert_thresh * 0.7:
            push_alarm(ts, "WARN", f"ML anomaly score={score:.2f}")

    status_placeholder.success(f"RUNNING – Last sample @ {ts}")
    time.sleep(1)
    st.experimental_rerun()
else:
    status_placeholder.warning("Simulation gestoppt")

# ---------------- OVERVIEW ----------------
with tab_overview:
    st.subheader("Gesamtzustand")
    if len(st.session_state.df):
        latest = st.session_state.df.iloc[-1]
        kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
        kpi1.metric("Temperatur (°C)", f"{latest['temperature_c']:.1f}")
        kpi2.metric("Vibration (RMS)", f"{latest['vibration_rms']:.2f}")
        kpi3.metric("Strom (A)", f"{latest['current_a']:.1f}")
        kpi4.metric("Spannung (V)", f"{latest['voltage_v']:.1f}")
        kpi5.metric("Lüfter (RPM)", f"{latest['fan_rpm']:.0f}")

        th_levels = []
        for k, v in THRESHOLDS.items():
            val = float(latest[k])
            if k == "fan_rpm":
                th_levels.append("ALERT" if val < v["alert"] else "WARN" if val < v["warn"] else "OK")
            else:
                th_levels.append("ALERT" if val > v["alert"] else "WARN" if val > v["warn"] else "OK")

        score, _ = ml_anomaly(st.session_state.df, window=window, contamination=contamination)
        lvl = overall_level(th_levels, score, ml_alert_thresh)

        colA, colB = st.columns([1, 3])
        with colA:
            badge = {"OK": "✅ OK", "WARN": "🟠 WARN", "ALERT": "🔴 ALERT"}[lvl]
            st.markdown(f"**Health:** {badge}")
        with colB:
            st.caption(f"ML-Score: {score:.2f}" if score is not None else "ML-Score: – (zu wenig Daten)")
    else:
        st.info("Noch keine Daten.")

# ---------------- LIVE CHARTS ----------------
with tab_live:
    if len(st.session_state.df):
        st.line_chart(st.session_state.df.set_index("ts")[METRICS])
    else:
        st.info("Noch keine Daten.")

# ---------------- ALERTS ----------------
with tab_alerts:
    st.subheader("Alarm-Feed (neueste zuerst)")
    if st.session_state.alarms:
        for a in reversed(st.session_state.alarms[-200:]):
            if a["level"] == "ALERT":
                st.error(f"[{a['ts']}] {a['message']}")
            else:
                st.warning(f"[{a['ts']}] {a['message']}")
    else:
        st.info("Keine Alarme.")
