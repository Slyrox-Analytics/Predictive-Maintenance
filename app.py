import time
import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(
    page_title="Predictive Maintenance – Rectifier",
    page_icon="🛠️",
    layout="wide"
)

# --- APP STATE (wird gleich vom Simulator befüllt) ---
if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame(columns=["ts","equipment_id","temperature_c","vibration_rms","current_a","voltage_v","fan_rpm"])
if "alarms" not in st.session_state:
    st.session_state.alarms = []  # Liste von dicts

# --- HEADER ---
colL, colR = st.columns([1,1])
with colL:
    st.markdown("### 🛠️ Predictive Maintenance – Gleichrichter")
    equipment_id = st.text_input("Equipment-ID", value="RECT-0001", help="Dummy-Nummer wie in SAP (z. B. RECT-0001).")
with colR:
    st.markdown("#### Live-Status")
    status_placeholder = st.empty()
    status_placeholder.info("Warte auf Daten …")

st.markdown("---")

# --- LAYOUT: Tabs wie in Enterprise-Dashboards ---
tab_overview, tab_live, tab_alerts, tab_settings = st.tabs(["Overview", "Live Charts", "Alerts", "Settings"])

with tab_overview:
    st.subheader("Gesamtzustand")
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    kpi1.metric("Temperatur (°C)", "–")
    kpi2.metric("Vibration (RMS)", "–")
    kpi3.metric("Strom (A)", "–")
    kpi4.metric("Spannung (V)", "–")
    kpi5.metric("Lüfter (RPM)", "–")

    st.caption("Hinweis: Hier erscheinen gleich KPIs und eine Health-Badge. Im nächsten Schritt fügen wir den Simulator und Schwellwerte/Anomalien hinzu.")

with tab_live:
    st.subheader("Live-Zeitreihen")
    st.line_chart(st.session_state.df.set_index("ts")[["temperature_c"]]) if len(st.session_state.df) else st.info("Noch keine Daten.")
    st.line_chart(st.session_state.df.set_index("ts")[["vibration_rms"]]) if len(st.session_state.df) else None
    st.line_chart(st.session_state.df.set_index("ts")[["current_a"]]) if len(st.session_state.df) else None
    st.line_chart(st.session_state.df.set_index("ts")[["voltage_v"]]) if len(st.session_state.df) else None
    st.line_chart(st.session_state.df.set_index("ts")[["fan_rpm"]]) if len(st.session_state.df) else None

with tab_alerts:
    st.subheader("Alarm-Feed")
    if st.session_state.alarms:
        for a in reversed(st.session_state.alarms):
            st.warning(f"[{a['ts']}] {a['level']} – {a['message']}")
    else:
        st.info("Noch keine Alarme.")

with tab_settings:
    st.subheader("Einstellungen (werden im nächsten Schritt aktiv)")
    st.write("- Schwellenwerte pro Metrik (WARN/ALERT)")
    st.write("- Fault Injection (z. B. Lüfterausfall, Spannungsschwankung)")
    st.write("- KI-Anomalie (IsolationForest)")

# Fußzeile
st.markdown("---")
st.caption("Demo-UI: nah an SAP PdM – Equipment, Live-Daten, Alarme, Settings. Nächster Schritt: Simulator + Schwellwerte + KI.")
