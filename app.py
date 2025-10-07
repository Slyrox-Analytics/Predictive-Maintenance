import streamlit as st
import numpy as np
import pandas as pd
import time
from sklearn.ensemble import IsolationForest

# -------------------------------
# App-Konfiguration
# -------------------------------
st.set_page_config(
    page_title="Predictive Maintenance – Gleichrichter",
    layout="wide"
)

# Equipment ID
equipment_id = "RECT-0001"

# Sidebar Navigation
menu = ["Overview", "Live Charts", "Alerts", "Settings"]
choice = st.sidebar.radio("Navigation", menu)

# Simulations-States (Session)
if "running" not in st.session_state:
    st.session_state.running = False
if "data" not in st.session_state:
    st.session_state.data = pd.DataFrame(columns=["time","temp","current","voltage","vibration","rpm","anomaly_score","alert"])


# -------------------------------
# Settings
# -------------------------------
if choice == "Settings":
    st.header("Fault Injection (während des Laufs umschaltbar)")
    fault_temp = st.checkbox("Cooling Degradation — Temperatur steigt")
    fault_fan = st.checkbox("Fan Wear — Lüfter RPM sinkt")
    fault_voltage = st.checkbox("Voltage Spikes — sporadische Spannungs-Ausreißer")

    st.subheader("Schwellwerte")
    temp_warn = st.number_input("Temperatur WARN (°C)", value=60.0)
    temp_alert = st.number_input("Temperatur ALERT (°C)", value=70.0)
    vib_warn = st.number_input("Vibration WARN (RMS)", value=0.60)
    vib_alert = st.number_input("Vibration ALERT (RMS)", value=0.80)
    i_warn = st.number_input("Strom WARN (A)", value=150.0)
    i_alert = st.number_input("Strom ALERT (A)", value=180.0)
    u_warn = st.number_input("Spannung WARN (V)", value=580.0)
    u_alert = st.number_input("Spannung ALERT (V)", value=620.0)
    rpm_warn = st.number_input("RPM WARN (unter)", value=2600.0)
    rpm_alert = st.number_input("RPM ALERT (unter)", value=2000.0)

    st.subheader("KI-Anomalie (IsolationForest)")

    st.markdown("""
    **Fensterprinzip (Bewertung):**
    - Die KI betrachtet ein Fenster der letzten Messwerte (z. B. 600).
    - Jeder Messwert besteht aus Temperatur, Strom, Spannung, Vibration und Lüfter.
    - Aus den 599 vergangenen Punkten lernt sie das **normale Verhalten**.
    - Den neuesten (600.) vergleicht sie damit:
      - passt er ins Muster → **normal**
      - weicht er stark ab → **Anomalie**

    **IsolationForest-Erklärung:**
    - **Name:** „Isolation“ = etwas isolieren, „Forest“ = viele kleine Entscheidungsbäume (wie ein Wald).
    - **Idee:** Anstatt nach Gemeinsamkeiten zu suchen, versucht der Algorithmus, ungewöhnliche Punkte möglichst schnell „abzuspalten“.
    - Er baut viele zufällige Entscheidungsbäume („Forest“).
    - Jeder Baum trennt die Daten nach Zufallsregeln (z. B. „Temperatur < 50 °C?“ → Ja/Nein).
    - **Normale Werte** brauchen viele Trennschritte, bis sie isoliert sind.
    - **Ausreißer** werden sehr schnell isoliert → weil sie nicht ins Muster passen.
    """)

    window_size = st.slider("Fenstergröße (Punkte)", 200, 2000, 600)
    contamination = st.slider("Kontamination (erwartete Ausreißer)", 0.0, 0.1, 0.02, step=0.01)
    ml_threshold = st.slider("ML-Alert-Schwelle (0-1)", 0.1, 0.9, 0.8)


    st.subheader("Simulation Control")
    if st.button("Start Simulation"):
        st.session_state.running = True
    if st.button("Stop Simulation"):
        st.session_state.running = False


# -------------------------------
# Overview
# -------------------------------
elif choice == "Overview":
    st.title("🔧 Predictive Maintenance – Gleichrichter")
    st.text(f"Equipment-ID: {equipment_id}")

    if st.session_state.running:
        st.success("Simulation läuft...")
    else:
        st.warning("Simulation gestoppt")

    st.subheader("Gesamtzustand")
    if len(st.session_state.data) == 0:
        st.info("Noch keine Daten.")
    else:
        last = st.session_state.data.iloc[-1]
        if last["alert"] == "ALERT":
            st.error("🚨 ALERT – Kritische Abweichung!")
        elif last["alert"] == "WARN":
            st.warning("⚠️ WARNUNG – Werte außerhalb der Norm")
        else:
            st.success("✅ Normalbetrieb")


# -------------------------------
# Live Charts
# -------------------------------
elif choice == "Live Charts":
    st.header("Live Charts")
    if len(st.session_state.data) == 0:
        st.info("Noch keine Daten.")
    else:
        st.line_chart(st.session_state.data[["temp","current","voltage","vibration","rpm"]])


# -------------------------------
# Alerts
# -------------------------------
elif choice == "Alerts":
    st.header("Alarm-Feed (neueste zuerst)")
    if len(st.session_state.data) == 0:
        st.info("Keine Alarme.")
    else:
        alerts = st.session_state.data[st.session_state.data["alert"]!="OK"].sort_values("time",ascending=False)
        if len(alerts)==0:
            st.success("Kein Alarm aktiv.")
        else:
            st.write(alerts[["time","temp","current","voltage","vibration","rpm","alert"]])


# -------------------------------
# Simulation Loop
# -------------------------------
if st.session_state.running:
    temp = np.random.normal(55,5)
    current = np.random.normal(140,10)
    voltage = np.random.normal(600,10)
    vibration = np.random.normal(0.5,0.1)
    rpm = np.random.normal(2800,200)

    # TODO: Fault Injection hier einbauen (vereinfachtes Beispiel)
    if "fault_temp" in locals() and fault_temp:
        temp += 20
    if "fault_fan" in locals() and fault_fan:
        rpm -= 1000
    if "fault_voltage" in locals() and fault_voltage:
        if np.random.rand()>0.8:
            voltage += 50

    # ML-Anomalie-Bewertung
    df = st.session_state.data.copy()
    new_point = pd.DataFrame([{
        "time": time.strftime("%H:%M:%S"),
        "temp": temp,
        "current": current,
        "voltage": voltage,
        "vibration": vibration,
        "rpm": rpm
    }])

    if len(df) > 20:
        features = df[["temp","current","voltage","vibration","rpm"]].values
        iso = IsolationForest(contamination=contamination, random_state=42)
        iso.fit(features[-window_size:])
        score = -iso.decision_function(new_point[["temp","current","voltage","vibration","rpm"]])[0]
        new_point["anomaly_score"] = score
        new_point["alert"] = "ALERT" if score > ml_threshold else "OK"
    else:
        new_point["anomaly_score"] = 0
        new_point["alert"] = "OK"

    # Schwellwerte
    if temp > temp_alert or current > i_alert or voltage > u_alert or vibration > vib_alert or rpm < rpm_alert:
        new_point["alert"] = "ALERT"
    elif temp > temp_warn or current > i_warn or voltage > u_warn or vibration > vib_warn or rpm < rpm_warn:
        new_point["alert"] = "WARN"

    st.session_state.data = pd.concat([st.session_state.data, new_point], ignore_index=True)
