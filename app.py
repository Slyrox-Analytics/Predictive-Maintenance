import time
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime
from sklearn.ensemble import IsolationForest

st.set_page_config(page_title="Predictive Maintenance – Rectifier", page_icon="🛠️", layout="wide")

# ---------------- EQUIPMENT-STAMMDATEN ----------------
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
if "eq_num" not in st.session_state:
    st.session_state.eq_num = "10109812-01"  # Default

# Default thresholds
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

    # Dropdown für EQ-Nummer + automatische Anzeige der Stammdaten
    eq_num = st.selectbox(
        "EQ-Nummer",
        options=list(EQUIPMENTS.keys()),
        index=list(EQUIPMENTS.keys()).index(st.session_state.eq_num),
        help="Wähle ein Equipment. Name & Standort werden automatisch gesetzt.",
    )
    st.session_state.eq_num = eq_num
    eq_name = EQUIPMENTS[eq_num]["name"]
    eq_loc  = EQUIPMENTS[eq_num]["location"]

    st.markdown(f"**Equipment:** {eq_name}")
    st.markdown(f"**Standort:** {eq_loc}")

with colR:
    st.markdown("#### Live-Status")
    status_placeholder = st.empty()

st.markdown("---")

# ---------------- TABS ----------------
tab_overview, tab_live, tab_alerts, tab_settings, tab_misc = st.tabs(
    ["Overview", "Live Charts", "Alerts", "Settings", "Sonstiges"]
)

# ---------------- SETTINGS ----------------
with tab_settings:
    # --- Simulation Control (oben) ---
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
        if st.button("🗑️ Daten löschen", help="Löscht NUR Simulationsdaten & Alarmfeed. Einstellungen bleiben erhalten."):
            st.session_state.df = pd.DataFrame(columns=st.session_state.df.columns)
            st.session_state.alarms = []
            st.success("Daten & Alarme gelöscht. Einstellungen unverändert.")

    st.markdown("---")

    # Fault Injection
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
    st.subheader("KI-Anomalie (IsolationForest)")

    st.markdown(
        """
**Fensterprinzip (Bewertung):**
- Die KI betrachtet ein Fenster der letzten Messwerte (z. B. 600).
- Jeder Messwert besteht aus Temperatur, Strom, Spannung, Vibration und Lüfter.
- Aus den 599 vergangenen Punkten lernt sie das **normale Verhalten**.
- Den neuesten (600.) vergleicht sie damit:
  - passt er ins Muster → **normal**
  - weicht er stark ab → **Anomalie**

**IsolationForest-Erklärung:**
- **Name:** „Isolation“ = etwas isolieren, „Forest“ = viele kleine Entscheidungsbäume (wie ein Wald).
- **Idee:** Statt Gemeinsamkeiten zu suchen, trennt der Algorithmus ungewöhnliche Punkte möglichst schnell ab.
- Er baut viele zufällige Entscheidungsbäume („Forest“).
- Jeder Baum trennt die Daten nach Zufallsregeln (z. B. „Temperatur < 50 °C?“ → Ja/Nein).
- **Normale Werte** brauchen viele Trennschritte, bis sie isoliert sind.
- **Ausreißer** werden sehr schnell isoliert → weil sie nicht ins Muster passen.
"""
    )

    # Regler
    c1, c2, c3 = st.columns(3)
    window = c1.slider("Fenstergröße (Punkte)", 200, 2000, 600, 50, key="ml_window")
    contamination = c2.slider("Kontamination (erwartete Ausreißer)", 0.001, 0.10, 0.02, 0.001, key="ml_cont")
    ml_alert_thresh = c3.slider("ML-Alert-Schwelle (0–1)", 0.10, 0.90, 0.80, 0.05, key="ml_thresh")

    # Kurz-Erklärungen unter den Reglern
    st.markdown("---")
    st.markdown("### Erläuterungen zu den Reglern")
    st.markdown("""
**Fenstergröße (Punkte)**  
➡️ Wie viele vergangene Messwerte die KI als Referenz nimmt (z. B. 600 = die letzten 600 Punkte).  
• Großes Fenster = stabiler, reagiert langsamer.  
• Kleines Fenster = reagiert schneller, aber empfindlicher.  

**Kontamination (erwartete Ausreißer)**  
➡️ Erwarteter Anteil an Ausreißern im Normalbetrieb.  
• Beispiel: 0.02 = 2 % der Punkte dürfen unauffällig abweichen, ohne sofort Alarm auszulösen.  
• Klein = empfindlicher (erkennt schneller ungewöhnliche Punkte).  
• Groß = toleranter (meldet nur stärkere Abweichungen).  

**ML-Alert-Schwelle (0–1)**  
➡️ Ab welchem Anomalie-Score das Machine-Learning-Modell (IsolationForest) Alarm gibt.  
• Score nahe 0 = Punkt ist normal.  
• Score nahe 1 = Punkt ist sehr ungewöhnlich.  
• Liegt der Score über dieser Schwelle (z. B. 0.8), wird ein Alarm im Dashboard ausgelöst.
""")

# ---------------- SIMULATOR ----------------
def generate_sample(t: int):
    base = {"temperature_c":45.0, "vibration_rms":0.35, "current_a":120.0, "voltage_v":540.0, "fan_rpm":3200.0}
    # Faults
    if st.session_state.faults.get("cooling"):
        base["temperature_c"] += 0.01 * t
    if st.session_state.faults.get("fan"):
        base["fan_rpm"] -= 0.5 * t
    if st.session_state.faults.get("voltage"):
        base["voltage_v"] += 20 * np.sin(t / 3.0)
    # Noise
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
    row = {"ts": ts, "equipment_id": st.session_state.eq_num, **vals}
    st.session_state.df = pd.concat([st.session_state.df, pd.DataFrame([row])], ignore_index=True)

    # Classical thresholds
    check_thresholds(vals, ts)

    # ML anomaly
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

        # CSV-Export für Messdaten
        data_csv = st.session_state.df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Export Messdaten (CSV)",
            data=data_csv,
            file_name=f"timeseries_{st.session_state.eq_num}.csv",
            mime="text/csv",
            use_container_width=True,
        )
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

    # CSV-Export für Alerts
    if st.session_state.alarms:
        df_alerts = pd.DataFrame(st.session_state.alarms)
        df_alerts["equipment_id"] = st.session_state.eq_num
        alerts_csv = df_alerts.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Export Alerts (CSV)",
            data=alerts_csv,
            file_name=f"alerts_{st.session_state.eq_num}.csv",
            mime="text/csv",
            use_container_width=True,
        )

        for a in reversed(st.session_state.alarms[-200:]):
            if a["level"] == "ALERT":
                st.error(f"[{a['ts']}] {a['message']}")
            else:
                st.warning(f"[{a['ts']}] {a['message']}")
    else:
        st.info("Keine Alarme.")

# ---------------- SONSTIGES ----------------
with tab_misc:
    st.subheader("Beispiel: Entscheidungsbaum (IsolationForest)")

    # Versuch, matplotlib on-demand zu laden (damit App ohne Abhängigkeit weiterläuft)
    MATPLOTLIB_OK = True
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyBboxPatch, ArrowStyle
        from matplotlib import patheffects
    except Exception:
        MATPLOTLIB_OK = False

    if MATPLOTLIB_OK:
        fig, ax = plt.subplots(figsize=(8,4))
        ax.axis("off")

        def add_box(ax, xy, text):
            box = FancyBboxPatch(xy, 0.38, 0.18, boxstyle="round,pad=0.02", fc="#E6F2FF", ec="#3973AC", lw=1.5)
            ax.add_patch(box)
            tx = ax.text(xy[0]+0.19, xy[1]+0.09, text, ha="center", va="center", fontsize=9, weight="bold")
            tx.set_path_effects([patheffects.withStroke(linewidth=3, foreground="white")])

        # Knoten
        add_box(ax, (0.05, 0.65), "Temperatur < 50 °C?")
        add_box(ax, (0.05, 0.30), "Ja → Spannung > 600 V?")
        add_box(ax, (0.55, 0.30), "Nein → normaler Bereich")
        add_box(ax, (0.05, 0.02), "Ja → normal")
        add_box(ax, (0.40, 0.02), "Nein → Ausreißer")

        # Pfeile
        arrow = ArrowStyle("-|>", head_length=1.0, head_width=0.6)
        ax.annotate("", xy=(0.24,0.39), xytext=(0.24,0.65), arrowprops=dict(arrowstyle=arrow, lw=1.5, color="#444"))
        ax.annotate("", xy=(0.55,0.39), xytext=(0.24,0.65), arrowprops=dict(arrowstyle=arrow, lw=1.5, color="#444"))
        ax.annotate("", xy=(0.24,0.11), xytext=(0.24,0.30), arrowprops=dict(arrowstyle=arrow, lw=1.5, color="#444"))
        ax.annotate("", xy=(0.40,0.11), xytext=(0.24,0.30), arrowprops=dict(arrowstyle=arrow, lw=1.5, color="#444"))

        st.pyplot(fig, use_container_width=True)
        st.caption("IsolationForest nutzt viele solcher zufälligen Entscheidungsbäume. Normale Punkte brauchen mehrere Trennschritte, Ausreißer werden schnell isoliert.")

        st.markdown("---")
        st.subheader("Beispiel: Zeitreihe mit Ausreißer")

        # synthetische Reihe mit Ausreißer
        n = 80
        x = np.arange(n)
        y = 45 + 0.2*np.sin(x/4) + np.random.normal(0,0.2,size=n)
        y[55] = y.mean() + 8.0  # Ausreißer

        fig2, ax2 = plt.subplots(figsize=(10,3))
        ax2.plot(x, y, linewidth=1.5)
        ax2.scatter([55],[y[55]], s=80, color="red", zorder=5)
        ax2.set_xlabel("Zeit (Messpunkte)")
        ax2.set_ylabel("Temperatur (°C)")
        ax2.set_title("Temperatur-Verlauf – markierter Ausreißer (rot)")
        st.pyplot(fig2, use_container_width=True)
    else:
        st.warning("Matplotlib ist nicht installiert. Die Beispielgrafiken werden deshalb nicht angezeigt. "
                   "Füge `matplotlib` zur requirements.txt hinzu, um die Grafiken zu sehen.")

    st.markdown("---")
    st.subheader("Beispiele: Excel-Exports (Vorschau)")

    # Beispiel-DataFrame für Alerts (so sieht CSV aus)
    sample_alerts = pd.DataFrame(
        [
            {"ts":"12:01:05","level":"WARN","message":"voltage_v hoch: 602.3","equipment_id":st.session_state.eq_num},
            {"ts":"12:03:10","level":"ALERT","message":"ML anomaly score=0.86","equipment_id":st.session_state.eq_num},
        ]
    )
    st.markdown("**Alerts-CSV (Struktur)**")
    st.dataframe(sample_alerts, use_container_width=True, hide_index=True)

    # Beispiel-DataFrame für Messdaten (so sieht CSV aus)
    sample_ts = pd.DataFrame(
        [
            {"ts":"12:00:00","equipment_id":st.session_state.eq_num,"temperature_c":45.2,"vibration_rms":0.36,"current_a":121.0,"voltage_v":541.2,"fan_rpm":3180},
            {"ts":"12:00:01","equipment_id":st.session_state.eq_num,"temperature_c":45.1,"vibration_rms":0.35,"current_a":120.8,"voltage_v":540.9,"fan_rpm":3195},
        ]
    )
    st.markdown("**Messdaten-CSV (Struktur)**")
    st.dataframe(sample_ts, use_container_width=True, hide_index=True)
