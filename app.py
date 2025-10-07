import time
import io
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime
from sklearn.ensemble import IsolationForest

st.set_page_config(page_title="Predictive Maintenance – Rectifier", page_icon="🛠️", layout="wide")

# ---------------- EQUIPMENT-STAMMDATEN ----------------
EQUIPMENTS = {
    "10109812-01": {"name": "Gleichrichter XD1", "location": "Schaltschrank 1 – Galvanik Halle (Schüttgutbereich)"},
    "10109812-02": {"name": "Gleichrichter XD2", "location": "Schaltschrank 2 – Galvanik Halle (Schüttgutbereich)"},
}

# ---------------- STATE ----------------
if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame(
        columns=["ts","equipment_id","temperature_c","vibration_rms","current_a","voltage_v","fan_rpm"]
    )
if "alarms" not in st.session_state:
    st.session_state.alarms = []   # dicts: {"ts","level","message"}
if "running" not in st.session_state:
    st.session_state.running = False
if "faults" not in st.session_state:
    st.session_state.faults = {"cooling": False, "fan": False, "voltage": False}
if "eq_num" not in st.session_state:
    st.session_state.eq_num = "10109812-01"  # Default

# ---------------- SCHWELLWERTE ----------------
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

    eq_num = st.selectbox(
        "EQ-Nummer",
        options=list(EQUIPMENTS.keys()),
        index=list(EQUIPMENTS.keys()).index(st.session_state.eq_num),
        help="Wähle ein Equipment. Name & Standort werden automatisch gesetzt.",
    )
    st.session_state.eq_num = eq_num
    st.markdown(f"**Equipment:** {EQUIPMENTS[eq_num]['name']}")
    st.markdown(f"**Standort:** {EQUIPMENTS[eq_num]['location']}")
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

    st.subheader("Fault Injection (während des Laufs umschaltbar)")
    c1, c2, c3 = st.columns(3)
    with c1: st.session_state.faults["cooling"] = st.checkbox("Cooling Degradation — Temperatur steigt")
    with c2: st.session_state.faults["fan"]     = st.checkbox("Fan Wear — Lüfter RPM sinkt")
    with c3: st.session_state.faults["voltage"] = st.checkbox("Voltage Spikes — sporadische Spannungsspitzen")

    st.markdown("---")
    st.subheader("Schwellwerte")
    r1, r2 = st.columns([3,3])
    with r1:
        st.number_input("Temperatur WARN (°C)", value=THRESHOLDS["temperature_c"]["warn"], step=1.0, key="t_warn")
        st.number_input("Temperatur ALERT (°C)", value=THRESHOLDS["temperature_c"]["alert"], step=1.0, key="t_alert")
        st.number_input("Vibration WARN (RMS)", value=THRESHOLDS["vibration_rms"]["warn"], step=0.01, format="%.2f", key="vib_warn")
        st.number_input("Vibration ALERT (RMS)", value=THRESHOLDS["vibration_rms"]["alert"], step=0.01, format="%.2f", key="vib_alert")
    with r2:
        st.number_input("Strom WARN (A)", value=THRESHOLDS["current_a"]["warn"], step=1.0, key="i_warn")
        st.number_input("Strom ALERT (A)", value=THRESHOLDS["current_a"]["alert"], step=1.0, key="i_alert")
        st.number_input("Spannung WARN (V)", value=THRESHOLDS["voltage_v"]["warn"], step=1.0, key="u_warn")
        st.number_input("Spannung ALERT (V)", value=THRESHOLDS["voltage_v"]["alert"], step=1.0, key="u_alert")
        st.number_input("Lüfter WARN (RPM, Untergrenze)", value=THRESHOLDS["fan_rpm"]["warn"], step=50.0, key="fan_warn")
        st.number_input("Lüfter ALERT (RPM, Untergrenze)", value=THRESHOLDS["fan_rpm"]["alert"], step=50.0, key="fan_alert")

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

    c1, c2, c3 = st.columns(3)
    window = c1.slider("Fenstergröße (Punkte)", 200, 2000, 600, 50, key="ml_window")
    contamination = c2.slider("Kontamination (erwartete Ausreißer)", 0.001, 0.10, 0.02, 0.001, key="ml_cont")
    ml_alert_thresh = c3.slider("ML-Alert-Schwelle (0–1)", 0.10, 0.90, 0.80, 0.05, key="ml_thresh")

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

# ---------------- FUNKTIONEN ----------------
def generate_sample(t: int):
    base = {"temperature_c":45.0, "vibration_rms":0.35, "current_a":120.0, "voltage_v":540.0, "fan_rpm":3200.0}
    if st.session_state.faults.get("cooling"): base["temperature_c"] += 0.01 * t
    if st.session_state.faults.get("fan"):     base["fan_rpm"] -= 0.5 * t
    if st.session_state.faults.get("voltage"): base["voltage_v"] += 20 * np.sin(t / 3.0)
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
            warn_thr  = st.session_state.get("fan_warn", THRESHOLDS["fan_rpm"]["warn"])
            alert_thr = st.session_state.get("fan_alert", THRESHOLDS["fan_rpm"]["alert"])
            if val < alert_thr: push_alarm(ts, "ALERT", f"{k} zu niedrig: {val:.1f} RPM")
            elif val < warn_thr: push_alarm(ts, "WARN", f"{k} niedrig: {val:.1f} RPM")
        else:
            if val > v["alert"]: push_alarm(ts, "ALERT", f"{k} zu hoch: {val:.1f}")
            elif val > v["warn"]: push_alarm(ts, "WARN", f"{k} hoch: {val:.1f}")

def ml_anomaly(df: pd.DataFrame, window: int, contamination: float):
    if len(df) < window: return None, None
    data = df.iloc[-window:].copy()
    X = data[METRICS].astype(float).to_numpy()
    mu = X.mean(axis=0); sigma = X.std(axis=0); sigma[sigma == 0] = 1e-6
    Z = (X - mu) / sigma
    Z_train, z_last = Z[:-1], Z[-1].reshape(1, -1)
    model = IsolationForest(contamination=contamination, random_state=42)
    model.fit(Z_train)
    raw_last  = -model.decision_function(z_last)[0]
    raw_train = -model.decision_function(Z_train)
    lo, hi = float(raw_train.min()), float(raw_train.max()) + 1e-9
    score = (raw_last - lo) / (hi - lo)
    return float(score), {"mu": mu.tolist(), "sigma": sigma.tolist()}

def overall_level(th_levels, ml_score, ml_thresh):
    order = {"OK": 0, "WARN": 1, "ALERT": 2}
    level = "OK"
    for lv in th_levels:
        if order[lv] > order[level]: level = lv
    if ml_score is not None:
        if ml_score >= ml_thresh: level = "ALERT"
        elif ml_score >= (ml_thresh * 0.7) and order["WARN"] > order[level]: level = "WARN"
    return level

def build_analysis_df():
    """Eine Liste: jeder Alarm + Mess-Kontext am selben ts."""
    if not st.session_state.alarms:
        return pd.DataFrame(columns=[
            "ts","equipment_id","level","message",
            "temperature_c","vibration_rms","current_a","voltage_v","fan_rpm"
        ])
    df_alerts = pd.DataFrame(st.session_state.alarms).copy()
    df_alerts["equipment_id"] = st.session_state.eq_num
    df_ts = st.session_state.df.copy()
    merged = df_alerts.merge(
        df_ts[["ts","equipment_id","temperature_c","vibration_rms","current_a","voltage_v","fan_rpm"]],
        on=["ts","equipment_id"], how="left"
    )
    cols = ["ts","equipment_id","level","message","temperature_c","vibration_rms","current_a","voltage_v","fan_rpm"]
    return merged.reindex(columns=cols)

# ---------------- LIVE LOOP ----------------
if st.session_state.running:
    t = len(st.session_state.df)
    vals = generate_sample(t)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = {"ts": ts, "equipment_id": st.session_state.eq_num, **vals}
    st.session_state.df = pd.concat([st.session_state.df, pd.DataFrame([row])], ignore_index=True)

    check_thresholds(vals, ts)

    score, _ = ml_anomaly(st.session_state.df, window=window, contamination=contamination)
    if score is not None:
        if score >= ml_alert_thresh: push_alarm(ts, "ALERT", f"ML anomaly score={score:.2f}")
        elif score >= ml_alert_thresh * 0.7: push_alarm(ts, "WARN", f"ML anomaly score={score:.2f}")

    status_placeholder.success(f"RUNNING – Last sample @ {ts}")
    time.sleep(1)
    st.experimental_rerun()
else:
    status_placeholder.warning("Simulation gestoppt")

# ---------------- OVERVIEW ----------------
with tab_overview:
    st.subheader("Gesamtzustand")

    # Export rechts oben: eine Liste (CSV + Excel)
    exp_l, exp_r = st.columns([3,2])
    with exp_r:
        analysis_df = build_analysis_df()
        st.download_button(
            "⬇️ Export Analyse (CSV)",
            data=analysis_df.to_csv(index=False).encode("utf-8"),
            file_name=f"analysis_{st.session_state.eq_num}.csv",
            mime="text/csv",
            use_container_width=True,
            key="dl_analysis_csv_overview",
        )
        try:
            import openpyxl  # noqa: F401
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as xw:
                analysis_df.to_excel(xw, index=False, sheet_name="analysis")
            st.download_button(
                "⬇️ Export Analyse (Excel)",
                data=buf.getvalue(),
                file_name=f"analysis_{st.session_state.eq_num}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                key="dl_analysis_xlsx_overview",
            )
        except Exception:
            st.info("Für Excel-Export `openpyxl` in requirements.txt ergänzen (z. B. openpyxl==3.1.5).")

    if len(st.session_state.df):
        latest = st.session_state.df.iloc[-1]
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Temperatur (°C)", f"{latest['temperature_c']:.1f}")
        k2.metric("Vibration (RMS)", f"{latest['vibration_rms']:.2f}")
        k3.metric("Strom (A)", f"{latest['current_a']:.1f}")
        k4.metric("Spannung (V)", f"{latest['voltage_v']:.1f}")
        k5.metric("Lüfter (RPM)", f"{latest['fan_rpm']:.0f}")

        th_levels = []
        for k, v in THRESHOLDS.items():
            val = float(latest[k])
            if k == "fan_rpm":
                warn_thr  = st.session_state.get("fan_warn", THRESHOLDS["fan_rpm"]["warn"])
                alert_thr = st.session_state.get("fan_alert", THRESHOLDS["fan_rpm"]["alert"])
                th_levels.append("ALERT" if val < alert_thr else "WARN" if val < warn_thr else "OK")
            else:
                th_levels.append("ALERT" if val > v["alert"] else "WARN" if val > v["warn"] else "OK")

        score, _ = ml_anomaly(st.session_state.df, window=window, contamination=contamination)
        lvl = overall_level(th_levels, score, ml_alert_thresh)

        cA, cB = st.columns([1,3])
        with cA:
            badge = {"OK": "✅ OK", "WARN": "🟠 WARN", "ALERT": "🔴 ALERT"}[lvl]
            st.markdown(f"**Health:** {badge}")
        with cB:
            st.caption(f"ML-Score: {score:.2f}" if score is not None else "ML-Score: – (zu wenig Daten)")
    else:
        st.info("Noch keine Daten.")

# ---------------- LIVE CHARTS ----------------
with tab_live:
    st.subheader("Live Charts")
    if len(st.session_state.df):
        st.line_chart(st.session_state.df.set_index("ts")[METRICS])
    else:
        st.info("Noch keine Daten.")

# ---------------- ALERTS ----------------
with tab_alerts:
    st.subheader("Alarm-Feed (neueste zuerst)")
    if st.session_state.alarms:
        for a in reversed(st.session_state.alarms[-200:]):
            (st.error if a["level"] == "ALERT" else st.warning)(f"[{a['ts']}] {a['message']}")
    else:
        st.info("Keine Alarme.")

# ---------------- SONSTIGES (Vorzeigen) ----------------
with tab_misc:
    st.subheader("IsolationForest – Normal vs. Anomalie (Illustration)")
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyBboxPatch, ArrowStyle

        # Scatter: Normal vs. Anomalie
        rng = np.random.default_rng(42)
        n = 150
        temp_norm = 50 + rng.normal(0, 2.0, n)
        curr_norm = 120 + rng.normal(0, 3.0, n)
        anom_temp, anom_curr = 65.0, 120.0

        fig_sc, ax_sc = plt.subplots(figsize=(8,5))
        ax_sc.scatter(temp_norm, curr_norm, s=25, label="Normal")
        ax_sc.scatter([anom_temp],[anom_curr], s=80, marker="x", linewidths=2.5, label="Anomalie")
        ax_sc.annotate("Anomalie", xy=(anom_temp, anom_curr),
                       xytext=(anom_temp-7, anom_curr+8),
                       arrowprops=dict(arrowstyle="->", lw=1.5))
        ax_sc.set_xlabel("Temperatur (°C)")
        ax_sc.set_ylabel("Strom (A)")
        ax_sc.legend()
        st.pyplot(fig_sc, use_container_width=True)

        st.markdown("---")
        st.subheader("Entscheidungsbaum – vereinfachte Logik (Illustration)")
        # Ein konsistenter Mini-Baum (einfach erklärbar)
        fig, ax = plt.subplots(figsize=(9,4))
        ax.axis("off")
        def box(xy, text):
            b = FancyBboxPatch(xy, 0.36, 0.18, boxstyle="round,pad=0.02", fc="#E6F2FF", ec="#3973AC", lw=1.5)
            ax.add_patch(b)
            ax.text(xy[0]+0.18, xy[1]+0.09, text, ha="center", va="center", fontsize=9, weight="bold")
        # Knoten
        box((0.05, 0.62), "Spannung > 600 V?")
        box((0.48, 0.62), "Ja → Ausreißer")
        box((0.05, 0.32), "Nein → Temp < 50 °C?")
        box((0.05, 0.02), "Ja → Normal")
        box((0.48, 0.32), "Nein → Vibration > 0.8?")
        box((0.48, 0.02), "Ja → Ausreißer")
        box((0.78, 0.02), "Nein → Normal")
        arr = ArrowStyle("-|>", head_length=1.0, head_width=0.6)
        ax.annotate("", xy=(0.41,0.41), xytext=(0.23,0.62), arrowprops=dict(arrowstyle=arr, lw=1.4))  # root->left
        ax.annotate("", xy=(0.48,0.70), xytext=(0.23,0.70), arrowprops=dict(arrowstyle=arr, lw=1.4))  # root->right
        ax.annotate("", xy=(0.23,0.11), xytext=(0.23,0.32), arrowprops=dict(arrowstyle=arr, lw=1.4))  # temp->normal
        ax.annotate("", xy=(0.66,0.41), xytext=(0.41,0.41), arrowprops=dict(arrowstyle=arr, lw=1.4))  # temp->vib
        ax.annotate("", xy=(0.66,0.11), xytext=(0.66,0.32), arrowprops=dict(arrowstyle=arr, lw=1.4))  # vib->anomaly
        ax.annotate("", xy=(0.83,0.11), xytext=(0.66,0.11), arrowprops=dict(arrowstyle=arr, lw=1.4))  # vib->normal
        st.pyplot(fig, use_container_width=True)

    except Exception:
        st.warning("Matplotlib fehlt – bitte `matplotlib` in requirements.txt ergänzen.")

    st.markdown("---")
    st.subheader("Beispiel-Export (eine Liste – Vorschau)")
    preview = build_analysis_df()
    if preview.empty:
        demo_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        demo = pd.DataFrame(
            [
                {"ts": demo_ts, "equipment_id": st.session_state.eq_num, "level": "WARN",  "message": "voltage_v hoch: 602.3",
                 "temperature_c": 45.2, "vibration_rms": 0.36, "current_a": 121.0, "voltage_v": 602.3, "fan_rpm": 3180},
                {"ts": demo_ts, "equipment_id": st.session_state.eq_num, "level": "ALERT", "message": "ML anomaly score=0.86",
                 "temperature_c": 45.2, "vibration_rms": 0.36, "current_a": 121.0, "voltage_v": 541.2, "fan_rpm": 3180},
            ]
        )
        st.dataframe(demo, use_container_width=True, hide_index=True)
    else:
        st.dataframe(preview.tail(10), use_container_width=True, hide_index=True)
