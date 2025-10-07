import time
import io
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, ArrowStyle
from matplotlib import patheffects

st.set_page_config(page_title="Predictive Maintenance – Rectifier", page_icon="🛠️", layout="wide")

# ---------------- EQUIPMENT-STAMMDATEN ----------------
EQUIPMENTS = {
    "10109812-01": {"name": "Gleichrichter XD1", "location": "Schaltschrank 1 – Galvanik Halle (Schüttgutbereich)"},
    "10109812-02": {"name": "Gleichrichter XD2", "location": "Schaltschrank 2 – Galvanik Halle (Schüttgutbereich)"},
}

# ---------------- CSV-SPALTEN (feste Reihenfolge) ----------------
CSV_COLUMNS_TS = ["ts","equipment_id","temperature_c","vibration_rms","current_a","voltage_v","fan_rpm"]
CSV_COLUMNS_ALERTS = ["ts","equipment_id","source","level","metric","value","note"]
CSV_COLUMNS_EVENTS = ["ts","equipment_id","record_type","temperature_c","vibration_rms","current_a","voltage_v","fan_rpm","source","level","metric","value","note"]

# ---------------- STATE ----------------
if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame(columns=CSV_COLUMNS_TS)
if "alarms" not in st.session_state:
    # list of dicts: keys = CSV_COLUMNS_ALERTS
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
tab_overview, tab_live, tab_alerts, tab_settings, tab_misc = st.tabs(["Overview", "Live Charts", "Alerts", "Settings", "Sonstiges"])

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
            st.session_state.df = pd.DataFrame(columns=CSV_COLUMNS_TS)
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
        st.number_input("Temperatur WARN (°C)", value=THRESHOLDS["temperature_c"]["warn"], step=1.0, key="t_warn")
        st.number_input("Temperatur ALERT (°C)", value=THRESHOLDS["temperature_c"]["alert"], step=1.0, key="t_alert")
        st.number_input("Vibration WARN (RMS)", value=THRESHOLDS["vibration_rms"]["warn"], step=0.01, format="%.2f", key="vib_warn")
        st.number_input("Vibration ALERT (RMS)", value=THRESHOLDS["vibration_rms"]["alert"], step=0.01, format="%.2f", key="vib_alert")
    with r2:
        st.number_input("Strom WARN (A)", value=THRESHOLDS["current_a"]["warn"], step=1.0, key="i_warn")
        st.number_input("Strom ALERT (A)", value=THRESHOLDS["current_a"]["alert"], step=1.0, key="i_alert")
        st.number_input("Spannung WARN (V)", value=THRESHOLDS["voltage_v"]["warn"], step=1.0, key="u_warn")
        st.number_input("Spannung ALERT (V)", value=THRESHOLDS["voltage_v"]["alert"], step=1.0, key="u_alert")

    st.markdown("---")
    st.subheader("KI-Anomalie (IsolationForest)")
    st.caption("Die KI bewertet den neuesten Punkt im Fenster gegen das erlernte Normalverhalten.")
    c1, c2, c3 = st.columns(3)
    window = c1.slider("Fenstergröße (Punkte)", 200, 2000, 600, 50, key="ml_window")
    contamination = c2.slider("Kontamination (erwartete Ausreißer)", 0.001, 0.10, 0.02, 0.001, key="ml_cont")
    ml_alert_thresh = c3.slider("ML-Alert-Schwelle (0–1)", 0.10, 0.90, 0.80, 0.05, key="ml_thresh")

# ---------------- SIMULATOR ----------------
def now_ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

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

def push_alarm(ts, level, source, metric, value, note=""):
    st.session_state.alarms.append({
        "ts": ts,
        "equipment_id": st.session_state.eq_num,
        "source": source,        # "THRESHOLD" | "ML"
        "level": level,          # "WARN" | "ALERT"
        "metric": metric,        # e.g. "temperature_c" | "anomaly_score"
        "value": round(float(value), 3) if value is not None else None,
        "note": note
    })

def check_thresholds(vals, ts):
    for k, v in THRESHOLDS.items():
        val = float(vals[k])
        if k == "fan_rpm":
            if val < v["alert"]:
                push_alarm(ts, "ALERT", "THRESHOLD", k, val, "unter Grenzwert alert")
            elif val < v["warn"]:
                push_alarm(ts, "WARN", "THRESHOLD", k, val, "unter Grenzwert warn")
        else:
            if val > v["alert"]:
                push_alarm(ts, "ALERT", "THRESHOLD", k, val, "über Grenzwert alert")
            elif val > v["warn"]:
                push_alarm(ts, "WARN", "THRESHOLD", k, val, "über Grenzwert warn")

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
    ts = now_ts()
    row = {"ts": ts, "equipment_id": st.session_state.eq_num, **vals}
    st.session_state.df = pd.concat([st.session_state.df, pd.DataFrame([row])], ignore_index=True)

    check_thresholds(vals, ts)

    score, _ = ml_anomaly(st.session_state.df, window=window, contamination=contamination)
    if score is not None:
        if score >= ml_alert_thresh:
            push_alarm(ts, "ALERT", "ML", "anomaly_score", score, f"score>= {ml_alert_thresh}")
        elif score >= ml_alert_thresh * 0.7:
            push_alarm(ts, "WARN", "ML", "anomaly_score", score, f"score>= {ml_alert_thresh*0.7:.2f}")

    status_placeholder.success(f"RUNNING – Last sample @ {ts}")
    time.sleep(1)
    st.experimental_rerun()
else:
    status_placeholder.warning("Simulation gestoppt")

# ---------------- OVERVIEW ----------------
with tab_overview:
    # ------- Export-Buttons: oben rechts, jederzeit nutzbar -------
    exp_left, exp_right = st.columns([3,2])
    with exp_right:
        # Messdaten (CSV)
        export_df = st.session_state.df.reindex(columns=CSV_COLUMNS_TS)
        data_csv = export_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Export Messdaten (CSV)",
            data=data_csv,
            file_name=f"timeseries_{st.session_state.eq_num}.csv",
            mime="text/csv",
            use_container_width=True,
        )

        # Alerts (CSV)
        export_alerts = pd.DataFrame(st.session_state.alarms)
        if export_alerts.empty:
            export_alerts = pd.DataFrame(columns=CSV_COLUMNS_ALERTS)
        export_alerts = export_alerts.reindex(columns=CSV_COLUMNS_ALERTS)
        alerts_csv = export_alerts.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Export Alerts (CSV)",
            data=alerts_csv,
            file_name=f"alerts_{st.session_state.eq_num}.csv",
            mime="text/csv",
            use_container_width=True,
        )

        # Events kombiniert (CSV)
        def build_events_df() -> pd.DataFrame:
            df_ts = st.session_state.df.reindex(columns=CSV_COLUMNS_TS).copy()
            df_ts["record_type"] = "MEAS"
            df_ts["source"] = ""
            df_ts["level"] = ""
            df_ts["metric"] = ""
            df_ts["value"] = np.nan
            df_ts["note"] = ""

            df_alerts = pd.DataFrame(st.session_state.alarms)
            if df_alerts.empty:
                df_alerts = pd.DataFrame(columns=CSV_COLUMNS_ALERTS)
            df_alerts = df_alerts.reindex(columns=CSV_COLUMNS_ALERTS)
            # Kontextmesswerte zum gleichen Zeitstempel mergen
            df_alerts_ctx = df_alerts.merge(
                df_ts[["ts","equipment_id","temperature_c","vibration_rms","current_a","voltage_v","fan_rpm"]],
                on=["ts","equipment_id"],
                how="left"
            )
            df_alerts_ctx.insert(2, "record_type", "ALERT")

            # Reihenfolge vereinheitlichen
            df_alerts_ctx = df_alerts_ctx.rename(columns={})
            df_ts_norm = df_ts[CSV_COLUMNS_EVENTS]
            df_alerts_norm = df_alerts_ctx[CSV_COLUMNS_EVENTS]

            events = pd.concat([df_ts_norm, df_alerts_norm], ignore_index=True)
            # sortiert nach Zeit, Alerts direkt neben Messungen
            events = events.sort_values(["ts","record_type"]).reset_index(drop=True)
            return events

        events_df = build_events_df()
        events_csv = events_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Export Events kombiniert (CSV)",
            data=events_csv,
            file_name=f"events_{st.session_state.eq_num}.csv",
            mime="text/csv",
            use_container_width=True,
        )

        # Optional: Excel (Events)
        try:
            import openpyxl  # nur falls vorhanden
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                events_df.to_excel(writer, index=False, sheet_name="events")
            st.download_button(
                "⬇️ Export Events (Excel)",
                data=buffer.getvalue(),
                file_name=f"events_{st.session_state.eq_num}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        except Exception:
            pass  # Excel-Button nur anzeigen, wenn openpyxl installiert ist

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
        order = {"OK": 0, "WARN": 1, "ALERT": 2}
        lvl = "OK"
        for lv in th_levels:
            if order[lv] > order[lvl]:
                lvl = lv
        if score is not None:
            if score >= ml_alert_thresh: lvl = "ALERT"
            elif score >= ml_alert_thresh*0.7 and order["WARN"]>order[lvl]: lvl = "WARN"

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
            badge = st.error if a["level"] == "ALERT" else st.warning
            badge(f"[{a['ts']}] {a['source']} | {a['metric']}={a['value']} → {a['level']} ({a['note']})")
    else:
        st.info("Keine Alarme.")

# ---------------- SONSTIGES ----------------
with tab_misc:
    st.subheader("Beispiel: Entscheidungsbaum (IsolationForest)")
    fig, ax = plt.subplots(figsize=(8,4)); ax.axis("off")

    def add_box(ax, xy, text):
        box = FancyBboxPatch(xy, 0.38, 0.18, boxstyle="round,pad=0.02", fc="#E6F2FF", ec="#3973AC", lw=1.5)
        ax.add_patch(box)
        tx = ax.text(xy[0]+0.19, xy[1]+0.09, text, ha="center", va="center", fontsize=9, weight="bold")
        tx.set_path_effects([patheffects.withStroke(linewidth=3, foreground="white")])

    add_box(ax, (0.05, 0.65), "Temperatur < 50 °C?")
    add_box(ax, (0.05, 0.30), "Ja → Spannung > 600 V?")
    add_box(ax, (0.55, 0.30), "Nein → normaler Bereich")
    add_box(ax, (0.05, 0.02), "Ja → normal")
    add_box(ax, (0.40, 0.02), "Nein → Ausreißer")

    arrow = ArrowStyle("-|>", head_length=1.0, head_width=0.6)
    ax.annotate("", xy=(0.24,0.39), xytext=(0.24,0.65), arrowprops=dict(arrowstyle=arrow, lw=1.5, color="#444"))
    ax.annotate("", xy=(0.55,0.39), xytext=(0.24,0.65), arrowprops=dict(arrowstyle=arrow, lw=1.5, color="#444"))
    ax.annotate("", xy=(0.24,0.11), xytext=(0.24,0.30), arrowprops=dict(arrowstyle=arrow, lw=1.5, color="#444"))
    ax.annotate("", xy=(0.40,0.11), xytext=(0.24,0.30), arrowprops=dict(arrowstyle=arrow, lw=1.5, color="#444"))

    st.pyplot(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("CSV-Strukturen (Vorschau)")
    events_preview = pd.DataFrame(columns=CSV_COLUMNS_EVENTS) if st.session_state.df.empty and not st.session_state.alarms else None
    if events_preview is not None:
        base_ts = now_ts()
        events_preview = pd.DataFrame(
            [
                {"ts":base_ts,"equipment_id":st.session_state.eq_num,"record_type":"MEAS","temperature_c":45.2,"vibration_rms":0.36,"current_a":121.0,"voltage_v":541.2,"fan_rpm":3180,"source":"","level":"","metric":"","value":np.nan,"note":""},
                {"ts":base_ts,"equipment_id":st.session_state.eq_num,"record_type":"ALERT","temperature_c":45.2,"vibration_rms":0.36,"current_a":121.0,"voltage_v":602.3,"fan_rpm":3180,"source":"THRESHOLD","level":"WARN","metric":"voltage_v","value":602.3,"note":"über Grenzwert warn"},
            ]
        )[CSV_COLUMNS_EVENTS]
        st.dataframe(events_preview, use_container_width=True, hide_index=True)
    else:
        st.caption("Die Events-Vorschau entspricht dem kombinierten Export (siehe Overview).")
