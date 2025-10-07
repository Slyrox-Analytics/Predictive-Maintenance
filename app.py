import time
import io
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

st.set_page_config(page_title="Predictive Maintenance – Rectifier", page_icon="🛠️", layout="wide")

# ---------------- EQUIPMENT-STAMMDATEN ----------------
EQUIPMENTS = {
    "10109812-01": {"name": "Gleichrichter XD1", "location": "Schaltschrank 1 – Galvanik Halle (Schüttgutbereich)"},
    "10109812-02": {"name": "Gleichrichter XD2", "location": "Schaltschrank 2 – Galvanik Halle (Schüttgutbereich)"},
}

# ---------------- CSV-SPALTEN (feste Reihenfolge) ----------------
CSV_COLUMNS_TS = ["ts","equipment_id","temperature_c","vibration_rms","current_a","voltage_v","fan_rpm"]
CSV_COLUMNS_ALERTS_INTERNAL = ["ts","equipment_id","source","level","metric","value","note"]   # interner Speicher
CSV_COLUMNS_ALERTS_DE = ["ts","equipment_id","quelle","stufe","merkmal","wert","notiz"]       # Export (deutsch)
CSV_COLUMNS_EVENTS = [
    "ts","equipment_id","typ",
    "temperature_c","vibration_rms","current_a","voltage_v","fan_rpm",
    "quelle","stufe","merkmal","wert","notiz"
]

# ---------------- STATE ----------------
if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame(columns=CSV_COLUMNS_TS)
if "alarms" not in st.session_state:
    # list of dicts mit Keys: ts, equipment_id, source, level, metric, value, note
    st.session_state.alarms = []
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

    # >>> Deine Erklärung 1:1 <<<
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

# ---------------- HILFSFUNKTIONEN ----------------
def now_ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

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

def push_alarm(ts, stufe, quelle, merkmal, wert, notiz=""):
    """Deutsch beschriftete Alarm-Erfassung (intern gespeichert, Export wird gemappt)."""
    st.session_state.alarms.append({
        "ts": ts,
        "equipment_id": st.session_state.eq_num,
        "source": quelle,        # intern 'source' -> Export 'quelle'
        "level": stufe,          # intern 'level'  -> Export 'stufe'
        "metric": merkmal,       # intern 'metric' -> Export 'merkmal'
        "value": round(float(wert), 3) if wert is not None else None,  # -> 'wert'
        "note": notiz            # -> 'notiz'
    })

def check_thresholds(vals, ts):
    for k, v in THRESHOLDS.items():
        val = float(vals[k])
        if k == "fan_rpm":
            if val < v["alert"]:
                push_alarm(ts, "ALERT", "Grenzwert", k, val, "unter Grenzwert (ALERT)")
            elif val < v["warn"]:
                push_alarm(ts, "WARN", "Grenzwert", k, val, "unter Grenzwert (WARN)")
        else:
            if val > v["alert"]:
                push_alarm(ts, "ALERT", "Grenzwert", k, val, "über Grenzwert (ALERT)")
            elif val > v["warn"]:
                push_alarm(ts, "WARN", "Grenzwert", k, val, "über Grenzwert (WARN)")

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

def build_events_df() -> pd.DataFrame:
    """Kombinierte Ereignis-Liste (Messpunkt + Alarm) in EINER Tabelle – deutsch."""
    df_ts = st.session_state.df.reindex(columns=CSV_COLUMNS_TS).copy()
    df_ts["typ"] = "Messpunkt"
    df_ts["quelle"] = ""
    df_ts["stufe"] = ""
    df_ts["merkmal"] = ""
    df_ts["wert"] = np.nan
    df_ts["notiz"] = ""

    df_alerts = pd.DataFrame(st.session_state.alarms)
    if df_alerts.empty:
        df_alerts = pd.DataFrame(columns=CSV_COLUMNS_ALERTS_INTERNAL)
    df_alerts = df_alerts.rename(columns={
        "source":"quelle",
        "level":"stufe",
        "metric":"merkmal",
        "value":"wert",
        "note":"notiz"
    })

    df_alerts_ctx = df_alerts.merge(
        df_ts[["ts","equipment_id","temperature_c","vibration_rms","current_a","voltage_v","fan_rpm"]],
        on=["ts","equipment_id"],
        how="left"
    )
    df_alerts_ctx.insert(2, "typ", "Alarm")

    df_ts_norm = df_ts[CSV_COLUMNS_EVENTS]
    df_alerts_norm = df_alerts_ctx[CSV_COLUMNS_EVENTS]
    events = pd.concat([df_ts_norm, df_alerts_norm], ignore_index=True)
    events = events.sort_values(["ts","typ"]).reset_index(drop=True)
    return events

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
            push_alarm(ts, "ALERT", "KI", "anomaly_score", score, f"Score ≥ {ml_alert_thresh}")
        elif score >= ml_alert_thresh * 0.7:
            push_alarm(ts, "WARN", "KI", "anomaly_score", score, f"Score ≥ {ml_alert_thresh*0.7:.2f}")

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

        # Alerts (CSV, deutsch benannt)
        export_alerts = pd.DataFrame(st.session_state.alarms)
        if export_alerts.empty:
            export_alerts = pd.DataFrame(columns=CSV_COLUMNS_ALERTS_INTERNAL)
        export_alerts_de = export_alerts.rename(columns={
            "source":"quelle","level":"stufe","metric":"merkmal","value":"wert","note":"notiz"
        })[CSV_COLUMNS_ALERTS_DE]
        alerts_csv = export_alerts_de.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Export Alerts (CSV)",
            data=alerts_csv,
            file_name=f"alerts_{st.session_state.eq_num}.csv",
            mime="text/csv",
            use_container_width=True,
        )

        # Ereignisse kombiniert (CSV)
        events_df = build_events_df()
        events_csv = events_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Export Ereignisse kombiniert (CSV)",
            data=events_csv,
            file_name=f"events_{st.session_state.eq_num}.csv",
            mime="text/csv",
            use_container_width=True,
        )

        # Optional: Excel-Export der Events (falls openpyxl vorhanden)
        try:
            import openpyxl  # noqa: F401
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                events_df.to_excel(writer, index=False, sheet_name="events")
            st.download_button(
                "⬇️ Export Ereignisse (Excel)",
                data=buffer.getvalue(),
                file_name=f"events_{st.session_state.eq_num}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        except Exception:
            pass

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
        df_view = pd.DataFrame(st.session_state.alarms).rename(columns={
            "source":"quelle","level":"stufe","metric":"merkmal","value":"wert","note":"notiz"
        })
        for _, a in df_view.tail(200).iloc[::-1].iterrows():
            badge = st.error if a["stufe"] == "ALERT" else st.warning
            badge(f"[{a['ts']}] {a['quelle']} | {a['merkmal']}={a['wert']} → {a['stufe']} ({a['notiz']})")
    else:
        st.info("Keine Alarme.")

# ---------------- SONSTIGES ----------------
with tab_misc:
    st.subheader("IsolationForest Prinzip – Normal vs. Anomalie")

    # Scatter: Temperatur vs. Strom, viele Normale + 1 Anomalie
    rng = np.random.default_rng(42)
    n = 150
    temp_norm = 50 + rng.normal(0, 2.0, n)   # um 50°C
    curr_norm = 120 + rng.normal(0, 3.0, n)  # um 120A
    anom_temp = 65.0
    anom_curr = 120.0

    fig_sc, ax_sc = plt.subplots(figsize=(8,5))
    ax_sc.scatter(temp_norm, curr_norm, s=25, label="Normal")
    ax_sc.scatter([anom_temp],[anom_curr], s=60, marker="x", linewidths=2.5, label="Anomalie")
    ax_sc.annotate("Anomalie: schnell isolierbar", xy=(anom_temp, anom_curr),
                   xytext=(anom_temp-7, anom_curr+8),
                   arrowprops=dict(arrowstyle="->", lw=1.5))
    ax_sc.set_xlabel("Temperatur (°C)")
    ax_sc.set_ylabel("Strom (A)")
    ax_sc.set_title("IsolationForest Prinzip – Normal vs. Anomalie")
    ax_sc.legend()
    st.pyplot(fig_sc, use_container_width=True)

    st.markdown("---")
    st.subheader("Zeitreihe mit Ausreißer")

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

    st.markdown("---")
    st.subheader("CSV-Strukturen (Vorschau)")
    if len(st.session_state.df)==0 and len(st.session_state.alarms)==0:
        base_ts = now_ts()
        preview = pd.DataFrame(
            [
                {"ts":base_ts,"equipment_id":st.session_state.eq_num,"typ":"Messpunkt","temperature_c":45.2,"vibration_rms":0.36,"current_a":121.0,"voltage_v":541.2,"fan_rpm":3180,"quelle":"","stufe":"","merkmal":"","wert":np.nan,"notiz":""},
                {"ts":base_ts,"equipment_id":st.session_state.eq_num,"typ":"Alarm","temperature_c":45.2,"vibration_rms":0.36,"current_a":121.0,"voltage_v":602.3,"fan_rpm":3180,"quelle":"Grenzwert","stufe":"WARN","merkmal":"voltage_v","wert":602.3,"notiz":"über Grenzwert (WARN)"},
            ]
        )[CSV_COLUMNS_EVENTS]
        st.dataframe(preview, use_container_width=True, hide_index=True)
    else:
        events_prev = build_events_df().tail(10)
        st.dataframe(events_prev, use_container_width=True, hide_index=True)
