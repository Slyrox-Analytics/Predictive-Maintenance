import time
import io
import sqlite3
import threading
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime
from sklearn.ensemble import IsolationForest
from streamlit import rerun  # für Button-Re-Runs (ok)

st.set_page_config(page_title="Predictive Maintenance – Rectifier", page_icon="🛠️", layout="wide")

# ---------- SAP/Fiori-ähnliches Styling ----------
st.markdown("""
<style>
:root {
  --sap-primary: #0a6ed1;   /* SAP Fiori Blau */
  --sap-warn:    #f0ab00;   /* SAP Warn-Gelb */
  --sap-alert:   #bb0000;   /* Rot */
  --sap-ok:      #107e3e;   /* Grün */
  --sap-text:    #32363a;
}
html, body, [class*="css"]  { color: var(--sap-text); }
h1, h2, h3, h4 { color: var(--sap-primary) !important; }
section.main > div { padding-top: 0.5rem; }

.stTabs [role="tablist"] { gap: .25rem; }
.stTabs [role="tab"] { border: 1px solid #e5e7eb; border-bottom: none; background: #f8fafc; }
.stTabs [aria-selected="true"] { background: white; border-bottom: 2px solid var(--sap-primary); color: var(--sap-primary); }

div[data-testid="stMetricValue"] { color: var(--sap-primary); font-weight: 600; }
div[data-testid="stMetric"] { border: 1px solid #eef2f7; border-radius: 10px; padding: .5rem .75rem; background: #fbfdff; }

button[kind="secondary"] { border-color: var(--sap-primary) !important; color: var(--sap-primary) !important; }
.stDownloadButton button { width: 100%; }

blockquote, .legend-box {
  border-left: 4px solid var(--sap-primary);
  padding: .5rem .75rem;
  background: #f4f9ff;
  border-radius: 6px;
  margin: .25rem 0 .75rem 0;
  font-size: 0.95rem;
}
.legend-inline { font-size: 0.92rem; margin-top: .25rem; }
.legend-inline strong { color: var(--sap-primary); }
</style>
""", unsafe_allow_html=True)

# ---------------- EQUIPMENT-STAMMDATEN ----------------
EQUIPMENTS = {
    "10109812-01": {"name": "Gleichrichter XD1", "location": "Schaltschrank 1 – Galvanik Halle (Schüttgutbereich)"},
    "10109812-02": {"name": "Gleichrichter XD2", "location": "Schaltschrank 2 – Galvanik Halle (Schüttgutbereich)"},
}

# ---------------- SOLLWERTE (Nominals) ----------------
NOMINALS = {
    "10109812-01": {  # XD1
        "temperature_c": 45.0,
        "vibration_rms": 0.35,
        "current_a": 120.0,
        "voltage_v": 540.0,
        "fan_rpm": 3200.0,
    },
    "10109812-02": {  # XD2
        "temperature_c": 45.0,
        "vibration_rms": 0.35,
        "current_a": 120.0,
        "voltage_v": 540.0,
        "fan_rpm": 3200.0,
    },
}
def defaults_from_nominals(eq_id: str):
    n = NOMINALS[eq_id]
    return {
        "temperature_c": {"warn": n["temperature_c"] + 15.0,  "alert": n["temperature_c"] + 25.0},
        "vibration_rms": {"warn": n["vibration_rms"] + 0.25,  "alert": n["vibration_rms"] + 0.45},
        "current_a":     {"warn": n["current_a"] * 1.25,      "alert": n["current_a"] * 1.50},
        "voltage_v":     {"warn": n["voltage_v"] * 1.07,      "alert": n["voltage_v"] * 1.15},
        "fan_rpm":       {"warn": n["fan_rpm"] - 600.0,       "alert": n["fan_rpm"] - 1200.0},  # Untergrenze
    }

METRICS = ["temperature_c","vibration_rms","current_a","voltage_v","fan_rpm"]

# ---------------- HINTERGRUND: DB + WORKER ----------------
def init_db(conn: sqlite3.Connection):
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS measurements (
        ts TEXT,
        equipment_id TEXT,
        temperature_c REAL,
        vibration_rms REAL,
        current_a REAL,
        voltage_v REAL,
        fan_rpm REAL
    )""")
    c.execute("""
    CREATE TABLE IF NOT EXISTS alarms (
        ts TEXT,
        equipment_id TEXT,
        level TEXT,
        message TEXT
    )""")
    c.execute("""
    CREATE TABLE IF NOT EXISTS control (
        id INTEGER PRIMARY KEY CHECK (id=1),
        running INTEGER DEFAULT 0,
        eq_id TEXT,
        fault_cooling INTEGER DEFAULT 0,
        fault_fan INTEGER DEFAULT 0,
        fault_voltage INTEGER DEFAULT 0,
        ml_window INTEGER,
        ml_cont REAL,
        ml_alert REAL
    )""")
    # einmalige Initialzeile
    c.execute("SELECT COUNT(*) FROM control WHERE id=1")
    if c.fetchone()[0] == 0:
        c.execute(
            "INSERT INTO control (id, running, eq_id, fault_cooling, fault_fan, fault_voltage, ml_window, ml_cont, ml_alert) "
            "VALUES (1, 0, ?, 0, 0, 0, 600, 0.02, 0.80)",
            ("10109812-01",)
        )
    conn.commit()

def control_get(conn):
    c = conn.cursor()
    c.execute("SELECT running, eq_id, fault_cooling, fault_fan, fault_voltage, ml_window, ml_cont, ml_alert FROM control WHERE id=1")
    row = c.fetchone()
    if not row:
        return {"running":0,"eq_id":"10109812-01","fault_cooling":0,"fault_fan":0,"fault_voltage":0,"ml_window":600,"ml_cont":0.02,"ml_alert":0.80}
    return {
        "running": int(row[0]),
        "eq_id": row[1],
        "fault_cooling": int(row[2]),
        "fault_fan": int(row[3]),
        "fault_voltage": int(row[4]),
        "ml_window": int(row[5] or 600),
        "ml_cont": float(row[6] or 0.02),
        "ml_alert": float(row[7] or 0.80),
    }

def control_update(conn, **kwargs):
    if not kwargs: return
    sets = ", ".join([f"{k}=?" for k in kwargs.keys()])
    vals = list(kwargs.values()) + [1]
    conn.execute(f"UPDATE control SET {sets} WHERE id=?", vals)
    conn.commit()

def db_save_measurement(conn, row):
    conn.execute("INSERT INTO measurements VALUES (?,?,?,?,?,?,?)", (
        row["ts"], row["equipment_id"], row["temperature_c"], row["vibration_rms"],
        row["current_a"], row["voltage_v"], row["fan_rpm"]
    ))
    conn.commit()

def db_save_alarm(conn, ts, eq, level, msg):
    conn.execute("INSERT INTO alarms VALUES (?,?,?,?)", (ts, eq, level, msg))
    conn.commit()

def db_load_measurements(conn, limit=2000, eq_id=None):
    q = "SELECT ts, equipment_id, temperature_c, vibration_rms, current_a, voltage_v, fan_rpm FROM measurements"
    params = []
    if eq_id:
        q += " WHERE equipment_id=?"
        params.append(eq_id)
    q += " ORDER BY ts DESC LIMIT ?"
    params.append(limit)
    df = pd.read_sql(q, conn, params=params)
    if df.empty:
        return df
    # chronologisch sortieren und ts -> datetime (WICHTIG für Live-Charts)
    df = df.iloc[::-1].reset_index(drop=True)
    df["ts"] = pd.to_datetime(df["ts"])
    return df

def db_load_alarms(conn, limit=2000, eq_id=None):
    q = "SELECT ts, equipment_id, level, message FROM alarms"
    params = []
    if eq_id:
        q += " WHERE equipment_id=?"
        params.append(eq_id)
    q += " ORDER BY ts DESC LIMIT ?"
    params.append(limit)
    return pd.read_sql(q, conn, params=params)

def sim_generate_sample(eq_id: str, t: int, faults: dict):
    base = NOMINALS[eq_id].copy()
    if faults.get("cooling"): base["temperature_c"] += 0.01 * t
    if faults.get("fan"):     base["fan_rpm"] -= 0.5 * t
    if faults.get("voltage"): base["voltage_v"] += 20 * np.sin(t / 3.0)
    base["temperature_c"] += float(np.random.uniform(-0.2, 0.2))
    base["vibration_rms"] += float(np.random.uniform(-0.02, 0.02))
    base["current_a"]     += float(np.random.uniform(-2, 2))
    base["voltage_v"]     += float(np.random.uniform(-1.5, 1.5))
    base["fan_rpm"]       += float(np.random.uniform(-30, 30))
    return base

def sim_check_thresholds(conn, eq_id, vals, ts):
    TH = defaults_from_nominals(eq_id)
    for k, v in TH.items():
        val = float(vals[k])
        if k == "fan_rpm":
            if val < v["alert"]:
                db_save_alarm(conn, ts, eq_id, "ALERT", f"{k} zu niedrig: {val:.1f} RPM")
            elif val < v["warn"]:
                db_save_alarm(conn, ts, eq_id, "WARN", f"{k} niedrig: {val:.1f} RPM")
        else:
            if val > v["alert"]:
                db_save_alarm(conn, ts, eq_id, "ALERT", f"{k} zu hoch: {val:.1f}")
            elif val > v["warn"]:
                db_save_alarm(conn, ts, eq_id, "WARN", f"{k} hoch: {val:.1f}")

def sim_ml_anomaly(conn, eq_id, window, contamination):
    df = db_load_measurements(conn, limit=window, eq_id=eq_id)
    if len(df) < window: return None
    X = df[METRICS].astype(float).to_numpy()
    mu = X.mean(axis=0); sigma = X.std(axis=0); sigma[sigma == 0] = 1e-6
    Z = (X - mu) / sigma
    Z_train, z_last = Z[:-1], Z[-1].reshape(1, -1)
    model = IsolationForest(contamination=float(contamination), random_state=42)
    model.fit(Z_train)
    raw_last  = -model.decision_function(z_last)[0]
    raw_train = -model.decision_function(Z_train)
    lo, hi = float(raw_train.min()), float(raw_train.max()) + 1e-9
    score = (raw_last - lo) / (hi - lo)
    return float(score)

def background_worker(conn: sqlite3.Connection, stop_evt: threading.Event):
    t = 0
    while not stop_evt.is_set():
        ctrl = control_get(conn)
        if ctrl["running"] == 1:
            eq_id = ctrl["eq_id"] or "10109812-01"
            faults = {"cooling": bool(ctrl["fault_cooling"]), "fan": bool(ctrl["fault_fan"]), "voltage": bool(ctrl["fault_voltage"])}
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            vals = sim_generate_sample(eq_id, t, faults)
            row = {"ts": ts, "equipment_id": eq_id, **vals}
            db_save_measurement(conn, row)
            sim_check_thresholds(conn, eq_id, vals, ts)
            score = sim_ml_anomaly(conn, eq_id, window=ctrl["ml_window"], contamination=ctrl["ml_cont"])
            if score is not None:
                if score >= ctrl["ml_alert"]:
                    db_save_alarm(conn, ts, eq_id, "ALERT", f"ML anomaly score={score:.2f}")
                elif score >= (ctrl["ml_alert"] * 0.7):
                    db_save_alarm(conn, ts, eq_id, "WARN", f"ML anomaly score={score:.2f}")
            t += 1
        time.sleep(1)

@st.cache_resource
def get_db_and_worker():
    conn = sqlite3.connect("data.db", check_same_thread=False)
    init_db(conn)
    stop_evt = threading.Event()
    thread = threading.Thread(target=background_worker, args=(conn, stop_evt), daemon=True)
    thread.start()
    return conn, stop_evt

# starte DB + Worker genau einmal pro Server-Prozess
DB_CONN, _STOP = get_db_and_worker()

# ---------------- (alter) STATE für UI-Kompatibilität ----------------
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

THRESHOLDS = defaults_from_nominals(st.session_state.eq_num)

# ---------------- TABS ----------------
tab_overview, tab_live, tab_alerts, tab_settings, tab_misc = st.tabs(
    ["Overview", "Live Charts", "Alerts", "Settings", "Sonstiges"]
)

# ---------------- SETTINGS ----------------
with tab_settings:
    st.subheader("Simulation Control")
    cstart, cdel = st.columns([2,1])
    with cstart:
        ctrl_now = control_get(DB_CONN)
        running_now = bool(ctrl_now["running"])
        if not running_now:
            if st.button("▶️ Start Simulation", use_container_width=True):
                control_update(DB_CONN, running=1, eq_id=st.session_state.eq_num)
                st.success("Hintergrund-Simulation gestartet.")
                rerun()
        else:
            if st.button("⏹ Stop Simulation", use_container_width=True):
                control_update(DB_CONN, running=0)
                st.warning("Hintergrund-Simulation gestoppt.")
                rerun()
    with cdel:
        if st.button("🗑️ Daten löschen", help="Löscht NUR Simulationsdaten & Alarmfeed. Einstellungen bleiben erhalten."):
            DB_CONN.execute("DELETE FROM measurements")
            DB_CONN.execute("DELETE FROM alarms")
            DB_CONN.commit()
            st.session_state.df = st.session_state.df.iloc[0:0]
            st.session_state.alarms = []
            st.success("Daten & Alarme gelöscht. Einstellungen unverändert.")

    st.markdown("---")

    st.subheader("Fault Injection (während des Laufs umschaltbar)")
    c1, c2, c3 = st.columns(3)
    with c1: st.session_state.faults["cooling"] = st.checkbox("Cooling Degradation — Temperatur steigt", value=bool(ctrl_now["fault_cooling"]))
    with c2: st.session_state.faults["fan"]     = st.checkbox("Fan Wear — Lüfter RPM sinkt", value=bool(ctrl_now["fault_fan"]))
    with c3: st.session_state.faults["voltage"] = st.checkbox("Voltage Spikes — sporadische Spannungsspitzen", value=bool(ctrl_now["fault_voltage"]))
    control_update(DB_CONN,
                   fault_cooling=1 if st.session_state.faults["cooling"] else 0,
                   fault_fan=1 if st.session_state.faults["fan"] else 0,
                   fault_voltage=1 if st.session_state.faults["voltage"] else 0,
                   eq_id=st.session_state.eq_num)

    st.markdown("---")
    st.subheader("Schwellwerte")
    r1, r2 = st.columns([3,3])
    with r1:
        t_warn  = st.number_input("Temperatur WARN (°C)",  value=THRESHOLDS["temperature_c"]["warn"], step=1.0, key="t_warn")
        t_alert = st.number_input("Temperatur ALERT (°C)", value=THRESHOLDS["temperature_c"]["alert"], step=1.0, key="t_alert")
        vib_warn  = st.number_input("Vibration WARN (RMS)",  value=THRESHOLDS["vibration_rms"]["warn"], step=0.01, format="%.2f", key="vib_warn")
        vib_alert = st.number_input("Vibration ALERT (RMS)", value=THRESHOLDS["vibration_rms"]["alert"], step=0.01, format="%.2f", key="vib_alert")
    with r2:
        i_warn  = st.number_input("Strom WARN (A)",        value=THRESHOLDS["current_a"]["warn"], step=1.0, key="i_warn")
        i_alert = st.number_input("Strom ALERT (A)",       value=THRESHOLDS["current_a"]["alert"], step=1.0, key="i_alert")
        u_warn  = st.number_input("Spannung WARN (V)",     value=THRESHOLDS["voltage_v"]["warn"], step=1.0, key="u_warn")
        u_alert = st.number_input("Spannung ALERT (V)",    value=THRESHOLDS["voltage_v"]["alert"], step=1.0, key="u_alert")
        fan_warn  = st.number_input("Lüfter WARN (RPM, Untergrenze)",  value=THRESHOLDS["fan_rpm"]["warn"],   step=50.0, key="fan_warn")
        fan_alert = st.number_input("Lüfter ALERT (RPM, Untergrenze)", value=THRESHOLDS["fan_rpm"]["alert"],  step=50.0, key="fan_alert")

    st.markdown(
        f"""
<div class="legend-box">
<b>SOLL &amp; Grenzwerte (für {EQUIPMENTS[st.session_state.eq_num]['name']}):</b><br/>
- Temperatur: <b>SOLL ~{NOMINALS[st.session_state.eq_num]['temperature_c']:.1f} °C</b> → WARN ab <b>{THRESHOLDS['temperature_c']['warn']:.1f} °C</b>, ALERT ab <b>{THRESHOLDS['temperature_c']['alert']:.1f} °C</b>.<br/>
- Vibration: <b>SOLL ~{NOMINALS[st.session_state.eq_num]['vibration_rms']:.2f} RMS</b> → WARN ab <b>{THRESHOLDS['vibration_rms']['warn']:.2f}</b>, ALERT ab <b>{THRESHOLDS['vibration_rms']['alert']:.2f}</b>.<br/>
- Strom: <b>SOLL ~{NOMINALS[st.session_state.eq_num]['current_a']:.0f} A</b> → WARN ab <b>{THRESHOLDS['current_a']['warn']:.0f} A</b>, ALERT ab <b>{THRESHOLDS['current_a']['alert']:.0f} A</b>.<br/>
- Spannung: <b>SOLL ~{NOMINALS[st.session_state.eq_num]['voltage_v']:.0f} V</b> → WARN ab <b>{THRESHOLDS['voltage_v']['warn']:.0f} V</b>, ALERT ab <b>{THRESHOLDS['voltage_v']['alert']:.0f} V</b>.<br/>
- Lüfter (Untergrenze): <b>SOLL ~{NOMINALS[st.session_state.eq_num]['fan_rpm']:.0f} RPM</b> → WARN <b>unter {st.session_state.get('fan_warn', THRESHOLDS['fan_rpm']['warn']):.0f}</b>, ALERT <b>unter {st.session_state.get('fan_alert', THRESHOLDS['fan_rpm']['alert']):.0f}</b>.
</div>
""",
        unsafe_allow_html=True
    )

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
    window = c1.slider("Fenstergröße (Punkte)", 200, 2000, ctrl_now["ml_window"], 50, key="ml_window")
    contamination = c2.slider("Kontamination (erwartete Ausreißer)", 0.001, 0.10, float(ctrl_now["ml_cont"]), 0.001, key="ml_cont")
    ml_alert_thresh = c3.slider("ML-Alert-Schwelle (0–1)", 0.10, 0.90, float(ctrl_now["ml_alert"]), 0.05, key="ml_thresh")
    control_update(DB_CONN, ml_window=int(window), ml_cont=float(contamination), ml_alert=float(ml_alert_thresh))

    ctrl_now2 = control_get(DB_CONN)
    if ctrl_now2["running"] == 1:
        status_placeholder.success("RUNNING – Hintergrund-Simulation aktiv")
    else:
        status_placeholder.warning("Simulation gestoppt (Hintergrund-Worker wartet)")

# ---------------- OVERVIEW ----------------
with tab_overview:
    st.subheader("Gesamtzustand")
    df_live = db_load_measurements(DB_CONN, eq_id=st.session_state.eq_num, limit=2000)
    st.session_state.df = df_live.copy()
    df_alarms = db_load_alarms(DB_CONN, eq_id=st.session_state.eq_num, limit=2000)
    st.session_state.alarms = df_alarms.to_dict("records")

    exp_l, exp_r = st.columns([3,2])
    with exp_r:
        def build_analysis_df_from_db():
            if df_alarms.empty:
                return pd.DataFrame(columns=[
                    "ts","equipment_id","level","message",
                    "temperature_c","vibration_rms","current_a","voltage_v","fan_rpm"
                ])
            merged = df_alarms.merge(
                df_live[["ts","equipment_id","temperature_c","vibration_rms","current_a","voltage_v","fan_rpm"]],
                on=["ts","equipment_id"], how="left"
            )
            cols = ["ts","equipment_id","level","message","temperature_c","vibration_rms","current_a","voltage_v","fan_rpm"]
            return merged.reindex(columns=cols)

        analysis_df = build_analysis_df_from_db()
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

    if len(df_live):
        latest = df_live.iloc[-1]
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Temperatur (°C)", f"{latest['temperature_c']:.1f}")
        k2.metric("Vibration (RMS)", f"{latest['vibration_rms']:.2f}")
        k3.metric("Strom (A)", f"{latest['current_a']:.1f}")
        k4.metric("Spannung (V)", f"{latest['voltage_v']:.1f}")
        k5.metric("Lüfter (RPM)", f"{latest['fan_rpm']:.0f}")

        def quick_ml_score(df, window, contamination):
            if len(df) < window: return None
            X = df[METRICS].astype(float).to_numpy()
            mu = X.mean(axis=0); sigma = X.std(axis=0); sigma[sigma == 0] = 1e-6
            Z = (X - mu) / sigma
            Z_train, z_last = Z[:-1], Z[-1].reshape(1, -1)
            model = IsolationForest(contamination=float(contamination), random_state=42)
            model.fit(Z_train)
            raw_last  = -model.decision_function(z_last)[0]
            raw_train = -model.decision_function(Z_train)
            lo, hi = float(raw_train.min()), float(raw_train.max()) + 1e-9
            return float((raw_last - lo) / (hi - lo))

        cvals = control_get(DB_CONN)
        score = quick_ml_score(df_live, cvals["ml_window"], cvals["ml_cont"])
        lvl = "OK"
        if score is not None:
            if score >= cvals["ml_alert"]: lvl = "ALERT"
            elif score >= (cvals["ml_alert"] * 0.7): lvl = "WARN"

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
    df_live = db_load_measurements(DB_CONN, eq_id=st.session_state.eq_num, limit=2000)
    if len(df_live):
        # ts ist jetzt datetime -> saubere Zeitachse
        st.line_chart(df_live.set_index("ts")[METRICS], use_container_width=True)
    else:
        st.info("Noch keine Daten.")

# ---------------- ALERTS ----------------
with tab_alerts:
    st.subheader("Alarm-Feed (neueste zuerst)")
    df_alarms = db_load_alarms(DB_CONN, eq_id=st.session_state.eq_num, limit=500)
    if not df_alarms.empty:
        for _, a in df_alarms.iloc[::-1].iterrows():
            (st.error if a["level"] == "ALERT" else st.warning)(f"[{a['ts']}] {a['message']}")
    else:
        st.info("Keine Alarme.")

# ---------------- SONSTIGES (Vorzeigen) ----------------
with tab_misc:
    st.subheader("IsolationForest – Normal vs. Anomalie (Illustration)")
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyBboxPatch, ArrowStyle

        rng = np.random.default_rng(42)
        npts = 150
        temp_norm = 50 + rng.normal(0, 2.0, npts)
        curr_norm = 120 + rng.normal(0, 3.0, npts)
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
        st.caption("Goldene Punkte = normales Verhalten. Rotes ✗ = Ausreißer (passt nicht ins gelernte Muster).")

        st.markdown("---")
        st.subheader("Entscheidungsbaum – vereinfachte Logik (Illustration)")
        fig, ax = plt.subplots(figsize=(9,4))
        ax.axis("off")
        def box(xy, text):
            b = FancyBboxPatch(xy, 0.36, 0.18, boxstyle="round,pad=0.02", fc="#E6F2FF", ec="#3973AC", lw=1.5)
            ax.add_patch(b)
            ax.text(xy[0]+0.18, xy[1]+0.09, text, ha="center", va="center", fontsize=9, weight="bold")
        box((0.05, 0.62), "Spannung > 600 V?")
        box((0.48, 0.62), "Ja → Ausreißer")
        box((0.05, 0.32), "Nein → Temp < 50 °C?")
        box((0.05, 0.02), "Ja → Normal")
        box((0.48, 0.32), "Nein → Vibration > 0.8?")
        box((0.48, 0.02), "Ja → Ausreißer")
        box((0.78, 0.02), "Nein → Normal")
        arr = ArrowStyle("-|>", head_length=1.0, head_width=0.6)
        ax.annotate("", xy=(0.41,0.41), xytext=(0.23,0.62), arrowprops=dict(arrowstyle=arr, lw=1.4))
        ax.annotate("", xy=(0.48,0.70), xytext=(0.23,0.70), arrowprops=dict(arrowstyle=arr, lw=1.4))
        ax.annotate("", xy=(0.23,0.11), xytext=(0.23,0.32), arrowprops=dict(arrowstyle=arr, lw=1.4))
        ax.annotate("", xy=(0.66,0.41), xytext=(0.41,0.41), arrowprops=dict(arrowstyle=arr, lw=1.4))
        ax.annotate("", xy=(0.66,0.11), xytext=(0.66,0.32), arrowprops=dict(arrowstyle=arr, lw=1.4))
        ax.annotate("", xy=(0.83,0.11), xytext=(0.66,0.11), arrowprops=dict(arrowstyle=arr, lw=1.4))
        st.pyplot(fig, use_container_width=True)
        st.caption("Ablauf: 1) Spannung prüfen (>600 V ⇒ Ausreißer). 2) Sonst Temperatur (<50 °C ⇒ normal). 3) Sonst Vibration prüfen (>0.8 ⇒ Ausreißer, sonst normal).")

    except Exception:
        st.warning("Matplotlib fehlt – bitte `matplotlib` in requirements.txt ergänzen.")

    st.markdown("---")
    st.subheader("Beispiel-Export (eine Liste – Vorschau)")
    st.markdown(
        """
**So liest du die Tabelle:**  
- Jede Zeile ist **ein Alarm** mit **den Messwerten zum gleichen Zeitpunkt**.  
- **level** = Stufe (WARN/ALERT), **message** = Grund.  
- **temperature_c, vibration_rms, current_a, voltage_v, fan_rpm** = Messkontext zur Analyse.

**Beispiel unten:**  
- Zeile 1 = **Grenzwert-WARN** wegen **Spannung (voltage_v)** über Warn-Grenze.  
- Zeile 2 = **ML-ALERT** vom IsolationForest (**Anomalie** erkannt).
"""
    )
    df_alarms = db_load_alarms(DB_CONN, eq_id=st.session_state.eq_num, limit=10)
    df_live = db_load_measurements(DB_CONN, eq_id=st.session_state.eq_num, limit=2000)
    if df_alarms.empty or df_live.empty:
        ts1 = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ts2 = (datetime.now() + pd.Timedelta(seconds=3)).strftime("%Y-%m-%d %H:%M:%S")
        demo = pd.DataFrame(
            [
                {"ts": ts1, "equipment_id": st.session_state.eq_num, "level": "WARN",  "message": "voltage_v hoch: 602.3",
                 "temperature_c": 45.2, "vibration_rms": 0.36, "current_a": 121.0, "voltage_v": 602.3, "fan_rpm": 3180},
                {"ts": ts2, "equipment_id": st.session_state.eq_num, "level": "ALERT", "message": "ML anomaly score=0.86",
                 "temperature_c": 45.2, "vibration_rms": 0.36, "current_a": 121.0, "voltage_v": 541.2, "fan_rpm": 3180},
            ]
        )
        st.dataframe(demo, use_container_width=True, hide_index=True)
    else:
        preview = df_alarms.merge(
            df_live[["ts","equipment_id","temperature_c","vibration_rms","current_a","voltage_v","fan_rpm"]],
            on=["ts","equipment_id"], how="left"
        )
        st.dataframe(preview.tail(10), use_container_width=True, hide_index=True)

# --------- AUTO-REFRESH: solange running=1, alle 1 s neu rendern ---------
try:
    _ctrl = control_get(DB_CONN)
    if int(_ctrl.get("running", 0)) == 1:
        time.sleep(1)   # Tickrate
        st.rerun()
except Exception:
    pass
