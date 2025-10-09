import time
import io
import math
import sqlite3
import threading
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest
from streamlit import rerun  # für Button-Re-Runs

st.set_page_config(page_title="Predictive Maintenance – Rectifier", page_icon="🛠️", layout="wide")

# ---------- SAP/Fiori-ähnliches Styling ----------
st.markdown("""
<style>
:root {
  --sap-primary: #0a6ed1;
  --sap-warn:    #f0ab00;
  --sap-alert:   #bb0000;
  --sap-ok:      #107e3e;
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

.small-muted { font-size: 0.9rem; color: #6b7280; }
.badge { display:inline-block; padding: .25rem .5rem; border-radius: .5rem; font-weight:600; }
.badge-ok { background:#ECFDF5; color:#065F46; border:1px solid #A7F3D0; }
.badge-warn { background:#FFFBEB; color:#92400E; border:1px solid #FDE68A; }
.badge-alert { background:#FEF2F2; color:#991B1B; border:1px solid #FCA5A5; }
.progress { height:10px; background:#eef2f7; border-radius:8px; overflow:hidden; }
.progress > div { height:100%; background:#0a6ed1; }
</style>
""", unsafe_allow_html=True)

# ---------------- EQUIPMENT-STAMMDATEN ----------------
EQUIPMENTS = {
    "10109812-01": {"name": "Gleichrichter XD1", "location": "Schaltschrank 1 – Galvanik Halle (Schüttgutbereich)"},
    "10109812-02": {"name": "Gleichrichter XD2", "location": "Schaltschrank 2 – Galvanik Halle (Schüttgutbereich)"},
}

# ---------------- SOLLWERTE (Nominals) ----------------
NOMINALS = {
    "10109812-01": {"temperature_c": 45.0, "vibration_rms": 0.35, "current_a": 120.0, "voltage_v": 540.0, "fan_rpm": 3200.0},
    "10109812-02": {"temperature_c": 45.0, "vibration_rms": 0.35, "current_a": 120.0, "voltage_v": 540.0, "fan_rpm": 3200.0},
}
def defaults_from_nominals(eq_id: str):
    n = NOMINALS[eq_id]
    return {
        "temperature_c": {"warn": n["temperature_c"] + 15.0,  "alert": n["temperature_c"] + 25.0},
        "vibration_rms": {"warn": n["vibration_rms"] + 0.25,  "alert": n["vibration_rms"] + 0.45},
        "current_a":     {"warn": n["current_a"] * 1.25,      "alert": n["current_a"] * 1.50},
        "voltage_v":     {"warn": n["voltage_v"] * 1.07,      "alert": n["voltage_v"] * 1.15},
        "fan_rpm":       {"warn": n["fan_rpm"] - 600.0,       "alert": n["fan_rpm"] - 1200.0},
    }
METRICS = ["temperature_c","vibration_rms","current_a","voltage_v","fan_rpm"]

# ---------------- DB-UTILS: eigene Connections + WAL ----------------
def open_db(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)  # autocommit
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA busy_timeout=4000;")
    return conn

def _execute_write_retry(conn: sqlite3.Connection, sql: str, params: tuple = (), retries: int = 8, base_sleep: float = 0.05):
    for i in range(retries):
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(sql, params)
            conn.execute("COMMIT")
            return
        except sqlite3.OperationalError:
            try:
                conn.execute("ROLLBACK")
            except Exception:
                pass
            time.sleep(base_sleep * (i + 1))
    conn.execute("BEGIN IMMEDIATE")
    conn.execute(sql, params)
    conn.execute("COMMIT")

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
    # neue Spalten bei Bedarf ergänzen
    new_cols = {
        "cool_rate": "REAL DEFAULT 0.00",      # °C pro Sekunde
        "fan_drop": "REAL DEFAULT 0.0",        # RPM pro Sekunde
        "volt_amp": "REAL DEFAULT 0.0",        # Volt
        "volt_freq": "REAL DEFAULT 0.50",      # Hz
        "tick_ms": "INTEGER DEFAULT 1000",     # Worker-Tick in ms
        "boost_temp": "REAL DEFAULT 0.0",      # einmaliger °C-Boost
        "boost_volt": "REAL DEFAULT 0.0"       # einmaliger Volt-Boost
    }
    for col, ddl in new_cols.items():
        try:
            c.execute(f"ALTER TABLE control ADD COLUMN {col} {ddl}")
        except sqlite3.OperationalError:
            pass  # gibt's schon
    # Initialzeile
    c.execute("SELECT COUNT(*) FROM control WHERE id=1")
    if c.fetchone()[0] == 0:
        c.execute(
            "INSERT INTO control (id, running, eq_id, fault_cooling, fault_fan, fault_voltage, ml_window, ml_cont, ml_alert, cool_rate, fan_drop, volt_amp, volt_freq, tick_ms, boost_temp, boost_volt) "
            "VALUES (1, 0, ?, 0, 0, 0, 600, 0.02, 0.80, 0.00, 0.0, 0.0, 0.50, 1000, 0.0, 0.0)",
            ("10109812-01",)
        )

def control_get(conn):
    c = conn.cursor()
    c.execute("""SELECT running, eq_id, fault_cooling, fault_fan, fault_voltage, ml_window, ml_cont, ml_alert,
                        cool_rate, fan_drop, volt_amp, volt_freq, tick_ms, boost_temp, boost_volt
                 FROM control WHERE id=1""")
    row = c.fetchone()
    if not row:
        return {"running":0,"eq_id":"10109812-01","fault_cooling":0,"fault_fan":0,"fault_voltage":0,
                "ml_window":600,"ml_cont":0.02,"ml_alert":0.80,
                "cool_rate":0.00, "fan_drop":0.0, "volt_amp":0.0, "volt_freq":0.50, "tick_ms":1000,
                "boost_temp":0.0, "boost_volt":0.0}
    return {
        "running": int(row[0]), "eq_id": row[1],
        "fault_cooling": int(row[2]), "fault_fan": int(row[3]), "fault_voltage": int(row[4]),
        "ml_window": int(row[5] or 600), "ml_cont": float(row[6] or 0.02), "ml_alert": float(row[7] or 0.80),
        "cool_rate": float(row[8] or 0.0), "fan_drop": float(row[9] or 0.0),
        "volt_amp": float(row[10] or 0.0), "volt_freq": float(row[11] or 0.50),
        "tick_ms": int(row[12] or 1000),
        "boost_temp": float(row[13] or 0.0), "boost_volt": float(row[14] or 0.0),
    }

def control_update(conn, **kwargs):
    if not kwargs: return
    sets = ", ".join([f"{k}=?" for k in kwargs.keys()])
    vals = list(kwargs.values()) + [1]
    _execute_write_retry(conn, f"UPDATE control SET {sets} WHERE id=?", tuple(vals))

def db_save_measurement(conn, row):
    _execute_write_retry(conn,
        "INSERT INTO measurements VALUES (?,?,?,?,?,?,?)",
        (row["ts"], row["equipment_id"], row["temperature_c"], row["vibration_rms"],
         row["current_a"], row["voltage_v"], row["fan_rpm"])
    )

def db_save_alarm(conn, ts, eq, level, msg):
    _execute_write_retry(conn, "INSERT INTO alarms VALUES (?,?,?,?)", (ts, eq, level, msg))

def db_load_measurements(conn, limit=2000, eq_id=None):
    q = "SELECT ts, equipment_id, temperature_c, vibration_rms, current_a, voltage_v, fan_rpm FROM measurements"
    params = []
    if eq_id:
        q += " WHERE equipment_id=?"
        params.append(eq_id)
    q += " ORDER BY ts DESC LIMIT ?"
    params.append(limit)
    df = pd.read_sql(q, conn, params=params)
    if df.empty: return df
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

# -------- Simulation --------
def sim_generate_sample(eq_id: str, t_sec: float, dt: float, ctrl):
    base = NOMINALS[eq_id].copy()

    # Faults (kontinuierlich)
    if ctrl["fault_cooling"]:
        base["temperature_c"] += ctrl["cool_rate"] * dt
    if ctrl["fault_fan"]:
        base["fan_rpm"] -= ctrl["fan_drop"] * dt
    if ctrl["fault_voltage"]:
        # Sinus mit 2π f t
        base["voltage_v"] += ctrl["volt_amp"] * math.sin(2 * math.pi * ctrl["volt_freq"] * t_sec)

    # One-shot Boosts (werden nach Anwendung im Worker auf 0 zurückgesetzt)
    base["temperature_c"] += ctrl.get("boost_temp", 0.0)
    base["voltage_v"]     += ctrl.get("boost_volt", 0.0)

    # Rauschen
    rng = np.random.default_rng()
    base["temperature_c"] += float(rng.uniform(-0.2, 0.2))
    base["vibration_rms"] += float(rng.uniform(-0.02, 0.02))
    base["current_a"]     += float(rng.uniform(-2, 2))
    base["voltage_v"]     += float(rng.uniform(-1.5, 1.5))
    base["fan_rpm"]       += float(rng.uniform(-30, 30))
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

def background_worker(stop_evt: threading.Event, db_path: str):
    conn = open_db(db_path)
    init_db(conn)
    t_sec = 0.0
    while not stop_evt.is_set():
        try:
            ctrl = control_get(conn)
            dt = max(0.05, (ctrl["tick_ms"] or 1000) / 1000.0)  # Sicherheitsuntergrenze 50 ms
            if ctrl["running"] == 1:
                eq_id = ctrl["eq_id"] or "10109812-01"
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                vals = sim_generate_sample(eq_id, t_sec, dt, ctrl)
                row = {"ts": ts, "equipment_id": eq_id, **vals}
                db_save_measurement(conn, row)
                sim_check_thresholds(conn, eq_id, vals, ts)
                score = sim_ml_anomaly(conn, eq_id, window=ctrl["ml_window"], contamination=ctrl["ml_cont"])
                if score is not None:
                    if score >= ctrl["ml_alert"]:
                        db_save_alarm(conn, ts, eq_id, "ALERT", f"ML anomaly score={score:.2f}")
                    elif score >= (ctrl["ml_alert"] * 0.7):
                        db_save_alarm(conn, ts, eq_id, "WARN", f"ML anomaly score={score:.2f}")

                # One-shot Boosts zurücksetzen, falls ungleich 0 angewendet
                if abs(ctrl.get("boost_temp", 0.0)) > 1e-9 or abs(ctrl.get("boost_volt", 0.0)) > 1e-9:
                    control_update(conn, boost_temp=0.0, boost_volt=0.0)

                t_sec += dt
            time.sleep(dt)
        except sqlite3.OperationalError:
            time.sleep(0.05)
            continue
        except Exception:
            time.sleep(0.1)
            continue
    try:
        conn.close()
    except Exception:
        pass

@st.cache_resource
def get_db_and_worker():
    ui_conn = open_db("data.db")
    init_db(ui_conn)
    stop_evt = threading.Event()
    thread = threading.Thread(target=background_worker, args=(stop_evt, "data.db"), daemon=True)
    thread.start()
    return ui_conn, stop_evt

DB_CONN, _STOP = get_db_and_worker()

# ---------------- SESSION STATE ----------------
if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame(columns=["ts","equipment_id","temperature_c","vibration_rms","current_a","voltage_v","fan_rpm"])
if "alarms" not in st.session_state:
    st.session_state.alarms = []
if "eq_num" not in st.session_state:
    st.session_state.eq_num = "10109812-01"
if "ml_level" not in st.session_state:
    st.session_state.ml_level = "OK"
if "ml_since" not in st.session_state:
    st.session_state.ml_since = datetime.now()

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

# ---------------- TABS (ohne Live-Charts – die ziehen wir ins Overview) ----------------
tab_overview, tab_alerts, tab_settings, tab_misc = st.tabs(
    ["Overview", "Alerts", "Settings", "Sonstiges"]
)

# ---------------- OVERVIEW ----------------
with tab_overview:
    st.subheader("Gesamtzustand")
    df_live = db_load_measurements(DB_CONN, eq_id=st.session_state.eq_num, limit=2000)
    st.session_state.df = df_live.copy()
    df_alarms = db_load_alarms(DB_CONN, eq_id=st.session_state.eq_num, limit=2000)
    st.session_state.alarms = df_alarms.to_dict("records")

    # KPIs
    if len(df_live):
        latest = df_live.iloc[-1]
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Temperatur (°C)", f"{latest['temperature_c']:.1f}")
        k2.metric("Vibration (RMS)", f"{latest['vibration_rms']:.2f}")
        k3.metric("Strom (A)", f"{latest['current_a']:.1f}")
        k4.metric("Spannung (V)", f"{latest['voltage_v']:.1f}")
        k5.metric("Lüfter (RPM)", f"{latest['fan_rpm']:.0f}")
    else:
        st.info("Noch keine Daten.")

    # --- Live Chart direkt im Overview ---
    st.markdown("#### Live Chart")
    if len(df_live):
        st.line_chart(df_live.set_index("ts")[METRICS], use_container_width=True)
    else:
        st.info("Noch keine Daten für das Diagramm.")

    st.markdown("---")

    # --- Fault Injection (Regler + Boost) ---
    st.subheader("Fault Injection (sofort wirksam)")
    ctrl_now = control_get(DB_CONN)
    fr1, fr2, fr3, fr4, fr5 = st.columns([1,1,1,1,1])

    with fr1:
        cool_rate = st.slider("Cooling ΔT/s", 0.00, 0.50, float(ctrl_now["cool_rate"]), 0.01, help="Temperatur-Steigrate pro Sekunde")
        cooling_on = st.toggle("Cooling ON", value=bool(ctrl_now["fault_cooling"]))
    with fr2:
        fan_drop = st.slider("Fan ΔRPM/s", 0.0, 30.0, float(ctrl_now["fan_drop"]), 1.0, help="RPM-Abfall pro Sekunde")
        fan_on = st.toggle("Fan ON", value=bool(ctrl_now["fault_fan"]))
    with fr3:
        volt_amp = st.slider("Voltage Amplitude (V)", 0.0, 60.0, float(ctrl_now["volt_amp"]), 1.0)
        volt_on = st.toggle("Voltage ON", value=bool(ctrl_now["fault_voltage"]))
    with fr4:
        volt_freq = st.slider("Voltage Frequenz (Hz)", 0.10, 2.00, float(ctrl_now["volt_freq"]), 0.05)
        tick_ms = st.slider("Tick (ms)", 200, 1000, int(ctrl_now["tick_ms"]), 50, help="Simulations-Tickrate")
    with fr5:
        st.caption("Boosts (einmalig beim nächsten Tick)")
        b1 = st.button("Temp +10°C", use_container_width=True)
        b2 = st.button("Volt +40V", use_container_width=True)

    # Nur schreiben, wenn sich Werte ändern (entlastet DB)
    updates = {}
    if abs(cool_rate - float(ctrl_now["cool_rate"])) > 1e-9: updates["cool_rate"] = float(cool_rate)
    if int(cooling_on) != int(ctrl_now["fault_cooling"]): updates["fault_cooling"] = 1 if cooling_on else 0
    if abs(fan_drop - float(ctrl_now["fan_drop"])) > 1e-9: updates["fan_drop"] = float(fan_drop)
    if int(fan_on) != int(ctrl_now["fault_fan"]): updates["fault_fan"] = 1 if fan_on else 0
    if abs(volt_amp - float(ctrl_now["volt_amp"])) > 1e-9: updates["volt_amp"] = float(volt_amp)
    if abs(volt_freq - float(ctrl_now["volt_freq"])) > 1e-9: updates["volt_freq"] = float(volt_freq)
    if int(volt_on) != int(ctrl_now["fault_voltage"]): updates["fault_voltage"] = 1 if volt_on else 0
    if int(tick_ms) != int(ctrl_now["tick_ms"]): updates["tick_ms"] = int(tick_ms)
    # EQ-Wechsel mitziehen
    if (ctrl_now.get("eq_id") or "") != st.session_state.eq_num:
        updates["eq_id"] = st.session_state.eq_num
    if updates:
        control_update(DB_CONN, **updates)

    if b1:
        control_update(DB_CONN, boost_temp=10.0)
        st.success("Booster gesetzt: +10°C")
    if b2:
        control_update(DB_CONN, boost_volt=40.0)
        st.success("Booster gesetzt: +40V")

    st.markdown("---")

    # --- ML Statusblock (Progress/ETA/Badge) ---
    st.subheader("KI-Anomalie – Status")
    cvals = control_get(DB_CONN)
    window = int(cvals["ml_window"])
    contamination = float(cvals["ml_cont"])
    alert_t = float(cvals["ml_alert"])
    tick_hz = 1000.0 / max(1, cvals["tick_ms"])
    st.caption(f"Fenstergröße: {window} • Kontamination: {contamination:.3f} • Alert-Schwelle: {alert_t:.2f} • Tick: {tick_hz:.1f} Hz")

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

    score = quick_ml_score(df_live, window, contamination)
    lvl = "OK"
    if score is not None:
        if score >= alert_t: lvl = "ALERT"
        elif score >= (alert_t * 0.7): lvl = "WARN"

    # Badge + „seit …“
    if lvl != st.session_state.ml_level:
        st.session_state.ml_level = lvl
        st.session_state.ml_since = datetime.now()
    since_delta = datetime.now() - st.session_state.ml_since
    since_txt = str(timedelta(seconds=int(since_delta.total_seconds())))

    badge_class = {"OK":"badge-ok","WARN":"badge-warn","ALERT":"badge-alert"}[lvl]
    st.markdown(f'<span class="badge {badge_class}">Health: {lvl}</span>  <span class="small-muted">seit {since_txt}</span>', unsafe_allow_html=True)

    # Score & Progress/ETA
    if score is None:
        have = len(df_live)
        pct = int(min(100, have * 100 / max(1, window)))
        missing = max(0, window - have)
        eta_sec = int(missing * (cvals["tick_ms"]/1000.0))
        st.markdown(f"**ML-Score:** – (zu wenig Daten) • {have}/{window}")
        st.markdown(f'<div class="progress"><div style="width:{pct}%"></div></div>', unsafe_allow_html=True)
        st.caption(f"ETA bis erste Bewertung: ~{timedelta(seconds=eta_sec)}")
    else:
        last_ts = df_live.iloc[-1]["ts"]
        st.markdown(f"**ML-Score:** {score:.2f}  •  zuletzt bewertet: {last_ts.strftime('%H:%M:%S')}")
        st.caption("WARN ab ~70 % der Alert-Schwelle, ALERT ab 100 % der Schwelle.")

    st.markdown("---")

    # --- Analyse-Export ---
    st.subheader("Analyse-Export")
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
    colA, colB = st.columns(2)
    with colA:
        st.download_button(
            "⬇️ Export Analyse (CSV)",
            data=analysis_df.to_csv(index=False).encode("utf-8"),
            file_name=f"analysis_{st.session_state.eq_num}.csv",
            mime="text/csv",
            use_container_width=True,
            key="dl_analysis_csv_overview",
        )
    with colB:
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

# ---------------- ALERTS ----------------
with tab_alerts:
    st.subheader("Alarm-Feed (neueste zuerst)")
    df_alarms = db_load_alarms(DB_CONN, eq_id=st.session_state.eq_num, limit=500)
    if not df_alarms.empty:
        for _, a in df_alarms.iloc[::-1].iterrows():
            (st.error if a["level"] == "ALERT" else st.warning)(f"[{a['ts']}] {a['message']}")
    else:
        st.info("Keine Alarme.")

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
            _execute_write_retry(DB_CONN, "DELETE FROM measurements", ())
            _execute_write_retry(DB_CONN, "DELETE FROM alarms", ())
            st.session_state.df = st.session_state.df.iloc[0:0]
            st.session_state.alarms = []
            st.success("Daten & Alarme gelöscht. Einstellungen unverändert.")

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

    st.markdown("---")
    st.subheader("KI-Anomalie (IsolationForest) – Parameter (mit Übernehmen)")

    # Staging der ML-Parameter
    ctrl_now = control_get(DB_CONN)
    if "ml_window_tmp" not in st.session_state: st.session_state.ml_window_tmp = int(ctrl_now["ml_window"])
    if "ml_cont_tmp" not in st.session_state:   st.session_state.ml_cont_tmp = float(ctrl_now["ml_cont"])
    if "ml_alert_tmp" not in st.session_state:  st.session_state.ml_alert_tmp = float(ctrl_now["ml_alert"])

    c1, c2, c3 = st.columns(3)
    with c1:
        st.session_state.ml_window_tmp = st.slider("Fenstergröße (Punkte)", 200, 2000, int(st.session_state.ml_window_tmp), 50, key="ml_window_tmp_slider")
    with c2:
        st.session_state.ml_cont_tmp = st.slider("Kontamination (erwartete Ausreißer)", 0.001, 0.10, float(st.session_state.ml_cont_tmp), 0.001, key="ml_cont_tmp_slider")
    with c3:
        st.session_state.ml_alert_tmp = st.slider("ML-Alert-Schwelle (0–1)", 0.10, 0.90, float(st.session_state.ml_alert_tmp), 0.05, key="ml_alert_tmp_slider")

    ucol1, ucol2 = st.columns([1,3])
    with ucol1:
        if st.button("✅ Übernehmen", use_container_width=True):
            control_update(
                DB_CONN,
                ml_window=int(st.session_state.ml_window_tmp),
                ml_cont=float(st.session_state.ml_cont_tmp),
                ml_alert=float(st.session_state.ml_alert_tmp),
            )
            st.success("ML-Parameter übernommen.")
    with ucol2:
        st.caption(f"Aktive Werte: window={ctrl_now['ml_window']}, contamination={ctrl_now['ml_cont']:.3f}, alert={ctrl_now['ml_alert']:.2f}")

# ---------------- SONSTIGES (Vorzeigen) ----------------
with tab_misc:
    st.subheader("IsolationForest – Illustration")
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
        st.caption("Goldene Punkte = normales Verhalten. Rotes ✗ = Ausreißer.")

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
    except Exception:
        st.warning("Matplotlib fehlt – bitte `matplotlib` in requirements.txt ergänzen.")

# --------- AUTO-REFRESH: solange running=1, alle ~Tick ms neu rendern ---------
try:
    _ctrl = control_get(DB_CONN)
    if int(_ctrl.get("running", 0)) == 1:
        time.sleep(max(0.2, _ctrl["tick_ms"]/1000.0))
        st.rerun()
except Exception:
    pass