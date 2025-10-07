# Predictive Maintenance – Rectifier (Streamlit)

Ein Demo-Dashboard **nah an SAP PdM**: ein Gleichrichter (Equipment-ID z. B. `RECT-0001`) wird live simuliert. 
**Schwellwerte** (WARN/ALERT) und **KI-Anomalie** (IsolationForest) erzeugen Alerts. 
Ziel: realistische PdM-Präsentation auf **Desktop & Handy** über **eine URL**.

## Features
- **Equipment-ID** wie in SAP (Dummy-Nummer frei wählbar)
- **Live-Simulation** (Temperatur, Vibration RMS, Strom, Spannung, Lüfter-RPM)
- **Fault Injection**: Cooling-Degradation, Fan-Wear, Voltage-Spikes
- **Schwellwert-Überwachung** (WARN/ALERT pro Metrik)
- **KI-Anomalie-Erkennung** (IsolationForest) mit Score & Health-Badge (OK/WARN/ALERT)
- **Alerts-Feed** (jüngste Ereignisse oben), **Live-Charts** & **KPIs**

## Bedienung
1. **Simulation starten** (▶️) im Tab **Settings**.
2. Optional **Faults aktivieren** (Cooling/Fan/Voltage).
3. **Schwellwerte** anpassen (WARN/ALERT) – sofort wirksam.
4. **KI**: Fenstergröße, Kontamination & Alert-Schwelle justieren.
5. **Overview** zeigt KPIs & Health; **Live Charts** die Zeitreihen; **Alerts** listet Ereignisse.

## Tech
- **Streamlit** für die UI  
- **scikit-learn (IsolationForest)** für ML-Anomalien  
- **pandas/numpy** für Datenhaltung

## Lokal starten (optional)
```bash
pip install -r requirements.txt
streamlit run app.py
