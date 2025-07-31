# Datenladung für Fehlerfortpflanzungsanalyse

Dieses Verzeichnis enthält Skripte zum Einlesen und Verwalten von Daten für die Fehlerfortpflanzungsanalyse von Michaelis-Menten-Enzymkinetiken.

## 📁 Dateien

- `daten_einlesen.py` - Hauptskript zum Einlesen aller Datentypen
- `beispiel_verwendung.py` - Beispiele für die Verwendung des DataLoaders
- `analysis_utilily.py` - Hilfsfunktionen für die Datenanalyse
- `README.md` - Diese Dokumentation

## 🚀 Schnellstart

### 1. Alle Daten laden und speichern

```python
from daten_einlesen import DataLoader

# DataLoader initialisieren
loader = DataLoader()

# Alle Daten laden
loader.load_all_data()

# Daten speichern
loader.save_data(format='pickle')  # Als Pickle-Datei
loader.save_data(format='excel')   # Als Excel-Dateien
```

### 2. Skript direkt ausführen

```bash
python daten_einlesen.py
```

Dies lädt automatisch alle verfügbaren Daten und speichert sie in verschiedenen Formaten.

## 📊 Unterstützte Datentypen

### Kalibrierungsdaten
- **Plate Reader**: NADH, NADPH Kalibriergeraden
- **GC-MS**: Kalibrierungen verschiedener Verbindungen

### Kinetik-Messungen
- **r1**: VL_NADH, NAD_PD Messungen
- **r2**: Lactol_NADH_PD Messungen  
- **r3**: NAD_Lactol, NADH Messungen
- **Enzym-Inaktivierung**: Stabilitätsmessungen

### GC-MS Daten
- **Hydrolyse**: Hydrolysereaktionen
- **Validierung**: Simulationsvalidierung

### Bestehende Daten
- **CSV-Dateien**: Bereits prozessierte Michaelis-Menten Daten

## 🔧 Verwendung

### DataLoader Klasse

```python
from daten_einlesen import DataLoader

# Initialisierung
loader = DataLoader()

# Spezifische Datentypen laden
loader.load_calibration_data()      # Nur Kalibrierungen
loader.load_kinetic_measurements()  # Nur Kinetik-Messungen
loader.load_gcms_data()            # Nur GC-MS Daten

# Alle Daten laden
loader.load_all_data()

# Daten abrufen
all_data = loader.get_data()                    # Alle Daten
calibration = loader.get_data('plate_reader_calibration')  # Spezifisches Experiment
```

### Datenformat

Die geladenen Daten werden in einem verschachtelten Dictionary gespeichert:

```python
{
    'experiment_name': {
        'file_name': pandas.DataFrame,
        'another_file': pandas.DataFrame
    },
    'another_experiment': pandas.DataFrame
}
```

### Datenzugriff

```python
# Übersicht der verfügbaren Experimente
experiments = loader.get_data()
print(list(experiments.keys()))

# Spezifisches Experiment
if 'plate_reader_calibration' in experiments:
    calibration_data = experiments['plate_reader_calibration']
    
    # Dateien in diesem Experiment
    print(list(calibration_data.keys()))
    
    # Spezifische Datei
    nadh_data = calibration_data['NADH_Kalibriergerade']
    print(nadh_data.head())
```

## 💾 Speicheroptionen

### Pickle Format (Empfohlen)
```python
loader.save_data(format='pickle')
```
- Schnell und kompakt
- Erhält alle Datentypen
- Ideal für Python-Weiterverarbeitung

### Excel Format
```python
loader.save_data(format='excel')
```
- Menschenlesbar
- Kompatibel mit Excel/LibreOffice
- Gut für Dateninspektion

## 📂 Datenstruktur

```
Daten/
├── Rohdaten/
│   ├── GC-MS/
│   │   ├── Hydrolyse/
│   │   ├── Kalibriergeraden/
│   │   └── Validierung Simulation/
│   └── Plate Reader/
│       ├── Kalibriergeraden/
│       └── Kinetik-Messungen/
│           ├── r1/
│           ├── r2/
│           ├── r3/
│           └── Enzym-Inaktivierung/
└── processed_data/
    ├── all_data.pkl
    ├── metadata.json
    └── [experiment_name].xlsx
```

## 🔍 Fehlerbehandlung

Das Skript behandelt automatisch:
- Fehlende Dateien (mit Warnung)
- Ungültige Dateiformate
- Leere Dateien
- Encoding-Probleme

## 📈 Beispiele

### Einfache Datenanalyse

```python
# Michaelis-Menten Parameter schätzen
if 'existing_mm_pdiol_lactol' in experiments:
    data = experiments['existing_mm_pdiol_lactol']
    
    x = data['x']
    y = data['y']
    
    # Grobe Parameterschätzung
    vmax_estimate = y.max() * 1.2
    km_estimate = x[abs(y - vmax_estimate/2).argmin()]
    
    print(f"Vmax ≈ {vmax_estimate:.6f}")
    print(f"Km ≈ {km_estimate:.2f}")
```

### Kalibrierungsdaten plotten

```python
import matplotlib.pyplot as plt

calibration = loader.get_data('plate_reader_calibration')
if 'NADH_Kalibriergerade' in calibration:
    nadh_data = calibration['NADH_Kalibriergerade']
    
    plt.figure(figsize=(10, 6))
    plt.scatter(nadh_data['Konzentration'], nadh_data['Absorption'])
    plt.xlabel('Konzentration (µM)')
    plt.ylabel('Absorption')
    plt.title('NADH Kalibrierungsgerade')
    plt.show()
```

## 🆘 Fehlerbehebung

### Häufige Probleme

1. **Datei nicht gefunden**
   - Prüfen Sie den Pfad zur Datei
   - Stellen Sie sicher, dass die Datei existiert

2. **Encoding-Fehler**
   - Speichern Sie Excel-Dateien neu
   - Verwenden Sie UTF-8 Encoding für CSV

3. **Leere DataFrames**
   - Prüfen Sie die Datenstruktur in Excel
   - Stellen Sie sicher, dass Header korrekt sind

### Debug-Modus

```python
# Ausführliche Ausgabe aktivieren
loader = DataLoader()
loader.load_all_data()

# Datenübersicht anzeigen
loader.print_data_overview()
```

## 📧 Support

Bei Problemen oder Fragen:
1. Prüfen Sie die Fehlermeldungen
2. Verwenden Sie `print_data_overview()` für Debug-Informationen
3. Kontaktieren Sie den Entwickler

---

*Entwickelt für die Fehlerfortpflanzungsanalyse von Michaelis-Menten-Enzymkinetiken*
