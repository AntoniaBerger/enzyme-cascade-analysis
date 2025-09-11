import os 
import pandas as pd
import numpy as np
from pyparsing import col
from scipy.stats import linregress

def is_linear(x, y, threshold=0.77):
    """Prüft ob Daten linear sind basierend auf R (gelockerte Kriterien für mehr Datenpunkte)"""
    if len(x) < 3 or len(y) < 3:
        return False
    try:
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        return abs(r_value) > threshold
    except Exception:
        return False

# bisher ohne concentrations noise
def add_noise_calibration(data, noise_level=0.1, conc_noise_level=0.0):
    
    conc_data = data["concentration"].copy()
    std_dev_conc = conc_data.std()
    conc_data += np.random.normal(0, std_dev_conc * conc_noise_level, size=conc_data.shape)

    RD_1_noisy = data["RD_1"].copy()
    std_dev_1 = RD_1_noisy.std()
    RD_1_noisy += np.random.normal(0, std_dev_1 * noise_level, size=RD_1_noisy.shape)
    
    RD_2_noisy = data["RD_2"].copy()
    std_dev_2 = RD_2_noisy.std()
    RD_2_noisy += np.random.normal(0, std_dev_2 * noise_level, size=RD_2_noisy.shape)

    noisy_data = pd.DataFrame({
        "concentration": conc_data,
        "RD_1": RD_1_noisy,
        "RD_2": RD_2_noisy
    })

    return noisy_data

def add_noise_substrate(data, noise_level=0.1, verbose=False):
    
    # extrahiere time which should remain unchanged
    time_points = data.iloc[[0]]
    data_to_noise = data.iloc[1:].copy()
    noisy_data = data_to_noise.copy()
    
    try:
        concentration_columns = [col for col in data_to_noise.columns if 'Konzentration_mM' in col or 'Konzentration' in col]

        # Identifizieren Sie die Absorptionsspalten (Raw Data (340))
        absorption_columns = [col for col in data_to_noise.columns if 'Raw Data' in col or '340' in col]

        for col in concentration_columns:
            mask = noisy_data[col].notna()  # Nur gültige Konzentrationen
            std_dev = noisy_data[col][mask].std()
            noise = np.random.normal(0, std_dev * 0.0, size=mask.sum())
            noisy_data.loc[mask, col] = data_to_noise.loc[mask, col] + noise

        for col in absorption_columns:
            mask = noisy_data[col].notna()  # Nur gültige Absorptionswerte
            std_dev = noisy_data[col][mask].std()
            noise = np.random.normal(0, std_dev * noise_level, size=mask.sum())
            noisy_data.loc[mask, col] = data_to_noise.loc[mask, col] + noise

    except Exception as e:
        print(f"Fehler beim Hinzufügen von Rauschen: {e}")
        return data
    
    noisy_data = pd.concat([time_points, noisy_data], ignore_index=True)
    return noisy_data

def add_noise_concentrations(data, noise_level=0.1, verbose=False):
    """
    Fügt Rauschen zu Konzentrationen in Plate Reader Daten hinzu.
    Sucht nach Spalten mit 'Konzentration_mM' oder 'Concentration' im Namen.
    
    Args:
        data: DataFrame mit Plate Reader Daten
        noise_level: Stärke des Rauschens (Standard: 0.1 = 10%)
        verbose: Debug-Ausgabe
    
    Returns:
        DataFrame mit verrauschten Konzentrationen
    """
    if noise_level == 0.0:
        return data
        
    noisy_data = data.copy()
    
    try:
        # Identifiziere Konzentrationsspalten
        concentration_columns = [col for col in data.columns if 'Konzentration_mM' in str(col) or 'Concentration' in str(col)]
        
        if verbose and concentration_columns:
            print(f"Gefundene Konzentrationsspalten: {concentration_columns}")
        
        for col in concentration_columns:
            if col in noisy_data.columns:
                # Nur numerische, gültige Konzentrationen verrauschen
                mask = noisy_data[col].notna()
                
                # Versuche die Werte zu numerischen Werten zu konvertieren
                numeric_values = pd.to_numeric(noisy_data[col], errors='coerce')
                numeric_mask = mask & numeric_values.notna() & (numeric_values > 0)
                
                if numeric_mask.sum() > 0:
                    valid_values = numeric_values[numeric_mask]
                    std_dev = valid_values.std()
                    
                    if std_dev > 0:
                        noise = np.random.normal(0, std_dev * noise_level, size=numeric_mask.sum())
                        noisy_values = valid_values + noise
                        
                        # Stelle sicher, dass Konzentrationen nicht negativ werden
                        noisy_values = np.maximum(noisy_values, 0.001)  # Mindestkonzentration
                        
                        noisy_data.loc[numeric_mask, col] = noisy_values
                        
                        if verbose:
                            print(f"Rauschen zu Spalte '{col}' hinzugefügt: {numeric_mask.sum()} Werte verändert")
                
    except Exception as e:
        if verbose:
            print(f"Fehler beim Hinzufügen von Konzentrations-Rauschen: {e}")
        return data
    
    return noisy_data
    

def add_noise_plate_reader_data(reaction_data_dict, noise_level=0.1, noise_level_conc=0.0, verbose=False):
    """
    Fügt Rauschen zu Plate Reader Daten hinzu.
    
    Args:
        reaction_data_dict: Dictionary mit Reaktionsdaten
        noise_level: Rauschen für Absorptionsdaten (Standard: 0.1 = 10%)
        noise_level_conc: Rauschen für Konzentrationen (Standard: 0.0 = kein Rauschen)
        verbose: Debug-Ausgabe
    
    Returns:
        Dictionary mit verrauschten Daten
    """
    noisy_dict = {}
    
    for reaction_name, dataframe in reaction_data_dict.items():
        if verbose:
            print(f"Verrausche Reaktion: {reaction_name}")
        reaction_dict = {}
        for component in dataframe:
            if verbose:
                print(f"  Komponente: {component}")
            # Erst Absorptionsdaten verrauschen
            noisy_component = add_noise_substrate(dataframe[component], noise_level, verbose=False)
            # Dann Konzentrationen verrauschen
            if noise_level_conc > 0:
                noisy_component = add_noise_concentrations(noisy_component, noise_level_conc, verbose=verbose)
            reaction_dict[component] = noisy_component
        noisy_dict[reaction_name] = reaction_dict

    return noisy_dict

def add_noise_processed_data(df, noise_level=0.1, noise_level_conc=0.0, verbose=False):
    """
    Fügt Rauschen zu verarbeiteten Daten hinzu.
    
    Args:
        df: DataFrame mit verarbeiteten Daten
        noise_level: Rauschen für Raten (Standard: 0.1 = 10%)
        noise_level_conc: Rauschen für Konzentrationen (Standard: 0.0 = kein Rauschen)
        verbose: Debug-Ausgabe
    
    Returns:
        DataFrame mit verrauschten Daten
    """
    noisy_df = df.copy()
    try:
        # Rauschen zu Konzentrationen hinzufügen (falls gewünscht)
        if noise_level_conc > 0:
            concentration_columns = [col for col in df.columns if col.startswith('c')]
            for col in concentration_columns:
                mask = noisy_df[col].notna()  # Nur gültige Konzentrationen
                if mask.sum() > 0:
                    std_dev = noisy_df[col][mask].std()
                    if std_dev > 0:
                        noise = np.random.normal(0, std_dev * noise_level_conc, size=mask.sum())
                        noisy_values = noisy_df.loc[mask, col] + noise
                        # Stelle sicher, dass Konzentrationen nicht negativ werden
                        noisy_values = np.maximum(noisy_values, 0.001)
                        noisy_df.loc[mask, col] = noisy_values
                        if verbose:
                            print(f"Rauschen zu Konzentrationsspalte '{col}' hinzugefügt")

        # Rauschen zu Raten hinzufügen
        if 'rates' in df.columns:
            mask = noisy_df['rates'].notna()  # Nur gültige Raten
            if mask.sum() > 0:
                std_dev = noisy_df['rates'][mask].std()
                if std_dev > 0:
                    noise = np.random.normal(0, std_dev * noise_level, size=mask.sum())
                    noisy_df.loc[mask, 'rates'] = df.loc[mask, 'rates'] + noise
                    if verbose:
                        print(f"Rauschen zu Ratenspalte hinzugefügt")

    except Exception as e:
        if verbose:
            print(f"Fehler beim Hinzufügen von Rauschen zu verarbeiteten Daten: {e}")
        return df

    return noisy_df

def get_concentrations_from_csv(csv_data):
    """Extrahiert die Konzentrationen aus CSV-Kinetikdaten"""
    # Spalte 2 (Index 1) enthält die Konzentrationen, ab Zeile 3 (Index 2)
    concentrations_raw = csv_data.iloc[1:, 1].dropna().values
    concentrations = [float(x) for x in concentrations_raw]

    return np.array(concentrations)

def get_absorption_data(csv_data):
    """Extrahiert die Absorptionsdaten aus CSV-Kinetikdaten"""
    # Ab Spalte 3 (Index 2) sind die Absorptionsdaten, ab Zeile 3 (Index 2)
    absorption_data = csv_data.iloc[1:, 2:].values
    absorption_data_clean = []
    
    for row in absorption_data:
        clean_row = []
        for val in row:
            try:
                if pd.notna(val):
                    clean_row.append(float(val))
                else:
                    break  # Stoppe bei NaN-Werten
            except (ValueError, TypeError):
                break
        if len(clean_row) > 0:
            absorption_data_clean.append(clean_row)
    
    return np.array(absorption_data_clean, dtype=object)

def get_time_points(csv_data):
    """Extrahiert die Zeitpunkte aus CSV-Kinetikdaten"""
    # Die Zeitpunkte stehen in der ERSTEN Zeile (Index 0), ab Spalte 3 (Index 2)
    time_row = csv_data.iloc[0, 2:].values  # Zeile 1 (Time [s]), ab Spalte 2
    
    # Konvertiere zu numerischen Werten und entferne NaN
    time_points = []
    for i, t in enumerate(time_row):
        try:
            if pd.notna(t):
                time_points.append(float(t))
            else:
                break
        except (ValueError, TypeError):
            break
    
    return np.array(time_points)

def calc_calibration_slope(data, verbose=False):
    """
    Berechnet die Kalibrierung basierend auf NADH-Konzentrationen und Extinktionen.
    Unterstützt sowohl pandas DataFrames als auch experiment_data-Dicts.
    """
    # Flexibler Zugriff: DataFrame oder Dict
    if isinstance(data, dict):
        # experiment_data-Dict Format
        x = data.get('concentrations', data.get('NADH', []))
        y = data.get('mean', data.get('Mittelwert', data.get('extinction', [])))
    else:
        # Pandas DataFrame 
        x = data["concentration"]
        y1 = data["RD_1"]
        y2 = data["RD_2"]

    y = np.mean(np.array([y1, y2]), axis=0) if isinstance(data, pd.DataFrame) else y
    
    if not is_linear(x, y):
        if verbose:
            print("Die Daten sind nicht linear. Bitte überprüfen Sie die Kalibrierung.")
        return None

    slope_cal, intercept_cal, r_value_cal, _, _ = linregress(x, y)
    if verbose:
        print(f"Kalibrierung: Steigung = {slope_cal:.2f}, R² = {r_value_cal**2:.4f}")
    
    return slope_cal

def get_rate_conc(concentrations, absorption_data, time_points, slope_cal, activ_param, verbose=True):
    """
    Berechnet Aktivitäten und gibt sie als experiment_data-Dict zurück.
    Kompatibel mit der neuen modularen Datenstruktur.
    """
    initial_rates = []
    valid_concentrations = []

    for i, conc in enumerate(concentrations):
        try:
            conc_float = float(conc)
        except (ValueError, TypeError):
            if verbose:
                print(f"Well {i+1}: Ungültige Konzentration '{conc}' - übersprungen")
            continue
            
        if pd.notna(conc_float) and i < len(absorption_data):
            # Absorptionswerte für dieses Well
            if isinstance(absorption_data, np.ndarray) and absorption_data.dtype == object:
                # Array of arrays (unterschiedliche Längen)
                absorbance = absorption_data[i]
            else:
                # Normale 2D Matrix
                absorbance = absorption_data[i, :]
            
            # Zu numerischen Werten konvertieren
            absorbance = pd.to_numeric(absorbance, errors='coerce')
            
            # Nur gültige Werte verwenden
            valid_indices = ~np.isnan(absorbance)
            if np.sum(valid_indices) < 3:
                if verbose:
                    print(f"Well {i+1}: Nicht genug gültige Absorptionswerte ({np.sum(valid_indices)} von {len(absorbance)})")
                continue
                
            # Stellen Sie sicher, dass valid_indices und time_points gleich lang sind
            min_len = min(len(valid_indices), len(time_points))
            valid_indices = valid_indices[:min_len]
            
            time_valid = time_points[:min_len][valid_indices]
            abs_valid = absorbance[:min_len][valid_indices]
            
            if len(time_valid) < 3:
                if verbose:
                    print(f"Well {i+1}: Nach Validierung zu wenig Punkte ({len(time_valid)})")
                continue
            
            # Zusätzlich prüfen, ob time_points auch gültig sind
            time_valid_mask = ~np.isnan(time_valid)
            if np.sum(time_valid_mask) < 3:
                if verbose:
                    print(f"Well {i+1}: Nicht genug gültige Zeitpunkte")
                continue
                
            time_final = time_valid[time_valid_mask]
            abs_final = abs_valid[time_valid_mask]
            
            # Linearität prüfen mit Debug-Info
            slope, intercept_test, r_value_test, p_value_test, std_err_test = linregress(time_final, abs_final)
            r_squared = r_value_test**2
            
            if verbose and r_squared < 0.50:
                print(f"Well {i+1} (Konz: {conc_float} mM): R² = {r_squared:.3f} - nicht linear genug")
                continue
            elif not is_linear(time_final, abs_final):
                if verbose:
                    print(f"Well {i+1} (Konz: {conc_float} mM): R² = {r_squared:.3f} - ist nicht linear")
                continue
                        
            # Umrechnung nach Ihrer Formel: A[U/mg] = (m1 * 60 * Vf_well * Vf_prod) / (m2 * c_prod)
            Vf_well = activ_param["Vf_well"]          # Verdünnung im Well (20μL + 180μL assay )
            Vf_prod = activ_param["Vf_prod"]          # Verdünnung der Proteinlösung
            c_prod = activ_param["c_prod"]            # Gemessene Proteinkonzentration [mg/L]

            activity_U_per_mg = (abs(slope) * 60 * Vf_well * Vf_prod) / (slope_cal * c_prod) #! Formel prüfen
            
            # Aktivität in U/mg
            activity_U = activity_U_per_mg  # Hier als U/mg belassen

            # Plausibilitätsprüfung
            if activity_U > 0 and activity_U < 1000:
                initial_rates.append(activity_U)
                valid_concentrations.append(conc_float)
                
                if verbose:
                    print(f"Well {i+1}: {conc_float:.2f} mM → Aktivität: {activity_U:.6f} U/mg")

    # Umwandlung in numpy Array
    initial_rates = np.array(initial_rates)
    valid_concentrations = np.array(valid_concentrations)
    
    # Längen-Check: Aktivitäten und Konzentrationen müssen gleich lang sein
    if len(initial_rates) != len(valid_concentrations):
        if verbose:
            print(f"FEHLER: Längen stimmen nicht überein - Activities: {len(initial_rates)}, Concentrations: {len(valid_concentrations)}")
        return None
        
    if len(initial_rates) < 3:
        if verbose:
            print("Nicht genug gültige initiale Raten für die Parameterabschätzung.")
        return None

    if verbose:
        print(f"Erfolgreich {len(initial_rates)} gültige Datenpunkte verarbeitet")

    return  initial_rates , valid_concentrations

def compute_processed_data(reaction_data_dict, slope, reaction_params_dict, noise_level_conc = 0.0, verbose=False):

    processed_data_dict = {
        "reaction" : [],
        "rates" : []
    }
    
    for c in range(reaction_params_dict["x_dimension"]):
        processed_data_dict[f"c{c+1}"] = []

    for reaction_name, reaction_data in reaction_data_dict.items():
        if verbose:
            print(f"Verarbeite Reaktion: {reaction_name}")
        
        try:
            # Hole die entsprechenden Aktivitätsparameter
            if reaction_name in reaction_params_dict:
                params_dict = reaction_params_dict[reaction_name]
            else:
                if verbose:
                    print(f"Warning: Keine Aktivitätsparameter für {reaction_name} gefunden")
                processed_data_dict = None
                continue

            i = 1
            for component_name, experimental_data in reaction_data.items():
                # Extrahiere Daten aus CSV
                concentrations = get_concentrations_from_csv(experimental_data)
                absorption_data = get_absorption_data(experimental_data)
                time_points = get_time_points(experimental_data)

                # Berechne Aktivitäten
                result = get_rate_conc(
                    concentrations, absorption_data, time_points, 
                    slope, params_dict, verbose=verbose
                )
                
                if result is not None:
                    activities, valid_concentrations = result
                    if (len(activities) != len(valid_concentrations)):
                        if verbose:
                            print(f"FEHLER: Längen stimmen nicht überein - Activities: {len(activities)}, Concentrations: {len(valid_concentrations)}")
                        continue

                    number_of_valid_concentrations = len(valid_concentrations)

                    processed_data_dict[f'c{i}'].extend(valid_concentrations)
                    processed_data_dict['rates'].extend(activities)
                    processed_data_dict['reaction'].extend([int(reaction_name[1:])] * number_of_valid_concentrations)

                    for j in range(reaction_params_dict["x_dimension"]):
                        if j+1 != i: 
                            # falls der eintrag existiert: 
                            if f'c{j+1}_const' in reaction_params_dict[reaction_name]:
                                constant_value = [reaction_params_dict[reaction_name][f'c{j+1}_const']] * number_of_valid_concentrations
                                if noise_level_conc > 0.0:
                                    const_value = reaction_params_dict[reaction_name][f'c{j+1}_const']
                                    std_dev_const = const_value * noise_level_conc
                                    noisy_constants = np.random.normal(const_value, std_dev_const, size=number_of_valid_concentrations)
                                    noisy_constants = np.maximum(noisy_constants, 0.000)
                                processed_data_dict[f'c{j+1}'].extend(constant_value)
                            else: 
                                processed_data_dict[f'c{j+1}'].extend([0.0] * number_of_valid_concentrations)
                    i += 1
                    if verbose:
                        print(f"✓ {reaction_name}: {len(activities)} gültige Datenpunkte")
                else:
                    if verbose:
                        print(f" {reaction_name}: Keine gültigen Daten")
                    processed_data_dict = None
            
        except Exception as e:
            if verbose:
                print(f" Fehler bei {reaction_name}: {e}")
            processed_data_dict = None

    df = pd.DataFrame(processed_data_dict)

    if verbose:
        print("Anzahl der Datenpunkte pro Reaction")
        print(df["reaction"].value_counts())

    return df
