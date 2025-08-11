import os 
import pandas as pd
import numpy as np
from scipy.stats import linregress

def is_linear(x, y, threshold=0.77):
    """Prüft ob Daten linear sind basierend auf R² (gelockerte Kriterien für mehr Datenpunkte)"""
    if len(x) < 3 or len(y) < 3:
        return False
    try:
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        return abs(r_value) > threshold
    except Exception:
        return False

def add_noise_calibration(data, noise_level=0.1):
    noisy_data = data.copy()
    noise = np.random.normal(0, noise_level, size=data.shape)
    return noisy_data + noise

# working but not optimal better: df version
def add_noise_reaction(data, noise_level=0.1, verbose=False):
    """
    Fügt Rauschen zu Reaktions-CSV-Daten hinzu.
    Speziell für die Struktur: Header, Units, Well-Daten mit Konzentrationen und Absorptionswerten
    
    Args:
        data: pandas DataFrame (CSV-Reaktionsdaten)
        noise_level: Relative Rauschstärke (0.1 = 10%)
        verbose: Bool - Debug-Ausgaben
        
    Returns:
        pandas DataFrame mit verrauschten Daten
    """
    noisy_data = data.copy()
    
    try:
        # Struktur der Reaktions-CSV:
        # Zeile 0: ["Time [s]", "Concentration", "0", "30", "60", ...]  
        # Zeile 1: ["", "mM", "0", "30", "60", ...]
        # Ab Zeile 2: ["Well X", concentration_value, abs_0, abs_30, abs_60, ...]
        
        if verbose:
            print(f"Original DataFrame shape: {data.shape}")
            print(f"Erste 3 Zeilen:\n{data.iloc[:3, :5]}")
        
        # 1. Verrausche Absorptionsdaten (ab Spalte 2, ab Zeile 2)
        absorption_start_row = 2
        absorption_start_col = 2
        
        if data.shape[0] > absorption_start_row and data.shape[1] > absorption_start_col:
            absorption_region = data.iloc[absorption_start_row:, absorption_start_col:]
            
            for row_idx in range(absorption_region.shape[0]):
                for col_idx in range(absorption_region.shape[1]):
                    cell_value = absorption_region.iloc[row_idx, col_idx]
                    
                    # Versuche numerische Konvertierung
                    try:
                        numeric_value = float(cell_value)
                        if not np.isnan(numeric_value):
                            # Addiere relatives Rauschen
                            noise = np.random.normal(0, abs(numeric_value) * noise_level)
                            noisy_value = numeric_value + noise
                            
                            # Setze zurück ins DataFrame
                            actual_row = absorption_start_row + row_idx
                            actual_col = absorption_start_col + col_idx
                            noisy_data.iloc[actual_row, actual_col] = noisy_value
                            
                    except (ValueError, TypeError):
                        # Nicht-numerische Werte ignorieren
                        continue
        
        # 2. Optional: Auch Konzentrationen verrauschen (Spalte 1, ab Zeile 2)  
        if data.shape[0] > absorption_start_row:
            conc_column = data.iloc[absorption_start_row:, 1]  # Spalte 1
            
            for row_idx, conc_value in enumerate(conc_column):
                try:
                    numeric_conc = float(conc_value)
                    if not np.isnan(numeric_conc) and numeric_conc > 0:
                        # Weniger Rauschen für Konzentrationen (meist bekannte Werte)
                        conc_noise = np.random.normal(0, numeric_conc * noise_level * 0.1)
                        noisy_conc = max(0, numeric_conc + conc_noise)  # Keine negativen Konzentrationen
                        
                        actual_row = absorption_start_row + row_idx
                        noisy_data.iloc[actual_row, 1] = noisy_conc
                        
                except (ValueError, TypeError):
                    continue
        
        if verbose:
            print(f"Rauschen hinzugefügt mit Level: {noise_level}")
            
    except Exception as e:
        print(f"Fehler beim Hinzufügen von Rauschen: {e}")
        return data  # Original bei Fehler zurückgeben
        
    return noisy_data

# Füge auch zu data_hadler.py hinzu
def add_noise_reaction_dict(reaction_data_dict, noise_level=0.1, verbose=False):
    """
    Fügt Rauschen zu einem Dictionary von Reaktionsdaten hinzu.
    
    Args:
        reaction_data_dict: Dict mit {reaction_name: DataFrame}
        noise_level: Rauschstärke
        verbose: Debug-Ausgaben
        
    Returns:
        Dict mit verrauschten DataFrames
    """
    noisy_dict = {}
    
    for reaction_name, dataframe in reaction_data_dict.items():
        if verbose:
            print(f"Verrausche Reaktion: {reaction_name}")
        reaction_dict = {}
        for component in dataframe:
            if verbose:
                print(f"  Komponente: {component}")
            reaction_dict[component] = add_noise_reaction(dataframe[component], noise_level, verbose=False)
        noisy_dict[reaction_name] = reaction_dict

    return noisy_dict


def get_concentrations_from_csv(csv_data):
    """Extrahiert die Konzentrationen aus CSV-Kinetikdaten"""
    # Spalte 2 (Index 1) enthält die Konzentrationen, ab Zeile 3 (Index 2)
    concentrations_raw = csv_data.iloc[2:, 1].dropna().values
    concentrations = [float(x) for x in concentrations_raw]

    return np.array(concentrations)

def get_absorption_data(csv_data):
    """Extrahiert die Absorptionsdaten aus CSV-Kinetikdaten"""
    # Ab Spalte 3 (Index 2) sind die Absorptionsdaten, ab Zeile 3 (Index 2)
    absorption_data = csv_data.iloc[2:, 2:].values
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
    # Die Zeitpunkte stehen in der ZWEITEN Zeile (Index 1), ab Spalte 3 (Index 2)
    time_row = csv_data.iloc[1, 2:].values  # Zeile 1 (Time [s]), ab Spalte 2
    
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

def calculate_calibration(data, verbose=False):
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

def calculate_activity(concentrations, absorption_data, time_points, slope_cal, activ_param, verbose=True):
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
            
            if verbose and r_squared < 0.70:
                print(f"Well {i+1} (Konz: {conc_float} mM): R² = {r_squared:.3f} - nicht linear genug")
                continue
            elif not is_linear(time_final, abs_final):
                if verbose:
                    print(f"Well {i+1} (Konz: {conc_float} mM): R² = {r_squared:.3f} - ist nicht linear")
                continue
                        
            # Umrechnung nach Ihrer Formel: A[U/mg] = (m1 * 60 * Vf_well * Vf_prod) / (m2 * c_prod)
            Vf_well = activ_param["Vf_well"]          # Verdünnung im Well
            Vf_prod = activ_param["Vf_prod"]          # Verdünnung der Proteinlösung
            c_prod = activ_param["c_prod"]            # Proteinkonzentration [mg/L]
            
            # m1 = slope [A340/s], m2 = slope_cal [A340/μM]
            activity_U_per_mg = (abs(slope) * 60 * Vf_well * Vf_prod) / (slope_cal * c_prod)
            
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

def get_rates_and_concentrations(reaction_data_dict, slope, reaction_params_dict, verbose=True):


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
                result = calculate_activity(
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
    df.to_csv("processed_reaction_data.csv", index=False)
    return df

def create_reaction_rates_dict(processed_data):
    """
    Erstellt ein rates_dict aus verarbeiteten Reaktionsdaten.
    
    Args:
        processed_data: Rückgabe von get_rates_and_concentrations
        
    Returns:
        Dict: {reaction_name + "_rates": activities}
    """
    rates_dict = {}
    activities = processed_data.get("activities", {})
    
    for reaction_name, activity_data in activities.items():
        rates_dict[f"{reaction_name}_rates"] = activity_data
    
    return rates_dict

def create_concentrations_dict(processed_data, constants_dict=None):
    """
    Erstellt ein concentrations_dict aus verarbeiteten Reaktionsdaten.
    
    Args:
        processed_data: Rückgabe von get_rates_and_concentrations
        constants_dict: Dict mit konstanten Konzentrationen {key: value}
        
    Returns:
        Dict: {reaction_name + "_conc": concentrations, ...constants}
    """
    concentrations_dict = {}
    concentrations = processed_data.get("concentrations", {})
    
    # Füge variable Konzentrationen hinzu
    for reaction_name, conc_data in concentrations.items():
        concentrations_dict[f"{reaction_name}_conc"] = conc_data
    
    # Füge konstante Konzentrationen hinzu
    if constants_dict:
        concentrations_dict.update(constants_dict)
    
    return concentrations_dict


def make_fitting_data(model_info, data_info, df, verbose=True):
    """ 
    Extrahiere aus dataframe die x und y Werte für das Fitting.
    """
    dim_x = data_info["x_dimension"]
    dim_y = data_info["y_dimension"]

    data_x = []
    data_y = []

    a = df.apply(lambda row: (row["reaction"],row["c1"],row["c2"],row["c3"]), axis=1).to_numpy()