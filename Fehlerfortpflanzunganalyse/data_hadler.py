import os 
import pandas as pd
import numpy as np
from scipy.stats import linregress

def is_linear(x, y, threshold=0.80):
    """Prüft ob Daten linear sind basierend auf R² (gelockerte Kriterien für mehr Datenpunkte)"""
    if len(x) < 3 or len(y) < 3:
        return False
    try:
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        return abs(r_value) > threshold
    except Exception:
        return False

def add_noise(data, noise_level=0.1):
    """
    Fügt den Daten zufälliges Rauschen hinzu.
    """
    noise = np.random.normal(0, noise_level, size=data.shape)
    return data + noise

def combine_full_reaction_system_data(
    r1_nad_valid_conc, r1_nad_rates, r1_pd_valid_conc, r1_pd_rates,
    r2_pd_valid_conc, r2_pd_rates, r2_nadh_valid_conc, r2_nadh_rates,
    r2_hp_valid_conc, r2_hp_rates, r3_nad_valid_conc, r3_nad_rates,
    r3_lactol_valid_conc, r3_lactol_rates
):
    r1_nad_constant = 500   # mM (PD konstant bei NAD-Variation)
    r1_pd_constant = 5.0    # mM (NAD konstant bei PD-Variation)
    r2_nadh_constant = 0.6   # mM (aus "0.6 mM NADH" Dateiname)
    r2_lactol_constant = 300.0  # mM (aus "300 mM HP" Dateiname)
    r3_nad_constant = 5.0     # mM (aus "5 mM NAD" Dateiname)
    r3_lactol_constant = 500.0  # mM (aus "500 mM Lactol" Dateiname)

    all_s1_values = []
    all_s2_values = []
    all_inhibitor_values = []
    all_activities = []
    all_reaction_ids = []

    # Reaktion 1 Daten hinzufügen
    all_s1_values.extend(r1_nad_valid_conc)
    all_s2_values.extend([r1_nad_constant] * len(r1_nad_valid_conc))
    all_inhibitor_values.extend([0.0] * len(r1_nad_valid_conc))
    all_activities.extend(r1_nad_rates)
    all_reaction_ids.extend([1] * len(r1_nad_valid_conc))

    all_s1_values.extend([r1_pd_constant] * len(r1_pd_valid_conc))
    all_s2_values.extend(r1_pd_valid_conc)
    all_inhibitor_values.extend([0.0] * len(r1_pd_valid_conc))
    all_activities.extend(r1_pd_rates)
    all_reaction_ids.extend([1] * len(r1_pd_valid_conc))

    # Reaktion 2 Daten hinzufügen
    all_s1_values.extend([r2_lactol_constant] * len(r2_pd_valid_conc))
    all_s2_values.extend([r2_nadh_constant] * len(r2_pd_valid_conc))
    all_inhibitor_values.extend(r2_pd_valid_conc)
    all_activities.extend(r2_pd_rates)
    all_reaction_ids.extend([2] * len(r2_pd_valid_conc))

    all_s1_values.extend([r2_lactol_constant] * len(r2_nadh_valid_conc))
    all_s2_values.extend(r2_nadh_valid_conc)
    all_inhibitor_values.extend([1.0] * len(r2_nadh_valid_conc))
    all_activities.extend(r2_nadh_rates)
    all_reaction_ids.extend([2] * len(r2_nadh_valid_conc))

    all_s1_values.extend(r2_hp_valid_conc)
    all_s2_values.extend([r2_nadh_constant] * len(r2_hp_valid_conc))
    all_inhibitor_values.extend([1.0] * len(r2_hp_valid_conc))
    all_activities.extend(r2_hp_rates)
    all_reaction_ids.extend([2] * len(r2_hp_valid_conc))

    # Reaktion 3 Daten hinzufügen
    all_s1_values.extend([r3_lactol_constant] * len(r3_nad_valid_conc))
    all_s2_values.extend(r3_nad_valid_conc)
    all_inhibitor_values.extend([0.0] * len(r3_nad_valid_conc))
    all_activities.extend(r3_nad_rates)
    all_reaction_ids.extend([3] * len(r3_nad_valid_conc))

    all_s1_values.extend(r3_lactol_valid_conc)
    all_s2_values.extend([r3_nad_constant] * len(r3_lactol_valid_conc))
    all_inhibitor_values.extend([0.0] * len(r3_lactol_valid_conc))
    all_activities.extend(r3_lactol_rates)
    all_reaction_ids.extend([3] * len(r3_lactol_valid_conc))

    S1_combined = np.array(all_s1_values)
    S2_combined = np.array(all_s2_values)
    Inhibitor_combined = np.array(all_inhibitor_values)
    activities_combined = np.array(all_activities)
    reaction_ids = np.array(all_reaction_ids)

    # Rückgabe als experiment_data-Dict für maximale Kompatibilität
    experiment_data = {
        "S1": S1_combined,
        "S2": S2_combined,
        "Inhibitor": Inhibitor_combined,
        "activities": activities_combined,
        "reaction_ids": reaction_ids
    }
    return experiment_data

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

def calculate_calibration(data):
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
        print("Die Daten sind nicht linear. Bitte überprüfen Sie die Kalibrierung.")
        return None

    slope_cal, intercept_cal, r_value_cal, _, _ = linregress(x, y)
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
            
            if verbose and r_squared < 0.90:
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

def get_rates_and_concentrations(reaction_data, slope, activity_params):

    #todo next erlaube dictionary als reaction_data
 

    concentrations = get_concentrations_from_csv(reaction_data)
    absorption_data = get_absorption_data(reaction_data)
    time_points = get_time_points(reaction_data)


    activities, concentrations = calculate_activity(concentrations, absorption_data, time_points, slope, activity_params)

    return  [activities, concentrations]

def make_fitting_data(model_info, concentrations, rates):
    """
    Erstellt die Datenstruktur für das Fitting basierend auf Modell-Informationen.
    Automatische Behandlung von variablen und konstanten Substratkonzentrationen.
    
    model_info: Dict mit Modell-Informationen (z.B. name, function, param_names, etc.)
    concentrations: Dict mit Substratkonzentrationen
    rates: Dict mit gemessenen Raten
    """
    
    # Extrahiere und kombiniere alle Aktivitäten
    activities_list = [rates[key] for key in rates if key.endswith('_rates') and rates[key] is not None]
    activities = np.concatenate(activities_list) if activities_list else np.array([])
    
    print(f"Debug - Total activities: {len(activities)}")
    
    # Allgemeine Behandlung für Multi-Substrat-Modelle
    substrate_keys = model_info['substrate_keys']
    n_substrates = len(substrate_keys)
    
    if n_substrates > 1:
        print(f"Debug - Multi-Substrat-Modell mit {n_substrates} Substraten")
        
        # Sammle alle variablen Konzentrationen und deren konstante Partner
        variable_data = []
        constant_values = []
        for substrate_key in substrate_keys:
            var_conc_key = substrate_key  
            
            if substrate_key.endswith('_conc'):
                const_key = substrate_key.replace('_conc', '_const')
            else:
                const_key = f"{substrate_key}_const"
            
            print(f"Debug - Suche nach: var='{var_conc_key}', const='{const_key}'")
            
            if var_conc_key in concentrations:
                variable_data.append((substrate_key, concentrations[var_conc_key]))
                
                # Suche nach entsprechendem konstanten Wert
                if const_key in concentrations:
                    constant_values.append((substrate_key, concentrations[const_key]))
                else:
                    print(f"Warning: Konstanter Wert für {substrate_key} nicht gefunden ({const_key})")
                    constant_values.append((substrate_key, 1.0))  # Default-Wert
        
        print(f"Debug - Variable Daten: {len(variable_data)}")
        print(f"Debug - Konstante Werte: {len(constant_values)}")
        
        # Erstelle kombinierte Substrat-Arrays
        substrate_arrays = [[] for _ in range(n_substrates)]
        
        # Für jedes variable Substrat: erstelle Experimente
        for var_idx, (var_substrate, var_concentrations) in enumerate(variable_data):
            var_concentrations = np.array(var_concentrations)
            n_var_points = len(var_concentrations)
            
            print(f"Debug - {var_substrate} variiert: {n_var_points} Punkte")
            
            # Für jedes Substrat: füge entweder variable oder konstante Werte hinzu
            for substrate_idx, substrate_key in enumerate(substrate_keys):
                if substrate_key == var_substrate:
                    # Das variable Substrat
                    substrate_arrays[substrate_idx].extend(var_concentrations)
                else:
                    # Alle anderen Substrate sind konstant
                    const_value = next((val for key, val in constant_values if key == substrate_key), 1.0)
                    substrate_arrays[substrate_idx].extend([const_value] * n_var_points)
        
        # Konvertiere zu numpy arrays
        substrate_data = [np.array(arr) for arr in substrate_arrays]
        
        # Debug-Ausgabe
        for i, (substrate_key, arr) in enumerate(zip(substrate_keys, substrate_data)):
            print(f"Debug - {substrate_key}: shape {arr.shape}, first 5 values: {arr[:5]}")
            print(f"Debug - {substrate_key}: unique values: {np.unique(arr)[:10]}")  # Zeige bis zu 10 einzigartige Werte
        
        return substrate_data, activities
    
    # Fallback für Ein-Substrat-Modelle
    else:
        substrate_data = []
        
        for substrate_key in substrate_keys:
            if substrate_key in concentrations:
                substrate_values = concentrations[substrate_key]
                
                if isinstance(substrate_values, (int, float)):
                    substrate_array = np.full(len(activities), substrate_values)
                else:
                    substrate_array = np.array(substrate_values)
                    
                    if len(activities) > len(substrate_array):
                        n_repeats = len(activities) // len(substrate_array)
                        remainder = len(activities) % len(substrate_array)
                        
                        substrate_array = np.tile(substrate_array, n_repeats)
                        if remainder > 0:
                            substrate_array = np.concatenate([substrate_array, substrate_array[:remainder]])
                    elif len(activities) < len(substrate_array):
                        substrate_array = substrate_array[:len(activities)]
                
                substrate_data.append(substrate_array)
                print(f"Debug - {substrate_key}: shape {substrate_array.shape}, first 3 values: {substrate_array[:3]}")
            else:
                print(f"Warning: Key '{substrate_key}' not found in concentrations")
                substrate_data.append(np.zeros(len(activities)))

        return substrate_data, activities
