from scipy.stats import linregress
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def michaelis_menten(S, Vmax, Km):
    """Michaelis-Menten Gleichung: v = (Vmax * S) / (Km + S)"""
    return (Vmax * S) / (Km + S)

def two_substrat_michaelis_menten(substrate_data, Vmax, Km1, Km2):
    """Zwei-Substrat Michaelis-Menten Gleichung"""
    
    S1_values, S2_values = substrate_data
    
    # Sicherstellen, dass es numpy arrays sind
    S1_values = np.asarray(S1_values)
    S2_values = np.asarray(S2_values)
    
    # Element-weise Berechnung für alle Datenpunkte
    rates = (Vmax * S1_values * S2_values) / ((Km1 + S1_values) * (Km2 + S2_values))
    
    return rates


# Verfügbare Modelle definieren
AVAILABLE_MODELS = {
    'michaelis_menten': {
        'function': michaelis_menten,
        'param_names': ['Vmax', 'Km'],
        'param_units': ['U', 'mM'],
        'initial_guess_func': lambda rates, concs: [max(rates), np.median(concs)],
        'description': 'Michaelis-Menten Kinetik'
    }
    ,  'two_substrat_michaelis_menten': {
        'function': two_substrat_michaelis_menten,
        'param_names': ['Vmax', 'Km1', 'Km2'],
        'param_units': ['U/mg', 'mM', 'mM'],
        'initial_guess_func': lambda rates, concs: [max(rates), 1.0, 1.0],
        'description': 'Zwei-Substrat Michaelis-Menten Kinetik',
        'special_data_format': True  # Flag für spezielle Datenbehandlung
    }
}

def is_linear(x, y, threshold=0.95):
    """Prüft ob Daten linear sind basierend auf R²"""
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    return abs(r_value) > threshold

def get_concentrations(kinetik_data):
    concentrations_with_duplicas = kinetik_data.iloc[1:, 1].dropna().values
    concentrations = [float(x) for x in concentrations_with_duplicas[::2]]  # Jedes zweite Element ab Index 0
    return np.array(concentrations)

def get_adsoption_data(kinetik_data):
    """Extrahiert die Absorptionsdaten aus den Kinetikdaten"""
    absorption_data_with_duplicas = kinetik_data.iloc[1:, 2:].values
    absorption_data = []
    for i in range(0, len(absorption_data_with_duplicas), 2):
        abs1 = absorption_data_with_duplicas[i] 
        abs2 = absorption_data_with_duplicas[i + 1] if i + 1 < len(absorption_data_with_duplicas) else None
        if abs2 is not None:
            mittelwert_abs = np.nanmean([abs1, abs2], axis=0)
        else:
            mittelwert_abs = abs1
        absorption_data.append(mittelwert_abs)
    return np.array(absorption_data)

def get_time_points(kinetik_data):
    """Extrahiert die Zeitpunkte aus den Kinetikdaten"""
    time_points = kinetik_data.iloc[0, 2:].values  # Erste Zeile (Index 0), ab Spalte 2
    time_points = pd.to_numeric(time_points, errors='coerce')  # Zu float konvertieren
    return time_points[~np.isnan(time_points)]  # Nur gültige Werte zurückgeben

def berechne_kalibrierung(nadh_kalibrierung):
    """Berechnet die Kalibrierung basierend auf NADH-Konzentrationen und Extinktionen"""
    x = nadh_kalibrierung.NADH
    y = nadh_kalibrierung.Mittelwert

    if not is_linear(x, y):
        print("Die Daten sind nicht linear. Bitte überprüfen Sie die Kalibrierung.")
        return None

    slope_cal, intercept_cal, r_value_cal, _, _ = linregress(x, y)
    print(f"Kalibrierung: Steigung = {slope_cal:.2f}, R = {r_value_cal**2:.4f}")
    
    return slope_cal, intercept_cal, r_value_cal

def berechne_aktivitaet(concentrations, absorption_data, time_points, slope_cal, activ_param, verbose=True):

    initial_rates = []
    valid_concentrations = []

    for i, conc in enumerate(concentrations):
        # Konzentration zu float konvertieren
        try:
            conc_float = float(conc)
        except (ValueError, TypeError):
            if verbose:
                print(f"Well {i+1}: Ungültige Konzentration '{conc}' - übersprungen")
            continue
            
        if pd.notna(conc_float) and i < len(absorption_data):
            # Absorptionswerte für dieses Well
            absorbance = absorption_data[i, :]
            
            # Zu numerischen Werten konvertieren
            absorbance = pd.to_numeric(absorbance, errors='coerce')
            
            # Nur gültige Werte verwenden
            valid_indices = ~np.isnan(absorbance)
            if np.sum(valid_indices) < 3:
                if verbose:
                    print(f"Well {i+1}: Nicht genug gültige Absorptionswerte")
                continue
                
            time_valid = time_points[valid_indices]
            abs_valid = absorbance[valid_indices]
            
            # Zusätzlich prüfen, ob time_points auch gültig sind
            time_valid_mask = ~np.isnan(time_valid)
            if np.sum(time_valid_mask) < 3:
                if verbose:
                    print(f"Well {i+1}: Nicht genug gültige Zeitpunkte")
                continue
                
            time_final = time_valid[time_valid_mask]
            abs_final = abs_valid[time_valid_mask]
            
            # Linearität prüfen
            if not is_linear(time_final, abs_final):
                if verbose:
                    print(f"Well {i+1} (Konz: {conc_float} mM) ist nicht linear.")
                continue
            
            # Lineare Regression für initiale Rate
            # Linearität prüfen
            if not is_linear(time_final, abs_final):
                if verbose:
                    print(f"Well {i+1} (Konz: {conc_float} mM) ist nicht linear.")
                continue
            
            # Lineare Regression für initiale Rate
            slope, intercept, r_value, p_value, std_err = linregress(time_final, abs_final)
            
            # Umrechnung nach Ihrer Formel: A[U/mg] = (m1 * 60 * Vf_well * Vf_prod) / (m2 * c_prod)
            Vf_well = activ_param["Vf_well"]          # Verdünnung im Well
            Vf_prod = activ_param["Vf_prod"]          # Verdünnung der Proteinlösung
            c_prod = activ_param["c_prod"]            # Proteinkonzentration [mg/L]
            
            # m1 = slope [A340/s], m2 = slope_cal [A340/μM]
            activity_U_per_mg = (abs(slope) * 60 * Vf_well * Vf_prod) / (slope_cal * c_prod)
            
            # Falls Sie U (ohne /mg) brauchen, multiplizieren Sie mit Proteinmenge:
            # protein_mass_mg = c_prod * volume_used_L * 1000  # je nach Ansatz
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
    if len(initial_rates) < 3:
        if verbose:
            print("Nicht genug gültige initiale Raten für die Parameterabschätzung.")
        return None

    return initial_rates, valid_concentrations

def schaetze_parameter(param, model_name='michaelis_menten'):
    
    # Modell validieren
    if model_name not in AVAILABLE_MODELS:
        print(f"Fehler: Modell '{model_name}' nicht verfügbar!")
        print(f"Verfügbare Modelle: {list(AVAILABLE_MODELS.keys())}")
        return None
    
    model_info = AVAILABLE_MODELS[model_name]
    model_func = model_info['function']
    param_names = model_info['param_names']
    param_units = model_info['param_units']
    
    print(f"Verwende Modell: {model_info['description']}")
    
    # NADH Kalibrierung laden (für alle Modelle gleich)
    nadh_kalibrierung = pd.read_excel(r"C:\Users\berger\Documents\Projekts\MichealisMenten\Dortmund\Fehlerfortpflanzunganalyse\Daten\Rohdaten\Plate_Reader\Kalibriergeraden\NADH_Kalibriergerade.xlsx")

    # Kalibrierungssteigung berechnen
    slope_cal, intercept_cal, r_value_cal = berechne_kalibrierung(nadh_kalibrierung)
    if slope_cal is None:
        return None

    # UNTERSCHIEDLICHE DATENBEHANDLUNG JE NACH MODELL:
    if model_name == 'two_substrat_michaelis_menten':
        # ZWEI-SUBSTRAT: Zwei separate Experimente laden
        print("\n--- Experiment 1: S1 variabel, S2 konstant ---")
        try:
            #todo
            r1_s1_var = pd.read_excel(r"C:\Users\berger\Documents\Projekts\MichealisMenten\Dortmund\Fehlerfortpflanzunganalyse\Daten\Rohdaten\Plate_Reader\Kinetik-Messungen\s1_variable\s1_var.xlsx")
            concentrations_s1 = get_concentrations(r1_s1_var)
            time_points_s1 = get_time_points(r1_s1_var)
            absorption_data_s1 = get_adsoption_data(r1_s1_var)
            
            initial_rates_s1, valid_concentrations_s1 = berechne_aktivitaet(
                concentrations_s1, absorption_data_s1, time_points_s1, slope_cal, param, verbose=True
            )
            #todo
            S2_constant = 2.0 
            
        except FileNotFoundError:
            print("WARNUNG: S1-variable Datei nicht gefunden!")
           
        
        print("\n--- Experiment 2: S2 variabel, S1 konstant ---")
        try:
            #todo
            r2_s2_var = pd.read_excel(r"C:\Users\berger\Documents\Projekts\MichealisMenten\Dortmund\Fehlerfortpflanzunganalyse\Daten\Rohdaten\Plate_Reader\Kinetik-Messungen\s2_variable\s2_var.xlsx")
            concentrations_s2 = get_concentrations(r2_s2_var)
            time_points_s2 = get_time_points(r2_s2_var)
            absorption_data_s2 = get_adsoption_data(r2_s2_var)
            
            initial_rates_s2, valid_concentrations_s2 = berechne_aktivitaet(
                concentrations_s2, absorption_data_s2, time_points_s2, slope_cal, param, verbose=True
            )
            #todo
            S1_constant = 1.5  #todo! gucken welcher wert das hier ist
            
        except FileNotFoundError:
            print("WARNUNG: S2-variable Datei nicht gefunden! ")
            
        
        if initial_rates_s1 is None or initial_rates_s2 is None:
            print("Fehler beim Verarbeiten der Zwei-Substrat-Kinetikdaten!")
            return None
        
        # Kombinierte Datenstruktur für Zwei-Substrat erstellen
        S1_combined = np.concatenate([
            valid_concentrations_s1,                          # Variable S1-Werte
            np.full(len(valid_concentrations_s2), S1_constant) # Konstante S1-Werte
        ])
        
        S2_combined = np.concatenate([
            np.full(len(valid_concentrations_s1), S2_constant), # Konstante S2-Werte
            valid_concentrations_s2                            # Variable S2-Werte
        ])
        
        activities_combined = np.concatenate([initial_rates_s1, initial_rates_s2])
        
        # Für Zwei-Substrat: verwende kombinierte Datenstruktur
        x_data = (S1_combined, S2_combined)  # Tupel für two_substrat_michaelis_menten
        y_data = activities_combined
        
        print(f"Zwei-Substrat Daten:")
        print(f"- Experiment 1 (S1 var, S2={S2_constant} mM): {len(valid_concentrations_s1)} Punkte")
        print(f"- Experiment 2 (S2 var, S1={S1_constant} mM): {len(valid_concentrations_s2)} Punkte")
        print(f"- Gesamt: {len(activities_combined)} Datenpunkte")
        
    else:
        # EIN-SUBSTRAT: Ihr bestehender Code (unverändert!)
        r1 = pd.read_excel(r"C:\Users\berger\Documents\Projekts\MichealisMenten\Dortmund\Fehlerfortpflanzunganalyse\Daten\Rohdaten\Plate_Reader\Kinetik-Messungen\r1\r1_NAD_PD_mod.xlsx")
        
        concentrations = get_concentrations(r1)
        time_points = get_time_points(r1)
        absorption_data = get_adsoption_data(r1)

        print(f"Ein-Substrat Daten:")
        print(f"Anzahl Wells: {len(concentrations)}")
        print(f"Zeitpunkte: {len(time_points)}")
        print(f"Konzentrationen: {concentrations}")

        initial_rates, valid_concentrations = berechne_aktivitaet(concentrations, absorption_data, time_points, slope_cal, param, verbose=True)
        
        # Für Ein-Substrat: verwende normale Datenstruktur
        x_data = valid_concentrations  # Array für michaelis_menten
        y_data = initial_rates
    
    # Parameter fitten
    try:
        # Initial guess basierend auf Modell
        if model_name == 'two_substrat_michaelis_menten':
            # Für Zwei-Substrat: spezielle initial guess
            p0 = [max(y_data), 1.0, 1.0]  # [Vmax, Km1, Km2]
        else:
            # Für Ein-Substrat: normale initial guess
            p0 = model_info['initial_guess_func'](y_data, x_data)
        
        print(f"Initial guess: {p0}")
        
        # curve_fit aufrufen
        params, covariance = curve_fit(model_func, x_data, y_data, p0=p0, maxfev=5000)
        
        # Fehler berechnen
        param_errors = np.sqrt(np.diag(covariance))
        
        # R² berechnen
        y_pred = model_func(x_data, *params)
        ss_res = np.sum((y_data - y_pred) ** 2)
        ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        print(f"\n=== {model_info['description'].upper()} PARAMETER ===")
        for i, (name, unit, value, error) in enumerate(zip(param_names, param_units, params, param_errors)):
            print(f"{name}: {value:.4f} ± {error:.4f} {unit}")
        print(f"R²: {r_squared:.4f}")
        
        # Zusätzliche Ausgabe für Zwei-Substrat
        if model_name == 'two_substrat_michaelis_menten':
            print(f"\nExperimentelle Bedingungen:")
            print(f"- S1 konstant: {S1_constant:.2f} mM")
            print(f"- S2 konstant: {S2_constant:.2f} mM")
        
        # Ergebnis-Dictionary erstellen
        result = {
            'R_squared': r_squared,
            'concentrations': x_data if model_name != 'two_substrat_michaelis_menten' else S1_combined,
            'activities': y_data,
            'model_name': model_name,
            'model_description': model_info['description']
        }
        
        # Parameter dynamisch hinzufügen
        for i, (name, unit, value, error) in enumerate(zip(param_names, param_units, params, param_errors)):
            result[name] = value
            result[f"{name}_error"] = error
            result[f"{name}_unit"] = unit
        
        # Zusätzliche Informationen für Zwei-Substrat
        if model_name == 'two_substrat_michaelis_menten':
            result['S1_constant'] = S1_constant
            result['S2_constant'] = S2_constant
            result['S1_combined'] = S1_combined
            result['S2_combined'] = S2_combined
        
        return result
        
    except Exception as e:
        print(f"Fehler beim Fitten der {model_info['description']} Parameter: {e}")
        return None
        
def add_noise(data, noise_level=0.05):
    """
    Fügt Gauß'sches Rauschen zu den Daten hinzu
    
    Parameters:
    data: numpy array - die ursprünglichen Daten
    noise_level: float - Rauschpegel als Bruchteil des Signals (0.05 = 5%)
    
    Returns:
    numpy array - verrauschte Daten
    """
    # Standardabweichung basierend auf dem Signal
    std_dev = np.abs(data) * noise_level
    
    # Gauß'sches Rauschen hinzufügen
    noise = np.random.normal(0, std_dev)
    
    return data + noise

def monte_carlo_simulation(param, model_name='michaelis_menten', n_iterations=1000, noise_level_calibration=0.03, noise_level_kinetics=0.02):
    """
    Monte Carlo Simulation für beliebige Modelle
    """
    
    print("=== MONTE CARLO SIMULATION ===")
    print(f"Iterationen: {n_iterations}")
    print(f"Rauschen Kalibrierung: {noise_level_calibration*100:.1f}%")
    print(f"Rauschen Kinetik: {noise_level_kinetics*100:.1f}%")
    print("="*50)
    
    # Listen für Ergebnisse
    param_results = {}  # Dictionary für alle Parameter
    r_squared_results = []
    
    # Originaldaten einmal laden
    try:
        nadh_kalibrierung = pd.read_excel(r"C:\Users\berger\Documents\Projekts\MichealisMenten\Dortmund\Fehlerfortpflanzunganalyse\Daten\Rohdaten\Plate_Reader\Kalibriergeraden\NADH_Kalibriergerade.xlsx")
        r1 = pd.read_excel(r"C:\Users\berger\Documents\Projekts\MichealisMenten\Dortmund\Fehlerfortpflanzunganalyse\Daten\Rohdaten\Plate_Reader\Kinetik-Messungen\r1\r1_NAD_PD_mod.xlsx")
    except Exception as e:
        print(f"Fehler beim Laden der Dateien: {e}")
        return None
    
    try:
        # Kalibrierungsdaten
        x_original = nadh_kalibrierung.NADH.values
        y_original = nadh_kalibrierung.Mittelwert.values
        
        # Konzentrationen 
        concentrations_original = get_concentrations(r1)
        
        # Zeitpunkte
        time_points_original = get_time_points(r1)
        
        # Absorptionsdaten 
        absorption_data_original = get_adsoption_data(r1)
        
        print("Originaldaten geladen:")
        print(f"- Kalibrierung: {len(x_original)} Punkte")
        print(f"- Konzentrationen: {len(concentrations_original)} Wells")
        print(f"- Zeitpunkte: {len(time_points_original)} Messungen")
        print(f"- Absorption: {absorption_data_original.shape}")
        
        # Validierung
        if len(concentrations_original) == 0:
            print("Keine gültigen Konzentrationen gefunden!")
            return None
            
        if len(time_points_original) == 0:
            print("Keine gültigen Zeitpunkte gefunden!")
            return None
        
    except Exception as e:
        print(f"Fehler beim Extrahieren der Originaldaten: {e}")
        import traceback
        traceback.print_exc()  # Detaillierte Fehlermeldung
        return None
    
    successful_iterations = 0
    failed_reasons = {"linearität": 0, "daten": 0, "fitting": 0, "andere": 0}
    
    for iteration in range(n_iterations):
        try:
            # 1. Kalibrierungsdaten verrauschen
            x_noisy = add_noise(x_original, noise_level_calibration)
            y_noisy = add_noise(y_original, noise_level_calibration)
            
            # Linearität der verrauschten Kalibrierung prüfen (weniger streng)
            if not is_linear(x_noisy, y_noisy, threshold=0.85):
                failed_reasons["linearität"] += 1
                continue
            
            # 2. Kinetikdaten verrauschen
            concentrations_noisy = add_noise(concentrations_original, noise_level_calibration)
            time_points_noisy = add_noise(time_points_original, noise_level_kinetics)
            
            # Absorption verrauschen
            absorption_data_noisy = add_noise(absorption_data_original.astype(float), noise_level_kinetics)
            
            # 3. Parameter berechnen mit verrauschten Daten
            result = schaetze_parameter_noisy(x_noisy, y_noisy, concentrations_noisy, 
                                            time_points_noisy, absorption_data_noisy, param, model_name)
            
            if result is not None:
                # Modell-spezifische Validierung
                model_info = AVAILABLE_MODELS[model_name]
                param_names = model_info['param_names']
                
                # Basis-Validierung: alle Parameter müssen positiv und realistisch sein
                all_params_valid = True
                for param_name in param_names:
                    param_value = result.get(param_name, 0)
                    if param_value <= 0 or param_value > 10000:  # Erweiterte Plausibilitätsprüfung
                        all_params_valid = False
                        break
                
                if all_params_valid and result['R_squared'] > 0.5:
                    # Sammle alle Parameter für die Statistik
                    for param_name in param_names:
                        if param_name not in param_results:
                            param_results[param_name] = []
                        param_results[param_name].append(result[param_name])
                    
                    r_squared_results.append(result['R_squared'])
                    successful_iterations += 1
                else:
                    failed_reasons["daten"] += 1
            else:
                failed_reasons["fitting"] += 1
                
            if (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}/{n_iterations} - Erfolgreiche: {successful_iterations}")
        
        except Exception as e:
            failed_reasons["andere"] += 1
            if iteration < 5:  # Erste 5 Fehler ausgeben
                print(f"Fehler in Iteration {iteration}: {e}")
            continue
    
    print(f"\nZusammenfassung der Fehlschläge:")
    print(f"- Linearitätsprobleme: {failed_reasons['linearität']}")
    print(f"- Datenprobleme: {failed_reasons['daten']}")
    print(f"- Fitting-Probleme: {failed_reasons['fitting']}")
    print(f"- Andere Fehler: {failed_reasons['andere']}")
    
    if successful_iterations < 10:
        print(f"Zu wenige erfolgreiche Iterationen: {successful_iterations}")
        return None
    
    # Statistische Auswertung
    r_squared_results = np.array(r_squared_results)
    
    # Modell-Info für Parameter
    model_info = AVAILABLE_MODELS[model_name]
    param_names = model_info['param_names']
    
    # Alle Parameter zu numpy arrays konvertieren
    param_arrays = {}
    for param_name in param_names:
        if param_name in param_results:
            param_arrays[param_name] = np.array(param_results[param_name])
    
    # Kovarianzmatrix und Korrelation berechnen (nur wenn mindestens 2 Parameter)
    if len(param_arrays) >= 2:
        # Matrix aus allen Parametern erstellen
        parameter_matrix = np.column_stack([param_arrays[name] for name in param_names if name in param_arrays])
        covariance_matrix = np.cov(parameter_matrix, rowvar=False)
        correlation_matrix = np.corrcoef(parameter_matrix, rowvar=False)
    else:
        covariance_matrix = None
        correlation_matrix = None
    
    # Ergebnis-Dictionary erstellen
    results = {
        'n_successful': successful_iterations,
        'model_name': model_name,
        'param_names': param_names,
        'R_squared_mean': np.mean(r_squared_results),
        'R_squared_std': np.std(r_squared_results),
        'r_squared_values': r_squared_results
    }
    
    # Parameter-spezifische Statistiken hinzufügen
    for param_name in param_names:
        if param_name in param_arrays:
            param_data = param_arrays[param_name]
            results[f'{param_name}_mean'] = np.mean(param_data)
            results[f'{param_name}_std'] = np.std(param_data)
            results[f'{param_name}_ci_lower'] = np.percentile(param_data, 2.5)
            results[f'{param_name}_ci_upper'] = np.percentile(param_data, 97.5)
            results[f'{param_name}_values'] = param_data
    
    # Kovarianz/Korrelation hinzufügen (falls verfügbar)
    if covariance_matrix is not None:
        results['covariance_matrix'] = covariance_matrix
        results['correlation_matrix'] = correlation_matrix
    
    print(f"\n=== MONTE CARLO ERGEBNISSE ===")
    print(f"Erfolgreiche Iterationen: {successful_iterations}/{n_iterations}")
    
    # Parameter-spezifische Ausgabe
    for param_name in param_names:
        if f'{param_name}_mean' in results:
            mean_val = results[f'{param_name}_mean']
            std_val = results[f'{param_name}_std']
            ci_lower = results[f'{param_name}_ci_lower']
            ci_upper = results[f'{param_name}_ci_upper']
            unit = model_info['param_units'][param_names.index(param_name)]
            
            print(f"{param_name}: {mean_val:.4f} ± {std_val:.4f} {unit}")
            print(f"{param_name} 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    
    print(f"R²: {results['R_squared_mean']:.4f} ± {results['R_squared_std']:.4f}")
    
    # Kovarianz und Korrelation ausgeben (falls verfügbar)
    if 'covariance_matrix' in results and len(param_names) >= 2:
        print(f"\n=== KOVARIANZ UND KORRELATION ===")
        
        # Für 2 Parameter: spezielle Ausgabe
        if len(param_names) == 2:
            param1, param2 = param_names[0], param_names[1]
            covariance = results['covariance_matrix'][0, 1]
            correlation = results['correlation_matrix'][0, 1]
            
            print(f"{param1}-{param2} Kovarianz: {covariance:.6f}")
            print(f"{param1}-{param2} Korrelation: {correlation:.4f}")
        
        # Vollständige Matrizen ausgeben
        print(f"\nKovarianzmatrix:")
        header = "        " + "".join([f"{name:>12}" for name in param_names])
        print(header)
        for i, param_name in enumerate(param_names):
            row = f"{param_name:<8}"
            for j in range(len(param_names)):
                row += f"{results['covariance_matrix'][i,j]:12.6f}"
            print(row)
        
        print(f"\nKorrelationsmatrix:")
        print(header)
        for i, param_name in enumerate(param_names):
            row = f"{param_name:<8}"
            for j in range(len(param_names)):
                row += f"{results['correlation_matrix'][i,j]:12.4f}"
            print(row)
    
    # Histogramme erstellen
    create_histograms(results)
    
    return results

def schaetze_parameter_noisy(x_cal, y_cal, concentrations, time_points, absorption_data, param, model_name='michaelis_menten'):
    """
    Parameter schätzen mit verrauschten Daten für beliebige Modelle
    """
    try:
        # Modell-Info holen
        if model_name not in AVAILABLE_MODELS:
            return None
            
        model_info = AVAILABLE_MODELS[model_name]
        model_func = model_info['function']
        param_names = model_info['param_names']
        
        # Kalibrierungssteigung berechnen
        slope_cal, intercept_cal, r_value_cal, _, _ = linregress(x_cal, y_cal)
        
        # WICHTIGER HINWEIS: Für Zwei-Substrat müsste hier eine erweiterte Datenbehandlung erfolgen
        # Da Monte Carlo aber normalerweise mit Ein-Substrat-Daten läuft, verwenden wir die einfache Version
        initial_rates, valid_concentrations = berechne_aktivitaet(concentrations, absorption_data, time_points, slope_cal, param, verbose=False)

        if initial_rates is None or valid_concentrations is None:
            return None

        try:
            # Initial guess basierend auf Modell
            if model_name == 'two_substrat_michaelis_menten':
                # Für Zwei-Substrat: vereinfachte Behandlung in Monte Carlo
                # (Normalerweise würde man hier die vollen Zwei-Substrat-Daten verwenden)
                p0 = [max(initial_rates), 1.0, 1.0]  # [Vmax, Km1, Km2]
                # Verwende vereinfachte Datenstruktur für Monte Carlo
                x_data = (valid_concentrations, np.full_like(valid_concentrations, 2.0))  # S2 konstant = 2.0
                y_data = initial_rates
            else:
                p0 = model_info['initial_guess_func'](initial_rates, valid_concentrations)
                x_data = valid_concentrations
                y_data = initial_rates
            
            params, covariance = curve_fit(model_func, x_data, y_data, p0=p0, maxfev=5000)
            
            # Plausibilitätsprüfung der Parameter
            for param_val in params:
                if param_val <= 0 or param_val > 10000:
                    return None
            
            param_errors = np.sqrt(np.diag(covariance))
            
            y_pred = model_func(x_data, *params)
            ss_res = np.sum((y_data - y_pred) ** 2)
            ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            
            # Ergebnis-Dictionary erstellen
            result = {'R_squared': r_squared, 'model_name': model_name}
            
            # Parameter dynamisch hinzufügen
            for i, (name, value, error) in enumerate(zip(param_names, params, param_errors)):
                result[name] = value
                result[f"{name}_error"] = error
            
            return result
            
        except Exception:
            return None
            
    except Exception:
        return None

def create_histograms(results):
    """Erstellt Histogramme der Monte Carlo Ergebnisse für beliebige Modelle"""
    
    # Modell-Info holen
    model_name = results['model_name']
    param_names = results['param_names']
    model_info = AVAILABLE_MODELS[model_name]
    
    # Anzahl Parameter bestimmen
    n_params = len(param_names)
    
    # Subplot-Layout bestimmen (Parameter + R²)
    total_plots = n_params + 1  # Parameter + R²
    
    if total_plots <= 3:
        fig, axes = plt.subplots(1, total_plots, figsize=(5*total_plots, 5))
    else:
        # Bei mehr als 3 Plots: 2 Zeilen
        cols = int(np.ceil(total_plots / 2))
        fig, axes = plt.subplots(2, cols, figsize=(5*cols, 10))
        axes = axes.flatten()  # Für einfacheren Zugriff
    
    # Sicherstellen, dass axes immer ein Array ist
    if total_plots == 1:
        axes = [axes]
    
    # Farben für Parameter
    colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    # Histogramme für alle Parameter erstellen
    for i, param_name in enumerate(param_names):
        if f'{param_name}_values' in results:
            values = results[f'{param_name}_values']
            mean_val = results[f'{param_name}_mean']
            ci_lower = results[f'{param_name}_ci_lower']
            ci_upper = results[f'{param_name}_ci_upper']
            unit = model_info['param_units'][i]
            
            color = colors[i % len(colors)]
            
            axes[i].hist(values, bins=50, alpha=0.7, color=color, edgecolor='black')
            axes[i].axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                           label=f'Mittel: {mean_val:.4f}')
            axes[i].axvline(ci_lower, color='orange', linestyle=':', label='95% CI')
            axes[i].axvline(ci_upper, color='orange', linestyle=':')
            
            # Achsenbeschriftung
            xlabel = f'{param_name} [{unit}]' if unit else param_name
            axes[i].set_xlabel(xlabel)
            axes[i].set_ylabel('Häufigkeit')
            axes[i].set_title(f'{param_name} Verteilung')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
    
    # R² Histogramm (letztes Subplot)
    r_idx = n_params
    axes[r_idx].hist(results['r_squared_values'], bins=50, alpha=0.7, color='purple', edgecolor='black')
    axes[r_idx].axvline(results['R_squared_mean'], color='red', linestyle='--', linewidth=2, 
                       label=f'Mittel: {results["R_squared_mean"]:.4f}')
    axes[r_idx].set_xlabel('R²')
    axes[r_idx].set_ylabel('Häufigkeit')
    axes[r_idx].set_title('R² Verteilung')
    axes[r_idx].legend()
    axes[r_idx].grid(True, alpha=0.3)
    
    # Ungenutzte Subplots ausblenden (falls vorhanden)
    for j in range(total_plots, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    
    # Dateiname basierend auf Modell
    filename = f'monte_carlo_results_{model_name}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Histogramme gespeichert als: {filename}")
    plt.show()

if __name__ == "__main__":
    # Parameter für die Messung - WICHTIG: Korrekte Werte eingeben!
    activ_param = {
        "Vf_well": 1.0,      # Verdünnungsfaktor der Gesamtansatzlösung im Well (z.B. 1.5 für 1:1.5)
        "Vf_prod": 1.0,      # Verdünnungsfaktor der Proteinlösung (z.B. 10 für 1:10) 
        "c_prod": 1.0        # Proteinkonzentration [mg/L] (z.B. 0.5 mg/L)
    }
    
    # Modell auswählen - ändern Sie hier das Modell!
    model_name = 'michaelis_menten'  # Verfügbare Modelle: 'michaelis_menten', 'hill', 'substrate_inhibition', 'linear', 'power_law'
    
    print(f"=== VERFÜGBARE MODELLE ===")
    for name, info in AVAILABLE_MODELS.items():
        params_str = ", ".join([f"{p} [{u}]" for p, u in zip(info['param_names'], info['param_units'])])
        print(f"- {name}: {info['description']} (Parameter: {params_str})")
    
    print(f"\n=== ORIGINALDATEN ANALYSE ===")
    print(f"Verwendetes Modell: {model_name}")
    result_original = schaetze_parameter(activ_param, model_name)
    
    if result_original:
        print("Analyse der Originaldaten erfolgreich!")
        model_info = AVAILABLE_MODELS[model_name]
        for param_name, unit in zip(model_info['param_names'], model_info['param_units']):
            value = result_original[param_name]
            error = result_original[f"{param_name}_error"]
            print(f"Original {param_name}: {value:.4f} ± {error:.4f} {unit}")
        print(f"Original R²: {result_original['R_squared']:.4f}")
    
    print("\n" + "="*60 + "\n")
    
    # Monte Carlo Simulation
    print("Starte Monte Carlo Simulation...")
    mc_results = monte_carlo_simulation(
        activ_param,
        model_name,
        n_iterations=10000,            # Erst weniger Iterationen zum Testen
        noise_level_calibration=0.01, # Weniger Rauschen: 1%
        noise_level_kinetics=0.01     # Weniger Rauschen: 1%
    )
    
    if mc_results and result_original:
        print("\n" + "="*60)
        print("=== VERGLEICH: ORIGINAL vs MONTE CARLO ===")
        print(f"{'Parameter':<12} {'Original':<15} {'MC Mittel':<15} {'MC Std':<10} {'Abweichung'}")
        print("-" * 70)
        
        model_info = AVAILABLE_MODELS[model_name]
        for param_name, unit in zip(model_info['param_names'], model_info['param_units']):
            if f'{param_name}_mean' in mc_results:
                original_val = result_original[param_name]
                mc_mean = mc_results[f'{param_name}_mean']
                mc_std = mc_results[f'{param_name}_std']
                diff = abs(original_val - mc_mean)
                
                param_display = f"{param_name} [{unit}]"
                print(f"{param_display:<12} {original_val:<15.4f} {mc_mean:<15.4f} {mc_std:<10.4f} {diff:.4f}")
        
        print(f"\nUnsicherheitsanalyse:")
        for param_name in model_info['param_names']:
            if f'{param_name}_mean' in mc_results:
                mc_mean = mc_results[f'{param_name}_mean']
                mc_std = mc_results[f'{param_name}_std']
                cv = (mc_std/mc_mean)*100
                print(f"{param_name} Variationskoeffizient: {cv:.2f}%")
        
        # Korrelation für 2-Parameter Modelle
        if len(model_info['param_names']) == 2 and 'correlation_matrix' in mc_results:
            param1, param2 = model_info['param_names']
            correlation = mc_results['correlation_matrix'][0, 1]
            print(f"\nParameter-Korrelation:")
            print(f"{param1}-{param2} Korrelationskoeffizient: {correlation:.4f}")
            if abs(correlation) > 0.7:
                print("  → Starke Korrelation zwischen den Parametern!")
            elif abs(correlation) > 0.3:
                print("  → Moderate Korrelation zwischen den Parametern")
            else:
                print("  → Schwache Korrelation zwischen den Parametern")
        
    else:
        print("Monte Carlo Simulation oder Original-Analyse fehlgeschlagen!")
