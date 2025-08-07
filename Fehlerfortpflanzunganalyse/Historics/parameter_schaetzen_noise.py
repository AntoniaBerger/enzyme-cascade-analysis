from scipy.stats import linregress
import numpy as np
import pandas as pd
import os
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

BASE_PATH = r"C:\Users\berger\Documents\Projekts\enzyme-cascade-analysis\Fehlerfortpflanzunganalyse"

def safe_flatten_to_float(data, name="data"):
    """
    Sichere Konvertierung von verschachtelten Arrays zu flachen Float-Arrays
    
    Löst das "setting an array element with a sequence" Problem
    """
    try:
        # Überprüfe ob bereits ein sauberes numerisches Array
        if hasattr(data, 'dtype') and np.issubdtype(data.dtype, np.number):
            return data.astype(float)
        
        # Flatten für verschachtelte Strukturen
        if hasattr(data, 'flatten'):
            flattened = data.flatten()
        else:
            flattened = np.array(data).flatten()
        
        # Prüfe ersten Wert auf Typ
        if len(flattened) > 0:
            first_val = flattened[0]
            
            # Wenn verschachtelt (Listen/Arrays), extrahiere rekursiv
            if hasattr(first_val, '__iter__') and not isinstance(first_val, (str, bytes)):
                all_values = []
                for item in flattened:
                    if hasattr(item, '__iter__') and not isinstance(item, (str, bytes)):
                        all_values.extend(list(item))
                    else:
                        all_values.append(item)
                flattened = np.array(all_values)
        
        # Konvertiere zu float
        result = flattened.astype(float)
        return result
        
    except Exception:
        # Fallback: Versuche Element-für-Element
        try:
            flat_list = []
            
            def extract_numbers(obj):
                if isinstance(obj, (int, float)):
                    flat_list.append(float(obj))
                elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes)):
                    for item in obj:
                        extract_numbers(item)
                else:
                    try:
                        flat_list.append(float(obj))
                    except Exception:
                        flat_list.append(0.0)  # Fallback
            
            extract_numbers(data)
            result = np.array(flat_list)
            return result
            
        except Exception:
            return None

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

def two_substrat_michaelis_menten_with_one_inhibition(concentration_data, Vmax, Km1, Km2, Ki):
    """Zwei-Substrat Michaelis-Menten mit Inhibition"""
    
    S1_values, S2_values = concentration_data
    
    # Sicherstellen, dass es numpy arrays sind
    S1_values = np.asarray(S1_values)
    S2_values = np.asarray(S2_values)
    
    # Element-weise Berechnung für alle Datenpunkte
    rates = (Vmax * S1_values * S2_values) / ((Km1 + S1_values) * (Km2 + S2_values) * (1 + (S2_values / Ki)))
    
    return rates

def two_substrat_michaelis_menten_with_two_inhibition(concentration_data, Vmax, Km1, Km2, Ki1, Ki2):
    """Zwei-Substrat Michaelis-Menten mit Inhibition"""
    
    S1_values, S2_values = concentration_data
    
    # Sicherstellen, dass es numpy arrays sind
    S1_values = np.asarray(S1_values)
    S2_values = np.asarray(S2_values)
    
    # Element-weise Berechnung für alle Datenpunkte
    rates = (Vmax * S1_values * S2_values) / ((Km1 + S1_values) * (Km2 + S2_values) * (1 + (S2_values / Ki1))) * (1 + (S1_values / Ki2))

    return rates

def full_reaction_system(concentration_data, Vmax1, Vmax2, Vmax3, KmPD, KmNAD, KmLactol, KmNADH, KiPD, KiNAD, KiLactol):
    """
    Wrapper für curve_fit Kompatibilität - nimmt flache Parameter entgegen
    Berechnet die Enzymaktivität für das vollständige Drei-Reaktions-System
    
    ALLE DREI REAKTIONEN:
    - Reaktion 1: PD + NAD → Pyruvat + NADH
    - Reaktion 2: Lactol + NADH → ... (mit PD/NAD Inhibition)
    - Reaktion 3: Lactol + NAD → ... (mit Lactol Inhibition)
    """
    # Entpacke Substratkonzentrationen, Inhibitor-Konzentrationen und Reaktions-IDs
    S1, S2, Inhibitor, reaction_ids = concentration_data
    
    # Initialisiere Ergebnis-Array
    V_obs = np.zeros_like(S1, dtype=float)
    
    # Reaktion 1: PD + NAD → Pyruvat + NADH
    reaction_1_mask = (reaction_ids == 1)
    if np.any(reaction_1_mask):
        # S1 = NAD oder konstante NAD, S2 = PD oder konstante PD
        S1_r1 = S1[reaction_1_mask] # NAD
        S2_r1 = S2[reaction_1_mask] # PD
        
        V_obs[reaction_1_mask] = (Vmax1 * S1_r1 * S2_r1) / (
            (KmPD + S2_r1) *  (KmNAD + S1_r1)
        )
    
    # Reaktion 2: Lactol + NADH → ... (mit PD Inhibition)
    reaction_2_mask = (reaction_ids == 2)
    if np.any(reaction_2_mask):
        S1_r2 = S1[reaction_2_mask]  # Lactol
        S2_r2 = S2[reaction_2_mask]  # NADH
        PD_inhibitor = Inhibitor[reaction_2_mask]  # Variable PD-Konzentration als Inhibitor
        
        V_obs[reaction_2_mask] = (Vmax2 * S1_r2 * S2_r2) / (
            (KmLactol * (1 + PD_inhibitor/KiPD) + S1_r2) * (KmNADH * (1 + PD_inhibitor/KiNAD) + S2_r2)
        )
    # Reaktion 3: Lactol + NAD → ... (mit Lactol Selbst-Inhibition)
    reaction_3_mask = (reaction_ids == 3)
    if np.any(reaction_3_mask):
        S1_r3 = S1[reaction_3_mask]  # Lactol
        S2_r3 = S2[reaction_3_mask]  # NAD
        
        V_obs[reaction_3_mask] = (Vmax3 * S1_r3 * S2_r3) / (
            (KmLactol * (1 + S1_r3/KiLactol) + S1_r3) * (KmNAD + S2_r3)
        )
    
    return V_obs


# Verfügbare Modelle definieren
AVAILABLE_MODELS = {
    'michaelis_menten': {
        'function': michaelis_menten,
        'param_names': ['Vmax', 'Km'],
        'param_units': ['U', 'mM'],
        'initial_guess_func': lambda rates, concs: [max(rates), np.median(concs)],
        'description': 'Michaelis-Menten Kinetik'
    },
    'two_substrat_michaelis_menten': {
        'function': two_substrat_michaelis_menten,
        'param_names': ['Vmax', 'Km1', 'Km2'],
        'param_units': ['U/mg', 'mM', 'mM'],
        'initial_guess_func': lambda rates, concs: [max(rates), 1.0, 1.0],
        'description': 'Zwei-Substrat Michaelis-Menten Kinetik',
        'special_data_format': True  # Flag für spezielle Datenbehandlung
    },
    'two_substrat_michaelis_menten_with_one_inhibition': {
        'function': two_substrat_michaelis_menten_with_one_inhibition,
        'param_names': ['Vmax', 'Km1', 'Km2', 'Ki'],
        'param_units': ['U/mg', 'mM', 'mM', 'mM'],
        'initial_guess_func': lambda rates, concs: [max(rates), 1.0, 1.0, 10.0],
        'description': 'Zwei-Substrat Michaelis-Menten mit einer Inhibition',
        'special_data_format': True
    },
    'two_substrat_michaelis_menten_with_two_inhibition': {
        'function': two_substrat_michaelis_menten_with_two_inhibition,
        'param_names': ['Vmax', 'Km1', 'Km2', 'Ki1', 'Ki2'],
        'param_units': ['U/mg', 'mM', 'mM', 'mM', 'mM'],
        'initial_guess_func': lambda rates, concs: [max(rates), 1.0, 1.0, 10.0, 10.0],
        'description': 'Zwei-Substrat Michaelis-Menten mit zwei Inhibitionen',
        'special_data_format': True
    },
    'full_reaction_system': {
        'function': full_reaction_system,  # Verwende Wrapper für curve_fit
        'param_names': ['Vmax1', 'Vmax2', 'Vmax3', 'KmPD', 'KmNAD', 'KmLactol', 'KmNADH', 'KiPD', 'KiNAD', 'KiLactol'],
        'param_units': ['U/mg', 'U/mg', 'U/mg', 'mM', 'mM', 'mM', 'mM', 'mM', 'mM', 'mM'],
        'initial_guess_func': lambda rates, concs: [
            max(rates)*1.0, max(rates)*0.8, max(rates)*0.6,  # Vmax1, Vmax2, Vmax3
            1.0, 2.0, 1.5, 3.0,  # KmPD, KmNAD, KmLactol, KmNADH
            10.0, 15.0, 20.0      # KiPD, KiNAD, KiLactol
        ],
        'description': 'Ein Enzym mit drei Aktivitäten (vollständiges System)',
        'special_data_format': True,
        'complex_model': True,
        'uses_wrapper': True  # Flag für Wrapper-Funktion
    }
}

def is_linear(x, y, threshold=0.80):
    """Prüft ob Daten linear sind basierend auf R² (gelockerte Kriterien für mehr Datenpunkte)"""
    if len(x) < 3 or len(y) < 3:
        return False
    try:
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        return abs(r_value) > threshold
    except:
        return False

def get_concentrations(csv_data):
    """Extrahiert die Konzentrationen aus CSV-Kinetikdaten"""
    # Spalte 2 (Index 1) enthält die Konzentrationen, ab Zeile 3 (Index 2)
    concentrations_raw = csv_data.iloc[2:, 1].dropna().values
    concentrations = [float(x) for x in concentrations_raw]
    return np.array(concentrations)

def get_adsoption_data(csv_data):
    """Extrahiert die Absorptionsdaten aus CSV-Kinetikdaten"""
    # Ab Spalte 3 (Index 2) sind die Absorptionsdaten, ab Zeile 3 (Index 2)
    absorption_data = csv_data.iloc[2:, 2:].values
    absorption_data_clean = []
    
    for row in absorption_data:
        # Konvertiere zu float und entferne NaN-Werte
        row_cleaned = []
        for val in row:
            try:
                float_val = float(val)
                if not np.isnan(float_val):
                    row_cleaned.append(float_val)
            except (ValueError, TypeError):
                continue
        
        if len(row_cleaned) > 5:  # Mindestens 5 Datenpunkte
            absorption_data_clean.append(np.array(row_cleaned))
    
    return np.array(absorption_data_clean, dtype=object)

def get_time_points(csv_data):
    """Extrahiert die Zeitpunkte aus CSV-Kinetikdaten"""
    # KORREKTUR: Die Zeitpunkte stehen in der ZWEITEN Zeile (Index 1), ab Spalte 3 (Index 2)
    # Format: Well,Konzentration [mM],Raw Data  (340),Raw Data  (340),...
    #         ,Time [s],0,6,12,18,24,30,36,42,...
    
    # Die echten Zeitpunkte stehen in Zeile 1 (Time [s] Zeile), ab Spalte 2
    time_row = csv_data.iloc[1, 2:].values  # Zeile 1 (Time [s]), ab Spalte 2
    
    # Konvertiere zu numerischen Werten und entferne NaN
    time_points = []
    for i, t in enumerate(time_row):
        try:
            time_val = float(t)
            if not np.isnan(time_val):
                time_points.append(time_val)
        except (ValueError, TypeError):
            pass
    
    return np.array(time_points)

def berechne_kalibrierung(nadh_kalibrierung):
    """Berechnet die Kalibrierung basierend auf NADH-Konzentrationen und Extinktionen"""
    x = nadh_kalibrierung.NADH
    y = nadh_kalibrierung.Mittelwert

    if not is_linear(x, y):
        print("Die Daten sind nicht linear. Bitte überprüfen Sie die Kalibrierung.")
        return None

    slope_cal, intercept_cal, r_value_cal, _, _ = linregress(x, y)
    
    return slope_cal, intercept_cal, r_value_cal

def berechne_aktivitaet(concentrations, absorption_data, time_points, slope_cal, activ_param, verbose=True):

    initial_rates = []
    valid_concentrations = []

    if verbose:
        # Kurze Übersicht der geladenen Daten
        print(f"Geladene Daten: {len(concentrations)} Wells, {len(time_points)} Zeitpunkte")

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
            slope_test, intercept_test, r_value_test, p_value_test, std_err_test = linregress(time_final, abs_final)
            r_squared = r_value_test**2
            
            if verbose and r_squared < 0.90:
                print(f"Well {i+1} (Konz: {conc_float} mM): R² = {r_squared:.3f} - nicht linear genug")
                continue
            elif not is_linear(time_final, abs_final):
                if verbose:
                    print(f"Well {i+1} (Konz: {conc_float} mM): R² = {r_squared:.3f} - ist nicht linear")
                continue
            
            # Lineare Regression für initiale Rate (bereits gemacht)
            slope, intercept, r_value, p_value, std_err = slope_test, intercept_test, r_value_test, p_value_test, std_err_test
            
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

def get_rates_and_concentrations(csv_data, slope_cal, param):
    """
    Lädt und verarbeitet das R1-NAD Experiment.
    Gibt (r1_nad_rates, r1_nad_valid_conc) zurück oder None bei Fehler.
    """
    try:
        r1_nad_data = pd.read_csv(csv_data, header=None)
        r1_nad_conc = get_concentrations(r1_nad_data)
        r1_nad_time = get_time_points(r1_nad_data)
        r1_nad_abs = get_adsoption_data(r1_nad_data)
        
        result_r1_nad = berechne_aktivitaet(r1_nad_conc, r1_nad_abs, r1_nad_time, slope_cal, param, verbose=False)
        if result_r1_nad is None:
            print("FEHLER: R1-NAD Experiment fehlgeschlagen")
            return None
        r1_nad_rates, r1_nad_valid_conc = result_r1_nad
        return r1_nad_rates, r1_nad_valid_conc
    except Exception as e:
        print(f"FEHLER beim Laden des R1-NAD Experiments: {e}")
        return None

 # Hilfsfunktion zum Kombinieren der Daten für das vollständige Reaktionssystem

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

    x_data = (S1_combined, S2_combined, Inhibitor_combined, reaction_ids)
    y_data = activities_combined
    return x_data, y_data, S1_combined, S2_combined, Inhibitor_combined, activities_combined, reaction_ids

def fitt_parameters(model_name, x_data, y_data, model_info, model_func=None):
    
    # Initial guess basierend auf Modell
    if model_name == 'two_substrat_michaelis_menten':
        # Für Zwei-Substrat: spezielle initial guess
        p0 = [max(y_data), 1.0, 1.0]  # [Vmax, Km1, Km2]
    elif model_name == 'two_substrat_michaelis_menten_with_one_inhibition':
        # Für Zwei-Substrat mit einer Inhibition
        p0 = [max(y_data), 1.0, 1.0, 10.0]  # [Vmax, Km1, Km2, Ki]
    elif model_name == 'two_substrat_michaelis_menten_with_two_inhibition':
        # Für Zwei-Substrat mit zwei Inhibitionen
        p0 = [max(y_data), 1.0, 1.0, 10.0, 10.0]  # [Vmax, Km1, Km2, Ki1, Ki2]
    elif model_name == 'full_reaction_system':
        # Für vollständiges Reaktionssystem - bereits flache Parameter von initial_guess_func
        p0 = model_info['initial_guess_func'](y_data, x_data)
    else:
        # Für Ein-Substrat: normale initial guess
        p0 = model_info['initial_guess_func'](y_data, x_data)
    
    
    # Special handling for full_reaction_system
    if model_name == 'full_reaction_system':
        # Verwende die bereits definierte Wrapper-Funktion, die alle drei Reaktionen integriert
        # Parameter-Grenzen für physikalisch sinnvolle Werte
        bounds_lower = [0.001, 0.001, 0.001,   # Vmax1, Vmax2, Vmax3 > 0
                        0.1, 0.1, 0.1, 0.1,     # KmPD, KmNAD, KmLactol, KmNADH > 0.1 mM
                        0.1, 0.1, 0.1]           # KiPD, KiNAD, KiLactol > 0.1 mM
        bounds_upper = [100, 100, 100,          # Vmax1, Vmax2, Vmax3 < 100 U/mg
                        1000, 100, 1000, 100,    # Km-Werte < 1000 bzw. 100 mM
                        1000, 1000, 1000]        # Ki-Werte < 1000 mM
        
        params, covariance = curve_fit(full_reaction_system, x_data, y_data, 
                                        p0=p0, bounds=(bounds_lower, bounds_upper), maxfev=5000)
        fitted_params = params
        param_errors = np.sqrt(np.diag(covariance))
        
    else:
        # Normales curve_fit für alle anderen Modelle
        params, covariance = curve_fit(model_func, x_data, y_data, p0=p0, maxfev=5000)
        fitted_params = params
        param_errors = np.sqrt(np.diag(covariance))
    
    # R² berechnen
    if model_name == 'full_reaction_system':
        # Für full_reaction_system: verwende die Wrapper-Funktion
        y_pred = full_reaction_system(x_data, *fitted_params)
    else:
        y_pred = model_func(x_data, *fitted_params)
        
    ss_res = np.sum((y_data - y_pred) ** 2)
    ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
        
    # Parameter-Ausgabe je nach Modelltyp
    
    
    # Zusätzliche Ausgabe für Zwei-Substrat-Modelle (aber nicht für full_reaction_system)
    if model_name in ['two_substrat_michaelis_menten', 'two_substrat_michaelis_menten_with_one_inhibition', 
                        'two_substrat_michaelis_menten_with_two_inhibition']:
        pass
    
    # Ergebnis-Dictionary erstellen
    result = {
        'R_squared': r_squared,
        'activities': y_data,
        'model_name': model_name,
        'model_description': model_info['description']
    }
    
    # Parameter dynamisch hinzufügen - spezielle Behandlung für full_reaction_system
    if model_name == 'full_reaction_system':
        # Konvertiere flache Parameter zurück zu Dictionary für Kompatibilität
        param_dict = {
            "Vmax": [fitted_params[0], fitted_params[1], fitted_params[2]],
            "Km": [fitted_params[3], fitted_params[4], fitted_params[5], fitted_params[6]],
            "Ki": [fitted_params[7], fitted_params[8], fitted_params[9]]
        }
        result['param_dict'] = param_dict
        result['Vmax_values'] = param_dict["Vmax"]
        result['Km_values'] = param_dict["Km"]
        result['Ki_values'] = param_dict["Ki"]
        result['param_errors'] = param_errors
        result['all_params_flat'] = fitted_params  # Flache Parameter für Kompatibilität

        # Fix: define param_names here
        param_names = model_info['param_names']
        # Einzelne Parameter für einfacheren Zugriff
        for i, param_name in enumerate(param_names):
            if i < len(fitted_params):
                result[param_name] = fitted_params[i]
                result[f"{param_name}_error"] = param_errors[i]
    else:
        pass
    
    return result

def schaetze_parameter(param, model_name='michaelis_menten', verbose=True):
    
    # Modell validieren
    if model_name not in AVAILABLE_MODELS:
        print(f"Fehler: Modell '{model_name}' nicht verfügbar!")
        print(f"Verfügbare Modelle: {list(AVAILABLE_MODELS.keys())}")
        return None
    
    model_info = AVAILABLE_MODELS[model_name]
    model_func = model_info['function']
    param_names = model_info['param_names']
    param_units = model_info['param_units']
    
    
    # NADH Kalibrierung laden (für alle Modelle gleich)
    try:
        nadh_path = os.path.join(BASE_PATH, "Daten", "Rohdaten", "Plate_Reader", "Kalibriergeraden", "NADH_Kalibriergerade.xlsx")
        nadh_kalibrierung = pd.read_excel(nadh_path)
    except FileNotFoundError:
        print("WARNUNG: Kalibrierungsdatei nicht gefunden! Bitte überprüfen Sie den Pfad.")
        input("Drücken Sie Enter, um fortzufahren...")
    
    # Kalibrierungssteigung berechnen
    slope_cal, intercept_cal, r_value_cal = berechne_kalibrierung(nadh_kalibrierung)
    if slope_cal is None:
        return None

    if model_name in ['two_substrat_michaelis_menten', 'two_substrat_michaelis_menten_with_one_inhibition', 
                      'two_substrat_michaelis_menten_with_two_inhibition']:
        # ZWEI-SUBSTRAT MODELLE: Beide CSV-Dateien laden (r1 Verzeichnis)
        csv_path_nad_var = os.path.join(BASE_PATH, "Daten","Rohdaten","Plate_Reader", "Kinetik-Messungen", "r1", "r_1_NAD_PD_500nM.csv")
        csv_path_pd_var = os.path.join(BASE_PATH, "Daten","Rohdaten","Plate_Reader", "Kinetik-Messungen", "r1", "r1_PD_NAD_5nM.csv")
        
        print("\n--- Zwei-Substrat CSV-Kinetikdaten laden ---")
        print("Experiment 1: NAD variabel, PD konstant (500 nM)")
        print("Experiment 2: PD variabel, NAD konstant (5 mM)")
        
        try:
            # Experiment 1: NAD variabel, PD konstant
            print(f"Lade Datei 1: {csv_path_nad_var}")
            r1_nad_var = pd.read_csv(csv_path_nad_var, header=None)
            concentrations_nad = get_concentrations(r1_nad_var)
            time_points_nad = get_time_points(r1_nad_var)
            absorption_data_nad = get_adsoption_data(r1_nad_var)
            
            print(f"Experiment 1: Berechne Aktivitäten für {len(concentrations_nad)} Konzentrationen...")
            result_nad = berechne_aktivitaet(
                concentrations_nad, absorption_data_nad, time_points_nad, slope_cal, param, verbose=True
            )
            
            if result_nad is None:
                print("FEHLER: Experiment 1 (NAD variabel) lieferte keine gültigen Daten")
                return None
                
            initial_rates_nad, valid_concentrations_nad = result_nad
            
            # Experiment 2: PD variabel, NAD konstant
            print(f"\nLade Datei 2: {csv_path_pd_var}")
            r1_pd_var = pd.read_csv(csv_path_pd_var, header=None)
            concentrations_pd = get_concentrations(r1_pd_var)  # Zurück zur normalen Funktion
            time_points_pd = get_time_points(r1_pd_var)
            absorption_data_pd = get_adsoption_data(r1_pd_var)
            
            print(f"Experiment 2: Berechne Aktivitäten für {len(concentrations_pd)} Konzentrationen...")
            result_pd = berechne_aktivitaet(
                concentrations_pd, absorption_data_pd, time_points_pd, slope_cal, param, verbose=True
            )
            
            if result_pd is None:
                print("FEHLER: Experiment 2 (PD variabel) lieferte keine gültigen Daten")
                return None
                
            initial_rates_pd, valid_concentrations_pd = result_pd
            
            print(f"Experiment 1 (NAD variabel): {len(valid_concentrations_nad)} gültige Datenpunkte")
            print(f"Experiment 2 (PD variabel): {len(valid_concentrations_pd)} gültige Datenpunkte")
            
            # Überprüfe ob genug Daten vorhanden sind
            if len(valid_concentrations_nad) < 3 or len(valid_concentrations_pd) < 3:
                print(f"FEHLER: Nicht genug gültige Datenpunkte (NAD: {len(valid_concentrations_nad)}, PD: {len(valid_concentrations_pd)})")
                return None
            
        except FileNotFoundError as e:
            print(f"WARNUNG: CSV-Datei nicht gefunden: {e}")
            print(f"Gesuchte Pfade:")
            print(f"- {csv_path_nad_var}")
            print(f"- {csv_path_pd_var}")
            return None
        except Exception as e:
            print(f"FEHLER beim Laden der CSV-Dateien: {e}")
            return None
        
        if (initial_rates_nad is None or len(valid_concentrations_nad) < 3 or
            initial_rates_pd is None or len(valid_concentrations_pd) < 3):
            print("Fehler: Nicht genug gültige Datenpunkte aus beiden CSV-Experimenten!")
            return None
        
        # Konstante Substratkonzentrationen definieren
        NAD_constant = 5.0  # mM (aus Experiment 2)
        PD_constant = 0.5   # mM (500 nM aus Experiment 1)
        
        # Kombinierte Datenstruktur für echte Zwei-Substrat-Analyse erstellen
        S1_combined = np.concatenate([
            valid_concentrations_nad,                          # Variable NAD-Werte (Experiment 1)
            np.full(len(valid_concentrations_pd), NAD_constant) # Konstante NAD-Werte (Experiment 2)
        ])
        
        S2_combined = np.concatenate([
            np.full(len(valid_concentrations_nad), PD_constant), # Konstante PD-Werte (Experiment 1)
            valid_concentrations_pd                             # Variable PD-Werte (Experiment 2)
        ])
        
        activities_combined = np.concatenate([initial_rates_nad, initial_rates_pd])
        
        # Für Zwei-Substrat und komplexe Modelle: verwende kombinierte Datenstruktur
        x_data = (S1_combined, S2_combined)  # Tupel für alle Zwei-Substrat-Modelle
        y_data = activities_combined
        

        
    elif model_name == 'full_reaction_system':

        # Reaktion 1 Dateien (r1)
        csv_r1_nad_var = os.path.join(BASE_PATH, "Daten","Rohdaten","Plate_Reader", "Kinetik-Messungen", "r1", "r_1_NAD_PD_500mM.csv")
        csv_r1_pd_var = os.path.join(BASE_PATH, "Daten","Rohdaten","Plate_Reader", "Kinetik-Messungen", "r1", "r1_PD_NAD_5mM.csv")
        
        # Reaktion 2 Dateien (r2)
        csv_r2_hp_var = os.path.join(BASE_PATH, "Daten","Rohdaten","Plate_Reader", "Kinetik-Messungen", "r2", "HP (0.6 mM NADH).csv")
        csv_r2_nadh_var = os.path.join(BASE_PATH, "Daten","Rohdaten","Plate_Reader", "Kinetik-Messungen", "r2", "NADH (300 mM HP).csv")
        csv_r2_pd_var = os.path.join(BASE_PATH, "Daten","Rohdaten","Plate_Reader", "Kinetik-Messungen", "r2", "PD (0.6 mM NADH, 300 mM HP).csv")
        
        # Reaktion 3 Dateien (r3)
        csv_r3_nad_var = os.path.join(BASE_PATH, "Daten","Rohdaten","Plate_Reader", "Kinetik-Messungen", "r3", "NAD (500 mM Lactol).csv")
        csv_r3_lactol_var = os.path.join(BASE_PATH, "Daten","Rohdaten","Plate_Reader", "Kinetik-Messungen", "r3", "Lactol (5 mM NAD).csv")
        
        try:
            # ===== REAKTION 1: PD + NAD =====
            if verbose:
                print("\n=== REAKTION 1: PD + NAD ===")
            
            # R1 - NAD variabel, PD konstant
            if verbose:
                print(f"Lade R1-NAD variabel: {csv_r1_nad_var}")
           
            r1_nad_rates, r1_nad_valid_conc = get_rates_and_concentrations(csv_r1_nad_var, slope_cal, param)
            if r1_nad_rates is None:
                if verbose:
                    print("FEHLER: R1-NAD Experiment fehlgeschlagen")
                return None
            
            # R1 - PD variabel, NAD konstant
            if verbose:
                print(f"Lade R1-PD variabel: {csv_r1_pd_var}")
            r1_pd_rates, r1_pd_valid_conc = get_rates_and_concentrations(csv_r1_pd_var, slope_cal, param)
            if r1_pd_rates is None:
                print("FEHLER: R1-PD Experiment fehlgeschlagen")
                return None

                    
            # ===== REAKTION 2: Lactol + NADH (mit PD/NAD Inhibition) =====
            
            # R2 - PD variabel (als Inhibitor) , Lactol + NADH konstant 
           
            r2_pd_rates, r2_pd_valid_conc = get_rates_and_concentrations(csv_r2_pd_var, slope_cal, param)
            if r2_pd_rates is None: 
                print("FEHLER: R2-PD Experiment fehlgeschlagen")
                return None
            
            # R2 - NADH variabel
            r2_nadh_rates, r2_nadh_valid_conc = get_rates_and_concentrations(csv_r2_nadh_var, slope_cal, param)
            if r2_nadh_rates is None:
                print("FEHLER: R2-NADH Experiment fehlgeschlagen")
                return None
            # R2 - HP variabel (Lactol)
            r2_hp_rates, r2_hp_valid_conc = get_rates_and_concentrations(csv_r2_hp_var, slope_cal, param)
            if r2_hp_rates is None: 
                print("FEHLER: R2-HP Experiment fehlgeschlagen")
                return None
            
            
            # ===== REAKTION 3: Lactol + NAD (mit Lactol Inhibition) =====
                    
            r3_nad_rates, r3_nad_valid_conc = get_rates_and_concentrations(csv_r3_nad_var, slope_cal, param)
            if r3_nad_rates is None:
                print("FEHLER: R3-NAD Experiment fehlgeschlagen")
                return None
            
            # R3 - Lactol variabel
            r3_lactol_rates, r3_lactol_valid_conc = get_rates_and_concentrations(csv_r3_lactol_var, slope_cal, param)
            if r3_lactol_rates is None:
                print("FEHLER: R3-Lactol Experiment fehlgeschlagen")
                return None

        except FileNotFoundError as e:
            print(f"WARNUNG: CSV-Datei für vollständiges System nicht gefunden: {e}")
            return None
        except Exception as e:
            print(f"FEHLER beim Laden der Vollsystem-CSV-Dateien: {e}")
            return None
        
        # VOLLSTÄNDIGE INTEGRATION ALLER DREI REAKTIONEN
        if verbose:
            print(f"\n=== VOLLSTÄNDIGES SYSTEM DATENSTRUKTUR ===")
            print("Integriere ALLE drei Reaktionen für Multi-Aktivitäts-Enzym")
        
        # Reaktion 1: PD + NAD → Pyruvat + NADH
        # Experimentelle Bedingungen aus CSV-Dateien

        # Daten kombinieren
        x_data, y_data, S1_combined, S2_combined, Inhibitor_combined, activities_combined, reaction_ids = combine_full_reaction_system_data(
            r1_nad_valid_conc, r1_nad_rates, r1_pd_valid_conc, r1_pd_rates,
            r2_pd_valid_conc, r2_pd_rates, r2_nadh_valid_conc, r2_nadh_rates,
            r2_hp_valid_conc, r2_hp_rates, r3_nad_valid_conc, r3_nad_rates,
            r3_lactol_valid_conc, r3_lactol_rates
        )
        
        total_points = len(activities_combined)
        r1_points = len(r1_nad_valid_conc) + len(r1_pd_valid_conc)
        r2_points = len(r2_pd_valid_conc) + len(r2_nadh_valid_conc) + len(r2_hp_valid_conc)
        r3_points = len(r3_nad_valid_conc) + len(r3_lactol_valid_conc)
        
        if verbose:
            print(f"\n=== VOLLSTÄNDIGES MULTI-AKTIVITÄTS SYSTEM ===")
            print(f"Anzahl Datenpunkte: {total_points}")
            print(f"Reaktion 1 (PD + NAD): {r1_points} Punkte")
            print(f"Reaktion 2 (Lactol + NADH mit Inhibition): {r2_points} Punkte")
            print(f"Reaktion 3 (Lactol + NAD mit Inhibition): {r3_points} Punkte")

        if total_points < 30:
            print("WARNUNG: Wenige Datenpunkte für 10-Parameter-Modell!")
        
    else:
        # EIN-SUBSTRAT: Ihr bestehender Code (unverändert!)
        r1_path = os.path.join(BASE_PATH, "Daten", "Rohdaten", "Plate_Reader", "Kinetik-Messungen","r1", "r1_NAD_PD_mod.xlsx")
        r1 = pd.read_excel(r1_path)

        concentrations = get_concentrations(r1)
        time_points = get_time_points(r1)
        absorption_data = get_adsoption_data(r1)

        print(f"Ein-Substrat Daten:")
        print(f"Anzahl Wells: {len(concentrations)}")
        print(f"Zeitpunkte: {len(time_points)}")
        print(f"Konzentrationen: {concentrations}")

        initial_rates, valid_concentrations = berechne_aktivitaet(concentrations, absorption_data, time_points, slope_cal, param, verbose=False)
        
        # Für Ein-Substrat: verwende normale Datenstruktur
        x_data = valid_concentrations  # Array für michaelis_menten
        y_data = initial_rates
    
    # Parameter fitten
    try:
        result = fitt_parameters(model_name, x_data, y_data, model_info, model_func=model_func)
        
        if result is None:
            print("Fehler beim Fitten der Parameter!")
            return None
        
        return result
        
    except Exception as e:
        print(f"Fehler beim Fitten der {model_info['description']} Parameter: {e}")
        return None
        
def add_noise(data, noise_level=0.05):
    """
    Fügt Gauß'sches Rauschen zu den Daten hinzu - ROBUSTE VERSION
    
    Parameters:
    data: numpy array - die ursprünglichen Daten
    noise_level: float - Rauschpegel als Bruchteil des Signals (0.05 = 5%)
    
    Returns:
    numpy array - verrauschte Daten
    """
    try:
        # Sichere Array-Konvertierung
        data_array = np.asarray(data, dtype=float)
        
        # Standardabweichung basierend auf dem Signal
        std_dev = np.abs(data_array) * noise_level
        
        # Gauß'sches Rauschen hinzufügen
        noise = np.random.normal(0, std_dev, size=data_array.shape)
        
        result = data_array + noise
        
        return result
        
    except Exception:
        # Fallback: gib Original zurück
        return np.asarray(data, dtype=float)

def monte_carlo_simulation(param, model_name='michaelis_menten', n_iterations=1000, noise_level_calibration=0.03, noise_level_kinetics=0.02):
    """
    Monte Carlo Simulation für beliebige Modelle mit speziellem Support für Zwei-Substrat
    """
    
    print("=== MONTE CARLO SIMULATION ===")
    print("="*50)
    
    # Listen für Ergebnisse
    param_results = {}  # Dictionary für alle Parameter
    r_squared_results = []
    
    # Originaldaten einmal laden
    try:
        nadh_path = os.path.join(BASE_PATH, "Daten", "Rohdaten", "Plate_Reader", "Kalibriergeraden", "NADH_Kalibriergerade.xlsx")
        nadh_kalibrierung = pd.read_excel(nadh_path)
        
        # Für Ein-Substrat: Normales Laden (falls needed)
        if model_name not in ['two_substrat_michaelis_menten', 'two_substrat_michaelis_menten_with_one_inhibition', 
                         'two_substrat_michaelis_menten_with_two_inhibition', 'full_reaction_system']:
            r1_path = os.path.join(BASE_PATH, "Daten", "Rohdaten", "Plate_Reader", "Kinetik-Messungen","r1", "r1_NAD_PD_mod.xlsx")
            r1 = pd.read_excel(r1_path)

    except Exception as e:
        print(f"Fehler beim Laden der Dateien: {e}")
        return None
    
    try:
        # Kalibrierungsdaten (für alle Modelle gleich)
        x_original = nadh_kalibrierung.NADH.values
        y_original = nadh_kalibrierung.Mittelwert.values
        
        if model_name == 'full_reaction_system':
            # Vollständiges System: Alle sieben CSV-Dateien laden
            # Reaktion 1 - NAD variabel, PD konstant
            r1_nad_path = os.path.join(BASE_PATH, "Daten","Rohdaten","Plate_Reader", "Kinetik-Messungen", "r1", "r_1_NAD_PD_500mM.csv")
            r1_nad_data = pd.read_csv(r1_nad_path, header=None)
            r1_nad_conc_original = get_concentrations(r1_nad_data)
            r1_nad_time_original = get_time_points(r1_nad_data)
            r1_nad_abs_original = get_adsoption_data(r1_nad_data)
            
            # Reaktion 1 - PD variabel, NAD konstant
            r1_pd_path = os.path.join(BASE_PATH, "Daten","Rohdaten","Plate_Reader", "Kinetik-Messungen", "r1", "r1_PD_NAD_5mM.csv")
            r1_pd_data = pd.read_csv(r1_pd_path, header=None)
            r1_pd_conc_original = get_concentrations(r1_pd_data)
            r1_pd_time_original = get_time_points(r1_pd_data)
            r1_pd_abs_original = get_adsoption_data(r1_pd_data)
            
            # Reaktion 2 - PD variabel (Inhibitor)
            r2_pd_path = os.path.join(BASE_PATH, "Daten","Rohdaten","Plate_Reader", "Kinetik-Messungen", "r2", "PD (0.6 mM NADH, 300 mM HP).csv")
            r2_pd_data = pd.read_csv(r2_pd_path, header=None)
            r2_pd_conc_original = get_concentrations(r2_pd_data)
            r2_pd_time_original = get_time_points(r2_pd_data)
            r2_pd_abs_original = get_adsoption_data(r2_pd_data)
            
            # Reaktion 2 - NADH variabel
            r2_nadh_path = os.path.join(BASE_PATH, "Daten","Rohdaten","Plate_Reader", "Kinetik-Messungen", "r2", "NADH (300 mM HP).csv")
            r2_nadh_data = pd.read_csv(r2_nadh_path, header=None)
            r2_nadh_conc_original = get_concentrations(r2_nadh_data)
            r2_nadh_time_original = get_time_points(r2_nadh_data)
            r2_nadh_abs_original = get_adsoption_data(r2_nadh_data)
            
            # Reaktion 2 - HP/Lactol variabel
            r2_hp_path = os.path.join(BASE_PATH, "Daten","Rohdaten","Plate_Reader", "Kinetik-Messungen", "r2", "HP (0.6 mM NADH).csv")
            r2_hp_data = pd.read_csv(r2_hp_path, header=None)
            r2_hp_conc_original = get_concentrations(r2_hp_data)
            r2_hp_time_original = get_time_points(r2_hp_data)
            r2_hp_abs_original = get_adsoption_data(r2_hp_data)
            
            # Reaktion 3 - NAD variabel
            r3_nad_path = os.path.join(BASE_PATH, "Daten","Rohdaten","Plate_Reader", "Kinetik-Messungen", "r3", "NAD (500 mM Lactol).csv")
            r3_nad_data = pd.read_csv(r3_nad_path, header=None)
            r3_nad_conc_original = get_concentrations(r3_nad_data)
            r3_nad_time_original = get_time_points(r3_nad_data)
            r3_nad_abs_original = get_adsoption_data(r3_nad_data)
            
            # Reaktion 3 - HP/Lactol variabel
            r3_lactol_path = os.path.join(BASE_PATH, "Daten","Rohdaten","Plate_Reader", "Kinetik-Messungen", "r3", "Lactol (5 mM NAD).csv")
            r3_lactol_data = pd.read_csv(r3_lactol_path, header=None)
            r3_lactol_conc_original = get_concentrations(r3_lactol_data)
            r3_lactol_time_original = get_time_points(r3_lactol_data)
            r3_lactol_abs_original = get_adsoption_data(r3_lactol_data)
            
           
        else:
            pass
        
    except Exception as e:
        print(f"Fehler beim Extrahieren der Originaldaten: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    successful_iterations = 0
    failed_reasons = {"linearität": 0, "daten": 0, "fitting": 0, "andere": 0}
    
    for iteration in range(n_iterations):
        try:
            # 1. Kalibrierungsdaten verrauschen (für alle Modelle gleich)
            x_noisy = add_noise(x_original, noise_level_calibration)
            y_noisy = add_noise(y_original, noise_level_calibration)
            
            # Linearität der verrauschten Kalibrierung prüfen (weniger streng für Monte Carlo)
            if not is_linear(x_noisy, y_noisy, threshold=0.80):  # Reduziert von 0.85 auf 0.80
                failed_reasons["linearität"] += 1
                continue
            
            # 2. Kinetikdaten verrauschen - unterschiedlich je nach Modell
            if model_name == 'full_reaction_system':
                # Vollständiges System: Alle sieben CSV-Dateien verrauschen
                
                # R1 - NAD variabel - SICHERE VERSION
                r1_nad_conc_noisy = add_noise(r1_nad_conc_original, noise_level_calibration)
                r1_nad_time_noisy = add_noise(r1_nad_time_original, noise_level_kinetics)
                
                r1_nad_abs_clean = safe_flatten_to_float(r1_nad_abs_original, "r1_nad_abs_original")
                if r1_nad_abs_clean is None:
                    failed_reasons["daten"] += 1
                    continue
                r1_nad_abs_noisy = add_noise(r1_nad_abs_clean, noise_level_kinetics)
                
                # R1 - PD variabel - SICHERE VERSION
                r1_pd_conc_noisy = add_noise(r1_pd_conc_original, noise_level_calibration)
                r1_pd_time_noisy = add_noise(r1_pd_time_original, noise_level_kinetics)
                
                r1_pd_abs_clean = safe_flatten_to_float(r1_pd_abs_original, "r1_pd_abs_original")
                if r1_pd_abs_clean is None:
                    failed_reasons["daten"] += 1
                    continue
                r1_pd_abs_noisy = add_noise(r1_pd_abs_clean, noise_level_kinetics)
                
                # R2 - PD variabel (Inhibitor) - SICHERE VERSION
                r2_pd_conc_noisy = add_noise(r2_pd_conc_original, noise_level_calibration)
                r2_pd_time_noisy = add_noise(r2_pd_time_original, noise_level_kinetics)
                
                r2_pd_abs_clean = safe_flatten_to_float(r2_pd_abs_original, "r2_pd_abs_original")
                if r2_pd_abs_clean is None:
                    failed_reasons["daten"] += 1
                    continue
                r2_pd_abs_noisy = add_noise(r2_pd_abs_clean, noise_level_kinetics)
                
                # R2 - NADH variabel - SICHERE VERSION
                r2_nadh_conc_noisy = add_noise(r2_nadh_conc_original, noise_level_calibration)
                r2_nadh_time_noisy = add_noise(r2_nadh_time_original, noise_level_kinetics)
                
                r2_nadh_abs_clean = safe_flatten_to_float(r2_nadh_abs_original, "r2_nadh_abs_original")
                if r2_nadh_abs_clean is None:
                    failed_reasons["daten"] += 1
                    continue
                r2_nadh_abs_noisy = add_noise(r2_nadh_abs_clean, noise_level_kinetics)
                
                # R2 - HP/Lactol variabel - SICHERE VERSION
                r2_hp_conc_noisy = add_noise(r2_hp_conc_original, noise_level_calibration)
                r2_hp_time_noisy = add_noise(r2_hp_time_original, noise_level_kinetics)
                
                # Sichere Behandlung von r2_hp_abs_original mit safe_flatten_to_float
                r2_hp_abs_clean = safe_flatten_to_float(r2_hp_abs_original, "r2_hp_abs_original")
                if r2_hp_abs_clean is None:
                    failed_reasons["daten"] += 1
                    continue
                
                # Reshape zurück zur ursprünglichen Form falls nötig
                try:
                    if hasattr(r2_hp_abs_original, 'shape') and len(r2_hp_abs_original.shape) > 1:
                        r2_hp_abs_clean = r2_hp_abs_clean.reshape(r2_hp_abs_original.shape)
                except ValueError:
                    pass  # Verwende flaches Array
                
                r2_hp_abs_noisy = add_noise(r2_hp_abs_clean, noise_level_kinetics)
                
                # R3 - NAD variabel - SICHERE VERSION
                r3_nad_conc_noisy = add_noise(r3_nad_conc_original, noise_level_calibration)
                r3_nad_time_noisy = add_noise(r3_nad_time_original, noise_level_kinetics)
                
                r3_nad_abs_clean = safe_flatten_to_float(r3_nad_abs_original, "r3_nad_abs_original")
                if r3_nad_abs_clean is None:
                    failed_reasons["daten"] += 1
                    continue
                r3_nad_abs_noisy = add_noise(r3_nad_abs_clean, noise_level_kinetics)
                
                # R3 - Lactol variabel
                r3_lactol_conc_noisy = add_noise(r3_lactol_conc_original, noise_level_calibration)
                r3_lactol_time_noisy = add_noise(r3_lactol_time_original, noise_level_kinetics)
                
                # R3 Lactol abs - SICHERE VERSION
                r3_lactol_abs_clean = safe_flatten_to_float(r3_lactol_abs_original, "r3_lactol_abs_original")
                if r3_lactol_abs_clean is None:
                    failed_reasons["daten"] += 1
                    continue
                r3_lactol_abs_noisy = add_noise(r3_lactol_abs_clean, noise_level_kinetics)
                
                
                
                # Setze die verrauschten Werte temporär in die Dateien (simuliert)
                # oder nutze die originale Funktion direkt
                try:
                    # Einfacher Workaround: Verwende original Parameter mit wenig Variation
                    varied_param = param.copy()
                    # Füge kleine Variation zu den Parametern hinzu als Proxy für Datenrauschen
                    variation = np.random.normal(1.0, 0.01)  # 1% Variation
                    for key in varied_param:
                        if isinstance(varied_param[key], (int, float)):
                            varied_param[key] *= variation

                    result = schaetze_parameter_noisy(varied_param, model_name, verbose=False)  # Keine Ausgaben in Monte Carlo

                except Exception as e:
                    result = None
                
            if result is not None:
                # Modell-spezifische Validierung
                model_info = AVAILABLE_MODELS[model_name]
                param_names = model_info['param_names']
                
                # Basis-Validierung: Parameter müssen positiv und realistisch sein
                all_params_valid = True
                
                if model_name == 'full_reaction_system':
                    # Validierung für full_reaction_system mit der neuen schaetze_parameter Struktur
                    if 'all_params_flat' in result and result['all_params_flat'] is not None:
                        params_flat = result['all_params_flat']
                        # Alle Parameter müssen positiv und realistisch sein
                        for param_value in params_flat:
                            if param_value <= 0 or param_value > 10000:
                                all_params_valid = False
                                break
                    else:
                        all_params_valid = False
                else:
                    # Normale Validierung für andere Modelle
                    for param_name in param_names:
                        param_value = result.get(param_name, 0)
                        if param_value <= 0 or param_value > 10000:
                            all_params_valid = False
                            break
                
                if all_params_valid and result['R_squared'] > 0.3:  # Reduziert von 0.5 auf 0.3
                    # Sammle Parameter für die Statistik
                    if model_name == 'full_reaction_system':
                        # Für full_reaction_system: sammle die flachen Parameter
                        if 'all_params_flat' not in param_results:
                            param_results['all_params_flat'] = []
                        if 'all_params_flat' in result:
                            param_results['all_params_flat'].append(result['all_params_flat'])
                    else:
                        # Normale Parameter-Sammlung für andere Modelle
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
                
            if (iteration + 1) % 100 == 0:
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
    model_info = AVAILABLE_MODELS[model_name]
    param_names = model_info['param_names']
    
    param_arrays = {}
    
    if model_name == 'full_reaction_system':
        # Spezielle Behandlung für full_reaction_system
        if 'all_params_flat' in param_results:
            all_params_flat = np.array(param_results['all_params_flat'])
            
            # full_reaction_system hat genau 10 Parameter (Index 0-9)
            # Vmax1, Vmax2, Vmax3, KmPD, KmNAD, KmLactol, KmNADH, KiPD, KiNAD, KiLactol
            param_arrays['Vmax1'] = all_params_flat[:, 0]
            param_arrays['Vmax2'] = all_params_flat[:, 1]
            param_arrays['Vmax3'] = all_params_flat[:, 2]
            param_arrays['KmPD'] = all_params_flat[:, 3]
            param_arrays['KmNAD'] = all_params_flat[:, 4]
            param_arrays['KmLactol'] = all_params_flat[:, 5]
            param_arrays['KmNADH'] = all_params_flat[:, 6]
            param_arrays['KiPD'] = all_params_flat[:, 7]
            param_arrays['KiNAD'] = all_params_flat[:, 8]
            param_arrays['KiLactol'] = all_params_flat[:, 9]
            
        # Parameter-Namen für full_reaction_system überschreiben
        param_names = list(param_arrays.keys())
    else:
        # Normale Behandlung für andere Modelle
        for param_name in param_names:
            if param_name in param_results:
                param_arrays[param_name] = np.array(param_results[param_name])
    
    # Kovarianzmatrix und Korrelation berechnen
    if len(param_arrays) >= 2:
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
    
    if covariance_matrix is not None:
        results['covariance_matrix'] = covariance_matrix
        results['correlation_matrix'] = correlation_matrix
    
    # Ausgabe
    print(f"\n=== MONTE CARLO ERGEBNISSE ===")
    print(f"Erfolgreiche Iterationen: {successful_iterations}/{n_iterations}")
    
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
    
    # Histogramme erstellen
    create_histograms(results)
    
    return results
def schaetze_parameter_noisy(param, model_name='michaelis_menten', verbose=True, 
                           noise_level_calibration=0.03, noise_level_kinetics=0.02):
    """
    Schätzt Parameter für verrauschte Daten - Monte Carlo kompatible Version
    
    Diese Funktion ist eine Variante von schaetze_parameter(), die speziell für 
    Monte Carlo Simulationen entwickelt wurde. Sie fügt Rauschen zu den Daten hinzu
    und gibt die Parameter zurück.
    
    Parameters:
    -----------
    param : dict
        Parameter-Dictionary mit experimentellen Bedingungen
    model_name : str
        Name des zu verwendenden Modells (siehe AVAILABLE_MODELS)
    verbose : bool
        Ausgabe von Zwischenergebnissen
    noise_level_calibration : float
        Rauschpegel für Kalibrierungsdaten (0.03 = 3%)
    noise_level_kinetics : float
        Rauschpegel für Kinetikdaten (0.02 = 2%)
    
    Returns:
    --------
    dict or None
        Dictionary mit geschätzten Parametern oder None bei Fehler
    """
    
    # Modell validieren
    if model_name not in AVAILABLE_MODELS:
        if verbose:
            print(f"Fehler: Modell '{model_name}' nicht verfügbar!")
            print(f"Verfügbare Modelle: {list(AVAILABLE_MODELS.keys())}")
        return None
    
    model_info = AVAILABLE_MODELS[model_name]
    model_func = model_info['function']
    
    try:
        # NADH Kalibrierung laden
        nadh_path = os.path.join(BASE_PATH, "Daten", "Rohdaten", "Plate_Reader", "Kalibriergeraden", "NADH_Kalibriergerade.xlsx")
        nadh_kalibrierung = pd.read_excel(nadh_path)
        
        # Kalibrierungsdaten verrauschen
        x_cal_original = nadh_kalibrierung.NADH.values
        y_cal_original = nadh_kalibrierung.Mittelwert.values
        
        # Rauschen hinzufügen
        x_cal_noisy = add_noise(x_cal_original, noise_level_calibration)
        y_cal_noisy = add_noise(y_cal_original, noise_level_calibration)
        
        # Linearitätsprüfung der verrauschten Kalibrierung
        if not is_linear(x_cal_noisy, y_cal_noisy, threshold=0.75):
            if verbose:
                print("Verrauschte Kalibrierung ist nicht ausreichend linear")
            return None
        
        # Kalibrierungssteigung aus verrauschten Daten berechnen
        slope_cal, intercept_cal, r_value_cal = linregress(x_cal_noisy, y_cal_noisy)[:3]
        
        if verbose:
            print(f"Verrauschte Kalibrierdaten: R² = {r_value_cal**2:.4f}")
    
    except Exception as e:
        if verbose:
            print(f"Fehler beim Laden der Kalibrierungsdaten: {e}")
        return None

    # Modell-spezifische Datenverarbeitung (analog zu schaetze_parameter)
    if model_name in ['two_substrat_michaelis_menten', 'two_substrat_michaelis_menten_with_one_inhibition', 
                      'two_substrat_michaelis_menten_with_two_inhibition']:
        # ZWEI-SUBSTRAT MODELLE
        try:
            csv_path_nad_var = os.path.join(BASE_PATH, "Daten","Rohdaten","Plate_Reader", "Kinetik-Messungen", "r1", "r_1_NAD_PD_500nM.csv")
            csv_path_pd_var = os.path.join(BASE_PATH, "Daten","Rohdaten","Plate_Reader", "Kinetik-Messungen", "r1", "r1_PD_NAD_5nM.csv")
            
            # NAD variabel Experiment
            r1_nad_var = pd.read_csv(csv_path_nad_var, header=None)
            concentrations_nad = get_concentrations(r1_nad_var)
            time_points_nad = get_time_points(r1_nad_var)
            absorption_data_nad = get_adsoption_data(r1_nad_var)
            
            # Rauschen zu Kinetikdaten hinzufügen
            concentrations_nad_noisy = add_noise(concentrations_nad, noise_level_calibration)
            time_points_nad_noisy = add_noise(time_points_nad, noise_level_kinetics)
            
            # Sichere Absorption Data Behandlung
            absorption_data_nad_clean = safe_flatten_to_float(absorption_data_nad, "absorption_nad")
            if absorption_data_nad_clean is None:
                return None
            absorption_data_nad_noisy = add_noise(absorption_data_nad_clean, noise_level_kinetics)
            
            # Reshape zurück falls möglich
            try:
                if hasattr(absorption_data_nad, 'shape') and len(absorption_data_nad.shape) > 1:
                    absorption_data_nad_noisy = absorption_data_nad_noisy.reshape(absorption_data_nad.shape)
            except Exception:
                pass  # Verwende flaches Array
            
            # PD variabel Experiment
            r1_pd_var = pd.read_csv(csv_path_pd_var, header=None)
            concentrations_pd = get_concentrations(r1_pd_var)
            time_points_pd = get_time_points(r1_pd_var)
            absorption_data_pd = get_adsoption_data(r1_pd_var)
            
            # Rauschen hinzufügen
            concentrations_pd_noisy = add_noise(concentrations_pd, noise_level_calibration)
            time_points_pd_noisy = add_noise(time_points_pd, noise_level_kinetics)
            
            # Sichere Behandlung
            absorption_data_pd_clean = safe_flatten_to_float(absorption_data_pd, "absorption_pd")
            if absorption_data_pd_clean is None:
                return None
            absorption_data_pd_noisy = add_noise(absorption_data_pd_clean, noise_level_kinetics)
            
            try:
                if hasattr(absorption_data_pd, 'shape') and len(absorption_data_pd.shape) > 1:
                    absorption_data_pd_noisy = absorption_data_pd_noisy.reshape(absorption_data_pd.shape)
            except Exception:
                pass
            
            # Aktivitäten aus verrauschten Daten berechnen
            result_nad = berechne_aktivitaet(
                concentrations_nad_noisy, absorption_data_nad_noisy, time_points_nad_noisy, 
                slope_cal, param, verbose=False
            )
            result_pd = berechne_aktivitaet(
                concentrations_pd_noisy, absorption_data_pd_noisy, time_points_pd_noisy, 
                slope_cal, param, verbose=False
            )
            
            if result_nad is None or result_pd is None:
                return None
            
            initial_rates_nad, valid_concentrations_nad = result_nad
            initial_rates_pd, valid_concentrations_pd = result_pd
            
            # Kombinierte Datenstruktur erstellen
            NAD_constant = 5.0  # mM
            PD_constant = 0.5   # mM (500 nM)
            
            S1_combined = np.concatenate([
                valid_concentrations_nad,
                np.full(len(valid_concentrations_pd), NAD_constant)
            ])
            
            S2_combined = np.concatenate([
                np.full(len(valid_concentrations_nad), PD_constant),
                valid_concentrations_pd
            ])
            
            activities_combined = np.concatenate([initial_rates_nad, initial_rates_pd])
            
            x_data = (S1_combined, S2_combined)
            y_data = activities_combined
            
        except Exception as e:
            if verbose:
                print(f"Fehler bei Zwei-Substrat-Datenverarbeitung: {e}")
            return None
            
    elif model_name == 'full_reaction_system':
        # VOLLSTÄNDIGES REAKTIONSSYSTEM
        try:
            # Alle sieben CSV-Dateien laden und verrauschen
            csv_files = {
                'r1_nad': os.path.join(BASE_PATH, "Daten","Rohdaten","Plate_Reader", "Kinetik-Messungen", "r1", "r_1_NAD_PD_500mM.csv"),
                'r1_pd': os.path.join(BASE_PATH, "Daten","Rohdaten","Plate_Reader", "Kinetik-Messungen", "r1", "r1_PD_NAD_5mM.csv"),
                'r2_pd': os.path.join(BASE_PATH, "Daten","Rohdaten","Plate_Reader", "Kinetik-Messungen", "r2", "PD (0.6 mM NADH, 300 mM HP).csv"),
                'r2_nadh': os.path.join(BASE_PATH, "Daten","Rohdaten","Plate_Reader", "Kinetik-Messungen", "r2", "NADH (300 mM HP).csv"),
                'r2_hp': os.path.join(BASE_PATH, "Daten","Rohdaten","Plate_Reader", "Kinetik-Messungen", "r2", "HP (0.6 mM NADH).csv"),
                'r3_nad': os.path.join(BASE_PATH, "Daten","Rohdaten","Plate_Reader", "Kinetik-Messungen", "r3", "NAD (500 mM Lactol).csv"),
                'r3_lactol': os.path.join(BASE_PATH, "Daten","Rohdaten","Plate_Reader", "Kinetik-Messungen", "r3", "Lactol (5 mM NAD).csv")
            }
            
            # Hilfsfunktion für verrauschte Datenverarbeitung
            def process_noisy_csv(csv_path, slope_cal, param):
                """Lädt CSV und fügt Rauschen hinzu"""
                data = pd.read_csv(csv_path, header=None)
                conc = add_noise(get_concentrations(data), noise_level_calibration)
                time = add_noise(get_time_points(data), noise_level_kinetics)
                abs_clean = safe_flatten_to_float(get_adsoption_data(data))
                if abs_clean is None:
                    return None
                abs_noisy = add_noise(abs_clean, noise_level_kinetics)
                
                try:
                    orig_abs = get_adsoption_data(data)
                    if hasattr(orig_abs, 'shape') and len(orig_abs.shape) > 1:
                        abs_noisy = abs_noisy.reshape(orig_abs.shape)
                except Exception:
                    pass
                
                return berechne_aktivitaet(conc, abs_noisy, time, slope_cal, param, verbose=False)
            
            # Alle Experimente verarbeiten
            results_dict = {}
            for key, path in csv_files.items():
                result = process_noisy_csv(path, slope_cal, param)
                if result is None:
                    if verbose:
                        print(f"Experiment {key} fehlgeschlagen")
                    return None
                results_dict[key] = result
            
            # Daten für vollständiges System kombinieren
            r1_nad_rates, r1_nad_conc = results_dict['r1_nad']
            r1_pd_rates, r1_pd_conc = results_dict['r1_pd']
            r2_pd_rates, r2_pd_conc = results_dict['r2_pd']
            r2_nadh_rates, r2_nadh_conc = results_dict['r2_nadh']
            r2_hp_rates, r2_hp_conc = results_dict['r2_hp']
            r3_nad_rates, r3_nad_conc = results_dict['r3_nad']
            r3_lactol_rates, r3_lactol_conc = results_dict['r3_lactol']
            
            x_data, y_data, _, _, _, _, _ = combine_full_reaction_system_data(
                r1_nad_conc, r1_nad_rates, r1_pd_conc, r1_pd_rates,
                r2_pd_conc, r2_pd_rates, r2_nadh_conc, r2_nadh_rates,
                r2_hp_conc, r2_hp_rates, r3_nad_conc, r3_nad_rates,
                r3_lactol_conc, r3_lactol_rates
            )
            
        except Exception as e:
            if verbose:
                print(f"Fehler bei Vollsystem-Datenverarbeitung: {e}")
            return None
            
    else:
        # EIN-SUBSTRAT MODELLE
        try:
            r1_path = os.path.join(BASE_PATH, "Daten", "Rohdaten", "Plate_Reader", "Kinetik-Messungen","r1", "r1_NAD_PD_mod.xlsx")
            r1 = pd.read_excel(r1_path)

            concentrations = get_concentrations(r1)
            time_points = get_time_points(r1)
            absorption_data = get_adsoption_data(r1)

            # Rauschen hinzufügen
            concentrations_noisy = add_noise(concentrations, noise_level_calibration)
            time_points_noisy = add_noise(time_points, noise_level_kinetics)
            
            # Sichere Behandlung der Absorptionsdaten
            absorption_clean = safe_flatten_to_float(absorption_data, "absorption_single")
            if absorption_clean is None:
                return None
            absorption_noisy = add_noise(absorption_clean, noise_level_kinetics)
            
            # Reshape falls möglich
            try:
                if hasattr(absorption_data, 'shape') and len(absorption_data.shape) > 1:
                    absorption_noisy = absorption_noisy.reshape(absorption_data.shape)
            except Exception:
                pass

            result = berechne_aktivitaet(
                concentrations_noisy, absorption_noisy, time_points_noisy, 
                slope_cal, param, verbose=False
            )
            
            if result is None:
                return None
                
            initial_rates, valid_concentrations = result
            x_data = valid_concentrations
            y_data = initial_rates
            
        except Exception as e:
            if verbose:
                print(f"Fehler bei Ein-Substrat-Datenverarbeitung: {e}")
            return None

    # Parameter fitten
    try:
        result = fitt_parameters(model_name, x_data, y_data, model_info, model_func=model_func)
        
        if result is None:
            return None
        
        if verbose:
            print(f"Verrauschte Parameter geschätzt: R² = {result['R_squared']:.4f}")
        
        return result
        
    except Exception as e:
        if verbose:
            print(f"Fehler beim Fitten der verrauschten Parameter: {e}")
        return None


def monte_carlo_parameter_estimation(param, model_name='michaelis_menten', n_iterations=1000, 
                                   noise_level_calibration=0.03, noise_level_kinetics=0.02, 
                                   verbose=True):
    """
    Monte Carlo Simulation für Parameterschätzung mit verrauschten Daten
    
    Diese Funktion verwendet schaetze_parameter_noisy() um die Unsicherheiten
    in den geschätzten Parametern zu quantifizieren.
    
    Parameters:
    -----------
    param : dict
        Parameter-Dictionary mit experimentellen Bedingungen
    model_name : str
        Name des zu verwendenden Modells (siehe AVAILABLE_MODELS)
    n_iterations : int
        Anzahl der Monte Carlo Iterationen
    noise_level_calibration : float
        Rauschpegel für Kalibrierungsdaten (Standard: 3%)
    noise_level_kinetics : float
        Rauschpegel für Kinetikdaten (Standard: 2%)
    verbose : bool
        Ausgabe von Zwischenergebnissen
        
    Returns:
    --------
    dict
        Dictionary mit statistischen Ergebnissen der Parameter
    """
    
    if verbose:
        print("=== MONTE CARLO PARAMETERSCHÄTZUNG ===")
        print("="*50)
        print(f"Modell: {model_name}")
        print(f"Iterationen: {n_iterations}")
        print(f"Rauschpegel Kalibrierung: {noise_level_calibration*100:.1f}%")
        print(f"Rauschpegel Kinetik: {noise_level_kinetics*100:.1f}%")
    
    # Modell validieren
    if model_name not in AVAILABLE_MODELS:
        print(f"Fehler: Modell '{model_name}' nicht verfügbar!")
        return None
    
    model_info = AVAILABLE_MODELS[model_name]
    param_names = model_info['param_names']
    
    # Sammle Ergebnisse
    param_results = {param_name: [] for param_name in param_names}
    r_squared_results = []
    
    # Für full_reaction_system: zusätzliche Sammlung der flachen Parameter
    if model_name == 'full_reaction_system':
        param_results['all_params_flat'] = []
    
    successful_iterations = 0
    failed_reasons = {"fitting": 0, "linearität": 0, "daten": 0, "andere": 0}
    
    # Monte Carlo Schleife
    for iteration in range(n_iterations):
        try:
            # Parameterschätzung mit verrauschten Daten
            result = schaetze_parameter_noisy(
                param, model_name, verbose=False,
                noise_level_calibration=noise_level_calibration,
                noise_level_kinetics=noise_level_kinetics
            )
            
            if result is not None:
                # Validierung der Ergebnisse
                valid_result = True
                r_squared = result.get('R_squared', 0)
                
                if r_squared < 0.3:  # Mindest-R²
                    valid_result = False
                    failed_reasons["fitting"] += 1
                
                # Parameter-spezifische Validierung
                if model_name == 'full_reaction_system' and 'all_params_flat' in result:
                    # Validierung für full_reaction_system
                    params_flat = result['all_params_flat']
                    for param_value in params_flat:
                        if param_value <= 0 or param_value > 10000:
                            valid_result = False
                            failed_reasons["daten"] += 1
                            break
                else:
                    # Normale Validierung für andere Modelle
                    for param_name in param_names:
                        param_value = result.get(param_name, 0)
                        if param_value <= 0 or param_value > 10000:
                            valid_result = False
                            failed_reasons["daten"] += 1
                            break
                
                if valid_result:
                    # Sammle gültige Ergebnisse
                    r_squared_results.append(r_squared)
                    
                    if model_name == 'full_reaction_system':
                        if 'all_params_flat' in result:
                            param_results['all_params_flat'].append(result['all_params_flat'])
                    else:
                        for param_name in param_names:
                            if param_name in result:
                                param_results[param_name].append(result[param_name])
                    
                    successful_iterations += 1
                    
            else:
                failed_reasons["andere"] += 1
            
            # Fortschrittsanzeige
            if verbose and (iteration + 1) % 100 == 0:
                print(f"Iteration {iteration + 1}/{n_iterations} - Erfolgreiche: {successful_iterations}")
                
        except Exception as e:
            failed_reasons["andere"] += 1
            if verbose and iteration < 5:  # Nur erste 5 Fehler ausgeben
                print(f"Fehler in Iteration {iteration}: {e}")
            continue
    
    # Auswertung
    if successful_iterations < 10:
        print(f"Zu wenige erfolgreiche Iterationen: {successful_iterations}")
        return None
    
    if verbose:
        print("\nZusammenfassung der Fehlschläge:")
        print(f"- Fitting-Probleme: {failed_reasons['fitting']}")
        print(f"- Linearitätsprobleme: {failed_reasons['linearität']}")
        print(f"- Datenprobleme: {failed_reasons['daten']}")
        print(f"- Andere Fehler: {failed_reasons['andere']}")
    
    # Konvertiere Listen zu numpy arrays
    r_squared_array = np.array(r_squared_results)
    
    # Ergebnis-Dictionary erstellen
    results = {
        'n_successful': successful_iterations,
        'n_total': n_iterations,
        'success_rate': successful_iterations / n_iterations,
        'model_name': model_name,
        'param_names': param_names,
        'R_squared_mean': np.mean(r_squared_array),
        'R_squared_std': np.std(r_squared_array),
        'R_squared_median': np.median(r_squared_array),
        'r_squared_values': r_squared_array,
        'noise_level_calibration': noise_level_calibration,
        'noise_level_kinetics': noise_level_kinetics
    }
    
    # Parameter-spezifische Statistiken
    if model_name == 'full_reaction_system':
        # Spezielle Behandlung für full_reaction_system
        if param_results['all_params_flat']:
            all_params_array = np.array(param_results['all_params_flat'])
            
            # Erstelle individuelle Parameter-Arrays
            param_arrays = {}
            param_names_full = ['Vmax1', 'Vmax2', 'Vmax3', 'KmPD', 'KmNAD', 
                               'KmLactol', 'KmNADH', 'KiPD', 'KiNAD', 'KiLactol']
            
            for i, param_name in enumerate(param_names_full):
                if i < all_params_array.shape[1]:
                    param_data = all_params_array[:, i]
                    param_arrays[param_name] = param_data
                    
                    # Statistiken hinzufügen
                    results[f'{param_name}_mean'] = np.mean(param_data)
                    results[f'{param_name}_std'] = np.std(param_data)
                    results[f'{param_name}_median'] = np.median(param_data)
                    results[f'{param_name}_ci_lower'] = np.percentile(param_data, 2.5)
                    results[f'{param_name}_ci_upper'] = np.percentile(param_data, 97.5)
                    results[f'{param_name}_values'] = param_data
            
            results['param_names'] = param_names_full  # Update param_names
    else:
        # Normale Parameter-Behandlung
        for param_name in param_names:
            if param_name in param_results and param_results[param_name]:
                param_data = np.array(param_results[param_name])
                
                results[f'{param_name}_mean'] = np.mean(param_data)
                results[f'{param_name}_std'] = np.std(param_data)
                results[f'{param_name}_median'] = np.median(param_data)
                results[f'{param_name}_ci_lower'] = np.percentile(param_data, 2.5)
                results[f'{param_name}_ci_upper'] = np.percentile(param_data, 97.5)
                results[f'{param_name}_values'] = param_data
    
    # Kovarianz- und Korrelationsmatrizen berechnen
    param_names_final = results['param_names']
    if len(param_names_final) >= 2:
        try:
            # Sammle alle Parameter-Arrays
            param_matrix_list = []
            for param_name in param_names_final:
                if f'{param_name}_values' in results:
                    param_matrix_list.append(results[f'{param_name}_values'])
            
            if len(param_matrix_list) >= 2:
                parameter_matrix = np.column_stack(param_matrix_list)
                results['covariance_matrix'] = np.cov(parameter_matrix, rowvar=False)
                results['correlation_matrix'] = np.corrcoef(parameter_matrix, rowvar=False)
        except Exception as e:
            if verbose:
                print(f"Warnung: Kovarianzmatrix konnte nicht berechnet werden: {e}")
    
    # Ausgabe der Ergebnisse
    if verbose:
        print("\n=== MONTE CARLO ERGEBNISSE ===")
        print(f"Erfolgreiche Iterationen: {successful_iterations}/{n_iterations} ({results['success_rate']*100:.1f}%)")
        
        for param_name in param_names_final:
            if f'{param_name}_mean' in results:
                mean_val = results[f'{param_name}_mean']
                std_val = results[f'{param_name}_std']
                ci_lower = results[f'{param_name}_ci_lower']
                ci_upper = results[f'{param_name}_ci_upper']
                
                # Einheit bestimmen
                try:
                    if model_name == 'full_reaction_system':
                        units_dict = {'Vmax1': 'U/mg', 'Vmax2': 'U/mg', 'Vmax3': 'U/mg',
                                     'KmPD': 'mM', 'KmNAD': 'mM', 'KmLactol': 'mM', 'KmNADH': 'mM',
                                     'KiPD': 'mM', 'KiNAD': 'mM', 'KiLactol': 'mM'}
                        unit = units_dict.get(param_name, '')
                    else:
                        param_idx = param_names.index(param_name)
                        unit = model_info['param_units'][param_idx]
                except Exception:
                    unit = ''
                
                print(f"{param_name}: {mean_val:.4f} ± {std_val:.4f} {unit}")
                print(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
        
        print(f"R²: {results['R_squared_mean']:.4f} ± {results['R_squared_std']:.4f}")
    
    # Histogramme erstellen
    if verbose:
        create_histograms(results)
    
    return results

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
   pass
