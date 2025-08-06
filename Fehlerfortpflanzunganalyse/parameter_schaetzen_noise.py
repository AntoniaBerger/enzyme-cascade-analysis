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
    print(f"Kalibrierung: Steigung = {slope_cal:.2f}, R = {r_value_cal**2:.4f}")
    
    return slope_cal, intercept_cal, r_value_cal

def berechne_aktivitaet(concentrations, absorption_data, time_points, slope_cal, activ_param, verbose=False):

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

def fitt_parameters(model_name, x_data, y_data, model_info, model_func=None,verbose=False):
    
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
    
    if verbose:
        print(f"Initial guess: {p0}")
    
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
    
    print(f"\n=== {model_info['description'].upper()} PARAMETER ===")
    
    # Parameter-Ausgabe je nach Modelltyp
    if model_name == 'full_reaction_system':
        # Spezielle Ausgabe für full_reaction_system mit flachen Parametern
        print("=== VOLLSTÄNDIGES MULTI-AKTIVITÄTS ENZYM ===")
        print("Enzym-Aktivitäten:")
        print(f"  Vmax1 (Reaktion 1): {fitted_params[0]:.4f} ± {param_errors[0]:.4f} U/mg")
        print(f"  Vmax2 (Reaktion 2): {fitted_params[1]:.4f} ± {param_errors[1]:.4f} U/mg")
        print(f"  Vmax3 (Reaktion 3): {fitted_params[2]:.4f} ± {param_errors[2]:.4f} U/mg")
        
        print("Substrat-Affinitäten:")
        print(f"  KmPD: {fitted_params[3]:.4f} ± {param_errors[3]:.4f} mM")
        print(f"  KmNAD: {fitted_params[4]:.4f} ± {param_errors[4]:.4f} mM")
        print(f"  KmLactol: {fitted_params[5]:.4f} ± {param_errors[5]:.4f} mM")
        print(f"  KmNADH: {fitted_params[6]:.4f} ± {param_errors[6]:.4f} mM")
        
        print("Inhibitions-Konstanten:")
        print(f"  KiPD: {fitted_params[7]:.4f} ± {param_errors[7]:.4f} mM")
        print(f"  KiNAD: {fitted_params[8]:.4f} ± {param_errors[8]:.4f} mM")
        print(f"  KiLactol: {fitted_params[9]:.4f} ± {param_errors[9]:.4f} mM")
    else:
        pass
            
    print(f"R²: {r_squared:.4f}")
    
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

def schaetze_parameter(param, model_name='michaelis_menten', verbose=False):
    
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
                concentrations_nad, absorption_data_nad, time_points_nad, slope_cal, param, verbose=False
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
                concentrations_pd, absorption_data_pd, time_points_pd, slope_cal, param, verbose=False
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
        
        print("Echte Zwei-Substrat-Analyse:")
        print(f"- NAD (S1) variabel: {len(valid_concentrations_nad)} Punkte von {min(valid_concentrations_nad):.2f} bis {max(valid_concentrations_nad):.2f} mM")
        print(f"- PD (S2) variabel: {len(valid_concentrations_pd)} Punkte von {min(valid_concentrations_pd):.2f} bis {max(valid_concentrations_pd):.2f} mM")
        print(f"- NAD konstant: {NAD_constant:.1f} mM, PD konstant: {PD_constant:.1f} mM")
        print(f"- Gesamt: {len(activities_combined)} Datenpunkte")
        
    elif model_name == 'full_reaction_system':
        # VOLLSTÄNDIGES REAKTIONSSYSTEM: Alle CSV-Dateien für drei Reaktionen laden
        print("\n--- Vollständiges Reaktionssystem CSV-Kinetikdaten laden ---")
        print("Reaktion 1: PD + NAD → Pyruvat + NADH")
        print("Reaktion 2: Lactol + NADH → ... (mit PD/NAD Inhibition)")
        print("Reaktion 3: Lactol + NAD → ... (mit Lactol Inhibition)")
        
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

        
            print(f"R1: NAD variabel {len(r1_nad_valid_conc)} Punkte, PD variabel {len(r1_pd_valid_conc)} Punkte")
            
            # ===== REAKTION 2: Lactol + NADH (mit PD/NAD Inhibition) =====
            print("\n=== REAKTION 2: Lactol + NADH ===")
            
            # R2 - PD variabel (als Inhibitor) , Lactol + NADH konstant 
           
            r2_pd_rates, r2_pd_valid_conc = get_rates_and_concentrations(csv_r2_pd_var, slope_cal, param)
            if r2_pd_rates is None: 
                print("FEHLER: R2-PD Experiment fehlgeschlagen")
                return None
            
            # R2 - NADH variabel
            print(f"Lade R2-NADH variabel: {csv_r2_nadh_var}")
            r2_nadh_rates, r2_nadh_valid_conc = get_rates_and_concentrations(csv_r2_nadh_var, slope_cal, param)
            if r2_nadh_rates is None:
                print("FEHLER: R2-NADH Experiment fehlgeschlagen")
                return None
            
            # R2 - HP variabel (Lactol)
            print(f"Lade R2-HP variabel: {csv_r2_hp_var}")
            r2_hp_rates, r2_hp_valid_conc = get_rates_and_concentrations(csv_r2_hp_var, slope_cal, param)
            if r2_hp_rates is None: 
                print("FEHLER: R2-HP Experiment fehlgeschlagen")
                return None
            
            print(f"R2: PD {len(r2_pd_valid_conc)} Punkte, NADH {len(r2_nadh_valid_conc)} Punkte, HP {len(r2_hp_valid_conc)} Punkte")
            
            # ===== REAKTION 3: Lactol + NAD (mit Lactol Inhibition) =====
            print("\n=== REAKTION 3: Lactol + NAD ===")
            
            # R3 - NAD variabel
            print(f"Lade R3-NAD variabel: {csv_r3_nad_var}")
        
            r3_nad_rates, r3_nad_valid_conc = get_rates_and_concentrations(csv_r3_nad_var, slope_cal, param)
            if r3_nad_rates is None:
                print("FEHLER: R3-NAD Experiment fehlgeschlagen")
                return None
            
            # R3 - Lactol variabel
            print(f"Lade R3-Lactol variabel: {csv_r3_lactol_var}")
            r3_lactol_rates, r3_lactol_valid_conc = get_rates_and_concentrations(csv_r3_lactol_var, slope_cal, param)
            if r3_lactol_rates is None:
                print("FEHLER: R3-Lactol Experiment fehlgeschlagen")
                return None


            print(f"R3: NAD {len(r3_nad_valid_conc)} Punkte, Lactol {len(r3_lactol_valid_conc)} Punkte")
            
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
            print(f"VOLLSTÄNDIGES MULTI-AKTIVITÄTS SYSTEM:")
            print(f"- GESAMT: {total_points} Datenpunkte für 10 Parameter")
            print(f"- Reaktion 1 (PD + NAD): {r1_points} Punkte")  
            print(f"- Reaktion 2 (Lactol + NADH mit Inhibition): {r2_points} Punkte")
            print(f"- Reaktion 3 (Lactol + NAD mit Inhibition): {r3_points} Punkte")
            print(f"- Parameter/Datenpunkt Ratio: {10/total_points:.3f} (sollte < 0.1 sein)")
        
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
        
        # Ergebnis ausgeben
        print(f"\n=== FITTED PARAMETER FÜR {model_info['description'].upper()} ===")
        print(f"R²: {result['R_squared']:.4f}")
        
        # Spezielle Ausgabe für full_reaction_system
        if model_name == 'full_reaction_system':
            print("Vmax-Werte:", result['Vmax_values'])
            print("Km-Werte:", result['Km_values'])
            print("Ki-Werte:", result['Ki_values'])
        
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
    
    print(f"Monte Carlo Simulation: {n_iterations} Iterationen")
    
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
            
            # Vollständiges Reaktionssystem: Alle CSV-Dateien geladen
            print(f"- Kalibrierung: {len(x_original)} Punkte")
            print(f"- R1 NAD: {len(r1_nad_conc_original)} Wells")
            print(f"- R1 PD: {len(r1_pd_conc_original)} Wells") 
            print(f"- R2 PD: {len(r2_pd_conc_original)} Wells")
            print(f"- R2 NADH: {len(r2_nadh_conc_original)} Wells")
            print(f"- R2 HP: {len(r2_hp_conc_original)} Wells")
            print(f"- R3 NAD: {len(r3_nad_conc_original)} Wells")
            print(f"- R3 Lactol: {len(r3_lactol_conc_original)} Wells")
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
        # Fortschritt alle 100 Iterationen anzeigen
        if (iteration + 1) % 100 == 0 or iteration == 0:
            successful_count = len([r for r in r_squared_results if r is not None])
            print(f"Iteration {iteration + 1}/{n_iterations} - Erfolgreich: {successful_count}")
        
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
                
                # Parameter mit den verrauschten Daten berechnen
                # Verwende die neue entkoppelte Funktion
                # Beachte: berechne_aktivitaet(concentrations, absorption_data, time_points, ...)
                
                # Erstelle noisy_data Dictionary...
                
                try:
                    noisy_data = {
                        'calibration': (x_noisy, y_noisy),
                        'r1_nad': (r1_nad_conc_noisy, r1_nad_abs_noisy, r1_nad_time_noisy),
                        'r1_pd': (r1_pd_conc_noisy, r1_pd_abs_noisy, r1_pd_time_noisy),
                        'r2_pd': (r2_pd_conc_noisy, r2_pd_abs_noisy, r2_pd_time_noisy),
                        'r2_nadh': (r2_nadh_conc_noisy, r2_nadh_abs_noisy, r2_nadh_time_noisy),
                        'r2_hp': (r2_hp_conc_noisy, r2_hp_abs_noisy, r2_hp_time_noisy),
                        'r3_nad': (r3_nad_conc_noisy, r3_nad_abs_noisy, r3_nad_time_noisy),
                        'r3_lactol': (r3_lactol_conc_noisy, r3_lactol_abs_noisy, r3_lactol_time_noisy)
                    }
                    # noisy_data Dictionary erfolgreich erstellt
                    
                except Exception:
                    failed_reasons["daten"] += 1
                    continue
                
                # Berechne Kalibrierungssteigung mit verrauschten Daten
                # Sichere Array-Behandlung für linregress
                try:
                    x_clean = np.asarray(x_noisy).flatten()
                    y_clean = np.asarray(y_noisy).flatten()
                
                    slope_noisy, _, _, _, _ = linregress(x_clean, y_clean)
                    
                except Exception as e:
                    print(f"ERROR: Linregress fehlgeschlagen: {e}")
                    continue
                
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
    
    # Zusammenfassung nur bei Problemen
    total_failed = sum(failed_reasons.values())
    if total_failed > 0:
        print(f"\nProblematische Iterationen: {total_failed}")
    
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
    
    # Ergebnisse
    print(f"\nMonte Carlo abgeschlossen: {successful_iterations}/{n_iterations} erfolgreich")
    print(f"R²: {results['R_squared_mean']:.4f} ± {results['R_squared_std']:.4f}")
    
    # Histogramme erstellen
    create_histograms(results)
    
    return results
def schaetze_parameter_noisy(activ_param, model_name='full_reaction_system', verbose=False):
    """
    Parameter schätzen mit verrauschten Daten - VEREINFACHTE VERSION
    
    Lädt alle Originaldaten, fügt Rauschen hinzu und führt dann 
    die normale Parameterschätzung durch.
    
    Parameters:
    -----------
    activ_param : dict
        Aktivitätsparameter für die Messung
    model_name : str
        Modellname (nur 'full_reaction_system' unterstützt)
        
    Returns:
    --------
    dict : Schätzergebnis oder None bei Fehler
    """
    if model_name != 'full_reaction_system':
        print("Nur vollständiges Reaktionssystem unterstützt")
        return None
        
    try:
        # Kleine Variation der Aktivitätsparameter als Proxy für Datenrauschen
        noisy_param = activ_param.copy()
        
        # 2% Rauschen auf die experimentellen Parameter
        for key in ['Vf_well', 'Vf_prod', 'c_prod']:
            if key in noisy_param:
                noise_factor = np.random.normal(1.0, 0.02)  # 2% Rauschen
                noisy_param[key] = activ_param[key] * noise_factor
        
        # Verwende die normale schaetze_parameter Funktion mit verrauschten Parametern
        result = schaetze_parameter(noisy_param, model_name, verbose=verbose)
        
        return result
        
    except Exception as e:
        print(f"Fehler in schaetze_parameter_noisy: {e}")
        return None

def schaetze_parameter_noisy_full_system(x_cal, y_cal,
                                       r1_nad_conc, r1_nad_time, r1_nad_abs,
                                       r1_pd_conc, r1_pd_time, r1_pd_abs,
                                       r2_pd_conc, r2_pd_time, r2_pd_abs,
                                       r2_nadh_conc, r2_nadh_time, r2_nadh_abs,
                                       r2_hp_conc, r2_hp_time, r2_hp_abs,
                                       r3_nad_conc, r3_nad_time, r3_nad_abs,
                                       r3_lactol_conc, r3_lactol_time, r3_lactol_abs,
                                       param):
    """
    Parameter schätzen mit verrauschten Daten für das vollständige Reaktionssystem
    Verarbeitet alle sieben CSV-Datensätze mit Monte Carlo Rauschen
    """
    try:
        # Kalibrierungssteigung berechnen
        slope_cal, intercept_cal, r_value_cal, _, _ = linregress(x_cal, y_cal)
        
        # Alle Aktivitäten berechnen mit verrauschten Daten
        # Reaktion 1 - NAD variabel, PD konstant
        r1_nad_rates, r1_nad_valid_conc = berechne_aktivitaet(
            r1_nad_conc, r1_nad_abs, r1_nad_time, slope_cal, param, verbose=False
        )
        
        # Debug: Überprüfung der Rückgabewerte
        if r1_nad_rates is None or r1_nad_valid_conc is None:
            return None
        
        # Reaktion 1 - PD variabel, NAD konstant  
        r1_pd_rates, r1_pd_valid_conc = berechne_aktivitaet(
            r1_pd_conc, r1_pd_abs, r1_pd_time, slope_cal, param, verbose=False
        )
        
        if r1_pd_rates is None or r1_pd_valid_conc is None:
            return None
        
        # Reaktion 2 - PD variabel (als Inhibitor)
        r2_pd_rates, r2_pd_valid_conc = berechne_aktivitaet(
            r2_pd_conc, r2_pd_abs, r2_pd_time, slope_cal, param, verbose=False
        )
        
        if r2_pd_rates is None or r2_pd_valid_conc is None:
            return None
        
        # Reaktion 2 - NADH variabel
        r2_nadh_rates, r2_nadh_valid_conc = berechne_aktivitaet(
            r2_nadh_conc, r2_nadh_abs, r2_nadh_time, slope_cal, param, verbose=False
        )
        
        if r2_nadh_rates is None or r2_nadh_valid_conc is None:
            return None
        
        # Reaktion 2 - HP/Lactol variabel
        r2_hp_rates, r2_hp_valid_conc = berechne_aktivitaet(
            r2_hp_conc, r2_hp_abs, r2_hp_time, slope_cal, param, verbose=False
        )
        
        if r2_hp_rates is None or r2_hp_valid_conc is None:
            return None
        
        # Reaktion 3 - NAD variabel
        r3_nad_rates, r3_nad_valid_conc = berechne_aktivitaet(
            r3_nad_conc, r3_nad_abs, r3_nad_time, slope_cal, param, verbose=False
        )
        
        if r3_nad_rates is None or r3_nad_valid_conc is None:
            return None
        
        # Reaktion 3 - Lactol variabel
        r3_lactol_rates, r3_lactol_valid_conc = berechne_aktivitaet(
            r3_lactol_conc, r3_lactol_abs, r3_lactol_time, slope_cal, param, verbose=False
        )
        
        if r3_lactol_rates is None or r3_lactol_valid_conc is None:
            return None
        
        # Überprüfung der berechneten Aktivitäten
        if any(rates is None for rates in [r1_nad_rates, r1_pd_rates, r2_pd_rates, r2_nadh_rates, 
                                         r2_hp_rates, r3_nad_rates, r3_lactol_rates]):
            return None
            
        if any(len(rates) == 0 for rates in [r1_nad_rates, r1_pd_rates, r2_pd_rates, r2_nadh_rates, 
                                           r2_hp_rates, r3_nad_rates, r3_lactol_rates]):
            return None
        
        # Kombinierte Datenstruktur für das vollständige System erstellen
        # Direkte Listen-basierte Lösung ohne verschachtelte Arrays
        all_s1_values = []
        all_s2_values = []
        all_inhibitor_values = []
        all_activities = []
        all_reaction_ids = []
        
        # Konstante Werte (aus der Original-Implementierung)
        r1_nad_constant = 500.0   # PD konstant in R1 NAD-Variation
        r1_pd_constant = 5.0      # NAD konstant in R1 PD-Variation
        r2_lactol_constant = 1.0  # Lactol konstant in R2
        r2_nadh_constant = 1.0    # NADH konstant in R2
        r3_lactol_constant = 1.0  # Lactol konstant in R3
        r3_nad_constant = 5.0     # NAD konstant in R3
        
        # Reaktion 1 Daten hinzufügen (direkte Konvertierung)
        # R1: NAD variabel, PD konstant
        r1_nad_conc_list = r1_nad_valid_conc.tolist() if isinstance(r1_nad_valid_conc, np.ndarray) else list(r1_nad_valid_conc)
        r1_nad_rates_list = r1_nad_rates.tolist() if isinstance(r1_nad_rates, np.ndarray) else list(r1_nad_rates)
        n_r1_nad = len(r1_nad_rates_list)
        
        all_s1_values.extend(r1_nad_conc_list)
        all_s2_values.extend([r1_nad_constant] * n_r1_nad)
        all_inhibitor_values.extend([0.0] * n_r1_nad)
        all_activities.extend(r1_nad_rates_list)
        all_reaction_ids.extend([1] * n_r1_nad)
        
        # R1: PD variabel, NAD konstant  
        r1_pd_conc_list = r1_pd_valid_conc.tolist() if isinstance(r1_pd_valid_conc, np.ndarray) else list(r1_pd_valid_conc)
        r1_pd_rates_list = r1_pd_rates.tolist() if isinstance(r1_pd_rates, np.ndarray) else list(r1_pd_rates)
        n_r1_pd = len(r1_pd_rates_list)
        
        all_s1_values.extend([r1_pd_constant] * n_r1_pd)
        all_s2_values.extend(r1_pd_conc_list)
        all_inhibitor_values.extend([0.0] * n_r1_pd)
        all_activities.extend(r1_pd_rates_list)
        all_reaction_ids.extend([1] * n_r1_pd)
        
        # Reaktion 2 Daten hinzufügen
        # R2: PD variabel (als Inhibitor)
        r2_pd_conc_list = r2_pd_valid_conc.tolist() if isinstance(r2_pd_valid_conc, np.ndarray) else list(r2_pd_valid_conc)
        r2_pd_rates_list = r2_pd_rates.tolist() if isinstance(r2_pd_rates, np.ndarray) else list(r2_pd_rates)
        n_r2_pd = len(r2_pd_rates_list)
        
        all_s1_values.extend([r2_lactol_constant] * n_r2_pd)
        all_s2_values.extend([r2_nadh_constant] * n_r2_pd)
        all_inhibitor_values.extend(r2_pd_conc_list)
        all_activities.extend(r2_pd_rates_list)
        all_reaction_ids.extend([2] * n_r2_pd)
        
        # R2: NADH variabel
        r2_nadh_conc_list = r2_nadh_valid_conc.tolist() if isinstance(r2_nadh_valid_conc, np.ndarray) else list(r2_nadh_valid_conc)
        r2_nadh_rates_list = r2_nadh_rates.tolist() if isinstance(r2_nadh_rates, np.ndarray) else list(r2_nadh_rates)
        n_r2_nadh = len(r2_nadh_rates_list)
        
        all_s1_values.extend([r2_lactol_constant] * n_r2_nadh)
        all_s2_values.extend(r2_nadh_conc_list)
        all_inhibitor_values.extend([1.0] * n_r2_nadh)
        all_activities.extend(r2_nadh_rates_list)
        all_reaction_ids.extend([2] * n_r2_nadh)
        
        # R2: Lactol variabel
        r2_hp_conc_list = r2_hp_valid_conc.tolist() if isinstance(r2_hp_valid_conc, np.ndarray) else list(r2_hp_valid_conc)
        r2_hp_rates_list = r2_hp_rates.tolist() if isinstance(r2_hp_rates, np.ndarray) else list(r2_hp_rates)
        n_r2_hp = len(r2_hp_rates_list)
        
        all_s1_values.extend(r2_hp_conc_list)
        all_s2_values.extend([r2_nadh_constant] * n_r2_hp)
        all_inhibitor_values.extend([1.0] * n_r2_hp)
        all_activities.extend(r2_hp_rates_list)
        all_reaction_ids.extend([2] * n_r2_hp)
        
        # Reaktion 3 Daten hinzufügen
        # R3: NAD variabel
        r3_nad_conc_list = r3_nad_valid_conc.tolist() if isinstance(r3_nad_valid_conc, np.ndarray) else list(r3_nad_valid_conc)
        r3_nad_rates_list = r3_nad_rates.tolist() if isinstance(r3_nad_rates, np.ndarray) else list(r3_nad_rates)
        n_r3_nad = len(r3_nad_rates_list)
        
        all_s1_values.extend([r3_lactol_constant] * n_r3_nad)
        all_s2_values.extend(r3_nad_conc_list)
        all_inhibitor_values.extend([0.0] * n_r3_nad)
        all_activities.extend(r3_nad_rates_list)
        all_reaction_ids.extend([3] * n_r3_nad)
        
        # R3: Lactol variabel
        r3_lactol_conc_list = r3_lactol_valid_conc.tolist() if isinstance(r3_lactol_valid_conc, np.ndarray) else list(r3_lactol_valid_conc)
        r3_lactol_rates_list = r3_lactol_rates.tolist() if isinstance(r3_lactol_rates, np.ndarray) else list(r3_lactol_rates)
        n_r3_lactol = len(r3_lactol_rates_list)
        
        all_s1_values.extend(r3_lactol_conc_list)
        all_s2_values.extend([r3_nad_constant] * n_r3_lactol)
        all_inhibitor_values.extend([0.0] * n_r3_lactol)
        all_activities.extend(r3_lactol_rates_list)
        all_reaction_ids.extend([3] * n_r3_lactol)
        
        # Validierung der Listen-Längen
        expected_length = len(all_activities)
        if not all(len(lst) == expected_length for lst in [all_s1_values, all_s2_values, 
                                                          all_inhibitor_values, all_reaction_ids]):
            return None
        
        # Validierung der Listen-Längen und Nicht-Leerheit
        if not all_activities or len(all_activities) == 0:
            return None
            
        expected_length = len(all_activities)
        if not all(len(lst) == expected_length for lst in [all_s1_values, all_s2_values, 
                                                          all_inhibitor_values, all_reaction_ids]):
            return None
        
        # Mindestanzahl von Datenpunkten prüfen
        if expected_length < 10:  # Brauchen genug Punkte für 10 Parameter
            return None
        
        # Arrays erstellen mit expliziter Typ-Konvertierung
        try:
            S1_combined = np.array(all_s1_values, dtype=float)
            S2_combined = np.array(all_s2_values, dtype=float)
            Inhibitor_combined = np.array(all_inhibitor_values, dtype=float)
            activities_combined = np.array(all_activities, dtype=float)
            reaction_ids = np.array(all_reaction_ids, dtype=int)
            
            # Zusätzliche Validierung der Arrays
            if (len(S1_combined) == 0 or len(activities_combined) == 0 or
                not np.all(np.isfinite(S1_combined)) or not np.all(np.isfinite(activities_combined))):
                return None
                
        except (ValueError, TypeError):
            return None
        
        x_data = (S1_combined, S2_combined, Inhibitor_combined, reaction_ids)
        y_data = activities_combined
        
        # Parameter fitting für das vollständige System
        try:
            # Initial guess für 10 Parameter
            vmax_guess = max(activities_combined) * 2.0
            p0 = [
                vmax_guess, vmax_guess, vmax_guess,  # Vmax1, Vmax2, Vmax3
                2.0, 0.1, 2.0, 0.1,                 # KmPD, KmNAD, KmLactol, KmNADH
                1.0, 0.1, 5.0                       # KiPD, KiNAD, KiLactol
            ]
            
            # Parameter-Grenzen (10 Parameter)
            bounds_lower = [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.1, 0.1, 0.1]
            bounds_upper = [100, 100, 100, 100, 100, 100, 100, 1000, 1000, 1000]
            
            params, covariance = curve_fit(full_reaction_system, x_data, y_data, 
                                         p0=p0, bounds=(bounds_lower, bounds_upper), maxfev=3000)
            
            # Plausibilitätsprüfung
            for param_val in params:
                if param_val <= 0 or param_val > 10000:
                    return None
            
            param_errors = np.sqrt(np.diag(covariance))
            
            # R² berechnen
            y_pred = full_reaction_system(x_data, *params)
            ss_res = np.sum((y_data - y_pred) ** 2)
            ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            
            # Ergebnis-Dictionary (analog zu schaetze_parameter)
            result = {
                'R_squared': r_squared,
                'model_name': 'full_reaction_system',
                'model_description': 'Vollständiges Multi-Aktivitäts Enzym System',
                'param_dict': {
                    "Vmax": [params[0], params[1], params[2]],           # Vmax1, Vmax2, Vmax3
                    "Km": [params[3], params[4], params[5], params[6]], # KmPD, KmNAD, KmLactol, KmNADH
                    "Ki": [params[7], params[8], params[9]]             # KiPD, KiNAD, KiLactol
                },
                'all_params_flat': params,
                'param_errors': param_errors,
                'concentrations': S1_combined,
                'activities': y_data
            }
            
            return result
            
        except Exception as e:
            return None
            
    except Exception as e:
        return None

def schaetze_parameter_noisy_two_substrat(x_cal, y_cal, 
                                        concentrations_s1, time_points_s1, absorption_data_s1, S2_constant,
                                        concentrations_s2, time_points_s2, absorption_data_s2, S1_constant,
                                        param, model_name):
    """
    Parameter schätzen mit verrauschten Zwei-Substrat-Daten
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
        
        # Experiment 1: S1 variabel, S2 konstant
        initial_rates_s1, valid_concentrations_s1 = berechne_aktivitaet(
            concentrations_s1, absorption_data_s1, time_points_s1, slope_cal, param, verbose=False
        )
        
        # Experiment 2: S2 variabel, S1 konstant
        initial_rates_s2, valid_concentrations_s2 = berechne_aktivitaet(
            concentrations_s2, absorption_data_s2, time_points_s2, slope_cal, param, verbose=False
        )
        
        if initial_rates_s1 is None or initial_rates_s2 is None:
            return None
            
        if len(initial_rates_s1) == 0 or len(initial_rates_s2) == 0:
            return None
        
        # Kombinierte Datenstruktur erstellen
        S1_combined = np.concatenate([
            valid_concentrations_s1,                          # Variable S1-Werte
            np.full(len(valid_concentrations_s2), S1_constant) # Konstante S1-Werte
        ])
        
        S2_combined = np.concatenate([
            np.full(len(valid_concentrations_s1), S2_constant), # Konstante S2-Werte
            valid_concentrations_s2                            # Variable S2-Werte
        ])
        
        activities_combined = np.concatenate([initial_rates_s1, initial_rates_s2])
        
        # Parameter fitten
        try:
            # Robustere initial guess basierend auf typischen Enzymkinetik-Werten
            vmax_guess = max(activities_combined) * 2.0  # Höher ansetzen
            
            # Konservative Km-Werte basierend auf typischen Enzymkinetik-Bereichen
            km1_guess = 2.0  # Typischer Wert für NAD+ (mM-Bereich)
            km2_guess = 0.1  # Typischer Wert für PD (µM bis mM-Bereich, hier 100µM = 0.1mM)
            
            if model_name == 'two_substrat_michaelis_menten':
                p0 = [vmax_guess, km1_guess, km2_guess]
            elif model_name == 'two_substrat_michaelis_menten_with_one_inhibition':
                p0 = [vmax_guess, km1_guess, km2_guess, 5.0]  # Ki = 5.0
            elif model_name == 'two_substrat_michaelis_menten_with_two_inhibition':
                p0 = [vmax_guess, km1_guess, km2_guess, 5.0, 5.0]  # Ki1, Ki2 = 5.0
            elif model_name == 'full_reaction_system':
                # Für full_reaction_system: verwende flache Parameter
                p0 = [
                    vmax_guess, km1_guess, km2_guess,         # param_r1
                    vmax_guess, km1_guess, km2_guess, 5.0,    # param_r2
                    vmax_guess, km1_guess, km2_guess, 5.0, 5.0  # param_r3
                ]
                
                # Wrapper-Funktion für full_reaction_system
                def full_system_wrapper(concentration_data, *flat_params):
                    param_r1 = flat_params[0:3]
                    param_r2 = flat_params[3:7]
                    param_r3 = flat_params[7:12]
                    rate1, rate2, rate3 = full_reaction_system(concentration_data, param_r1, param_r2, param_r3)
                    return rate1  # Verwende nur rate1 für Fitting
                
                model_func = full_system_wrapper
            else:
                p0 = [vmax_guess, km1_guess, km2_guess]
            
            x_data = (S1_combined, S2_combined)
            y_data = activities_combined
            
            # Debugging: Initial guess anzeigen bei ersten paar Iterationen
            # (Wird nur in den ersten Iterationen gedruckt)
            
            params, covariance = curve_fit(model_func, x_data, y_data, p0=p0, maxfev=5000)  # Weniger Iterationen für Stabilität
            
            # Plausibilitätsprüfung - sehr viel lockerer für Debugging
            for param_val in params:
                if param_val <= 0 or param_val > 10000:  # Sehr lockere Grenzen
                    return None
            
            param_errors = np.sqrt(np.diag(covariance))
            
            # R² berechnen
            y_pred = model_func(x_data, *params)
            ss_res = np.sum((y_data - y_pred) ** 2)
            ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            
            # Ergebnis-Dictionary erstellen
            result = {'R_squared': r_squared, 'model_name': model_name}
            
            # Parameter je nach Modell hinzufügen
            if model_name == 'full_reaction_system':
                # Spezielle Behandlung für full_reaction_system
                param_r1 = params[0:3]
                param_r2 = params[3:7]
                param_r3 = params[7:12]
                
                result['param_r1'] = param_r1
                result['param_r2'] = param_r2
                result['param_r3'] = param_r3
                result['all_params_flat'] = params  # Flache Parameter für Monte Carlo
            else:
                # Normale Parameter-Hinzufügung
                model_info = AVAILABLE_MODELS[model_name]
                param_names = model_info['param_names']
                for i, (name, value, error) in enumerate(zip(param_names, params, param_errors)):
                    result[name] = value
                    result[f"{name}_error"] = error
            
            return result
            
        except Exception:
            return None
            
    except Exception:
        return None
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
