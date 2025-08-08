import os
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import pickle
from data_hadler import  add_noise_reaction_dict, calculate_calibration, add_noise_calibration, create_concentrations_dict, create_reaction_rates_dict, get_rates_and_concentrations , make_fitting_data



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
    # test if concentration_data has the right shape
    if len(concentration_data) != 4 or len(concentration_data[0]) == 0:
        raise ValueError("concentration_data must contain 4 arrays: S1, S2, Inhibitor, reaction_ids")
    
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

# Setze Funktions-Referenzen in AVAILABLE_MODELS

def fit_parameters(substrate_data, activities, model_info, verbose=True):
    """
    Allgemeine Fitting-Funktion mit modularer Datenstruktur.
    """

    result = {
        'success': False,
        'params': {},
        'param_errors': {},
        'r_squared': 0.0,
        'model_name': model_info['name'],
        'description': model_info['description']
    }
    
    # Überprüfe zuerst, ob activities leer ist
    if len(activities) == 0:
        result['success'] = False
        return result
    
    # Initial guess berechnen
    p0 = model_info['initial_guess_func'](activities, substrate_data)
    if verbose:
        print(f"Initial guess: {p0}")
    
    # Parameter-Grenzen für physikalisch sinnvolle Werte
    bounds_lower = model_info['bounds_lower']
    bounds_upper = model_info['bounds_upper']
    
    try:
        params, covariance = curve_fit(model_info['function'], substrate_data, activities, 
                                        p0=p0, bounds=(bounds_lower, bounds_upper), maxfev=5000)
        fitted_params = params
        param_errors = np.sqrt(np.diag(covariance))
            
        # R² berechnen
        y_pred = model_info['function'](substrate_data, *fitted_params)
            
        ss_res = np.sum((activities - y_pred) ** 2)
        ss_tot = np.sum((activities - np.mean(activities)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        result['success'] = True
        result['params'] = fitted_params
        result['param_errors'] = param_errors
        result['r_squared'] = r_squared

        return result
        
    except Exception as e:
        if verbose:
            print(f"Fehler beim Fitten: {e}")
        result['success'] = False
        return result

def monte_carlo_simulation_r1(calibration_data, reaction_data, model_info, data_info, noise_level, n_iterations=1000):
    """
    Monte Carlo Simulation für Reaktion 1 mit verbesserter Fehlerbehandlung und Statistik.
    
    Args:
        calibration_data: DataFrame mit Kalibrierungsdaten
        reaction_data: Dict mit Reaktionsdaten {reaction_name: DataFrame}
        model_info: Dict mit Modellinformationen
        noise_level: Dict mit {"calibration": float, "reaction": float}
        n_iterations: Anzahl der Iterationen
        
    Returns:
        Dict mit statistischen Auswertungen
    """
    print(f"\n{'='*60}")
    print(f"🔬 MONTE CARLO SIMULATION")
    print(f"{'='*60}")
    print(f"Modell: {model_info['description']}")
    print(f"Iterationen: {n_iterations}")
    print(f"Kalibrierungs-Rauschen: {noise_level['calibration']*100:.1f}%")
    print(f"Reaktions-Rauschen: {noise_level['reaction']*100:.1f}%")
    print(f"{'='*60}")
    
    successful_results = []
    failed_counts = {"calibration": 0, "data_processing": 0, "fitting": 0, "validation": 0}
    
    # Progress Bar Setup
    progress_interval = max(1, n_iterations // 20)  # 20 Updates
    
    for iteration in range(n_iterations):
        try:
            # 1. Verrausche Kalibrierungsdaten
            try:
                calibration_data_noisy = add_noise_calibration(calibration_data, noise_level=noise_level["calibration"])
                calibration_slope_noisy = calculate_calibration(calibration_data_noisy)
            except Exception:
                failed_counts["calibration"] += 1
                continue
            
            # 2. Verrausche Reaktionsdaten und verarbeite sie (OHNE Well-Informationen)
            try:
                active_params = data_info["active_params"]
                reaction_data_noisy = add_noise_reaction_dict(reaction_data, noise_level=noise_level["reaction"])
                
                processed_data_noisy = get_rates_and_concentrations(
                    reaction_data_noisy, calibration_slope_noisy, active_params, verbose=False  # Keine Well-Ausgaben
                )

                if not processed_data_noisy or not processed_data_noisy.get("activities"):
                    failed_counts["data_processing"] += 1
                    continue
                    
            except Exception:
                failed_counts["data_processing"] += 1
                continue
            
            # 3. Erstelle Konzentrations- und Raten-Dictionaries
            try:
                constants_dict = data_info["constants"] #todo add noise here
                concentration_noisy = create_concentrations_dict(processed_data_noisy, constants_dict)
                rates_noisy = create_reaction_rates_dict(processed_data_noisy)
            except Exception:
                failed_counts["data_processing"] += 1
                continue
            
            # 4. Parameter-Schätzung (OHNE Debug-Ausgaben)
            try:
                result = estimate_parameters(model_info, data_info, concentration_noisy, rates_noisy, verbose=False)

                if not result or not result.get('success', False):
                    failed_counts["fitting"] += 1
                    continue
                    
            except Exception:
                failed_counts["fitting"] += 1
                continue
            
            # 5. Validierung der Parameter
            try:
                params = result['params']
                r_squared = result.get('r_squared', 0)
                
                # Plausibilitätsprüfung
                if (len(params) >= 3 and 
                    all(p > 0 for p in params) and  
                    params[0] < 100 and  
                    all(1e-6 < p < 1000 for p in params[1:]) and  
                    r_squared > 0.1):  
                    
                    successful_results.append(result)
                else:
                    failed_counts["validation"] += 1
                    
            except Exception:
                failed_counts["validation"] += 1
                continue
        
        except Exception:
            failed_counts["data_processing"] += 1
            continue
        
        # Progress-Update
        if (iteration + 1) % progress_interval == 0 or iteration + 1 == n_iterations:
            success_rate = len(successful_results) / (iteration + 1) * 100
            progress = (iteration + 1) / n_iterations * 100
            print(f"🔄 Fortschritt: {progress:5.1f}% ({iteration + 1:4d}/{n_iterations}) | ✅ Erfolg: {len(successful_results):3d} ({success_rate:5.1f}%)")
    
            continue
        
        # Progress-Update
        if (iteration + 1) % 100 == 0:
            success_rate = len(successful_results) / (iteration + 1) * 100
            print(f"Iteration {iteration + 1}/{n_iterations} - Erfolgreiche: {len(successful_results)} ({success_rate:.1f}%)")
    
    # Auswertung der Ergebnisse
    n_successful = len(successful_results)
    print(f"\n=== ZUSAMMENFASSUNG ===")
    
    if n_iterations > 0:
        success_percentage = n_successful/n_iterations*100
        print(f"Erfolgreiche Iterationen: {n_successful}/{n_iterations} ({success_percentage:.1f}%)")
    else:
        print(f"Erfolgreiche Iterationen: {n_successful}/{n_iterations} (0.0%)")
        
    print(f"Fehlschläge:")
    for reason, count in failed_counts.items():
        print(f"  - {reason}: {count}")
    
    if n_successful < 10:
        print(f"⚠️  Zu wenige erfolgreiche Iterationen für sinnvolle Statistik!")
        return {
            'successful_results': successful_results,
            'failed_counts': failed_counts,
            'n_successful': n_successful,
            'n_iterations': n_iterations
        }
    
    # Statistische Auswertung
    param_names = model_info['param_names']
    param_units = model_info['param_units']
    
    # Extrahiere Parameter-Arrays
    param_arrays = {}
    r_squared_values = []
    
    for result in successful_results:
        params = result['params']
        r_squared_values.append(result.get('r_squared', 0))
        
        for i, param_name in enumerate(param_names):
            if param_name not in param_arrays:
                param_arrays[param_name] = []
            param_arrays[param_name].append(params[i])
    
    # Konvertiere zu numpy arrays
    for param_name in param_arrays:
        param_arrays[param_name] = np.array(param_arrays[param_name])
    
    r_squared_values = np.array(r_squared_values)
    
    # Erstelle Ergebnis-Dictionary
    mc_results = {
        'n_successful': n_successful,
        'n_total': n_iterations,
        'success_rate': n_successful / n_iterations,
        'model_name': model_info['name'],
        'param_names': param_names,
        'failed_counts': failed_counts,
        
        # R² Statistiken
        'R_squared_mean': np.mean(r_squared_values),
        'R_squared_std': np.std(r_squared_values),
        'R_squared_median': np.median(r_squared_values),
        'R_squared_values': r_squared_values
    }
    
    # Parameter-spezifische Statistiken
    print(f"\n=== PARAMETER-STATISTIKEN ===")
    for i, param_name in enumerate(param_names):
        if param_name in param_arrays:
            values = param_arrays[param_name]
            unit = param_units[i]
            
            mean_val = np.mean(values)
            std_val = np.std(values)
            median_val = np.median(values)
            ci_lower = np.percentile(values, 2.5)
            ci_upper = np.percentile(values, 97.5)
            
            mc_results[f'{param_name}_mean'] = mean_val
            mc_results[f'{param_name}_std'] = std_val
            mc_results[f'{param_name}_median'] = median_val
            mc_results[f'{param_name}_ci_lower'] = ci_lower
            mc_results[f'{param_name}_ci_upper'] = ci_upper
            mc_results[f'{param_name}_values'] = values
            
            print(f"{param_name}: {mean_val:.4f} ± {std_val:.4f} {unit}")
            print(f"  Median: {median_val:.4f} {unit}")
            print(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}] {unit}")
    
    print(f"\nR²: {mc_results['R_squared_mean']:.4f} ± {mc_results['R_squared_std']:.4f}")
    print(f"R² Median: {mc_results['R_squared_median']:.4f}")
    
    # Korrelationsmatrix berechnen
    if len(param_arrays) >= 2:
        parameter_matrix = np.column_stack([param_arrays[name] for name in param_names])
        correlation_matrix = np.corrcoef(parameter_matrix, rowvar=False)
        mc_results['correlation_matrix'] = correlation_matrix
        
        print(f"\n=== PARAMETER-KORRELATIONEN ===")
        for i, name1 in enumerate(param_names):
            for j, name2 in enumerate(param_names):
                if i < j:
                    corr = correlation_matrix[i, j]
                    print(f"{name1} ↔ {name2}: {corr:.3f}")

    # speicher ergebnisse in pickle
    with open(f"monte_carlo_results_{model_info['name']}.pkl", "wb") as f:
        pickle.dump(mc_results, f)

    return mc_results

def validate_parameters(params_dict, r_squared=None):
    """
    Validiert Parameter-Dictionary auf Plausibilität.
    params_dict: Dict mit Parameter-Namen als Keys und Werten
    r_squared: Optional, R²-Wert für zusätzliche Validierung
    """
    for param_name, value in params_dict.items():
        if not isinstance(value, (int, float)):
            return False
        if value <= 0 or value > 10000:
            return False
    
    if r_squared is not None and r_squared < 0.3:
        return False
    
    return True

def estimate_parameters(model_info, data_info, concentrations, rates, verbose=True): 
    """
    Schätzt die Parameter für ein gegebenes Modell basierend auf Konzentrationen und Raten.
    
    model_info: Dict mit Modell-Informationen (z.B. name, function, param_names, etc.)
    data_info: Dict mit Daten-Informationen (z.B. constants, active_params)
    concentrations: Dict mit Substratkonzentrationen
    rates: Dict mit gemessenen Raten
    verbose: Bool - ob Debug-Ausgaben gezeigt werden sollen
    """
    
    substrate_data, activities = make_fitting_data(model_info, data_info, concentrations, rates, verbose=verbose)
    
    # Fitting durchführen
    result = fit_parameters(substrate_data, activities, model_info, verbose=verbose)

    return result

if __name__ == "__main__":

    
    BASE_PATH = r"C:\Users\berger\Documents\Projekts\enzyme-cascade-analysis\Fehlerfortpflanzunganalyse"
    
    calibration_data = pd.read_csv(os.path.join(BASE_PATH, 'Data', 'NADH_Kalibriergerade.csv'))
    calibration_slope = calculate_calibration(calibration_data)

    r1_path = os.path.join(BASE_PATH, 'Data', 'Reaction1')
    r1_nad_data = pd.read_csv(os.path.join(r1_path, 'r_1_NAD_PD_500mM.csv'), header=None)
    r1_pd_data = pd.read_csv(os.path.join(r1_path, 'r1_PD_NAD_5mM.csv'), header=None)


    r2_path = os.path.join(BASE_PATH, 'Data', 'Reaction2')
    r2_hp_data = pd.read_csv(os.path.join(r2_path, 'r_2_HP_NADH_06mM.csv'), header=None)
    r2_nadh_data = pd.read_csv(os.path.join(r2_path, 'r_2_NADH_HP_300mM.csv'), header=None)
    r2_pd_data = pd.read_csv(os.path.join(r2_path, 'r_2_PD_NADH_06nM_HP_300mM.csv'), header=None)

    r3_path = os.path.join(BASE_PATH, 'Data', 'Reaction3')
    r3_lactol = pd.read_csv(os.path.join(r3_path, 'r_3_Lactol_NAD_5mM.csv'))
    r3_nad = pd.read_csv(os.path.join(r3_path, 'r_3_NAD_Lactol_500mM.csv'))


    full_system_data = {
        "r1": {
            "r1_nad": r1_nad_data,
            "r1_pd": r1_pd_data
        },
        "r2": {
            "r2_hp": r2_hp_data,
            "r2_nadh": r2_nadh_data,
            "r2_pd": r2_pd_data
        },
        "r3": {
            "r3_lactol": r3_lactol,
            "r3_nad": r3_nad
        }
    }


    full_system_param = {
        "r1": {
            "Vf_well": 10.0,
            "Vf_prod": 1.0,
            "c_prod": 2.2108,
            "c_nad_const": 5.0,
            "c_pd_const": 500.0
        },
        "r2": {
            "Vf_well": 10.0,
            "Vf_prod": 5.0,
            "c_prod": 2.15,
            "c_hd_const": 6.0,
            "c_nadh_const": 300.0,
            "c_pd_const": 6.0
        },
        "r3": {
            "Vf_well": 10.0,
            "Vf_prod": 10.0,
            "c_prod": 2.15,
            "c_lactol_const": 5.0,
            "c_nad_const": 500.0
        },
        "x_dimension": 3,
        "y_dimension": 1
    }

    full_system_constants_dict = {
        "r1_nad_const": 5.0,    # NAD konstant wenn PD variiert
        "r1_pd_const": 500.0,   # PD konstant wenn NAD variiert
        "r2_hp_const": 6.0,     # HP konstant wenn NADH variiert
        "r2_nadh_const": 300.0,   # NADH konstant wenn HP variiert
        "r2_pd_const": 6.0,      # PD konstant wenn NADH variiert
        "r3_lactol_const": 5.0,  # Lactol konstant wenn NAD variiert
        "r3_nad_const": 500.0   # NAD konstant wenn Lactol variiert
    }

    processed_data = get_rates_and_concentrations(
        full_system_data,
        calibration_slope,
        full_system_param
    )

    # Fix: get_rates function needs 3 parameters, not 2
    full_system_rates = create_reaction_rates_dict(processed_data)

    full_system_concentrations = create_concentrations_dict(processed_data, full_system_constants_dict)

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
    

    full_reaction_system_model_info = {
        "name": "full_reaction_system",
        "function": full_reaction_system,
        "param_names": [
            "Vmax1", "Vmax2", "Vmax3",
            "KmPD", "KmNAD", "KmLactol", "KmNADH",
            "KiPD", "KiNAD", "KiLactol"
        ],
        "param_units": [
            "U", "U", "U",
            "mM", "mM", "mM", "mM",
            "mM", "mM", "mM"
        ],
        "substrate_keys": ["S1", "S2", "Inhibitor", "reaction_ids"],
        "initial_guess_func": lambda activities, substrate_data: [
            max(activities) if len(activities) > 0 else 1.0,  # Vmax1
            max(activities) if len(activities) > 0 else 1.0,  # Vmax2
            max(activities) if len(activities) > 0 else 1.0,  # Vmax3
            1.0,  # KmPD
            1.0,  # KmNAD
            1.0,  # KmLactol
            1.0,  # KmNADH
            1.0,  # KiPD
            1.0,  # KiNAD
            1.0   # KiLactol
        ],
        "bounds_lower": [0]*10,
        "bounds_upper": [np.inf]*10,
        "description": "Komplettes Drei-Reaktions-System mit Inhibitionen"
    }

    reac_1_parameters = estimate_parameters(full_reaction_system_model_info, full_system_param, full_system_concentrations, full_system_rates)

    print("\n=== Parameter Schätzung für das vollständige System ==="
          f"\nModell: {full_reaction_system_model_info['description']}"
          f"\nErgebnis: {reac_1_parameters}"
          f"\nR²: {reac_1_parameters['r_squared']:.4f}")
    for i, param_name in enumerate(full_reaction_system_model_info['param_names']):
        param_val = reac_1_parameters['params'][i]
        param_err = reac_1_parameters['param_errors'][i]
        unit = full_reaction_system_model_info['param_units'][i]
        print(f"{param_name}: {param_val:.4f} ± {param_err:.4f} {unit}")

    noise_level = {
        "calibration": 0.01,
        "reaction": 0.01
    }

    monte_carlo_results = monte_carlo_simulation_r1(
        calibration_data,
        full_system_data,
        full_reaction_system_model_info,
        full_system_param,
        noise_level,
        n_iterations=1000
    )

    if monte_carlo_results:
        print("\n" + "="*60)
        print("📊 MONTE CARLO ERGEBNISSE")
        print("="*60)
        print(f"✅ Erfolgreiche Iterationen: {monte_carlo_results['n_successful']}/{monte_carlo_results['n_total']}")
        print(f"📈 Erfolgsrate: {monte_carlo_results['success_rate']*100:.1f}%")
        print(f"📊 R² (Mittelwert): {monte_carlo_results['R_squared_mean']:.4f} ± {monte_carlo_results['R_squared_std']:.4f}")

        param_names = monte_carlo_results['param_names']
        param_units = full_reaction_system_model_info['param_units']

        for i, param_name in enumerate(param_names):
            mean_key = f"{param_name}_mean"
            std_key = f"{param_name}_std"
            if mean_key in monte_carlo_results and std_key in monte_carlo_results:
                unit = param_units[i] if i < len(param_units) else ""
                print(f"🔸 {param_name}: {monte_carlo_results[mean_key]:.4f} ± {monte_carlo_results[std_key]:.4f} {unit}")
    else:
        print("\n❌ Monte Carlo Simulation fehlgeschlagen!")
