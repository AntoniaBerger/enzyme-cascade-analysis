import os
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import pickle
from simulator import cadet_simulation_full_system
from data_handler import  add_noise_reaction_dict, calculate_calibration, add_noise_calibration, create_concentrations_dict, create_reaction_rates_dict, get_rates_and_concentrations , make_fitting_data

from plotter import plot_monte_carlo_results, create_monte_carlo_report, plot_fitting_quality, plot_parameter_convergence


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
    simulation_results = []
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
            
            # 2. Verrausche Reaktionsdaten und verarbeite sie
            try:
                # KORRIGIERT: Verwende data_info direkt (das ist schon reaction_params_dict!)
                reaction_data_noisy = add_noise_reaction_dict(reaction_data, noise_level=noise_level["reaction"])
                
                processed_data_noisy = get_rates_and_concentrations(
                    reaction_data_noisy, 
                    calibration_slope_noisy, 
                    data_info,  # NICHT data_info["active_params"]!
                    verbose=False
                )

                # Prüfe ob DataFrame zurückgegeben wurde
                if processed_data_noisy is None or len(processed_data_noisy) == 0:
                    failed_counts["data_processing"] += 1
                    continue
                    
            except Exception as e:
                print(f"Debug: Data processing error: {e}")  # Für Debug
                failed_counts["data_processing"] += 1
                continue
            
            # 3. Parameter-Schätzung (DIREKT mit DataFrame!)
            try:
                result = estimate_parameters(
                    model_info, 
                    data_info, 
                    processed_data_noisy,  # DataFrame direkt verwenden
                    verbose=False
                )

                if not result or not result.get('success', False):
                    failed_counts["fitting"] += 1
                    continue
                    
            except Exception as e:
                print(f"Debug: Fitting error: {e}")  # Für Debug
                failed_counts["fitting"] += 1
                continue
            
            try: 
                # convert results into param dict:
                params_dict = {name: value for name, value in zip(model_info["param_names"], result['params'])}
                return_code = cadet_simulation_full_system(params_dict, cm_iteration=iteration)

                if return_code == 0: 
                    simulation_results.append(return_code)
           
            except Exception as e:
                print(f"Debug: Simulation error: {e}")  # Für Debug
                failed_counts["simulation"] += 1
                continue

            # 5. Validierung der Parameter
            try:
                params = result['params']
                r_squared = result.get('r_squared', 0)
                
                # Plausibilitätsprüfung
                if (len(params) >= len(model_info["param_names"]) and 
                    all(p > 0 for p in params) and  
                    params[0] < 100 and  
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
    
    n_success_sim = 0
    if n_iterations > 0:
        success_percentage = n_successful/n_iterations*100
        print(f"Erfolgreiche Iterationen: {n_successful}/{n_iterations} ({success_percentage:.1f}%)")
    else:
        print(f"Erfolgreiche Iterationen: {n_successful}/{n_iterations} (0.0%)")
    
    if len(simulation_results) > 0:
        n_success_sim = np.sum(np.array(simulation_results) == 0)  # Simulations-Ergebnisse zählen
        print(f"Erfolgreiche Simulationen: {n_success_sim}/{len(simulation_results)} ({n_success_sim / len(simulation_results) * 100:.1f}%)")
    else:
        print(f"Erfolgreiche Simulationen: {n_success_sim}/{len(simulation_results)} (0.0%)")


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
        'n_successful_mc': n_successful,
        'n_successful_sim': n_success_sim,
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

    # Stelle sicher, dass das Results-Verzeichnis existiert
    os.makedirs('Results', exist_ok=True)
    
    # speicher ergebnisse in pickle
    results_path = os.path.join('Results', f"monte_carlo_results_{model_info['name']}.pkl")
    with open(results_path, "wb") as f:
        pickle.dump(mc_results, f)
    print(f"💾 Monte Carlo Ergebnisse gespeichert: {results_path}")

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

def estimate_parameters(model_info, data_info, processed_data, verbose=False): 
    """
    Schätzt die Parameter für ein gegebenes Modell basierend auf verarbeiteten Daten.
    """
    
    if verbose:
        print(f"DataFrame Info:")
        print(f"Shape: {processed_data.shape}")
        print(f"Columns: {list(processed_data.columns)}")
        print(f"Erste 3 Zeilen:")
        print(processed_data.head(3))
    
    try:
        # Extrahiere Daten als separate Arrays (NICHT als Liste von Listen!)
        reaction_ids = processed_data['reaction'].values
        c1_values = processed_data['c1'].values if 'c1' in processed_data.columns else np.zeros(len(processed_data))
        c2_values = processed_data['c2'].values if 'c2' in processed_data.columns else np.zeros(len(processed_data))
        c3_values = processed_data['c3'].values if 'c3' in processed_data.columns else np.zeros(len(processed_data))
        
        # Activities als flaches Array
        activities = processed_data['rates'].values
        
        if verbose:
            print(f"Extrahierte Arrays:")
            print(f"  reaction_ids: {reaction_ids[:3]}... (shape: {reaction_ids.shape})")
            print(f"  c1_values: {c1_values[:3]}... (shape: {c1_values.shape})")  
            print(f"  c2_values: {c2_values[:3]}... (shape: {c2_values.shape})")
            print(f"  c3_values: {c3_values[:3]}... (shape: {c3_values.shape})")
            print(f"  activities: {activities[:3]}... (shape: {activities.shape})")
        
        # Erstelle concentration_data für curve_fit (4 separate Arrays!)
        # Achtung: Reihenfolge muss zu deiner full_reaction_system Funktion passen!
        concentration_values = [c1_values, c2_values, c3_values, reaction_ids]  # S1, S2, Inhibitor, reaction_ids
        
        if verbose:
            print(f"concentration_values shapes: {[arr.shape for arr in concentration_values]}")
            print(f"activities shape: {activities.shape}")
        
    except Exception as e:
        if verbose:
            print(f"Fehler beim Extrahieren der Daten: {e}")
        return {'success': False, 'error': f'Datenextraktion fehlgeschlagen: {e}'}
    
    # Fitting durchführen
    result = fit_parameters(concentration_values, activities, model_info, verbose=verbose)
    
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
    r2_pd_data = pd.read_csv(os.path.join(r2_path, 'r_2_PD_NADH_06mM_HP_300mM.csv'), header=None)

    r3_path = os.path.join(BASE_PATH, 'Data', 'Reaction3')
    r3_lactol = pd.read_csv(os.path.join(r3_path, 'r_3_Lactol_NAD_5mM.csv'), header=None)
    r3_nad = pd.read_csv(os.path.join(r3_path, 'r_3_NAD_Lactol_500mM.csv'), header=None)


    full_system_data = {
        "r1": {
            "c1": r1_nad_data,
            "c2": r1_pd_data
        },
        "r2": {
            "c1": r2_hp_data,
            "c2": r2_nadh_data,
            "c3": r2_pd_data
        },
        "r3": {
            "c1": r3_lactol,
            "c2": r3_nad
        }
    }


    full_system_param = {
        "r1": {
            "Vf_well": 10.0,
            "Vf_prod": 1.0,
            "c_prod": 2.2108,
            "c1_const": 5.0,
            "c2_const": 500.0
        },
        "r2": {
            "Vf_well": 10.0,
            "Vf_prod": 5.0,
            "c_prod": 2.15,
            "c1_const": 300.0,
            "c2_const": 0.6,
            "c3_const": 100.0
        },
        "r3": {
            "Vf_well": 10.0,
            "Vf_prod": 10.0,
            "c_prod": 2.15,
            "c1_const": 500.0,
            "c2_const": 5.0
        },
        "x_dimension": 3,
        "y_dimension": 1
    }

    df = get_rates_and_concentrations(
        full_system_data,
        calibration_slope,
        full_system_param
    )
    
    # Stelle sicher, dass das Results-Verzeichnis existiert
    os.makedirs('Results', exist_ok=True)
    
    # Speichere im Results-Ordner
    csv_path = os.path.join('Results', "full_system_processed_reaction_data.csv")
    pkl_path = os.path.join('Results', "full_system_processed_reaction_data.pkl")
    
    df.to_csv(csv_path, index=True)
    df.to_pickle(pkl_path)
    
    print(f"💾 Verarbeitete Reaktionsdaten gespeichert:")
    print(f"   CSV: {csv_path}")
    print(f"   PKL: {pkl_path}")

    def full_reaction_system(concentration_data, Vmax1, Vmax2, Vmax3, KmPD, KmNAD, KmLactol, KmNADH, KiPD):
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
        
        # Reaktion 1: PD + NAD → HD + NADH
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
                (KmLactol * (1 + PD_inhibitor/KiPD) + S1_r2) * (KmNADH  + S2_r2)
            )
        # Reaktion 3: Lactol + NAD 
        reaction_3_mask = (reaction_ids == 3)
        if np.any(reaction_3_mask):
            S1_r3 = S1[reaction_3_mask]  # Lactol
            S2_r3 = S2[reaction_3_mask]  # NAD
            V_obs[reaction_3_mask] = (Vmax3 * S1_r3 * S2_r3) / (
                (KmLactol  + S1_r3) * (KmNAD + S2_r3)
            )
        
        return V_obs
    

    full_reaction_system_model_info = {
        "name": "full_reaction_system",
        "function": full_reaction_system,
        "param_names": [
            "Vmax1", "Vmax2", "Vmax3",
            "KmPD", "KmNAD", "KmLactol", "KmNADH",
            "KiPD"
        ],
        "param_units": [
            "U", "U", "U",
            "mM", "mM", "mM", "mM",
            "mM"
        ],
        "substrate_keys": ["S1", "S2", "Inhibitor", "reaction_ids"],
        "initial_guess_func": lambda activities, substrate_data: [
            max(activities) if len(activities) > 0 else 1.0,  # Vmax1
            max(activities) if len(activities) > 0 else 1.0,  # Vmax2
            max(activities) if len(activities) > 0 else 1.0,  # Vmax3
            84.0,  # KmPD
            2.2,  # KmNAD
            75.0,  # KmLactol
            2.0,  # KmNADH
            90.0  # KiPD
        ],
        "bounds_lower": [0]*8,
        "bounds_upper": [np.inf]*8,
        "description": "Komplettes Drei-Reaktions-System mit Inhibitionen"
    }

    full_system_parameters = estimate_parameters(full_reaction_system_model_info, full_system_param,df)



    print("\n=== Parameter Schätzung für das vollständige System ==="
          f"\nModell: {full_reaction_system_model_info['description']}"
          f"\nErgebnis: {full_system_parameters}"
          f"\nR²: {full_system_parameters['r_squared']:.4f}")
    for i, param_name in enumerate(full_reaction_system_model_info['param_names']):
        param_val = full_system_parameters['params'][i]
        param_err = full_system_parameters['param_errors'][i]
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
        n_iterations=10
    )

    if monte_carlo_results:
        print("\n✅ Monte Carlo Simulation erfolgreich abgeschlossen!")
        print(f"Erfolgreiche Iterationen: {monte_carlo_results['n_successful']}/{monte_carlo_results['n_total']} ({monte_carlo_results['success_rate']*100:.1f}%)")
        
        # Stelle sicher, dass das Results-Verzeichnis existiert
        os.makedirs('Results', exist_ok=True)
        
        # Plot Ergebnisse - alle werden automatisch in Results gespeichert
        plot_monte_carlo_results(monte_carlo_results, full_reaction_system_model_info)
        #create_monte_carlo_report(monte_carlo_results, full_reaction_system_model_info)
        
        # Fitting-Qualität und Konvergenz
        #plot_fitting_quality(monte_carlo_results, full_reaction_system_model_info)
        #plot_parameter_convergence(monte_carlo_results, full_reaction_system_model_info)
    else:
        print("\n❌ Monte Carlo Simulation fehlgeschlagen!")
