import os
import shutil
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import pickle

from data_handler import  add_noise_reaction_dict, calculate_calibration, add_noise_calibration, create_concentrations_dict, create_reaction_rates_dict, get_rates_and_concentrations , make_fitting_data
from parameter_estimator import estimate_parameters, monte_carlo_simulation

from simulator import cadet_simulation_full_system

from plotter import plot_monte_carlo_results, create_monte_carlo_report, plot_fitting_quality, plot_parameter_convergence, plot_component_analysis



BASE_PATH = r"C:\Users\berger\Documents\Projekts\enzyme-cascade-analysis\Fehlerfortpflanzunganalyse"

calibration_data = pd.read_csv(os.path.join(BASE_PATH, 'Data', 'NADH_Kalibriergerade.csv'))
calibration_slope = calculate_calibration(calibration_data)

r1_path = os.path.join(BASE_PATH, 'Data', 'Reaction1')
r1_nad_data = pd.read_csv(os.path.join(r1_path, 'r_1_NAD_PD_500mM.csv'))
r1_pd_data = pd.read_csv(os.path.join(r1_path, 'r1_PD_NAD_5mM.csv'))


r2_path = os.path.join(BASE_PATH, 'Data', 'Reaction2')
r2_hp_data = pd.read_csv(os.path.join(r2_path, 'r_2_HP_NADH_06mM.csv'))
r2_nadh_data = pd.read_csv(os.path.join(r2_path, 'r_2_NADH_HP_300mM.csv'))
r2_pd_data = pd.read_csv(os.path.join(r2_path, 'r_2_PD_NADH_06mM_HP_300mM.csv'))

r3_path = os.path.join(BASE_PATH, 'Data', 'Reaction3')
r3_lactol = pd.read_csv(os.path.join(r3_path, 'r_3_Lactol_NAD_5mM.csv'))
r3_nad = pd.read_csv(os.path.join(r3_path, 'r_3_NAD_Lactol_500mM.csv'))


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
        "c3_const": 0.0
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

print(f" Verarbeitete Reaktionsdaten gespeichert:")
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

# todo add new level reaction_ad and reaction_co
noise_level = {
    "calibration": 0.01,
    "reaction": 0.01,
}

monte_carlo_results = monte_carlo_simulation(
    calibration_data,
    full_system_data,
    full_reaction_system_model_info,
    full_system_param,
    noise_level,
    n_iterations=5
)

if monte_carlo_results:
    print("\n✅ Monte Carlo Simulation erfolgreich abgeschlossen!")
    print(f"Erfolgreiche Iterationen: {monte_carlo_results['n_successful']}/{monte_carlo_results['n_total']} ({monte_carlo_results['success_rate']*100:.1f}%)")
    
    # Stelle sicher, dass das Results-Verzeichnis existiert
    os.makedirs('Results', exist_ok=True)
    
    # Plot Ergebnisse - alle werden automatisch in Results gespeichert
    simulation_dir = "Results/Simulations"
    if os.path.exists(simulation_dir):
        shutil.rmtree(simulation_dir)
        print(f"🗑️ Simulations-Ordner gelöscht: {simulation_dir}")
    plot_monte_carlo_results(monte_carlo_results, full_reaction_system_model_info)
    create_monte_carlo_report(monte_carlo_results, full_reaction_system_model_info)
    
    # Fitting-Qualität und Konvergenz
    plot_fitting_quality(monte_carlo_results, full_reaction_system_model_info)
    plot_parameter_convergence(monte_carlo_results, full_reaction_system_model_info)

    plot_component_analysis()

else:
    print("\n❌ Monte Carlo Simulation fehlgeschlagen!")