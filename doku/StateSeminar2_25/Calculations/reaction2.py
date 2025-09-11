# includes
import os
import sys
import shutil
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import pickle

BASE_PATH = r"C:\Users\berger\Documents\Projekts\enzyme-cascade-analysis\Fehlerfortpflanzunganalyse"

# Add BASE_PATH to Python path so we can import modules from there
sys.path.append(BASE_PATH)

from data_handler import calc_calibration_slope, compute_processed_data
from parameter_estimator import estimate_parameters, monte_carlo_simulation
from simulator import cadet_simulation_full_system
from plotter import plot_monte_carlo_results, create_monte_carlo_report, plot_fitting_quality, plot_parameter_convergence, plot_component_analysis,plot_corner_plot


BASE_PATH = r"C:\Users\berger\Documents\Projekts\enzyme-cascade-analysis\Fehlerfortpflanzunganalyse"

calibration_data = pd.read_csv(os.path.join(BASE_PATH, 'Data', 'NADH_Kalibriergerade.csv'))
calibration_slope = calc_calibration_slope(calibration_data)

r2_path = os.path.join(BASE_PATH, 'Data', 'Reaction2')
r2_hp_data = pd.read_csv(os.path.join(r2_path, 'r_2_HP_NADH_06mM.csv'))
r2_nadh_data = pd.read_csv(os.path.join(r2_path, 'r_2_NADH_HP_300mM.csv'))
r2_pd_data = pd.read_csv(os.path.join(r2_path, 'r_2_PD_NADH_06mM_HP_300mM.csv'))


reaction_2_data = {
    "r2": {
        "c1": r2_hp_data,
        "c2": r2_nadh_data,
        "c3": r2_pd_data
    },
}


reaction2_data_info = {
    "r2": {
        "Vf_well": 10.0,
        "Vf_prod": 5.0,
        "c_prod": 2.15,
        "c1_const": 300.0,
        "c2_const": 0.6,
        "c3_const": 0.0
    },
    "x_dimension": 3,
    "y_dimension": 1
}

reaction2_dataframe = compute_processed_data(
    reaction_2_data,
    calibration_slope,
    reaction2_data_info
)

print("Anzahl der Datenpunkte pro Reaction")
print(reaction2_dataframe["reaction"].value_counts())

reaction2_dataframe.to_pickle(os.path.join(os.path.dirname(__file__), 'Results', 'reaction2_dataframe.pkl'))
reaction2_dataframe.to_csv(os.path.join(os.path.dirname(__file__), 'Results', 'reaction2_dataframe.csv'))

def reaction_2(concentration_data, Vmax2, KmLactol, KmNADH, KiPD):
    """
    Wrapper für curve_fit Kompatibilität - nimmt flache Parameter entgegen
    Berechnet die Enzymaktivität für die zweite Reaktion

    ALLE DREI REAKTIONEN:
    - Reaktion 1: PD + NAD → Pyruvat + NADH
    """
    # Entpacke Substratkonzentrationen, Inhibitor-Konzentrationen und Reaktions-IDs
    S1, S2, Inhibitor, reaction_ids = concentration_data

    # Initialisiere Ergebnis-Array
    V_obs = np.zeros_like(S1, dtype=float)
    
    # Reaktion 1: PD + NAD → HD + NADH
    reaction_2_mask = (reaction_ids == 2)
    if np.any(reaction_2_mask):
        S1_r2 = S1[reaction_2_mask]  # Lactol
        S2_r2 = S2[reaction_2_mask]  # NADH
        PD_inhibitor = Inhibitor[reaction_2_mask]  # Variable PD-Konzentration als Inhibitor
        V_obs[reaction_2_mask] = (Vmax2 * S1_r2 * S2_r2) / (
            (KmLactol *(1 + PD_inhibitor / KiPD)  + S1_r2) * (KmNADH  + S2_r2)
        )
    
    return V_obs


reaction2_model_info = {
    "name": "reaction_2",
    "function": reaction_2,
    "param_names": [
        "Vmax2", "KmLactol", "KmNADH", "KiPD"
    ],
    "param_units": [
        "U",
        "mM", "mM", "mM"
    ],
    "substrate_keys": ["S1", "S2", "Inhibitor", "reaction_ids"],
    "initial_guess_func": lambda activities, substrate_data: [
        max(activities) if len(activities) > 0 else 1.0,  # Vmax2
        111,  # KmLactol
        2.9,   # KmNADH
        90.0   # KiPD
        ],
    "bounds_lower": [0]*4,
    "bounds_upper": [np.inf]*4,
    "description": "Komplettes Drei-Reaktions-System mit Inhibitionen"
}

reaction2_parameters = estimate_parameters(reaction2_model_info, reaction2_data_info, reaction2_dataframe,sigma = 0.01)

print("\n=== Parameter Schätzung für das vollständige System ==="
        f"\nModell: {reaction2_parameters['description']}"
        f"\nR²: {reaction2_parameters['r_squared']:.4f}"
        f"\ncovariance: {reaction2_parameters['correlation_matrix']}")

for i, param_name in enumerate(reaction2_model_info['param_names']):
    param_val = reaction2_parameters['params'][i]
    param_err = reaction2_parameters['param_errors'][i]
    unit = reaction2_model_info['param_units'][i]
    print(f"{param_name}: {param_val:.4f} ± {param_err:.4f} {unit}")


noise_level = {
    "calibration": 0.00,
    "reaction": 0.01,
    "concentration": 0.00
}
simulation_dir = "Results/Simulations"

if os.path.exists(simulation_dir):
    shutil.rmtree(simulation_dir)
    print(f"Simulations-Ordner gelöscht: {simulation_dir}")

monte_carlo_results = monte_carlo_simulation(
    calibration_data,
    reaction_2_data,
    reaction2_model_info,
    reaction2_data_info,
    noise_level,
    noise_model="plate_reader",
    n_iterations=5000
)

# Erstelle alle Ergebnisse mit demselben Zeitstempel in einem Ordner
results_dir = os.path.join(os.path.dirname(__file__), 'Results_reaction2')
timestamp = plot_monte_carlo_results(monte_carlo_results, reaction2_model_info, save_path=results_dir)
create_monte_carlo_report(monte_carlo_results, reaction2_model_info, save_path=results_dir, timestamp=timestamp)
plot_corner_plot(monte_carlo_results, reaction2_model_info, show_plots=True, save_path=results_dir, timestamp=timestamp)

monte_carlo_results = monte_carlo_simulation(
    calibration_data,
    reaction_2_data,
    reaction2_model_info,
    reaction2_data_info,
    noise_level,
    noise_model="processed_data",
    n_iterations=5000
)

# Erstelle alle Ergebnisse mit demselben Zeitstempel in einem Ordner
results_dir = os.path.join(os.path.dirname(__file__), 'Results_reaction2')
timestamp = plot_monte_carlo_results(monte_carlo_results, reaction2_model_info, save_path=results_dir)
create_monte_carlo_report(monte_carlo_results, reaction2_model_info, save_path=results_dir, timestamp=timestamp)
plot_corner_plot(monte_carlo_results, reaction2_model_info, show_plots=True, save_path=results_dir, timestamp=timestamp)
