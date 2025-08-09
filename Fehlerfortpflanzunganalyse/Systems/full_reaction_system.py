import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Fehlerfortpflanzunganalyse.parameter_estimator import estimate_parameters,calculate_calibration, monte_carlo_simulation_r1
from Fehlerfortpflanzunganalyse.data_handler import calculate_calibration, get_rates_and_concentrations

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
        "c1_const": 6.0,
        "c2_const": 300.0,
        "c3_const": 6.0
    },
    "r3": {
        "Vf_well": 10.0,
        "Vf_prod": 10.0,
        "c_prod": 2.15,
        "c1_const": 5.0,
        "c2_const": 500.0
    },
    "x_dimension": 3,
    "y_dimension": 1
}

df = get_rates_and_concentrations(
    full_system_data,
    calibration_slope,
    full_system_param
)



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