from parameter_schaetzen_noise import *

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
    
else:
    print("Monte Carlo Simulation oder Original-Analyse fehlgeschlagen!")
