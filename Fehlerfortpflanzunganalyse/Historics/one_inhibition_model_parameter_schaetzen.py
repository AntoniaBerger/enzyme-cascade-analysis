from parameter_schaetzen_noise import *

# Parameter für die Messung - WICHTIG: Korrekte Werte eingeben!
activ_param = {
    "Vf_well": 1.0,      # Verdünnungsfaktor der Gesamtansatzlösung im Well 
    "Vf_prod": 10.0,      # Verdünnungsfaktor der Proteinlösung 
    "c_prod": 2.2108        # Proteinkonzentration [mg/L] 
}

# Zwei-Substrat mit einer Inhibition Modell auswählen
model_name = 'two_substrat_michaelis_menten_with_one_inhibition'

print("=" * 80)
print("=== ZWEI-SUBSTRAT MICHAELIS-MENTEN MIT EINER INHIBITION ===")
print("=" * 80)

print(f"\n=== VERFÜGBARE MODELLE ===")
for name, info in AVAILABLE_MODELS.items():
    params_str = ", ".join([f"{p} [{u}]" for p, u in zip(info['param_names'], info['param_units'])])
    print(f"- {name}: {info['description']} (Parameter: {params_str})")

print(f"\n=== ORIGINALDATEN ANALYSE ===")
print(f"Verwendetes Modell: {model_name}")
print(f"Beschreibung: {AVAILABLE_MODELS[model_name]['description']}")
print(f"Parameter: {', '.join(AVAILABLE_MODELS[model_name]['param_names'])}")
print(f"Einheiten: {', '.join(AVAILABLE_MODELS[model_name]['param_units'])}")
print("\nModell-Gleichung: v = (Vmax * S1 * S2) / ((Km1 + S1) * (Km2 + S2) * (1 + S2/Ki))")
print("Inhibition: Kompetitive Inhibition durch S2")

print("\nStarte Parameterschätzung für Zwei-Substrat-Modell mit einer Inhibition...")
result_original = schaetze_parameter(activ_param, model_name)

if result_original:
    print("\n" + "=" * 60)
    print("=== ERGEBNISSE DER ORIGINALDATEN ===")
    print("=" * 60)
    
    model_info = AVAILABLE_MODELS[model_name]
    for param_name, unit in zip(model_info['param_names'], model_info['param_units']):
        value = result_original[param_name]
        error = result_original[f"{param_name}_error"]
        print(f"Original {param_name}: {value:.4f} ± {error:.4f} {unit}")
    
    print(f"Original R²: {result_original['R_squared']:.4f}")
    
    # Zusätzliche Zwei-Substrat-spezifische Informationen
    if 'S1_constant' in result_original and 'S2_constant' in result_original:
        print(f"\nExperimentelle Bedingungen:")
        print(f"- S1 konstant (Exp. 2): {result_original['S1_constant']:.2f} mM")
        print(f"- S2 konstant (Exp. 1): {result_original['S2_constant']:.2f} mM")
    
    print(f"Modellbeschreibung: {result_original['model_description']}")
    
    # Spezielle Interpretation für Inhibitionsmodell
    if 'Ki' in result_original:
        ki_value = result_original['Ki']
        s2_constant = result_original.get('S2_constant', 500.0)
        inhibition_factor = s2_constant / (ki_value + s2_constant)
        print(f"\nInhibitionsanalyse:")
        print(f"- Inhibitionskonstante Ki: {ki_value:.4f} mM")
        print(f"- Bei S2 = {s2_constant:.0f} mM: Inhibitionsfaktor = {inhibition_factor:.3f}")
        print(f"- Relative Aktivität: {inhibition_factor*100:.1f}% der maximal möglichen")
    
else:
    print("FEHLER: Parameterschätzung fehlgeschlagen!")
    print("Bitte überprüfen Sie:")
    print("- Verfügbarkeit der Datendateien")
    print("- Korrektheit der Pfade")
    print("- Format der Excel-Dateien")
    print("- Ob genügend Datenpunkte für 4 Parameter vorhanden sind")
    exit(1)

print("\n" + "=" * 80)
print("=== MONTE CARLO SIMULATION ===")
print("=" * 80)

# Monte Carlo Simulation für Inhibitionsmodell
print("Starte Monte Carlo Simulation für Zwei-Substrat-Modell mit einer Inhibition...")
print("Hinweis: Dies kann mehrere Minuten dauern!")
print("INFO: Inhibitionsmodelle benötigen oft mehr Iterationen für stabile Ergebnisse.")

mc_results = monte_carlo_simulation(
    activ_param,
    model_name,
    n_iterations=7500,             # Mehr Iterationen für komplexeres Modell
    noise_level_calibration=0.02,  # 2% Rauschen in Kalibrierung
    noise_level_kinetics=0.025     # 2.5% Rauschen in Kinetikdaten (etwas weniger als Standard)
)

if mc_results and result_original:
    print("\n" + "=" * 80)
    print("=== VERGLEICH: ORIGINAL vs MONTE CARLO ===")
    print("=" * 80)
    
    print(f"{'Parameter':<15} {'Original':<15} {'MC Mittel':<15} {'MC Std':<12} {'Abweichung':<12} {'CV [%]'}")
    print("-" * 90)
    
    model_info = AVAILABLE_MODELS[model_name]
    for param_name, unit in zip(model_info['param_names'], model_info['param_units']):
        if f'{param_name}_mean' in mc_results:
            original_val = result_original[param_name]
            mc_mean = mc_results[f'{param_name}_mean']
            mc_std = mc_results[f'{param_name}_std']
            diff = abs(original_val - mc_mean)
            cv = (mc_std/mc_mean)*100
            
            param_display = f"{param_name} [{unit}]"
            print(f"{param_display:<15} {original_val:<15.4f} {mc_mean:<15.4f} {mc_std:<12.4f} {diff:<12.4f} {cv:<8.2f}")
    
    # R² Vergleich
    if 'R_squared_mean' in mc_results:
        original_r2 = result_original['R_squared']
        mc_r2_mean = mc_results['R_squared_mean']
        mc_r2_std = mc_results['R_squared_std']
        r2_diff = abs(original_r2 - mc_r2_mean)
        print(f"{'R²':<15} {original_r2:<15.4f} {mc_r2_mean:<15.4f} {mc_r2_std:<12.4f} {r2_diff:<12.4f}")
    
    print("\n" + "=" * 60)
    print("=== UNSICHERHEITSANALYSE ===")
    print("=" * 60)
    
    for param_name, unit in zip(model_info['param_names'], model_info['param_units']):
        if f'{param_name}_mean' in mc_results:
            mc_mean = mc_results[f'{param_name}_mean']
            mc_std = mc_results[f'{param_name}_std']
            cv = (mc_std/mc_mean)*100
            
            # Konfidenzintervall (2σ ≈ 95%)
            ci_lower = mc_mean - 2*mc_std
            ci_upper = mc_mean + 2*mc_std
            
            print(f"{param_name} ({unit}):")
            print(f"  Variationskoeffizient: {cv:.2f}%")
            print(f"  95% Konfidenzintervall: [{ci_lower:.4f}, {ci_upper:.4f}]")
            
            # Spezielle Interpretation für Ki
            if param_name == 'Ki':
                print(f"  Inhibitionsstärke: ", end="")
                if mc_mean < 10:
                    print("Starke Inhibition")
                elif mc_mean < 100:
                    print("Moderate Inhibition")
                else:
                    print("Schwache Inhibition")
            
            # Bewertung der Unsicherheit
            if cv < 5:
                print(f"  → Sehr präzise Schätzung")
            elif cv < 10:
                print(f"  → Präzise Schätzung")
            elif cv < 20:
                print(f"  → Moderate Unsicherheit")
            else:
                print(f"  → Hohe Unsicherheit")
            print()
    
    # Korrelationsanalyse für 4-Parameter-Modell
    if 'correlation_matrix' in mc_results:
        print("=" * 60)
        print("=== PARAMETER-KORRELATIONSANALYSE ===")
        print("=" * 60)
        
        correlation_matrix = mc_results['correlation_matrix']
        param_names = model_info['param_names']
        
        print("Korrelationsmatrix:")
        print(f"{'Param':<8}", end="")
        for param in param_names:
            print(f"{param:<12}", end="")
        print()
        
        for i, param_i in enumerate(param_names):
            print(f"{param_i:<8}", end="")
            for j, param_j in enumerate(param_names):
                corr_val = correlation_matrix[i, j]
                print(f"{corr_val:<12.3f}", end="")
            print()
        
        print("\nWichtige Korrelationen:")
        for i in range(len(param_names)):
            for j in range(i+1, len(param_names)):
                corr = correlation_matrix[i, j]
                param1, param2 = param_names[i], param_names[j]
                
                print(f"{param1}-{param2}: {corr:.4f}", end="")
                if abs(corr) > 0.8:
                    print("  → Sehr starke Korrelation!")
                elif abs(corr) > 0.6:
                    print("  → Starke Korrelation")
                elif abs(corr) > 0.3:
                    print("  → Moderate Korrelation")
                else:
                    print("  → Schwache Korrelation")
        
        # Spezielle Warnung für Ki-Korrelationen
        ki_idx = param_names.index('Ki') if 'Ki' in param_names else -1
        if ki_idx >= 0:
            print(f"\nINFO: Inhibitionsparameter Ki:")
            for j, param in enumerate(param_names):
                if j != ki_idx:
                    corr = correlation_matrix[ki_idx, j]
                    if abs(corr) > 0.7:
                        print(f"  WARNUNG: Starke Korrelation Ki-{param} ({corr:.3f})")
                        print(f"           Dies kann die Parameterschätzung erschweren!")
    
    print("\n" + "=" * 60)
    print("=== SIMULATIONSSTATISTIK ===")
    print("=" * 60)
    print(f"Erfolgreiche Iterationen: {mc_results['n_successful']}/{mc_results.get('n_total', 'N/A')}")
    
    total_iterations = mc_results.get('n_total', mc_results['n_successful'])
    success_rate = (mc_results['n_successful']/total_iterations)*100
    print(f"Erfolgsrate: {success_rate:.1f}%")
    
    if mc_results['n_successful'] < 1000:
        print("  WARNUNG: Wenige erfolgreiche Iterationen!")
        print("   Inhibitionsmodelle sind sensitiver - versuchen Sie:")
        print("   - Mehr Iterationen (n_iterations=10000)")
        print("   - Weniger Rauschen (noise_level_kinetics=0.02)")
    
    if success_rate < 30:
        print("  WARNUNG: Niedrige Erfolgsrate!")
        print("   Das deutet auf Fitting-Probleme hin:")
        print("   - Überprüfen Sie die Datenqualität")
        print("   - Ki-Parameter schwer zu schätzen?")
    
    print(f"\nHistogramme werden erstellt und gespeichert...")
    print(f"Dateiname: monte_carlo_results_{model_name}.png")

else:
    print("FEHLER: Monte Carlo Simulation oder Original-Analyse fehlgeschlagen!")
    print("\nMögliche Ursachen bei Inhibitionsmodellen:")
    print("- Ki-Parameter schwer zu bestimmen (braucht gute S2-Variation)")
    print("- Starke Parameter-Korrelationen")
    print("- Zu viel Rauschen in den Daten")
    print("- Ungeeignete experimentelle Bedingungen für Inhibitionsnachweis")
    
    print("\nLösungsvorschläge:")
    print("- Reduzieren Sie das Rauschen (noise_level_kinetics=0.015)")
    print("- Erhöhen Sie die Iterationen (n_iterations=10000)")
    print("- Überprüfen Sie, ob S2-Konzentrationen im Ki-Bereich liegen")
    print("- Verwenden Sie das einfachere 'two_substrat_michaelis_menten' Modell")

print("\n" + "=" * 80)
print("=== INHIBITIONSMODELL - ZUSÄTZLICHE INFORMATIONEN ===")
print("=" * 80)
print("Dieses Modell beschreibt kompetitive Inhibition durch das zweite Substrat (S2).")
print("Die Inhibition tritt auf, wenn S2 in hohen Konzentrationen vorliegt.")
print("\nTipps für bessere Ergebnisse:")
print("- Stellen Sie sicher, dass Ihre S2-Konzentrationen einen weiten Bereich abdecken")
print("- Niedrige S2-Konzentrationen zeigen maximale Aktivität")
print("- Hohe S2-Konzentrationen zeigen Inhibitionseffekte")
print("- Ki sollte im Bereich Ihrer S2-Konzentrationen liegen")

print("\n" + "=" * 80)
print("=== PROGRAMMENDE ===")
print("=" * 80)
