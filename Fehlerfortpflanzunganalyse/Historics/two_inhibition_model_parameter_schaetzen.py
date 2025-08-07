from parameter_schaetzen_noise import *

# Parameter für die Messung - WICHTIG: Korrekte Werte eingeben!
activ_param = {
    "Vf_well": 1.0,      # Verdünnungsfaktor der Gesamtansatzlösung im Well 
    "Vf_prod": 10.0,      # Verdünnungsfaktor der Proteinlösung 
    "c_prod": 2.2108        # Proteinkonzentration [mg/L] 
}

# Zwei-Substrat mit zwei Inhibitionen Modell auswählen
model_name = 'two_substrat_michaelis_menten_with_two_inhibition'

print("=" * 80)
print("=== ZWEI-SUBSTRAT MICHAELIS-MENTEN MIT ZWEI INHIBITIONEN ===")
print("=" * 80)

print("\n=== VERFÜGBARE MODELLE ===")
for name, info in AVAILABLE_MODELS.items():
    params_str = ", ".join([f"{p} [{u}]" for p, u in zip(info['param_names'], info['param_units'])])
    print(f"- {name}: {info['description']} (Parameter: {params_str})")

print("\n=== ORIGINALDATEN ANALYSE ===")
print(f"Verwendetes Modell: {model_name}")
print(f"Beschreibung: {AVAILABLE_MODELS[model_name]['description']}")
print(f"Parameter: {', '.join(AVAILABLE_MODELS[model_name]['param_names'])}")
print(f"Einheiten: {', '.join(AVAILABLE_MODELS[model_name]['param_units'])}")
print("\nModell-Gleichung: v = (Vmax * S1 * S2) / ((Km1 + S1) * (Km2 + S2) * (1 + S2/Ki1) * (1 + S1/Ki2))")
print("Inhibition: Doppelte kompetitive Inhibition - S1 und S2 inhibieren sich gegenseitig")

print("\nStarte Parameterschätzung für Zwei-Substrat-Modell mit zwei Inhibitionen...")
print("ACHTUNG: Dieses Modell hat 5 Parameter - benötigt sehr gute Datenqualität!")
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
        print("\nExperimentelle Bedingungen:")
        print(f"- S1 konstant (Exp. 2): {result_original['S1_constant']:.2f} mM")
        print(f"- S2 konstant (Exp. 1): {result_original['S2_constant']:.2f} mM")
    
    print(f"Modellbeschreibung: {result_original['model_description']}")
    
    # Spezielle Interpretation für doppeltes Inhibitionsmodell
    if 'Ki1' in result_original and 'Ki2' in result_original:
        ki1_value = result_original['Ki1']
        ki2_value = result_original['Ki2']
        s1_constant = result_original.get('S1_constant', 5.0)
        s2_constant = result_original.get('S2_constant', 500.0)
        
        # Inhibitionsfaktoren berechnen
        s2_inhibition = s2_constant / (ki1_value + s2_constant)
        s1_inhibition = s1_constant / (ki2_value + s1_constant)
        combined_inhibition = s2_inhibition * s1_inhibition
        
        print("\nDoppelte Inhibitionsanalyse:")
        print(f"- Ki1 (S2-Inhibition): {ki1_value:.4f} mM")
        print(f"- Ki2 (S1-Inhibition): {ki2_value:.4f} mM")
        print(f"- S2-Inhibitionsfaktor: {s2_inhibition:.3f}")
        print(f"- S1-Inhibitionsfaktor: {s1_inhibition:.3f}")
        print(f"- Kombinierte Inhibition: {combined_inhibition:.3f}")
        print(f"- Relative Gesamtaktivität: {combined_inhibition*100:.1f}%")
        
        # Dominante Inhibition identifizieren
        if s2_inhibition < s1_inhibition:
            print(f"- Dominante Inhibition: S2 (stärker)")
        elif s1_inhibition < s2_inhibition:
            print(f"- Dominante Inhibition: S1 (stärker)")
        else:
            print(f"- Beide Inhibitionen sind etwa gleich stark")
    
else:
    print("FEHLER: Parameterschätzung fehlgeschlagen!")
    print("Bitte überprüfen Sie:")
    print("- Verfügbarkeit der Datendateien")
    print("- Korrektheit der Pfade")
    print("- Format der Excel-Dateien")
    print("- Ob genügend Datenpunkte für 5 Parameter vorhanden sind")
    print("- Datenqualität (5-Parameter-Modell ist sehr anspruchsvoll!)")
    exit(1)

print("\n" + "=" * 80)
print("=== MONTE CARLO SIMULATION ===")
print("=" * 80)

# Monte Carlo Simulation für doppeltes Inhibitionsmodell
print("Starte Monte Carlo Simulation für Zwei-Substrat-Modell mit zwei Inhibitionen...")
print("Hinweis: Dies kann längere Zeit dauern!")
print("WARNUNG: 5-Parameter-Modell ist sehr komplex - niedrige Erfolgsrate normal!")

mc_results = monte_carlo_simulation(
    activ_param,
    model_name,
    n_iterations=10000,            # Deutlich mehr Iterationen für komplexes Modell
    noise_level_calibration=0.015, # Weniger Rauschen für stabiles Fitting
    noise_level_kinetics=0.02      # Weniger Rauschen in Kinetikdaten
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
            
            # Spezielle Interpretation für Ki1 und Ki2
            if param_name in ['Ki1', 'Ki2']:
                inhibitor = 'S2' if param_name == 'Ki1' else 'S1'
                print(f"  Inhibition durch {inhibitor}: ", end="")
                if mc_mean < 10:
                    print("Starke Inhibition")
                elif mc_mean < 100:
                    print("Moderate Inhibition")
                else:
                    print("Schwache Inhibition")
            
            # Bewertung der Unsicherheit (relaxierte Kriterien für komplexes Modell)
            if cv < 10:
                print("  → Sehr gute Schätzung für 5-Parameter-Modell")
            elif cv < 20:
                print("  → Akzeptable Schätzung")
            elif cv < 40:
                print("  → Moderate Unsicherheit (normal für komplexe Modelle)")
            else:
                print("  → Hohe Unsicherheit - Parameter schwer bestimmbar")
            print()
    
    # Erweiterte Korrelationsanalyse für 5-Parameter-Modell
    if 'correlation_matrix' in mc_results:
        print("=" * 60)
        print("=== PARAMETER-KORRELATIONSANALYSE (5 PARAMETER) ===")
        print("=" * 60)
        
        correlation_matrix = mc_results['correlation_matrix']
        param_names = model_info['param_names']
        
        print("Korrelationsmatrix:")
        print(f"{'Param':<8}", end="")
        for param in param_names:
            print(f"{param:<10}", end="")
        print()
        
        for i, param_i in enumerate(param_names):
            print(f"{param_i:<8}", end="")
            for j, param_j in enumerate(param_names):
                corr_val = correlation_matrix[i, j]
                print(f"{corr_val:<10.3f}", end="")
            print()
        
        print("\nKritische Korrelationen (|r| > 0.7):")
        critical_correlations = []
        for i in range(len(param_names)):
            for j in range(i+1, len(param_names)):
                corr = correlation_matrix[i, j]
                param1, param2 = param_names[i], param_names[j]
                
                if abs(corr) > 0.7:
                    critical_correlations.append((param1, param2, corr))
                    print(f"  {param1}-{param2}: {corr:.4f} → KRITISCH!")
        
        if not critical_correlations:
            print("  Keine kritischen Korrelationen gefunden - gutes Zeichen!")
        else:
            print(f"\nWARNUNG: {len(critical_correlations)} kritische Korrelationen gefunden!")
            print("Dies erschwert die eindeutige Parameterschätzung.")
        
        # Spezielle Analyse der Ki-Parameter
        ki_params = ['Ki1', 'Ki2']
        for ki in ki_params:
            if ki in param_names:
                ki_idx = param_names.index(ki)
                print(f"\nKorrelationen von {ki}:")
                for j, param in enumerate(param_names):
                    if j != ki_idx:
                        corr = correlation_matrix[ki_idx, j]
                        strength = "schwach" if abs(corr) < 0.3 else "moderat" if abs(corr) < 0.6 else "stark"
                        print(f"  {ki}-{param}: {corr:.3f} ({strength})")
    
    print("\n" + "=" * 60)
    print("=== SIMULATIONSSTATISTIK ===")
    print("=" * 60)
    print(f"Erfolgreiche Iterationen: {mc_results['n_successful']}/{mc_results.get('n_total', 'N/A')}")
    
    total_iterations = mc_results.get('n_total', mc_results['n_successful'])
    success_rate = (mc_results['n_successful']/total_iterations)*100
    print(f"Erfolgsrate: {success_rate:.1f}%")
    
    if mc_results['n_successful'] < 500:
        print("  WARNUNG: Sehr wenige erfolgreiche Iterationen!")
        print("   5-Parameter-Modell ist extrem sensitiv:")
        print("   - Erwägen Sie einfachere Modelle")
        print("   - Mehr Iterationen (n_iterations=20000)")
        print("   - Weniger Rauschen (noise_level_kinetics=0.01)")
    
    if success_rate < 10:
        print("  KRITISCH: Sehr niedrige Erfolgsrate!")
        print("   Das Modell ist wahrscheinlich überparametrisiert:")
        print("   - Verwenden Sie das Ein-Inhibitions-Modell")
        print("   - Überprüfen Sie Ihre Datenqualität")
        print("   - Mehr experimentelle Datenpunkte nötig")
    elif success_rate < 20:
        print("  WARNUNG: Niedrige Erfolgsrate für komplexes Modell")
        print("   Ergebnisse mit Vorsicht interpretieren")
    
    print(f"\nHistogramme werden erstellt und gespeichert...")
    print(f"Dateiname: monte_carlo_results_{model_name}.png")

else:
    print("FEHLER: Monte Carlo Simulation oder Original-Analyse fehlgeschlagen!")
    print("\nHäufige Probleme bei 5-Parameter-Modellen:")
    print("- Überparametrisierung (zu viele Parameter für verfügbare Daten)")
    print("- Starke Parameter-Korrelationen")
    print("- Ki1 und Ki2 nicht eindeutig bestimmbar")
    print("- Unzureichende experimentelle Variation")
    
    print("\nEmpfehlungen:")
    print("- Verwenden Sie das einfachere Ein-Inhibitions-Modell")
    print("- Sammeln Sie mehr experimentelle Daten")
    print("- Reduzieren Sie das Rauschen drastisch (noise_level_kinetics=0.01)")
    print("- Erhöhen Sie die Iterationen auf 20000+")
    print("- Überprüfen Sie, ob beide Inhibitionseffekte wirklich vorhanden sind")

print("\n" + "=" * 80)
print("=== ZWEI-INHIBITIONS-MODELL - ZUSÄTZLICHE INFORMATIONEN ===")
print("=" * 80)
print("Dieses Modell beschreibt doppelte kompetitive Inhibition:")
print("- S2 inhibiert bei hohen Konzentrationen (Ki1)")
print("- S1 inhibiert bei hohen Konzentrationen (Ki2)")
print("\nModellkomplexität:")
print("- 5 Parameter müssen geschätzt werden")
print("- Sehr hohe Datenanforderungen")
print("- Starke Parameter-Korrelationen möglich")
print("\nWann dieses Modell verwenden:")
print("- Klare Evidenz für beide Inhibitionseffekte")
print("- Sehr gute Datenqualität")
print("- Breite Konzentrationsbereiche für S1 und S2")
print("- Wenn einfachere Modelle unzureichend sind")

print("\n" + "=" * 80)
print("=== PROGRAMMENDE ===")
print("=" * 80)
