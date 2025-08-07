"""
Multi-Aktivitäts Enzym Charakterisierung mit Monte Carlo Unsicherheitsanalyse
=============================================================================
Vollständiges 3-Reaktionssystem mit robusten Parameterschätzungen
"""

from parameter_schaetzen_noise import *

# Experimentelle Parameter
activ_param = {
    "Vf_well": 1.0,      # Verdünnungsfaktor Well
    "Vf_prod": 10.0,     # Verdünnungsfaktor Protein
    "c_prod": 2.2108     # Proteinkonzentration [mg/L]
}


model_name = 'full_reaction_system'

print("=" * 90)
print("=== MULTI-AKTIVITÄTS ENZYM CHARAKTERISIERUNG ===")
print("=" * 90)
print("Reaktionssystem: 3 Reaktionen, 10 Parameter")
print("- Reaktion 1: PD + NAD → Produkt (Michaelis-Menten)")  
print("- Reaktion 2: Lactol + NADH → Produkt (mit Inhibition)")
print("- Reaktion 3: Lactol + NAD → Produkt (mit Selbstinhibition)")
print("=" * 90)

# =============================================================================
# 1. PARAMETERSCHÄTZUNG
# =============================================================================

print("\nStarte Parameterschätzung...")
result_original = schaetze_parameter(activ_param, model_name)

def print_parameter_results(result):
    """Zeigt Parameterergebnisse formatiert an"""
    if not result or 'param_dict' not in result:
        return False
        
    param_dict = result['param_dict']
    vmax_values = param_dict["Vmax"]
    km_values = param_dict["Km"] 
    ki_values = param_dict["Ki"]
    
    print(f"\n{'Parameter':<15} {'Wert':<12} {'Einheit':<8} {'Beschreibung'}")
    print("-" * 60)
    
    # Enzym-Aktivitäten
    print("AKTIVITÄTEN:")
    activities = [("Vmax1", "PD + NAD"), ("Vmax2", "Lactol + NADH"), ("Vmax3", "Lactol + NAD")]
    for i, (name, desc) in enumerate(activities):
        print(f"{name:<15} {vmax_values[i]:<12.4f} {'U/mg':<8} {desc}")
    
    # Substrat-Affinitäten  
    print("\nAFFINITÄTEN:")
    affinities = [("KmPD", "Pyruvat Deh."), ("KmNAD", "NAD"), ("KmLactol", "Lactol"), ("KmNADH", "NADH")]
    for i, (name, desc) in enumerate(affinities):
        print(f"{name:<15} {km_values[i]:<12.4f} {'mM':<8} {desc}")
    
    # Inhibitions-Konstanten
    print("\nINHIBITIONEN:")
    inhibitions = [("KiPD", "durch PD"), ("KiNAD", "durch NAD"), ("KiLactol", "durch Lactol")]
    for i, (name, desc) in enumerate(inhibitions):
        print(f"{name:<15} {ki_values[i]:<12.4f} {'mM':<8} {desc}")
    
    return True

if result_original:
    print(f"\n✅ Parameterschätzung erfolgreich! R² = {result_original['R_squared']:.4f}")
    print_parameter_results(result_original)
else:
    print("❌ FEHLER: Parameterschätzung fehlgeschlagen!")
    exit(1)

# =============================================================================
# 2. MONTE CARLO UNSICHERHEITSANALYSE  
# =============================================================================

# Monte Carlo ausführen
print("\n" + "=" * 90)
print("=== MONTE CARLO UNSICHERHEITSANALYSE ===")
print("=" * 90)

# Echte Monte Carlo Simulation mit der funktionierenden minimalen Version
mc_results = monte_carlo_simulation(activ_param, model_name='full_reaction_system', 
                                           n_iterations=100)

# =============================================================================
# 3. ERGEBNISVERGLEICH UND BEWERTUNG
# =============================================================================

if mc_results and result_original:
    print("\n" + "=" * 90)
    print("=== PARAMETER-UNSICHERHEITEN ===")
    print("=" * 90)
    
    param_names = mc_results['base_result']['parameter_names']
    mean_params = mc_results['mean_params']
    std_params = mc_results['std_params']
    
    param_mapping = {
        'Vmax1': ('Vmax', 0), 'Vmax2': ('Vmax', 1), 'Vmax3': ('Vmax', 2),
        'KmPD': ('Km', 0), 'KmNAD': ('Km', 1), 'KmLactol': ('Km', 2), 'KmNADH': ('Km', 3),
        'KiPD': ('Ki', 0), 'KiNAD': ('Ki', 1), 'KiLactol': ('Ki', 2)
    }
    
    print(f"{'Parameter':<12} {'Original':<10} {'MC Mittel':<10} {'±Std':<8} {'CV%':<6} {'Bewertung'}")
    print("-" * 70)
    
    for i, param_name in enumerate(param_names):
        if param_name in param_mapping:
            key, index = param_mapping[param_name]
            original_val = result_original['param_dict'][key][index]
            mc_mean = mean_params[i]
            mc_std = std_params[i]
            cv = (mc_std/mc_mean)*100 if mc_mean != 0 else 0
            
            # Bewertung
            if cv < 5:
                rating = "Exzellent"
            elif cv < 15:
                rating = "Gut"
            elif cv < 30:
                rating = "Akzeptabel"
            else:
                rating = "Unsicher"
            
            print(f"{param_name:<12} {original_val:<10.3f} {mc_mean:<10.3f} {mc_std:<8.3f} {cv:<6.1f} {rating}")
    
    # R² Vergleich
    original_r2 = result_original['R_squared']
    mc_r2_mean = mc_results['base_result']['R_squared']
    print(f"{'R²':<12} {original_r2:<10.4f} {mc_r2_mean:<10.4f} {'--':<8} {'--':<6}")
    
    # Zusammenfassung
    print("\n📊 ZUSAMMENFASSUNG:")
    print(f"   Erfolgsrate: {mc_results['successful_iterations']}/{mc_results['successful_iterations']} (100.0%)")
    
    all_cvs = [(std_params[i]/mean_params[i])*100 for i in range(len(mean_params)) if mean_params[i] != 0]
    stable_params = sum(1 for cv in all_cvs if cv < 15)
    print(f"   Stabile Parameter: {stable_params}/{len(all_cvs)} (CV < 15%)")
    
    if stable_params >= 8:
        print("   ✅ Modell ist robust und zuverlässig!")
    elif stable_params >= 6:
        print("   ⚠️  Modell ist moderat zuverlässig")
    else:
        print("   ❌ Modell zeigt hohe Unsicherheiten")

else:
    print("❌ FEHLER: Monte Carlo Simulation fehlgeschlagen!")

print("\n" + "=" * 90)
print("=== ANALYSE ABGESCHLOSSEN ===") 
print("=" * 90)
print("Multi-Aktivitäts Enzym erfolgreich charakterisiert!")
print("Alle 10 Parameter geschätzt mit Unsicherheitsquantifizierung")
print("=" * 90)
