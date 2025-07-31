from ReaktionsGeschwindigkeiten import multiSubstratMM
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from cadetrdm import ProjectRepo


def simulation_reaction_system(
    anfangskonzentrationen, # von substraten und produkten/inhibitoren 
    vmax,
    km,
    gleichgewichtskonstanten,
    inhibitor_typen,
    stoicheometrie_matrix,
    simulationszeit, 
    inhibitor_indices
):
    n_reactions = stoicheometrie_matrix.shape[1]
    n_species = stoicheometrie_matrix.shape[0]

    #dc/dt = S*r
    def ode_system(t,y):
        
        species = y[:n_species]
        reaction_rates = np.zeros(n_reactions)
        for i in range(n_reactions):
            # finde substrate von reaktion i 
            substrat_indices = np.where(stoicheometrie_matrix[:,i] < 0)[0]

            substrat_konz = np.array([species[idx] for idx in substrat_indices])
            km_substrate = km[i]
            hemmung_substrate = inhibitor_typen[i]
            
            inhibitor_konz = np.zeros(len(substrat_indices))
            j = 0
            for idx in inhibitor_indices[i]: 
                if idx is not None:
                    inhibitor_konz[j] = species[idx]
                    j += 1
                    
            reaction_rates[i] = multiSubstratMM(
                substrat_konz,
                vmax[i],
                km_substrate,
                inhibitor_konz,
                gleichgewichtskonstanten[i],
                hemmung_substrate
            )
        dy_dt = np.dot(stoicheometrie_matrix, reaction_rates)
        return dy_dt
        
    solution = solve_ivp(
        ode_system, 
        simulationszeit,
        anfangskonzentrationen,
        method='RK45', 
        rtol = 1e-6,
        atol = 1e-6
    )

    return solution

def simulation_reaction_system_manual(
    anfangskonzentrationen, 
    vmax,
    km,
    gleichgewichtskonstanten,
    inhibitor_typen,
    simulationszeit, 
    inhibitor_indices=None
):
    """
    Simuliert das Reaktionssystem mit manuell aufgestellten ODEs für die drei spezifischen Reaktionen.
    
    Reaktionen:
    1: PD + NAD -> LTOL + NADH
    2: LTOL + NADH -> PD + NAD
    3: LTOL + NAD -> NADH + LTON
    
    Indizes:
    0: PD, 1: NAD, 2: LTOL, 3: NADH, 4: LTON
    """
    
    # Manuell definierte ODE-Funktion für die spezifischen Reaktionen
    def ode_system_manual(t, y):
        # y enthält die Konzentrationen: [PD, NAD, LTOL, NADH, LTON]
        PD, NAD, LTOL, NADH, LTON = y
        
        # Berechne die Reaktionsraten manuell
        # Reaktion 1: PD + NAD -> LTOL + NADH
        Vmax1 = vmax[0]
        Km_PD = km[0][0]
        Km_NAD = km[0][1]
        

        rate1 = Vmax1 * Vmax1 * PD * NAD / ((Km_PD + PD)* (Km_NAD + NAD))

        # Reaktion 2: LTOL + NADH -> PD + NAD
        
        Vmax2 =  vmax[1]
        Km_LTOL = km[1][0]
        Km_NADH = km[1][1]
        Ki_PD = gleichgewichtskonstanten[1][0]
        Ki_NAD = gleichgewichtskonstanten[1][1]
        

        rate2 = Vmax2 * Vmax2 * LTOL * NADH / ((LTOL + Km_LTOL*(1 + PD/Ki_PD))*(NADH + Km_NADH*(1 + NAD/Ki_NAD)))


        # Reaktion 3: LTOL + NAD -> NADH + LTON
        
        Vmax3 = vmax[2]
        Km_LTOL = km[2][1]
        Km_NAD = km[2][0]
        Ki_LTON = gleichgewichtskonstanten[2][1]


        rate3 = Vmax3 * Vmax3 * LTOL * NAD / ((LTOL + Km_LTOL*(1 + LTON/Ki_LTON))*(NAD + Km_NAD))

       
        dPD_dt = -rate1 + rate2               # PD wird in Reaktion 1 verbraucht, in Reaktion 2 erzeugt
        dNAD_dt = -rate1 + rate2 - rate3        # NAD wird in Reaktion 1 und 3 verbraucht, in Reaktion 2 erzeugt
        dLTOL_dt = rate1 - rate2 - rate3     # LTOL wird in Reaktion 1 erzeugt, in Reaktion 2 und 3 verbraucht
        dNADH_dt = rate1 - rate2 + rate3    # NADH wird in Reaktion 1 und 3 erzeugt, in Reaktion 2 verbraucht
        dLTON_dt = rate3                    # LTON wird nur in Reaktion 3 erzeugt
        
        return np.array([dPD_dt, dNAD_dt, dLTOL_dt, dNADH_dt, dLTON_dt])
    
    # Löse das ODE-System
    solution = solve_ivp(
        ode_system_manual, 
        simulationszeit,
        anfangskonzentrationen,
        method='RK45', 
        rtol=1e-6,
        atol=1e-6
    )
    
    return solution


if __name__ == "__main__":

    repo = ProjectRepo()
    debug = True
    if not debug: 
        input("Do you want to run the file with RDM?")

    commit_message = "Zeitsimulation des Reaktionssystems"
    repo.enter_context(debug=debug)

    # Beispiel: Zweistufiges System mit 3 Spezies
    # Reaktion 1: PD + NAD -> LTOL + NADH
    # Reaktion 2: LTOL + NADH -> PD + NAD
    # Reaktion 3: LTOL + NAD -> NADH + LTON
    
    # Indizes: 0: PD, 1: NAD, 2: LTOL, 3: NADH, 4: LTON
    
    # Stöchiometrie-Matrix (Zeilen sind Spezies, Spalten sind Reaktionen)
    stoichiometrie_matrix = np.array([
        [-1, 1, 0],     # PD
        [-1, 1,-1],     # NAD
        [ 1,-1,-1],     # LTOL
        [ 1,-1, 1],     # NADH
        [ 0, 0, 1]      # LTON
    ])
    
    # Vmax-Werte für jede Reaktion
    vmax_values = np.array([np.sqrt(0.07), np.sqrt(2.26), np.sqrt(2.3)])
    
    # Km-Werte für jede Reaktion und jedes relevante Substrat
    # Reaktion 1: PD + NAD -> HA + NADH braucht Km für PD und NAD
    # Reaktion 2: LTOL + NADH -> PD + NAD braucht Km für LTOL und NADH
    km_values = [
        [84.0, 2.2],      # Km-Werte für Reaktion 1: [Km_PD, Km_NAD]
        [111.0, 2.9],      # Km-Werte für Reaktion 2: [Km_LTOL, Km_NADH]
        [2.8, 62.0]      # Km-Werte für Reaktion 3: [Km_NAD,Km_LTOL]
    ]
    
    # Gleichgewichtskonstanten für die Inhibitoren
    # Für Reaktion 2 sind PD und NAD Inhibitoren
    gleichgewichtskonstanten = [
        [0.0, 0.0],         # Keine Inhibitoren für Reaktion 1
        [90.0, 1.12],    # Inhibitoren für Reaktion 2: [Ki_PD, Ki_NAD]
        [0,108.0]        # Inhibitoren für Reaktion 3: [Ki_LTON, -]
    ]
    
    # Inhibitor-Typen
    inhibitor_typen = [
        ["none", "none"],               # Keine Inhibition in Reaktion 1
        ["kompetitiv", "kompetitiv"],   # Kompetitive Inhibition in Reaktion 2
        ["kompetitiv", "none"]    # Nicht-kompetitive Inhibition in Reaktion 3
    ]
    
    # Anfangskonzentrationen: [PD, NAD, LTOL, NADH, LTON]
    anfangskonzentrationen = np.array([100.0, 8.0, 0.0, 0.0, 0.0])
    
    # Simulationszeit
    simulationszeit = (0, 500)
    
    # Inhibitor-Indizes (welche Spezies sind Inhibitoren)
    # Für Reaktion 2: PD (Index 0), NAD (Index 1) Lacton (Index 4) sind Inhibitoren
    inhibitor_indices = [
        [None,None],         # Keine Inhibitoren für Reaktion 1
        [0, 1],         # PD und NAD sind Inhibitoren für Reaktion 2
        [4, None]             # LTON ist Inhibitor für Reaktion 3
    ]
    
    # Führe Simulation durch
    solution1 = simulation_reaction_system(
        anfangskonzentrationen,
        vmax_values,
        km_values,
        gleichgewichtskonstanten,
        inhibitor_typen,
        stoichiometrie_matrix,
        simulationszeit,
        inhibitor_indices
    )
    print("-----------------------------------------------------------------")
    solution2 = simulation_reaction_system_manual(
        anfangskonzentrationen,
        vmax_values,
        km_values,
        gleichgewichtskonstanten,
        inhibitor_typen,
        simulationszeit,
        inhibitor_indices
    )

    
    # Plotte Ergebnisse
    plt.figure(figsize=(10, 6))
    species_names = ['PD', 'NAD', 'LTOL', 'NADH', 'LTON']
    plt.plot(solution1.t, solution1.y[1], label=species_names[1])
    plt.plot(solution2.t, solution2.y[1], label=species_names[1] + " (manuell)", linestyle = '--')
    plt.plot(solution1.t, solution1.y[2], label=species_names[2])
    plt.plot(solution2.t, solution2.y[2], label=species_names[2] + " (manuell)", linestyle = '--')
    plt.plot(solution1.t, solution1.y[3], label=species_names[3])
    plt.plot(solution2.t, solution2.y[3], label=species_names[3] + " (manuell)", linestyle = '--')
    plt.plot(solution1.t, solution1.y[4], label=species_names[4])
    plt.plot(solution2.t, solution2.y[4], label=species_names[4] + " (manuell)", linestyle = '--')

    
    plt.xlabel('Zeit')
    plt.ylabel('Konzentration')
    plt.legend()
    plt.grid(True)
    plt.title('Simulation eines enzymatischen Reaktionssystems')
    plt.savefig(fr'{repo.output_path}\simulation_MM_system.png')
    plt.show()

    if not debug:
        repo.exit_context(message = commit_message)