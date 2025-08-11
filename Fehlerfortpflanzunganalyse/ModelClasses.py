import numpy as np

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
