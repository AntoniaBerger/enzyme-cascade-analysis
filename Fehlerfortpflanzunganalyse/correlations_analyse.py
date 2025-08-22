import numpy as np

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit

import pandas as pd


# CSV-Datei einlesen und als DataFrame speichern
data = pd.read_csv(r"C:\Users\berger\Documents\Projekts\enzyme-cascade-analysis\Fehlerfortpflanzunganalyse\Data\full_system_dataframe.csv")


# Künstliche Daten für Zwei-Substrat-System (Lactol und NADH), breite Spanne für minimale Korrelation
num_points = 50
S1 = data["c1"]
S2 = data["c2"]
reaction_ids = data["reaction"]

S1_grid, S2_grid = np.meshgrid(S1, S2)
S1_flat = S1_grid.flatten()
S2_flat = S2_grid.flatten()

# "Wahre" Parameter
Vmax2_true = 2.5


# Generiere künstliche Messwerte mit Rauschen
y_data = data["rates"]


x_data_ = []
y_data = []

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
            (KmLactol *(1 + PD_inhibitor / KiPD)  + S1_r2) * (KmNADH  + S2_r2)
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


# Parametric Bootstrap für das vollständige Drei-Reaktions-System

iterations = 100
params_boot = []

# Extrahiere Substratdaten und IDs aus DataFrame
S1 = data["c1"].values
S2 = data["c2"].values
Inhibitor = data["c3"].values if "c3" in data.columns else np.zeros_like(S1)
reaction_ids = data["reaction"].values
y_data = data["rates"].values

for i in range(iterations):
    # 1. Daten verrauschen (parametric: Modell + Fehler)
    y_sim = y_data + np.random.normal(0, 0.1, size=y_data.shape)
    # 2. Fit
    try:
        def fit_func(X, Vmax1, Vmax2, Vmax3, KmPD, KmNAD, KmLactol, KmNADH, KiPD):
            S1, S2, Inhibitor, reaction_ids = X
            return full_reaction_system((S1, S2, Inhibitor, reaction_ids), Vmax1, Vmax2, Vmax3, KmPD, KmNAD, KmLactol, KmNADH, KiPD)
        p0 = [2.0, 2.5, 2.0, 80.0, 2.0, 1.2, 0.3, 90.0]
        popt, _ = curve_fit(
            fit_func,
            (S1, S2, Inhibitor, reaction_ids),
            y_sim,
            p0=p0,
            maxfev=10000
        )
        params_boot.append(popt)
    except Exception:
        continue

params_boot = np.array(params_boot)

# Korrelationsmatrix
corr = np.corrcoef(params_boot.T)
print("Korrelationsmatrix:")
print(corr)

# Corner-Plot
param_names = ["Vmax1", "Vmax2", "Vmax3", "KmPD", "KmNAD", "KmLactol", "KmNADH", "KiPD"]
df = pd.DataFrame(params_boot, columns=param_names)

# Nur ausgewählte Parameter plotten
# subset = ["Vmax2", "KmLactol", "KmNADH"]
# sns.pairplot(df[subset])
# plt.suptitle("Parametric Bootstrap: Vmax2, KmLactol, KmNADH")
# plt.show()

plt.figure(figsize=(6, 5))
sns.heatmap(corr, annot=True, fmt=".2f", xticklabels=param_names, yticklabels=param_names, cmap="coolwarm")
plt.title("Korrelationsmatrix der Bootstrap-Parameter")
plt.tight_layout()
plt.show()