import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from time_simulation import*

repo = ProjectRepo()
debug = True
if not debug: 
    input("Do you want to run the file with RDM?")

commit_message = "Julians Zeit-Simulation mit std = 0 und Error Plots der Konzentrationen"
repo.enter_context(debug=debug)

# Enzymkonzentration in mg/L
enzyme_conc = 1  # Beispielwert in mg/L

# Parameterwerte mit Standardabweichungen
vmax1_per_mg, vmax1_std_per_mg = 0.07, 0.002
Km_PD, Km_PD_std = 84, 16
Km_NAD1, Km_NAD1_std = 2.2, 0.2

vmax2_per_mg, vmax2_std_per_mg = 2.26, 1.63
Km_Lactol, Km_Lactol_std = 111, 16
Km_NADH, Km_NADH_std = 2.9, 2.42
Ki_PD, Ki_PD_std = 90, 19
Ki_NAD, Ki_NAD_std = 1.12, 0.4

vmax3_per_mg, vmax3_std_per_mg = 2.3, 0.1
Km_Lactol2, Km_Lactol2_std = 62, 9
Ki_Lacton, Ki_Lacton_std = 108, 48
Km_NAD2, Km_NAD2_std = 2.8, 0.3

# Anfangswerte
c_PD0, c_NAD0, c_Lactol0, c_NADH0, c_Lacton0 = 100, 8, 0, 0, 0

# Anzahl der Monte-Carlo-Simulationen
n_sim = 10500

# Zeitbereich
t_span = (0, 1000)
t_eval = np.linspace(t_span[0], t_span[1], 200)

# Speicherung der Ergebnisse
c_PD_all, c_NAD_all, c_Lactol_all, c_NADH_all, c_Lacton_all = [], [], [], [], []
r1_all, r2_all, r3_all = [], [], []

for _ in range(n_sim):
    # Ziehen zufälliger Parameterwerte vmax in [mmol / (L * min)]
    vmax1 = max(1e-6, np.random.normal(vmax1_per_mg, 0)) * enzyme_conc
    vmax2 = max(1e-6, np.random.normal(vmax2_per_mg, 0)) * enzyme_conc
    vmax3 = max(1e-6, np.random.normal(vmax3_per_mg, 0)) * enzyme_conc
    
    Km_PD_sample = max(1e-6, np.random.normal(Km_PD, 0))
    Km_NAD1_sample = max(1e-6, np.random.normal(Km_NAD1, 0))
    Km_NAD2_sample = max(1e-6, np.random.normal(Km_NAD2, 0))
    Km_Lactol_sample = max(1e-6, np.random.normal(Km_Lactol, 0))
    Km_NADH_sample = max(1e-6, np.random.normal(Km_NADH, 0))
    Ki_PD_sample = max(1e-6, np.random.normal(Ki_PD, 0))
    Ki_NAD_sample = max(1e-6, np.random.normal(Ki_NAD, 0))
    Km_Lactol2_sample = max(1e-6, np.random.normal(Km_Lactol2, 0))
    Ki_Lacton_sample = max(1e-6, np.random.normal(Ki_Lacton, 0))
    
    
    def reaction_rates(y):
        c_PD, c_NAD, c_Lactol, c_NADH, c_Lacton = [max(1e-6, val) for val in y]  # Verhindert zu kleine Werte
        
        r1 = (vmax1 * c_PD * c_NAD) / ((c_PD + Km_PD_sample) * (c_NAD + Km_NAD1_sample))
        r2 = (vmax2 * c_Lactol * c_NADH) / ((c_Lactol + Km_Lactol_sample * (1 + c_PD / Ki_PD_sample)) * (c_NADH + Km_NADH_sample * (1 + c_NAD / Ki_NAD_sample)))
        r3 = (vmax3 * c_Lactol * c_NAD) / ((c_Lactol + Km_Lactol2_sample * (1 + c_Lacton / Ki_Lacton_sample)) * (c_NAD + Km_NAD2_sample))
        
        return [-r1 + r2, -r1 + r2 - r3, r1 - r2 - r3, r1 - r2 + r3, r3], r1, r2, r3
    
    def ode_system(t, y):
        rates, r1, r2, r3 = reaction_rates(y)
        return rates
    
    sol = solve_ivp(ode_system, t_span, [c_PD0, c_NAD0, c_Lactol0, c_NADH0, c_Lacton0], t_eval=t_eval, method='LSODA')
    
    c_PD_all.append(sol.y[0])
    c_NAD_all.append(sol.y[1])
    c_Lactol_all.append(sol.y[2])
    c_NADH_all.append(sol.y[3])
    c_Lacton_all.append(sol.y[4])
    
    r1_vals, r2_vals, r3_vals = [], [], []
    for i in range(len(sol.t)):
        _, r1, r2, r3 = reaction_rates([sol.y[j][i] for j in range(5)])
        r1_vals.append(r1)
        r2_vals.append(r2)
        r3_vals.append(r3)
    
    r1_all.append(r1_vals)
    r2_all.append(r2_vals)
    r3_all.append(r3_vals)

# Umwandlung in numpy Arrays zur Fehlervermeidung
r1_all = np.array(r1_all)
r2_all = np.array(r2_all)
r3_all = np.array(r3_all)

# Berechnung der Mittelwerte und 95%-Konfidenzintervalle für Reaktionsraten
r1_mean = np.mean(r1_all, axis=0)
r1_lower = np.percentile(r1_all, 2.5, axis=0)
r1_upper = np.percentile(r1_all, 97.5, axis=0)
r2_mean = np.mean(r2_all, axis=0)
r2_lower = np.percentile(r2_all, 2.5, axis=0)
r2_upper = np.percentile(r2_all, 97.5, axis=0)
r3_mean = np.mean(r3_all, axis=0)
r3_lower = np.percentile(r3_all, 2.5, axis=0)
r3_upper = np.percentile(r3_all, 97.5, axis=0)

# Berechnung der Mittelwerte und 95%-Konfidenzintervalle für Konzentrationen
c_PD_mean = np.mean(c_PD_all, axis=0)
c_PD_lower = np.percentile(c_PD_all, 2.5, axis=0)
c_PD_upper = np.percentile(c_PD_all, 97.5, axis=0)

c_NAD_mean = np.mean(c_NAD_all, axis=0)
c_NAD_lower = np.percentile(c_NAD_all, 2.5, axis=0)
c_NAD_upper = np.percentile(c_NAD_all, 97.5, axis=0)

c_Lactol_mean = np.mean(c_Lactol_all, axis=0)
c_Lactol_lower = np.percentile(c_Lactol_all, 2.5, axis=0)
c_Lactol_upper = np.percentile(c_Lactol_all, 97.5, axis=0)

c_NADH_mean = np.mean(c_NADH_all, axis=0)
c_NADH_lower = np.percentile(c_NADH_all, 2.5, axis=0)
c_NADH_upper = np.percentile(c_NADH_all, 97.5, axis=0)

c_Lacton_mean = np.mean(c_Lacton_all, axis=0)
c_Lacton_lower = np.percentile(c_Lacton_all, 2.5, axis=0)
c_Lacton_upper = np.percentile(c_Lacton_all, 97.5, axis=0)

 # FZJ-Simulation -----------------------------------------------------------------
stoichiometrie_matrix = np.array([
        [-1, 1, 0],     # PD
        [-1, 1,-1],     # NAD
        [ 1,-1,-1],     # LTOL
        [ 1,-1, 1],     # NADH
        [ 0, 0, 1]      # LTOL
    ])
    
    # Vmax-Werte für jede Reaktion
vmax_values = np.array([np.sqrt(0.07), np.sqrt(2.26), np.sqrt(2.3)])

# Km-Werte für jede Reaktion und jedes relevante Substrat
# Reaktion 1: PD + NAD -> HA + NADH braucht Km für PD und NAD
# Reaktion 2: LTOL + NADH -> PD + NAD braucht Km für LTOL und NADH
km_values = [
    [84, 2.2],      # Km-Werte für Reaktion 1: [Km_PD, Km_NAD]
    [111, 2.9],      # Km-Werte für Reaktion 2: [Km_LTOL, Km_NADH]
    [62, 2.8]      # Km-Werte für Reaktion 3: [Km_LTOL, Km_NAD]
]

# Gleichgewichtskonstanten für die Inhibitoren
# Für Reaktion 2 sind PD und NAD Inhibitoren
gleichgewichtskonstanten = [
    [0, 0],          # Keine Inhibitoren für Reaktion 1
    [90.0, 1.12],    # Inhibitoren für Reaktion 2: [Ki_PD, Ki_NAD]
    [108, 0]         # Inhibitoren für Reaktion 3: [Ki_LTON, -]
]

# Inhibitor-Typen
inhibitor_typen = [
    ["none", "none"],               # Keine Inhibition in Reaktion 1
    ["kompetitiv", "kompetitiv"],   # Kompetitive Inhibition in Reaktion 2
    ["kompetitiv", "none"]    # Nicht-kompetitive Inhibition in Reaktion 3
]

# Anfangskonzentrationen: [PD, NAD, HA, NADH, LTOL]
anfangskonzentrationen = np.array([100.0, 8.0, 0.0, 0.0, 0.0])

# Simulationszeit
simulationszeit = (0, 1000)

# Inhibitor-Indizes (welche Spezies sind Inhibitoren)
# Für Reaktion 2: PD (Index 0), NAD (Index 1) Lacton (Index 4) sind Inhibitoren
inhibitor_indices = [
    [None,None],         # Keine Inhibitoren für Reaktion 1
    [0, 1],         # PD und NAD sind Inhibitoren für Reaktion 2
    [2, None]             # LTOL ist Inhibitor für Reaktion 3
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





# Erstellung der Plots

# Definiere t_max als das maximale Ende von tspan
t_max = np.max(t_span)

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Deutsches Design aktivieren
plt.rcParams.update({
    'font.family': 'serif',  # Wissenschaftliche Schriftart
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'axes.grid': True,  # Standardmäßiges Raster
    'grid.linestyle': '--',
    'grid.alpha': 0.5
})

# Funktion zur Formatierung der Achsenlabels mit deutschen Dezimalzeichen
def deutsche_ticks(ax):
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.0f}".replace(".", ",")))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:.2f}".replace(".", ",")))


import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Schriftart auf Arial setzen
plt.rcParams.update({
    'font.family': 'Arial',  # Setze Arial als Standard
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'axes.grid': True,
    'grid.linestyle': '--',
    'grid.alpha': 0.5
})

# Funktion zur Anpassung der Dezimaltrennzeichen (Punkte → Kommas)
def deutsche_ticks(ax):
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.0f}".replace(".", ",")))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:.2f}".replace(".", ",")))

# Maximale Zeit bestimmen
t_max = np.max(t_span)

# 1. Konzentration von Pentandiol (c_PD) über die Zeit
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(t_eval, c_PD_mean, label=r"$c_{PD}$", color="blue")
#ax.fill_between(t_eval, c_PD_lower, c_PD_upper, color="blue", alpha=0.3)
ax.set_xlabel("Zeit [min]")
ax.set_ylabel("Konzentration [mM]")
ax.set_title("Konzentration von Pentandiol (c_PD) über die Zeit mit std = 0")
ax.set_xlim(0, t_max)
ax.set_ylim(max(0, np.min(c_PD_upper) - 0.5), np.max(c_PD_upper) + 0.5)
ax.legend()
deutsche_ticks(ax)
plt.savefig(fr'{repo.output_path}\c_PD_std0.png')
plt.show()

# 2. Restliche Konzentrationen über die Zeit
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(t_eval, c_NAD_mean, label=r"$c_{NAD}$", color="red")
ax.fill_between(t_eval, c_NAD_lower, c_NAD_upper, color="red", alpha=0.3)
ax.plot(t_eval, c_Lactol_mean, label=r"$c_{Lactol}$", color="green")
ax.fill_between(t_eval, c_Lactol_lower, c_Lactol_upper, color="green", alpha=0.3)
ax.plot(t_eval, c_NADH_mean, label=r"$c_{NADH}$", color="purple")
ax.fill_between(t_eval, c_NADH_lower, c_NADH_upper, color="purple", alpha=0.3)
ax.plot(t_eval, c_Lacton_mean, label=r"$c_{Lacton}$", color="orange")
ax.fill_between(t_eval, c_Lacton_lower, c_Lacton_upper, color="orange", alpha=0.3)
ax.set_xlabel("Zeit [min]")
ax.set_ylabel("Konzentration [mM]")

ax.set_xlim(0, t_max)
max_c_rest = max(np.max(c_NAD_upper), np.max(c_Lactol_upper), np.max(c_NADH_upper), np.max(c_Lacton_upper))
ax.set_ylim(0, max_c_rest + 0.05 * max_c_rest)
ax.set_title("Restliche Konzentrationen über die Zeit mit std = 0")
# Legende in das zweite Viertel des Diagramms (oben rechts, innerhalb der Achse)
ax.legend(loc='upper right', bbox_to_anchor=(1, 0.75))  
plt.savefig(fr'{repo.output_path}\c_NAD_LTOL_NADH_LTON_std0.png')
deutsche_ticks(ax)


# 3. Reaktionsraten über die Zeit
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(t_eval, r1_mean, label=r"$r_1$", color="blue")
ax.fill_between(t_eval, np.percentile(r1_all, 2.5, axis=0), np.percentile(r1_all, 97.5, axis=0), color="blue", alpha=0.3)
ax.plot(t_eval, r2_mean, label=r"$r_2$", color="red")
ax.fill_between(t_eval, np.percentile(r2_all, 2.5, axis=0), np.percentile(r2_all, 97.5, axis=0), color="red", alpha=0.3)
ax.plot(t_eval, r3_mean, label=r"$r_3$", color="green")
ax.fill_between(t_eval, np.percentile(r3_all, 2.5, axis=0), np.percentile(r3_all, 97.5, axis=0), color="green", alpha=0.3)
ax.set_xlabel("Zeit [min]")
ax.set_ylabel("Reaktionsrate [U]")


ax.set_xlim(0, t_max)
max_r = max(np.max(np.percentile(r1_all, 97.5, axis=0)), np.max(np.percentile(r2_all, 97.5, axis=0)), np.max(np.percentile(r3_all, 97.5, axis=0)))
ax.set_ylim(0, max_r + 0.05 * max_r)
# Y-Achse auf 3 Nachkommastellen formatieren
ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.3f}"))

# Legende ebenfalls im zweiten Viertel des Diagramms
ax.legend(loc='upper right', bbox_to_anchor=(1, 0.75))  
ax.set_title("Reaktionsraten über die Zeit mit std = 0")
deutsche_ticks(ax)
plt.savefig(fr'{repo.output_path}\raten_st0png')
plt.show()


# 4. Fehlerplot für die PD-Konzentration
from scipy.interpolate import interp1d

interp_PD =  interp1d(solution1.t, solution1.y[0], bounds_error=False, fill_value="extrapolate")
interp_PD = interp_PD(t_eval)

difference_PD = c_PD_mean - interp_PD

plt.figure(figsize=(10, 6))
plt.plot(t_eval, difference_PD, label=r"$c_{PD}$", color="blue")
plt.title('Fehlerplot zwischen TU_DO-FZJ /n für  Konzentrationen mit std = 0')
plt.xlabel('Zeit')
plt.ylabel('Differenz der Konzentrationen')
plt.legend()
plt.savefig(fr'{repo.output_path}\Fehlerplot_PD.png')
plt.show()


interp_NAD =  interp1d(solution1.t, solution1.y[1], bounds_error=False, fill_value="extrapolate")
interp_NAD = interp_NAD(t_eval)

difference_NAD = c_NAD_mean - interp_NAD

interp_TOL =  interp1d(solution1.t, solution1.y[2], bounds_error=False, fill_value="extrapolate")
interp_TOL = interp_TOL(t_eval)

difference_TOL = c_Lactol_mean - interp_TOL

interp_NADH =  interp1d(solution1.t, solution1.y[3], bounds_error=False, fill_value="extrapolate")
interp_NADH = interp_NADH(t_eval)

difference_NADH = c_NADH_mean - interp_NADH

interp_NTON =  interp1d(solution1.t, solution1.y[4], bounds_error=False, fill_value="extrapolate")
interp_NTON = interp_NTON(t_eval)

difference_NTON = c_Lacton_mean - interp_NTON

plt.figure(figsize=(10, 6))
species_names = ['NAD', 'LTOL', 'NADH', 'LTON']
for i, difference in enumerate([difference_NAD, difference_TOL, difference_NADH, difference_NTON]):
    plt.plot(t_eval, difference, label=species_names[i])
plt.title('Fehlerplot zwischen TU_DO-FZJ /n für  Konzentrationen mit std = 0')
plt.xlabel('Zeit')
plt.ylabel('Differenz der Konzentrationen')
plt.legend()
plt.savefig(fr'{repo.output_path}\Fehlerplot_rest.png')
plt.show()

if not debug:
        repo.exit_context(message = commit_message)
        repo.push()