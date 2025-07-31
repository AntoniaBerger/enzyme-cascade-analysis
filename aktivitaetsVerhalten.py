import numpy as np
from ReaktionsGeschwindigkeiten import multiSubstratMM
import matplotlib.pyplot as plt

from cadetrdm import ProjectRepo
repo = ProjectRepo()
debug = True
commit_message = "Repdoduziertes Aktivitätsverhalten"
repo.enter_context(debug=debug)


plt.figure(figsize=(15, 10))

# Reaktion 1 : PD + NAD -> HA + NADH
# Plot 1: NAD
vmaxNAD = 0.07
kmNAD = 2.2
hemmung = ["none"]
konz_NAD_reac1 = np.arange(0.0, 10.1, 0.1)  # Feinere Auflösung
activität_NAD = []

for c in konz_NAD_reac1:
    activität_NAD.append(multiSubstratMM([c], vmaxNAD, [kmNAD], hemmung=hemmung))

plt.subplot(2, 3, 1)
plt.plot(konz_NAD_reac1, activität_NAD, 'b-')
plt.title('NAD (Reaktion 1)')
plt.xlabel('Konzentration [mM]')
plt.ylabel('Aktivität')
plt.grid(True)

# Plot 2: PD
kmPD = 84
vmaxPD = 0.05
hemmung = ["none"]
konz_PD = np.arange(0.0, 800.1, 5)  # Angepasste Schrittgröße
activität_PD = []
for c in konz_PD:
    activität_PD.append(multiSubstratMM([c], vmaxPD, [kmPD], hemmung=hemmung))

plt.subplot(2, 3, 2)
plt.plot(konz_PD, activität_PD, 'g-')
plt.title('PD (Reaktion 1)')
plt.xlabel('Konzentration [mM]')
plt.ylabel('Aktivität')
plt.grid(True)

# Reaktion 2: LTOL + NADH -> PD + NAD
# Plot 3: LTOL
vmaxLTOL = 0.5
kmLTOL = 112.0
hemmung = ["none"]
konz_LTOL = np.arange(0.0, 800.1, 5)  
activität_LTOL = []
for c in konz_LTOL:
    activität_LTOL.append(multiSubstratMM([c], vmaxLTOL, [kmLTOL], hemmung=hemmung))

plt.subplot(2, 3, 3)
plt.plot(konz_LTOL, activität_LTOL, 'r-')
plt.title('LTOL (Reaktion 2)')
plt.xlabel('Konzentration [mM]')
plt.ylabel('Aktivität')
plt.grid(True)

# Plot 4: NADH
vmaxNADH = 2.5
kmNADH = 2.9
hemmung = ["none"]
konz_NADH = np.arange(0.0, 1.1, 0.01)  
activität_NADH = []
for c in konz_NADH:
    activität_NADH.append(multiSubstratMM([c], vmaxNADH, [kmNADH], hemmung=hemmung))

plt.subplot(2, 3, 4)
plt.plot(konz_NADH, activität_NADH, 'c-')
plt.title('NADH (Reaktion 2)')
plt.xlabel('Konzentration [mM]')
plt.ylabel('Aktivität')
plt.grid(True)

# Reaktion 3: LTOL + NAD -> LTON + NADH
# Plot 5: LTOL
vmaxLTON = 1.7
kmLTON = 62
hemmung = ["none"]
konz_LTON = np.arange(0.0, 800.1, 2) 
activität_LTON = []
for c in konz_LTON:
    activität_LTON.append(multiSubstratMM([c], vmaxLTON, [kmLTON], hemmung=hemmung))

plt.subplot(2, 3, 5)
plt.plot(konz_LTON, activität_LTON, 'm-')
plt.title('LTOL (Reaktion 3)')
plt.xlabel('Konzentration [mM]')
plt.ylabel('Aktivität')
plt.grid(True)

# Plot 6: NAD2
vmaxNAD2 = 2.5 
kmNAD2 = 2.8
hemmung = ["none"]
konz_NAD2 = np.arange(0.0, 10.1, 0.1)  # Feinere Auflösung
activität_NAD2 = []
for c in konz_NAD2:
    activität_NAD2.append(multiSubstratMM([c], vmaxNAD2, [kmNAD2], hemmung=hemmung))

plt.subplot(2, 3, 6)
plt.plot(konz_NAD2, activität_NAD2, 'y-')
plt.title('NAD (Reaktion 3)')
plt.xlabel('Konzentration [mM]')
plt.ylabel('Aktivität')
plt.grid(True)

# Gesamttitel und Layout optimieren
plt.suptitle('Aktivitätsverhalten aller Substrate in den Reaktionen', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])  # Platz für den Haupttitel lassen
plt.savefig(fr'{repo.output_path}\MMaktivitaetsVerhalten.png')

if not debug:
    repo.exit_context(message = commit_message) 
