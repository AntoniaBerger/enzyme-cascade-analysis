import numpy as np

def michelisMentenFlux(
    substrat_konzentration, 
    Vmax, 
    Km, 
    inhibitor_konzentration = None, 
    gleichgewichtskonstante = None, 
    hemmung = "none"
    ):
    reaktionsgeschwindigkeit = 0
    if hemmung == "none":
        reaktionsgeschwindigkeit = Vmax * substrat_konzentration / (Km + substrat_konzentration)
        return reaktionsgeschwindigkeit
    
    # Berechnung des Inhibierungsterms
    inhibitor = 0
    for i in range(len(inhibitor_konzentration)):
        if gleichgewichtskonstante[i] <= 0:
            continue
        inhibitor  += inhibitor_konzentration[i] / gleichgewichtskonstante[i]
    
    # Berechnung der Reaktionsgeschwindigkeit 

    if hemmung == "kompetitiv":
        reaktionsgeschwindigkeit = Vmax * substrat_konzentration / (Km * (1 + inhibitor) + substrat_konzentration)
    if hemmung == "nicht-kompetitiv":
        reaktionsgeschwindigkeit = Vmax * substrat_konzentration / (Km + (1 + inhibitor) * substrat_konzentration)
    if hemmung == "unkompetitiv":
        reaktionsgeschwindigkeit = Vmax * substrat_konzentration / ((Km + substrat_konzentration)*(1 + inhibitor))
    
    return reaktionsgeschwindigkeit

def multiSubstratMM(
    substrat_konzentrationen: np.array,
    Vmax: float,
    Km: np.array,
    inhibitor_konzentrationen: np.ndarray =  np.zeros((1,1)),
    gleichgewichtskonstante: np.ndarray = np.zeros((1,1)),
    hemmung: np.array = "none"
    ):

    # multipliziere flux von jedem substrat basierend auf stiochiometrie
    reaktionsgeschwindigkeit = 1
    for i in range(len(substrat_konzentrationen)):
        csub = substrat_konzentrationen[i]
        hsub = hemmung[i]
        km = Km[i]
        if hemmung[i] == "none":
                reaktionsgeschwindigkeit *= michelisMentenFlux(csub, Vmax, km)
        else:
            isub =[inhibitor_konzentrationen[i]]
            gsub = [gleichgewichtskonstante[i]]
            # multipliziere flux wenn substrat in stiochiometrie > 1 
            reaktionsgeschwindigkeit *= michelisMentenFlux(csub, Vmax, km, isub, gsub, hsub)
            
    return reaktionsgeschwindigkeit