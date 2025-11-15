import numpy as np
import pandas as pd

def reaction1_synthetic_data(parameters, noise_level=0.01, num_points=50):
    Vmax, Km1, Km2 = parameters
    S1 = np.linspace(0.1, 10, num_points)
    S2 = np.linspace(0.1, 10, num_points)
    S1_grid, S2_grid = np.meshgrid(S1, S2)
    
    # Michaelis-Menten equation
    data = []
    for s1 in S1: 
        for s2 in S2:
            rate = (Vmax * s1 * s2) / ((Km1 + s1) * (Km2 + s2))
            data.append({'PD_mM': s1, 'NAD_mM': s2, 'activity_U/mg': rate})

    df = pd.DataFrame(data)

    return df



def reaction2_synthetic_data(parameters, noise_level=0.01, num_points=50):
    Vmax, Km1, Km2, Ki1 = parameters
    S1 = np.linspace(0.1, 500, num_points)
    S2 = np.linspace(0.1, 10, num_points)
    S3 = np.linspace(0.1, 500, num_points)
    
    data = []
    for s1 in S1:
        for s2 in S2:
            for s3 in S3:
                rate = (Vmax * s1 * s2) / ((Km1*(1+s3/Ki1) + s1) * (Km2 + s2))
                data.append({'HP_mM': s1, 'NADH_mM': s2, 'PD_mM': s3, 'activity_U/mg': rate})
    df = pd.DataFrame(data)

    return df
