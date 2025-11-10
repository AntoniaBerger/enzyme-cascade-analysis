import numpy as np
import pandas as pd
import scipy.optimize as opt

from artifical_data import reaction1_synthetic_data

def estimate_parameters(data, model_func, initial_guess):
    """
    Estimate parameters for a given model function using non-linear curve fitting.

    Parameters:
    data (pd.DataFrame): DataFrame containing the experimental data.
    model_func (callable): The model function to fit to the data.
    initial_guess (list): Initial guess for the parameters to be estimated.

    Returns:
    tuple: Estimated parameters and covariance matrix.
    """
    # Extract independent variables and dependent variable from the DataFrame
    S1 = [float(idx.split('_')[1]) for idx in data.index]
    S2 = [float(col.split('_')[1]) for col in data.columns]
    rates = data.values.flatten()

    # Create meshgrid for S1 and S2
    S1_grid, S2_grid = np.meshgrid(S1, S2)
    S1_flat = S1_grid.flatten()
    S2_flat = S2_grid.flatten()

    # Fit the model to the data
    popt, pcov = opt.curve_fit(model_func, (S1_flat, S2_flat), rates, p0=initial_guess)

    return popt, pcov

if __name__ == "__main__":

    # Generate synthetic data for testing
    true_parameters = (100, 2, 3)  # Vmax, Km1, Km2
    synthetic_data = reaction1_synthetic_data(true_parameters)

    # Define the Michaelis-Menten model function
    def michaelis_menten(S, Vmax, Km1, Km2):
        S1, S2 = S
        return (Vmax * S1 * S2) / ((Km1 + S1) * (Km2 + S2))

    # Initial guess for parameters
    initial_guess = [80, 1, 1]

    # Estimate parameters
    estimated_params, covariance = estimate_parameters(synthetic_data, michaelis_menten, initial_guess)

    print("Estimated Parameters:")
    print(f"Vmax: {estimated_params[0]}")
    print(f"Km1: {estimated_params[1]}")
    print(f"Km2: {estimated_params[2]}")