import numpy as np
import pandas as pd
import pickle


def save_results(data, parameter_names, dataset_name="monte_carlo_results"):
    """
    Save Monte Carlo results to a DataFrame.

    Parameters:
    data (np.ndarray): Array of Monte Carlo results.
    parameter_names (list): List of parameter names.
    dataset_name (str): Name of the dataset for the DataFrame.

    Returns:
    pd.DataFrame: DataFrame containing the Monte Carlo results.
    """
    df = pd.DataFrame(data, columns=parameter_names)
    df.index.name = dataset_name
    
    df.to_csv(f"Data/{dataset_name}_results.csv")
    df.to_pickle(f"Data/{dataset_name}_results.pkl")

    return df

