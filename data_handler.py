import numpy as np
import pandas as pd
import pickle


def save_results(data, parameter_names, save_path="monte_carlo_results.csv"):
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
    
    try:
        df.to_csv(save_path)
        df.to_pickle(save_path.replace('.csv', '.pkl'))
        
    except Exception as e:
        print(f"Error saving results: {e}")

    return df

