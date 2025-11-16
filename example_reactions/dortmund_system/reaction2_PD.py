import sys
import os
import pandas as pd
sys.path.append(r"C:\Users\berger\Documents\Projekts\enzyme-cascade-analysis")
import numpy as np

from monte_carlo_estimator import monte_carlo_parameter_estimation
from plotter import compare_multiple_ellipses, print_monte_carlo_info, corner_plot_monte_carlo_results, correlation_matrix_plot, compare_corner_plots, compare_error_ellipses
from data_handler import save_results
from noise_function_libary import add_noise_rate, full_experiment_processing_with_noise

# Experimental data paths
EXPERIMENTAL_DATA_PATH = "C:\\Users\\berger\\Documents\\Projekts\\enzyme-cascade-analysis\\example_reactions\\dortmund_system\\documentation\\experimental_data"

# Processed data path
PROCESSED_DATA_PATH = "C:\\Users\\berger\\Documents\\Projekts\\enzyme-cascade-analysis\\example_reactions\\dortmund_system\\documentation\\processed_data"

# Results path
RESULTS_PATH = "C:\\Users\\berger\\Documents\\Projekts\\enzyme-cascade-analysis\\example_reactions\\dortmund_system\\results"

np.random.seed(42)

# Define model
parameters = ['Vmax', 'Km1', 'Ki']
substrates = ["PD_mM"]

def michaelis_menten_inhibition_PD(S, *parameters):
    S1 = S
    Vmax, Km1, Ki = parameters
    return (Vmax * 300 * 0.6) / ((Km1 * (1+ S1/Ki) + 300))

# Perform Monte Carlo parameter estimation with experimental data
data = pd.read_csv(os.path.join(EXPERIMENTAL_DATA_PATH, "Reaction2", "r_2_PD.csv"))
cal_data = pd.read_csv(os.path.join(EXPERIMENTAL_DATA_PATH, "NADH_Kalibriergerade.csv"))

cal_parameters = {
    "Vf_well": 10.0,  # mL
    "Vf_prod": 5.0,   # mL
    "c_prod": 2.15    # mg/mL
}

initial_guess = [10, 60, 10]

noise_level = {
    'fehler_wage': 0.02,
    'fehler_pipettieren': 0.01,
    'fehler_time_points': 0.001,
    'fehler_od': 0.01
}
num_iterations = 1000

mc_reaction2_noisy_plate_reader = monte_carlo_parameter_estimation(
    data, 
    cal_data, 
    substrates, 
    cal_parameters,
    michaelis_menten_inhibition_PD, 
    full_experiment_processing_with_noise, 
    initial_guess, 
    noise_level = noise_level, 
    num_iterations = num_iterations
)

df_reaction2_noisy_plate_reader = save_results(
    mc_reaction2_noisy_plate_reader, 
    parameters, 
    save_path=os.path.join(RESULTS_PATH, "MC_reaction2_full_experiment_PD.csv")
)

# Print results and generate plots
print_monte_carlo_info(parameters, df_reaction2_noisy_plate_reader)

noise_level = (noise_level['fehler_od']  + noise_level['fehler_time_points'] + noise_level['fehler_pipettieren'])/3

mc_reaction2_noisy_rate = monte_carlo_parameter_estimation(
    data, 
    cal_data, 
    substrates, 
    cal_parameters,
    michaelis_menten_inhibition_PD, 
    add_noise_rate, 
    initial_guess,
    noise_level = noise_level, 
    num_iterations = num_iterations
)

df_reaction2_noisy_rate = save_results(
    mc_reaction2_noisy_rate, 
    parameters, 
    save_path=os.path.join(RESULTS_PATH, "MC_reaction2_rate_noise_PD.csv")
)

print_monte_carlo_info(parameters, df_reaction2_noisy_rate)