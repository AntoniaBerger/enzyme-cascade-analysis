import sys
import os
import numpy as np
import pandas as pd
sys.path.append(r"C:\\Users\\berger\\Documents\\Projekts\\enzyme-cascade-analysis")

from artifical_data import reaction1_synthetic_data
from monte_carlo_estimator import monte_carlo_parameter_estimation
from plotter import print_monte_carlo_info, corner_plot_monte_carlo_results, correlation_matrix_plot, compare_corner_plots, compare_error_ellipses
from data_handler import save_results
from noise_function_libary import  full_experiment_processing_with_noise, add_noise_rate
import os
import pandas as pd


# experimental data
EXPERIMENTAL_DATA_PATH = r"C:\\Users\\berger\\Documents\\Projekts\\enzyme-cascade-analysis\\example_reactions\\dortmund_system\\documentation\\experimental_data"

# processed data
PROCESSED_DATA_PATH = r"C:\\Users\\berger\\Documents\\Projekts\\enzyme-cascade-analysis\\example_reactions\\dortmund_system\\processed_data"

# results path
RESULTS_PATH = r"C:\\Users\\berger\\Documents\\Projekts\\enzyme-cascade-analysis\\example_reactions\\dortmund_system\\results"

# set seed 
np.random.seed(42)

# define model
    
parameters = ['Vmax', 'Km2']
substrates = ["PD_mM"]

def michaelis_menten(S, *parameters):
    S1 = S
    Vmax, Km2 = parameters

    return (Vmax * S1 ) / ((Km2 + S1))

 # Perform Monte Carlo parameter estimation with experimental data

data = pd.read_csv(os.path.join(EXPERIMENTAL_DATA_PATH, "Reaction1", "r_1_PD.csv"))
cal_data = pd.read_csv(os.path.join(EXPERIMENTAL_DATA_PATH, "NADH_Kalibriergerade.csv"))

cal_parameters = {
    "Vf_well": 10.0,  # mL
    "Vf_prod": 1.0,  # mL
    "c_prod": 2.2108    # mg/mL
}

initial_guess = [0.1, 1]


noise_level = {
    'fehler_wage': 0.001,
    'fehler_pipettieren': 0.02,
    'fehler_time_points': 0.001,
    'fehler_od': 0.007
}
num_iterations = 1000

mc_reaction1_noisy_plate_reader = monte_carlo_parameter_estimation(data, 
                                                            cal_data, substrates, cal_parameters,
                                                            michaelis_menten, full_experiment_processing_with_noise, 
                                                            initial_guess, noise_level = noise_level, num_iterations= num_iterations)


df_reaction1 = save_results(mc_reaction1_noisy_plate_reader, parameters, save_path=os.path.join(RESULTS_PATH, "MC_reaction1_full_experiment_PD.csv"))

print_monte_carlo_info(parameters, df_reaction1)

# Perform Monte Carlo parameter estimation with experimental data
noise_level = (noise_level["fehler_pipettieren"] + noise_level["fehler_od"] + noise_level["fehler_time_points"])/3

mc_reaction1_noisy_plate_reader = monte_carlo_parameter_estimation(data,
                                                            cal_data, substrates, cal_parameters,
                                                            michaelis_menten, add_noise_rate,
                                                            initial_guess, noise_level = noise_level, num_iterations = num_iterations)


df_reaction1_2= save_results(mc_reaction1_noisy_plate_reader, parameters, save_path=os.path.join(RESULTS_PATH, "MC_reaction1_rate_noise_PD.csv"))


print_monte_carlo_info(parameters, df_reaction1_2)
