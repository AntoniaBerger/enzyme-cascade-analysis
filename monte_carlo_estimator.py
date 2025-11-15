from data_handler import save_results
from parameter_estimator import  estimate_parameters_adaptive
import corner
import numpy as np
import pandas as pd
import scipy.optimize as opt
import numpy.random as random
import os

from plotter import corner_plot_monte_carlo_results, correlation_matrix_plot, print_monte_carlo_info
from process_data import get_time_points, get_concentration_data, get_absorbance_data, get_calibration_slope, process_duplicates, get_processed_data, add_noise
from noise_function_libary import add_noise_plate_reader, add_noise_rate

def monte_carlo_parameter_estimation(data:pd.DataFrame, 
                                     cal_data:pd.DataFrame, 
                                     substrate:list[str], 
                                     cal_param:dict[str, float], 
                                     model_func:callable, 
                                     noise_function:callable,
                                     initial_guess:list[float], 
                                     noise_level=0.1,
                                     estimate_method='standard',
                                     num_iterations=1000):
    estimated_params = []
    local_data = data.copy()
    local_cal_data = cal_data.copy()

    for _ in range(num_iterations):
        
        # Add Gaussian noise to the data
        data_noisy = noise_function(local_data, local_cal_data, substrate, cal_param, noise_level)

        # Estimate parameters using the noisy data
        popt, _ = estimate_parameters_adaptive(data_noisy, model_func,substrate, initial_guess =initial_guess, method=estimate_method)
        estimated_params.append(popt)

    return np.array(estimated_params)


if __name__ == "__main__":

    # Pfade

    # experimental data
    EXPERIMENTAL_DATA_PATH = r"C:\Users\berger\Documents\Projekts\enzyme-cascade-analysis\example_reactions\dortmund_system\experimental_data"

    # processed data
    PROCESSED_DATA_PATH = r"C:\Users\berger\Documents\Projekts\enzyme-cascade-analysis\example_reactions\dortmund_system\processed_data"

    # results path
    RESULTS_PATH = r"C:\Users\berger\Documents\Projekts\enzyme-cascade-analysis\example_reactions\dortmund_system\results"

    # define model

    parameters = ['Vmax', 'Km1', 'Km2', "Ki1"]
    substrates = ["HP_mM", "NADH_mM", "PD_mM"]

    def michaelis_menten2(S, *parameters):
        S1, S2, S3 = S # S1: HP, S2: NADH, S3: PD
        Vmax, Km1, Km2, Ki1 = parameters

        return (Vmax * S1 * S2) / ((Km1*(1+S3/Ki1) + S1) * (Km2 + S2))
    
    data = pd.read_csv(os.path.join(EXPERIMENTAL_DATA_PATH, "Reaction2", "r_2_HP_NADH_PD.csv"))
    cal_data = pd.read_csv(os.path.join(EXPERIMENTAL_DATA_PATH, "NADH_Kalibriergerade.csv"))

    cal_parameters = {
        "Vf_well": 10.0,  # mL
        "Vf_prod": 5.0,  # mL
        "c_prod": 2.15    # mg/mL
    }

    initial_guess = [2.6,111,3,90]
    noise_level = 0.01
    num_iterations = 1000

    mc_reaction2_noisy_plate_reader = monte_carlo_parameter_estimation(data, 
                                                                cal_data, substrates, cal_parameters,
                                                                michaelis_menten2, add_noise_plate_reader, 
                                                                initial_guess, noise_level, num_iterations)


    df_reaction2_noisy_plate_reader = save_results(mc_reaction2_noisy_plate_reader, parameters, save_path=os.path.join(RESULTS_PATH, "experimental_reaction2_noisy_plate_reader_results.csv"))


    print_monte_carlo_info(parameters, df_reaction2_noisy_plate_reader)
    corner_plot_monte_carlo_results(df_reaction2_noisy_plate_reader,parameters)
    correlation_matrix_plot(df_reaction2_noisy_plate_reader,parameters)