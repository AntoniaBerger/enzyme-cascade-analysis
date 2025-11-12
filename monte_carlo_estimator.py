from data_handler import save_results
from parameter_estimator import estimate_parameters
import corner
import numpy as np
import pandas as pd
import scipy.optimize as opt
import numpy.random as random

from process_data import get_time_points, get_concentration_data, get_absorbance_data, get_calibration_slope, process_duplicates, get_processed_data, add_noise


def monte_carlo_parameter_estimation(data, 
                                     cal_data, 
                                     substrate:list[str], 
                                     cal_param, 
                                     model_func, 
                                     noise_function, 
                                     initial_guess, 
                                     noise_level=0.1, 
                                     num_iterations=1000):
    estimated_params = []
    local_data = data.copy()

    for _ in range(num_iterations):
        
        # Add Gaussian noise to the data
        data_noisy = noise_function(local_data, cal_data, substrate, cal_param, noise_level)

        # Estimate parameters using the noisy data
        popt, _ = estimate_parameters(data_noisy, model_func, initial_guess=initial_guess)
        estimated_params.append(popt)

    return np.array(estimated_params)


if __name__ == "__main__":


    # define model function
    def michaelis_menten(S, Vmax, Km1, Km2):
        S1, S2 = S
        return (Vmax * S1 * S2) / ((Km1 + S1) * (Km2 + S2))
    
    # define noise function (signiture must match: data, cal_data, noise_level, cal_parameters)
    def add_noise_plate_reader(data, cal_data, model_param, cal_parameters, noise_level):
        
        time_points = get_time_points(data)
        concentration_data = get_concentration_data(data,model_param)


        ad_data = get_absorbance_data(data)
        ad_data_noisy = add_noise(ad_data,ad_data.columns, noise_level)


        ad_data_noisy = process_duplicates(ad_data_noisy)

        slope, r = get_calibration_slope(cal_data)

        processed_data = get_processed_data(time_points, concentration_data, ad_data_noisy, slope, cal_parameters)

        return processed_data

    # Perform Monte Carlo parameter estimation

    data = pd.read_csv(r"C:\Users\berger\Documents\Projekts\enzyme-cascade-analysis\Experimental_Data\Reaction1\r_1_PD_NAD.csv")
    cal_data = pd.read_csv(r"C:\Users\berger\Documents\Projekts\enzyme-cascade-analysis\Experimental_Data\NADH_Kalibriergerade.csv")
    data_process_params = {
        "substrate": ["PD_mM", "NAD_mM"],
        "Vf_well": 10.0,  # mL
        "Vf_prod": 1.0,  # mL
        "c_prod": 2.2108    # mg/mL
    }
    
    parameters = ["PD_mM", "NAD_mM"]
    initial_guess = [80, 1, 1]


    monte_carlo_results_r1 = monte_carlo_parameter_estimation(data, cal_data, parameters, data_process_params,
                                                               michaelis_menten, add_noise_plate_reader, 
                                                               initial_guess=initial_guess, noise_level=0.01, num_iterations=500)
  
    
    parameters = ["Vmax", "Km1", "Km2"]
    monte_carlo_results_r1 = save_results(monte_carlo_results_r1, parameters, dataset_name="noisy_plate_reader_reaction1")
    # Calculate mean and standard deviation of estimated parameters
    
