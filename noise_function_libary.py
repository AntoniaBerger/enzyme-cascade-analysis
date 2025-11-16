import numpy as np
from process_data import add_noise_function, get_time_points, get_concentration_data, get_absorbance_data, get_calibration_slope, process_duplicates, get_processed_data, add_noise, process_duplicates2, get_calibration_slope_with_noise


def no_noise(data, cal_data, substrate, cal_param, noise_level):
    return data

# Now process the synthetic data as if it were experimental data
def add_noise_processed_data(data, cal_data, substrates, cal_parameters, noise_level):
    data_noisy = data.copy()
    rates = data_noisy["activity_U/mg"].values
    noisy_rates = rates + np.random.normal(0, noise_level, size=rates.shape)
    data_noisy["activity_U/mg"] = noisy_rates
    return data_noisy

 # define noise function (signiture must match: data, cal_data, noise_level, cal_parameters)
def add_noise_plate_reader(data, cal_data, substrates, cal_parameters, noise_level):
    
    time_points = get_time_points(data)
    time_points_strings = [dp for dp in data.columns if "data_" in dp]
    
    data_noisy = add_noise(data,time_points_strings, noise_level)
    data_noisy = process_duplicates2(data_noisy,substrates)

    slope = get_calibration_slope(cal_data)
    
    concentration_data = data_noisy[substrates]

    processed_data, regi_results = get_processed_data(time_points, concentration_data, data_noisy[time_points_strings], slope, substrates, cal_parameters)

    return processed_data


def add_noise_rate(data, cal_data, substrates, cal_parameters, noise_level):
    
    time_points = get_time_points(data)
    time_points_strings = [dp for dp in data.columns if "data_" in dp]
    

    data_noisy = data.copy()
    slope = get_calibration_slope(cal_data)
    concentration_data = data_noisy[substrates]

    processed_data, regi_results = get_processed_data(time_points, concentration_data, data_noisy[time_points_strings], slope, substrates, cal_parameters)
    
    processed_data = process_duplicates2(processed_data,substrates)

    rates = processed_data["activity_U/mg"].values
    noisy_rates = rates + np.random.normal(0, noise_level, size=rates.shape)
    processed_data["activity_U/mg"] = noisy_rates

    return processed_data


def full_experiment_processing_with_noise(data, cal_data, substrates, cal_parameters, noise_levels):

    fehler_wage = noise_levels['fehler_wage']
    fehler_pipettieren = noise_levels['fehler_pipettieren']
    fehler_time_points = noise_levels['fehler_time_points']
    fehler_od = noise_levels['fehler_od']


    time_points_strings = [dp for dp in data.columns if "data_" in dp]
    time_points = get_time_points(data)
    noisy_time_points = time_points + np.random.normal(0, fehler_time_points, size=len(time_points))
    
    
    data_noisy = add_noise(data,time_points_strings, fehler_od)
    data_noisy = process_duplicates2(data_noisy,substrates)

    data_noisy = data_noisy.sort_values(by=substrates, ascending=False)
    concentration_data_noisy = data_noisy[substrates].to_numpy()

    n = len(concentration_data_noisy)
    
    for i in range(0, n): 
        concentration_data_noisy[i] = (concentration_data_noisy[i] + np.random.normal(0, fehler_pipettieren)*2*i)

    data_noisy[substrates] = concentration_data_noisy

    slope_noisy = get_calibration_slope_with_noise(cal_data,noise_levels)
    
    processed_data, regi_results = get_processed_data(noisy_time_points, data_noisy[substrates], data_noisy[time_points_strings], slope_noisy, substrates, cal_parameters)
    
    return processed_data