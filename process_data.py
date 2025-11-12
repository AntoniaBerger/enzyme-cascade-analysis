import os 
import pandas as pd
import numpy as np
from pyparsing import col
from scipy.stats import linregress
import pickle

def is_linear(x,y, threshold=0.80):
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    is_linear = r_value**2 >= threshold
    if is_linear:
        return slope, r_value**2
    else:
        return False, r_value**2
    
def get_time_points(data):
    
    time_points_strings = data.columns
    time_points = [int(tp.split("_")[1]) for tp in time_points_strings if "data_" in tp]
    
    return time_points

def get_concentration_data(data, concentration_columns):

    concentration_data = data[concentration_columns].drop_duplicates()

    return concentration_data

def get_absorbance_data(data):

    time_points_strings = [dp for dp in data.columns if "data_" in dp]
    absorbance_data = data[time_points_strings]

    return absorbance_data

def get_calibration_slope(data):

    x = data["c"]
    y = data[["ad1", "ad2"]].mean(axis=1)
    
    slope = is_linear(x, y)

    return slope

def get_reaction_slope(time_points, absorbance_values):

    slope, r_value = is_linear(time_points, absorbance_values)

    return slope, r_value

def convert_ad_to_concentration(ad_slope, calc_slope, parameters):

    activity_U_per_mg = (abs(ad_slope) * 60 * parameters["Vf_well"] * parameters["Vf_prod"]) / (calc_slope * parameters["c_prod"]) #! Formel prüfen

    return activity_U_per_mg


def process_duplicates(ad_data):

    data_without_duplicates = pd.DataFrame(columns=ad_data.columns)
    
    values = ad_data.values
    for i in range(0, len(ad_data.values), 2):
        row1 = values[i]
        row2 = values[i+1]

        averaged_row = (row1 + row2) / 2
        data_without_duplicates.loc[i] = averaged_row

    return data_without_duplicates


def get_processed_data(time_points, conc_data, ad_data, cal_scope, substrates, cal_parameters):

    activity_list = []
    for index, row in ad_data.iterrows():
        test = row.values
        # Remove NaN values and corresponding time points
        mask = ~np.isnan(test)
        test = test[mask]
        time_points_filtered = np.array(time_points)[mask]
        
        # Use filtered time points for linear regression
        if len(test) > 1:  # Need at least 2 points for linear regression
            ad_slope, r = is_linear(time_points_filtered, test)
            
            if ad_slope:
                activity = convert_ad_to_concentration(ad_slope, cal_scope, cal_parameters)
                activity_list.append(activity)
                print(f"Row {index}: Activity = {activity} U/mg, R^2 = {r}")
            else:
                print(f"Row {index}: No linear region found.Row {index}: R^2 = {r}")
                conc_data = conc_data.drop(index)
        else:
            print(f"Row {index}: Not enough data points for linear regression.")
            conc_data = conc_data.drop(index)
        
    
    dict_results = { }
    for sub_names in substrates:
        dict_results[sub_names] = conc_data[sub_names]
    dict_results["activity_U/mg"] = activity_list
    
    df = pd.DataFrame(dict_results)

    return df


def add_noise(df, keys, noise_level=0.01):
    noisy_df = df.copy()
    for key in keys:
        if key in noisy_df.columns:
            noise = np.random.normal(0, noise_level, size=noisy_df[key].shape)
            noisy_df[key] += noise
    return noisy_df
    
if __name__ == "__main__":

    data_path = r"C:\Users\berger\Documents\Projekts\enzyme-cascade-analysis\MinimalWorkingExample\Data\Reaction1\r_1_PD_NAD.csv"
    data = pd.read_csv(data_path)
    time_points = get_time_points(data)
    print(time_points)

    concentration_data = get_concentration_data(data,["PD_mM","NAD_mM"])
    print(concentration_data)

    ad_data = get_absorbance_data(data)
    print(ad_data)

    ads_data = pd.read_csv(r"C:\Users\berger\Documents\Projekts\enzyme-cascade-analysis\MinimalWorkingExample\Data\NADH_Kalibriergerade.csv")
    slope, r = get_calibration_slope(ads_data)
    print(slope, r)

    ad_data = process_duplicates(ad_data)
    print(ad_data)

    data_r1 = {
        "time_points": time_points,
        "concentration_data": concentration_data,
        "ad_data": ad_data,
        "calslope": slope,
        "parameters": {
            "substrate": ["PD_mM", "NAD_mM"],
            "Vf_well": 10.0,  # mL
            "Vf_prod": 1.0,  # mL
            "c_prod": 2.2108    # mg/mL
        }
    }

    processed_data = get_processed_data(time_points, concentration_data, ad_data, slope, data_r1["parameters"])

    print(processed_data)
