import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


def get_data_xlsx(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist.")
    data = pd.read_excel(file_path, header=0)
    return data


def plot_kalibriungs_grade(data_x, data_y, title="Kalibrierungsgerade", xlabel="Konzentration (µM)", ylabel="Absorbance"):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=data_x, y=data_y, color='blue', label='Messwerte')
    sns.regplot(x=data_x, y=data_y, scatter=False, color='red', label='Regressionslinie')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.show()

def get_regression(x, y):
    """
    Perform linear regression on the given data.
    
    Parameters:
    x (pd.Series): Independent variable.
    y (pd.Series): Dependent variable.
    
    Returns:
    tuple: Slope and intercept of the regression line.
    """
    slope, intercept = np.polyfit(x, y, 1)
    return slope, intercept

def get_stats(data_x, data_y):
    """
    Calculate statistics for the given data.
    
    Parameters:
    data_x (pd.Series): Independent variable.
    data_y (pd.Series): Dependent variable.
    
    Returns:
    dict: Dictionary containing slope, intercept, R-squared, and standard error.
    """
    slope, intercept = get_regression(data_x, data_y)
    residuals = data_y - (slope * data_x + intercept)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((data_y - np.mean(data_y))**2)
    r_squared = 1 - (ss_res / ss_tot)
    
    n = len(data_x)
    std_err = np.sqrt(ss_res / (n - 2))
    
    return {
        'r_squared': r_squared,
        'std_err': std_err
    }