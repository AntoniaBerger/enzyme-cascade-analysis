import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from artifical_data import reaction1_synthetic_data
from monte_carlo_estimator import monte_carlo_parameter_estimation
from plotter import print_monte_carlo_info, corner_plot_monte_carlo_results, correlation_matrix_plot
from data_handler import save_results

parameters = ['Vmax', 'Km1', 'Km2']

def michaelis_menten(S, *parameters):
    S1, S2 = S
    Vmax, Km1, Km2 = parameters
    
    return (Vmax * S1 * S2) / ((Km1 + S1) * (Km2 + S2))
    
# Generate synthetic data for testing
true_parameters = (100, 2, 3)  # Vmax, Km1, Km2
synthetic_data_r1 = reaction1_synthetic_data(true_parameters)

# Perform Monte Carlo parameter estimation
monte_carlo_results_r1 = monte_carlo_parameter_estimation(synthetic_data_r1, michaelis_menten, noise_level=0.5, num_iterations=500)
monte_carlo_results_r1 = save_results(monte_carlo_results_r1, parameters, dataset_name="artificial_reaction1")

# Print Monte Carlo info
result_path = os.path.dirname(os.path.abspath(__file__)) + "/Results/"

print_monte_carlo_info(parameters, monte_carlo_results_r1, save_to_file=result_path, dataset_name="artificial_reaction1")

# Create corner plot
corner_plot_monte_carlo_results(monte_carlo_results_r1, parameters, save_to_file=result_path, dataset_name="artificial_reaction1")

# Create correlation matrix plot
correlation_matrix_plot(monte_carlo_results_r1, parameters, save_to_file=result_path, dataset_name="artificial_reaction1")