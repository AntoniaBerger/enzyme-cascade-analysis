from parameter_estimator import estimate_parameters
import corner
import numpy as np
import pandas as pd
import scipy.optimize as opt
import numpy.random as random


def monte_carlo_parameter_estimation(data, model_func, noise_level=0.1, num_iterations=1000):
    """
    Perform Monte Carlo simulation to estimate parameter uncertainties.

    Parameters:
    data (pd.DataFrame): DataFrame containing the experimental data.
    model_func (callable): The model function to fit to the data.
    noise_level (float): Standard deviation of the Gaussian noise to be added.
    num_iterations (int): Number of Monte Carlo iterations.

    Returns:
    np.ndarray: Array of estimated parameters from each iteration.
    """
    estimated_params = []

    for _ in range(num_iterations):
        # Add Gaussian noise to the data
        noisy_data = data + random.normal(0, noise_level, data.shape)
        
        # Estimate parameters using the noisy data

        popt, _ = estimate_parameters(noisy_data, model_func, initial_guess=[80, 1, 1])
        estimated_params.append(popt)

    return np.array(estimated_params)


if __name__ == "__main__":

    def michaelis_menten(S, Vmax, Km1, Km2):
        S1, S2 = S
        return (Vmax * S1 * S2) / ((Km1 + S1) * (Km2 + S2))
    
    # Generate synthetic data for testing
    true_parameters = (100, 2, 3)  # Vmax, Km1, Km2
    from artifical_data import reaction1_synthetic_data
    synthetic_data = reaction1_synthetic_data(true_parameters)

    # Perform Monte Carlo parameter estimation
    monte_carlo_results = monte_carlo_parameter_estimation(synthetic_data, michaelis_menten, noise_level=0.5, num_iterations=500)
    # Calculate mean and standard deviation of estimated parameters
    param_means = np.mean(monte_carlo_results, axis=0)
    param_stds = np.std(monte_carlo_results, axis=0)
    print("Monte Carlo Parameter Estimation Results:")
    print(f"Vmax: {param_means[0]} ± {param_stds[0]}")
    print(f"Km1: {param_means[1]} ± {param_stds[1]}")
    print(f"Km2: {param_means[2]} ± {param_stds[2]}")

    import matplotlib.pyplot as plt

    # Plot histograms for each parameter
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Vmax histogram
    axes[0].hist(monte_carlo_results[:, 0], bins=30, alpha=0.7, color='blue')
    axes[0].set_xlabel('Vmax')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title(f'Vmax Distribution\nMean: {param_means[0]:.2f} ± {param_stds[0]:.2f}')
    axes[0].axvline(true_parameters[0], color='red', linestyle='--', label='True value')
    axes[0].legend()

    # Km1 histogram
    axes[1].hist(monte_carlo_results[:, 1], bins=30, alpha=0.7, color='green')
    axes[1].set_xlabel('Km1')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title(f'Km1 Distribution\nMean: {param_means[1]:.2f} ± {param_stds[1]:.2f}')
    axes[1].axvline(true_parameters[1], color='red', linestyle='--', label='True value')
    axes[1].legend()

    # Km2 histogram
    axes[2].hist(monte_carlo_results[:, 2], bins=30, alpha=0.7, color='orange')
    axes[2].set_xlabel('Km2')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title(f'Km2 Distribution\nMean: {param_means[2]:.2f} ± {param_stds[2]:.2f}')
    axes[2].axvline(true_parameters[2], color='red', linestyle='--', label='True value')
    axes[2].legend()

    plt.tight_layout()
    plt.show()

    # Create corner plot
    figure = corner.corner(monte_carlo_results, 
                          labels=['Vmax', 'Km1', 'Km2'],
                          show_titles=True,
                          title_kwargs={"fontsize": 12})
    
    plt.show()