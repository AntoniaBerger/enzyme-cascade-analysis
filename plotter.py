import matplotlib.pyplot as plt
import corner

import numpy as np
import datetime

def print_monte_carlo_info(params,monte_carlo_results, save_to_file="", dataset_name=""):
    

    param_means = monte_carlo_results.mean(axis=0)
    param_stds = monte_carlo_results.std(axis=0)
    correlation_matrix = monte_carlo_results[params].corr()

    print("Monte Carlo Parameter Estimation Results:")
    
    if save_to_file:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if dataset_name != "":
            filename = f"Results/parameter_info_{dataset_name}_{timestamp}.txt"
        else:
            filename = f"Results/parameter_info_{timestamp}.txt"
        
        with open(filename, 'w') as f:
            f.write("Monte Carlo Parameter Estimation Results:\n\n")
            for i, param in enumerate(params):
                f.write(f"{param}: {param_means[i]} ± {param_stds[i]}\n")
                print(f"{param}: {param_means[i]} ± {param_stds[i]}")
            f.write("\n")
    
    
    print("\nParameter Correlation Matrix:")
    params = np.array(params, dtype=object)
    correlation_matrix = correlation_matrix.astype(object)

    correlation_matrix = np.insert(correlation_matrix, 0, params, axis=1)
    correlation_matrix = np.insert(correlation_matrix, 0, np.insert(params, 0, ""), axis=0)

    if save_to_file:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if dataset_name != "":
            filename = f"Results/parameter_info_{dataset_name}_{timestamp}.txt"
        else:
            filename = f"Results/parameter_info_{timestamp}.txt"

        with open(filename, 'a') as f:
            f.write("Parameter Correlation Matrix:\n")
            for row in correlation_matrix:
                f.write("\t".join([str(elem) for elem in row]) + "\n")

    print(correlation_matrix)


def corner_plot_monte_carlo_results(monte_carlo_results,parameters,save_to_file="", dataset_name=""):
    # Create corner plot
    data = monte_carlo_results[parameters].values
    figure = corner.corner(data, 
                          labels=parameters,
                          title_kwargs={"fontsize": 12})
    if save_to_file:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if dataset_name != "":
            figure.savefig(f"Results/monte_carlo_corner_plot_{dataset_name}_{timestamp}.png")
        else:
            figure.savefig(f"Results/monte_carlo_corner_plot__{timestamp}.png")

    plt.show()


def correlation_matrix_plot(monte_carlo_results,parameters,save_to_file="", dataset_name=""):
    correlation_matrix = monte_carlo_results[parameters].corr()

    # Only show lower triangle and diagonal
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)
    correlation_matrix_masked = np.ma.array(correlation_matrix, mask=mask)

    fig, ax = plt.subplots()
    cax = ax.matshow(correlation_matrix_masked, cmap='coolwarm')
    fig.colorbar(cax)

    ax.set_xticks(np.arange(len(parameters)))
    ax.set_yticks(np.arange(len(parameters)))
    ax.set_xticklabels(parameters)
    ax.set_yticklabels(parameters)

    for i in range(len(parameters)):
        for j in range(len(parameters)):
            if not mask[i, j]:
                text = ax.text(j, i, f"{correlation_matrix_masked[i, j]:.2f}",
                            ha="center", va="center", color="w")

    ax.set_title("Parameter Correlation Matrix")
    plt.tight_layout()

    if save_to_file:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if dataset_name != "":
            plt.savefig(f"Results/correlation_matrix_plot_{dataset_name}_{timestamp}.png")
        else:
            plt.savefig(f"Results/correlation_matrix_plot__{timestamp}.png")

    plt.show()

if __name__ == "__main__":

    # Example usage with synthetic data
    from artifical_data import reaction1_synthetic_data
    from monte_carlo_estimator import monte_carlo_parameter_estimation

    def michaelis_menten(S, Vmax, Km1, Km2):
        S1, S2 = S
        return (Vmax * S1 * S2) / ((Km1 + S1) * (Km2 + S2))
    
    # Generate synthetic data for testing
    true_parameters = (100, 2, 3)  # Vmax, Km1, Km2
    synthetic_data = reaction1_synthetic_data(true_parameters)

    # Perform Monte Carlo parameter estimation
    monte_carlo_results = monte_carlo_parameter_estimation(synthetic_data, michaelis_menten, noise_level=0.5, num_iterations=500)

    # Print Monte Carlo info
    #print_monte_carlo_info(['Vmax', 'Km1', 'Km2'], monte_carlo_results)

    # Create corner plot
    #coner_plot_monte_carlo_results(monte_carlo_results, true_parameters)

    # Create correlation matrix plot
    correlation_matrix_plot(monte_carlo_results, ['Vmax', 'Km1', 'Km2'])
