import numpy as np
import pandas as pd

def reaction1_synthetic_data(parameters, noise_level=0.5, num_points=50):
    Vmax, Km1, Km2 = parameters
    S1 = np.linspace(0.1, 10, num_points)
    S2 = np.linspace(0.1, 10, num_points)
    S1_grid, S2_grid = np.meshgrid(S1, S2)
    
    # Michaelis-Menten equation
    rates = (Vmax * S1_grid * S2_grid) / ((Km1 + S1_grid) * (Km2 + S2_grid))
    
    # Add noise
    noise = np.random.normal(0, noise_level, rates.shape)
    noisy_rates = rates + noise

    df = pd.DataFrame(noisy_rates, columns=[f'S2_{s2:.2f}' for s2 in S2], index=[f'S1_{s1:.2f}' for s1 in S1])
    df.index.name = 'S1'
    df.columns.name = 'S2'

    return df

if __name__ == "__main__":
    # Example usage
    parameters = (100, 2, 3)  # Vmax, Km1, Km2
    synthetic_data = reaction1_synthetic_data(parameters)

    import matplotlib.pyplot as plt

    # Convert the DataFrame to numpy arrays for plotting
    S1_values = [float(idx.split('_')[1]) for idx in synthetic_data.index]
    S2_values = [float(col.split('_')[1]) for col in synthetic_data.columns]
    S1_grid, S2_grid = np.meshgrid(S2_values, S1_values)

    # Create the plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface
    surf = ax.plot_surface(S1_grid, S2_grid, synthetic_data.values, 
                        cmap='viridis', alpha=0.8)

    # Add labels and title
    ax.set_xlabel('S2 concentration')
    ax.set_ylabel('S1 concentration')
    ax.set_zlabel('Reaction rate')
    ax.set_title('Synthetic Enzyme Kinetic Data')

    # Add colorbar
    plt.colorbar(surf, ax=ax, shrink=0.5)

    plt.tight_layout()
    plt.show()