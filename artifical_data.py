import numpy as np
import pandas as pd

def reaction1_synthetic_data(parameters, noise_level=0.01, num_points=50):
    Vmax, Km1, Km2 = parameters
    S1 = np.linspace(0.1, 10, num_points)
    S2 = np.linspace(0.1, 10, num_points)
    S1_grid, S2_grid = np.meshgrid(S1, S2)
    
    # Michaelis-Menten equation
    data = []
    for s1 in S1: 
        for s2 in S2:
            rate = (Vmax * s1 * s2) / ((Km1 + s1) * (Km2 + s2))
            data.append({'S1': s1, 'S2': s2, 'rate': rate})

    df = pd.DataFrame(data)

    return df



def reaction2_synthetic_data(parameters, noise_level=0.5, num_points=50):
    Vmax, Km1, Km2, Ki1 = parameters
    S1 = np.linspace(0.1, 10, num_points)
    S2 = np.linspace(0.1, 10, num_points)
    S3 = np.linspace(0.1, 10, num_points)
    
    data = []
    for s1 in S1:
        for s2 in S2:
            for s3 in S3:
                rate = (Vmax * s1 * s2) / ((Km1 + s1 + s3/Ki1) * (Km2 + s2))
                data.append({'S1': s1, 'S2': s2, 'S3': s3, 'rate': rate})
    df = pd.DataFrame(data)

    return df

if __name__ == "__main__":
    # Example usage
    parameters = (100, 2, 3)  # Vmax, Km1, Km2
    synthetic_data = reaction1_synthetic_data(parameters)

    import matplotlib.pyplot as plt

    # Convert the DataFrame to numpy arrays for plotting
    S1_unique = synthetic_data["S1"].unique()
    S2_unique = synthetic_data["S2"].unique()
    
    S1_grid, S2_grid = np.meshgrid(S1_unique, S2_unique)

    # Reshape rates to match the grid
    rates = synthetic_data["rate"].values.reshape(len(S2_unique), len(S1_unique))

    # Create the plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface
    surf = ax.plot_surface(S1_grid, S2_grid, rates, 
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