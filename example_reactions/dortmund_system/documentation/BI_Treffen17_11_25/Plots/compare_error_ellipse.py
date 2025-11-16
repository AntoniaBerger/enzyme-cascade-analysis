import corner 
import pandas as pd
import os
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt

from scipy.stats import chi2



MC_RESULTS_PATH = r"C:\Users\berger\Documents\Projekts\enzyme-cascade-analysis\example_reactions\dortmund_system\results"


# Reaction 1

df_reaction1_NAD_full = pd.read_csv(os.path.join(MC_RESULTS_PATH, "MC_reaction1_full_experiment_NAD.csv"))
df_reaction1_NAD_rate = pd.read_csv(os.path.join(MC_RESULTS_PATH, "MC_reaction1_rate_noise_NAD.csv"))

df_reaction1_PD_full = pd.read_csv(os.path.join(MC_RESULTS_PATH, "MC_reaction1_full_experiment_PD.csv"))
df_reaction1_PD_rate = pd.read_csv(os.path.join(MC_RESULTS_PATH, "MC_reaction1_rate_noise_PD.csv"))

# Reaction 2
df_reaction2_HP_full = pd.read_csv(os.path.join(MC_RESULTS_PATH, "MC_reaction2_full_experiment_HP.csv"))
df_reaction2_HP_rate = pd.read_csv(os.path.join(MC_RESULTS_PATH, "MC_reaction2_rate_noise_HP.csv"))

df_reaction2_NADH_full = pd.read_csv(os.path.join(MC_RESULTS_PATH, "MC_reaction2_full_experiment_NADH.csv"))
df_reaction2_NADH_rate = pd.read_csv(os.path.join(MC_RESULTS_PATH, "MC_reaction2_rate_noise_NADH.csv"))

df_reaction2_PD_full = pd.read_csv(os.path.join(MC_RESULTS_PATH, "MC_reaction2_full_experiment_PD.csv"))
df_reaction2_PD_rate = pd.read_csv(os.path.join(MC_RESULTS_PATH, "MC_reaction2_rate_noise_PD.csv"))

# Reaction 3
df_reaction3_LACTOL_full = pd.read_csv(os.path.join(MC_RESULTS_PATH, "MC_reaction3_full_experiment_LACTOL.csv"))
df_reaction3_LACTOL_rate = pd.read_csv(os.path.join(MC_RESULTS_PATH, "MC_reaction3_rate_noise_LACTOL.csv"))

df_reaction3_NAD_full = pd.read_csv(os.path.join(MC_RESULTS_PATH, "MC_reaction3_full_experiment_NAD.csv"))
df_reaction3_NAD_rate = pd.read_csv(os.path.join(MC_RESULTS_PATH, "MC_reaction3_rate_noise_NAD.csv"))


PLOT_RESULT_PATH = r"C:\Users\berger\Documents\Projekts\enzyme-cascade-analysis\example_reactions\dortmund_system\documentation\BI_Treffen17_11_25\Plots"

data_frames = [
    [df_reaction1_NAD_full, df_reaction1_NAD_rate],
    [df_reaction1_PD_full, df_reaction1_PD_rate],
    [df_reaction2_HP_full, df_reaction2_HP_rate],
    [df_reaction2_NADH_full, df_reaction2_NADH_rate],
    [df_reaction2_PD_full, df_reaction2_PD_rate],
    [df_reaction2_PD_full, df_reaction2_PD_rate],
    [df_reaction3_LACTOL_full, df_reaction3_LACTOL_rate],
    [df_reaction3_NAD_full, df_reaction3_NAD_rate]
]


parameter_names_list = [
    ['Vmax', 'Km1'],
    ['Vmax', 'Km2'],
    ['Vmax', 'Km1'],
    ['Vmax', 'Km2'],
    ['Vmax', 'Ki'],
    ['Vmax','Km1'],
    ['Vmax', 'Km1'],
    ['Vmax', 'Km2']
]


plot_titles = [
    "Reaction 1: NAD Full Experiment vs Rate Noise \n Vmax und Km2",
    "Reaction 1: PD Full Experiment vs Rate Noise \n Vmax und Km2",
    "Reaction 2: HP Full Experiment vs Rate Noise \n Vmax und Km2",
    "Reaction 2: NADH Full Experiment vs Rate Noise \n Vmax und Km2",
    "Reaction 2: PD Full Experiment vs Rate Noise \n Vmax und Kmi",
    "Reaction 2: PD Full Experiment vs Rate Noise \n Vmax und Km2",
    "Reaction 3: LACTOL Full Experiment vs Rate Noise \n Vmax und Km2",
    "Reaction 3: NAD Full Experiment vs Rate Noise \n Vmax und Km2"
]

file_names = [
    "compare_ellipse_reaction1_NAD_full_vs_rate_noise_vmax_Km2.png",
    "compare_ellipse_reaction1_PD_full_vs_rate_noise_vmax_Km2.png",
    "compare_ellipse_reaction2_HP_full_vs_rate_noise_vmax_Km2.png",
    "compare_ellipse_reaction2_NADH_full_vs_rate_noise_vmax_Km2.png",
    "compare_ellipse_reaction2_PD_full_vs_rate_noise_vmax_Kmi.png",
    "compare_ellipse_reaction2_PD_full_vs_rate_noise_vmax_Km2.png",
    "compare_ellipse_reaction3_LACTOL_full_vs_rate_noise_vmax_Km2.png",
    "compare_ellipse_reaction3_NAD_full_vs_rate_noise_vmax_Km2.png"
]


# ...existing code...

for (df_full, df_rate), parameter_names, title, file_name in zip(data_frames, parameter_names_list, plot_titles, file_names):
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Extract parameter data
    x_full = df_full[parameter_names[0]]
    y_full = df_full[parameter_names[1]]
    x_rate = df_rate[parameter_names[0]]
    y_rate = df_rate[parameter_names[1]]
    
    
    # Calculate and plot error ellipses (1 and 2 sigma)
    from matplotlib.patches import Ellipse
    import numpy as np
    
    # Function to calculate ellipse parameters
    def calculate_ellipse_params(x, y, n_std):
        cov = np.cov(x, y)
        vals, vecs = np.linalg.eigh(cov)
        angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        width, height = 2 * n_std * np.sqrt(vals)
        return width, height, angle
    
    # 1 sigma ellipses
    w1_full, h1_full, a1_full = calculate_ellipse_params(x_full, y_full, 1)
    w1_rate, h1_rate, a1_rate = calculate_ellipse_params(x_rate, y_rate, 1)
    # Add markers for mean values
    ax.plot(x_full.mean(), y_full.mean(), marker='x', markersize=10, markeredgewidth=3, color='darkblue', label='Mittelwert Vollständiges Experiment')
    ax.plot(x_rate.mean(), y_rate.mean(), marker='x', markersize=10, markeredgewidth=3, color='darkred', label='Mittelwert Rauschen auf Reaktionsraten')
    ellipse1_full = Ellipse((x_full.mean(), y_full.mean()), w1_full, h1_full, angle=a1_full, 
                           facecolor='blue', alpha=0.3, edgecolor='blue', linewidth=2)
    ellipse1_rate = Ellipse((x_rate.mean(), y_rate.mean()), w1_rate, h1_rate, angle=a1_rate, 
                           facecolor='red', alpha=0.3, edgecolor='red', linewidth=2)
    
    # 2 sigma ellipses
    w2_full, h2_full, a2_full = calculate_ellipse_params(x_full, y_full, 2)
    w2_rate, h2_rate, a2_rate = calculate_ellipse_params(x_rate, y_rate, 2)
    
    ellipse2_full = Ellipse((x_full.mean(), y_full.mean()), w2_full, h2_full, angle=a2_full, 
                           facecolor='blue', alpha=0.15, edgecolor='blue', linewidth=1, linestyle='--')
    ellipse2_rate = Ellipse((x_rate.mean(), y_rate.mean()), w2_rate, h2_rate, angle=a2_rate, 
                           facecolor='red', alpha=0.15, edgecolor='red', linewidth=1, linestyle='--')
    
    # Add ellipses to plot
    ax.add_patch(ellipse1_full)
    ax.add_patch(ellipse1_rate)
    ax.add_patch(ellipse2_full)
    ax.add_patch(ellipse2_rate)
    
    # Set axis limits based on ellipses (not data points)
    # Calculate ellipse boundaries considering rotation
    import math
    
    def get_ellipse_bounds(center_x, center_y, width, height, angle_deg):
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        
        # Calculate rotated ellipse bounds
        a = width / 2
        b = height / 2
        
        dx = math.sqrt((a * cos_a)**2 + (b * sin_a)**2)
        dy = math.sqrt((a * sin_a)**2 + (b * cos_a)**2)
        
        return center_x - dx, center_x + dx, center_y - dy, center_y + dy
    
    # Get bounds for all 2-sigma ellipses
    x_min_full, x_max_full, y_min_full, y_max_full = get_ellipse_bounds(
        x_full.mean(), y_full.mean(), w2_full, h2_full, a2_full)
    x_min_rate, x_max_rate, y_min_rate, y_max_rate = get_ellipse_bounds(
        x_rate.mean(), y_rate.mean(), w2_rate, h2_rate, a2_rate)
    
    # Set limits with padding
    padding = 0.1
    x_min = min(x_min_full, x_min_rate)
    x_max = max(x_max_full, x_max_rate)
    y_min = min(y_min_full, y_min_rate)
    y_max = max(y_max_full, y_max_rate)
    
    x_range = x_max - x_min
    y_range = y_max - y_min
    
    ax.set_xlim(x_min - padding * x_range, x_max + padding * x_range)
    ax.set_ylim(y_min - padding * y_range, y_max + padding * y_range)
    
    # Set labels and title
    ax.set_xlabel(parameter_names[0])
    ax.set_ylabel(parameter_names[1])
    ax.set_title(title)
    
    # Add legend
    legend_elements = [
        Line2D([0], [0], color='blue', lw=2, label='Vollständiges Experiment'),
        Line2D([0], [0], color='red', lw=2, label='Rauschen auf Reaktionsraten'),
        Line2D([0], [0], marker='x', color='darkblue', markeredgewidth=3, linestyle='None', label='Mittelwert Full'),
        Line2D([0], [0], marker='x', color='darkred', markeredgewidth=3, linestyle='None', label='Mittelwert Rate'),
        Line2D([0], [0], color='black', lw=2, label='1σ'),
        Line2D([0], [0], color='black', lw=1, linestyle='--', label='2σ'
        )
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Save and show
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_RESULT_PATH, file_name), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()