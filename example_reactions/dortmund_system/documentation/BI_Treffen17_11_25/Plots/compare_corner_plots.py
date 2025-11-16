import corner 
import pandas as pd
import os
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt

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
    [df_reaction3_LACTOL_full, df_reaction3_LACTOL_rate],
    [df_reaction3_NAD_full, df_reaction3_NAD_rate]
]

parameter_names_list = [
    ['Vmax', 'Km1'],
    ['Vmax', 'Km2'],
    ['Vmax', 'Km1'],
    ['Vmax', 'Km2'],
    ['Vmax', 'Km1', 'Ki'],
    ['Vmax', 'Km1'],
    ['Vmax', 'Km2']
]

plot_titles = [
    "Reaction 1 NAD ",
    "Reaction 1 PD ",
    "Reaction 2 HP ",
    "Reaction 2 NADH ",
    "Reaction 2 PD ",
    "Reaction 3 LACTOL ",
    "Reaction 3 NAD "
]

file_names = [
    "reaction1_NAD_full_vs_rate.png",
    "reaction1_PD_full_vs_rate.png",
    "reaction2_HP_full_vs_rate.png",
    "reaction2_NADH_full_vs_rate.png",
    "reaction2_PD_full_vs_rate.png",
    "reaction3_LACTOL_full_vs_rate.png",
    "reaction3_NAD_full_vs_rate.png"
]

for (df_full, df_rate), parameter_names, title, file_name in zip(data_frames, parameter_names_list, plot_titles, file_names):
    samples_full = df_full[parameter_names].values
    samples_rate = df_rate[parameter_names].values

    figure = corner.corner(samples_full, labels=parameter_names, color='blue', label_kwargs={"fontsize": 12}, 
                           plot_datapoints=False, fill_contours=True, levels=(0.68, 0.95), 
                           title_kwargs={"fontsize": 14}, title_fmt=".2f")

    corner.corner(samples_rate, labels=parameter_names, color='red', label_kwargs={"fontsize": 12}, 
                  plot_datapoints=False, fill_contours=True, levels=(0.68, 0.95), 
                  title_kwargs={"fontsize": 14}, title_fmt=".2f", fig=figure)
    
    axes = figure.get_axes()
    for ax in axes:
        ax.tick_params(axis='both', which='major', labelsize=10)
    
    # Add legend
    legend_elements = [Line2D([0], [0], color='blue', lw=2, label='Alle Rauschquellen'),
                      Line2D([0], [0], color='red', lw=2, label='Nur Ratenrauschen')]
    figure.legend(handles=legend_elements, fontsize=12, bbox_to_anchor=(0.95, 0.8))

    figure.suptitle(title, fontsize=16, y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    figure.savefig(os.path.join(PLOT_RESULT_PATH, file_name))
    plt.show()


