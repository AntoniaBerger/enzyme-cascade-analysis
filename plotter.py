import matplotlib.pyplot as plt
import corner

import numpy as np
import datetime
import pandas as pd

def print_monte_carlo_info(params,monte_carlo_results, save_to_file="", dataset_name=""):
    

    param_means = monte_carlo_results.mean(axis=0)
    param_stds = monte_carlo_results.std(axis=0)
    correlation_matrix = monte_carlo_results[params].corr()

    print("Monte Carlo Parameter Estimation Results:")
    for i, param in enumerate(params):
        print(f"{param}: {param_means[i]} ± {param_stds[i]}")

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


def corner_plot_monte_carlo_results(monte_carlo_results,parameters,save_to_file="", title="Corner Plot of Monte Carlo Results"):
    # Create corner plot
    data = monte_carlo_results[parameters].values
    figure = corner.corner(data, 
                          labels=parameters,
                          title_kwargs={"fontsize": 12},
                          title=title)
    if save_to_file:
        figure.savefig(save_to_file)

    plt.show()


def correlation_matrix_plot(monte_carlo_results,parameters,save_to_file="", titel="Parameter Correlation Matrix"):
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

    ax.set_title(titel)
    plt.tight_layout()

    if save_to_file:
        plt.savefig(save_to_file)

    plt.show()


def compare_error_ellipses(monte_carlo_results1, monte_carlo_results2, parameters, 
                          labels=None, save_to_file="", title="Comparison of Error Ellipses",
                          zoom_factor=3.0, show_data_points=True, ellipse_scale=2.0,
                          min_ellipse_size=0.05, manual_limits=None):
    """
    Vergleicht Fehlerellipsen zwischen zwei Monte Carlo Simulationen mit besserer Sichtbarkeit.
    
    Args:
        monte_carlo_results1: DataFrame mit ersten Monte Carlo Ergebnissen
        monte_carlo_results2: DataFrame mit zweiten Monte Carlo Ergebnissen
        parameters: Liste von Parameternamen für Vergleich (genau 2 Parameter)
        labels: Labels für die beiden Simulationen ['Simulation 1', 'Simulation 2']
        save_to_file: Pfad zum Speichern des Plots
        title: Titel des Plots
        zoom_factor: Faktor um die Achsen zu erweitern (größer = mehr Zoom out)
        show_data_points: Ob Datenpunkte gezeigt werden sollen
        ellipse_scale: Skalierungsfaktor für Ellipsengröße (größer = sichtbarer)
    """
    import matplotlib.patches as patches
    from scipy.stats import chi2
    
    if len(parameters) != 2:
        raise ValueError("Bitte genau 2 Parameter für Ellipsen-Vergleich angeben")
    
    if labels is None:
        labels = ['Simulation 1', 'Simulation 2']
    
    # Daten für beide Simulationen extrahieren
    data1 = monte_carlo_results1[parameters].values
    data2 = monte_carlo_results2[parameters].values
    
    # Statistiken berechnen
    mean1 = np.mean(data1, axis=0)
    mean2 = np.mean(data2, axis=0)
    cov1 = np.cov(data1.T)
    cov2 = np.cov(data2.T)
    
    # Plot erstellen
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Konfidenzlevels für Ellipsen
    confidence_levels = [0.68, 0.95]  # 1σ und 2σ
    colors = ['blue', 'red']
    alphas = [0.6, 0.3]  # Erhöhte Sichtbarkeit
    
    # Für beide Simulationen
    datasets = [(data1, mean1, cov1, labels[0], colors[0]), 
                (data2, mean2, cov2, labels[1], colors[1])]
    
    all_data = np.vstack([data1, data2])  # Für Achsenlimits
    all_ellipse_points = []  # Sammle Ellipsenpunkte für Achsenlimits
    
    for data, mean, cov, label, color in datasets:
        # Scatter plot der Datenpunkte (reduziert für Übersichtlichkeit)
        if show_data_points:
            # Nur jeden 20. Punkt zeigen
            sample_indices = np.arange(0, len(data), max(1, len(data)//50))
            ax.scatter(data[sample_indices, 0], data[sample_indices, 1], 
                      alpha=0.5, s=30, color=color, label=f'{label} (samples)')
        
        # Mittelwert markieren (sehr groß und auffällig)
        ax.plot(mean[0], mean[1], 'o', color=color, markersize=20, 
                markeredgecolor='black', markeredgewidth=3, 
                label=f'{label} (mean)', zorder=10)
        
        # Fehlerellipsen für verschiedene Konfidenzlevel
        for i, confidence in enumerate(confidence_levels):
            # Chi-squared Wert für gegebenes Konfidenzlevel
            chi2_val = chi2.ppf(confidence, df=2)
            
            # Eigenwerte und Eigenvektoren der Kovarianzmatrix
            eigenvals, eigenvecs = np.linalg.eigh(cov)
            
            # Winkel der Hauptachse
            angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
            
            # Halbachsen der Ellipse (mit Skalierung für bessere Sichtbarkeit)
            width = 2 * np.sqrt(chi2_val * eigenvals[0]) * ellipse_scale
            height = 2 * np.sqrt(chi2_val * eigenvals[1]) * ellipse_scale
            
            # Mindestgröße für bessere Sichtbarkeit
            data_range_x = np.ptp(all_data[:, 0])  # Range der x-Daten
            data_range_y = np.ptp(all_data[:, 1])  # Range der y-Daten
            min_width = data_range_x * min_ellipse_size   # Mindestens % der Datenrange
            min_height = data_range_y * min_ellipse_size  # Mindestens % der Datenrange
            
            width = max(width, min_width)
            height = max(height, min_height)
            
            # Ellipse erstellen
            ellipse = patches.Ellipse(mean, width, height, angle=angle,
                                    facecolor=color, alpha=alphas[i],
                                    edgecolor=color, linewidth=4,
                                    label=f'{label} ({int(confidence*100)}% CI)')
            ax.add_patch(ellipse)
            
            # Sammle Ellipsenpunkte für Achsenlimits
            ellipse_radius = max(width, height) / 2
            all_ellipse_points.append([mean[0] - ellipse_radius, mean[1] - ellipse_radius])
            all_ellipse_points.append([mean[0] + ellipse_radius, mean[1] + ellipse_radius])
    
    # Intelligente Achsenlimits basierend auf Daten und Ellipsen
    if manual_limits is not None:
        # Verwende manuelle Limits: [x_min, x_max, y_min, y_max]
        ax.set_xlim(manual_limits[0], manual_limits[1])
        ax.set_ylim(manual_limits[2], manual_limits[3])
    else:
        # Automatische Limits basierend auf Mittelwerten und Ellipsen
        all_means = np.array([mean1, mean2])
        
        # Berechne Limits basierend auf Mittelwerten und Standardabweichungen
        x_center = np.mean(all_means[:, 0])
        y_center = np.mean(all_means[:, 1])
        
        # Verwende die größten Standardabweichungen für Limits
        x_std = max(np.sqrt(cov1[0,0]), np.sqrt(cov2[0,0]))
        y_std = max(np.sqrt(cov1[1,1]), np.sqrt(cov2[1,1]))
        
        # Setze Limits basierend auf 3-4 Standardabweichungen
        x_margin = x_std * 4 * ellipse_scale
        y_margin = y_std * 4 * ellipse_scale
        
        ax.set_xlim(x_center - x_margin, x_center + x_margin)
        ax.set_ylim(y_center - y_margin, y_center + y_margin)
    
    # Plot formatieren
    ax.set_xlabel(parameters[0], fontsize=14, fontweight='bold')
    ax.set_ylabel(parameters[1], fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.4, linewidth=1)
    
    # Legende außerhalb des Plots
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
    
    # Tick-Größe erhöhen
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    plt.tight_layout()
    
    if save_to_file:
        plt.savefig(save_to_file, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    # Detaillierte numerische Ausgabe
    print(f"\n{'='*50}")
    print(f"VERGLEICH DER SIMULATIONEN")
    print(f"{'='*50}")
    
    for i, (data, mean, cov, label) in enumerate([(data1, mean1, cov1, labels[0]), 
                                                  (data2, mean2, cov2, labels[1])]):
        print(f"\n{label}:")
        print(f"  Anzahl Samples: {len(data)}")
        print(f"  {parameters[0]}: {mean[0]:.4f} ± {np.sqrt(cov[0,0]):.4f}")
        print(f"  {parameters[1]}: {mean[1]:.4f} ± {np.sqrt(cov[1,1]):.4f}")
        
        # Korrelation
        correlation = cov[0,1]/(np.sqrt(cov[0,0]*cov[1,1]))
        print(f"  Korrelation: {correlation:.4f}")
        
        # Ellipsenflächen
        eigenvals, _ = np.linalg.eigh(cov)
        area_68 = np.pi * np.sqrt(eigenvals[0] * eigenvals[1]) * chi2.ppf(0.68, df=2)
        area_95 = np.pi * np.sqrt(eigenvals[0] * eigenvals[1]) * chi2.ppf(0.95, df=2)
        print(f"  68% Ellipsenfläche: {area_68:.4f}")
        print(f"  95% Ellipsenfläche: {area_95:.4f}")
    
    # Vergleich der Unsicherheiten
    print(f"\nVERGLEICH DER UNSICHERHEITEN:")
    ratio_0 = np.sqrt(cov2[0,0]) / np.sqrt(cov1[0,0])
    ratio_1 = np.sqrt(cov2[1,1]) / np.sqrt(cov1[1,1])
    print(f"  {parameters[0]} Unsicherheitsverhältnis ({labels[1]}/{labels[0]}): {ratio_0:.3f}")
    print(f"  {parameters[1]} Unsicherheitsverhältnis ({labels[1]}/{labels[0]}): {ratio_1:.3f}")


def compare_error_ellipses_enhanced(monte_carlo_results1, monte_carlo_results2, parameters,
                                  labels=None, true_values=None, save_to_file="", 
                                  title="Enhanced Error Ellipse Comparison"):
    """
    Erweiterte Version mit wahren Werten und zusätzlichen Features.
    """
    # Erst den normalen Plot
    compare_error_ellipses(monte_carlo_results1, monte_carlo_results2, parameters,
                          labels, "", title, zoom_factor=2.5, ellipse_scale=1.5)
    
    # Dann mit wahren Werten falls verfügbar
    if true_values is not None and len(true_values) >= 2:
        fig = plt.gcf()
        ax = plt.gca()
        
        # Wahre Werte als großes X markieren
        ax.plot(true_values[0], true_values[1], 'x', color='black', 
                markersize=25, markeredgewidth=5, 
                label='True Values', zorder=15)
        
        # Legende aktualisieren
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
        
        if save_to_file:
            plt.savefig(save_to_file, dpi=300, bbox_inches='tight')
        
        plt.show()


def compare_multiple_ellipses(monte_carlo_results_list, parameters, labels=None, 
                            save_to_file="", title="Comparison of Multiple Error Ellipses",
                            ellipse_scale=2.0, min_ellipse_size=0.05, manual_limits=None,
                            show_data_points=True, confidence_level=0.95):
    """
    Vergleicht Fehlerellipsen für mehrere Simulationen (erweiterte Version).
    
    Args:
        monte_carlo_results_list: Liste von DataFrames mit Monte Carlo Ergebnissen
        parameters: Liste von Parameternamen (genau 2)
        labels: Liste von Labels für die Simulationen
        save_to_file: Pfad zum Speichern
        title: Titel des Plots
        ellipse_scale: Skalierungsfaktor für Ellipsengröße
        min_ellipse_size: Mindestgröße der Ellipsen als Prozent der Datenrange
        manual_limits: [x_min, x_max, y_min, y_max] für manuelle Achsenlimits
        show_data_points: Ob Datenpunkte gezeigt werden sollen
        confidence_level: Konfidenzlevel für Ellipsen (Standard: 0.95)
    """
    import matplotlib.patches as patches
    from scipy.stats import chi2
    
    if len(parameters) != 2:
        raise ValueError("Bitte genau 2 Parameter für Ellipsen-Vergleich angeben")
    
    n_simulations = len(monte_carlo_results_list)
    
    if labels is None:
        labels = [f'Simulation {i+1}' for i in range(n_simulations)]
    
    # Farben für verschiedene Simulationen
    colors = plt.cm.tab10(np.linspace(0, 1, n_simulations))
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Sammle alle Daten für Limits
    all_data_list = [results[parameters].values for results in monte_carlo_results_list]
    all_data = np.vstack(all_data_list)
    all_means = []
    all_covs = []
    
    for i, (results, label, color) in enumerate(zip(monte_carlo_results_list, labels, colors)):
        data = results[parameters].values
        mean = np.mean(data, axis=0)
        cov = np.cov(data.T)
        all_means.append(mean)
        all_covs.append(cov)
        
        # Scatter plot (nur wenige Punkte zeigen)
        if show_data_points:
            sample_indices = np.random.choice(len(data), min(50, len(data)), replace=False)
            ax.scatter(data[sample_indices, 0], data[sample_indices, 1], 
                      alpha=0.4, s=20, color=color, label=f'{label} (samples)')
        
        # Mittelwert markieren
        ax.plot(mean[0], mean[1], 'o', color=color, markersize=15, 
                markeredgecolor='black', markeredgewidth=2, 
                label=f'{label} (mean)', zorder=10)
        
        # Konfidenzellipse
        chi2_val = chi2.ppf(confidence_level, df=2)
        eigenvals, eigenvecs = np.linalg.eigh(cov)
        angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
        width = 2 * np.sqrt(chi2_val * eigenvals[0]) * ellipse_scale
        height = 2 * np.sqrt(chi2_val * eigenvals[1]) * ellipse_scale
        
        # Mindestgröße für bessere Sichtbarkeit
        data_range_x = np.ptp(all_data[:, 0])
        data_range_y = np.ptp(all_data[:, 1])
        min_width = data_range_x * min_ellipse_size
        min_height = data_range_y * min_ellipse_size
        
        width = max(width, min_width)
        height = max(height, min_height)
        
        ellipse = patches.Ellipse(mean, width, height, angle=angle,
                                facecolor=color, alpha=0.3,
                                edgecolor=color, linewidth=3,
                                label=f'{label} ({int(confidence_level*100)}% CI)')
        ax.add_patch(ellipse)
    
    # Intelligente Achsenlimits
    if manual_limits is not None:
        ax.set_xlim(manual_limits[0], manual_limits[1])
        ax.set_ylim(manual_limits[2], manual_limits[3])
    else:
        # Automatische Limits basierend auf allen Mittelwerten und Standardabweichungen
        all_means = np.array(all_means)
        x_center = np.mean(all_means[:, 0])
        y_center = np.mean(all_means[:, 1])
        
        # Verwende die größten Standardabweichungen
        max_x_std = max([np.sqrt(cov[0,0]) for cov in all_covs])
        max_y_std = max([np.sqrt(cov[1,1]) for cov in all_covs])
        
        x_margin = max_x_std * 4 * ellipse_scale
        y_margin = max_y_std * 4 * ellipse_scale
        
        ax.set_xlim(x_center - x_margin, x_center + x_margin)
        ax.set_ylim(y_center - y_margin, y_center + y_margin)
    
    # Plot formatieren
    ax.set_xlabel(parameters[0], fontsize=14, fontweight='bold')
    ax.set_ylabel(parameters[1], fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.4, linewidth=1)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    plt.tight_layout()
    
    if save_to_file:
        plt.savefig(save_to_file, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    # Detaillierte numerische Ausgabe für alle Simulationen
    print(f"\n{'='*60}")
    print("VERGLEICH ALLER SIMULATIONEN")
    print(f"{'='*60}")
    
    for i, (results, label) in enumerate(zip(monte_carlo_results_list, labels)):
        data = results[parameters].values
        mean = all_means[i]
        cov = all_covs[i]
        
        print(f"\n{i+1}. {label}:")
        print(f"   Anzahl Samples: {len(data)}")
        print(f"   {parameters[0]}: {mean[0]:.4f} ± {np.sqrt(cov[0,0]):.4f}")
        print(f"   {parameters[1]}: {mean[1]:.4f} ± {np.sqrt(cov[1,1]):.4f}")
        
        correlation = cov[0,1]/(np.sqrt(cov[0,0]*cov[1,1]))
        print(f"   Korrelation: {correlation:.4f}")
        
        eigenvals, _ = np.linalg.eigh(cov)
        area = np.pi * np.sqrt(eigenvals[0] * eigenvals[1]) * chi2.ppf(confidence_level, df=2)
        print(f"   {int(confidence_level*100)}% Ellipsenfläche: {area:.4f}")
    
    # Vergleiche alle Paare
    print(f"\nVERGLEICH DER UNSICHERHEITEN:")
    for i in range(len(labels)):
        for j in range(i+1, len(labels)):
            ratio_x = np.sqrt(all_covs[j][0,0]) / np.sqrt(all_covs[i][0,0])
            ratio_y = np.sqrt(all_covs[j][1,1]) / np.sqrt(all_covs[i][1,1])
            print(f"   {labels[j]} vs {labels[i]}:")
            print(f"     {parameters[0]} Verhältnis: {ratio_x:.3f}")
            print(f"     {parameters[1]} Verhältnis: {ratio_y:.3f}")

if __name__ == "__main__":

    # Example usage with synthetic data
    from artifical_data import reaction1_synthetic_data
    from monte_carlo_estimator import monte_carlo_parameter_estimation

    monte_carlo_results1 = pd.read_csv(r"C:\Users\berger\Documents\Projekts\enzyme-cascade-analysis\example_reactions\dortmund_system\results\experimental_reaction2_HP_noisy_plate_reader_results.csv")    

    monte_carlo_results2 = pd.read_csv(r"C:\Users\berger\Documents\Projekts\enzyme-cascade-analysis\example_reactions\dortmund_system\results\experimental_reaction2_NADH_noisy_plate_reader_results.csv")

    # Create correlation matrix plot
    compare_error_ellipses(monte_carlo_results1 , monte_carlo_results2, 
                          parameters=['Vmax', 'Km1'])
