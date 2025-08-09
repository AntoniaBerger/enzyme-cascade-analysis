import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
import warnings
import os
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def plot_monte_carlo_results(monte_carlo_results, model_info, save_path=None, show_plots=True):
    """
    Erstellt umfassende Plots für Monte Carlo Simulationsergebnisse
    
    Args:
        monte_carlo_results: Dict mit Monte Carlo Ergebnissen
        model_info: Dict mit Modellinformationen (param_names, param_units, etc.)
        save_path: Optional - Pfad zum Speichern der Plots (Standard: Results/monte_carlo_results.png)
        show_plots: Bool - Ob Plots angezeigt werden sollen
    """
    
    if not monte_carlo_results or monte_carlo_results.get('n_successful', 0) == 0:
        print("⚠️ Keine erfolgreichen Monte Carlo Iterationen zum Plotten verfügbar")
        return
    
    # Standard-Speicherpfad im Results-Ordner
    if save_path is None:
        save_path = os.path.join('Results', 'monte_carlo_results.png')
    
    # Parameter-Namen und Einheiten
    param_names = model_info.get('param_names', [])
    param_units = model_info.get('param_units', [''] * len(param_names))
    
    # Filter nur Parameter, die auch Daten haben
    available_params = []
    for param_name in param_names:
        if f'{param_name}_values' in monte_carlo_results:
            available_params.append(param_name)
    
    if not available_params:
        print("⚠️ Keine Parameter-Daten zum Plotten verfügbar")
        return
    
    # Erstelle große Figure mit Subplots
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Parameter-Verteilungen (Histogramme)
    n_params = len(available_params)
    n_cols = min(4, n_params)
    n_rows = int(np.ceil(n_params / n_cols))
    
    for i, param_name in enumerate(available_params):
        ax = plt.subplot(4, n_cols, i + 1)
        
        # Hole die tatsächlichen Daten aus dem Monte Carlo Ergebnis
        param_values = monte_carlo_results[f'{param_name}_values']
        mean_val = monte_carlo_results[f'{param_name}_mean']
        std_val = monte_carlo_results[f'{param_name}_std']
        
        # Bestimme die Einheit
        unit_idx = param_names.index(param_name) if param_name in param_names else 0
        unit = param_units[unit_idx] if unit_idx < len(param_units) else ''
        
        # Histogramm
        ax.hist(param_values, bins=30, alpha=0.7, color=f'C{i}', edgecolor='black')
        
        # Mittelwert und Standardabweichung
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
        ax.axvline(mean_val - std_val, color='orange', linestyle=':', alpha=0.7, label=f'±σ: {std_val:.3f}')
        ax.axvline(mean_val + std_val, color='orange', linestyle=':', alpha=0.7)
        
        ax.set_title(f'{param_name} ({unit})', fontsize=10, fontweight='bold')
        ax.set_xlabel(f'{param_name} [{unit}]')
        ax.set_ylabel('Frequency')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 2. Parameter-Korrelationsmatrix
    if 'correlation_matrix' in monte_carlo_results:
        plt.figure(figsize=(12, 10))
        
        # Erstelle Korrelationsmatrix
        corr_matrix = monte_carlo_results['correlation_matrix']
        
        # Heatmap (nur für verfügbare Parameter)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Obere Dreiecksmatrix ausblenden
        sns.heatmap(corr_matrix, 
                   mask=mask,
                   annot=True, 
                   cmap='coolwarm', 
                   center=0,
                   square=True, 
                   linewidths=0.5,
                   xticklabels=available_params,
                   yticklabels=available_params,
                   cbar_kws={"shrink": 0.8})
        
        plt.title('Parameter Korrelationen (Monte Carlo)', fontsize=14, fontweight='bold', pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
    
    # 3. R² Verteilung
    if 'R_squared_array' in monte_carlo_results:
        plt.figure(figsize=(10, 6))
        
        r2_values = monte_carlo_results['R_squared_array']
        r2_mean = monte_carlo_results['R_squared_mean']
        r2_std = monte_carlo_results['R_squared_std']
        
        plt.hist(r2_values, bins=30, alpha=0.7, color='green', edgecolor='black')
        plt.axvline(r2_mean, color='red', linestyle='--', linewidth=2, label=f'Mean R²: {r2_mean:.4f}')
        plt.axvline(r2_mean - r2_std, color='orange', linestyle=':', alpha=0.7, label=f'±σ: {r2_std:.4f}')
        plt.axvline(r2_mean + r2_std, color='orange', linestyle=':', alpha=0.7)
        
        plt.title('R² Verteilung (Monte Carlo)', fontsize=14, fontweight='bold')
        plt.xlabel('R² Werte')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
    
    # 4. Parameter-Boxplots (Übersicht)
    plt.figure(figsize=(15, 8))
    
    param_data = []
    param_labels = []
    
    for param_name in available_params:
        param_data.append(monte_carlo_results[f'{param_name}_values'])
        # Finde die Einheit
        unit_idx = param_names.index(param_name) if param_name in param_names else 0
        unit = param_units[unit_idx] if unit_idx < len(param_units) else ''
        param_labels.append(f'{param_name}\n({unit})')
    
    box_plot = plt.boxplot(param_data, labels=param_labels, patch_artist=True)
    
    # Färbe Boxplots
    colors = plt.cm.Set3(np.linspace(0, 1, len(param_data)))
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    plt.title('Parameter Unsicherheiten (Monte Carlo)', fontsize=14, fontweight='bold')
    plt.ylabel('Parameter Werte')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 5. Erfolgsrate und Fehlschläge
    plt.figure(figsize=(12, 8))
    
    # Subplot 1: Erfolgsrate
    plt.subplot(2, 2, 1)
    
    success_rate = monte_carlo_results['success_rate']
    labels = ['Erfolgreich', 'Fehlgeschlagen']
    sizes = [success_rate, 1 - success_rate]
    colors = ['green', 'red']
    
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.title(f'Erfolgsrate\n({monte_carlo_results["n_successful"]}/{monte_carlo_results["n_total"]} Iterationen)')
    
    # Subplot 2: Fehlschlag-Kategorien
    plt.subplot(2, 2, 2)
    
    if 'failed_counts' in monte_carlo_results:
        fail_categories = list(monte_carlo_results['failed_counts'].keys())
        fail_counts = list(monte_carlo_results['failed_counts'].values())
        
        # Nur Kategorien mit Fehlern zeigen
        non_zero_fails = [(cat, count) for cat, count in zip(fail_categories, fail_counts) if count > 0]
        
        if non_zero_fails:
            categories, counts = zip(*non_zero_fails)
            plt.bar(categories, counts, color='red', alpha=0.7)
            plt.title('Fehlschlag-Kategorien')
            plt.ylabel('Anzahl Fehlschläge')
            plt.xticks(rotation=45)
        else:
            plt.text(0.5, 0.5, 'Keine Fehlschläge', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Fehlschlag-Kategorien')
    
    # Subplot 3: Parameter-Konfidenzintervalle
    plt.subplot(2, 1, 2)
    
    means = [monte_carlo_results[f'{param_name}_mean'] for param_name in available_params]
    stds = [monte_carlo_results[f'{param_name}_std'] for param_name in available_params]
    
    x_pos = np.arange(len(available_params))
    
    plt.errorbar(x_pos, means, yerr=stds, fmt='o', capsize=5, capthick=2, 
                ecolor='red', markersize=8, linewidth=2)
    
    for i, (mean, std, name) in enumerate(zip(means, stds, available_params)):
        plt.annotate(f'{mean:.3f}±{std:.3f}', 
                    (i, mean), 
                    textcoords="offset points", 
                    xytext=(0,10), 
                    ha='center', fontsize=8)
    
    plt.xticks(x_pos, available_params, rotation=45, ha='right')
    plt.ylabel('Parameter Werte')
    plt.title('Parameter-Mittelwerte mit Standardabweichungen')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Speichern falls gewünscht
    if save_path:
        # Stelle sicher, dass das Results-Verzeichnis existiert
        os.makedirs('Results', exist_ok=True)
        
        if save_path.endswith('.png'):
            filename = save_path
        else:
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            filename = f"Results/monte_carlo_results_{timestamp}.png"
            
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"📊 Plots gespeichert als: {filename}")
    
    if show_plots:
        plt.show()


def plot_parameter_convergence(monte_carlo_results, model_info, save_path=None, show_plots=True):
    """
    Zeigt die Konvergenz der Parameter über die Monte Carlo Iterationen
    
    Args:
        monte_carlo_results: Dict mit Monte Carlo Ergebnissen  
        model_info: Dict mit Modellinformationen
        save_path: Optional - Pfad zum Speichern (Standard: Results/parameter_convergence.png)
        show_plots: Bool - Ob Plots angezeigt werden sollen
    """
    
    # Standard-Speicherpfad im Results-Ordner
    if save_path is None:
        save_path = os.path.join('Results', 'parameter_convergence.png')
    
    # Prüfe verfügbare Parameter
    param_names = model_info.get('param_names', [])
    available_params = []
    for param_name in param_names:
        if f'{param_name}_values' in monte_carlo_results:
            available_params.append(param_name)
    
    if not available_params:
        print("⚠️ Keine Parameter-Arrays für Konvergenz-Plot verfügbar")
        return
    
    n_params = len(available_params)
    n_cols = min(3, n_params)
    n_rows = int(np.ceil(n_params / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    if n_params == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, param_name in enumerate(available_params):
        row, col = divmod(i, n_cols)
        ax = axes[row, col] if n_rows > 1 else axes[col]
        
        param_values = monte_carlo_results[f'{param_name}_values']
        
        # Kumulative Mittelwerte berechnen
        cumulative_means = np.cumsum(param_values) / np.arange(1, len(param_values) + 1)
        
        # Plot
        ax.plot(cumulative_means, color=f'C{i}', linewidth=2)
        
        # Finale Werte
        final_mean = monte_carlo_results[f'{param_name}_mean']
        final_std = monte_carlo_results[f'{param_name}_std']
        
        ax.axhline(final_mean, color='red', linestyle='--', alpha=0.7, 
                  label=f'Final Mean: {final_mean:.3f}')
        ax.fill_between(range(len(cumulative_means)), 
                       final_mean - final_std, final_mean + final_std,
                       alpha=0.2, color='red', label=f'±σ: {final_std:.3f}')
        
        ax.set_title(f'{param_name} Konvergenz', fontweight='bold')
        ax.set_xlabel('Monte Carlo Iteration')
        ax.set_ylabel(f'{param_name} Wert')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Leere Subplots ausblenden
    for i in range(n_params, n_rows * n_cols):
        row, col = divmod(i, n_cols)
        if n_rows > 1:
            axes[row, col].set_visible(False)
        else:
            axes[col].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        # Stelle sicher, dass das Results-Verzeichnis existiert
        os.makedirs('Results', exist_ok=True)
        
        if save_path.endswith('.png'):
            filename = save_path
        else:
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            filename = f"Results/parameter_convergence_{timestamp}.png"
            
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"📈 Konvergenz-Plot gespeichert als: {filename}")
    
    if show_plots:
        plt.show()


def plot_fitting_quality(monte_carlo_results, processed_data=None, model_info=None, save_path=None, show_plots=True):
    """
    Visualisiert die Fitting-Qualität der Monte Carlo Simulation
    
    Args:
        monte_carlo_results: Dict mit Monte Carlo Ergebnissen
        processed_data: Optional - Originaldaten zum Vergleich
        model_info: Dict mit Modellinformationen
        save_path: Optional - Pfad zum Speichern (Standard: Results/fitting_quality.png)
        show_plots: Bool - Ob Plots angezeigt werden sollen
    """
    
    # Standard-Speicherpfad im Results-Ordner
    if save_path is None:
        save_path = os.path.join('Results', 'fitting_quality.png')
    
    if 'R_squared_array' not in monte_carlo_results:
        print("⚠️ Keine R²-Daten für Fitting-Qualität verfügbar")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. R² Verteilung mit Statistiken
    ax1 = axes[0, 0]
    r2_values = monte_carlo_results['R_squared_array']
    
    ax1.hist(r2_values, bins=30, alpha=0.7, color='blue', edgecolor='black')
    
    # Statistiken
    r2_mean = np.mean(r2_values)
    r2_median = np.median(r2_values)
    r2_std = np.std(r2_values)
    
    ax1.axvline(r2_mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {r2_mean:.4f}')
    ax1.axvline(r2_median, color='orange', linestyle=':', linewidth=2, label=f'Median: {r2_median:.4f}')
    
    ax1.set_title('R² Verteilung', fontweight='bold')
    ax1.set_xlabel('R² Werte')
    ax1.set_ylabel('Häufigkeit')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Textbox mit Statistiken
    stats_text = f'Mean: {r2_mean:.4f}\nMedian: {r2_median:.4f}\nStd: {r2_std:.4f}\nMin: {np.min(r2_values):.4f}\nMax: {np.max(r2_values):.4f}'
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 2. R² vs Iteration (falls verfügbar)
    ax2 = axes[0, 1]
    ax2.plot(r2_values, color='green', alpha=0.7, linewidth=1)
    ax2.axhline(r2_mean, color='red', linestyle='--', alpha=0.7, label=f'Mean: {r2_mean:.4f}')
    
    ax2.set_title('R² über Monte Carlo Iterationen', fontweight='bold')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('R²')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Kumulative R² Statistiken
    ax3 = axes[1, 0]
    cumulative_r2_mean = np.cumsum(r2_values) / np.arange(1, len(r2_values) + 1)
    
    ax3.plot(cumulative_r2_mean, color='purple', linewidth=2, label='Kumulative Mean')
    ax3.axhline(r2_mean, color='red', linestyle='--', alpha=0.7, label=f'Final Mean: {r2_mean:.4f}')
    
    ax3.set_title('R² Konvergenz', fontweight='bold')
    ax3.set_xlabel('Monte Carlo Iteration')
    ax3.set_ylabel('Kumulative R² Mean')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Qualitätskategorien
    ax4 = axes[1, 1]
    
    # Kategorisiere R² Werte
    excellent = np.sum(r2_values >= 0.95)
    good = np.sum((r2_values >= 0.9) & (r2_values < 0.95))
    fair = np.sum((r2_values >= 0.8) & (r2_values < 0.9))
    poor = np.sum(r2_values < 0.8)
    
    categories = ['Excellent\n(≥0.95)', 'Good\n(0.9-0.95)', 'Fair\n(0.8-0.9)', 'Poor\n(<0.8)']
    counts = [excellent, good, fair, poor]
    colors = ['darkgreen', 'green', 'orange', 'red']
    
    bars = ax4.bar(categories, counts, color=colors, alpha=0.7, edgecolor='black')
    
    # Prozentangaben hinzufügen
    total = len(r2_values)
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        if height > 0:
            ax4.text(bar.get_x() + bar.get_width()/2., height + total*0.01,
                    f'{count}\n({count/total*100:.1f}%)',
                    ha='center', va='bottom', fontweight='bold')
    
    ax4.set_title('Fitting-Qualität Kategorien', fontweight='bold')
    ax4.set_ylabel('Anzahl Iterationen')
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        # Stelle sicher, dass das Results-Verzeichnis existiert
        os.makedirs('Results', exist_ok=True)
        
        if save_path.endswith('.png'):
            filename = save_path
        else:
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            filename = f"Results/fitting_quality_{timestamp}.png"
            
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"📊 Fitting-Qualität Plot gespeichert als: {filename}")
    
    if show_plots:
        plt.show()
        plt.show()


def create_monte_carlo_report(monte_carlo_results, model_info, save_path=None):
    """
    Erstellt einen umfassenden Bericht der Monte Carlo Ergebnisse
    
    Args:
        monte_carlo_results: Dict mit Monte Carlo Ergebnissen
        model_info: Dict mit Modellinformationen
        save_path: Optional - Basis-Pfad zum Speichern des Berichts (Standard: Results/)
    """
    
    # Standard-Speicherpfad im Results-Ordner
    if save_path is None:
        save_path = 'Results'
    
    # Stelle sicher, dass das Results-Verzeichnis existiert
    os.makedirs(save_path, exist_ok=True)
    
    # Erstelle alle Plots mit spezifischen Namen
    print("📊 Erstelle Monte Carlo Plots...")
    plot_monte_carlo_results(monte_carlo_results, model_info, 
                           os.path.join(save_path, 'monte_carlo_results.png'), show_plots=False)
    
    print("📈 Erstelle Konvergenz-Plots...")
    plot_parameter_convergence(monte_carlo_results, model_info, 
                             os.path.join(save_path, 'parameter_convergence.png'), show_plots=False)
    
    print("📊 Erstelle Fitting-Qualität Plots...")
    plot_fitting_quality(monte_carlo_results, 
                        save_path=os.path.join(save_path, 'fitting_quality.png'), show_plots=False)
    
    # Text-Bericht erstellen
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    report_file = os.path.join(save_path, f"monte_carlo_report_{timestamp}.txt")
    
    with open(report_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("MONTE CARLO SIMULATION BERICHT\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Modell: {model_info.get('description', 'Unbekannt')}\n")
            f.write(f"Datum: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Simulationsstatistiken
            f.write("SIMULATIONSSTATISTIKEN:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Gesamte Iterationen: {monte_carlo_results['n_total']}\n")
            f.write(f"Erfolgreiche Iterationen: {monte_carlo_results['n_successful']}\n")
            f.write(f"Erfolgsrate: {monte_carlo_results['success_rate']*100:.2f}%\n\n")
            
            # Parameter-Ergebnisse
            f.write("PARAMETER-ERGEBNISSE:\n")
            f.write("-" * 30 + "\n")
            param_names = model_info.get('param_names', [])
            param_units = model_info.get('param_units', [''] * len(param_names))
            
            for i, (name, unit) in enumerate(zip(param_names, param_units)):
                if f'{name}_mean' in monte_carlo_results:
                    mean = monte_carlo_results[f'{name}_mean']
                    std = monte_carlo_results[f'{name}_std']
                    values = monte_carlo_results[f'{name}_values']
                    min_val = np.min(values)
                    max_val = np.max(values)
                    
                    f.write(f"{name:12s}: {mean:8.4f} ± {std:6.4f} {unit:4s} "
                           f"[{min_val:8.4f}, {max_val:8.4f}]\n")
            
            # R² Statistiken
            if 'R_squared_mean' in monte_carlo_results:
                f.write("\nR² STATISTIKEN:\n")
                f.write("-" * 30 + "\n")
                f.write(f"R² Mittelwert: {monte_carlo_results['R_squared_mean']:.6f}\n")
                f.write(f"R² Standardabweichung: {monte_carlo_results['R_squared_std']:.6f}\n")
                f.write(f"R² Minimum: {np.min(monte_carlo_results['R_squared_values']):.6f}\n")
                f.write(f"R² Maximum: {np.max(monte_carlo_results['R_squared_values']):.6f}\n")
            
            # Fehlschläge
            if 'failed_counts' in monte_carlo_results:
                f.write("\nFEHLSCHLAG-ANALYSE:\n")
                f.write("-" * 30 + "\n")
                for category, count in monte_carlo_results['failed_counts'].items():
                    f.write(f"{category:20s}: {count:6d} ({count/monte_carlo_results['n_total']*100:.2f}%)\n")
    
    print(f"📄 Bericht gespeichert als: {report_file}")
    print("✅ Monte Carlo Analyse abgeschlossen!")


# Utility-Funktion für schnelle Plots
def quick_monte_carlo_plot(monte_carlo_results, model_info):
    """
    Erstellt schnelle Übersichts-Plots für Monte Carlo Ergebnisse
    """
    plot_monte_carlo_results(monte_carlo_results, model_info, save_path=None, show_plots=True)