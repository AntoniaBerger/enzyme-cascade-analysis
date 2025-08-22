from copyreg import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
import warnings
import os
import pickle
import glob
    
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def plot_monte_carlo_results_from_file(pkl_file_path, save_path=None, show_plots=True):
    """
    Lädt Monte Carlo Ergebnisse aus PKL-Datei und erstellt Plots
    
    Args:
        pkl_file_path: Pfad zur PKL-Datei mit Monte Carlo Ergebnissen
        save_path: Optional - Pfad zum Speichern der Plots
        show_plots: Bool - Ob Plots angezeigt werden sollen
    """
    
    try:
        with open(pkl_file_path, 'rb') as f:
            data = pickle.load(f)
        
        # Extrahiere Monte Carlo Ergebnisse und Model Info
        if isinstance(data, dict) and 'monte_carlo_results' in data and 'model_info' in data:
            monte_carlo_results = data['monte_carlo_results']
            model_info = data['model_info']
        else:
            # Falls direkt Monte Carlo Ergebnisse gespeichert wurden
            monte_carlo_results = data
            # Standard Model Info falls nicht vorhanden
            model_info = {
                'param_names': [key.replace('_values', '') for key in monte_carlo_results.keys() if key.endswith('_values')],
                'param_units': [''] * len([key for key in monte_carlo_results.keys() if key.endswith('_values')]),
                'description': 'Loaded from PKL file'
            }
        
        print(f"📂 Monte Carlo Daten geladen aus: {pkl_file_path}")
        
        # Rufe die ursprüngliche Plot-Funktion auf
        plot_monte_carlo_results(monte_carlo_results, model_info, save_path, show_plots)
        
    except Exception as e:
        print(f"❌ Fehler beim Laden der PKL-Datei {pkl_file_path}: {e}")

def plot_monte_carlo_results(monte_carlo_results_or_file, model_info=None, save_path=None, show_plots=True):
    """
    Erstellt umfassende Plots für Monte Carlo Simulationsergebnisse
    
    Args:
        monte_carlo_results_or_file: Dict mit Monte Carlo Ergebnissen ODER Pfad zur PKL-Datei
        model_info: Dict mit Modellinformationen (nur nötig wenn erste Argument Dict ist)
        save_path: Optional - Pfad zum Speichern der Plots
        show_plots: Bool - Ob Plots angezeigt werden sollen
    """
    
    # Prüfe ob es ein String (Dateipfad) oder Dict ist
    if isinstance(monte_carlo_results_or_file, str):
        # Es ist ein Dateipfad - lade die Daten
        plot_monte_carlo_results_from_file(monte_carlo_results_or_file, save_path, show_plots)
        return
    
    # Es ist ein Dict - verwende wie bisher
    monte_carlo_results = monte_carlo_results_or_file
    
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
                    median = monte_carlo_results.get(f'{name}_median', None)
                    ci_lower = monte_carlo_results.get(f'{name}_ci_lower', None)
                    ci_upper = monte_carlo_results.get(f'{name}_ci_upper', None)
                    values = monte_carlo_results[f'{name}_values']
                    min_val = np.min(values)
                    max_val = np.max(values)
                    f.write(f"{name:12s}: {mean:8.4f} ± {std:6.4f} {unit:4s} "
                           f"[{min_val:8.4f}, {max_val:8.4f}]\n")
                    if median is not None:
                        f.write(f"    Median: {median:8.4f} {unit}\n")
                    if ci_lower is not None and ci_upper is not None:
                        f.write(f"    95% CI: [{ci_lower:8.4f}, {ci_upper:8.4f}] {unit}\n")
                # Statistische Kennzahlen für Parameter
                
            
            # R² Statistiken
            if 'R_squared_mean' in monte_carlo_results:
                f.write("\nR² STATISTIKEN:\n")
                f.write("-" * 30 + "\n")
                f.write(f"R² Mittelwert: {monte_carlo_results['R_squared_mean']:.6f}\n")
                f.write(f"R² Standardabweichung: {monte_carlo_results['R_squared_std']:.6f}\n")
                f.write(f"R² Median: {monte_carlo_results.get('R_squared_median', float('nan')):.6f}\n")
                f.write(f"R² Minimum: {np.min(monte_carlo_results['R_squared_values']):.6f}\n")
                f.write(f"R² Maximum: {np.max(monte_carlo_results['R_squared_values']):.6f}\n")
            
            # Fehlschläge
            if 'failed_counts' in monte_carlo_results:
                f.write("\nFEHLSCHLAG-ANALYSE:\n")
                f.write("-" * 30 + "\n")
                for category, count in monte_carlo_results['failed_counts'].items():
                    f.write(f"{category:20s}: {count:6d} ({count/monte_carlo_results['n_total']*100:.2f}%)\n")
            
            report_lines = []
            report_lines.append('=== PARAMETER-STATISTIKEN ===')
            for param_name in param_names:
                mean_val = monte_carlo_results.get(f'{param_name}_mean', None)
                std_val = monte_carlo_results.get(f'{param_name}_std', None)
                median_val = monte_carlo_results.get(f'{param_name}_median', None)
                ci_lower = monte_carlo_results.get(f'{param_name}_ci_lower', None)
                ci_upper = monte_carlo_results.get(f'{param_name}_ci_upper', None)
                unit = model_info.get('param_units', [''])[param_names.index(param_name)] if param_name in param_names else ''
                if mean_val is not None:
                    report_lines.append(f"{param_name}: {mean_val:.4f} ± {std_val:.4f} {unit}")
                    report_lines.append(f"  Median: {median_val:.4f} {unit}")
                    report_lines.append(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}] {unit}")

            # R² Statistiken
            r2_mean = monte_carlo_results.get('R_squared_mean', None)
            r2_std = monte_carlo_results.get('R_squared_std', None)
            r2_median = monte_carlo_results.get('R_squared_median', None)
            if r2_mean is not None:
                report_lines.append(f"\nR²: {r2_mean:.4f} ± {r2_std:.4f}")
                report_lines.append(f"R² Median: {r2_median:.4f}")

            # Korrelationsmatrix
            correlation_matrix = monte_carlo_results.get('correlation_matrix', None)
            if correlation_matrix is not None:
                report_lines.append('\n=== PARAMETER-KORRELATIONEN ===')
                for i, name1 in enumerate(param_names):
                    for j, name2 in enumerate(param_names):
                        if i < j:
                            corr = correlation_matrix[i, j]
                            report_lines.append(f"{name1} ↔ {name2}: {corr:.3f}")

            # Schreibe Report in Datei (report_file) und gebe ihn ggf. auf der Konsole aus
            for line in report_lines:
                f.write(line + '\n')
            for line in report_lines:
                print(line)
    
    print(f"📄 Bericht gespeichert als: {report_file}")
    print("✅ Monte Carlo Analyse abgeschlossen!")

def plot_simulation_results(simulation_dir="Results/Simulations", save_path=None, show_plots=True, max_files=None):
    """
    Lädt und plottet CADET Simulationsergebnisse aus PKL-Dateien
    
    Args:
        simulation_dir: Pfad zum Ordner mit den PKL-Dateien
        save_path: Optional - Pfad zum Speichern des Plots (Standard: Results/simulation_results.png)
        show_plots: Bool - Ob Plots angezeigt werden sollen
        max_files: Optional - Maximale Anzahl der zu ladenden Dateien (für Performance)
    """
    

    # Standard-Speicherpfad im Results-Ordner
    if save_path is None:
        save_path = os.path.join('Results', 'simulation_results.png')
    
    # Finde alle PKL-Dateien
    pkl_pattern = os.path.join(simulation_dir, "full_system_simulation_results_*.pkl")
    pkl_files = glob.glob(pkl_pattern)
    
    if not pkl_files:
        print(f"⚠️ Keine Simulationsdateien in {simulation_dir} gefunden!")
        return
    
    # Sortiere nach Dateinamen (Iteration)
    pkl_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
    # Begrenze Anzahl der Dateien falls gewünscht
    if max_files and len(pkl_files) > max_files:
        pkl_files = pkl_files[:max_files]
        print(f"📊 Lade ersten {max_files} von {len(glob.glob(pkl_pattern))} Simulationsdateien...")
    
    print(f"📂 Lade {len(pkl_files)} Simulationsdateien...")
    
    # Listen für alle Simulationen
    all_concentrations = []
    all_timepoints = []
    iteration_numbers = []
    
    # Lade alle Simulationsergebnisse
    for i, pkl_file in enumerate(pkl_files):
        try:
            with open(pkl_file, 'rb') as f:
                model = pickle.load(f)
            
            # Extrahiere Konzentrationen und Zeitpunkte
            concentrations = model.root.output.solution.unit_001.solution_outlet
            timepoints = model.root.output.solution.solution_times
            
            all_concentrations.append(concentrations)
            all_timepoints.append(timepoints)
            
            # Extrahiere Iterations-Nummer aus Dateiname
            iteration_num = int(pkl_file.split('_')[-1].split('.')[0])
            iteration_numbers.append(iteration_num)
            
            if (i + 1) % 10 == 0:
                print(f"  Geladen: {i + 1}/{len(pkl_files)} Dateien")
                
        except Exception as e:
            print(f"⚠️ Fehler beim Laden von {pkl_file}: {e}")
            continue
    
    if not all_concentrations:
        print("❌ Keine gültigen Simulationsdaten gefunden!")
        return
    
    print(f"✅ {len(all_concentrations)} Simulationen erfolgreich geladen")
    
    # Komponentennamen (basierend auf deiner Simulation)
    component_names = ['PD', 'NAD', 'Lactol', 'NADH', 'Lacton']
    n_components = len(component_names)
    
    # Erstelle umfassende Plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('CADET Simulationsergebnisse - Monte Carlo Ensemble', fontsize=16, fontweight='bold')
    
    # 1. Alle Simulationen übereinander (alle Komponenten)
    ax1 = axes[0, 0]
    colors = plt.cm.Set1(np.linspace(0, 1, n_components))
    
    for i, (concentrations, timepoints) in enumerate(zip(all_concentrations, all_timepoints)):
        alpha = 0.1 if len(all_concentrations) > 50 else 0.3
        
        for comp_idx in range(min(n_components, concentrations.shape[1])):
            if i == 0:  # Label nur für erste Simulation
                ax1.plot(timepoints, concentrations[:, comp_idx], 
                        color=colors[comp_idx], alpha=alpha, 
                        label=component_names[comp_idx])
            else:
                ax1.plot(timepoints, concentrations[:, comp_idx], 
                        color=colors[comp_idx], alpha=alpha)
    
    ax1.set_xlabel('Zeit [s]')
    ax1.set_ylabel('Konzentration [mM]')
    ax1.set_title('Alle Simulationen - Alle Komponenten')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')  # Logarithmische Skala wegen großer Wertebereiche
    
    # 2. Mittelwert und Standardabweichung
    ax2 = axes[0, 1]
    
    # Interpoliere alle Simulationen auf gemeinsame Zeitachse
    common_time = all_timepoints[0]  # Verwende erste Zeitachse als Referenz
    
    interpolated_concentrations = []
    for concentrations, timepoints in zip(all_concentrations, all_timepoints):
        interp_conc = np.zeros((len(common_time), concentrations.shape[1]))
        for comp_idx in range(concentrations.shape[1]):
            interp_conc[:, comp_idx] = np.interp(common_time, timepoints, concentrations[:, comp_idx])
        interpolated_concentrations.append(interp_conc)
    
    # Berechne Statistiken
    conc_array = np.array(interpolated_concentrations)  # [n_sims, n_time, n_components]
    mean_conc = np.mean(conc_array, axis=0)  # [n_time, n_components]
    std_conc = np.std(conc_array, axis=0)    # [n_time, n_components]
    
    for comp_idx in range(min(n_components, mean_conc.shape[1])):
        ax2.plot(common_time, mean_conc[:, comp_idx], 
                color=colors[comp_idx], linewidth=2, 
                label=f'{component_names[comp_idx]} (Mean)')
        ax2.fill_between(common_time, 
                        mean_conc[:, comp_idx] - std_conc[:, comp_idx],
                        mean_conc[:, comp_idx] + std_conc[:, comp_idx],
                        color=colors[comp_idx], alpha=0.3)
    
    ax2.set_xlabel('Zeit [s]')
    ax2.set_ylabel('Konzentration [mM]')
    ax2.set_title('Mittelwert ± Standardabweichung')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # 3. Einzelne Komponenten in separaten Subplots
    component_plots = [axes[0, 2], axes[1, 0], axes[1, 1]]
    for comp_idx, ax in enumerate(component_plots):
        if comp_idx < n_components:
            # Alle Simulationen für diese Komponente
            for concentrations, timepoints in zip(all_concentrations, all_timepoints):
                alpha = 0.1 if len(all_concentrations) > 50 else 0.3
                ax.plot(timepoints, concentrations[:, comp_idx], 
                       color=colors[comp_idx], alpha=alpha)
            
            # Mittelwert hervorheben
            ax.plot(common_time, mean_conc[:, comp_idx], 
                   color='black', linewidth=3, 
                   label=f'Mean {component_names[comp_idx]}')
            
            ax.set_xlabel('Zeit [s]')
            ax.set_ylabel('Konzentration [mM]')
            ax.set_title(f'{component_names[comp_idx]} Konzentration')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Setze Y-Achse je nach Komponente
            if component_names[comp_idx] in ['NADH', 'Lactol', 'Lacton']:
                ax.set_yscale('log')
    
    # 4. Endkonzentrationen (nach 500s)
    ax_final = axes[1, 2]
    
    final_concentrations = []
    for concentrations in all_concentrations:
        final_concentrations.append(concentrations[-1, :])  # Letzte Zeitpunkt
    
    final_conc_array = np.array(final_concentrations)
    
    # Boxplot der Endkonzentrationen
    box_data = [final_conc_array[:, i] for i in range(min(n_components, final_conc_array.shape[1]))]
    box_labels = component_names[:len(box_data)]
    
    box_plot = ax_final.boxplot(box_data, labels=box_labels, patch_artist=True)
    
    # Färbe Boxplots
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax_final.set_ylabel('Endkonzentration [mM]')
    ax_final.set_title('Endkonzentrationen (t=500s)')
    ax_final.set_yscale('log')
    ax_final.grid(True, alpha=0.3)
    ax_final.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Speichern
    if save_path:
        os.makedirs('Results', exist_ok=True)
        
        if save_path.endswith('.png'):
            filename = save_path
        else:
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            filename = f"Results/simulation_results_{timestamp}.png"
            
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"📊 Simulationsplots gespeichert als: {filename}")
    
    if show_plots:
        plt.show()
    
    # Zusätzliche Statistik-Ausgabe
    print("\n" + "="*50)
    print("📊 SIMULATIONSSTATISTIKEN")
    print("="*50)
    print(f"Anzahl Simulationen: {len(all_concentrations)}")
    print(f"Simulationszeit: {common_time[-1]:.1f} s")
    print(f"Komponenten: {', '.join(component_names)}")
    
    print("\nENDKONZENTRATIONEN (Mittelwert ± Std):")
    for i, comp_name in enumerate(component_names):
        if i < final_conc_array.shape[1]:
            mean_final = np.mean(final_conc_array[:, i])
            std_final = np.std(final_conc_array[:, i])
            print(f"  {comp_name:8s}: {mean_final:8.4f} ± {std_final:6.4f} mM")

def plot_single_simulation(pkl_file, save_path=None, show_plots=True):
    """
    Plottet eine einzelne Simulation aus einer PKL-Datei
    
    Args:
        pkl_file: Pfad zur PKL-Datei
        save_path: Optional - Pfad zum Speichern 
        show_plots: Bool - Ob Plot angezeigt werden soll
    """
    
    try:
        with open(pkl_file, 'rb') as f:
            model = pickle.load(f)
        
        # Extrahiere Daten
        concentrations = model.root.output.solution.unit_001.solution_outlet
        timepoints = model.root.output.solution.solution_times
        
    except Exception as e:
        print(f"❌ Fehler beim Laden von {pkl_file}: {e}")
        return
    
    # Komponentennamen
    component_names = ['PD', 'NAD', 'Lactol', 'NADH', 'Lacton']
    colors = plt.cm.Set1(np.linspace(0, 1, len(component_names)))
    
    plt.figure(figsize=(12, 8))
    
    for comp_idx in range(min(len(component_names), concentrations.shape[1])):
        plt.plot(timepoints, concentrations[:, comp_idx], 
                color=colors[comp_idx], linewidth=2, 
                label=component_names[comp_idx], marker='o', markersize=3)
    
    plt.xlabel('Zeit [s]', fontsize=12)
    plt.ylabel('Konzentration [mM]', fontsize=12)
    plt.title(f'CADET Simulation: {os.path.basename(pkl_file)}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Annotationen für Endwerte
    for comp_idx in range(min(len(component_names), concentrations.shape[1])):
        final_conc = concentrations[-1, comp_idx]
        plt.annotate(f'{final_conc:.3f}', 
                    (timepoints[-1], final_conc),
                    textcoords="offset points", 
                    xytext=(10,0), 
                    ha='left', fontsize=8,
                    color=colors[comp_idx])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Einzelsimulation gespeichert als: {save_path}")
    
    if show_plots:
        plt.show()

def analyze_simulation_convergence(simulation_dir="Results/Simulations", save_path=None):
    """
    Analysiert die Konvergenz der Simulationsergebnisse über Monte Carlo Iterationen
    
    Args:
        simulation_dir: Pfad zum Ordner mit PKL-Dateien
        save_path: Optional - Pfad zum Speichern der Analyse
    """
    
    import glob
    
    pkl_files = glob.glob(os.path.join(simulation_dir, "*.pkl"))
    pkl_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
    if len(pkl_files) < 2:
        print("⚠️ Zu wenige Simulationen für Konvergenzanalyse")
        return
    
    component_names = ['PD', 'NAD', 'Lactol', 'NADH', 'Lacton']
    final_concentrations = []
    
    # Sammle Endkonzentrationen
    for pkl_file in pkl_files:
        try:
            with open(pkl_file, 'rb') as f:
                model = pickle.load(f)
            concentrations = model.root.output.solution.unit_001.solution_outlet
            final_concentrations.append(concentrations[-1, :])
        except:
            continue
    
    final_conc_array = np.array(final_concentrations)
    
    # Plot Konvergenz
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Simulation Konvergenz-Analyse', fontsize=16, fontweight='bold')
    
    for comp_idx, ax in enumerate(axes.flat):
        if comp_idx < len(component_names):
            values = final_conc_array[:, comp_idx]
            cumulative_mean = np.cumsum(values) / np.arange(1, len(values) + 1)
            
            ax.plot(cumulative_mean, linewidth=2, color=f'C{comp_idx}')
            ax.axhline(np.mean(values), color='red', linestyle='--', 
                      label=f'Final Mean: {np.mean(values):.4f}')
            
            ax.set_title(f'{component_names[comp_idx]} Konvergenz')
            ax.set_xlabel('Monte Carlo Iteration')
            ax.set_ylabel('Kumulative Endkonzentration [mM]')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📈 Konvergenz-Analyse gespeichert als: {save_path}")
    
    plt.show()

def plot_component_analysis(simulation_dir="Results/Simulations", save_path=None, show_plots=True, max_files=None):
    """
    Detaillierte Analyse einzelner Komponenten aus CADET Simulationsergebnissen
    
    Args:
        simulation_dir: Pfad zum Ordner mit PKL-Dateien
        save_path: Optional - Pfad zum Speichern (Standard: Results/component_analysis.png)
        show_plots: Bool - Ob Plots angezeigt werden sollen
        max_files: Optional - Maximale Anzahl Dateien
    """
    
    if save_path is None:
        save_path = os.path.join('Results', 'component_analysis.png')
    
    # Lade Simulationsdaten
    pkl_pattern = os.path.join(simulation_dir, "full_system_simulation_results_*.pkl")
    pkl_files = glob.glob(pkl_pattern)
    
    if not pkl_files:
        print(f"⚠️ Keine Simulationsdateien gefunden!")
        return
    
    pkl_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    if max_files:
        pkl_files = pkl_files[:max_files]
    
    all_concentrations = []
    all_timepoints = []
    
    for pkl_file in pkl_files:
        try:
            with open(pkl_file, 'rb') as f:
                model = pickle.load(f)
            concentrations = model.root.output.solution.unit_001.solution_outlet
            timepoints = model.root.output.solution.solution_times
            all_concentrations.append(concentrations)
            all_timepoints.append(timepoints)
        except:
            continue
    
    if not all_concentrations:
        print("❌ Keine gültigen Daten!")
        return
    
    component_names = ['PD', 'NAD', 'Lactol', 'NADH', 'Lacton']
    colors = plt.cm.Set1(np.linspace(0, 1, len(component_names)))
    
    # Erstelle Figure mit 5 Komponenten
    fig, axes = plt.subplots(5, 2, figsize=(16, 20))
    fig.suptitle('Detaillierte Komponenten-Analyse (Monte Carlo Ensemble)', fontsize=16, fontweight='bold')
    
    # Interpoliere auf gemeinsame Zeitachse
    common_time = all_timepoints[0]
    interpolated_concentrations = []
    for concentrations, timepoints in zip(all_concentrations, all_timepoints):
        interp_conc = np.zeros((len(common_time), concentrations.shape[1]))
        for comp_idx in range(concentrations.shape[1]):
            interp_conc[:, comp_idx] = np.interp(common_time, timepoints, concentrations[:, comp_idx])
        interpolated_concentrations.append(interp_conc)
    
    conc_array = np.array(interpolated_concentrations)
    mean_conc = np.mean(conc_array, axis=0)
    std_conc = np.std(conc_array, axis=0)
    
    for comp_idx in range(min(len(component_names), conc_array.shape[2])):
        # Linke Spalte: Zeitverlauf
        ax_time = axes[comp_idx, 0]
        
        # Alle Simulationen (dünn)
        for sim_idx in range(len(all_concentrations)):
            alpha = 0.05 if len(all_concentrations) > 50 else 0.2
            ax_time.plot(all_timepoints[sim_idx], all_concentrations[sim_idx][:, comp_idx], 
                        color=colors[comp_idx], alpha=alpha, linewidth=0.5)
        
        # Mittelwert (dick)
        ax_time.plot(common_time, mean_conc[:, comp_idx], 
                    color='black', linewidth=3, label=f'Mean')
        
        # Konfidenzbereich
        ax_time.fill_between(common_time, 
                           mean_conc[:, comp_idx] - std_conc[:, comp_idx],
                           mean_conc[:, comp_idx] + std_conc[:, comp_idx],
                           alpha=0.3, color=colors[comp_idx], label='±1σ')
        
        ax_time.set_title(f'{component_names[comp_idx]} - Zeitverlauf', fontweight='bold')
        ax_time.set_xlabel('Zeit [s]')
        ax_time.set_ylabel('Konzentration [mM]')
        ax_time.legend()
        ax_time.grid(True, alpha=0.3)
        ax_time.set_yscale('log')
        
        # Rechte Spalte: Endkonzentrations-Verteilung
        ax_dist = axes[comp_idx, 1]
        
        final_conc = conc_array[:, -1, comp_idx]  # Endkonzentrationen
        
        # Histogramm
        ax_dist.hist(final_conc, bins=20, alpha=0.7, color=colors[comp_idx], edgecolor='black')
        
        # Statistiken
        mean_final = np.mean(final_conc)
        std_final = np.std(final_conc)
        median_final = np.median(final_conc)
        
        ax_dist.axvline(mean_final, color='red', linestyle='--', linewidth=2, 
                       label=f'Mean: {mean_final:.4f}')
        ax_dist.axvline(median_final, color='orange', linestyle=':', linewidth=2, 
                       label=f'Median: {median_final:.4f}')
        
        ax_dist.set_title(f'{component_names[comp_idx]} - Endkonzentrations-Verteilung', fontweight='bold')
        ax_dist.set_xlabel('Endkonzentration [mM]')
        ax_dist.set_ylabel('Häufigkeit')
        ax_dist.legend()
        ax_dist.grid(True, alpha=0.3)
        
        # Statistik-Textbox
        stats_text = f'Mean: {mean_final:.4f}\nStd: {std_final:.4f}\nCV: {std_final/mean_final*100:.1f}%'
        ax_dist.text(0.02, 0.98, stats_text, transform=ax_dist.transAxes, 
                    verticalalignment='top', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Komponenten-Analyse gespeichert als: {save_path}")
    
    if show_plots:
        plt.show()
    
    # Konzentrationsstatistiken ausgeben
    print("\n" + "="*60)
    print("📊 KOMPONENTEN-STATISTIKEN")
    print("="*60)
    for comp_idx in range(min(len(component_names), conc_array.shape[2])):
        final_conc = conc_array[:, -1, comp_idx]
        print(f"{component_names[comp_idx]:8s}: {np.mean(final_conc):8.4f} ± {np.std(final_conc):6.4f} mM "
              f"(CV: {np.std(final_conc)/np.mean(final_conc)*100:.1f}%)")



if __name__ == "__main__":

    plot_component_analysis()# Utility-Funktion für schnelle Plots
