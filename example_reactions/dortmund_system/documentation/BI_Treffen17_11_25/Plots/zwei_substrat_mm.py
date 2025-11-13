"""
Einfacher 3D-Plot für Zwei-Substrat Michaelis-Menten Kinetik
============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import os

def lade_experimentelle_daten(datei_pfad=None):
    """
    Lädt experimentelle Daten aus CSV-Datei
    
    Erwartetes Format:
    - Spalte 1: [Substrat A] in mM
    - Spalte 2: [Substrat B] in mM  
    - Spalte 3: Reaktionsgeschwindigkeit v in μM/min
    
    Oder alternativ: manuell definierte Testdaten
    """
    if datei_pfad and os.path.exists(datei_pfad):
        try:
            data = pd.read_csv(datei_pfad)
            if len(data.columns) >= 3:
                sa_data = data.iloc[:, 0].values
                sb_data = data.iloc[:, 1].values
                v_data = data.iloc[:, 2].values
                print(f"Experimentelle Daten aus {datei_pfad} geladen: {len(sa_data)} Datenpunkte")
                return sa_data, sb_data, v_data
        except Exception as e:
            print(f"Fehler beim Laden der Datei: {e}")
    
    # Fallback: Simulierte "experimentelle" Daten mit Rauschen
    print("Erstelle simulierte experimentelle Daten mit Rauschen...")
    np.random.seed(42)  # Für Reproduzierbarkeit
    
    # Zufällige Substratkonzentrationen
    n_points = 20
    sa_data = np.random.uniform(1, 30, n_points)
    sb_data = np.random.uniform(1, 25, n_points)
    
    # "Wahre" Werte mit Modell berechnen
    v_true = zwei_substrat_mm(sa_data, sb_data, vmax=95, ka=8, kb=6)
    
    # Experimentelles Rauschen hinzufügen (5-15% relativer Fehler)
    rauschen = np.random.normal(0, 0.1, n_points)  # 10% Standardabweichung
    v_data = v_true * (1 + rauschen)
    v_data = np.maximum(v_data, 0.1)  # Negative Werte vermeiden
    
    return sa_data, sb_data, v_data

def erstelle_beispiel_daten():
    """Erstellt eine CSV-Datei mit Beispieldaten"""
    np.random.seed(42)
    n_points = 25
    
    sa_data = np.random.uniform(2, 40, n_points)
    sb_data = np.random.uniform(2, 35, n_points)
    v_true = zwei_substrat_mm(sa_data, sb_data, vmax=98, ka=9, kb=7)
    
    # Rauschen hinzufügen
    rauschen = np.random.normal(0, 0.08, n_points)
    v_data = v_true * (1 + rauschen)
    v_data = np.maximum(v_data, 0.1)
    
    # DataFrame erstellen und speichern
    df = pd.DataFrame({
        'Substrat_A_mM': sa_data,
        'Substrat_B_mM': sb_data,
        'Reaktionsgeschwindigkeit_uM_min': v_data
    })
    
    datei_name = 'beispiel_zwei_substrat_daten.csv'
    df.to_csv(datei_name, index=False)
    print(f"Beispieldaten in '{datei_name}' erstellt!")
    return datei_name

def zwei_substrat_mm(sa, sb, vmax=100, ka=10, kb=5):
    """
    Zwei-Substrat Michaelis-Menten Kinetik (Ordered Sequential)
    
    v = (Vmax * [A] * [B]) / (Ka*Kb + Kb*[A] + Ka*[B] + [A]*[B])
    
    Parameter:
    - sa, sb: Substratkonzentrationen A und B
    - vmax: Maximale Reaktionsgeschwindigkeit (μM/min)
    - ka, kb: Michaelis-Konstanten (mM)
    """
    numerator = vmax * sa * sb
    denominator = ka * kb + kb * sa + ka * sb + sa * sb
    return numerator / denominator

def plot_3d_mit_daten(datei_pfad=None):
    """Erstellt 3D-Plot mit experimentellen Daten"""
    
    # Experimentelle Daten laden
    sa_data, sb_data, v_data = lade_experimentelle_daten(datei_pfad)
    
    # Theoretische Oberfläche
    sa_range = np.linspace(0.5, 50, 50)
    sb_range = np.linspace(0.5, 50, 50)
    SA, SB = np.meshgrid(sa_range, sb_range)
    V = zwei_substrat_mm(SA, SB)
    
    # 3D-Plot erstellen
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Theoretische Oberfläche (transparenter)
    surf = ax.plot_surface(SA, SB, V, cmap='viridis', alpha=0.6, 
                          linewidth=0, antialiased=True)
    
    # Experimentelle Datenpunkte plotten
    scatter = ax.scatter(sa_data, sb_data, v_data, 
                        c='red', s=100, alpha=0.9, 
                        marker='o', edgecolors='darkred', linewidth=2,
                        label='Experimentelle Daten')
    
    # Residuen berechnen und anzeigen
    v_predicted = zwei_substrat_mm(sa_data, sb_data)
    residuen = v_data - v_predicted
    
    # Residuen-Linien (Fehlerbalken)
    for i in range(len(sa_data)):
        if residuen[i] > 0:
            farbe = 'green'
        else:
            farbe = 'orange'
        ax.plot([sa_data[i], sa_data[i]], 
                [sb_data[i], sb_data[i]], 
                [v_predicted[i], v_data[i]], 
                color=farbe, linewidth=2, alpha=0.7)
    
    # Beschriftungen
    ax.set_xlabel('[Substrat A] (mM)', fontsize=12)
    ax.set_ylabel('[Substrat B] (mM)', fontsize=12)
    ax.set_zlabel('Reaktionsgeschwindigkeit v (μM/min)', fontsize=12)
    ax.set_title('Zwei-Substrat MM-Kinetik\nTheorie vs. Experimentelle Daten', 
                fontsize=14, pad=20)
    
    # Statistiken berechnen
    r_squared = 1 - np.sum(residuen**2) / np.sum((v_data - np.mean(v_data))**2)
    rmse = np.sqrt(np.mean(residuen**2))
    
    # Textbox mit Statistiken
    stats_text = f'R² = {r_squared:.3f}\nRMSE = {rmse:.2f} μM/min\nn = {len(sa_data)} Punkte'
    ax.text2D(0.02, 0.98, stats_text, transform=ax.transAxes, 
              fontsize=10, verticalalignment='top',
              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Colorbar und Legend
    fig.colorbar(surf, ax=ax, shrink=0.5, label='v (μM/min)')
    ax.legend(loc='upper right')
    
    # Blickwinkel optimieren
    ax.view_init(elev=25, azim=45)
    
    plt.tight_layout()
    plt.savefig('zwei_substrat_mm_mit_daten.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"3D-Plot mit experimentellen Daten erstellt!")
    print(f"Modellgüte: R² = {r_squared:.3f}, RMSE = {rmse:.2f} μM/min")

def plot_3d_mit_projektionen_und_daten(datei_pfad=None):
    """Erstellt 3D-Plot mit Projektionen UND experimentellen Daten"""
    
    # Experimentelle Daten laden
    sa_data, sb_data, v_data = lade_experimentelle_daten(datei_pfad)
    
    # Theoretische Oberfläche
    sa_range = np.linspace(0.5, 50, 50)
    sb_range = np.linspace(0.5, 50, 50)
    SA, SB = np.meshgrid(sa_range, sb_range)
    V = zwei_substrat_mm(SA, SB)
    
    # 3D-Plot erstellen
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Theoretische Oberfläche
    surf = ax.plot_surface(SA, SB, V, cmap='viridis', alpha=0.5, 
                          linewidth=0, antialiased=True)
    
    # Projektionslinien (weniger, um übersichtlich zu bleiben)
    sb_konstant = [10, 20]  
    sa_konstant = [10, 20]  
    
    # Projektionen für konstantes [B]
    for sb_const in sb_konstant:
        sa_proj = sa_range
        sb_proj = np.full_like(sa_range, sb_const)
        v_proj = zwei_substrat_mm(sa_proj, sb_proj)
        ax.plot(sa_proj, sb_proj, v_proj, color='blue', linewidth=2, alpha=0.7,
                label=f'[B] = {sb_const} mM' if sb_const == sb_konstant[0] else "")
    
    # Projektionen für konstantes [A]
    for sa_const in sa_konstant:
        sa_proj = np.full_like(sb_range, sa_const)
        sb_proj = sb_range
        v_proj = zwei_substrat_mm(sa_proj, sb_proj)
        ax.plot(sa_proj, sb_proj, v_proj, color='orange', linewidth=2, alpha=0.7,
                label=f'[A] = {sa_const} mM' if sa_const == sa_konstant[0] else "")
    
    # Experimentelle Datenpunkte
    scatter = ax.scatter(sa_data, sb_data, v_data, 
                        c='red', s=120, alpha=0.9, 
                        marker='o', edgecolors='darkred', linewidth=2,
                        label='Experimentelle Daten')
    
    # Beschriftungen
    ax.set_xlabel('[Substrat A] (mM)', fontsize=12)
    ax.set_ylabel('[Substrat B] (mM)', fontsize=12)
    ax.set_zlabel('Reaktionsgeschwindigkeit v (μM/min)', fontsize=12)
    ax.set_title('Zwei-Substrat MM-Kinetik\nTheorie, Projektionen & Daten', 
                fontsize=14, pad=20)
    
    # Modellgüte berechnen
    v_predicted = zwei_substrat_mm(sa_data, sb_data)
    residuen = v_data - v_predicted
    r_squared = 1 - np.sum(residuen**2) / np.sum((v_data - np.mean(v_data))**2)
    rmse = np.sqrt(np.mean(residuen**2))
    
    # Statistiken anzeigen
    stats_text = f'R² = {r_squared:.3f}\nRMSE = {rmse:.2f}\nn = {len(sa_data)}'
    ax.text2D(0.02, 0.98, stats_text, transform=ax.transAxes, 
              fontsize=10, verticalalignment='top',
              bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Colorbar und Legend
    fig.colorbar(surf, ax=ax, shrink=0.4, label='v (μM/min)')
    ax.legend(loc='center left', bbox_to_anchor=(1.1, 0.5), fontsize=10)
    
    # Blickwinkel optimieren
    ax.view_init(elev=25, azim=45)
    
    plt.tight_layout()
    plt.savefig('zwei_substrat_mm_komplett.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Kompletter 3D-Plot mit Projektionen und Daten erstellt!")
    print(f"Modellgüte: R² = {r_squared:.3f}, RMSE = {rmse:.2f} μM/min")

def plot_3d_mit_projektionen():
    """Erstellt 3D-Oberflächenplot mit Ein-Substrat-Projektionen"""
    
    # Substratkonzentrationen definieren
    sa_range = np.linspace(0.5, 50, 50)
    sb_range = np.linspace(0.5, 50, 50)
    
    # Meshgrid erstellen
    SA, SB = np.meshgrid(sa_range, sb_range)
    
    # Reaktionsgeschwindigkeiten berechnen
    V = zwei_substrat_mm(SA, SB)
    
    # 3D-Plot erstellen
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Oberfläche plotten
    surf = ax.plot_surface(SA, SB, V, cmap='viridis', alpha=0.7, 
                          linewidth=0, antialiased=True)
    
    # Konstante Substratkonzentrationen für Projektionen
    sb_konstant = [5, 10, 20, 30]  # mM
    sa_konstant = [5, 10, 20, 30]  # mM
    
    # Farben für die Projektionslinien
    farben_a = ['red', 'orange', 'yellow', 'pink']
    farben_b = ['blue', 'cyan', 'green', 'purple']
    
    # Projektionen für konstantes [B] (variiert [A])
    for i, sb_const in enumerate(sb_konstant):
        sa_proj = sa_range
        sb_proj = np.full_like(sa_range, sb_const)
        v_proj = zwei_substrat_mm(sa_proj, sb_proj)
        
        # Linie auf der Oberfläche
        ax.plot(sa_proj, sb_proj, v_proj, color=farben_a[i], linewidth=3, 
                label=f'[B] = {sb_const} mM konstant')
        
        # Projektion auf x-z Ebene (Boden)
        ax.plot(sa_proj, np.full_like(sa_proj, 0.5), v_proj, 
                color=farben_a[i], linewidth=2, alpha=0.6, linestyle='--')
    
    # Projektionen für konstantes [A] (variiert [B])
    for i, sa_const in enumerate(sa_konstant):
        sa_proj = np.full_like(sb_range, sa_const)
        sb_proj = sb_range
        v_proj = zwei_substrat_mm(sa_proj, sb_proj)
        
        # Linie auf der Oberfläche
        ax.plot(sa_proj, sb_proj, v_proj, color=farben_b[i], linewidth=3, 
                label=f'[A] = {sa_const} mM konstant')
        
        # Projektion auf y-z Ebene (Seite)
        ax.plot(np.full_like(sb_proj, 0.5), sb_proj, v_proj, 
                color=farben_b[i], linewidth=2, alpha=0.6, linestyle='--')
    
    # Beschriftungen
    ax.set_xlabel('[Substrat A] (mM)', fontsize=12)
    ax.set_ylabel('[Substrat B] (mM)', fontsize=12)
    ax.set_zlabel('Reaktionsgeschwindigkeit v (μM/min)', fontsize=12)
    ax.set_title('Zwei-Substrat Michaelis-Menten Kinetik\nmit Ein-Substrat Projektionen', 
                fontsize=14, pad=20)
    
    # Achsengrenzen setzen
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 50)
    ax.set_zlim(0, np.max(V))
    
    # Colorbar hinzufügen
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, label='v (μM/min)')
    
    # Legend hinzufügen
    ax.legend(bbox_to_anchor=(1.15, 1), loc='upper left', fontsize=9)
    
    # Blickwinkel optimieren
    ax.view_init(elev=25, azim=45)
    
    # Plot speichern und anzeigen
    plt.tight_layout()
    plt.savefig('zwei_substrat_mm_3d_projektionen.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("3D-Plot mit Projektionen wurde erfolgreich erstellt!")

def plot_3d():
    """Erstellt einfachen 3D-Oberflächenplot"""
    
    # Substratkonzentrationen definieren
    sa_range = np.linspace(0.5, 50, 50)
    sb_range = np.linspace(0.5, 50, 50)
    
    # Meshgrid erstellen
    SA, SB = np.meshgrid(sa_range, sb_range)
    
    # Reaktionsgeschwindigkeiten berechnen
    V = zwei_substrat_mm(SA, SB)
    
    # 3D-Plot erstellen
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Oberfläche plotten
    surf = ax.plot_surface(SA, SB, V, cmap='viridis', alpha=0.8, 
                          linewidth=0, antialiased=True)
    
    # Beschriftungen
    ax.set_xlabel('[Substrat A] (mM)', fontsize=12)
    ax.set_ylabel('[Substrat B] (mM)', fontsize=12)
    ax.set_zlabel('Reaktionsgeschwindigkeit v (μM/min)', fontsize=12)
    ax.set_title('Zwei-Substrat Michaelis-Menten Kinetik\n(Ordered Sequential Mechanismus)', 
                fontsize=14, pad=20)
    
    # Colorbar hinzufügen
    fig.colorbar(surf, ax=ax, shrink=0.6, label='v (μM/min)')
    
    # Blickwinkel optimieren
    ax.view_init(elev=30, azim=45)
    
    # Plot speichern und anzeigen
    plt.tight_layout()
    plt.savefig('zwei_substrat_mm_3d.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("3D-Plot wurde erfolgreich erstellt und als 'zwei_substrat_mm_3d.png' gespeichert!")

if __name__ == "__main__":
    print("=== Zwei-Substrat Michaelis-Menten Plot Generator ===")
    print("Welchen Plot möchten Sie erstellen?")
    print("1 - Einfacher 3D-Plot (nur Theorie)")
    print("2 - 3D-Plot mit Ein-Substrat Projektionen")
    print("3 - 3D-Plot mit experimentellen Daten")
    print("4 - 3D-Plot mit Projektionen UND experimentellen Daten")
    print("5 - Beispieldaten erstellen")
    print("6 - Alle Plots erstellen")
    
    choice = input("\nEingabe (1-6): ").strip()
    
    if choice == "1":
        plot_3d()
    elif choice == "2":
        plot_3d_mit_projektionen()
    elif choice == "3":
        datei = input("Pfad zur CSV-Datei (Enter für simulierte Daten): ").strip()
        plot_3d_mit_daten(datei if datei else None)
    elif choice == "4":
        datei = input("Pfad zur CSV-Datei (Enter für simulierte Daten): ").strip()
        plot_3d_mit_projektionen_und_daten(datei if datei else None)
    elif choice == "5":
        erstelle_beispiel_daten()
        print("Möchten Sie die Beispieldaten jetzt plotten? (j/n): ")
        if input().lower().startswith('j'):
            plot_3d_mit_daten('beispiel_zwei_substrat_daten.csv')
    elif choice == "6":
        print("Erstelle alle Plots...")
        plot_3d()
        plot_3d_mit_projektionen()
        plot_3d_mit_daten()
        plot_3d_mit_projektionen_und_daten()
    else:
        print("Erstelle standardmäßig den Plot mit Projektionen und simulierten Daten...")
        plot_3d_mit_projektionen_und_daten()
