import os 
import pandas as pd
import numpy as np
from scipy.stats import linregress
import pickle

def is_linear(x, y, threshold=0.60):
    """
    Überprüft, ob eine lineare Beziehung zwischen x und y besteht.
    
    Args:
        x: Unabhängige Variable (z.B. Zeit)
        y: Abhängige Variable (z.B. Absorption)
        threshold: Mindest-R-Wert für Linearität (Standard: 0.80)
    
    Returns:
        tuple: (slope, r_squared) wenn linear, sonst (False, r_squared)
    """
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    is_linear = r_value**2 >= threshold
    if is_linear:
        return slope, r_value**2
    else:
        return False, r_value**2
    
def get_time_points(data):
    """
    Extrahiert Zeitpunkte aus den Spaltennamen der Daten.
    
    Args:
        data: DataFrame mit Spaltennamen im Format "data_<time>"
    
    Returns:
        list: Liste der Zeitpunkte als Integer
    """
    
    time_points_strings = data.columns
    time_points = [int(tp.split("_")[1]) for tp in time_points_strings if "data_" in tp]
    
    return time_points

def get_concentration_data(data, concentration_columns):
    """
    Extrahiert die Konzentrationsdaten der Substrate.
    
    Args:
        data: DataFrame mit allen Daten
        concentration_columns: Liste der Spaltennamen für Konzentrationen
    
    Returns:
        DataFrame: Eindeutige Kombinationen der Substratkonzentrationen
    """

    concentration_data = data[concentration_columns].drop_duplicates().reset_index(drop=True)

    return concentration_data

def get_absorbance_data(data):
    """
    Extrahiert die Absorptionsdaten (Zeitreihen) aus dem DataFrame.
    
    Args:
        data: DataFrame mit allen Daten
    
    Returns:
        DataFrame: Nur die Absorptionsspalten (data_<time>)
    """

    time_points_strings = [dp for dp in data.columns if "data_" in dp]
    absorbance_data = data[time_points_strings]

    return absorbance_data

def get_calibration_slope(data):
    """
    Berechnet die Kalibriersteigung aus den Kalibrierdaten.
    
    Args:
        data: DataFrame mit Kalibrierdaten (Spalten: "c", "ad1", "ad2")
    
    Returns:
        tuple: Steigung und R²-Wert der Kalibrierung
    """

    x = data["c"]
    y = data[["ad1", "ad2"]].mean(axis=1)

    slope, r_squared = is_linear(x, y, threshold=0.98)

    return slope


def get_calibration_slope_with_noise(data, noise_levels):
    """
    Berechnet die Kalibriersteigung aus den Kalibrierdaten.
    
    Args:
        data: DataFrame mit Kalibrierdaten (Spalten: "c", "ad1", "ad2")
    
    Returns:
        tuple: Steigung und R²-Wert der Kalibrierung
    """

    x = data["c"]
    y = data[["ad1", "ad2"]].mean(axis=1)

    x_noise = x + np.random.normal(0, noise_levels['fehler_pipettieren'], size=x.shape)
    y_noise = y + np.random.normal(0, noise_levels['fehler_od'], size=y.shape)

    slope, r_squared = is_linear(x, y)
        
    # Always return the slope, even if it's not considered "linear enough"
    if slope is False:
            # Calculate slope anyway using linear regression
            from scipy.stats import linregress
            slope_val, intercept, r_value, p_value, std_err = linregress(x, y)
            print(f"Warning: Calibration data is not linear (R² = {r_squared:.3f}), using slope = {slope_val:.3f}")
            return slope_val
        
    return slope

def get_reaction_slope(time_points, absorbance_values):
    """
    Berechnet die Reaktionssteigung aus Zeitpunkten und Absorptionswerten.
    
    Args:
        time_points: Liste/Array der Zeitpunkte
        absorbance_values: Liste/Array der Absorptionswerte
    
    Returns:
        tuple: (slope, r_value) der linearen Regression
    """

    slope, r_value = is_linear(time_points, absorbance_values)

    return slope, r_value

def convert_ad_to_concentration(ad_slope, calc_slope, parameters):
    """
    Konvertiert die Absorptionssteigung in Enzymaktivität (U/mg).
    
    Args:
        ad_slope: Steigung der Absorption über Zeit
        calc_slope: Steigung der Kalibrierung
        parameters: Dictionary mit experimentellen Parametern (Vf_well, Vf_prod, c_prod)
    
    Returns:
        float: Enzymaktivität in U/mg
    """
    if calc_slope is None or calc_slope == 0:
        print("Warning: Invalid calibration slope, using default value of 1.0")
        calc_slope = 1.0
    
    # Formel zur Berechnung der Enzymaktivität in U/mg
    activity_U_per_mg = (abs(ad_slope) * 60 * parameters["Vf_well"] * parameters["Vf_prod"]) / (calc_slope * parameters["c_prod"])

    return activity_U_per_mg


def process_duplicates(ad_data):
    """
    Verarbeitet Duplikate in den Absorptionsdaten durch Mittelung.
    
    Args:
        ad_data: DataFrame mit Absorptionsdaten (enthält Duplikate)
    
    Returns:
        DataFrame: Gemittelte Daten ohne Duplikate
    """

    data_without_duplicates = pd.DataFrame(columns=ad_data.columns)
    
    values = ad_data.values
    # Verarbeite jeweils zwei aufeinanderfolgende Zeilen als Duplikate
    for i in range(0, len(ad_data.values), 2):
        row1 = values[i]
        row2 = values[i+1]
        print(row1-row2)

        # Berechne Mittelwert der beiden Duplikate
        averaged_row = (row1 + row2) / 2
        data_without_duplicates.loc[i] = averaged_row

    return data_without_duplicates


def process_duplicates2(full_data, concentration_columns):
    """
    Verarbeitet Duplikate durch Mittelung aller Zeilen mit gleichen Konzentrationen.
    
    Args:
        full_data: DataFrame mit allen Daten (Konzentrationen + Absorptionen)
        concentration_columns: Liste der Konzentrationsspalten
    
    Returns:
        DataFrame: Gemittelte Daten ohne Duplikate
    """
    unique_concentrations = full_data[concentration_columns].drop_duplicates().reset_index(drop=True)
    processed_data = pd.DataFrame()

    for index, conc_row in unique_concentrations.iterrows():
        # Erstelle Maske für Zeilen mit gleichen Konzentrationen
        mask = True
        for col in concentration_columns:
            mask &= (full_data[col] == conc_row[col])
        
        # Finde alle Zeilen mit dieser Konzentrationskombination
        matching_rows = full_data[mask]
        
        if len(matching_rows) > 1:
            # Mittelwert aller numerischen Spalten bilden
            averaged_row = matching_rows.mean(numeric_only=True)

            # Konzentrationswerte explizit setzen
            for col in concentration_columns:
                averaged_row[col] = conc_row[col]
        else:
            averaged_row = matching_rows.iloc[0]
        
        processed_data = pd.concat([processed_data, averaged_row.to_frame().T], ignore_index=True)
    
    return processed_data

def get_processed_data(time_points, conc_data, ad_data, cal_scope, substrates, cal_parameters):

    activity_list = []
    regi_results = []
    
    # Verarbeite jede Zeile der Absorptionsdaten
    for index, row in ad_data.iterrows():
        test = row.values
        # Entferne NaN-Werte und entsprechende Zeitpunkte
        mask = ~np.isnan(test)
        test = test[mask]
        time_points_filtered = np.array(time_points)[mask]
        
        # Verwende gefilterte Zeitpunkte für lineare Regression
        if len(test) > 1:  # Benötige mindestens 2 Punkte für lineare Regression
            ad_slope, r = is_linear(time_points_filtered, test)
            regi_results.append((index, ad_slope, r))
            if ad_slope:
                activity = convert_ad_to_concentration(ad_slope, cal_scope, cal_parameters)
                activity_list.append(activity)
            else:
                #print(f"Row {index}: No linear region found.Row {index}: R^2 = {r}")
                conc_data = conc_data.drop(index)
        else:
            print(f"Row {index}: Not enough data points for linear regression.")
            conc_data = conc_data.drop(index)
        
    
    # Erstelle Ergebnis-Dictionary mit Substratkonzentrationen und Aktivitäten
    dict_results = { }
    for sub_names in substrates:
        dict_results[sub_names] = conc_data[sub_names]
    dict_results["activity_U/mg"] = activity_list
    
    df = pd.DataFrame(dict_results)

    return df, regi_results


def add_noise(df, keys, noise_level=0.01):
    """
    Fügt Gaussian-Rauschen zu spezifischen Spalten eines DataFrames hinzu.
    
    Args:
        df: Eingabe-DataFrame
        keys: Liste der Spaltennamen, zu denen Rauschen hinzugefügt werden soll
        noise_level: Standardabweichung des Rauschens (Standard: 0.01)
    
    Returns:
        DataFrame: DataFrame mit hinzugefügtem Rauschen
    """
    noisy_df = df.copy()
    # Füge zu jeder angegebenen Spalte Gaussian-Rauschen hinzu
    for key in keys:
        if key in noisy_df.columns:
            noise = np.random.normal(0, noise_level, size=noisy_df[key].shape)
            noisy_df[key] += noise
    return noisy_df

def add_noise_function(df, keys, function, noise_level=0.01):

    noise_level = function(noise_level)
    noisy_df = df.copy()
    # Füge zu jeder angegebenen Spalte Gaussian-Rauschen hinzu
    for ikey, key in enumerate(keys):
        if key in noisy_df.columns:
            noise = np.random.normal(0, noise_level[ikey], size=noisy_df[key].shape)
            noisy_df[key] += noise
    return noisy_df

def get_noise_level_od(df,index1, index2):

    ad_values1 = df.loc[index1].values
    ad_values1 = ad_values1[~np.isnan(ad_values1)]

    ad_values2 = df.loc[index2].values
    ad_values2 = ad_values2[~np.isnan(ad_values2)]
    diffs = ad_values1 - ad_values2
    noise_level_od = np.std(diffs)

    return noise_level_od

    
if __name__ == "__main__":
    """
    Beispiel-Ausführung zur Demonstration der Datenverarbeitung.
    """

    data_path = r"C:\Users\berger\Documents\Projekts\enzyme-cascade-analysis\example_reactions\dortmund_system\documentation\experimental_data\Reaction3\r_3_NAD.csv"
    data = pd.read_csv(data_path)
    keys = [dp for dp in data.columns if "data_" in dp]

    data = data[keys]

    std = []
    for i in range(0, len(data.values), 2):
        print(get_noise_level_od(data,i,i+1))
        std.append( get_noise_level_od(data,i,i+1))

    print("Durchschnittliches Rauschen OD:", np.mean(std))