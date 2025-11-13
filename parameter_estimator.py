import numpy as np
import pandas as pd
import scipy.optimize as opt

from artifical_data import reaction1_synthetic_data

def estimate_parameters_standard(data, model_func, initial_guess):
    x_data = []
    for col in data.columns:
        if col != "activity_U/mg" and col != "rate":
            x_data.append(data[col].values)

    # Fit the model to the data
    y_data = data["activity_U/mg"].values
    try:
        popt, pcov = opt.curve_fit(model_func, x_data, y_data, p0=initial_guess, maxfev=10000, bounds=(0, 500))
    except RuntimeError as e:
        print(f"Error occurred during curve fitting: {e}")
        input("Press Enter to continue...")
        return None, None
        
    return popt, pcov

def estimate_parameters_adaptive(data, model_func, substrate, initial_guess, method='multi_start'):
    x_data = []

    if len(substrate) == 1:
        x_data = data[substrate[0]].values
    else:
        for col in substrate:
            x_data.append(data[col].values)

    y_data = data["activity_U/mg"].values

    if len(y_data) < len(initial_guess):
        raise ValueError("Not enough data points to estimate parameters.")
    
    if method == "multi_start":
        return _multi_start_fit(x_data, y_data, model_func, initial_guess)
    elif method == 'regularized':
        return _regularized_fitting(x_data, y_data, model_func, initial_guess)
    elif method == 'bootstrap':
        return _bootstrap_fitting(x_data, y_data, model_func, initial_guess)
    else:
        return _standard_fitting(x_data, y_data, model_func, initial_guess)

def _multi_start_fit(x_data, y_data, model_func, initial_guess, n_starts=10):
    best_result = None
    best_error = np.inf

    for _ in range(n_starts):
        noise_factor =  0.3
        perburbed_guess = [guess * (1 + np.random.uniform(-noise_factor, noise_factor))
            for i, guess in enumerate(initial_guess)]

        try:
            popt, pcov = opt.curve_fit(
                    model_func, x_data, y_data, 
                    p0=perburbed_guess, 
                    maxfev=10000,
                    bounds=(0, 1000),  # Erweiterte Bounds
                    method='trf'  # Trust Region Reflective - robuster
                )
            
            residuals = y_data - model_func(x_data, *popt)
            error = np.sum(residuals**2)

            if error < best_error:
                best_error = error
                best_result = (popt, pcov)

        except (RuntimeError, ValueError) as e:
            continue

    if best_result is None:
        print("All fitting attempts failed.")
        input("Press Enter to continue...")
        return None, None
    
    return best_result


def _standard_fitting(x_data, y_data, model_func, initial_guess):
    try:
        popt, pcov = opt.curve_fit(
            model_func, x_data, y_data, 
            p0=initial_guess, 
            maxfev=10000,
            bounds=(0, 500)  # Beispielhafte Bounds
        )
    except RuntimeError as e:
        print(f"Error occurred during curve fitting: {e}")
        input("Press Enter to continue...")
        return None, None

    best_result = (popt, pcov)
    return best_result

def _regularized_fitting(x_data, y_data, model_func, initial_guess):
    """Regularized fitting für wenige Datenpunkte"""
    def regularized_objective(params, x_data, y_data, lambda_reg=0.01):
        try:
            predicted = model_func(x_data, *params)
            mse = np.mean((y_data - predicted)**2)
            # L2-Regularisierung
            regularization = lambda_reg * np.sum(params**2)
            return mse + regularization
        except:
            return 1e10
    
    try:
        result = opt.minimize(
            regularized_objective,
            initial_guess,
            args=(x_data, y_data),
            method='L-BFGS-B',
            bounds=[(0.1, 1000) for _ in initial_guess]
        )
        
        if result.success:
            # Approximiere Kovarianz
            pcov = np.eye(len(result.x)) * 0.1
            return result.x, pcov
        else:
            return None, None
    except Exception as e:
        print(f"Regularized fitting failed: {e}")
        return None, None

def _bootstrap_fitting(x_data, y_data, model_func, initial_guess, n_bootstrap=100):
    """Bootstrap für Unsicherheitsschätzung bei wenigen Daten"""
    
    bootstrap_params = []
    
    for i in range(n_bootstrap):
        # Bootstrap sample
        indices = np.random.choice(len(y_data), size=len(y_data), replace=True)
        x_boot = [x[indices] for x in x_data]
        y_boot = y_data[indices]
        
        try:
            popt, _ = opt.curve_fit(
                model_func, x_boot, y_boot,
                p0=initial_guess,
                maxfev=10000,
                bounds=(0, 500)
            )
            bootstrap_params.append(popt)
        except:
            continue
    
    if bootstrap_params:
        bootstrap_params = np.array(bootstrap_params)
        mean_params = np.mean(bootstrap_params, axis=0)
        param_cov = np.cov(bootstrap_params.T)
        return mean_params, param_cov
    else:
        return _standard_fitting(x_data, y_data, model_func, initial_guess)

if __name__ == "__main__":

    # Generate synthetic data for testing
    true_parameters = (100, 2, 3)  # Vmax, Km1, Km2
    synthetic_data = reaction1_synthetic_data(true_parameters)

    # Define the Michaelis-Menten model function
    def michaelis_menten(S, Vmax, Km1, Km2):
        S1, S2 = S
        return (Vmax * S1 * S2) / ((Km1 + S1) * (Km2 + S2))

    # Initial guess for parameters
    initial_guess = [80, 1, 1]

    # Estimate parameters
    estimated_params, covariance = estimate_parameters_adaptive(synthetic_data, michaelis_menten, initial_guess)

    print("Estimated Parameters:")
    print(f"Vmax: {estimated_params[0]}")
    print(f"Km1: {estimated_params[1]}")
    print(f"Km2: {estimated_params[2]}")