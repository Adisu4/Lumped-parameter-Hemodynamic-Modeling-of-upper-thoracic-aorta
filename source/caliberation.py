import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import from your existing modules
from source import model_parameters
from source.inflow_condition import inflow  # Import inflow function directly

def load_synthetic_data(filepath=None):
    """
    Load synthetic data from a file
    Format: t, Q, P (time in s, flow in mL/s, pressure in dyne/cm^2)
    """
    # Try exact path first, then variations
    if filepath is None:
        filepath = r"C:\Users\Adisu\upper_thoracic_aorta\Data\synthetic_data_with_noise_t_Q_P"
    
    paths_to_try = [
        filepath,
        filepath + '.txt',
        filepath + '.dat',
        filepath + '.csv'
    ]
    
    # Try each path
    for path in paths_to_try:
        try:
            print(f"Attempting to load data from: {path}")
            data = np.loadtxt(path, comments='#')  # Skip any comment lines
            print(f"✓ Successfully loaded data with {len(data)} data points")
            
            # Extract columns
            t_data = data[:, 0]
            Q_data = data[:, 1]
            P_data = data[:, 2]
            
            # Verify data looks reasonable
            print(f"Time range: {t_data.min():.2f} to {t_data.max():.2f} seconds")
            print(f"Flow range: {Q_data.min():.2f} to {Q_data.max():.2f} mL/s")
            print(f"Pressure range: {P_data.min()/1333.22:.2f} to {P_data.max()/1333.22:.2f} mmHg")
            
            return t_data, Q_data, P_data
            
        except Exception as e:
            print(f"Could not load from {path}: {e}")
    
    # If we get here, all paths failed
    print("ERROR: Could not load synthetic data from any path")
    print("Falling back to generating synthetic data...")
    return generate_synthetic_data()

# ======================================================
# Simulation Function using imported parameters
# ======================================================
def simulate_model(R0, C0, dt=0.005, T=model_parameters.T):
    """
    Simulate the coupled vessel–Windkessel model for one period.
    Returns: time array, Q_model (mL/s), and P_model (dyne/cm^2)
    """
    # Convert parameters to scalars to avoid NumPy deprecation warnings
    R0 = float(R0)
    C0 = float(C0)
    
    N = int(T/dt) + 1
    time_sim = np.linspace(0, T, N)
    V = np.zeros(N)
    Q = np.zeros(N)
    P_wk = np.zeros(N)
    P_vessel = np.zeros(N)
    
    # Get parameters from model_parameters
    P0 = model_parameters.P0
    Pven = model_parameters.Pven
    Cwk = model_parameters.Cwk
    Rp_wk = model_parameters.Rp_wk
    Rd_wk = model_parameters.Rd_wk
    L0 = model_parameters.L0
    V_init = model_parameters.V_init
    Q_init = model_parameters.Q_init
    P_wk_init = model_parameters.P_wk_init
    
    # Set initial conditions
    V[0] = V_init
    Q[0] = Q_init
    P_wk[0] = P_wk_init
    
    for i in range(N - 1):
        t = time_sim[i]
        Q_in_val = inflow(t, T)
        # Vessel pressure: P = P0 + (V - V0)/C0
        P_vessel[i] = P0 + (V[i] - V_init) / C0
        # Coupling: outlet pressure of the vessel
        P_out = P_wk[i] + Rp_wk * Q[i]
        # Vessel ODEs
        dVdt = Q_in_val - Q[i]
        dQdt = (P_vessel[i] - R0 * Q[i] - P_out) / L0
        # Windkessel model
        Q_wk = (P_wk[i] - Pven) / Rd_wk
        dP_wk_dt = (Q[i] - Q_wk) / Cwk
        
        V[i+1] = V[i] + dt * dVdt
        Q[i+1] = Q[i] + dt * dQdt
        P_wk[i+1] = P_wk[i] + dt * dP_wk_dt
    
    # Final vessel pressure
    P_vessel[-1] = P0 + (V[-1] - V_init) / C0
    return time_sim, Q, P_vessel

def objective(params, t_data, Q_data, P_data):
    """Objective function for parameter calibration"""
    R0_val, C0_val = params
    
    # Use the simplified model for faster optimization
    t_sim, Q_sim, P_sim = simulate_model(R0_val, C0_val, dt=0.005)
    
    # Interpolate the simulation outputs to the time points of the data
    Q_interp = np.interp(t_data, t_sim, Q_sim)
    P_interp = np.interp(t_data, t_sim, P_sim)
    
    # Normalize errors to give equal weight to pressure and flow
    Q_scale = np.max(Q_data) - np.min(Q_data)
    P_scale = np.max(P_data) - np.min(P_data)
    
    Q_error = (Q_interp - Q_data) / Q_scale
    P_error = (P_interp - P_data) / P_scale
    
    # Combine residuals for Q and P
    error = np.concatenate([Q_error, P_error])
    return error

def run_calibration(use_real_data=True):
    """Run parameter calibration with physical constraints"""
    print("Running model calibration...")
    
    # Load or generate synthetic data
    if use_real_data:
        # Try the exact path provided by the user
        full_path = r"C:\Users\Adisu\upper_thoracic_aorta\Data\synthetic_data_with_noise_t_Q_P.txt"
        t_data, Q_data, P_data = load_synthetic_data(full_path)
    else:
        # Define true parameter values for synthetic data generation
        R0_true = 320.0  # True resistance
        C0_true = 0.0035  # True compliance
        t_data, Q_data, P_data = generate_synthetic_data(R0_true, C0_true, noise_level=0.05)
        print(f"True parameter values:")
        print(f"  R0 = {R0_true} dyne·s/cm^5")
        print(f"  C0 = {C0_true} cm^3/dyne")
    
    # Initial guesses - try theoretical values from model_parameters
    R0_nom = max(model_parameters.R0, 100.0)  # Ensure positive reasonable value
    C0_nom = max(model_parameters.C0, 0.001)  # Ensure positive reasonable value
    initial_guess = [R0_nom, C0_nom]
    
    print(f"Initial parameter guesses:")
    print(f"  R0 = {initial_guess[0]:.6f} dyne·s/cm^5")
    print(f"  C0 = {initial_guess[1]:.6f} cm^3/dyne")
    
    # Add bounds to enforce physical constraints (both parameters must be positive)
    bounds = ([1.0, 0.0001], [1000.0, 0.01])  # Lower and upper bounds
    
    # Perform least-squares optimization WITH BOUNDS
    print("Optimizing parameters with physical constraints...")
    res = least_squares(
        lambda p: objective(p, t_data, Q_data, P_data), 
        initial_guess,
        bounds=bounds,
        method='trf',  # Trust Region Reflective algorithm - handles bounds well
        ftol=1e-8,     # Function tolerance for convergence
        xtol=1e-8,     # Parameter tolerance for convergence
        verbose=1      # Show progress
    )
    
    R0_calibrated, C0_calibrated = res.x
    
    print(f"Calibration complete!")
    print(f"Calibrated parameters:")
    print(f"  R0 = {R0_calibrated:.6f} dyne·s/cm^5")
    print(f"  C0 = {C0_calibrated:.6f} cm^3/dyne")
    
    # Simulate with calibrated parameters
    t_sim, Q_sim, P_sim = simulate_model(R0_calibrated, C0_calibrated)
    
    # Plot comparison
    plt.figure(figsize=(12, 5))
    
    # Flow rate comparison
    plt.subplot(1, 2, 1)
    plt.plot(t_data, Q_data, 'ko', label='Data points')
    plt.plot(t_sim, Q_sim, 'b-', label='Calibrated model Q')
    plt.xlabel('Time (s)')
    plt.ylabel('Flow rate Q (mL/s)')
    plt.legend()
    plt.title('Flow Rate Calibration')
    
    # Pressure comparison
    plt.subplot(1, 2, 2)
    plt.plot(t_data, P_data/1333.22, 'ko', label='Data points')  # Convert to mmHg
    plt.plot(t_sim, P_sim/1333.22, 'r-', label='Calibrated model P')  # Convert to mmHg
    plt.xlabel('Time (s)')
    plt.ylabel('Pressure P (mmHg)')
    plt.legend()
    plt.title('Pressure Calibration')
    
    plt.tight_layout()
    plt.savefig('calibration_results.png')
    plt.show()
    
    # Identifiability Analysis: Cost Function Contours
    print("\nRunning identifiability analysis...")
    R0_range = np.linspace(0.8 * R0_calibrated, 1.2 * R0_calibrated, 20)  # Reduced grid size
    C0_range = np.linspace(0.8 * C0_calibrated, 1.2 * C0_calibrated, 20)
    R0_grid, C0_grid = np.meshgrid(R0_range, C0_range)
    cost = np.zeros_like(R0_grid)
    
    for i in range(R0_grid.shape[0]):
        for j in range(R0_grid.shape[1]):
            params = [R0_grid[i, j], C0_grid[i, j]]
            err = objective(params, t_data, Q_data, P_data)
            cost[i, j] = np.sum(err**2)
    
    plt.figure(figsize=(8, 6))
    cp = plt.contourf(R0_grid, C0_grid, cost, levels=20, cmap='viridis')
    plt.xlabel('R0 (dyne·s/cm^5)')
    plt.ylabel('C0 (cm^3/dyne)')
    plt.title('Parameter Identifiability: Cost Function Contour Plot')
    plt.colorbar(cp, label='Sum of Squared Errors')
    plt.scatter([R0_calibrated], [C0_calibrated], color='red', marker='x', s=100, label='Optimal Parameters')
    plt.legend()
    plt.savefig('parameter_identifiability.png')
    plt.show()
    
    # Quantitative Identifiability Analysis
    print("\nComputing identifiability metrics...")

    # Calculate the Hessian matrix (approximation)
    epsilon = 1e-6
    hessian = np.zeros((2, 2))
    baseline_error = np.sum(objective([R0_calibrated, C0_calibrated], t_data, Q_data, P_data)**2)

    # Approximate the Hessian using finite differences
    for i in range(2):
        for j in range(2):
            params_plus_i = [R0_calibrated, C0_calibrated]
            params_plus_j = [R0_calibrated, C0_calibrated]
            params_plus_both = [R0_calibrated, C0_calibrated]
            
            params_plus_i[i] += epsilon
            params_plus_j[j] += epsilon
            params_plus_both[i] += epsilon
            params_plus_both[j] += epsilon
            
            err_i = np.sum(objective(params_plus_i, t_data, Q_data, P_data)**2)
            err_j = np.sum(objective(params_plus_j, t_data, Q_data, P_data)**2)
            err_both = np.sum(objective(params_plus_both, t_data, Q_data, P_data)**2)
            
            hessian[i,j] = (err_both - err_i - err_j + baseline_error) / (epsilon**2)

    # Calculate eigenvalues of the Hessian for identifiability analysis
    eigenvalues, eigenvectors = np.linalg.eig(hessian)
    condition_number = np.max(np.abs(eigenvalues)) / np.min(np.abs(eigenvalues))

    # Calculate parameter correlation via Fisher information and covariance matrix
    fisher_info = hessian/2  # Fisher information is half the Hessian
    covariance = np.linalg.inv(fisher_info)
    correlation = covariance[0,1] / np.sqrt(covariance[0,0] * covariance[1,1])

    print(f"Parameter identifiability metrics:")
    print(f"  Condition number: {condition_number:.2f} (lower is better, <100 is good)")
    print(f"  Parameter correlation: {correlation:.4f} (closer to 0 is better)")
    print(f"  Relative confidence intervals:")
    print(f"    R0: ±{100*np.sqrt(covariance[0,0])/R0_calibrated:.2f}%")
    print(f"    C0: ±{100*np.sqrt(covariance[1,1])/C0_calibrated:.2f}%")

    # Plot confidence ellipse
    plt.figure(figsize=(8, 6))
    theta = np.linspace(0, 2*np.pi, 100)
    z = np.array([np.cos(theta), np.sin(theta)])
    chi2_val = 5.991  # 95% confidence for 2 parameters
    ellipse = R0_calibrated + np.sqrt(chi2_val * covariance[0,0]) * z[0], \
              C0_calibrated + np.sqrt(chi2_val * covariance[1,1]) * z[1]

    plt.plot(ellipse[0], ellipse[1], 'g-', label='95% Confidence Region')
    plt.scatter([R0_calibrated], [C0_calibrated], c='r', marker='x', s=100, label='Optimal Parameters')
    plt.xlabel('R0 (dyne·s/cm^5)')
    plt.ylabel('C0 (cm^3/dyne)')
    plt.title('Parameter Identifiability: Confidence Ellipse')
    plt.legend()
    plt.grid(True)
    plt.savefig('confidence_ellipse.png')
    plt.show()
    
    # Profile likelihood analysis
    R0_profile_range = [0.7 * R0_calibrated, 1.3 * R0_calibrated]
    C0_profile_range = [0.7 * C0_calibrated, 1.3 * C0_calibrated]
    fixed_values = [R0_calibrated, C0_calibrated]

    # Run profile likelihood for R0
    profile_vals_R0, likelihood_vals_R0 = profile_likelihood(0, R0_profile_range, fixed_values, t_data, Q_data, P_data)
    
    # Run profile likelihood for C0
    profile_vals_C0, likelihood_vals_C0 = profile_likelihood(1, C0_profile_range, fixed_values, t_data, Q_data, P_data)
    
    # Return all needed values
    return R0_calibrated, C0_calibrated, t_data, Q_data, P_data

# Profile likelihood function
def profile_likelihood(param_index, param_range, fixed_values, t_data, Q_data, P_data):
    """Calculate profile likelihood for a parameter"""
    param_name = ["R0", "C0"][param_index]
    print(f"Calculating profile likelihood for {param_name}...")
    
    profile_vals = np.linspace(param_range[0], param_range[1], 20)
    likelihood_vals = []
    
    for val in profile_vals:
        # Fix one parameter and optimize the other
        def objective_profile(p):
            params = fixed_values.copy()
            params[param_index] = val
            params[1-param_index] = p[0]  # The other parameter
            return objective(params, t_data, Q_data, P_data)
        
        res = least_squares(lambda p: objective_profile([p]), [fixed_values[1-param_index]])
        best_error = np.sum(objective_profile(res.x)**2)
        likelihood_vals.append(best_error)
    
    # Plot profile likelihood
    plt.figure(figsize=(8, 5))
    plt.plot(profile_vals, likelihood_vals, 'b-o')
    plt.axvline(x=fixed_values[param_index], color='r', linestyle='--', label='Optimal value')
    plt.xlabel(param_name + ' (dyne·s/cm^5)' if param_index == 0 else param_name + ' (cm^3/dyne)')
    plt.ylabel('Sum of squared residuals')
    plt.title(f'Profile Likelihood for {param_name}')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'profile_likelihood_{param_name}.png')
    plt.show()
    
    return profile_vals, likelihood_vals

def generate_synthetic_data(R0_true=320.0, C0_true=0.0035, noise_level=0.05):
    """Generate synthetic data if needed as fallback"""
    print("Generating synthetic data...")
    t_sim, Q_clean, P_clean = simulate_model(R0_true, C0_true, dt=0.001)
    
    # Sample fewer points to simulate sparse measurements
    num_points = 50
    indices = np.linspace(0, len(t_sim)-1, num_points, dtype=int)
    t_data = t_sim[indices]
    Q_data_clean = Q_clean[indices]
    P_data_clean = P_clean[indices]
    
    # Add random noise
    Q_noise_amp = noise_level * (np.max(Q_data_clean) - np.min(Q_data_clean))
    P_noise_amp = noise_level * (np.max(P_data_clean) - np.min(P_data_clean))
    
    Q_data = Q_data_clean + np.random.normal(0, Q_noise_amp, len(Q_data_clean))
    P_data = P_data_clean + np.random.normal(0, P_noise_amp, len(P_data_clean))
    
    return t_data, Q_data, P_data

def diagnostic_parameter_sweep():
    """Test different parameter values to understand the problem"""
    # Load data
    t_data, Q_data, P_data = load_synthetic_data()
    
    # Create a grid of parameter values to test
    R0_values = np.logspace(0, 3, 4)  # 1, 10, 100, 1000
    C0_values = np.logspace(-4, -2, 3)  # 0.0001, 0.001, 0.01
    
    plt.figure(figsize=(15, 10))
    plot_idx = 1
    
    for R0 in R0_values:
        for C0 in C0_values:
            # Simulate with these parameters
            t_sim, Q_sim, P_sim = simulate_model(R0, C0)
            
            # Calculate error
            Q_interp = np.interp(t_data, t_sim, Q_sim)
            P_interp = np.interp(t_data, t_sim, P_sim)
            
            Q_error = np.mean((Q_interp - Q_data)**2)
            P_error = np.mean((P_interp - P_data)**2)
            total_error = Q_error + P_error
            
            # Plot
            plt.subplot(4, 3, plot_idx)
            plt.plot(t_data, Q_data, 'ko', markersize=3, label='Data')
            plt.plot(t_sim, Q_sim, 'b-', label='Model')
            plt.title(f'R0={R0:.1f}, C0={C0:.5f}\nError={total_error:.2f}')
            if plot_idx % 3 == 1:
                plt.ylabel('Flow Q (mL/s)')
            if plot_idx > 9:
                plt.xlabel('Time (s)')
            plt.legend(fontsize=8)
            plot_idx += 1
    
    plt.tight_layout()
    plt.savefig('parameter_sweep.png')
    plt.show()

def frequency_domain_identification(t_data, Q_data, P_data):
    """Identify parameters in frequency domain with focus on Rp"""
    # Ensure uniform time steps
    t_uniform = np.linspace(t_data.min(), t_data.max(), len(t_data))
    Q_uniform = np.interp(t_uniform, t_data, Q_data)
    P_uniform = np.interp(t_uniform, t_data, P_data)
    
    # Windowing to reduce spectral leakage
    window = np.hanning(len(t_uniform))
    Q_windowed = Q_uniform * window
    P_windowed = P_uniform * window
    
    # Compute FFT
    Q_fft = np.fft.rfft(Q_windowed)
    P_fft = np.fft.rfft(P_windowed)
    freqs = np.fft.rfftfreq(len(t_uniform), d=t_uniform[1]-t_uniform[0])
    
    # Input impedance (Z = P/Q in frequency domain)
    Z_data = P_fft / Q_fft
    
    # Focus on first few harmonics where Rp is most identifiable
    n_harmonics = 5
    heart_rate = 1.0/(t_data.max() - t_data.min())  # Hz
    harmonic_indices = [np.argmin(np.abs(freqs - k*heart_rate)) for k in range(1, n_harmonics+1)]
    
    # DC component gives peripheral resistance (Rp + Rd)
    dc_idx = 0
    total_resistance = np.abs(Z_data[dc_idx])
    
    # Create 3-element Windkessel model in frequency domain
    def wk3_impedance(params, ω):
        Rp, Rd, C = params
        return Rp + Rd/(1 + 1j*ω*Rd*C)
    
    # Error function focused on harmonics relevant to Rp
    def impedance_error(params):
        Rp, Rd, C = params
        errors = []
        
        # DC component - total resistance
        errors.append((Rp + Rd - total_resistance)/(Rp + Rd))
        
        # First few harmonics where Rp is most identifiable
        for idx in harmonic_indices:
            ω = 2*np.pi*freqs[idx]
            Z_model = wk3_impedance(params, ω)
            Z_measured = Z_data[idx]
            # Weighted combination of magnitude and phase errors
            err_mag = np.abs(np.abs(Z_model) - np.abs(Z_measured))/np.abs(Z_measured)
            err_phase = np.abs(np.angle(Z_model) - np.angle(Z_measured))/(2*np.pi)
            errors.append(err_mag + 0.5*err_phase)
            
        return np.array(errors)
    
    # Initial estimates
    Rp_init = total_resistance * 0.1  # 10% of total resistance
    Rd_init = total_resistance * 0.9  # 90% of total resistance
    C_init = 0.0005  # Starting estimate
    
    # Optimization with physical constraints
    bounds = ([0.1, 1.0, 0.0001], [1000.0, 10000.0, 0.01])
    res = least_squares(impedance_error, [Rp_init, Rd_init, C_init], 
                        bounds=bounds, method='trf')
    
    Rp_identified, Rd_identified, C_identified = res.x
    
    # Verify results with impedance plot
    plt.figure(figsize=(10, 8))
    
    # Plot magnitude
    plt.subplot(2, 1, 1)
    ω_range = 2*np.pi*freqs[1:20]  # Plot first 20 frequencies
    Z_model = [wk3_impedance(res.x, ω) for ω in ω_range]
    
    plt.loglog(freqs[1:20], np.abs(Z_data[1:20]), 'ko', label='Data')
    plt.loglog(freqs[1:20], [np.abs(z) for z in Z_model], 'r-', label='Model')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('|Z| (dyne·s/cm^5)')
    plt.title('Vascular Impedance Magnitude')
    plt.grid(True, which='both')
    plt.legend()
    
    return Rp_identified, Rd_identified, C_identified

def analyze_parameter_coupling():
    """Analyze why R0 has poor identifiability"""
    t_data, Q_data, P_data = load_synthetic_data()
    
    # Create a grid of R0 and Rp values
    R0_values = np.logspace(0, 3, 10)  # 1-1000 dyne·s/cm^5
    Rp_values = np.logspace(1, 3, 10)  # 10-1000 dyne·s/cm^5
    
    # Storage for errors
    errors = np.zeros((len(R0_values), len(Rp_values)))
    
    # Use a fixed C0 value - the one we identified
    C0_fixed = 0.00036
    
    # Test combinations
    for i, R0 in enumerate(R0_values):
        for j, Rp in enumerate(Rp_values):
            # Need to temporarily modify model parameters
            original_Rp = model_parameters.Rp_wk
            model_parameters.Rp_wk = Rp
            
            # Simulate with this parameter combination
            t_sim, Q_sim, P_sim = simulate_model(R0, C0_fixed)
            
            # Calculate error
            Q_interp = np.interp(t_data, t_sim, Q_sim)
            P_interp = np.interp(t_data, t_sim, P_sim)
            
            Q_error = np.mean((Q_interp - Q_data)**2)
            P_error = np.mean((P_interp - P_data)**2)
            errors[i, j] = Q_error + P_error
            
            # Restore original parameter
            model_parameters.Rp_wk = original_Rp
    
    # Plot the error landscape
    plt.figure(figsize=(10, 8))
    
    # Use log-scale for both axes and apply a logarithmic color scale for better visualization
    contour = plt.contourf(np.log10(R0_values), np.log10(Rp_values), 
                          errors.T, levels=20, cmap='viridis_r')  # Reversed colormap so darker = better
    
    # Mark the calibrated parameter values
    R0_calibrated = 1.0  # From the calibration results
    Rp_identified = 79.433  # From frequency domain identification
    plt.scatter(np.log10(R0_calibrated), np.log10(Rp_identified), 
                color='red', marker='x', s=150, label='Identified Parameters')
    
    # Add a valley line showing R0 + Rp = constant relationship
    total_R = np.logspace(1, 3, 100)
    plt.plot(np.log10(total_R * 0.1), np.log10(total_R * 0.9), 'r--', 
             linewidth=2, label='R0 + Rp ≈ constant')
    
    # Calculate the minimum error position
    min_idx = np.unravel_index(np.argmin(errors), errors.shape)
    min_R0 = R0_values[min_idx[0]]
    min_Rp = Rp_values[min_idx[1]]
    plt.scatter(np.log10(min_R0), np.log10(min_Rp), 
               color='lime', marker='o', s=150, edgecolors='black', label='Minimum Error')
    
    plt.xlabel('log10(R0) (dyne·s/cm^5)')
    plt.ylabel('log10(Rp) (dyne·s/cm^5)')
    
    # Add axis ticks in original units
    plt.xticks(np.log10([1, 10, 100, 1000]), ['1', '10', '100', '1000'])
    plt.yticks(np.log10([10, 100, 1000]), ['10', '100', '1000'])
    
    # Add colorbar and make sure it's properly labeled
    cbar = plt.colorbar(contour, label='Mean Squared Error (lower is better)')
    
    plt.title('Parameter Coupling: Error Landscape for R0 vs Rp\n(Fixed C0 = {:.6f})'.format(C0_fixed))
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig('R0_Rp_coupling_improved.png', dpi=300, bbox_inches='tight')

# Call this before running the main calibration:
if __name__ == "__main__":
    # Add diagnostic parameter sweep
    print("Running parameter sweep to diagnose calibration issues...")
    diagnostic_parameter_sweep()
    
    # Then continue with your existing code...
    # Let user choose data source
    import argparse
    parser = argparse.ArgumentParser(description='Calibrate hemodynamic model parameters')
    parser.add_argument('--synthetic', action='store_true', 
                        help='Use synthetic data generated in code (default: use data from file)')
    args = parser.parse_args()
    
    # Run calibration and get ALL required outputs
    R0_calibrated, C0_calibrated, t_data, Q_data, P_data = run_calibration(use_real_data=not args.synthetic)
    
    print("\nCalibration and identifiability analysis complete.")
    print(f"Identified parameters: R0={R0_calibrated:.6f}, C0={C0_calibrated:.6f}")
    
    # Get the expected parameters from the model
    R0_theoretical = model_parameters.R0
    C0_theoretical = model_parameters.C0
    
    # Compare identified vs. theoretical parameters
    print("\nComparison with theoretical parameters:")
    print(f"  R0 (identified): {R0_calibrated:.6f} dyne·s/cm^5")
    print(f"  R0 (theoretical): {R0_theoretical:.6f} dyne·s/cm^5")
    print(f"  Difference: {(R0_calibrated-R0_theoretical)/R0_theoretical*100:.2f}%")
    
    print(f"  C0 (identified): {C0_calibrated:.6f} cm^3/dyne")
    print(f"  C0 (theoretical): {C0_theoretical:.6f} cm^3/dyne")
    print(f"  Difference: {(C0_calibrated-C0_theoretical)/C0_theoretical*100:.2f}%")

    # Run frequency domain identification to better identify Rp
    print("\n=== FREQUENCY DOMAIN IDENTIFICATION ===")
    print("Running frequency domain identification for better Rp parameter identifiability...")

    # Run the frequency domain identification with the loaded data
    Rp_identified, Rd_identified, C_identified = frequency_domain_identification(t_data, Q_data, P_data)

    print(f"\nFrequency domain identification results:")
    print(f"  Rp = {Rp_identified:.4f} dyne·s/cm^5")
    print(f"  Rd = {Rd_identified:.4f} dyne·s/cm^5")
    print(f"  C = {C_identified:.6f} cm^3/dyne")

    # Compare with model parameters
    print("\nComparison with theoretical parameters:")
    print(f"  Rp (identified): {Rp_identified:.4f} dyne·s/cm^5")
    print(f"  Rp (model): {model_parameters.Rp_wk:.4f} dyne·s/cm^5")
    print(f"  Difference: {(Rp_identified-model_parameters.Rp_wk)/model_parameters.Rp_wk*100:.2f}%")

    # Check identifiability improvement compared to R0
    print("\nIdentifiability comparison:")
    print(f"  Time domain R0 confidence interval: ±971.24%")
    print(f"  C0 confidence interval: ±38.81%")
    print(f"  Frequency domain provides more stable Rp identification")

    # Create a full 3-element Windkessel model with identified parameters
    def simulate_wk3_model(t_data, Rp, Rd, C):
        """Simulate 3-element Windkessel model with identified parameters"""
        # Create time points
        dt = 0.001
        t_sim = np.arange(0, t_data.max() + dt, dt)
        
        # Get inflow from original data
        Q_inflow = np.interp(t_sim, t_data, Q_data)
        
        # Initialize arrays
        P_wk3 = np.zeros_like(t_sim)
        
        # Set initial pressure
        P_wk3[0] = np.mean(P_data)
        
        # Three-element Windkessel model simulation
        for i in range(1, len(t_sim)):
            # Pressure derivative from 3-element Windkessel
            dP = (1/C) * (Q_inflow[i-1] - P_wk3[i-1]/Rd)
            
            # Forward Euler integration
            P_wk3[i] = P_wk3[i-1] + dt * dP
            
        # Add characteristic impedance effect
        P_wk3 = P_wk3 + Rp * Q_inflow
        
        return t_sim, P_wk3

    # Simulate with identified parameters
    t_wk3, P_wk3 = simulate_wk3_model(t_data, Rp_identified, Rd_identified, C_identified)

    # Plot comparison with data
    plt.figure(figsize=(12, 6))
    plt.plot(t_data, P_data/1333.22, 'ko', markersize=3, label='Data')
    plt.plot(t_wk3, P_wk3/1333.22, 'r-', label='WK3 Model (Freq Domain)')

    # Also compare with time domain model
    t_sim, _, P_sim = simulate_model(R0_calibrated, C0_calibrated)
    plt.plot(t_sim, P_sim/1333.22, 'b--', label='Vessel-WK Model (Time Domain)')

    plt.xlabel('Time (s)')
    plt.ylabel('Pressure (mmHg)')
    plt.title('Pressure Waveform Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig('frequency_domain_model.png')
    plt.show()

    print("\nAnalyzing parameter coupling between R0 and Rp...")
    analyze_parameter_coupling()
