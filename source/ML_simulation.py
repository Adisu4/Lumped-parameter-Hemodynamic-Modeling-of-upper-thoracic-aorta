import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from scipy.interpolate import interp1d
from source import model_parameters, hemodynamic_solver
import os
from pathlib import Path

def get_writable_path(filename):
    """Get a path where we can write files"""
    import tempfile
    paths = [
        os.path.join('.', filename),
        os.path.join(str(Path.home()), filename),
        os.path.join(tempfile.gettempdir(), filename)
    ]
    
    for path in paths:
        try:
            directory = os.path.dirname(path) or '.'
            if not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
            with open(path, 'a') as test:
                pass
            return path
        except (IOError, PermissionError):
            continue
    return None

def analyze_time_steps():
    """Analyze the effect of different time steps using data-driven approach"""
    
    # Store the original dt value
    original_dt = model_parameters.dt
    
    # Generate more dt values - 50 points
    dt_values = np.logspace(-4, -1.5, num=50)  # Broader range, more points
    
    # Compute a reference solution with the smallest dt
    smallest_dt = min(dt_values)
    model_parameters.dt = smallest_dt
    print(f"\n📊 Computing reference solution with dt={smallest_dt:.6f}s...")
    reference_data = hemodynamic_solver.simulate()
    
    # Extract reference data for the final period
    T = model_parameters.T
    num_periods = model_parameters.num_periods
    start_idx_ref = int((num_periods-1) * T / smallest_dt)
    ref_time = reference_data["time"][start_idx_ref:]
    ref_pressure = reference_data["P_vessel"][start_idx_ref:]
    ref_flow = reference_data["Q"][start_idx_ref:]
    
    # Create a uniform time grid for comparison
    uniform_time = np.linspace(ref_time[0], ref_time[-1], 1000)
    
    # Interpolate reference data to uniform grid
    ref_pressure_interp = interp1d(ref_time, ref_pressure, kind='cubic')
    ref_flow_interp = interp1d(ref_time, ref_flow, kind='cubic')
    
    ref_pressure_uniform = ref_pressure_interp(uniform_time)
    ref_flow_uniform = ref_flow_interp(uniform_time)
    
    # Local dataset
    data_records = []
    
    print("\n🔄 Running simulations with different time steps...\n")
    
    for dt in dt_values[1:]:  # Skip the smallest dt as we already computed it
        model_parameters.dt = dt
        start_time = time.time()
        
        # Run the simulation
        data = hemodynamic_solver.simulate()
        
        computation_time = time.time() - start_time
        
        # Calculate error metrics compared to reference solution
        start_idx = int((num_periods-1) * T / dt)
        sim_time = data["time"][start_idx:]
        sim_pressure = data["P_vessel"][start_idx:]
        sim_flow = data["Q"][start_idx:]
        
        # Interpolate simulation data to uniform grid
        try:
            sim_pressure_interp = interp1d(sim_time, sim_pressure, kind='cubic', bounds_error=False, fill_value="extrapolate")
            sim_flow_interp = interp1d(sim_time, sim_flow, kind='cubic', bounds_error=False, fill_value="extrapolate")
            
            sim_pressure_uniform = sim_pressure_interp(uniform_time)
            sim_flow_uniform = sim_flow_interp(uniform_time)
            
            # Calculate error metrics
            pressure_rmse = np.sqrt(np.mean((sim_pressure_uniform - ref_pressure_uniform)**2))
            flow_rmse = np.sqrt(np.mean((sim_flow_uniform - ref_flow_uniform)**2))
            pressure_relative_error = np.mean(np.abs(sim_pressure_uniform - ref_pressure_uniform)) / np.mean(np.abs(ref_pressure_uniform))
            flow_relative_error = np.mean(np.abs(sim_flow_uniform - ref_flow_uniform)) / np.mean(np.abs(ref_flow_uniform))
            
            # Combined metric: balance accuracy and speed
            # Lower is better
            combined_error = 0.7 * (pressure_relative_error + flow_relative_error) + 0.3 * (computation_time / max(0.1, computation_time))
            
            # Store data
            data_records.append([
                dt,
                computation_time, 
                pressure_rmse,
                flow_rmse,
                pressure_relative_error,
                flow_relative_error,
                combined_error
            ])
            
            print(f" ✅ dt={dt:.5f}s: time={computation_time:.2f}s, pressure error={pressure_relative_error:.4f}, flow error={flow_relative_error:.4f}")
        except:
            print(f" ❌ dt={dt:.5f}s: Failed to interpolate data, skipping")
    
    # Restore original dt
    model_parameters.dt = original_dt
    
    # Convert to DataFrame
    columns = ["dt", "computation_time", "pressure_rmse", "flow_rmse", "pressure_relative_error", "flow_relative_error", "combined_error"]
    df = pd.DataFrame(data_records, columns=columns)
    
    # Plot error vs dt
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.loglog(df["dt"], df["pressure_relative_error"], 'o-')
    plt.xlabel("Time Step Size (dt)")
    plt.ylabel("Pressure Relative Error")
    plt.title("Pressure Error vs Time Step")
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.loglog(df["dt"], df["flow_relative_error"], 'o-')
    plt.xlabel("Time Step Size (dt)")
    plt.ylabel("Flow Relative Error")
    plt.title("Flow Error vs Time Step")
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    plt.semilogx(df["dt"], df["computation_time"], 'o-')
    plt.xlabel("Time Step Size (dt)")
    plt.ylabel("Computation Time (s)")
    plt.title("Computation Time vs Time Step")
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    plt.loglog(df["dt"], df["combined_error"], 'o-')
    plt.xlabel("Time Step Size (dt)")
    plt.ylabel("Combined Error Metric")
    plt.title("Combined Error vs Time Step")
    plt.grid(True)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = get_writable_path("dt_analysis.png")
    if plot_path:
        try:
            plt.savefig(plot_path, dpi=300)
            print(f"\n📊 Plot saved as {plot_path}")
        except Exception as e:
            print(f"❌ Warning: Could not save plot: {e}")
    
    plt.show()
    
    # Now train a simple ML model to predict the optimal dt
    return train_dt_predictor(df)

def train_dt_predictor(df):
    """Train a simple ML model to predict the optimal time step."""
    
    print("\n🔬 Training ML model to predict optimal dt")
    
    # Find the dt with the lowest combined error
    best_dt_idx = df["combined_error"].idxmin()
    best_dt = df.loc[best_dt_idx, "dt"]
    best_error = df.loc[best_dt_idx, "combined_error"]
    
    print(f"\n🏆 Based on data analysis, the optimal dt is {best_dt:.6f}s")
    print(f"   This has a combined error metric of {best_error:.6f}")
    
    # Use a simple model: Decision Tree
    # We'll train it to predict whether a dt value is "good" (low error)
    df["is_good_dt"] = df["combined_error"] < (best_error * 1.2)  # Within 20% of best error
    
    X = df[["pressure_rmse", "flow_rmse", "computation_time"]]
    y = df["is_good_dt"]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a simple model
    model = DecisionTreeRegressor(max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    
    # Print feature importance
    feature_importance = dict(zip(X.columns, model.feature_importances_))
    print("\n⚙️ Feature Importance:")
    for feature, importance in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
        print(f"   {feature}: {importance:.4f}")
    
    # Return the recommended dt directly
    return best_dt

def run_optimal_simulation():
    """Run simulation with data-driven optimal dt."""
    print("\n🚀 Starting time step analysis...")
    optimal_dt = analyze_time_steps()
    
    print(f"\n🔮 Using optimal dt = {optimal_dt:.6f}s for final simulation")
    
    # Run the final simulation with the selected dt
    model_parameters.dt = optimal_dt
    final_results = hemodynamic_solver.simulate()
    
    return final_results
