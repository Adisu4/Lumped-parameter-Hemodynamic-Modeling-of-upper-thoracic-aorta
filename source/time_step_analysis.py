import numpy as np
import matplotlib.pyplot as plt
import time
from source import model_parameters, hemodynamic_solver
import os

def analyze_time_steps():
    """Analyze the effect of different time steps on the simulation results"""
    
    # Store the original dt value to restore it later
    original_dt = model_parameters.dt
    
    # Using only the specified time steps
    dt_values = [0.01, 0.006866, 0.0005]
    
    # Store results for each dt
    results = {}
    computation_times = {}
    
    # For normalization and comparison
    final_period_indices = {}
    
    print("Running simulations with different time steps...")
    
    for dt in dt_values:
        # Update the model parameter
        model_parameters.dt = dt
        
        # Time the simulation
        start_time = time.time()
        
        # Run the simulation
        data = hemodynamic_solver.simulate()
        
        # Record computation time
        computation_times[dt] = time.time() - start_time
        
        # Calculate starting index for the final period
        T = model_parameters.T
        num_periods = model_parameters.num_periods
        start_idx = int((num_periods - 1) * T / dt)
        final_period_indices[dt] = start_idx
        
        # Store relevant results
        results[dt] = {
            "time": data["time"],
            "P_vessel": data["P_vessel"] / 1333.22,  # Convert to mmHg
            "Q": data["Q"]
        }
        
        print(f"  Completed dt={dt:.5f}s in {computation_times[dt]:.2f} seconds")
    
    # Restore original dt
    model_parameters.dt = original_dt
    
    # Plot pressure results with distinctive colors
    plt.figure(figsize=(12, 10))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    # Plot Pressure
    plt.subplot(2, 1, 1)
    for i, dt in enumerate(dt_values):
        start_idx = final_period_indices[dt]
        time_data = results[dt]["time"][start_idx:]
        # Normalize time to show the same period for all dt values
        normalized_time = time_data - time_data[0]
        pressure_data = results[dt]["P_vessel"][start_idx:]
        plt.plot(normalized_time, pressure_data, color=colors[i], linewidth=2, label=f'dt={dt:.5f}s')
    
    plt.xlabel('Time (s)')
    plt.ylabel('Pressure (mmHg)')
    plt.title('Effect of Time Step Size on Pressure Waveform (Final Period)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Plot Flow
    plt.subplot(2, 1, 2)
    for i, dt in enumerate(dt_values):
        start_idx = final_period_indices[dt]
        time_data = results[dt]["time"][start_idx:]
        # Normalize time to show the same period for all dt values
        normalized_time = time_data - time_data[0]
        flow_data = results[dt]["Q"][start_idx:]
        plt.plot(normalized_time, flow_data, color=colors[i], linewidth=2, label=f'dt={dt:.5f}s')
    
    plt.xlabel('Time (s)')
    plt.ylabel('Flow Rate (mL/s)')
    plt.title('Effect of Time Step Size on Flow Waveform (Final Period)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.tight_layout()
    plot_dir = "plots"
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, 'time_step_analysis.png')
    if os.path.exists(plot_path):
       os.remove(plot_path)

    # Save inside `plots/` folder
    plt.savefig(plot_path, dpi=300)
    plt.show()
    
    # Updated computational efficiency plot
    plt.figure(figsize=(10, 6))
    dt_list = list(dt_values)
    comp_times = [computation_times[dt] for dt in dt_list]
    
    bar_width = 0.5  # Wider bars for better visibility
    bars = plt.bar(range(len(dt_list)), comp_times, width=bar_width, 
        color=colors[:len(dt_list)], edgecolor='black', alpha=0.85)
        
    # Add value labels on top of each bar
    for bar, time_val in zip(bars, comp_times):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{time_val:.2f}s', ha='center', fontweight='bold')

    plt.xticks(range(len(dt_list)), [f"{dt:.5f}" for dt in dt_list])
    plt.xlabel('Time Step Size (s)')
    plt.ylabel('Computation Time (s)')
    plt.title('Computational Efficiency vs Time Step Size')
    plt.grid(True, linestyle='--', axis='y', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'time_step_computation.png'), dpi=300)
    plt.show()
    
    # Calculate error relative to the smallest dt
    smallest_dt = min(dt_values)
    reference_pressure = results[smallest_dt]["P_vessel"][final_period_indices[smallest_dt]:]
    reference_time = results[smallest_dt]["time"][final_period_indices[smallest_dt]:]
    
    # Calculate RMS error for the other time steps
    rms_errors = {}
    for dt in dt_values:
        if dt == smallest_dt:
            rms_errors[dt] = 0.0
            continue
            
        # Interpolate pressure to match reference time points
        current_pressure = results[dt]["P_vessel"][final_period_indices[dt]:]
        current_time = results[dt]["time"][final_period_indices[dt]:]
        
        # Normalize times to start at 0
        norm_ref_time = reference_time - reference_time[0]
        norm_current_time = current_time - current_time[0]
        
        # Only compare over the overlapping time range
        max_time = min(norm_ref_time[-1], norm_current_time[-1])
        mask_ref = norm_ref_time <= max_time
        mask_current = norm_current_time <= max_time
        
        # Interpolate current data to reference time points
        interp_pressure = np.interp(
            norm_ref_time[mask_ref], 
            norm_current_time[mask_current],
            current_pressure[mask_current]
        )
        
        # Calculate RMS error
        rms_error = np.sqrt(np.mean((interp_pressure - reference_pressure[mask_ref])**2))
        rms_errors[dt] = rms_error
    
    # Print analysis with the specified time steps
    print("\nTime Step Analysis Summary:")
    print("---------------------------")
    print(f"{'dt (s)':10} {'Comp Time (s)':15} {'RMS Error':15} {'Notes':30}")
    print("-" * 75)
    
    for dt in dt_values:
        if dt == smallest_dt:
            notes = "Reference (highest resolution)"
        elif dt == 0.01:
            notes = "Fast but lower accuracy"
        elif dt == 0.006866:
            notes = "Model-specific value"
        else:
            notes = "Balanced accuracy and efficiency"
            
        print(f"{dt:<10.5f} {computation_times[dt]:<15.2f} {rms_errors.get(dt, 0):<15.5f} {notes:<30}")
    
    print("\nRecommendation:")
    # Modified recommendation logic based on the three available time steps
    if computation_times[smallest_dt] < 10:  # If computation is fast enough with smallest dt
        recommended_dt = smallest_dt
        reason = "provides highest accuracy with acceptable computation time"
    elif computation_times[0.006866] < 5:
        recommended_dt = 0.006866
        reason = "balances accuracy and efficiency, specific to this model"
    else:
        recommended_dt = 0.01
        reason = "offers fastest computation with acceptable accuracy"
        
    print(f"Recommended time step: dt = {recommended_dt} seconds")
    print(f"Reason: This time step {reason}.")
    
    # Physiological checks remain the same
    print("\nPhysiological Reality Check:")
    
    # Get systolic and diastolic pressures for the recommended dt
    recommended_data = results[recommended_dt]
    start_idx = final_period_indices[recommended_dt]
    pressure_final_period = recommended_data["P_vessel"][start_idx:]
    systolic = np.max(pressure_final_period)
    diastolic = np.min(pressure_final_period)
    
    print(f"Simulated blood pressure: {systolic:.1f}/{diastolic:.1f} mmHg")
    print(f"Normal range: ~120/80 mmHg")
    
    if 90 <= systolic <= 140 and 60 <= diastolic <= 90:
        print("✓ Pressure values are within physiological range")
    else:
        print("⚠ Pressure values may be outside normal physiological range")
    
    flow_final_period = recommended_data["Q"][start_idx:]
    peak_flow = np.max(flow_final_period)
    
    print(f"Peak aortic flow: {peak_flow:.1f} mL/s")
    print(f"Typical peak aortic flow: ~400-500 mL/s")
    
    if 300 <= peak_flow <= 600:
        print("✓ Flow values are within physiological range")
    else:
        print("⚠ Flow values may be outside normal physiological range")
        
    return recommended_dt