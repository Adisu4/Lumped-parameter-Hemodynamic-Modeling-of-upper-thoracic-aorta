import matplotlib.pyplot as plt
import numpy as np
from source import model_parameters


def plot_results(simulation_data: dict) -> None:
    time = simulation_data["time"]
    Q = simulation_data["Q"]
    P_vessel = simulation_data["P_vessel"]
    
    # Convert pressure to mmHg (1 mmHg = 1333.22 dyne/cm^2)
    P_vessel_mmHg = P_vessel / 1333.22
    dt = model_parameters.dt  
    start_idx = int((model_parameters.num_periods - 1) * model_parameters.T / dt)
    end_idx = len(time)
    
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(time[start_idx:end_idx], Q[start_idx:end_idx])
    plt.xlabel('Time (s)')
    plt.ylabel('Flow Rate (mL/s)')
    plt.title('Flow Rate in Vessel (Final Period)')
    
    plt.subplot(2, 1, 2)
    plt.plot(time[start_idx:end_idx], P_vessel_mmHg[start_idx:end_idx])
    plt.xlabel('Time (s)')
    plt.ylabel('Pressure (mmHg)')
    plt.title('Vessel Pressure (Final Period)')
    plt.tight_layout()
    plt.show()

def sensitivity_analysis(simulation_data: dict) -> None:
    dt = model_parameters.dt  # Fixed variable name
    T = model_parameters.T
    num_periods = model_parameters.num_periods  # Fixed variable name
    time = simulation_data["time"]
    N = len(time)
    start_idx = int((num_periods - 1) * T / dt)
    end_idx = N
    
    C0_factors = [0.25, 0.5, 1.0, 2.0, 4.0]
    plt.figure(figsize=(10, 6))
    
    for factor in C0_factors:
        modified_C0 = model_parameters.C0 * factor
        V_temp = np.zeros(N)  # Using numpy arrays for better performance
        Q_temp = np.zeros(N)
        P_wk_temp = np.zeros(N)
        P_vessel_temp = np.zeros(N)
        
        V_temp[0] = model_parameters.V_init
        Q_temp[0] = model_parameters.Q_init
        P_wk_temp[0] = model_parameters.P_wk_init
        
        for i in range(N - 1):
            t = time[i]
            current_inflow = model_parameters.inflow(t, T)
            P_vessel_temp[i] = model_parameters.P0 + (V_temp[i] - model_parameters.V0) / modified_C0
            P_out = P_wk_temp[i] + model_parameters.Rp_wk * Q_temp[i]
            dVdt = current_inflow - Q_temp[i]
            dQdt = (P_vessel_temp[i] - model_parameters.R0 * Q_temp[i] - P_out) / model_parameters.L0  # Fixed variable name
            Q_wk_temp = (P_wk_temp[i] - model_parameters.Pven) / model_parameters.Rd_wk
            dP_wk_dt = (Q_temp[i] - Q_wk_temp) / model_parameters.Cwk
            
            V_temp[i + 1] = V_temp[i] + dt * dVdt
            Q_temp[i + 1] = Q_temp[i] + dt * dQdt
            P_wk_temp[i + 1] = P_wk_temp[i] + dt * dP_wk_dt
        
        P_vessel_temp[-1] = model_parameters.P0 + (V_temp[-1] - model_parameters.V0) / modified_C0  # Fixed variable name
        P_vessel_mmHg_temp = P_vessel_temp / 1333.22  # Vectorized conversion
        plt.plot(time[start_idx:end_idx], P_vessel_mmHg_temp[start_idx:end_idx],
                 label=f'C0 factor {factor}')
    
    plt.xlabel('Time (s)')
    plt.ylabel('Pressure (mmHg)')
    plt.title('Sensitivity Analysis: Vessel Pressure for Varying C0')
    plt.legend()
    plt.show()


