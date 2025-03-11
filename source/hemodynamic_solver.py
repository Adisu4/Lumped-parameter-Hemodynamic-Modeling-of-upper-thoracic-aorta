import numpy as np
from source import model_parameters  # Fixed import syntax

def simulate() -> dict:
    dt = model_parameters.dt
    T = model_parameters.T
    num_periods = model_parameters.num_periods
    t_end = num_periods * T
    time = np.arange(0, t_end + dt, dt)
    N = len(time)
    
    # Initialize state arrays
    V = np.zeros(N)         # Vessel volume (mL)
    Q = np.zeros(N)         # Flow rate (mL/s)
    P_wk = np.zeros(N)      # Windkessel pressure (dyne/cm^2)
    P_vessel = np.zeros(N)  # Vessel pressure (dyne/cm^2)
    Q_in = np.zeros(N)      # Inflow (mL/s)
    
    # Initial conditions
    V[0] = model_parameters.V_init
    Q[0] = model_parameters.Q_init
    P_wk[0] = model_parameters.P_wk_init
    
    # Forward Euler integration
    for i in range(N - 1):
        t = time[i]
        Q_in[i] = model_parameters.inflow(t, T)
        P_vessel[i] = model_parameters.P0 + (V[i] - model_parameters.V0) / model_parameters.C0
        P_out = P_wk[i] + model_parameters.Rp_wk * Q[i]
        dVdt = Q_in[i] - Q[i]
        dQdt = (P_vessel[i] - model_parameters.R0 * Q[i] - P_out) / model_parameters.L0
        Q_wk = (P_wk[i] - model_parameters.Pven) / model_parameters.Rd_wk
        dP_wk_dt = (Q[i] - Q_wk) / model_parameters.Cwk
        
        V[i + 1] = V[i] + dt * dVdt
        Q[i + 1] = Q[i] + dt * dQdt
        P_wk[i + 1] = P_wk[i] + dt * dP_wk_dt
    
    # Final vessel pressure calculation
    P_vessel[-1] = model_parameters.P0 + (V[-1] - model_parameters.V0) / model_parameters.C0
    
    return {
        "time": time,
        "V": V,
        "Q": Q,
        "P_wk": P_wk,
        "P_vessel": P_vessel,
        "Q_in": Q_in
    }
