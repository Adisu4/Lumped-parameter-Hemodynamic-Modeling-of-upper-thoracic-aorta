import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# 1) Global / Vessel Parameters
# =============================================================================
T = 0.955  # Cardiac period
dt = 0.001 # Small time step for stability

# Parent (Qin, Qout)
l_p = 24.137
r0_p = 1.2
h0_p = 0.12
rho_p = 1.06
mu_p = 0.040
E_p = 4e6
P0_p = 1e5

V0_p = np.pi*(r0_p**2)*l_p
R0_p = 8*mu_p*l_p/(np.pi*(r0_p**4))
L0_p = rho_p*l_p/(np.pi*(r0_p**2))
C0_p = (3/2)*np.pi*(r0_p**3)*l_p/(E_p*h0_p)

# Minimal load resistor for parent outlet:
R_out = 150.0  # dyne·s/cm^5

# Daughter (Qin, Pout)
# We'll assume two identical daughters for simplicity
l_d = 20.0
r0_d = 1.0
h0_d = 0.1
rho_d = 1.06
mu_d = 0.040
E_d = 3e6
P0_d = 1e5

V0_d = np.pi*(r0_d**2)*l_d
R0_d = 8*mu_d*l_d/(np.pi*(r0_d**4))
L0_d = rho_d*l_d/(np.pi*(r0_d**2))
C0_d = (3/2)*np.pi*(r0_d**3)*l_d/(E_d*h0_d)

# Prescribed outlet pressure for daughters:
Pout_d = 8e4  # 80,000 dyne/cm^2 ~ 60 mmHg

# =============================================================================
# 2) Inflow Function for the Parent
# =============================================================================
def inflow(t):
    # Example: single sine wave for demonstration
    return 500.0 * np.sin(2.0*np.pi*t/T)

# =============================================================================
# 3) Bifurcation Solver (Configuration B) with Iterative Coupling
# =============================================================================
def simulate_configB_full_iter(num_periods=5, dt=0.001, tol=1e-3, max_iter=50, alpha=0.5):
    """
    Solve parent (Qin, Qout) + 2 daughters (Qin, Pout) with iterative coupling.
    Equations:
      Parent:
        dV/dt = Qin - Qp
        dQ/dt = (Pp - R0_p*Qp - R_out*Qp) / L0_p
        Pp = P0_p + (V - V0_p)/C0_p
      Daughter i:
        dVd/dt = Q_in - Qd
        dQd/dt = (Pd - R0_d*Qd - Pout_d) / L0_d
        Pd = P0_d + (Vd - V0_d)/C0_d
    Coupling:
      Qp -> parent's outlet flow
      Qd1_in + Qd2_in = Qp  (symmetric: Qd1_in = Qd2_in = 0.5*Qp)
      Pressure continuity => parent's outlet pressure = daughters' inlet pressure.
      We do an iterative loop to match these.
    """
    T_sim = num_periods * T
    time = np.arange(0, T_sim+dt, dt)
    N = len(time)
    
    # Parent arrays
    Vp = np.zeros(N)
    Qp = np.zeros(N)
    Pp = np.zeros(N)
    
    # Daughter 1 arrays
    Vd1 = np.zeros(N)
    Qd1 = np.zeros(N)
    Pd1 = np.zeros(N)
    
    # Daughter 2 arrays
    Vd2 = np.zeros(N)
    Qd2 = np.zeros(N)
    Pd2 = np.zeros(N)
    
    # Initial conditions
    Vp[0] = V0_p
    Qp[0] = 0.0
    
    Vd1[0] = V0_d
    Qd1[0] = 0.0
    
    Vd2[0] = V0_d
    Qd2[0] = 0.0
    
    for i in range(N-1):
        t = time[i]
        Qin_val = inflow(t)
        
        # Parent step: ODE for (Qin, Qout)
        #   dV/dt = Qin_val - Qp[i]
        #   dQ/dt = [Pp[i] - R0_p*Qp[i] - R_out*Qp[i]] / L0_p
        Pp[i] = P0_p + (Vp[i] - V0_p)/C0_p
        P_load = R_out * Qp[i]  # outlet load
        Vp_prov = Vp[i] + dt*(Qin_val - Qp[i])
        Qp_prov = Qp[i] + dt*((Pp[i] - R0_p*Qp[i] - P_load)/L0_p)
        
        # Iterative coupling for pressure continuity
        # We'll guess the parent's new pressure from Vp_prov:
        Pp_new = P0_p + (Vp_prov - V0_p)/C0_p
        P_j = Pp_new  # initial guess for junction pressure
        
        # For daughters, we want P_d(inlet) = P_j
        for _ in range(max_iter):
            # Flow to each daughter is half the parent's Qp_prov
            Q_d_in = 0.5 * Qp_prov
            
            # Daughter 1: (Qin, Pout)
            #   dVd1/dt = Q_d_in - Qd1[i]
            #   dQd1/dt = [Pd1 - R0_d*Qd1[i] - Pout_d]/L0_d
            #   Pd1 = P0_d + (Vd1 - V0_d)/C0_d
            # We want the inlet pressure = P_j => Vd1_target = V0_d + C0_d*(P_j - P0_d).
            Vd1_target = V0_d + C0_d*(P_j - P0_d)
            # Update Qd1 by forward Euler:
            dQd1 = dt*((P0_d + (Vd1[i] - V0_d)/C0_d - R0_d*Qd1[i] - Pout_d)/L0_d)
            Qd1_new = Qd1[i] + dQd1
            
            # Daughter 2 (same approach)
            Vd2_target = V0_d + C0_d*(P_j - P0_d)
            dQd2 = dt*((P0_d + (Vd2[i] - V0_d)/C0_d - R0_d*Qd2[i] - Pout_d)/L0_d)
            Qd2_new = Qd2[i] + dQd2
            
            # The daughters' new inlet pressures from these volumes:
            Pd1_new = P0_d + (Vd1_target - V0_d)/C0_d
            Pd2_new = P0_d + (Vd2_target - V0_d)/C0_d
            # Average them for a guess of the new P_j:
            P_j_new = 0.5*(Pd1_new + Pd2_new)
            
            # Apply damping
            old_P_j = P_j
            P_j = alpha*P_j_new + (1-alpha)*old_P_j
            
            if np.abs(P_j_new - old_P_j) < tol:
                break
        
        # Once converged, finalize parent step:
        Vp[i+1] = Vp_prov
        Qp[i+1] = Qp_prov
        Pp[i+1] = P0_p + (Vp[i+1] - V0_p)/C0_p
        
        # Final volumes & flows for daughters:
        Vd1[i+1] = Vd1_target
        Qd1[i+1] = 0.5 * Qp_prov  # from mass conservation
        Pd1[i+1] = P0_d + (Vd1[i+1] - V0_d)/C0_d
        
        Vd2[i+1] = Vd2_target
        Qd2[i+1] = 0.5 * Qp_prov
        Pd2[i+1] = P0_d + (Vd2[i+1] - V0_d)/C0_d
    
    # Final pressures
    Pp[-1] = P0_p + (Vp[-1] - V0_p)/C0_p
    Pd1[-1] = P0_d + (Vd1[-1] - V0_d)/C0_d
    Pd2[-1] = P0_d + (Vd2[-1] - V0_d)/C0_d
    
    return time, Qp, Pp, Qd1, Pd1, Qd2, Pd2

# =============================================================================
# Main script
# =============================================================================
if __name__ == "__main__":
    num_periods = 5
    time, Qp, Pp, Qd1, Pd1, Qd2, Pd2 = simulate_configB_full_iter(num_periods, dt=0.001, tol=1e-3, max_iter=50, alpha=0.5)
    
    # Extract final period
    start_idx = int((num_periods-1)*T/0.001)
    time_final = time[start_idx:] - time[start_idx]
    
    Qp_final = Qp[start_idx:]
    Pp_final = Pp[start_idx:]
    Qd1_final = Qd1[start_idx:]
    Pd1_final = Pd1[start_idx:]
    Qd2_final = Qd2[start_idx:]
    Pd2_final = Pd2[start_idx:]
    
    # Mass check
    mass_error = Qp_final - (Qd1_final + Qd2_final)
    max_mass_error = np.max(np.abs(mass_error))
    
    # Energy flux
    fE_parent = np.trapz(Qp_final * Pp_final, time_final)
    # If we assume the junction pressure ~ parent's outlet pressure for the energy flux,
    # we might do:
    fE_daughters = np.trapz((Qd1_final + Qd2_final)*Pp_final, time_final)
    
    # Convert pressure to mmHg
    mmHg_factor = 1.0/1333.22
    Pp_mmHg = Pp_final*mmHg_factor
    Pd1_mmHg = Pd1_final*mmHg_factor
    Pd2_mmHg = Pd2_final*mmHg_factor
    
    print("Max mass error (final period):", max_mass_error)
    print("Energy flux (parent vessel):", fE_parent, "dyne·mL/s")
    print("Energy flux (daughters combined):", fE_daughters, "dyne·mL/s")
    
    # Plot flows
    plt.figure()
    plt.plot(time_final, Qp_final, label='Parent Q (mL/s)')
    plt.plot(time_final, Qd1_final, '--', label='Daughter1 Q (mL/s)')
    plt.plot(time_final, Qd2_final, '--', label='Daughter2 Q (mL/s)')
    plt.xlabel("Time (s)")
    plt.ylabel("Flow Rate (mL/s)")
    plt.title("Configuration B: Flow (Final Period)")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Plot pressures
    plt.figure()
    plt.plot(time_final, Pp_mmHg, label='Parent P (mmHg)')
    plt.plot(time_final, Pd1_mmHg, '--', label='Daughter1 P (mmHg)')
    plt.plot(time_final, Pd2_mmHg, '--', label='Daughter2 P (mmHg)')
    plt.xlabel("Time (s)")
    plt.ylabel("Pressure (mmHg)")
    plt.title("Configuration B: Pressure (Final Period)")
    plt.legend()
    plt.grid(True)
    plt.show()
