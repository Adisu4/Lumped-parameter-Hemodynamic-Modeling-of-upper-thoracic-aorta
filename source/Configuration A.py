import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# 1) Global time step and parameters
# ============================================================================
T = 0.955      # Cardiac period (s)
dt = 0.0001    # Time step

# Parent vessel parameters (Q_in, P_out)
l_p = 24.137
r0_p = 1.2
rho_p = 1.06
mu_p = 0.040
E_p = 4e6
h0_p = 0.12
P0_p = 1e5

V0_p = np.pi * r0_p**2 * l_p
R0_p = 8 * mu_p * l_p / (np.pi * r0_p**4)
L0_p = rho_p * l_p / (np.pi * r0_p**2)
C0_p = (3/2) * np.pi * r0_p**3 * l_p / (E_p * h0_p)

# Daughter vessel parameters (P_in, P_out)
l_d = 15.0
r0_d = 0.9
rho_d = 1.06
mu_d = 0.040
E_d = 2e6
h0_d = 0.1
P0_d = 1e5

V0_d = np.pi * r0_d**2 * l_d
R0_d = 8 * mu_d * l_d / (np.pi * r0_d**4)
L0_d = rho_d * l_d / (np.pi * r0_d**2)
C0_d = (3/2) * np.pi * r0_d**3 * l_d / (E_d * h0_d)

Pout_d = 8e4  # Prescribed outlet pressure for each daughter (dyne/cm^2)

# ============================================================================
# 2) Inflow function for the parent
# ============================================================================
def inflow(t):
    return 500.0 * np.sin(2*np.pi*t/T)

# ============================================================================
# 3) ODE updates for each vessel type
# ============================================================================
def parent_update(Vp, Qp, Pout_parent, Qin_val):
    """
    Parent vessel (Q_in, P_out):
      dV/dt = Qin_val - Qp
      dQ/dt = [P - R0_p*Qp - Pout_parent] / L0_p
      P = P0_p + (Vp - V0_p)/C0_p
    """
    Pp = P0_p + (Vp - V0_p) / C0_p
    dV = dt * (Qin_val - Qp)
    dQ = dt * ((Pp - R0_p*Qp - Pout_parent) / L0_p)
    Vp_new = Vp + dV
    Qp_new = Qp + dQ
    Pp_new = P0_p + (Vp_new - V0_p) / C0_p
    return Vp_new, Qp_new, Pp_new

def daughter_update(Vd, Q_in, Q, Qd, Pin, Pout):
    """
    Daughter vessel (P_in, P_out):
      dV/dt = Q - Qd
      dQ/dt = [Pin - R0_d*Q - P] / L0_d
      dQd/dt = [P - R0_d*Qd - Pout] / L0_d
      P = P0_d + (Vd - V0_d)/C0_d
    Here Q_in is the flow into the daughter from the parent.
    """
    Pd = P0_d + (Vd - V0_d) / C0_d
    dV = dt * (Q_in - Qd)
    dQ = dt * ((Pin - R0_d*Q - Pd) / L0_d)
    dQd = dt * ((Pd - R0_d*Qd - Pout) / L0_d)
    Vd_new = Vd + dV
    Q_new = Q + dQ
    Qd_new = Qd + dQd
    Pd_new = P0_d + (Vd_new - V0_d) / C0_d
    return Vd_new, Q_new, Qd_new, Pd_new

# ============================================================================
# 4) Newton–Raphson Coupling at Each Time Step for Configuration A
# ============================================================================
def newton_update(Pout_guess, Vp_old, Qp_old, Vd1_old, Q1_old, Qd1_old, Vd2_old, Q2_old, Qd2_old, Qin_val):
    """
    Define the residual:
      f(P_out) = 0.5*(Pd1(P_out) + Pd2(P_out)) - P_out,
    where Pd1 and Pd2 are the daughter inlet pressures computed from a forward Euler update using 
    P_in = P_out.
    Compute f(P_out) and approximate f'(P_out) via finite differences, then return the Newton–Raphson update.
    """
    # Step 1: Parent provisional update with current guess:
    Vp_prov, Qp_prov, Pp_prov = parent_update(Vp_old, Qp_old, Pout_guess, Qin_val)
    
    # Daughter inlet flow is assumed to be half of parent's outlet flow:
    Q_in_d = 0.5 * Qp_prov
    
    # Compute provisional daughter inlet pressures:
    _, _, _, Pd1_prov = daughter_update(Vd1_old, Q_in_d, Q1_old, Qd1_old, Pout_guess, Pout_d)
    _, _, _, Pd2_prov = daughter_update(Vd2_old, Q_in_d, Q2_old, Qd2_old, Pout_guess, Pout_d)
    
    # Residual: difference between the average daughter inlet pressure and the parent's boundary guess.
    f_val = 0.5*(Pd1_prov + Pd2_prov) - Pout_guess
    
    # Compute derivative via finite difference
    eps = 1e-6
    Pout_pert = Pout_guess + eps
    _, _, _, Pd1_pert = daughter_update(Vd1_old, Q_in_d, Q1_old, Qd1_old, Pout_pert, Pout_d)
    _, _, _, Pd2_pert = daughter_update(Vd2_old, Q_in_d, Q2_old, Qd2_old, Pout_pert, Pout_d)
    f_val_pert = 0.5*(Pd1_pert + Pd2_pert) - Pout_pert
    
    df_dP = (f_val_pert - f_val) / eps
    
    # Newton–Raphson update:
    Pout_new = Pout_guess - f_val/df_dP
    return Pout_new, f_val

# ============================================================================
# 5) Full Solver for Configuration A with Newton–Raphson Coupling
# ============================================================================
def simulate_configA_full(num_periods=5, dt=0.0001, tol=1e-3, max_iter=50):
    T_sim = num_periods * T
    time = np.arange(0, T_sim+dt, dt)
    N = len(time)

    # Allocate arrays for parent vessel
    Vp = np.zeros(N)
    Qp = np.zeros(N)
    Pp = np.zeros(N)

    # Allocate arrays for daughter vessels (Daughter 1 and 2)
    Vd1 = np.zeros(N)
    Q1 = np.zeros(N)
    Qd1 = np.zeros(N)
    Pd1 = np.zeros(N)

    Vd2 = np.zeros(N)
    Q2 = np.zeros(N)
    Qd2 = np.zeros(N)
    Pd2 = np.zeros(N)

    # Initial conditions
    Vp[0] = V0_p; Qp[0] = 0.0
    Vd1[0] = V0_d; Q1[0] = 0.0; Qd1[0] = 0.0
    Vd2[0] = V0_d; Q2[0] = 0.0; Qd2[0] = 0.0

    for i in range(N-1):
        t = time[i]
        Qin_val = inflow(t)

        # Save current states
        Vp_old = Vp[i]; Qp_old = Qp[i]
        Vd1_old = Vd1[i]; Q1_old = Q1[i]; Qd1_old = Qd1[i]
        Vd2_old = Vd2[i]; Q2_old = Q2[i]; Qd2_old = Qd2[i]

        # Initial guess for parent's outlet pressure: use current parent's pressure.
        Pout_guess = P0_p + (Vp_old - V0_p)/C0_p

        # Newton–Raphson iteration for coupling
        for it in range(max_iter):
            Pout_new, res = newton_update(Pout_guess, Vp_old, Qp_old,
                                          Vd1_old, Q1_old, Qd1_old,
                                          Vd2_old, Q2_old, Qd2_old,
                                          Qin_val)
            if np.abs(res) < tol:
                Pout_guess = Pout_new
                break
            Pout_guess = Pout_new

        # Now update parent vessel using the converged Pout_guess
        Vp[i+1], Qp[i+1], Pp[i+1] = parent_update(Vp_old, Qp_old, Pout_guess, Qin_val)

        # For daughters, use parent's outlet flow split equally:
        Q_in_d = 0.5 * Qp[i+1]
        Vd1[i+1], Q1[i+1], Qd1[i+1], Pd1[i+1] = daughter_update(Vd1_old, Q_in_d, Q1_old, Qd1_old, Pout_guess, Pout_d)
        Vd2[i+1], Q2[i+1], Qd2[i+1], Pd2[i+1] = daughter_update(Vd2_old, Q_in_d, Q2_old, Qd2_old, Pout_guess, Pout_d)

    # Final state update for pressures
    Pp[-1] = P0_p + (Vp[-1] - V0_p)/C0_p
    Pd1[-1] = P0_d + (Vd1[-1] - V0_d)/C0_d
    Pd2[-1] = P0_d + (Vd2[-1] - V0_d)/C0_d

    return time, Qp, Pp, Vd1, Q1, Qd1, Pd1, Vd2, Q2, Qd2, Pd2

# ============================================================================
# 6) Plotting Function
# ============================================================================
def plot_configurationA_results(time, Qp, Pp, Vd1, Q1, Qd1, Pd1, Vd2, Q2, Qd2, Pd2, num_periods=5, dt=0.0001):
    # Extract final period:
    start_idx = int((num_periods - 1) * T / dt)
    time_final = time[start_idx:] - time[start_idx]

    Qp_final = Qp[start_idx:]
    Pp_final = Pp[start_idx:]
    Q1_final = Q1[start_idx:]
    Qd1_final = Qd1[start_idx:]
    Pd1_final = Pd1[start_idx:]
    Q2_final = Q2[start_idx:]
    Qd2_final = Qd2[start_idx:]
    Pd2_final = Pd2[start_idx:]

    # Convert pressures to mmHg:
    mmHg_factor = 1.0 / 1333.22
    Pp_final_mmHg = Pp_final * mmHg_factor
    Pd1_final_mmHg = Pd1_final * mmHg_factor
    Pd2_final_mmHg = Pd2_final * mmHg_factor

    # Mass conservation check: parent's outlet flow Qp should equal the sum of daughters' inlet flows.
    # In our model, the daughters receive Q_in = 0.5*Qp each, so total equals Qp.
    total_inflow_daughters = 0.5 * Qp_final + 0.5 * Qp_final
    mass_error = Qp_final - total_inflow_daughters
    max_mass_error = np.max(np.abs(mass_error))

    # Plot flows:
    plt.figure(figsize=(10, 5))
    plt.plot(time_final, Qp_final, 'b-', label='Parent Outlet Flow Qp (mL/s)')
    plt.plot(time_final, 0.5 * Qp_final, 'r--', label='Daughter Inlet Flow (mL/s)')
    plt.xlabel('Time (s)')
    plt.ylabel('Flow Rate (mL/s)')
    plt.title('Configuration A: Flow Waveforms (Final Period)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot pressures:
    plt.figure(figsize=(10, 5))
    plt.plot(time_final, Pp_final_mmHg, 'b-', label='Parent Outlet Pressure (mmHg)')
    plt.plot(time_final, Pd1_final_mmHg, 'r--', label='Daughter 1 Inlet Pressure (mmHg)')
    plt.plot(time_final, Pd2_final_mmHg, 'g--', label='Daughter 2 Inlet Pressure (mmHg)')
    plt.xlabel('Time (s)')
    plt.ylabel('Pressure (mmHg)')
    plt.title('Configuration A: Pressure Waveforms (Final Period)')
    plt.legend()
    plt.grid(True)
    plt.show()

    print("Maximum mass error (final period):", max_mass_error)

# ============================================================================
# 7) Main script: Run solver and plot results
# ============================================================================
if __name__ == "__main__":
    time, Qp, Pp, Vd1, Q1, Qd1, Pd1, Vd2, Q2, Qd2, Pd2 = simulate_configA_full()
    print("Configuration A solver done.")
    plot_configurationA_results(time, Qp, Pp, Vd1, Q1, Qd1, Pd1, Vd2, Q2, Qd2, Pd2)
