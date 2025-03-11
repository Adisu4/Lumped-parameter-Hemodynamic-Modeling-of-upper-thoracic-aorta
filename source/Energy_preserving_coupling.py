import numpy as np
import matplotlib.pyplot as plt
from source.model_parameters import l, r0, h0, rho, mu, E, P0, Pven, Cwk, Rp_wk, Rd_wk, T
from source.inflow_condition import inflow

# For each segment, the effective length is half the total
l_seg = l / 2
V0_seg = np.pi * r0**2 * l_seg  # Reference volume for a segment

# Compute segment parameters using formulas (note: they scale with l)
R0_seg = 8 * mu * l_seg / (np.pi * r0**4)       # Resistance for one segment
L0_seg = rho * l_seg / (np.pi * r0**2)            # Inertance for one segment
C0_seg = (3/2) * (np.pi * r0**3 * l_seg) / (E * h0) # Compliance for one segment

# --------------------------
# Simulation Function for Split Vessel
# --------------------------
def simulate_model_split(dt=0.006866, T=0.955):
    """
    Simulate the two-segment (split vessel) model with energy-preserving coupling.
    Each segment uses the same 0D vessel model with parameters computed using l_seg.
    Coupling is enforced at the junction via averaging.
    
    Returns: time array, Q (mL/s) at the interface (common flow), and P (dyne/cm^2)
    measured at the outlet of segment 2.
    """
    N = int(T/dt) + 1
    time_sim = np.linspace(0, T, N)
    
    # Initialize state arrays for segment 1 and 2, and for the Windkessel
    V1 = np.zeros(N); Q1 = np.zeros(N)
    V2 = np.zeros(N); Q2 = np.zeros(N)
    P_wk = np.zeros(N)
    
    # Set initial conditions for each segment and WK
    V1[0] = V0_seg; Q1[0] = 0.0
    V2[0] = V0_seg; Q2[0] = 0.0
    P_wk[0] = P0
    
    # For convenience, define the Windkessel term function:
    def WK_pressure(P_wk_val, Q_val):
        return P_wk_val + Rp_wk * Q_val
    
    for i in range(N-1):
        t = time_sim[i]
        # ------------------
        # Segment 1 update
        # ------------------
        # Inlet flow from the heart
        Q_in_val = inflow(t, T)
        # Compute pressure in segment 1 from its volume
        P1 = P0 + (V1[i] - V0_seg)/C0_seg
        # For segment 1, the outlet (junction) pressure is not yet determined.
        # We update its ODE using its own state:
        dV1 = dt * (Q_in_val - Q1[i])
        V1_star = V1[i] + dV1
        P1_star = P0 + (V1_star - V0_seg)/C0_seg
        dQ1 = dt * ((P1_star - R0_seg * Q1[i] - P1_star)/L0_seg)  # Note: here we assume the junction load equals P1_star
        Q1_star = Q1[i] + dQ1
        
        # ------------------
        # Segment 2 update
        # ------------------
        # The inlet flow for segment 2 will be corrected later;
        # first update its state as if it were decoupled.
        # (We use the previous value of Q2 for its ODE.)
        # For segment 2, the inlet pressure is unknown, but its outlet is coupled with the WK.
        dV2 = dt * (Q1[i] - Q2[i])  # temporarily use Q1[i] as Q_in (will be corrected)
        V2_star = V2[i] + dV2
        P2_star = P0 + (V2_star - V0_seg)/C0_seg
        # For segment 2, the load at the outlet is from the WK:
        P_out = WK_pressure(P_wk[i], Q2[i])
        dQ2 = dt * ((P2_star - R0_seg * Q2[i] - P_out)/L0_seg)
        Q2_star = Q2[i] + dQ2
        
        # ------------------
        # Interface (junction) coupling correction
        # ------------------
        # Enforce mass conservation: set interface flow as the average of Q1 and Q2 updates.
        Q_interface = 0.5*(Q1_star + Q2_star)
        # Enforce energy conservation: set the junction pressure as the average of the two segment pressures.
        P_interface = 0.5*(P1_star + P2_star)
        # Adjust segment 2 state so that its pressure matches the interface value.
        V2_star = V0_seg + C0_seg*(P_interface - P0)
        # Override the updated flows with the interface value.
        Q1_new = Q_interface
        Q2_new = Q_interface
        
        # ------------------
        # Windkessel update (coupled to the outlet of segment 2)
        # ------------------
        Q_wk = (P_wk[i] - Pven) / Rd_wk
        dP_wk = dt * ((Q_interface - Q_wk) / Cwk)
        P_wk_new = P_wk[i] + dP_wk
        
        # Store the updated values for the next time step:
        V1[i+1] = V1_star
        Q1[i+1] = Q1_new
        V2[i+1] = V2_star
        Q2[i+1] = Q2_new
        P_wk[i+1] = P_wk_new
        
    # For the split vessel, we report the interface flow (common to both segments)
    # and define the vessel pressure as the outlet pressure of segment 2:
    P2 = P0 + (V2 - V0_seg)/C0_seg  # Pressure in segment 2 over time
    # For final output, we return the common flow and the pressure at the outlet (after WK load is applied)
    # Here, one might also include the WK-coupled outlet pressure:
    P_out_final = P_wk + Rp_wk * Q2
    return time_sim, Q1, P_out_final

# --------------------------
# Run the Simulation for the Split Vessel Model
# --------------------------
dt = 0.006866
time_split, Q_split, P_out_split = simulate_model_split(dt=dt, T=T)

# Convert pressure to mmHg for plotting (1 mmHg ≈ 1333.22 dyne/cm^2)
P_out_split_mmHg = P_out_split / 1333.22

# --------------------------
# For comparison, run the original (unsplit) model simulation (from our previous code)
# --------------------------
def simulate_model_single(dt=0.006866, T=0.955):
    N = int(T/dt) + 1
    time_sim = np.linspace(0, T, N)
    V = np.zeros(N)
    Q = np.zeros(N)
    P_wk = np.zeros(N)
    P_vessel = np.zeros(N)
    
    V[0] = np.pi * r0**2 * l   # original V0
    Q[0] = 0.0
    P_wk[0] = P0
    
    for i in range(N-1):
        t = time_sim[i]
        Q_in_val = inflow(t, T)
        P_vessel[i] = P0 + (V[i] - (np.pi * r0**2 * l))/((3/2)*(np.pi*r0**3*l)/(E*h0))
        P_out = P_wk[i] + Rp_wk * Q[i]
        dVdt = dt * (Q_in_val - Q[i])
        dQdt = dt * ((P_vessel[i] - R0_seg*2 * Q[i] - P_out)/ ( (rho*l)/(np.pi*r0**2) ))  # note: R0 and L0 for full vessel approximated
        V[i+1] = V[i] + dVdt
        Q[i+1] = Q[i] + dQdt
        Q_wk = (P_wk[i]-Pven)/Rd_wk
        dP_wk = dt*((Q[i]-Q_wk)/Cwk)
        P_wk[i+1] = P_wk[i] + dP_wk
    P_vessel[-1] = P0 + (V[-1] - (np.pi*r0**2*l))/((3/2)*(np.pi*r0**3*l)/(E*h0))
    return time_sim, Q, P_vessel

time_single, Q_single, P_single = simulate_model_single(dt=dt, T=T)
P_single_mmHg = P_single / 1333.22

# --------------------------
# Plot Comparison of Split vs. Single Vessel Model
# --------------------------
plt.figure(figsize=(12, 8))

plt.subplot(2,1,1)
plt.plot(time_single, Q_single, 'b-', label='Single Vessel Q')
plt.plot(time_split, Q_split, 'r--', label='Split Vessel Q')
plt.xlabel('Time (s)')
plt.ylabel('Flow rate Q (mL/s)')
plt.legend()
plt.title('Comparison of Flow Rate: Single vs. Split Vessel Model')

plt.subplot(2,1,2)
plt.plot(time_single, P_single_mmHg, 'b-', label='Single Vessel P')
plt.plot(time_split, P_out_split_mmHg, 'r--', label='Split Vessel P')
plt.xlabel('Time (s)')
plt.ylabel('Pressure P (mmHg)')
plt.legend()
plt.title('Comparison of Pressure at Outlet: Single vs. Split Vessel Model')

plt.tight_layout()
plt.show()

# --------------------------
# Energy Flux Calculation
# --------------------------
# Compute energy flux f_E = ∫_0^T Q(t)*P(t) dt for both models
fE_single = np.trapezoid(Q_single * P_single, time_single)
fE_split = np.trapezoid(Q_split * P_out_split, time_split)
print("Energy flux (single vessel):", fE_single, "dyne·mL/s")
print("Energy flux (split vessel):", fE_split, "dyne·mL/s")
