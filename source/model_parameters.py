import numpy as np
from source.inflow_condition import inflow

# Upper thoracic aorta parameters (Table 1)
l = 24.137      # Vessel length in cm
r0 = 1.2        # Reference vessel radius in cm
h0 = 0.12       # Wall thickness in cm
rho = 1.06      # Blood density in g/cm^3
mu = 0.040      # Blood viscosity in g/cm/s
E = 4e6         # Elastic modulus in dyne/cm^2

# Pressure and Windkessel properties
P0 = 1e5        # Reference pressure in dyne/cm^2
Pven = 0        # Distal venous pressure in dyne/cm^2
Cwk = 1.0163e-3 # Windkessel compliance in cm^5/dyne
Rp_wk = 1.1752e2   # Windkessel proximal resistance in dyne·s/cm^5
Rd_wk = 1.1167e3   # Windkessel distal resistance in dyne·s/cm^5

# Cardiac period and initial conditions
T = 0.955       # Cardiac period in seconds
V0 = np.pi * r0**2 * l  # Reference vessel volume (mL)
V_init = V0     # Initial vessel volume
Q_init = 0.0    # Initial flow rate (mL/s)
P_wk_init = P0  # Initial Windkessel pressure (dyne/cm^2)

# Computed model parameters (Equation 5)
R0 = 8 * mu * l / (np.pi * r0**4)      # Resistance (dyne·s/cm^5)
L0 = rho * l / (np.pi * r0**2)           # Inertance (g/cm)
C0 = (3/2) * (np.pi * r0**3 * l) / (E * h0)  # Wall compliance (cm^3/dyne)

# Time discretization
dt = 0.001       # Time step (s)
num_periods = 5  # Number of cardiac cycles for simulation
