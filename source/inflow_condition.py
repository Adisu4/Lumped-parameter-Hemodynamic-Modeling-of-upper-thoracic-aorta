import numpy as np

def inflow(t: float, T: float = 0.955) -> float:
    """Compute the inlet flow rate Q_in (mL/s) at time t using the provided periodic function."""
    PI = np.pi
    term1 = 0.20617
    term2 = 0.37759 * np.sin(2 * PI * t / T + 0.59605)
    term3 = 0.2804  * np.sin(4 * PI * t / T - 0.35859)
    term4 = 0.15337 * np.sin(6 * PI * t / T - 1.2509)
    term5 = -0.049889 * np.sin(8 * PI * t / T + 1.3921)
    term6 = 0.038107 * np.sin(10 * PI * t / T - 1.1068)
    term7 = -0.041699 * np.sin(12 * PI * t / T + 1.3985)
    term8 = -0.020754 * np.sin(14 * PI * t / T + 0.72921)
    term9 = 0.013367 * np.sin(16 * PI * t / T - 1.5394)
    term10 = -0.021983 * np.sin(18 * PI * t / T + 0.95617)
    term11 = -0.013072 * np.sin(20 * PI * t / T - 0.022417)
    term12 = 0.0037028 * np.sin(22 * PI * t / T - 1.4146)
    term13 = -0.013973 * np.sin(24 * PI * t / T + 0.77416)
    term14 = -0.012423 * np.sin(26 * PI * t / T - 0.46511)
    term15 = 0.0040098 * np.sin(28 * PI * t / T + 0.95145)
    term16 = -0.0059704 * np.sin(30 * PI * t / T + 0.86369)
    term17 = -0.0073439 * np.sin(32 * PI * t / T - 0.64769)
    term18 = 0.0037006 * np.sin(34 * PI * t / T + 0.74663)
    term19 = -0.0032069 * np.sin(36 * PI * t / T + 0.85926)
    term20 = -0.0048171 * np.sin(38 * PI * t / T - 1.0306)
    term21 = 0.0040403 * np.sin(40 * PI * t / T + 0.28009)
    term22 = -0.0032409 * np.sin(42 * PI * t / T + 1.202)
    term23 = -0.0032517 * np.sin(44 * PI * t / T - 0.93316)
    term24 = 0.0029112 * np.sin(46 * PI * t / T + 0.21405)
    term25 = -0.0022708 * np.sin(48 * PI * t / T + 1.1869)
    term26 = -0.0021566 * np.sin(50 * PI * t / T - 1.1574)
    term27 = 0.0025511 * np.sin(52 * PI * t / T - 0.12915)
    term28 = -0.0024448 * np.sin(54 * PI * t / T + 1.1185)
    term29 = -0.0019032 * np.sin(56 * PI * t / T - 0.99244)
    term30 = 0.0019476 * np.sin(58 * PI * t / T - 0.059885)
    term31 = -0.0019477 * np.sin(60 * PI * t / T + 1.1655)
    term32 = -0.0014545 * np.sin(62 * PI * t / T - 0.85829)
    term33 = 0.0013979 * np.sin(64 * PI * t / T + 0.042912)
    term34 = -0.0014305 * np.sin(66 * PI * t / T + 1.2439)
    term35 = -0.0010775 * np.sin(68 * PI * t / T - 0.79464)
    term36 = 0.0010368 * np.sin(70 * PI * t / T - 0.0043058)
    term37 = -0.0012162 * np.sin(72 * PI * t / T + 1.211)
    term38 = -0.00095707 * np.sin(74 * PI * t / T - 0.66203)
    term39 = 0.00077733 * np.sin(76 * PI * t / T + 0.25642)
    term40 = -0.00092407 * np.sin(78 * PI * t / T + 1.3954)
    term41 = -0.00079585 * np.sin(80 * PI * t / T - 0.49973)
    
    inflow_value = 500 * (
        term1 + term2 + term3 + term4 + term5 + term6 + term7 +
        term8 + term9 + term10 + term11 + term12 + term13 + term14 +
        term15 + term16 + term17 + term18 + term19 + term20 + term21 +
        term22 + term23 + term24 + term25 + term26 + term27 + term28 +
        term29 + term30 + term31 + term32 + term33 + term34 + term35 +
        term36 + term37 + term38 + term39 + term40 + term41
    )
    return inflow_value
