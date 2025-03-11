# PhD_Interview_Task
PhD Interview Task: Hemodynamic Modeling and ODEs. This repository contains the Python source code and detailed report for the task on 0D lumped-parameter blood flow modeling, parameter calibration, and identifiability analysis
# Upper Thoracic Aorta Hemodynamic Modeling

## Overview
This project implements a zero-dimensional (0D) hemodynamic model of the upper thoracic aorta coupled with a Windkessel terminal model. The simulation is based on a system of ordinary differential equations (ODEs) that model blood flow and pressure in the vessel using the forward Euler method.

The model simulates the hemodynamics of the upper thoracic aorta over several cardiac cycles to reach a periodic steady state. It includes detailed time step analysis to optimize simulation accuracy and performance, sensitivity analysis to study wall compliance (C0) effects, and various vessel configurations.

## Key Features
Time Step Analysis: Systematically evaluates different time step sizes to determine the optimal balance between computational efficiency and simulation accuracy
Sensitivity Analysis: Studies the effect of varying wall compliance (C0) on vessel pressure
Multiple Vessel Configurations: Supports different vessel arrangements:
Single vessel with Windkessel model
Configuration A: Parent-daughter vessel bifurcation with pressure-based coupling
Configuration B: Parent-daughter vessel bifurcation with flow-based coupling
Energy-Preserving Coupling: Implementation of energy-consistent coupling schemes
ML-Enhanced Simulation: Data-driven approach to determine optimal simulation parameters
Parameter Calibration: Tools for model parameter optimization

## Repository Structure
upper_thoracic_aorta/
├── source/
│   ├── __init__.py
│   ├── inflow_condition.py       # Periodic inflow function
│   ├── model_parameters.py       # Upper thoracic aorta model parameters
│   ├── hemodynamic_solver.py     # Coupled vessel-Windkessel solver using forward Euler
│   ├── data_visualization.py     # Plotting functions for simulation results
│   ├── time_step_analysis.py     # Analyzes optimal time step size for simulations
│   ├── Energy_preserving_coupling.py  # Energy-consistent coupling methods
│   ├── Configuration_A.py        # Parent-daughter vessel with pressure continuity
│   ├── Configuration_B.py        # Parent-daughter vessel with flow continuity
│   ├── ML_simulation.py          # Data-driven simulation optimization
│   └── caliberation.py           # Model parameter calibration tools
├── tests/
│   ├── __init__.py
│   ├── test_inflow.py            # Unit tests for the inflow function
│   └── test_solver.py            # Unit tests for the solver module
├── plots/                        # Output directory for generated plots
├── main.py                       # Entry point for standard simulation
├── run_ml.py                     # Entry point for ML-based optimization
├── Makefile                      # Task automation
├── .flake8                       # Linting configuration
└── README.md                     # This file


- Task Automation: 
  A Makefile is provided to simplify common tasks such as installation, linting, testing, and running the application.

## Setup and Installation

### Using Poetry (Recommended)

1. **Install Poetry**  
   Follow the instructions at [python-poetry.org](https://python-poetry.org/) to install Poetry if you haven't already.

2. **Install Dependencies:**

   ```bash
   poetry install
   ```

3. Run the Application:

To run the simulation with the default configuration (adjust the source code in source/main.py to choose your desired analysis):

To run the time step analysis:
poetry run python -m source.time_step_analysis

To run the main application:
poetry run python main.py
To run the bifurcation anlysis
poetry run python configuration A.py
poetry run python configuration B.py

Without Poetry
Alternatively, you can create a virtual environment and install the required packages manually:

python -m venv venv
# On Windows:
venv\Scripts\activate
# On Unix/Mac:
source venv/bin/activate

pip install numpy matplotlib scipy pandas scikit-learn pytest

# Run standard simulation
python main.py

# Run ML optimization
python run_ml.py

# To run the bifurcation anlysis
python configuration B.py
python configuration A.py


Running Tests
To run the unit tests using pytest, execute:

poetry run pytest

Or, if pytest is installed globally:
pytest


Technical Details
Hemodynamic Model
The model couples a 0D vessel representation with a three-element Windkessel model and solves:

Vessel volume conservation: dV/dt = Qin - Q
Momentum balance: dQ/dt = (P - R₀Q - Pout)/L₀
Pressure-volume relationship: P = P₀ + (V - V₀)/C₀
Windkessel dynamics: dPwk/dt = (Q - Qwk)/Cwk, where Qwk = (Pwk - Pven)/Rd
Time Step Analysis
The time_step_analysis.py module performs a detailed evaluation of simulation accuracy and computational efficiency across different time steps:

Tests multiple time steps (0.01s, 0.006866s, 0.0005s)
Calculates RMS error using the smallest time step as reference
Evaluates computation time for each step size
Generates visualizations comparing flow and pressure waveforms
Provides recommendations for optimal time step selection based on accuracy vs. speed tradeoffs
Performs physiological reality checks on simulation outputs
ML-Enhanced Simulation
The ML_simulation.py module uses machine learning techniques to:

Analyze simulation performance across many time step values
Predict optimal simulation parameters
Reduce computational cost while maintaining accuracy
Identify correlations between model parameters and simulation results



Contact
For questions or further information, please contact:
Name: Adisu Mengesha
Email: adisumengesha315@gmail.com

