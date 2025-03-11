from source import hemodynamic_solver, data_visualization
from source.time_step_analysis import analyze_time_steps

def main():
    # First analyze different time steps
    print("Analyzing effect of different time steps...")
    recommended_dt = analyze_time_steps()
    
    # Now run the main simulation with the recommended time step
    print(f"\nRunning main simulation with dt={recommended_dt}s...")
    simulation_data = hemodynamic_solver.simulate()
    
    print("Generating plots...")
    data_visualization.plot_results(simulation_data)
    data_visualization.sensitivity_analysis(simulation_data)
    print("Simulation complete!")

if __name__ == "__main__":
    main()
