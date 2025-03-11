from source.ML_simulation import run_optimal_simulation
from source import data_visualization

def main():
    print("Running ML-based time step optimization...")
    results = run_optimal_simulation()
    
    print("\nGenerating visualization of final results...")
    data_visualization.plot_results(results)
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()