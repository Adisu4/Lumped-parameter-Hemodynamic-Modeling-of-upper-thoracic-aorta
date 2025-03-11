.PHONY: install test run analyze visualize clean help

# Install required Python packages
install:
	pip install -r requirements.txt

# Run all tests
test:
	pytest -v

# Run the main simulation
run:
	python main.py

# Run time step analysis
analyze:
	python -c "from source.time_step_analysis import analyze_time_steps; analyze_time_steps()"

# Run data visualization 
visualize:
	python -c "from source.hemodynamic_solver import simulate; from source.data_visualization import plot_results, sensitivity_analysis; data = simulate(); plot_results(data); sensitivity_analysis(data)"

# Remove generated files and cache
clean:
	rm -f *.png
	rm -rf __pycache__
	rm -rf source/__pycache__
	rm -rf tests/__pycache__
	rm -rf plots/*.png

# Show available Makefile commands
help:
	@echo "Available targets:"
	@echo "  install   - Install required packages"
	@echo "  test      - Run all tests"
	@echo "  run       - Run the main simulation"
	@echo "  analyze   - Run time step analysis"
	@echo "  visualize - Generate visualization plots"
	@echo "  clean     - Remove generated files and cache"
	@echo "  help      - Show this help message"

