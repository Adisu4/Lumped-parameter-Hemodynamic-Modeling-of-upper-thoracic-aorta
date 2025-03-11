import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from source import hemodynamic_solver, model_parameters


def test_simulation_length() -> None:
    data = hemodynamic_solver.simulate()
    expected_length = int(round(model_parameters.num_periods * model_parameters.T / model_parameters.dt)) + 1
    assert len(data["time"]) == expected_length
