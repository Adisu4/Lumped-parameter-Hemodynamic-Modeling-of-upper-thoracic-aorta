from source.inflow_condition import inflow  

def test_inflow_output() -> None:
    value = inflow(0)
    assert isinstance(value, float)
    assert value >= 0
