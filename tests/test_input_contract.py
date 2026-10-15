import pandas as pd

EXPECTED_COLUMNS = [
    "studytime", "failures", "absences", "G1", "G2",
    "schoolsup", "famsup", "internet"
]

def test_input_schema():
    row = pd.DataFrame([{
        "studytime": 2,
        "failures": 0,
        "absences": 4,
        "G1": 11,
        "G2": 12,
        "schoolsup": "no",
        "famsup": "yes",
        "internet": "yes",
    }])
    assert list(row.columns) == EXPECTED_COLUMNS
    assert row.shape == (1, 8)
