"""Dataset quality tests: run validation and assert that every check passes."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import validate_dataset as v


def test_all_validation_checks_pass():
    results = v.validate()
    failed = [r for r in results if not r["passed"]]
    assert not failed, f"Validation failed: {[(r['check'], r['detail']) for r in failed]}"


def test_positive_rate_in_range():
    import pandas as pd
    df = pd.read_csv(os.path.join(v.C.ROOT, "data", "raw", "component_snapshots.csv"))
    assert 0.015 <= df["target"].mean() <= 0.025


def test_row_count():
    import pandas as pd
    df = pd.read_csv(os.path.join(v.C.ROOT, "data", "raw", "component_snapshots.csv"))
    assert 240000 <= len(df) <= 320000
