from __future__ import annotations

import unittest

import pandas as pd

from src.predict import assign_priority


class PriorityAssignmentTest(unittest.TestCase):
    def test_priority_respects_high_threshold(self) -> None:
        df = pd.DataFrame({"predicted_fill_pct": [82, 88, 92, 96]})

        result = assign_priority(df, threshold=95)

        self.assertEqual(result["priority"].tolist(), ["Skip", "Skip", "Skip", "Critical"])

    def test_priority_respects_low_threshold(self) -> None:
        df = pd.DataFrame({"predicted_fill_pct": [49, 50, 79, 80, 90]})

        result = assign_priority(df, threshold=50)

        self.assertEqual(result["priority"].tolist(), ["Skip", "Medium", "Medium", "High", "Critical"])


if __name__ == "__main__":
    unittest.main()
