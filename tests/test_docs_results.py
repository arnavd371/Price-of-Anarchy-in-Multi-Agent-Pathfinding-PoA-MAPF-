import json
import unittest
from pathlib import Path


class TestDocsResults(unittest.TestCase):
    def test_results_json_schema_and_bounds(self) -> None:
        results_path = Path(__file__).resolve().parents[1] / "docs" / "results.json"
        rows = json.loads(results_path.read_text())

        self.assertIsInstance(rows, list)
        self.assertGreaterEqual(len(rows), 3)

        required = {
            "agents",
            "ne_with_shortcut",
            "ne_without_shortcut",
            "social_optimum_with_shortcut",
            "poa_with_shortcut",
            "avg_time_with_shortcut",
            "avg_time_without_shortcut",
        }

        for row in rows:
            self.assertTrue(required.issubset(row.keys()))
            self.assertGreaterEqual(row["poa_with_shortcut"], 1.0)
            self.assertGreaterEqual(row["ne_with_shortcut"], row["social_optimum_with_shortcut"])


if __name__ == "__main__":
    unittest.main()
