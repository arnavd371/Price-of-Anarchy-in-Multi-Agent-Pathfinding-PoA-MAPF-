import unittest
from pathlib import Path


class TestReadmePagesURL(unittest.TestCase):
    def test_readme_contains_github_io_url(self) -> None:
        readme = (Path(__file__).resolve().parents[1] / "README.md").read_text()
        self.assertIn("https://arnavd371.github.io/Price-of-Anarchy-in-Multi-Agent-Pathfinding-PoA-MAPF-/", readme)
        self.assertIn("/docs/", readme)


if __name__ == "__main__":
    unittest.main()
