"""Package interface checks."""
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]


class TestInterfaces(unittest.TestCase):
    def test_imports(self):
        from pcrort import conditional_risk_matrix, relative_contrast
        from pcrort.robust import UncertaintySet, robust_select
        from pcrort.tabular import ExperimentConfig, discover, run_experiment
        for item in (conditional_risk_matrix, relative_contrast, robust_select, discover, run_experiment):
            self.assertTrue(callable(item))
        self.assertEqual(ExperimentConfig().clients.count, 3)

    def test_demo_cli(self):
        process = subprocess.run([sys.executable, "-m", "pcrort.cli", "demo"],
                                 cwd=ROOT, text=True, capture_output=True, check=True)
        self.assertIn("full covariance", process.stdout)
        self.assertIn("diagonal only", process.stdout)


if __name__ == "__main__":
    unittest.main()
