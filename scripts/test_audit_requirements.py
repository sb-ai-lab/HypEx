"""Regression checks for auditing incompatible locked dependency stacks."""

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from audit_requirements import audit_batches, main


class AuditRequirementsTests(unittest.TestCase):
    def test_alternative_stacks_keep_every_applicable_pin(self):
        batches = audit_batches(
            [
                "numpy==1.26.4",
                "numpy==2.4.6",
                "pandas==2.2.3",
                "pandas==3.0.0",
                "NumPy==1.26.4",
                'numpy==1.24.4; python_version < "3.0"',
                "# exported comment",
                "",
            ]
        )
        self.assertEqual(
            batches,
            [
                ["numpy==1.26.4", "pandas==2.2.3"],
                ["numpy==2.4.6", "pandas==3.0.0"],
            ],
        )

    def test_unresolved_requirements_fail_closed(self):
        for line in [
            "numpy>=1.26",
            "numpy==1.*",
            "numpy",
            "numpy @ https://example.org/numpy.whl",
        ]:
            with self.subTest(line=line), self.assertRaises(ValueError):
                audit_batches([line])

    def test_audits_remaining_batches_after_findings_and_returns_failure(self):
        audited = []

        def run(command, check):
            audited.append(
                Path(command[command.index("--requirement") + 1]).read_text()
            )
            self.assertIn("--no-deps", command)
            self.assertIn("--disable-pip", command)
            self.assertEqual(command[-2:], ["--progress-spinner", "off"])
            return type("Result", (), {"returncode": 1 if len(audited) == 1 else 0})()

        with TemporaryDirectory() as directory:
            requirements = Path(directory) / "requirements.txt"
            requirements.write_text("numpy==1.26.4\nnumpy==2.4.6\n")
            arguments = ["audit", str(requirements), "--progress-spinner", "off"]
            with patch.object(sys, "argv", arguments):
                with patch("audit_requirements.subprocess.run", side_effect=run):
                    self.assertEqual(main(), 1)
        self.assertEqual(audited, ["numpy==1.26.4\n", "numpy==2.4.6\n"])


if __name__ == "__main__":
    unittest.main()
