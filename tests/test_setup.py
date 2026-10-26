"""Check setup reporting and run metadata without reading any audio."""
import argparse
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import scipy

from speech_dft import experiment


class SetupTests(unittest.TestCase):
    def test_doctor_reports_versions_without_recordings(self):
        with tempfile.TemporaryDirectory() as folder:
            output = io.StringIO()
            arguments = ["experiment", "doctor", "--data", folder]
            with patch("sys.argv", arguments), contextlib.redirect_stdout(output):
                experiment.main()
        report = json.loads(output.getvalue())
        self.assertEqual(report["scipy"], scipy.__version__)
        self.assertEqual(report["dataset_wavs"], 0)

    def test_run_identity_records_scipy_version(self):
        with tempfile.TemporaryDirectory() as folder:
            args = argparse.Namespace(run="setup-check", smoke=False, data=Path(folder))
            with patch.object(
                experiment, "EXPERIMENTS_DIR", Path(folder)
            ), contextlib.redirect_stdout(io.StringIO()):
                destination = experiment.folder(args)
            identity = json.loads((destination / "identity.json").read_text())
        self.assertEqual(identity["scipy"], scipy.__version__)
        self.assertFalse(identity["smoke"])


if __name__ == "__main__":
    unittest.main()
