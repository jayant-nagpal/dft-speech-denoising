"""Verify reference files and compare a completed cleaned-code run with them."""
import argparse
import csv
import hashlib
import json
import math

from .paths import ROOT, EXPERIMENTS_DIR, COMPARISONS_DIR

REFERENCE = ROOT / "reference"


def read_csv(path):
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def compare_csv(expected_path, actual_path, keys, fields):
    def key(row):
        return tuple(float(row[k]) if k == "input_snr_db" else row[k] for k in keys)

    expected_rows, actual_rows = read_csv(expected_path), read_csv(actual_path)
    expected = {key(row): row for row in expected_rows}
    actual = {key(row): row for row in actual_rows}
    if len(expected) != len(expected_rows) or len(actual) != len(actual_rows):
        raise ValueError("Duplicate result rows.")
    if expected.keys() != actual.keys():
        raise ValueError("Result identities differ from the reference.")
    largest_difference = 0.0
    for identity, old in expected.items():
        for field in fields:
            before, after = float(old[field]), float(actual[identity][field])
            if not math.isfinite(after) or not math.isclose(
                before, after, rel_tol=1e-9, abs_tol=1e-8
            ):
                raise ValueError(
                    f"Result mismatch: {identity}, {field}: {before} versus {after}"
                )
            largest_difference = max(largest_difference, abs(before - after))
    return {
        "rows_compared": len(expected),
        "largest_absolute_difference": largest_difference,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", help="Completed cleaned-code full run to compare")
    parser.add_argument(
        "--comparison",
        action="store_true",
        help="Also compare its completed constant-gain control",
    )
    args = parser.parse_args()
    if args.comparison and not args.run:
        parser.error("--comparison requires --run")
    provenance = json.loads((REFERENCE / "provenance.json").read_text())
    for name, expected in provenance["reference_files"].items():
        actual = hashlib.sha256((REFERENCE / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Reference file changed: {name}")
    result = {"reference_files_verified": len(provenance["reference_files"])}
    if args.run:
        if not args.run.replace("-", "").replace("_", "").isalnum():
            parser.error("Use a run-folder name, not a path")
        run = EXPERIMENTS_DIR / args.run
        identity = json.loads((run / "identity.json").read_text())
        if identity["smoke"]:
            raise ValueError("A smoke run cannot replace a full-run comparison.")
        selection = json.loads((run / "selection.json").read_text())
        expected_selection = json.loads(
            (REFERENCE / "experiment/selection.json").read_text()
        )
        if selection["alpha"] != expected_selection["alpha"]:
            raise ValueError("Selected alpha differs from the original full run.")
        result["experiment"] = compare_csv(
            REFERENCE / "experiment/per_utterance_results.csv",
            run / "per_utterance_results.csv",
            ["file", "noise", "input_snr_db", "method"],
            [
                "output_snr_db",
                "snr_improvement_db",
                "normalized_mse",
                "noise_attenuation_db",
                "speech_distortion_ratio_db",
            ],
        )
        if args.comparison:
            comparison = COMPARISONS_DIR / f"global-gain-{args.run}"
            result["comparison"] = compare_csv(
                REFERENCE / "comparison/per-recording-comparison.csv",
                comparison / "per-recording-comparison.csv",
                ["file", "noise", "input_snr_db"],
                [
                    "constant_gain",
                    "noisy_snr_db",
                    "constant_gain_snr_db",
                    "classical_snr_db",
                    "classical_minus_constant_db",
                ],
            )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
