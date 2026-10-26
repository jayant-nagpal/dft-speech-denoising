"""Select and evaluate one constant-gain control on an existing full run."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import platform
import sys
import numpy as np
import scipy
from threadpoolctl import threadpool_limits
from . import core as c
from .paths import EXPERIMENTS_DIR, COMPARISONS_DIR

CANDIDATES = np.linspace(0.0, 1.0, 101)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def context(args):
    if not args.run.replace("-", "").replace("_", "").isalnum():
        raise SystemExit("Use the existing full run folder name, not a path.")
    run = EXPERIMENTS_DIR / args.run
    identity = json.loads((run / "identity.json").read_text())
    if identity["smoke"]:
        raise SystemExit("This comparison requires the full original run.")
    if digest(Path(c.__file__)) != identity["core_sha256"]:
        raise SystemExit(
            "core.py differs from the original run; preserve the comparison protocol."
        )
    evaluation = json.loads((run / "evaluation.json").read_text())
    for key, name in [
        ("selection_sha256", "selection.json"),
        ("summary_sha256", "summary.csv"),
        ("example_sha256", "fixed_example.npz"),
    ]:
        if digest(run / name) != evaluation[key]:
            raise SystemExit(f"Original run input changed: {name}")
    data = Path(identity["dataset_path"])
    manifest = read_csv(run / "manifest.csv")
    if len(manifest) != 3000:
        raise SystemExit("Expected 3000 original manifest entries.")
    out = COMPARISONS_DIR / f"global-gain-{args.run}"
    out.mkdir(parents=True, exist_ok=True)
    provenance = {
        "original_run": args.run,
        "original_identity_sha256": digest(run / "identity.json"),
        "original_manifest_sha256": digest(run / "manifest.csv"),
        "original_results_sha256": digest(run / "per_utterance_results.csv"),
        "core_sha256": digest(Path(c.__file__)),
        "comparison_script_sha256": digest(Path(__file__)),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "data": str(data),
        "scope": "Post hoc supplementary control; original test outcomes already inspected.",
    }
    marker = out / "identity.json"
    if marker.exists():
        if json.loads(marker.read_text()) != provenance:
            raise SystemExit(
                "Comparison identity changed; do not mix implementations or results."
            )
    else:
        write_json(marker, provenance)
    return (run, data, manifest, out)


def verified_paths(data, manifest, split):
    paths = []
    for row in manifest:
        if row["split"] != split:
            continue
        path = data / row["file"]
        if digest(path) != row["sha256"]:
            raise SystemExit(f"Recording differs from original manifest: {path.name}")
        paths.append(path)
    return sorted(paths)


def select(data, manifest, out):
    if (out / "selection.json").exists():
        raise SystemExit(
            "Selection exists; do not overwrite an established comparison."
        )
    paths = [
        p
        for p in verified_paths(data, manifest, "validation")
        if int(p.stem.split("_")[-1]) < 10
    ]
    if len(paths) != 100:
        raise SystemExit("Expected the same 100 prespecified validation recordings.")
    coefficients = []
    for path in paths:
        for snr in [0, 5, 10]:
            s, _, y, *_ = c.create_mixture(path, "white", snr, "alpha_validation")
            reference, observed = (
                s[c.CALIBRATION_SAMPLES :],
                y[c.CALIBRATION_SAMPLES :],
            )
            coefficients.append(
                (
                    float(observed @ observed),
                    float(observed @ reference),
                    float(reference @ reference),
                )
            )
    ay, cross, energy = np.asarray(coefficients).T
    candidates = []
    for gain in CANDIDATES:
        error = gain * gain * ay - 2 * gain * cross + energy
        if not np.all(error > 0):
            raise RuntimeError("Unexpected nonpositive validation error energy.")
        scores = 10 * np.log10((energy + 1e-18) / (error + 1e-18))
        candidates.append(
            {"gain": float(gain), "mean_validation_snr_db": float(scores.mean())}
        )
    best = max(candidates, key=lambda row: row["mean_validation_snr_db"])
    c.save_csv(out / "validation-candidates.csv", candidates)
    result = {
        "gain": best["gain"],
        "mean_validation_snr_db": best["mean_validation_snr_db"],
        "validation_recordings": 100,
        "validation_mixtures": 300,
        "candidates": 101,
        "candidate_range": [0.0, 1.0],
        "candidate_step": 0.01,
        "noise": "white",
        "input_snrs_db": [0, 5, 10],
        "mixture_seed_phase": "alpha_validation",
        "objective": "Arithmetic mean of individual validation reconstruction-SNR dB scores",
        "policy": "One fixed gain for every sample, recording, noise type and input SNR; no test-time tuning.",
        "candidate_csv_sha256": digest(out / "validation-candidates.csv"),
    }
    write_json(out / "selection.json", result)
    print(json.dumps(result, indent=2))
    print("Selection saved. Interpret it before running evaluate.")


def evaluate(run, data, manifest, out):
    if (out / "comparison-summary.csv").exists() or (out / "evaluation.json").exists():
        raise SystemExit("Comparison evaluation already exists; refusing to overwrite.")
    selected = json.loads((out / "selection.json").read_text())
    if digest(out / "validation-candidates.csv") != selected["candidate_csv_sha256"]:
        raise SystemExit("Validation candidates changed since selection.")
    gain = float(selected["gain"])
    original = read_csv(run / "per_utterance_results.csv")
    lookup = {
        (r["file"], r["noise"], int(float(r["input_snr_db"])), r["method"]): r
        for r in original
    }
    if len(original) != 12000 or len(lookup) != 12000:
        raise SystemExit("Original results must contain 12000 unique method rows.")
    paths = verified_paths(data, manifest, "test")
    if len(paths) != 1000:
        raise SystemExit("Expected the original 1000 test recordings.")
    rows = []
    for noise in ["white", "colored"]:
        for snr in [0, 5, 10]:
            for path in paths:
                s, _, y, *_ = c.create_mixture(path, noise, snr, "test")
                ref, observed = (s[c.CALIBRATION_SAMPLES :], y[c.CALIBRATION_SAMPLES :])
                noisy = c.reconstruction_snr(ref, observed)
                saved_noisy = float(
                    lookup[path.name, noise, snr, "Noisy"]["output_snr_db"]
                )
                if abs(noisy - saved_noisy) > 1e-08:
                    raise RuntimeError(
                        "Regenerated baseline does not match original input SNR."
                    )
                scalar = float(c.reconstruction_snr(ref, gain * observed))
                spectral = float(
                    lookup[path.name, noise, snr, "Classical"]["output_snr_db"]
                )
                rows.append(
                    {
                        "file": path.name,
                        "noise": noise,
                        "input_snr_db": snr,
                        "constant_gain": gain,
                        "noisy_snr_db": float(noisy),
                        "constant_gain_snr_db": scalar,
                        "classical_snr_db": spectral,
                        "constant_minus_noisy_db": scalar - noisy,
                        "classical_minus_constant_db": spectral - scalar,
                    }
                )
            print(f"Scored {noise} at {snr} dB: 1000 recordings", flush=True)
    c.save_csv(out / "per-recording-comparison.csv", rows)
    summary = []
    for noise in ["white", "colored"]:
        for snr in [0, 5, 10]:
            group = [
                r for r in rows if r["noise"] == noise and r["input_snr_db"] == snr
            ]
            differences = np.array([r["classical_minus_constant_db"] for r in group])
            summary.append(
                {
                    "noise": noise,
                    "input_snr_db": snr,
                    "recordings": len(group),
                    "constant_gain": gain,
                    "mean_noisy_snr_db": float(
                        np.mean([r["noisy_snr_db"] for r in group])
                    ),
                    "mean_constant_gain_snr_db": float(
                        np.mean([r["constant_gain_snr_db"] for r in group])
                    ),
                    "mean_classical_snr_db": float(
                        np.mean([r["classical_snr_db"] for r in group])
                    ),
                    "mean_classical_minus_constant_db": float(differences.mean()),
                    "number_classical_better": int(np.sum(differences > 1e-09)),
                    "number_constant_better": int(np.sum(differences < -1e-09)),
                    "number_tied": int(np.sum(abs(differences) <= 1e-09)),
                }
            )
    c.save_csv(out / "comparison-summary.csv", summary)
    write_json(
        out / "evaluation.json",
        {
            "mixtures": 6000,
            "selection_sha256": digest(out / "selection.json"),
            "summary_sha256": digest(out / "comparison-summary.csv"),
            "paired_results_sha256": digest(out / "per-recording-comparison.csv"),
            "scope": "Supplementary comparison on previously inspected test conditions; not independent confirmation.",
        },
    )
    for row in summary:
        print(json.dumps(row))
    print(
        "Original full-run files unchanged. Compare all conditions, including any scalar-control wins."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["select", "evaluate"])
    parser.add_argument("--run", default="reproduction")
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 12):
        raise SystemExit("Use the existing Python 3.12 environment.")
    run, data, manifest, out = context(args)
    print("Supplementary constant-gain control; output:", out, flush=True)
    with threadpool_limits(limits=2):
        if args.stage == "select":
            select(data, manifest, out)
        else:
            evaluate(run, data, manifest, out)


if __name__ == "__main__":
    main()
