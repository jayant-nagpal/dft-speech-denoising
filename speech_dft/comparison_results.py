"""Check saved constant-gain comparisons and generate their report table."""
import argparse
import csv
import hashlib
import json
import statistics
from .paths import EXPERIMENTS_DIR, COMPARISONS_DIR


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    if not args.run.replace("-", "").replace("_", "").isalnum():
        raise SystemExit("Use a run folder name.")
    folder = COMPARISONS_DIR / f"global-gain-{args.run}"
    meta = json.loads((folder / "evaluation.json").read_text())
    for key, name in [
        ("selection_sha256", "selection.json"),
        ("summary_sha256", "comparison-summary.csv"),
        ("paired_results_sha256", "per-recording-comparison.csv"),
    ]:
        assert digest(folder / name) == meta[key], f"Changed comparison input: {name}"
    selection = json.loads((folder / "selection.json").read_text())
    assert (
        digest(folder / "validation-candidates.csv")
        == selection["candidate_csv_sha256"]
    )
    assert (
        selection["validation_recordings"] == 100
        and selection["validation_mixtures"] == 300
    )
    summaries = read_csv(folder / "comparison-summary.csv")
    paired = read_csv(folder / "per-recording-comparison.csv")
    assert len(summaries) == 6 and len(paired) == 6000
    assert len({(r["file"], r["noise"], r["input_snr_db"]) for r in paired}) == 6000
    original = read_csv(EXPERIMENTS_DIR / args.run / "per_utterance_results.csv")
    lookup = {
        (r["file"], r["noise"], float(r["input_snr_db"]), r["method"]): float(
            r["output_snr_db"]
        )
        for r in original
    }
    lines = [
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "Noise & Input SNR & Fixed gain & Spectral & Advantage & \\shortstack{Number\\\\spectral better} \\\\",
        "\\midrule",
    ]
    checked = []
    for row in summaries:
        noise, snr = (row["noise"], float(row["input_snr_db"]))
        group = [
            r for r in paired if r["noise"] == noise and float(r["input_snr_db"]) == snr
        ]
        assert len(group) == 1000
        for r in group:
            assert float(r["constant_gain"]) == selection["gain"]
            assert (
                abs(
                    float(r["classical_snr_db"])
                    - lookup[r["file"], noise, snr, "Classical"]
                )
                < 1e-10
            )
            assert (
                abs(float(r["noisy_snr_db"]) - lookup[r["file"], noise, snr, "Noisy"])
                < 1e-08
            )
            assert (
                abs(
                    float(r["classical_minus_constant_db"])
                    - (float(r["classical_snr_db"]) - float(r["constant_gain_snr_db"]))
                )
                < 1e-10
            )
        for field, source in [
            ("mean_constant_gain_snr_db", "constant_gain_snr_db"),
            ("mean_classical_snr_db", "classical_snr_db"),
            ("mean_classical_minus_constant_db", "classical_minus_constant_db"),
        ]:
            assert (
                abs(
                    float(row[field])
                    - statistics.mean((float(r[source]) for r in group))
                )
                < 1e-10
            )
        differences = [float(r["classical_minus_constant_db"]) for r in group]
        assert sum((x > 1e-09 for x in differences)) == int(
            row["number_classical_better"]
        )
        assert sum((x < -1e-09 for x in differences)) == int(
            row["number_constant_better"]
        )
        assert sum((abs(x) <= 1e-09 for x in differences)) == int(row["number_tied"])
        lines.append(
            f"{noise.capitalize()} & {snr:g} & {float(row['mean_constant_gain_snr_db']):.2f} & {float(row['mean_classical_snr_db']):.2f} & {float(row['mean_classical_minus_constant_db']):.2f} & {row['number_classical_better']} "
            + "\\\\"
        )
        checked.append(
            {
                "noise": noise,
                "input_snr_db": snr,
                "recordings": 1000,
                "minimum_spectral_minus_constant_db": min(differences),
                "number_spectral_better": int(row["number_classical_better"]),
                "number_constant_better": int(row["number_constant_better"]),
                "number_tied": int(row["number_tied"]),
            }
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    output = folder / "report-analysis"
    output.mkdir(exist_ok=True)
    (output / "global-gain-table.tex").write_text("\n".join(lines) + "\n")
    result = {
        "scope": "Postprocessing existing student-run comparison; no experiment rerun.",
        "constant_gain": selection["gain"],
        "conditions": checked,
        "inputs": {
            name: digest(folder / name)
            for name in [
                "selection.json",
                "comparison-summary.csv",
                "per-recording-comparison.csv",
            ]
        },
    }
    (output / "checks.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
