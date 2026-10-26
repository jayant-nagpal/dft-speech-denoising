"""Check saved full-run scores and generate report tables without rerunning denoising."""
import argparse
import csv
import hashlib
import json
import statistics
from .paths import EXPERIMENTS_DIR


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table(headers, rows, alignment):
    lines = [
        "\\begin{tabular}{" + alignment + "}",
        "\\toprule",
        " & ".join(headers) + " \\\\",
        "\\midrule",
    ]
    lines += [" & ".join(map(str, row)) + " \\\\" for row in rows]
    return "\n".join(lines + ["\\bottomrule", "\\end{tabular}"]) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    if not args.run.replace("-", "").replace("_", "").isalnum():
        raise SystemExit("Use a run-folder name, not a path.")
    run = EXPERIMENTS_DIR / args.run
    identity = json.loads((run / "identity.json").read_text())
    if identity["smoke"]:
        raise SystemExit("Smoke results are not full report evidence.")
    evaluation = json.loads((run / "evaluation.json").read_text())
    for key, name in [
        ("selection_sha256", "selection.json"),
        ("summary_sha256", "summary.csv"),
        ("example_sha256", "fixed_example.npz"),
    ]:
        assert sha256(run / name) == evaluation[key], f"Changed source: {name}"
    with (run / "per_utterance_results.csv").open(newline="") as file:
        rows = list(csv.DictReader(file))
    with (run / "summary.csv").open(newline="") as file:
        means = list(csv.DictReader(file))
    assert len(rows) == 12000
    assert (
        len({(r["file"], r["noise"], r["input_snr_db"], r["method"]) for r in rows})
        == 12000
    )
    groups, performance, consistency = ([], [], [])
    for noise in ["white", "colored"]:
        for snr in [0, 5, 10]:
            group = [
                r
                for r in rows
                if r["noise"] == noise and float(r["input_snr_db"]) == snr
            ]
            noisy = {r["file"]: r for r in group if r["method"] == "Noisy"}
            enhanced = {r["file"]: r for r in group if r["method"] == "Classical"}
            assert (
                len(noisy) == len(enhanced) == 1000 and noisy.keys() == enhanced.keys()
            )
            differences = [
                float(enhanced[f]["output_snr_db"]) - float(noisy[f]["output_snr_db"])
                for f in sorted(noisy)
            ]
            for f in noisy:
                assert abs(float(noisy[f]["output_snr_db"]) - snr) < 1e-08
                assert (
                    abs(
                        float(enhanced[f]["snr_improvement_db"])
                        - (
                            float(enhanced[f]["output_snr_db"])
                            - float(noisy[f]["output_snr_db"])
                        )
                    )
                    < 1e-08
                )
            mean = statistics.mean(
                (float(r["output_snr_db"]) for r in enhanced.values())
            )
            saved = next(
                (
                    r
                    for r in means
                    if r["noise"] == noise
                    and float(r["input_snr_db"]) == snr
                    and (r["method"] == "Classical")
                )
            )
            assert abs(mean - float(saved["mean_output_snr_db"])) < 1e-10
            worst = min(
                enhanced, key=lambda f: float(enhanced[f]["snr_improvement_db"])
            )
            item = dict(
                noise=noise,
                input_snr_db=snr,
                n=1000,
                mean_output_snr_db=mean,
                mean_improvement_db=statistics.mean(differences),
                median_improvement_db=statistics.median(differences),
                minimum_improvement_db=min(differences),
                worst_file=worst,
                improved=sum((d > 1e-09 for d in differences)),
                worsened=sum((d < -1e-09 for d in differences)),
                unchanged=sum((abs(d) <= 1e-09 for d in differences)),
            )
            groups.append(item)
            performance.append(
                [
                    noise.capitalize(),
                    f"{snr:.2f}",
                    f"{snr:.2f}",
                    f"{mean:.2f}",
                    f"{item['mean_improvement_db']:.2f}",
                ]
            )
            consistency.append(
                [
                    noise.capitalize(),
                    str(snr),
                    f"{item['median_improvement_db']:.3f}",
                    f"{item['minimum_improvement_db']:.3f}",
                    str(item["improved"]),
                ]
            )
    example = next(
        (
            r
            for r in rows
            if r["file"] == "7_jackson_10.wav"
            and r["noise"] == "white"
            and (float(r["input_snr_db"]) == 0)
            and (r["method"] == "Classical")
        )
    )
    noise_fraction = 10 ** (-float(example["noise_attenuation_db"]) / 10)
    distortion_fraction = 10 ** (-float(example["speech_distortion_ratio_db"]) / 10)
    total_fraction = float(example["normalized_mse"])
    result = {
        "run": args.run,
        "scope": "Postprocessing of the student-executed full reproduction; no new experiment",
        "input_sha256": {
            name: sha256(run / name)
            for name in [
                "identity.json",
                "selection.json",
                "summary.csv",
                "per_utterance_results.csv",
                "fixed_example.npz",
            ]
        },
        "conditions": groups,
        "example": {
            "file": example["file"],
            "noise": "white",
            "input_snr_db": 0,
            "output_snr_db": float(example["output_snr_db"]),
            "noise_attenuation_db": float(example["noise_attenuation_db"]),
            "speech_distortion_ratio_db": float(example["speech_distortion_ratio_db"]),
            "noise_energy_fraction_of_reference": noise_fraction,
            "speech_distortion_energy_fraction_of_reference": distortion_fraction,
            "total_error_energy_fraction_of_reference": total_fraction,
            "cross_term_fraction_of_reference": total_fraction
            - noise_fraction
            - distortion_fraction,
        },
    }
    output = run / "report-analysis"
    output.mkdir(exist_ok=True)
    (output / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    (output / "performance-table.tex").write_text(
        table(
            ["Noise", "Input SNR", "Noisy", "Classical", "$\\Delta$ SNR"],
            performance,
            "lrrrr",
        )
    )
    (output / "consistency-table.tex").write_text(
        table(
            [
                "Noise",
                "Input SNR",
                "Median $\\Delta$",
                "Minimum $\\Delta$",
                "Number improved",
            ],
            consistency,
            "lrrrr",
        )
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
