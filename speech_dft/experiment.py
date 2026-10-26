"""Run data inspection, transform checks, selection, evaluation or plotting."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import platform
import sys
from .paths import ROOT, DATA_DIR, EXPERIMENTS_DIR

# Keep plotting caches local to this project.
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
import numpy as np
import scipy
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from threadpoolctl import threadpool_limits
from . import core as c


def write(path, value):
    path.write_text(json.dumps(value, indent=2))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def data_paths(args):
    files = sorted(args.data.glob("*.wav"))
    if len(files) != 3000:
        raise SystemExit(
            f"Expected 3000 WAV files at {args.data}. Set DFT_DATA or use --data."
        )
    paths = {
        split: [p for p in files if p.stem.split("_")[1] in speakers]
        for split, speakers in c.SPLITS.items()
    }
    for split, pp in paths.items():
        assert len(pp) == 500 * len(c.SPLITS[split])
    if args.smoke:
        paths = {
            k: [p for p in pp if int(p.stem.split("_")[-1]) < 2]
            for k, pp in paths.items()
        }
        paths["test"] = sorted(set(paths["test"] + [args.data / "7_jackson_10.wav"]))
    return paths


def folder(args):
    if not args.run.replace("-", "").replace("_", "").isalnum():
        raise SystemExit("Use a simple --run folder name.")
    out = EXPERIMENTS_DIR / args.run
    out.mkdir(parents=True, exist_ok=True)
    identity = {
        "smoke": args.smoke,
        "dataset_path": str(args.data.resolve()),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "core_sha256": digest(Path(c.__file__)),
        "runner_sha256": digest(Path(__file__)),
        "scope": "SMOKE execution check"
        if args.smoke
        else "FULL classical pilot reproduction",
    }
    marker = out / "identity.json"
    if marker.exists():
        old = json.loads(marker.read_text())
        if old != identity:
            raise SystemExit(
                "Run identity changed. Choose a new --run (do not mix smoke/full, data or code versions)."
            )
    else:
        write(marker, identity)
    print(identity["scope"])
    print("Outputs:", out)
    return out


def guard(path, args):
    if path.exists() and (not args.replace):
        raise SystemExit(
            f"{path.name} exists. Choose a new run or explicitly add --replace."
        )


def inspect(args, out):
    guard(out / "inspection.json", args)
    paths = data_paths(args)
    rows = []
    durations = []
    dtypes = set()
    for split, pp in paths.items():
        for p in pp:
            fs, x = c.wavfile.read(p)
            assert fs == c.SAMPLE_RATE and x.ndim == 1
            rows.append({"file": p.name, "split": split, "sha256": digest(p)})
            durations.append(len(x) / fs)
            dtypes.add(str(x.dtype))
    c.save_csv(out / "manifest.csv", rows)
    report = {
        "counts": {k: len(v) for k, v in paths.items()},
        "speakers": c.SPLITS,
        "sample_rate_hz": c.SAMPLE_RATE,
        "duration_seconds": {
            "min": min(durations),
            "mean": float(np.mean(durations)),
            "max": max(durations),
        },
        "dtypes": sorted(dtypes),
        "note": "Original split labels retained for provenance; the reference group supplies transform-check recordings.",
    }
    write(out / "inspection.json", report)
    print(json.dumps(report, indent=2))


def toy(args, out):
    guard(out / "toy.json", args)
    n = np.arange(8)
    x = np.cos(2 * np.pi * n / 8) + 0.5 * np.sin(4 * np.pi * n / 8)
    F = np.exp(-2j * np.pi * np.outer(n, n) / 8)
    X = F @ x
    report = {
        "DFT_real": X.real.tolist(),
        "DFT_imag": X.imag.tolist(),
        "time_energy": float(np.sum(x * x)),
        "frequency_energy_divided_by_8": float(np.sum(abs(X) ** 2) / 8),
        "inverse_error": float(np.max(abs(F.conj().T @ X / 8 - x))),
    }
    assert report["inverse_error"] < 1e-10
    write(out / "toy.json", report)
    fig, ax = plt.subplots(1, 2, figsize=(8, 3), constrained_layout=True)
    ax[0].stem(n, x, basefmt=" ")
    ax[0].set(xlabel="Sample n", ylabel="Amplitude", title="Eight-sample signal")
    ax[1].stem(n, abs(X), basefmt=" ")
    ax[1].set(xlabel="DFT bin k", ylabel="Magnitude", title="Fourier coordinates")
    save_figure(fig, out, "dft_worked_example")
    print(json.dumps(report, indent=2))


def select(args, out):
    guard(out / "selection.json", args)
    paths = data_paths(args)
    checks = c.verify_transforms(paths["train"])
    alpha, details = c.select_smoothing(paths["validation"])
    report = {
        "alpha": alpha,
        "validation_candidates": details,
        "transform_checks": checks,
    }
    write(out / "selection.json", report)
    print(json.dumps(report, indent=2))


def evaluate(args, out):
    guard(out / "evaluation.json", args)
    paths = data_paths(args)
    if not (out / "selection.json").exists():
        raise SystemExit("Run select first with the same run name/mode.")
    alpha = json.loads((out / "selection.json").read_text())["alpha"]
    rows = []
    for noise in ["white", "colored"]:
        for snr in [0, 5, 10]:
            for p in paths["test"]:
                s, v, y, sz, vz, yz, pn = c.create_mixture(p, noise, snr, "test")
                g = c.estimate_gains(yz, pn, alpha)
                for name, gain in [("Noisy", np.ones_like(g)), ("Classical", g)]:
                    estimate = c.inverse_stft(gain * yz, len(y))
                    speech = c.inverse_stft(gain * sz, len(y))
                    residual = c.inverse_stft(gain * vz, len(y))
                    assert np.max(abs(estimate - speech - residual)) < 1e-09
                    reference = s[c.CALIBRATION_SAMPLES :]
                    output = estimate[c.CALIBRATION_SAMPLES :]
                    achieved = c.reconstruction_snr(
                        reference, y[c.CALIBRATION_SAMPLES :]
                    )
                    assert abs(achieved - snr) < 1e-08
                    score = c.reconstruction_snr(reference, output)
                    rows.append(
                        {
                            "file": p.name,
                            "speaker": p.stem.split("_")[1],
                            "noise": noise,
                            "input_snr_db": snr,
                            "method": name,
                            "output_snr_db": float(score),
                            "snr_improvement_db": float(score - achieved),
                            "normalized_mse": float(
                                np.sum((reference - output) ** 2)
                                / np.sum(reference**2)
                            ),
                            "noise_attenuation_db": float(
                                10
                                * np.log10(
                                    np.sum(v[c.CALIBRATION_SAMPLES :] ** 2)
                                    / (
                                        np.sum(residual[c.CALIBRATION_SAMPLES :] ** 2)
                                        + 1e-18
                                    )
                                )
                            ),
                            "speech_distortion_ratio_db": float(
                                c.reconstruction_snr(
                                    reference, speech[c.CALIBRATION_SAMPLES :]
                                )
                            ),
                        }
                    )
                if p.name == "7_jackson_10.wav" and noise == "white" and (snr == 0):
                    np.savez(
                        out / "fixed_example.npz",
                        reference=s,
                        noise=v,
                        mixture=y,
                        mixture_stft=yz,
                        gain=g,
                        estimate=c.inverse_stft(g * yz, len(y)),
                        noise_power=pn,
                    )
            print(
                "Scored",
                noise,
                snr,
                "dB;",
                len(paths["test"]),
                "recordings",
                flush=True,
            )
    c.save_csv(out / "per_utterance_results.csv", rows)
    summary = []
    for noise in ["white", "colored"]:
        for snr in [0, 5, 10]:
            for method in ["Noisy", "Classical"]:
                rr = [
                    r
                    for r in rows
                    if (r["noise"], r["input_snr_db"], r["method"])
                    == (noise, snr, method)
                ]
                avg, low, high = c.mean_ci(
                    [r["output_snr_db"] for r in rr], c.RANDOM_SEED
                )
                summary.append(
                    {
                        "noise": noise,
                        "input_snr_db": snr,
                        "method": method,
                        "n_utterances": len(rr),
                        "mean_output_snr_db": avg,
                        "ci_low": low,
                        "ci_high": high,
                        "mean_noise_attenuation_db": float(
                            np.mean([r["noise_attenuation_db"] for r in rr])
                        ),
                        "mean_speech_distortion_ratio_db": float(
                            np.mean([r["speech_distortion_ratio_db"] for r in rr])
                        ),
                    }
                )
    c.save_csv(out / "summary.csv", summary)
    report = {
        "alpha": alpha,
        "test_recordings": len(paths["test"]),
        "mixtures": 6 * len(paths["test"]),
        "method_rows": len(rows),
        "selection_sha256": digest(out / "selection.json"),
        "summary_sha256": digest(out / "summary.csv"),
        "example_sha256": digest(out / "fixed_example.npz"),
        "scope": "Smoke check only"
        if args.smoke
        else "Frozen original pilot reproduced, not fresh unseen evaluation",
        "uncertainty": "Utterance-bootstrap descriptive intervals conditional on two test speakers; not population-level confidence.",
    }
    write(out / "evaluation.json", report)
    print("Scoring complete. Next run plot.")


def save_figure(fig, out, name):
    dest = out / "figures"
    dest.mkdir(exist_ok=True)
    fig.savefig(dest / f"{name}.png", dpi=170, bbox_inches="tight")
    fig.savefig(dest / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot(args, out):
    guard(out / "plot.json", args)
    if not (out / "evaluation.json").exists():
        raise SystemExit("Run evaluate first.")
    report = json.loads((out / "evaluation.json").read_text())
    for key, name in [
        ("selection_sha256", "selection.json"),
        ("summary_sha256", "summary.csv"),
        ("example_sha256", "fixed_example.npz"),
    ]:
        if report[key] != digest(out / name):
            raise SystemExit(
                "Scoring inputs changed; rerun evaluate --replace before plotting."
            )
    import csv

    with (out / "summary.csv").open() as f:
        summary = list(csv.DictReader(f))
    fig, axes = plt.subplots(
        1, 2, figsize=(8, 3.2), constrained_layout=True, sharey=True
    )
    for ax, noise in zip(axes, ["white", "colored"]):
        for method, color, style in [
            ("Noisy", "#777777", "--"),
            ("Classical", "#23687d", "-"),
        ]:
            rr = [r for r in summary if r["noise"] == noise and r["method"] == method]
            ax.plot(
                [float(r["input_snr_db"]) for r in rr],
                [float(r["mean_output_snr_db"]) for r in rr],
                style,
                marker="o",
                label=method,
                color=color,
            )
        ax.set(
            xlabel="Input SNR (dB)",
            title=noise.capitalize() + " noise",
            xticks=[0, 5, 10],
        )
        ax.grid(alpha=0.15)
    axes[0].set_ylabel("Mean reconstruction SNR (dB)")
    axes[0].legend(frameon=False)
    save_figure(fig, out, "performance")
    z = np.load(out / "fixed_example.npz")
    signals = {
        "Reference": z["reference"],
        "Noisy": z["mixture"],
        "Classical": z["estimate"],
    }
    ref = max(float(np.max(abs(c.stft(z["reference"])) ** 2)), 1e-12)
    fig, axes = plt.subplots(
        3, 1, figsize=(8, 6), constrained_layout=True, sharex=True, sharey=True
    )
    for ax, (name, wave) in zip(axes, signals.items()):
        zz = c.stft(wave)
        t = (
            np.arange(len(zz)) * c.HOP_LENGTH / c.SAMPLE_RATE
            - c.CALIBRATION_SAMPLES / c.SAMPLE_RATE
        )
        im = ax.pcolormesh(
            t,
            np.fft.rfftfreq(c.FRAME_LENGTH, 1 / c.SAMPLE_RATE) / 1000,
            10 * np.log10(abs(zz.T) ** 2 / ref + 1e-12),
            shading="auto",
            vmin=-65,
            vmax=0,
            cmap="cividis",
            rasterized=True,
        )
        ax.set(
            title=name,
            ylabel="Frequency (kHz)",
            xlim=(0, (len(wave) - c.CALIBRATION_SAMPLES) / c.SAMPLE_RATE),
        )
    axes[-1].set_xlabel("Time since original recording starts (s)")
    fig.colorbar(
        im,
        ax=list(axes),
        label="Power relative to common reference maximum (dB)",
        shrink=0.8,
    )
    save_figure(fig, out, "spectrograms")
    fig, ax = plt.subplots(figsize=(8, 3), constrained_layout=True)
    t = (
        np.arange(len(z["gain"])) * c.HOP_LENGTH / c.SAMPLE_RATE
        - c.CALIBRATION_SAMPLES / c.SAMPLE_RATE
    )
    im = ax.pcolormesh(
        t,
        np.fft.rfftfreq(c.FRAME_LENGTH, 1 / c.SAMPLE_RATE) / 1000,
        z["gain"].T,
        shading="auto",
        vmin=0,
        vmax=1,
        cmap="cividis",
        rasterized=True,
    )
    ax.set(
        xlabel="Time since original recording starts (s)",
        ylabel="Frequency (kHz)",
        title="Classical amplitude gain",
        xlim=(0, (len(z["reference"]) - c.CALIBRATION_SAMPLES) / c.SAMPLE_RATE),
    )
    fig.colorbar(im, ax=ax, label="Amplitude multiplier")
    save_figure(fig, out, "gains")
    fig, ax = plt.subplots(figsize=(8, 3), constrained_layout=True)
    for name, wave in signals.items():
        color = {"Reference": "#333333", "Noisy": "#aaaaaa", "Classical": "#23687d"}[
            name
        ]
        ax.plot(
            np.arange(len(wave) - c.CALIBRATION_SAMPLES) / c.SAMPLE_RATE,
            wave[c.CALIBRATION_SAMPLES :],
            label=name,
            color=color,
            alpha=0.75,
            linewidth=0.8,
        )
    ax.set(
        xlabel="Time since original recording starts (s)",
        ylabel="Amplitude",
        title="One preselected example",
    )
    ax.legend(frameon=False)
    save_figure(fig, out, "waveforms")
    lines = [
        "% Generated by speech_dft.experiment; smoke tables are not report evidence.",
        "\\begin{tabular}{llrr}",
        "\\toprule",
        "Noise & Input SNR (dB) & Noisy & Classical \\\\",
        "\\midrule",
    ]
    for noise in ["white", "colored"]:
        for snr in [0, 5, 10]:
            rr = {
                r["method"]: r
                for r in summary
                if r["noise"] == noise and float(r["input_snr_db"]) == snr
            }
            lines.append(
                f"{noise.capitalize()} & {snr} & {float(rr['Noisy']['mean_output_snr_db']):.2f} & {float(rr['Classical']['mean_output_snr_db']):.2f} "
                + "\\\\"
            )
    lines += ["\\bottomrule", "\\end{tabular}"]
    (out / "results-table.tex").write_text("\n".join(lines) + "\n")
    write(
        out / "plot.json",
        {
            "smoke": args.smoke,
            "summary_sha256": digest(out / "summary.csv"),
            "fixed_example": "7_jackson_10.wav, white noise, 0 dB",
            "figures": ["performance", "spectrograms", "gains", "waveforms"],
            "numerical_plots_only": True,
        },
    )
    print(
        "Created four computed figure pairs and results-table.tex. Read their limitations before using them."
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "stage", choices=["doctor", "inspect", "toy", "select", "evaluate", "plot"]
    )
    p.add_argument(
        "--data",
        type=Path,
        default=Path(
            os.environ.get(
                "DFT_DATA",
                str(DATA_DIR / "free-spoken-digit-dataset-1.0.10/recordings"),
            )
        ),
    )
    p.add_argument("--run", default="reproduction")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--replace", action="store_true")
    a = p.parse_args()
    if a.stage == "doctor":
        print(
            json.dumps(
                {
                    "python": platform.python_version(),
                    "executable": sys.executable,
                    "architecture": platform.machine(),
                    "numpy": np.__version__,
                    "scipy": scipy.__version__,
                    "matplotlib": matplotlib.__version__,
                    "dataset_path": str(a.data.resolve()),
                    "dataset_wavs": len(list(a.data.glob("*.wav"))),
                },
                indent=2,
            )
        )
        return
    if sys.version_info[:2] != (3, 12):
        raise SystemExit("Use the Python 3.12 environment for recorded reproduction.")
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )
    out = folder(a)
    with threadpool_limits(limits=2):
        stages = {
            "inspect": inspect,
            "toy": toy,
            "select": select,
            "evaluate": evaluate,
            "plot": plot,
        }
        stages[a.stage](a, out)


if __name__ == "__main__":
    main()
