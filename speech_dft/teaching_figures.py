"""Generate analytic teaching plots; these are not speech test results."""
import argparse
import json
import os
from .paths import ROOT, TEACHING_DIR

# Keep plotting caches local to this project.
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = TEACHING_DIR
TEAL, BROWN, DARK = ("#23687d", "#986330", "#292929")


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def coefficient_sequence():
    fs, size = (8000, 256)
    n = np.arange(size)
    s = np.cos(2 * np.pi * 1000 * n / fs) + 0.45 * np.cos(2 * np.pi * 2000 * n / fs)
    v = 0.8 * np.cos(2 * np.pi * 2500 * n / fs)
    y = s + v
    Y = np.fft.rfft(y)
    gain = np.ones_like(Y.real)
    gain[80] = 0.05
    result = np.fft.irfft(gain * Y, n=size)
    assert np.allclose(result, s + 0.05 * v, atol=1e-12)
    assert np.allclose(np.fft.irfft(Y, n=size), y, atol=1e-12)
    f = np.fft.rfftfreq(size, 1 / fs) / 1000
    fig, axes = plt.subplots(
        2, 3, figsize=(7.3, 4.0), sharex="row", sharey="row", constrained_layout=True
    )
    names = ["1. Reference: two tones", "2. Add a third tone", "3. Attenuate bin 80"]
    waves = [s, y, result]
    for col, (name, wave) in enumerate(zip(names, waves)):
        axes[0, col].plot(
            n[:64] / fs * 1000,
            wave[:64],
            color=TEAL if col != 1 else BROWN,
            lw=1,
            marker=".",
            ms=2,
        )
        axes[0, col].set(title=name, xlabel="Time (ms)", xlim=(0, 8), ylim=(-2.4, 2.4))
        amplitude = 2 * np.abs(np.fft.rfft(wave)) / size
        amplitude[[0, -1]] /= 2
        indices = np.flatnonzero(amplitude > 1e-10)
        marker, stems, base = axes[1, col].stem(
            f[indices], amplitude[indices], basefmt=" "
        )
        plt.setp(marker, color=TEAL if col != 1 else BROWN, markersize=4)
        plt.setp(stems, color=TEAL if col != 1 else BROWN, linewidth=1.5)
        axes[1, col].set(
            xlabel="Frequency (kHz)",
            xlim=(0, 4),
            ylim=(0, 1.12),
            xticks=[0, 1, 2, 3, 4],
        )
        for row in range(2):
            axes[row, col].grid(alpha=0.16)
    axes[0, 0].set_ylabel("Amplitude")
    axes[1, 0].set_ylabel("One-sided tone amplitude")
    axes[1, 2].annotate(
        "0.80 becomes 0.04",
        xy=(2.5, 0.04),
        xytext=(1.55, 0.7),
        fontsize=8,
        arrowprops={"arrowstyle": "->", "color": DARK},
    )
    save(fig, "coordinate_sequence")
    np.savez(
        OUT / "coordinate_sequence.npz",
        sample_rate=fs,
        reference=s,
        interference=v,
        mixture=y,
        gain=gain,
        reconstructed=result,
    )
    return {
        "sample_rate_hz": fs,
        "samples": size,
        "tone_frequencies_hz": [1000, 2000, 2500],
        "tone_amplitudes": [1, 0.45, 0.8],
        "attenuated_bin": 80,
        "amplitude_gain": 0.05,
        "result_max_error_against_s_plus_005v": float(
            np.max(abs(result - s - 0.05 * v))
        ),
        "scope": "Unwindowed exactly bin-centred tone example; not speech-denoising evidence.",
    }


def gain_cost():
    g = np.linspace(0, 1, 501)
    ps, pv = (4.0, 1.0)
    damage = (1 - g) ** 2 * ps
    residual = g * g * pv
    total = damage + residual
    optimum = ps / (ps + pv)
    assert np.isclose(optimum, 0.8)
    assert np.isclose((1 - optimum) ** 2 * ps + optimum**2 * pv, 0.8)
    fig, ax = plt.subplots(figsize=(6.8, 3.25), constrained_layout=True)
    ax.plot(g, damage, "--", color=BROWN, lw=1.6, label="Speech loss")
    ax.plot(g, residual, ":", color=TEAL, lw=2, label="Residual noise")
    ax.plot(g, total, "-", color=DARK, lw=1.7, label="Total expected error")
    ax.scatter([optimum], [0.8], color=DARK, zorder=5)
    ax.vlines(optimum, 0, 0.8, color="#777777", lw=0.8, ls="--")
    ax.annotate(
        "Minimum at gain 0.8",
        xy=(0.8, 0.8),
        xytext=(0.45, 2.05),
        arrowprops={"arrowstyle": "->", "color": DARK},
        fontsize=9,
    )
    ax.set(
        xlabel="Amplitude gain g (0: silence; 1: unchanged)",
        ylabel="Expected squared coefficient error",
        xlim=(0, 1),
        ylim=(0, 4.2),
    )
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax.grid(alpha=0.16)
    save(fig, "gain_compromise")
    np.savez(
        OUT / "gain_compromise.npz",
        gain=g,
        speech_loss=damage,
        residual_noise=residual,
        total=total,
    )
    return {
        "speech_power": ps,
        "noise_power": pv,
        "optimal_gain": optimum,
        "minimum_expected_error": 0.8,
        "scope": "Analytic error model; not a measured SNR or a guarantee for each mixture.",
    }


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )
    metadata = {
        "numpy": np.__version__,
        "matplotlib": matplotlib.__version__,
        "coordinate_sequence": coefficient_sequence(),
        "gain_compromise": gain_cost(),
    }
    (OUT / "teaching-figures.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("Wrote computed teaching figures and their source arrays:", OUT)


if __name__ == "__main__":
    main()
