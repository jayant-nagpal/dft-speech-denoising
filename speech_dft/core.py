"""Numerical core: windowed DFT, reconstruction, noise construction and spectral gains."""
import csv
import hashlib
import numpy as np
from scipy.io import wavfile
from scipy.signal import lfilter

SAMPLE_RATE = 8000
FRAME_LENGTH = 256
HOP_LENGTH = 64
CALIBRATION_SAMPLES = 2560
WINDOW = np.sqrt(np.hanning(FRAME_LENGTH + 1)[:-1])
SPLITS = {
    "train": ["george", "lucas", "theo"],
    "validation": ["yweweler"],
    "test": ["jackson", "nicolas"],
}
RANDOM_SEED = 20261015


def stft(x):
    """Centred, padded, unnormalised forward DFT of windowed frames."""
    z = np.pad(x, (FRAME_LENGTH // 2, FRAME_LENGTH))
    frames = np.lib.stride_tricks.sliding_window_view(z, FRAME_LENGTH)[::HOP_LENGTH]
    return np.fft.rfft(frames * WINDOW, axis=1)


def inverse_stft(z, length):
    """Weighted overlap-add, divided by accumulated squared windows."""
    frames = np.fft.irfft(z, n=FRAME_LENGTH, axis=1) * WINDOW
    x = np.zeros((len(z) - 1) * HOP_LENGTH + FRAME_LENGTH)
    w = np.zeros_like(x)
    for i, frame in enumerate(frames):
        sl = slice(i * HOP_LENGTH, i * HOP_LENGTH + FRAME_LENGTH)
        x[sl] += frame
        w[sl] += WINDOW**2
    x = np.divide(x, w, out=np.zeros_like(x), where=w > 1e-12)
    return x[FRAME_LENGTH // 2 : FRAME_LENGTH // 2 + length]


def load_recording(path):
    rate, x = wavfile.read(path)
    if rate != SAMPLE_RATE or x.ndim != 1:
        raise ValueError(f"Unexpected format in {path}: {rate}, {x.shape}")
    x = x.astype(np.float64)
    x -= x.mean()
    return x / max(np.sqrt(np.mean(x**2)), 1e-12) * 0.1


def key_seed(name, noise, snr, phase):
    key = f"{RANDOM_SEED}/{name}/{noise}/{snr}/{phase}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "little")


def create_mixture(path, noise, snr, phase):
    """Create seeded noise, a mixture and its calibration-based bin-power estimate."""
    clean = load_recording(path)
    s = np.pad(clean, (CALIBRATION_SAMPLES, 0))
    rng = np.random.default_rng(key_seed(path.name, noise, snr, phase))
    raw = rng.standard_normal(len(s) + 1024)
    if noise == "colored":
        raw = lfilter([1], [1, -0.85], raw)
    v = raw[1024:]
    # Reference energy sets input difficulty; it is not a gain-estimator input.
    v *= np.sqrt(
        np.mean(clean**2) / (np.mean(v[CALIBRATION_SAMPLES:] ** 2) * 10 ** (snr / 10))
    )
    y = s + v
    yz, sz, vz = (stft(y), stft(s), stft(v))
    centers = np.arange(len(yz)) * HOP_LENGTH
    calibration = (centers >= FRAME_LENGTH // 2) & (
        centers <= CALIBRATION_SAMPLES - FRAME_LENGTH // 2
    )
    # Use only complete frames inside the known noise-only prefix.
    pn = np.mean(np.abs(yz[calibration]) ** 2, axis=0) + 1e-12
    return (s, v, y, sz, vz, yz, pn)


def estimate_gains(yz, pn, alpha):
    """Estimate decision-directed Wiener-style amplitude gains, with a 0.05 floor."""
    gamma = np.abs(yz) ** 2 / pn
    gains = np.zeros_like(gamma)
    previous = np.zeros(gamma.shape[1])
    for i, posterior in enumerate(gamma):
        xi = alpha * previous + (1 - alpha) * np.maximum(posterior - 1, 0)
        gains[i] = np.clip(xi / (1 + xi), 0.05, 1)
        previous = gains[i] ** 2 * posterior
    return gains


def reconstruction_snr(s, estimate):
    """Score reference energy relative to total reconstruction-error energy."""
    return 10 * np.log10(
        (np.sum(s**2) + 1e-18) / (np.sum((s - estimate) ** 2) + 1e-18)
    )


def verify_transforms(paths):
    x = load_recording(paths[0])
    error = float(np.max(np.abs(x - inverse_stft(stft(x), len(x)))))
    n = np.arange(8)
    f = np.exp(-2j * np.pi * np.outer(n, n) / 8)
    toy = np.cos(2 * np.pi * n / 8) + 0.5 * np.sin(4 * np.pi * n / 8)
    matrix_error = float(np.max(np.abs(f @ toy - np.fft.fft(toy))))
    parseval_error = float(abs(np.sum(toy**2) - np.sum(abs(f @ toy) ** 2) / 8))
    assert error < 1e-10 and matrix_error < 1e-10 and (parseval_error < 1e-10)
    return {
        "identity_reconstruction_max_error": error,
        "matrix_vs_fft_max_error": matrix_error,
        "parseval_error": parseval_error,
    }


def select_smoothing(paths):
    """Choose alpha on the prespecified validation subset, never on test scores."""
    rows = []
    subset = [p for p in paths if int(p.stem.split("_")[-1]) < 10]
    for alpha in [0.8, 0.9, 0.98]:
        values = []
        for path in subset:
            for snr in [0, 5, 10]:
                s, v, y, sz, vz, yz, pn = create_mixture(
                    path, "white", snr, "alpha_validation"
                )
                gain = estimate_gains(yz, pn, alpha)
                out = inverse_stft(yz * gain, len(y))
                values.append(
                    reconstruction_snr(
                        s[CALIBRATION_SAMPLES:], out[CALIBRATION_SAMPLES:]
                    )
                )
        rows.append(
            {"alpha": alpha, "validation_mean_output_snr_db": float(np.mean(values))}
        )
    alpha = max(rows, key=lambda a: a["validation_mean_output_snr_db"])["alpha"]
    return (alpha, rows)


def save_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean_ci(values, seed):
    """Utterance-resampled intervals, conditional on these two test speakers."""
    a = np.asarray(values)
    rng = np.random.default_rng(seed)
    means = np.mean(a[rng.integers(0, len(a), (1000, len(a)))], axis=1)
    return (float(a.mean()), *np.quantile(means, [0.025, 0.975]).tolist())
