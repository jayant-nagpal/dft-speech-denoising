# Reproducibility guide

## Protocol

| Setting | Value |
|---|---|
| Dataset | Free Spoken Digit Dataset v1.0.10; 3,000 recordings |
| Sample rate | 8,000 Hz, mono |
| Reference group | george, lucas, theo; 1,500 recordings; transform checks |
| Validation | yweweler; 500 recordings |
| Test | jackson, nicolas; 1,000 recordings |
| Selection subset | Recording indices below 10 per digit in validation; 100 recordings |
| Frame / hop | 256 / 64 samples; 32 / 8 ms |
| Window | Square root of a periodic Hann |
| Calibration | 2,560 noise-only samples; 320 ms |
| Reference scaling | Mean removal, then RMS 0.1 |
| Noise | White or first-order coloured, coefficient 0.85 |
| Input SNRs | 0, 5 and 10 dB; exact finite-record scaling |
| Noise keys | SHA-256 with base seed 20261015 |
| Alpha candidates | 0.8, 0.9, 0.98; white-noise validation at all three SNRs |
| Gain floor | 0.05 in amplitude |
| Score | Reference energy / total error energy, in dB |

Calibration and transform padding are excluded from scoring; silences inside the
original recordings remain included. Condition means average per-recording dB
scores. Utterance-bootstrap intervals describe these recordings, conditional on
the two test speakers.

The constant-gain control selects from 0:0.01:1 using the same validation subset
and noise conditions. One selected gain applies to every test sample and condition;
it does not adapt using calibration. The control was proposed after the initial
test results were inspected, so it is supplementary rather than an independent
blind confirmation. It does not isolate frequency selection from all other parts
of the spectral pipeline.

## Outputs

```text
results/
  experiments/<run>/
    identity.json                 # versions, data location and source hashes
    manifest.csv                  # recording names, groups and checksums
    inspection.json
    toy.json
    selection.json
    evaluation.json
    per_utterance_results.csv
    summary.csv
    fixed_example.npz             # prespecified example arrays
    figures/                      # computed PDF/PNG pairs
    results-table.tex
    report-analysis/              # checked tables and diagnostics
  comparisons/global-gain-<run>/
    identity.json
    selection.json
    validation-candidates.csv
    evaluation.json
    per-recording-comparison.csv
    comparison-summary.csv
    report-analysis/
  teaching/                       # analytic figures, arrays and parameters
```

The prespecified plotted example is `7_jackson_10.wav`, white noise, 0 dB input.
It was not selected for a favourable score. Plot scales are shared where required
for comparison. Teaching figures are mathematical examples, not test results.

## Checks

```sh
python -m unittest discover -s tests
python -m speech_dft.verify
```

The eight tests use synthetic arrays, a temporary WAV and temporary metadata;
they do not process the dataset. The verification command checks the included
reference files against their recorded checksums.

After a full run and its constant-gain comparison:

```sh
python -m speech_dft.verify --run baseline --comparison
```

This checks selected alpha, result identities and per-recording numerical fields
against the reference. Missing/duplicate rows, non-finite values and differences
outside relative tolerance 1e-9 or absolute tolerance 1e-8 fail verification.
Use this alongside the dataset manifest, recorded environment and plotted evidence.

## Evidence

The tested implementation matched 12,000 main result rows and 6,000 paired control
rows with largest absolute difference 0.0 in the checked fields. Dataset manifests,
validation selections and the saved example also matched the reference evidence.
These checks establish numerical agreement on the tested system, not performance
on new data or a cross-platform guarantee. Details are in
[PROVENANCE.md](PROVENANCE.md).
