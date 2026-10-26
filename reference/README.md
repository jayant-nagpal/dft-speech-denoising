# Reference outputs

These recorded scores and metadata support result verification:

- `experiment/`: main scores, dataset manifest, inspection, parameter selection
  and completion metadata.
- `comparison/`: constant-gain selection, paired scores and summaries.
- `environment.json`: recorded numerical package versions and core/runner hashes.
- `provenance.json`: source mapping and checksums for these files.

New runs belong in `results/`; reference files remain unchanged. The current
numerical implementation has been checked against these outputs. See
[verification and provenance](../docs/PROVENANCE.md) for their source history.

Audio, waveform arrays, local machine paths and environments are not included.
Completion metadata may reference an example-array checksum retained in the
working archive; the data and experiment commands regenerate that array locally.
