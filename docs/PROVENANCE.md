# Verification and provenance

The project retains reference numerical outputs and source hashes so that changes
can be checked without replacing the evidence behind the report. Reference scores
are kept in `reference/`; new executions are written to `results/`.

## Verification record

- Eight numerical/setup unit tests pass.
- All 11 numerical function bodies were checked against the preceding recorded
  implementation after documented name changes.
- The author executed the full implementation and constant-gain control.
- 12,000 main result rows and 6,000 paired comparison rows matched the reference
  values exactly in the checked fields.
- Dataset manifests, validation selections and the saved example also matched.

The machine-readable records are [cleanup-verification.json](cleanup-verification.json)
and [reproduction-verification.json](reproduction-verification.json). Reference-file
checksums and source mapping are in `reference/provenance.json`.

A presentation-only runner revision during verification changed two explanatory
strings. Its earlier source and run identity were preserved in the run's
`source-history/wording-cleanup/` directory. Numerical statements and existing
scores were unchanged; the identity was advanced to the documented revision so
plotting could continue. Other source/data/environment mismatch guards remained
in force.

The reference outputs originate from the author's recorded study, and the current
numerical implementation was subsequently checked against them. Exact agreement
is evidence of numerical consistency, not a new statistical evaluation. In
particular, the constant-gain control was introduced after initial test inspection.
