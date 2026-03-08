# Blood Duplicate Harmonization

- [in_progress] Audit blood biomarker duplicates across names, units, and missing-unit aliases in `data/processed/lab_variable_manifest.parquet`.
- [pending] Update `src/build_analysis_dataset.py` to merge true duplicates using smarter name normalization, unit parsing, and conversion handling.
- [pending] Emit blood duplicate merge documentation into processed/output artifacts and update `README.md`.
- [pending] Rebuild blood dataset/dashboard outputs and verify the merged catalog.
- [pending] Run focused tests, review the diff, commit changes, and push to the repository.
