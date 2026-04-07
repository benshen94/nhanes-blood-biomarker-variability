# Agent Directives

1. **File Hygiene & Organization**: Whenever you generate new script files or output files (.csv, .json, etc.), DO NOT leave them in the root directory. Always move them to the appropriate folders (e.g., `scripts/`, `data/`, `output/`).
2. **Clean Up Temporary Files**: Always delete temporary files created during processing (e.g., intermediate JSON lists, temporary data dumps) once they are no longer needed. 
3. **Documentation**: Always update the `README.md` file reflecting any major architectural or dataset changes.
4. **Commit & Push**: After completing a task that modifies the project successfully, commit your changes with a descriptive message and push them to the repository.
5. **Public Dashboard Source Of Truth**: Make dashboard code, template, and data-processing changes in this `nhanes_dashboard` repo. Do not hand-edit the published HTML or JSON files in the public repo as the primary workflow.
6. **Public Dashboard Export Flow**: After updating the audience-facing dashboard here, run `python3 scripts/export_public_dashboard_site.py`. This exports the built public site into the sibling repo at `../biomarker_dashboard/`.
7. **Public Repo Publish Flow**: After exporting, commit and push the changed files from the sibling repo at `/Users/benshenhar/Library/CloudStorage/GoogleDrive-benshenhar@gmail.com/My Drive/Weizmann/Alon Lab/Aging/biomarker_dashboard`. That repo is the GitHub Pages deployment target.
