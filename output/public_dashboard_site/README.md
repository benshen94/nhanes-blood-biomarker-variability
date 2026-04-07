# Public Dashboard Site Export

This folder is ready to become a separate public GitHub Pages repository.

## Files included
- `index.html`
- `longevity-explorer.html`
- `aging_biomarkers_dashboard.html`
- `aging_biomarkers_public/`
- `data/series/` for the curated biomarker subset only

## Suggested deployment
1. Create a new GitHub repository for the public site only.
2. Copy the contents of this folder into that repo root.
3. Push the repo.
4. Enable GitHub Pages from the repo root branch.

## Notes
- This export includes only the files needed by the audience-facing dashboard.
- It does not include the full analysis repo, source notebooks, or unrelated dashboard assets.
- The dashboard expects `aging_biomarkers_public/` and `data/series/` to live next to the HTML files.
