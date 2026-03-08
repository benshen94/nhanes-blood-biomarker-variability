# Lessons

- Keep the generator script as the source of truth for dashboard HTML; patching generated outputs directly creates drift.
- For large static HTML generators, moving the shared shell into a template file is safer than repeatedly editing a giant inline string in the Python source.
