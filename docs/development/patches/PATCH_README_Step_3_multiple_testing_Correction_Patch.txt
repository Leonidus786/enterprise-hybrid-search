Step 3 follow-up patch: family-wise correction for multiple paired-bootstrap comparisons.

Merge these folders into E:\enterprise-hybrid-search, preserving folder structure.

Replace:
  scripts\compare_runs.py

Add:
  tests\test_compare_adjustment.py
  docs\development\step3-multiple-comparison-correction.md

`src\hs\compare.py` does not need changes; its `confidence` argument is already supported.

This patch preserves the nominal 95% CI and makes the significance column use a Bonferroni-adjusted interval over all pair/metric tests requested in the command. It does not change retrieval code or existing benchmark JSON.
