# Reproducibility and Artifact Audit

## Environment

The controlled study was prepared on:

- Windows 11;
- Python 3.12;
- PyTorch 2.10 CPU build;
- torchvision 0.25;
- no CUDA device.

The precise installed package versions can be captured with:

```powershell
python -m pip freeze > outputs/environment.txt
```

Install the project and development checks with:

```powershell
python -m pip install -e ".[dev]"
python -m pytest -q
python -m ruff check .
```

## Controlled CPU study

Run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_pilot.ps1
```

This script uses fixed, explicitly disclosed subsets. It is intended to produce
real evidence on a CPU in a bounded time. It must not be described as a
full-dataset benchmark.

## Full study

On a CUDA machine:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_all.ps1
powershell -ExecutionPolicy Bypass -File scripts/build_report.ps1
```

The full sweep is expensive because each width/seed pair is independently
trained. Reduce neither the seed count nor post-interpolation widths merely to
obtain a visually attractive curve.

## Build the formal report

The source is `report/main.tex`. A portable Tectonic compiler can build it:

```powershell
Push-Location report
tectonic main.tex --outdir .
Copy-Item main.pdf corrective_neural_tree_report.pdf -Force
Pop-Location
```

The committed `report/corrective_neural_tree_report.pdf` is the submission
artifact. After changing experimental outputs, update the numerical statements
in `main.tex`, rebuild, and verify them against the JSON files.

## Audit checklist

Before submission:

1. confirm the repository remains private and the instructor has access;
2. confirm the commit hash in the report is the submitted commit;
3. check every reported accuracy against `metrics.json`;
4. check every parameter count against `summary.json`;
5. confirm subset sizes and label-noise probability from `config.json`;
6. confirm a double-descent statement meets the automated repeated-seed rule;
7. ensure figures are generated files rather than hand-drawn curves;
8. ensure failed or incomplete runs are not silently omitted;
9. open the PDF on another machine and test the repository hyperlink;
10. rerun the test suite.

## Artifact integrity

The code never fabricates default metrics. The lightweight report generator
prints “not run” when an expected artifact is absent. The formal paper follows
the same policy: a controlled result is labeled controlled, a full result is
labeled full, and a hypothesis is not rewritten as an observation.

For stronger archival integrity, future runs should add:

- Git commit hash;
- SHA-256 hashes of split indices;
- package lock file;
- elapsed training time and peak memory;
- checkpoint hash;
- explicit run status (`complete`, `failed`, or `interrupted`).
