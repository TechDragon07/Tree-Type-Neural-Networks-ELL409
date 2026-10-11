# Result artifacts

The repository retains only artifacts used by the paper:

- `cifar10/main/` and `catsdogs/main/` contain the final tree checkpoints,
  configurations, stage metrics, pruning summaries, and figures.
- `*/interpolation_sweep/` contains the aggregated per-seed capacity rows and
  the plotted sweep.

Per-capacity checkpoints are intentionally omitted. They can be regenerated
with `scripts/run_pilot.ps1`; the aggregate JSON contains every capacity point
reported in the paper.
