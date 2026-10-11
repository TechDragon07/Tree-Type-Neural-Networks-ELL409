$ErrorActionPreference = "Stop"

python -m tree_nn.cli train --dataset cifar10 --root-target-accuracy 0.95 `
    --root-epochs 150 --max-nodes 30 --node-epochs 40 `
    --target-accuracy 0.999 --run-name main
python -m tree_nn.cli train --dataset catsdogs --image-size 64 --trunk-width 32 `
    --root-target-accuracy 0.95 --root-epochs 120 --max-nodes 20 --node-epochs 40 `
    --target-accuracy 0.999 --run-name main

python -m tree_nn.cli sweep --dataset cifar10 --train-size 10000 --label-noise 0.10 `
    --widths 4 8 12 16 24 32 48 64 --seeds 42 43 44 --root-epochs 120 `
    --eval-every 5 --max-nodes 20 --node-epochs 30 --sweep-name full_sweep
python -m tree_nn.cli sweep --dataset catsdogs --label-noise 0.10 `
    --widths 4 8 12 16 24 32 48 --seeds 42 43 44 --root-epochs 100 `
    --eval-every 5 --max-nodes 15 --node-epochs 30 --sweep-name full_sweep

python scripts/summarize_results.py
Write-Output "Run scripts/build_report.ps1 with Tectonic to rebuild the formal PDF."
