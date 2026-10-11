$ErrorActionPreference = "Stop"

# Controlled CPU study. These are real but deliberately small subset
# experiments; they are not full-dataset benchmarks.

python -m tree_nn.cli train `
    --dataset cifar10 `
    --cifar-source huggingface `
    --train-size 256 `
    --test-size 1000 `
    --cache-images `
    --batch-size 64 `
    --trunk-width 16 `
    --root-hidden 64 `
    --root-target-accuracy 0.85 `
    --root-epochs 100 `
    --max-nodes 20 `
    --node-hidden 64 `
    --node-epochs 30 `
    --node-lr 0.005 `
    --target-accuracy 0.999 `
    --prune-accuracy 0.99 `
    --run-name main

python -m tree_nn.cli train `
    --dataset catsdogs `
    --pet-source huggingface `
    --train-size 256 `
    --test-size 256 `
    --cache-images `
    --batch-size 64 `
    --trunk-width 16 `
    --root-hidden 64 `
    --root-target-accuracy 0.90 `
    --root-epochs 100 `
    --max-nodes 12 `
    --node-hidden 128 `
    --node-epochs 100 `
    --node-lr 0.01 `
    --target-accuracy 0.999 `
    --prune-accuracy 0.99 `
    --run-name main

# A bounded capacity diagnostic. The report may describe double descent only
# if the predeclared repeated-seed criterion is met.
python -m tree_nn.cli sweep `
    --dataset cifar10 `
    --cifar-source huggingface `
    --train-size 64 `
    --test-size 256 `
    --cache-images `
    --batch-size 64 `
    --label-noise 0.10 `
    --widths 1 2 4 8 16 32 64 `
    --seeds 42 43 44 `
    --root-target-accuracy 0.95 `
    --root-epochs 80 `
    --eval-every 5 `
    --max-nodes 10 `
    --node-epochs 50 `
    --node-lr 0.005 `
    --sweep-name interpolation_sweep

python -m tree_nn.cli sweep `
    --dataset catsdogs `
    --pet-source huggingface `
    --train-size 64 `
    --test-size 256 `
    --cache-images `
    --batch-size 64 `
    --label-noise 0.10 `
    --widths 1 2 4 8 16 32 64 `
    --seeds 42 43 44 `
    --root-target-accuracy 0.95 `
    --root-epochs 80 `
    --eval-every 5 `
    --max-nodes 4 `
    --node-epochs 50 `
    --node-lr 0.005 `
    --sweep-name interpolation_sweep

python scripts/summarize_results.py
