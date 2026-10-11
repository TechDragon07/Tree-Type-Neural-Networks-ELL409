# Implementation Guide

## 1. Package map

### `tree_nn/models.py`

`TinyConvTrunk` owns the shared image representation. `NodeMLP` is the common
two-layer neural-network primitive. `PairCorrectionNode` implements the exact
multiclass sign constraint. `CorrectionTree` combines the trunk, root, and an
ordered `ModuleList` of correction nodes.

`CorrectionTree.logits_from_features` is intentionally separate from
`forward`. Root training uses images; stage-wise growth uses cached embeddings.
The `active_children` argument permits leave-one-node-out evaluation without
mutating the checkpoint.

### `tree_nn/data.py`

`get_datasets` creates official CIFAR-10 or Oxford-IIIT Pet splits, deterministic
subsets, optional symmetric training-label noise, and transformations.
`NoisyLabels` stores only changed labels. `_label_at` avoids decoding every
image while constructing noise.

CIFAR-10 can use torchvision or the Hugging Face mirror. Both represent the
same official dataset; the source is recorded in run configuration.

### `tree_nn/training.py`

This module contains deterministic seeding, loader construction, image-space
and feature-space evaluation, feature caching, confusion selection, root
training, child growth, pruning, and checkpoint I/O.

The checkpoint format includes:

- constructor arguments for the tree;
- each child's true class, predicted class, and hidden width;
- the complete `state_dict`;
- dataset and pruning metadata.

This avoids relying on pickled Python class instances and allows architecture
reconstruction before weights are loaded.

### `tree_nn/experiments.py`

`ExperimentConfig` is the complete run schema. `run_experiment` owns one
training run and writes an artifact directory. `run_double_descent` launches
independent width/seed combinations and collects their terminal metrics.

### `tree_nn/visualize.py`

The module renders stage-wise learning curves, repeated-seed capacity sweeps,
tree/confusion diagrams, and effective receptive fields. It uses a noninteractive
backend, so the same commands work on headless compute nodes.

### `tree_nn/analysis.py`

This module implements the predeclared per-seed and repeated-seed
double-descent criteria independently of report rendering.

### `tree_nn/cli.py`

The CLI exposes `train` and `sweep`. Every experimental hyperparameter is
explicit and therefore preserved in shell history and configuration artifacts.

## 2. Forward-pass invariants

The test suite verifies that a correction:

- is non-negative in its true-class coordinate;
- is non-positive in its predicted-class coordinate;
- is zero in unrelated coordinates;
- sums to zero across classes;
- survives checkpoint save/load exactly.

These invariants are more important than matching one numerical output because
they define the intended optimization geometry.

## 3. Root training

`train_root` jointly optimizes the trunk and root with AdamW. At each reporting
epoch it evaluates both splits and can stop when `root_target_accuracy` is
reached. This target is deliberately separate from the final
`target_accuracy`; otherwise a successful root would prevent studying tree
growth.

## 4. Feature caching

After root training, `freeze_base` disables trunk/root gradients.
`cache_features` performs one deterministic pass and stores embeddings and
labels in a `TensorDataset`. Benefits include:

- child epochs avoid repeated convolutions;
- all children see the same representation;
- wall-clock cost scales mainly with head size;
- stage-wise attribution is easier because representation drift is removed.

The corresponding cost is reduced adaptability. The report treats failure to
interpolate with cached features as evidence about this design choice, not as a
generic statement about all neural trees.

## 5. Child proposal and fitting

`dominant_confusion` counts ordered \((y,\hat y)\) pairs among current
mistakes. The most common pair is proposed. The child final layer starts near
the zero function.

During fitting, old logits are evaluated without gradients. The route target is
recomputed from the frozen old model for each batch. Positive route examples
receive a capped inverse-frequency weight. After every epoch, full cached
training accuracy is measured; the best state is retained.

If no epoch improves training accuracy, the child is deleted and growth stops.
This is conservative: it prevents a larger tree from being reported when its
newest node has no demonstrated training benefit.

## 6. Pruning

`greedy_prune` begins with all children active. For every active child, it
measures accuracy with that child omitted. The removable child with the best
result is deleted from the active index set if accuracy remains above the
floor. The procedure repeats to a fixed point.

Weights are not physically rewritten; the checkpoint retains the complete
growth history and stores retained indices in metadata. This permits audit of
both the original and pruned models.

## 7. Artifact schema

A run directory has:

```text
config.json
metrics.json
metrics.csv
summary.json
model.pt
learning_curves.png
tree.png
receptive_fields.png
```

`metrics.json` is a list of stage records. A root record identifies its epoch;
a child record identifies node count and ordered confusion pair. `summary.json`
stores class names, device, retained nodes, and both parameter counts.

A capacity sweep additionally has:

```text
capacity_sweep.json
capacity_sweep.csv
double_descent.png
```

Each sweep row contains seed, width, measured parameters, node count, train/test
accuracy and error, and a Boolean interpolation indicator.

## 8. Determinism

Python, NumPy, CPU PyTorch, and CUDA PyTorch seeds are set together. Subset
indices and noisy-label replacements use deterministic generators. Exact
bitwise determinism can still differ by PyTorch version and hardware; version
information should therefore accompany a submitted run.

## 9. Tests

The fast suite covers:

- correction signs and zero-sum behavior;
- complete checkpoint reconstruction;
- dominant-confusion selection;
- child masking for pruning;
- PDF generation with missing data;
- the repeated-seed guard on double-descent claims.

The suite is a structural verification layer. It does not replace end-to-end
dataset experiments.

## 10. Known limitations

- The structure is additive and evaluates every retained child.
- Children use pooled shared features rather than spatial adapters.
- Node acceptance optimizes training accuracy because interpolation is the
  assignment objective.
- Pruning has no post-pruning fine-tuning.
- The current ERF implementation visualizes total gated child activation, not
  separate magnitude and routing maps.
- Full-dataset repeated-seed sweeps are computationally expensive.

Each limitation maps to a concrete future experiment rather than a vague call
for “more work.”
