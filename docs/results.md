# Measured Results

This file is generated from JSON artifacts. Controlled-subset measurements are not full-dataset benchmarks.

## Main tree-growth runs

### CIFAR-10

- Controlled subset: 256 train / 1000 test.
- Root: 91.02% train, 29.50% test at epoch 20.
- Final tree: 100.00% train, 29.80% test, 11 accepted children.
- Change from root: 8.98 training points and 0.30 test points; test cross-entropy 2.041 → 2.148.
- Interpolation criterion reached: yes; parameters: 128,176.
- Greedy pruning retained 9 children and 119,340 parameters (6.9% reduction), with 99.22% train / 29.70% test.

### Cats vs Dogs

- Controlled subset: 256 train / 256 test.
- Root: 91.41% train, 68.75% test at epoch 29.
- Final tree: 100.00% train, 63.67% test, 2 accepted children.
- Change from root: 8.59 training points and -5.08 test points; test cross-entropy 0.645 → 1.315.
- Interpolation criterion reached: yes; parameters: 96,470.
- Greedy pruning retained 2 children and 96,470 parameters (0.0% reduction), with 100.00% train / 63.67% test.

## Capacity diagnostics

### CIFAR-10

- 21 independent models, 3 seeds, widths 1, 2, 4, 8, 16, 32, 64.
- Train-accuracy range: 39.06–100.00%; test-accuracy range: 6.64–23.83%.
- Supporting seeds: 2; repeated-seed double-descent criterion: met.

### Cats vs Dogs

- 21 independent models, 3 seeds, widths 1, 2, 4, 8, 16, 32, 64.
- Train-accuracy range: 68.75–100.00%; test-accuracy range: 48.44–67.58%.
- Supporting seeds: 1; repeated-seed double-descent criterion: not met.

## Interpretive findings

### Memorization versus generalization

The CIFAR-10 corrections added 8.98 training-accuracy points but only 0.30 test points. Test cross-entropy worsened from 2.041 to 2.148. The tree therefore memorized residuals without a corresponding generalization gain.

Cats vs Dogs is stronger negative evidence: corrections added 8.59 training points but changed test accuracy by -5.08 points, while test cross-entropy moved from 0.645 to 1.315.

### Routing insight from receptive fields

The maximum-gate CIFAR bird<-frog example is an actual frog, so that gate identifies source-class evidence rather than a missed bird subclass. In Cats vs Dogs, the dog->cat child likewise reaches gate 1.000 on an actual dog. Exact correction signs do not prevent harm when routing fails out of sample.

### Capacity conclusion

The CIFAR repeated-seed criterion is met with 2/3 supporting seeds. The Cats vs Dogs criterion is not met with 1/3 support. The contrast is reported as dataset/protocol dependence, not as a universal double-descent claim.

## Evidence locations

- Raw stage records: `outputs/<dataset>/main/metrics.json`.
- Run configuration: `outputs/<dataset>/main/config.json`.
- Pruning metadata: `outputs/<dataset>/main/summary.json`.
- Capacity rows: `outputs/<dataset>/<sweep>/capacity_sweep.json`.
- Figures are stored beside the corresponding JSON.
