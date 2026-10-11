# Experimental Protocol

## 1. Questions and hypotheses

The experiments are organized around four questions.

**Q1: Can a stage-wise neural correction tree interpolate a finite image
training set?**  
Hypothesis: a root deliberately stopped below interpolation, followed by
confusion-specific neural nodes, can reach at least 99.9% training accuracy.

**Q2: What limits correction growth?**  
Hypothesis: early nodes fix coherent class confusions, while late nodes become
limited by frozen-feature collisions and increasingly small routed subsets.

**Q3: Does test error exhibit double descent near interpolation?**  
Hypothesis: adding 10% symmetric training-label noise makes the interpolation
region visible, but a claim is accepted only if the rise and second fall repeat
across seeds.

**Q4: Can the interpolating tree be made smaller without losing the fit?**  
Hypothesis: some earlier nodes become redundant after later corrections and can
be removed by leave-one-node-out pruning.

## 2. Datasets

### CIFAR-10

CIFAR-10 contains 50,000 training and 10,000 test images from ten balanced
classes. Images are resized to \(32\times32\), converted to tensors, and
normalized with the standard CIFAR-10 channel statistics. The project can
obtain the same official data through torchvision or the Hugging Face mirror.

### Cats vs Dogs

The binary task uses Oxford-IIIT Pet. Torchvision's `binary-category` target
maps the 37 breeds to cat and dog species without a manually maintained breed
list. The official `trainval` split is used for training and the official
`test` split for testing. Images are converted to RGB, resized, and normalized
with the same channel convention used by the trunk.

### Controlled CPU study versus full study

The repository distinguishes:

- **controlled CPU study:** a seeded subset used to obtain real interpolation,
  growth, ERF, and capacity evidence on the available CPU-only machine;
- **full study:** all official training examples, intended for a CUDA machine.

Subset results are never described as full-dataset accuracy. Their purpose is
to study mechanisms under a compute budget. Exact subset sizes are stored in
each run's `config.json`.

## 3. Data transformations

Augmentation defaults to off in the interpolation experiment. The empirical
object is a fixed finite sample, so stochastic crops would change the sample
being interpolated at every epoch. A separate generalization experiment can
enable random crop and horizontal flip. Test data are never augmented.

Symmetric label noise is deterministic for a given seed. A selected training
label is replaced uniformly by one of the other classes. No test label is
modified.

## 4. Main tree-growth experiment

For each dataset:

1. choose the seeded training and test subsets;
2. train trunk and root until the root target or epoch budget is reached;
3. freeze and cache all trunk features;
4. identify the most frequent ordered residual confusion;
5. add and train one sign-constrained correction node;
6. retain the best epoch if it improves training accuracy, otherwise reject
   the node and stop;
7. repeat until 99.9% accuracy or the node budget;
8. prune nodes while preserving the configured accuracy floor;
9. save metrics, model, tree diagram, learning curve, and ERF figure.

Important hyperparameters are the trunk width, root hidden width, root stopping
accuracy, child hidden width, gate-loss coefficient, gate sparsity coefficient,
and node budget. They are serialized rather than inferred from filenames.

## 5. Capacity sweep

Capacity is varied through trunk width \(w\). Root and child hidden widths are
set to \(4w\). A capacity point is a separately initialized and trained model;
weights are not inherited from a narrower model. The same subset construction
and noise probability are used within a seed.

The default sweep uses widths

\[
\{4,8,12,16,24,32,48,64\}
\]

and seeds \(\{42,43,44\}\). The x-axis is measured total parameters after tree
growth. Mean and standard deviation are plotted across seeds.

The broad sweep is exploratory. If it suggests a peak, additional widths and
at least five seeds should be run around that region. This local refinement is
necessary because an apparent peak can be a coarse-grid artifact.

## 6. Metrics

Every stage records:

- cross-entropy and accuracy on training data;
- cross-entropy and accuracy on untouched test data;
- accepted node count;
- total parameter count;
- confusion direction of the accepted child.

The summary additionally records:

- unpruned parameter count;
- retained node indices;
- pruned parameter count;
- class names and device.

The current implementation reports test metrics during training for diagnosis,
but no optimizer step, node direction, acceptance decision, or pruning
decision uses them. A stronger future protocol would reserve a validation set
for model selection and inspect test data only once.

## 7. Predeclared interpretation rules

### Interpolation

Interpolation is reached at training accuracy \(\ge99.9\%\). “Near
interpolation” is not silently rounded to 100%.

### Double descent

The automated report requires three or more seeds and two-thirds seed
agreement. For a supporting seed, test error must be lower somewhere before
the first interpolating model, rise at that model, and fall at a larger model.
If these conditions fail, the conclusion is “not established,” not “absent in
all settings.”

### Receptive fields

The root uses a fixed sample index. Each child uses the maximum-gate image from
the first 256 test examples, a deterministic alternative to cherry-picking.
Map normalization is fixed. A root map becoming more localized than a leaf map
is reported as an observation, not forced into the expected
semantic-to-spurious narrative.

### Pruning

The result is called “greedy empirical minimization.” It is not described as a
proof of minimum network size.

## 8. Threats to validity

1. **Subset scale.** Controlled CPU results may not transfer to the complete
   dataset.
2. **Optimization versus capacity.** A model can fail to interpolate because
   of insufficient epochs rather than insufficient parameters.
3. **Frozen-feature collisions.** Child heads cannot recover distinctions
   absent from the cached embedding.
4. **Repeated test inspection.** Although not used for decisions, repeatedly
   viewing test curves can influence later human choices.
5. **Noise dependence.** Label noise sharpens interpolation but changes the
   learning problem.
6. **ERF limitations.** Input gradients are local sensitivity, can be noisy,
   and do not prove causal feature use.
7. **Additive execution.** All retained children are evaluated, so nominal
   tree depth does not equal hard-path compute.
8. **Seed count.** Three seeds are a minimum diagnostic, not a definitive
   uncertainty analysis.

## 9. Reproduction commands

The exact controlled commands are stored in `scripts/run_pilot.ps1`; full
commands are in `scripts/run_all.ps1`. The final report is built from committed
JSON and PNG artifacts. See `docs/reproducibility.md` for environment and
artifact details.
