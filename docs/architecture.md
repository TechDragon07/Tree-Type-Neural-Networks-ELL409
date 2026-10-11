# Architecture and Mathematical Formulation

## 1. Design objective

The project implements the corrective tree discussed in class without reducing
each node to a linear threshold. The root and every corrective node are small
neural networks. The implementation preserves three properties of the original
construction:

1. growth is driven by residual training errors;
2. a child has a fixed corrective direction;
3. existing nodes are frozen while a new node is fitted.

The resulting model is best described as a **stage-wise additive corrective
tree**. The ancestry is the sequence in which residual-confusion nodes are
created. All retained corrections are evaluated at inference. This differs from
a decision tree that executes exactly one hard path, and that distinction is
important when interpreting parameter count and latency.

## 2. Shared feature trunk

For an RGB image \(x\), a shared convolutional trunk computes

\[
z(x)=\phi_\theta(x)\in\mathbb{R}^{4w},
\]

where \(w\) is the configurable trunk width. The trunk contains:

- a \(3\times3\) convolution, GroupNorm, and SiLU stem;
- a width-\(w\) two-convolution block;
- a width-\(2w\) block whose first convolution has stride two;
- a width-\(4w\) block whose first convolution has stride two;
- global average pooling.

Each convolutional block uses two \(3\times3\) convolutions. GroupNorm is used
instead of BatchNorm because late corrective nodes can be associated with very
few errors; batch statistics are then noisy or undefined. Before global
pooling, one map location has a theoretical \(25\times25\) receptive field on a
\(32\times32\) input. Global pooling lets a head aggregate evidence across the
whole image while retaining a compact embedding.

The current implementation freezes this trunk after root training and caches
all embeddings. This makes node growth cheap and isolates the effect of added
heads. It also imposes a real limitation: no child can separate examples that
the frozen representation maps to nearly the same point. That limitation is a
central experimental question, not an implementation accident.

## 3. Root node

The root is a two-layer neural classifier

\[
L_0(x)=W_2\,\sigma(W_1\,\operatorname{LN}(z(x))+b_1)+b_2,
\]

where \(\sigma\) is SiLU and \(L_0(x)\in\mathbb{R}^C\). Root optimization uses
ordinary cross-entropy. A separate root-accuracy target deliberately stops root
training before complete interpolation, leaving structured residual errors for
the tree. This avoids the vacuous case in which an overpowered root memorizes
the finite sample and no children are ever created.

## 4. Multiclass corrective nodes

Suppose the current model most frequently predicts class \(p\) when the true
class is \(t\). A child is created for the ordered confusion pair \((t,p)\).
Its neural network produces two scalars:

\[
(u_k(x),v_k(x))=
f_k(z(x);\theta_k).
\]

These become a non-negative magnitude and a soft gate:

\[
m_k(x)=\operatorname{softplus}(u_k(x)),\qquad
g_k(x)=\operatorname{sigmoid}(v_k(x)),
\]

\[
a_k(x)=m_k(x)g_k(x)\ge 0.
\]

The child correction is

\[
\Delta_k(x)=a_k(x)(e_t-e_p),
\]

where \(e_c\) is the class-\(c\) standard basis vector. Consequently,

\[
\Delta_{k,t}(x)\ge0,\qquad
\Delta_{k,p}(x)\le0,\qquad
\sum_c\Delta_{k,c}(x)=0.
\]

The sign constraint is structural rather than a soft regularizer. A child can
only raise the missed class and lower the falsely predicted class. The complete
model after \(K\) stages is

\[
L_K(x)=L_0(x)+\sum_{k=1}^{K}\Delta_k(x).
\]

For binary Cats-vs-Dogs classification, the two possible ordered pairs recover
the positive and negative correction branches in the original formulation.
For CIFAR-10, ordered confusion pairs provide a sparse alternative to ten
independent one-vs-rest trees.

## 5. Learned routing

The target used to train a new gate is

\[
r_i=\mathbb{1}
\left[y_i=t\ \land\
\arg\max_c L_{k-1,c}(x_i)=p\right].
\]

This target is computed only during training. At inference, \(y\) is unknown,
so the network uses \(g_k(x)\). This avoids a common conceptual error: using
ground-truth error categories as inference-time routes would leak the answer.

The final layer of a new node is initialized with zero weights and biases of
\(-4\). Therefore its initial correction is close to zero and the old model is
approximately nested inside the expanded model. The training procedure also
stores the best node state by training accuracy and rejects a node if no epoch
improves on the pre-expansion accuracy.

## 6. Stage-wise objective

For a candidate child, all old parameters are frozen. Let
\(\ell_i=\operatorname{CE}(L_{k-1}(x_i)+\Delta_k(x_i),y_i)\). Positive route
examples are rare, so classification and gate losses are reweighted. If
\(w_+=\min(N_-/N_+,100)\), the objective is

\[
\mathcal{J}_k=
\frac1N\sum_i [1+r_i(w_+-1)]\ell_i
+\lambda_g\operatorname{BCE}_{w_+}(g_k(x_i),r_i)
+\lambda_s\frac1N\sum_i g_k(x_i).
\]

The first term optimizes corrected classification, the second teaches
label-free routing, and the third discourages a child from firing everywhere.
The cap on \(w_+\) prevents one very small residual set from producing
unbounded gradients.

Stage-wise fitting addresses a credit-assignment failure observed in
end-to-end additive systems: if all nodes move together, the root can become
lazy and children lose an identifiable purpose. Freezing makes the statement
“node \(k\) was added for \(p\to t\)” empirically auditable.

## 7. Interpolation and double descent

Interpolation is defined before running experiments:

\[
\widehat R_{\mathrm{train}}\le 0.001,
\]

equivalently training accuracy of at least \(99.9\%\). Capacity sweeps vary the
trunk width and scale root/node hidden dimensions proportionally. Every plotted
x-coordinate uses the measured total parameter count, not the width label.

A non-monotone test curve is not automatically double descent. The report
generator requires:

1. at least three independent seeds;
2. a measured interpolation capacity for a seed;
3. lower test error at some pre-interpolation capacity;
4. lower test error again at a larger post-interpolation capacity;
5. agreement from at least two-thirds of seeds.

The noisy-label diagnostic changes only training labels. Test labels remain
clean. This can sharpen the interpolation threshold, but noisy and clean
experiments must be identified separately.

## 8. Receptive-field visualization

The effective receptive field (ERF) of a scalar node activation \(q(x)\) is

\[
E_{uv}(x)=\frac13\sum_{c=1}^{3}
\left|\frac{\partial q(x)}{\partial x_{cuv}}\right|.
\]

For the root, \(q\) is the largest root logit. For a child, \(q=a_k=m_kg_k\).
Maps are independently normalized to \([0,1]\). The root uses a fixed sample.
Each child uses the maximum-gate example among the first 256 test examples.
This predeclared rule avoids manually selecting whichever image gives the most
attractive explanation and ensures a child is visualized where it actually
responds.

An ERF is a sensitivity map, not a causal explanation. A concentrated
background map is evidence consistent with spurious memorization, but a
background-replacement intervention would be needed for a causal claim.

## 9. Network minimization

After growth, the implementation greedily tests removing each node. A removal
is accepted if training accuracy remains above a predeclared floor. Iteration
continues until no further child can be removed. Both unpruned and retained
parameter counts are saved.

This algorithm finds a locally irredundant subset, not a globally minimal tree.
Its value is operational: it asks whether the apparent benefit of depth
survives deletion, and it exposes correction nodes whose contribution was
subsumed by later nodes.

## 10. Extensions

### Regression

For regression, replace confusion pairs by signed residual groups. A node may
predict \(a_k(x)\ge0\) and use a fixed sign \(s_k\in\{-1,+1\}\):

\[
\hat y_K(x)=\hat y_0(x)+\sum_k s_k a_k(x).
\]

Positive-residual and negative-residual subsets play the roles of false
negatives and false positives. Squared, Huber, or quantile loss can be used.

### Node-specific adapters

When the shared trunk prevents interpolation, a child can receive a small
depthwise-separable convolutional adapter or low-rank residual update. This
adds representation diversity without copying the entire trunk and offers a
direct experiment on the cost of specialization.
