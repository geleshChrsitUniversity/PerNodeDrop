# PerNeuronDropDense

A drop-in Keras `Dense` layer that regularizes networks by perturbing
**individual weight connections**, per training sample, without the
custom bookkeeping or CUDA kernels that per-connection perturbation
usually requires.

> **Patent notice.** This layer implements an idea closely related to
> *PerNodeDrop*, for which a provisional patent has been filed by
> Chemophilic Data Sage. See [References](#references) below. This
> repository is provided for research/educational purposes; the MIT
> license below covers the code in this repo only — it does not grant,
> and cannot grant, any rights under that patent. If you intend to use
> this technique commercially, get your own legal advice on the patent
> status first.

## Why not just `Dropout`, or DropConnect?

The useful way to compare these methods is to ask: **who owns the mask,
and at what granularity?**

| Method | Mask lives on | Where it's applied | Consequence |
|---|---|---|---|
| **Dropout** / **GaussianDropout** | A separate layer, between two `Dense` layers | The *activation* leaving the upstream layer | Every downstream neuron sees the same perturbed activation from a given upstream neuron — the perturbation is shared across all receivers. That sharing can push receiving neurons toward learning correlated, redundant representations of it. |
| **DropConnect** ([Wan et al., 2013](https://cs.nyu.edu/~wanli/dropc/dropc.pdf)) | The *transmitting* layer, over its full `Nin × Nout` weight matrix | The *weight*, before the matmul | Theoretically the richest option — a genuinely different stochastic realization per connection, per sample. But the mask is an `Nin × Nout` object that must exist before the matmul can run, so exact per-sample DropConnect has real memory and bookkeeping costs, and fast implementations traditionally needed custom kernels. |
| **PerNeuronDropDense** (this repo) | The *receiving* neuron | The neuron's own *copy of the input*, right before its own dot product | Each output unit masks its own view of the input and reduces it independently. This is mathematically identical to masking the weight column feeding that unit — multiplication commutes — but expressed as an operation each node can do on its own, with no cross-node coordination. |

**Why the reordering matters in practice.** For output unit $u$:

$$Y_u = \sum_i M_{iu}\, x_i$$

Masking the weight, $Y_u = \sum_i (\text{mask}_{u,i} M_{iu})\, x_i$, and masking
the node's view of the input, $Y_u = \sum_i M_{iu} (\text{mask}_{u,i}\, x_i)$,
give the exact same answer. What changes is which computational unit is
responsible for applying the mask. Putting that responsibility on the
receiving node means node $u$'s operation — mask my inputs, dot with my
weight column — doesn't depend on what any other node is doing. There's
no shared `Nin × Nout` object that has to be fully assembled first. That
maps directly onto ordinary broadcast-multiply-then-reduce, which
TensorFlow/PyTorch already implement as fused, vectorized primitives —
so there's nothing here that needs a bespoke kernel.

**What this buys you, and what it doesn't.** The reordering solves an
*implementation* problem (drop-in Dense-layer semantics, plain tensor
ops, no custom CUDA) and recovers DropConnect's per-sample fidelity
(unlike the common batch-shared-mask compromise many frameworks use in
practice). It does **not** reduce the underlying memory cost: with
`fixed_mask=False`, this layer still materializes a `(batch, input_dim,
units)` tensor every forward pass, the same order of cost DropConnect
pays for a per-sample mask. In practice, though, because it's just a
broadcast-multiply feeding straight into `einsum`, it has measured
close to `Dropout`-level wall-clock training time — noticeably faster
than per-sample DropConnect (including PyTorch's batch-shared variant),
which pays extra for constructing and indexing a separate mask object
that doesn't fuse as cleanly with the matmul.

## Installation

Single-file layer, one dependency:

```bash
pip install tensorflow
```

Then drop `per_neuron_drop_dense.py` into your project, or copy the
`PerNeuronDropDense` class directly.

## Quick start

```python
import tensorflow as tf
from per_neuron_drop_dense import PerNeuronDropDense

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(784,)),
    PerNeuronDropDense(256, drop_rate=0.3, stir_type="drop", activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax"),
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
```

## API

```python
PerNeuronDropDense(
    units,
    drop_rate=0.2,
    fixed_mask=False,
    stir_type="drop",
    activation=None,
    **kwargs,
)
```

| Argument | Type | Description |
|---|---|---|
| `units` | `int` | Output dimensionality. |
| `drop_rate` | `float` in `[0, 1)` | Strength of the perturbation. `0.0` disables it (behaves as plain `Dense`). |
| `fixed_mask` | `bool` | If `True`, one mask is sampled at build time and reused on every call (train **and** inference). If `False` (default), a new mask is sampled every training step, and the layer collapses to plain `Dense` at inference. |
| `stir_type` | `"drop"` or `"gaussian"` | `"drop"`: Bernoulli mask, scaled by `1/(1-drop_rate)` (inverted dropout on weights). `"gaussian"`: multiplicative Gaussian noise, mean `1`, variance-matched to the Bernoulli case. |
| `activation` | `str` or callable | Standard Keras activation, applied after the bias add. |

Calling the layer with `return_masks=True` also returns the sampled mask
tensor, e.g. `y, mask = layer(x, training=True, return_masks=True)`.

## Repository contents

```
.
├── per_neuron_drop_dense.py         # the layer
├── notebook/
│   └── cifar10_experiments.ipynb    # CIFAR-10 benchmark notebook
├── requirements.txt
├── LICENSE
└── README.md
```

## Running the CIFAR-10 experiments

The notebook in `notebook/cifar10_experiments.ipynb` trains a small CNN on
CIFAR-10 with a `PerNeuronDropDense` layer as the penultimate (classifier
head) layer, and compares:

- a plain `Dense` + `Dropout` baseline,
- `PerNeuronDropDense` with `stir_type="drop"`,
- `PerNeuronDropDense` with `stir_type="gaussian"`,
- `PerNeuronDropDense` with `fixed_mask=True` (a frozen sub-network),

across a couple of `drop_rate` values, and plots training/validation
accuracy and loss curves for each configuration.

```bash
pip install -r requirements.txt
jupyter notebook notebook/cifar10_experiments.ipynb
```

## Notes & caveats

- With `fixed_mask=False` (the default), the mask is resampled **per
  training batch**, independently for every example in the batch — this
  means the layer allocates a full `(batch, input_dim, units)` tensor on
  every forward pass, which is more memory-hungry than standard `Dropout`
  for large layers. Keep this layer's `units` modest, or use it near the
  end of the network (as in the notebook) rather than on very wide hidden
  layers.
- At inference, a layer with `fixed_mask=False` behaves exactly like a
  plain `Dense` layer (no perturbation). A layer with `fixed_mask=True`
  keeps applying its single frozen mask at inference too — this is by
  design, but worth knowing if you expect deterministic-Dense behavior at
  test time.
- Any wall-clock comparisons in this README or the notebook are from
  informal local experiments, not a controlled benchmark suite. Numbers
  will vary by hardware, batch size, and layer width — run the notebook
  on your own setup before drawing conclusions for your use case.

## References

- L. Wan, M. Zeiler, S. Zhang, Y. LeCun, R. Fergus. ["Regularization of
  Neural Networks using DropConnect."](https://cs.nyu.edu/~wanli/dropc/dropc.pdf) ICML 2013.
- Sreeja C. S. ["PerNodeDrop: A Method Balancing Specialized Subnets and
  Regularization in Deep Neural Networks."](https://arxiv.org/abs/2512.12663)
  arXiv:2512.12663. Notes a provisional patent filed by Chemophilic Data
  Sage related to the PerNodeDrop method — see the notice at the top of
  this README.
- ["PerNodeDrop: A Practical and Efficient Alternative to DropConnect —
  Exploring Node-Owned Stochasticity."](https://github.com/keras-team/keras/discussions/23090)
  keras-team/keras Discussion #23090.

## License

MIT — see [LICENSE](LICENSE). Note the patent notice above: this license
covers the code in this repository and does not grant any rights under
the referenced patent. This work as a system and a method is under patent consideration/

## Aknoledgement
This work was encoraged and partialy funded by Chemophilic Data Sage LLP. 
