# PixelCNN research code for Ising and Wegner models

Masked-convolution neural-network code for autoregressive lattice sampling. The runnable training entry point in this snapshot targets the three-dimensional Ising model. Shared modules also contain other Ising and gauge-model energy functions; separate end-to-end drivers for those experiments are not included.

| File | Role in the code |
|---|---|
| `3d_ising_pixel.py` | Lightning training driver, sequential spin sampling, variational loss and effective-sample-size logging. |
| `gate_pixel.py` | Residual masked-convolution network used by the current driver. |
| `gate_ising.py` | Alternative multi-stack masked-convolution implementation. |
| `py_func.py` | Network components and Ising/gauge energy helpers. Import this module; it is not a standalone command. |

The sampler fills lattice sites sequentially and calls the network repeatedly. This repository does not contain saved quantitative benchmarks or an executable MCMC comparison, so no sampling-speed or accuracy result is claimed here.

## How to run

Create an environment from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python 3d_ising_pixel.py --help
```

Requirements follow the actual imports: NumPy, PyTorch, both Lightning import namespaces used by the driver, and TensorBoard for logging. Choose a PyTorch build compatible with your GPU and driver before installing the remaining requirements if necessary.

The training driver explicitly uses the GPU accelerator. Pass `--dev` explicitly, because its current default is inconsistent with the list handling in the script. The example already present in the source comments is:

```bash
python 3d_ising_pixel.py --beta 0.22 --batch 1500 --L 8 --epochs 1 --lr 0.003 --dev 0
```

This is an example configuration, not a benchmark or a validated physics result. The driver creates experiment logs and may write checkpoints and observable files in the working directory. Review its epoch-dependent save conditions before a longer run. It does not offer a CPU training option in its current command-line interface.

## Validation and limitations

There is no automated test suite in this snapshot. Syntax, imports and command-line help can be checked without training. Full training and physics validation require a suitable GPU environment and were not run during this cleanup. Model code is unchanged.

TODO: establish a pinned GPU training environment, add numerical and sampling checks, and save reproducible observables before reporting performance or physics accuracy.

## Related publication

[Hierarchical autoregressive neural networks in three-dimensional statistical system](https://arxiv.org/abs/2503.08610), Piotr Białas, Vaibhav Chahar, Piotr Korcyl, Tomasz Stebel, Mateusz Winiarski and Dawid Zapolski. *Computer Physics Communications* **318** (2026), 109892. [DOI](https://doi.org/10.1016/j.cpc.2025.109892).

TODO: confirm the exact relationship between this repository snapshot and the published implementation before labelling this as the paper's reproduction code.
