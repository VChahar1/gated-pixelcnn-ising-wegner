# Gated PixelCNN for the 2D/3D Ising Model and the Wegner Model

Autoregressive neural-network sampling of lattice spin and gauge configurations using a **Gated PixelCNN** architecture. The network learns the Boltzmann distribution of the model directly, enabling efficient generation of statistically independent configurations as an alternative to traditional Markov Chain Monte Carlo (MCMC).

> **[VERIFY] Relationship to publication.** This repository contains code associated with the work published as:
> *"Simulation of the Ising model using a hierarchical autoregressive network in three spatial dimensions,"* **Computer Physics Communications** 318 (2026) 109892.
> *(If this repo is a separate/earlier Gated-PixelCNN implementation rather than the exact published code, reword this line accordingly — e.g. "an earlier autoregressive approach related to the methods in...".)*

---

## Background

The **Ising model** is the canonical model of statistical physics: a lattice of spins (+1/-1) interacting with their neighbours, with configurations distributed according to the Boltzmann weight `exp(-E/T)`. Computing thermal averages requires sampling configurations from this distribution. The standard tool is MCMC (e.g. Metropolis, Wolff), which generates *correlated* samples and suffers from **critical slowing down** near phase transitions — autocorrelation times explode, making independent samples expensive.

**Autoregressive neural networks** offer an alternative. By factorizing the joint distribution over spins into a product of conditionals,

```
p(s_1, s_2, ..., s_N) = p(s_1) · p(s_2 | s_1) · p(s_3 | s_1, s_2) · ...
```

the network can **directly generate** a configuration in a single forward pass and assign it an exact probability. Combined with importance reweighting, this yields statistically independent samples and sidesteps the autocorrelation problem of MCMC.

**Gated PixelCNN** is the architecture used here to model those conditionals. Originally developed for image generation, its masked convolutions enforce the autoregressive ordering (each site depends only on previously-generated sites), and its gated activation units increase expressive power. The same machinery applies naturally to spin lattices in 2D and 3D, and is extended here to the **Wegner model** — a lattice gauge theory whose degrees of freedom live on links rather than sites.

---

## What this repository does

- Implements a Gated PixelCNN that models the conditional distributions of lattice configurations.
- Trains the network (via the variational / KL objective against the Boltzmann distribution) to sample the 2D and 3D Ising model. **[VERIFY: training objective — confirm whether you used the variational free-energy loss, maximum-likelihood on MCMC samples, or another scheme.]**
- Extends the approach to the **Wegner model** (lattice gauge theory).
- Compares sampling quality and efficiency against baseline MCMC. **[VERIFY: which baselines and metrics — e.g. autocorrelation time, free-energy estimate, magnetization.]**

---

## Repository structure

> **[VERIFY all file descriptions below]** — these are inferred from filenames; correct any that are wrong.

| File | Purpose (inferred) |
|------|--------------------|
| `3d_ising_pixel.py` | Main script for the **3D Ising** model — builds the 3D Gated PixelCNN, trains it, and samples configurations. |
| `gate_pixel.py` | The **Gated PixelCNN architecture** — masked convolutions and gated activation units that enforce the autoregressive ordering. |
| `gate_ising.py` | Ising-specific model setup — energy/Hamiltonian, Boltzmann weights, and the training loop tying the network to the Ising distribution. |
| `py_func.py` | Utility functions — observables (energy, magnetization), reweighting, I/O, and analysis helpers. |
| `README.md` | This file. |

---

## Method in brief

1. **Factorize** the joint distribution over lattice sites into autoregressive conditionals.
2. **Model** each conditional with a Gated PixelCNN, masked so site `i` sees only sites `< i`.
3. **Train** the network so its distribution matches the target Boltzmann distribution at temperature `T`. **[VERIFY objective]**
4. **Sample** configurations in a single forward pass, with exact known probabilities.
5. **Reweight / estimate** thermodynamic observables (energy, magnetization, specific heat) and compare against MCMC.

---

## Requirements

> **[VERIFY versions and framework]** — fill in what you actually used.

```
python >= 3.x
[PyTorch or TensorFlow — specify which]
numpy
matplotlib
```

```bash
pip install -r requirements.txt   # [add a requirements.txt if not present]
```

---

## Usage

> **[VERIFY commands and arguments]** — replace with the real entry points and flags.

```bash
# Train and sample the 3D Ising model
python 3d_ising_pixel.py   # [add arguments: lattice size, temperature, epochs, etc.]
```

**[Add a short example: what the script outputs, where results/figures are written, and roughly how long a run takes / what hardware — e.g. trained on JUWELS / PLGrid multi-GPU nodes.]**

---

## Results

**[Add 1-2 key results here — this is the most important section for a visitor.]**
Suggested content:
- A figure comparing the network-sampled observables (energy or magnetization vs temperature) against exact / MCMC results, showing they agree.
- The headline efficiency claim — e.g. reduced autocorrelation / sampling-efficiency gain over MCMC, and the largest lattice reached (up to 32³).
- Embed an image with: `![Energy vs temperature](figures/energy_vs_T.png)`

A single good plot here does more than any amount of prose.

---

## Citation

If you use this code, please cite:

```bibtex
@article{Chahar2026Ising,
  title   = {Simulation of the Ising model using a hierarchical autoregressive network in three spatial dimensions},
  author  = {Chahar, Vaibhav and [VERIFY co-authors]},
  journal = {Computer Physics Communications},
  volume  = {318},
  pages   = {109892},
  year    = {2026},
  doi     = {[VERIFY DOI]}
}
```

---

## References

- van den Oord et al., *Conditional Image Generation with PixelCNN Decoders* (Gated PixelCNN), 2016.
- Wu, Wang, Zhang, *Solving Statistical Mechanics using Variational Autoregressive Networks*, Phys. Rev. Lett. 2019.
- **[VERIFY]** Add the Wegner-model and any hierarchical-autoregressive references central to your paper.
