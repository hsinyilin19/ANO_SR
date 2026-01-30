# Quantum Super-Resolution with Adaptive Non-Local Observables (ANO-VQC)

This repository hosts the code for the paper [Quantum Super-Resolution by Adaptive Non-Local Observables (ICASSP 2026)](https://arxiv.org/abs/2601.14433) by Hsin-Yi Lin, Huan-Hsin Tseng, Samuel Yen-Chi Chen, and Shinjae Yoo.


## Overview

We propose a novel quantum framework for image super-resolution using Variational Quantum Circuits (VQCs) with **Adaptive Non-Local Observables (ANO)**. Unlike conventional VQCs that use fixed Pauli measurements, our approach treats the Hermitian measurement operators as trainable parameters, enabling:

- **Enhanced expressivity** through learnable multi-qubit observables
- **Richer feature extraction** by exploring broader Hilbert space subspaces
- **Resource-efficient scaling** without requiring deeper circuits


## Requirements

- Python ≥ 3.8
- PyTorch ≥ 2.0
- PennyLane ≥ 0.30
- lpips
- scikit-image
- torchvision
- tqdm
- matplotlib
- numpy

## Quick Start

### Training

```bash
# Train with default settings (4×4 → 12×12, 2-local observables)
python ANO_SR_train.py

```

### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--input-size` | 4 | Input image size (LR) |
| `--output-size` | 12 | Output image size (HR) |
| `--n-local` | 2 | Locality of observables (2 or 3) |
| `--lr` | 0.01 | Learning rate for linear layer |
| `--lr-H` | 0.1 | Learning rate for Hermitian parameters |
| `--batch-size` | 100 | Training batch size |
| `--epochs` | 20 | Number of training epochs |
| `--mse-weight` | 0.3 | Weight for MSE loss |
| `--lpips-weight` | 0.7 | Weight for LPIPS loss |



## Method

### ANO-VQC Architecture

The model consists of three stages:

1. **Encoding**: Low-resolution input is embedded into an n-qubit Hilbert space
2. **Variational Transformation**: Parameterized quantum gates explore the state space
3. **Adaptive Measurement**: Trainable k-local Hermitian observables extract HR features

### Trainable Hermitian Observables

For k-local observables on K = 2^k dimensions:

```
H(ϕ) = | c₁₁    a₁₂+ib₁₂  ...  a₁ₖ+ib₁ₖ |
       | *      c₂₂       ...  a₂ₖ+ib₂ₖ |
       | *      *         ...  ...      |
       | *      *         ...  cₖₖ      |
```

where ϕ = (aᵢⱼ, bᵢⱼ, cᵢᵢ) are K² learnable parameters.

### Loss Function

We use a combined loss balancing pixel-level accuracy and perceptual quality:

```
L(θ, ϕ) = c₁ · MSE + c₂ · LPIPS
```

