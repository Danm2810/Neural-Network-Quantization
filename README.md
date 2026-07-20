# Systematic Analysis of Weight Quantization Thresholds in CNNs

**A dataset-dependent study of weight bitwidth vs. accuracy on MNIST and CIFAR-10.**

This repository contains the paper and the accompanying PyTorch implementation for a systematic study of how far you can compress a convolutional neural network's weights before accuracy breaks down — and how that threshold shifts with dataset complexity.

By isolating weight quantization while holding activations and gradients at full precision, this study attributes accuracy degradation to bitwidth reduction alone, without confounds from activation or gradient quantization.

---

## Key findings

- **MNIST tolerates extreme quantization.** 1-bit weights achieve **99.06%** accuracy — statistically indistinguishable from the 32-bit baseline (98.96%) — while shrinking the model **32×**.
- **CIFAR-10 has a clear precision cliff.** Accuracy degrades monotonically from 70.45% (32-bit) → 66.81% (8-bit) → 63.71% (4-bit) → 62.71% (1-bit).
- **8-bit is the sweet spot for complex tasks.** 4× compression for only 3.64% accuracy loss on CIFAR-10 — 94.8% accuracy retention.
- **Quantization tolerance is dataset-dependent, not architecture-dependent.** Same model, same training — MNIST shows −0.10% degradation at 1-bit, CIFAR-10 shows +7.74%. A **77× difference** in sensitivity.
- **The precision-accuracy tradeoff is non-linear.** CIFAR-10's 4-bit and 1-bit runs converge near a ~63% "accuracy floor," suggesting a lower bound where further precision cuts stop mattering.

## Results

| Dataset  | Bitwidth (W) | Best Test Acc (%) | Model Size (MB) | Compression |
| -------- | :----------: | :---------------: | :-------------: | :---------: |
| MNIST    | 32           | 98.96             | 2.08            | 1.0×        |
| MNIST    | 8            | 99.06             | 0.52            | 4.0×        |
| MNIST    | 4            | 98.81             | 0.26            | 8.0×        |
| MNIST    | 1            | 99.06             | 0.06            | 32.0×       |
| CIFAR-10 | 32           | 70.45             | 2.08            | 1.0×        |
| CIFAR-10 | 8            | 66.81             | 0.52            | 4.0×        |
| CIFAR-10 | 4            | 63.71             | 0.26            | 8.0×        |
| CIFAR-10 | 1            | 62.71             | 0.06            | 32.0×       |

*Reproduced from Table 1 of the paper. Activations and gradients held at 32-bit throughout.*

## Repository contents

```
.
├── nn_quantizations.pdf              # The full paper
├── Neural_Network_Quantization.ipynb # PyTorch implementation and experiments
└── README.md                         # This file
```

## Method

The implementation follows a weight-only variant of [DoReFa-Net](https://arxiv.org/abs/1606.06160) (Zhou et al., 2016):

- **k-bit quantization** (k > 1) uses `tanh` normalization followed by uniform rounding into 2ᵏ − 1 levels.
- **1-bit (binary) quantization** uses `sign(W) · mean(|W|)` — the scale-preserving binarization from [XNOR-Net](https://arxiv.org/abs/1603.05279) / DoReFa-Net.
- **Dual-precision training**: weights are stored at 32-bit, quantized before each forward pass, and updated in full precision after backprop.
- **Straight-through estimator (STE)** handles the non-differentiable quantization step by passing the incoming gradient through unchanged.

**Architecture**: `SimpleCNN`, two `QuantizableConv2d` blocks (3→32→64, 3×3 kernels) + max pooling + two FC layers (128 hidden, 10 out). ~545K parameters, all subject to the target bitwidth. Trained with Adam (lr=1e-3), batch size 128, cross-entropy loss, for 5 epochs.


Data is downloaded automatically by `torchvision.datasets` — no manual download step. Full run time on a T4 GPU is roughly 5–10 minutes for all eight configurations (4 bitwidths × 2 datasets × 5 epochs).

## Requirements

```
torch >= 1.10
torchvision
numpy
matplotlib
pandas
tqdm
```

## The Paper

The paper is included as `nn_quantizations.pdf`. It covers:

1. Motivation and prior work on network compression
2. Weight-only quantization methodology and the STE
3. Experimental setup (architecture, datasets, training procedure)
4. Results and analysis of the accuracy-compression tradeoff
5. Discussion of dataset-dependent thresholds and deployment guidance

## Citation

If you find this work useful, please cite:

```bibtex
@misc{mikhail2025quantization,
  title  = {Systematic Analysis of Weight Quantization Thresholds in Convolutional Neural Networks: A Dataset-Dependent Study},
  author = {Mikhail, Daniel},
  year   = {2025},
  note   = {University of Delaware, Mathematics Department}
}
```

## References

Key works this study builds on:

- Zhou et al. (2016), [DoReFa-Net: Training Low Bitwidth CNNs with Low Bitwidth Gradients](https://arxiv.org/abs/1606.06160)
- Courbariaux et al. (2016), [Binarized Neural Networks](https://arxiv.org/abs/1602.02830)
- Rastegari et al. (2016), [XNOR-Net](https://arxiv.org/abs/1603.05279)
- Hinton et al. (2015), [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)
- Han et al. (2015), [Deep Compression](https://arxiv.org/abs/1510.00149)

## License

MIT — see `LICENSE` for details.
