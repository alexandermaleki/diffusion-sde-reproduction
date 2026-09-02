# Reproducing Score-Based Diffusion Models

<p align="center">
  <img src="assets/oxford-flowers-ema-samples.png" alt="EMA-generated Oxford Flowers samples" width="700">
</p>

The original paper, Song et al.'s *Score-Based Generative Modeling through
Stochastic Differential Equations*, trains on large-scale datasets,
high-capacity models, and thousands of sampling steps. Our setup was
considerably more constrained. Rather than aim to match their results
directly, we treated that constraint as the basis for a different question:
how much of the framework's behavior holds up under a substantially reduced
budget?

The full report is [here](./report/report.pdf).

## The setup

Two SDE families - variance-preserving (VP) and variance-exploding (VE) -
across three score-network backbones and two datasets, with and without
augmentation. Eight trained configurations in total.

| Dataset | Backbone | SDE | Params | Resolution | Train / Test |
|---|---|---|---|---|---|
| CIFAR-10 Bird | U-Net (custom) | VP | 62.14M | 32×32 | 5,000 / 1,000 |
| CIFAR-10 Bird | DDPM++ | VP | 10.09M | 32×32 | 5,000 / 1,000 |
| CIFAR-10 Bird | NCSN++ | VE | 62.76M | 32×32 | 5,000 / 1,000 |
| Oxford Flowers 102 | DDPM++ | VP | 9.89M | 48×48 | 1,020 / 6,198 |

DDPM++ and NCSN++ are adapted from `score_sde_pytorch`; the U-Net is a
from-scratch encoder-decoder baseline. Every model trained for 1,760 epochs
(AdamW, lr 1e-4, batch size 128), with an EMA copy of the weights tracked
alongside the raw ones throughout.

## The numbers

Test-set FID and the nearest-neighbor gap - mean pixel-MSE distance to the
nearest *test* image minus the nearest *train* image. Positive means
generated samples sit closer, on average, to what the model trained on:

| Dataset | Backbone | SDE | Aug. | Test FID ↓ | NN gap |
|---|---|---|---|---|---|
| CIFAR-10 Bird | U-Net | VP | No | 59.30 | 0.01342 |
| CIFAR-10 Bird | U-Net | VP | Yes | 71.56 | 0.00203 |
| CIFAR-10 Bird | DDPM++ | VP | No | 63.18 | 0.00318 |
| CIFAR-10 Bird | DDPM++ | VP | Yes | 73.85 | 0.00040 |
| CIFAR-10 Bird | NCSN++ | VE | No | 62.95 | 0.00227 |
| CIFAR-10 Bird | NCSN++ | VE | Yes | 68.88 | 0.00088 |
| Oxford Flowers | DDPM++ | VP | No | 96.01 | 0.01927 |
| Oxford Flowers | DDPM++ | VP | Yes | 128.44 | -0.01248 |

## Quality vs. Memorization Trade-off

<p align="center">
  <img src="assets/generated-vs-nearest-neighbors.png" alt="Generated samples vs. nearest training/test neighbors" width="700">
</p>

EMA weights improved FID in every single case where both raw and EMA were
evaluated. Taken alone, that would suggest just always use EMA and move on -
except the *best* FID in the whole study (52.12, non-augmented VP U-Net) came
from the model with the strongest memorization signal by a wide margin: its
mean nearest-neighbor distance was roughly an order of magnitude below every
other configuration, meaning a lot of its "generated" birds were close to
being training-set copies.

The figure above makes this concrete - left column is generated, middle is
the nearest training image, right is the nearest test image. For the
non-augmented U-Net, generated and train are nearly identical (MSE as low as
0.0016). The augmented Oxford Flowers model, by contrast, sits much further
from both train and test - worse FID, but far less evidence of copying.

Augmentation reliably pushed samples away from the training set (for NCSN++,
mean NN MSE roughly doubled), but its effect on FID was inconsistent -
slightly better for NCSN++, worse for DDPM++ and the U-Net. The best
practical trade-off we found was the augmented VE-SDE NCSN++ with EMA: FID
58.32, without the memorization red flag.

## Corrector-Step Ablation

<p align="center">
  <img src="assets/corrector-step-ablation.png" alt="FID and samples across 0-4 Langevin corrector steps" width="700">
</p>

Standard sampling here used 1,000 Euler-Maruyama predictor steps. We also
tried adding 1-4 Langevin corrector steps on top, to see if it would sharpen
results. It didn't: 0 corrector steps gave the best FID, and the generated
birds look essentially the same regardless of how many correction passes ran.
More sampling machinery isn't free - it just wasn't the bottleneck here.

## Notebooks

| What it covers | Notebook(s) |
|---|---|
| FID + memorization aggregation, corrector-step ablation | `evaluation_fid_memorization.ipynb` |
| Sampling-step sweep, 100–1000 steps | `raw_sampling_step_sweep.ipynb` |
| VE-SDE / NCSN++, Birds | `ve_sde_bird_ncsnpp_non_augmented.ipynb`, `ve_sde_bird_ncsnpp_augmented.ipynb` |
| VP-SDE / DDPM++, Birds | `vp_sde_bird_ddpmpp_non_augmented.ipynb`, `vp_sde_bird_ddpmpp_augmented.ipynb` |
| VP-SDE / U-Net, Birds | `vp_sde_bird_unet_non_augmented.ipynb`, `vp_sde_bird_unet_augmented.ipynb` |
| VP-SDE / DDPM++, Flowers | `vp_sde_flowers_ddpmpp_non_augmented.ipynb`, `vp_sde_flowers_ddpmpp_augmented.ipynb` |

Each notebook is built for Colab with a GPU runtime and Drive mounted; update
the Drive path near the top if your layout differs. The DDPM++ and NCSN++
notebooks pull in `score_sde_pytorch` as part of setup.

## Repo layout

```
.
├── assets/       # Images used in this README
├── notebooks/    # The 10 notebooks above
├── report/       # Full write-up (report.pdf)
├── results/      # Per-run FID + memorization logs
├── src/          # Shared SDE kernels, configs, EMA, plotting helpers
├── requirements.txt
└── README.md
```

## Limitations

This is a compute-limited reproduction, not a state-of-the-art result.

The eight runs vary in more than one way at once. For example, the two
VP-SDE models differ a lot in size (10M vs. 62M parameters), so we can't
fully separate the effect of model size from the effect of the SDE type.
FID numbers should always be checked against the nearest-neighbor scores
too, not read alone. Full math, the sampling algorithm, and more result
images are in the report.
