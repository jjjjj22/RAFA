# RAFA: Rarity-Aware Feature Adaptation for Long-Tailed Recognition

**A lightweight, plug-and-play feature adapter for CNN and Transformer classifiers.**

> **Release status:** Research-code consolidation of the RAFA core and available project scripts. A running pipeline is provided for training, evaluation, paired comparisons and diagnostics. The reconstructed baseline integrations are **not guaranteed to reproduce the published paper numbers**. See [implementation scope](docs/IMPLEMENTATION_SCOPE.md) before citing benchmark results.

中文使用说明：[README_zh.md](README_zh.md).

## 1. Architecture

For backbone feature \(z\in\mathbb R^D\), the Feature-conditioned Soft Rarity Router (FC-SRR) uses **train-only** class frequencies and predictions:

\[
\rho_c=\frac{\log n_{\max}-\log n_c}{\max(\log n_{\max}-\log n_{\min},\epsilon)},\quad
p_c(x)=\operatorname{softmax}(W_r\operatorname{LN}(z)/\tau+b_r/\tau)_c
\]

\[
r(x)=\sum_c p_c(x)\rho_c,\quad
\gamma(x)=\sigma(b_g+\operatorname{Softplus}(s_g)\,r(x))
\]

\[
z'=z+\gamma(x)W_{\mathrm{up}}\operatorname{GELU}(W_{\mathrm{down}}\operatorname{LN}(z)).
\]

Training includes an auxiliary **Balanced Softmax** router objective with weight \(\lambda_r=0.1\). Its adjusted logits are used **only in the auxiliary loss**. At inference RAFA does not use any ground-truth label or test-set class-frequency statistics.

Default parameters: rank = 16, router temperature = 1.0, gate bias = -2.0, gate raw slope = 1.85, zero-initialized up-projection.

For a 100-class ResNet-18 (512-D features), the adapter introduces **68,710 trainable parameters** and about **67,584 additional linear-projection MACs per image** (not total model MACs). Those are determined by the architecture; timing must be measured on the actual deployment hardware.

## 2. Setup

Python 3.10+; recommended PyTorch with a matching CUDA build for your GPU.

```bash
pip install -r requirements.txt
# or pip install -e .
```

All commands run in the repository root. A CPU sanity check is available:

```bash
python scripts/profile_model.py --backbone resnet18 --classes 100
python -m unittest discover -s tests -v
```

## 3. Prepare datasets ONCE

The generated split manifest **must be reused** for every baseline and +RAFA run. Dataset images are not included.

### CIFAR-100-LT (IF = 100)

Extract the original CIFAR-100 Python archives:

```text
/path/to/cifar-100-python/
    train
    test
    meta
```

```bash
python prepare_data.py --dataset cifar100lt --root /path/to/cifar-100-python \
    --output splits/cifar100lt_if100_seed42.json --imb-factor 100 --seed 42 --val-ratio 0.1
```

The original CIFAR test set stays untouched. The training pool is subsampled exponentially; the validation pool is taken from the selected training pool.

### DLCPD-25 / PlantWild v2 (natural long-tailed datasets)

Folder layout with class names as directories:

```text
/path/to/PlantWildV2/
    class_0/001.jpg
    class_0/002.jpg
    class_1/003.jpg
    ...
```

Alternatively supply **existing** `train/val/test/class_name/*.jpg` directories, which will be preserved. If not already split, the script builds seeded stratified splits (72:8:20).

```bash
python prepare_data.py --dataset folder --root /path/to/PlantWildV2 \
    --output splits/plantwild_v2_seed42.json --seed 42

python prepare_data.py --dataset folder --root /path/to/DLCPD25 \
    --output splits/dlcpd25_seed42.json --seed 42
```

**Important:** This script discovers classes from the actual folder. Check the class count and preserve the exact manuscript protocol; the name of a dataset alone does not identify a unique local split.

## 4. Train RAFA or the baseline

```bash
# Cross-entropy baseline
python train.py --manifest splits/cifar100lt_if100_seed42.json \
    --method ce --variant baseline --backbone resnet18 \
    --pretrained --epochs 50 --batch-size 16 --output runs/ce_baseline

# Same backbone, same training split and seed, now +RAFA
python train.py --manifest splits/cifar100lt_if100_seed42.json \
    --method ce --variant rafa --backbone resnet18 \
    --pretrained --epochs 50 --batch-size 16 --output runs/ce_rafa

# Balanced Softmax / Balanced Contrastive Learning
python train.py --manifest splits/cifar100lt_if100_seed42.json \
    --method bs --variant rafa --pretrained --output runs/bs_rafa
python train.py --manifest splits/cifar100lt_if100_seed42.json \
    --method bcl --variant rafa --pretrained --output runs/bcl_rafa
```

Other integrated modes: `bbn`, `ride`, `dbm`, `h2t` (see limitations below). Backbones: `resnet18`, `resnet34`, `resnet50`, `convnext_tiny`, `vit_b_16`, `swin_t`. TorchVision ViT uses image size 224.

Training uses AdamW + cosine learning-rate scheduling. Default epochs: 50; default LR: `1e-4` (ViT: `2e-5`), weight decay `1e-4`; seed 42. Model weights and metrics are saved to the supplied run folder.

## 5. Paired comparison for multiple methods/backbones/seeds

```bash
# CE/BS/BCL paired experiments on ResNet-18 with independent seeds
python compare.py --manifest splits/cifar100lt_if100_seed42.json \
    --methods ce,bs,bcl --backbones resnet18 --seeds 42,43,44 \
    --output runs/cifar_baselines --pretrained --epochs 50

# Adapted expert / modern-method integrations (READ SCOPE NOTES FIRST)
python compare.py --manifest splits/plantwild_v2_seed42.json \
    --methods bbn,ride,dbm,h2t --backbones resnet18 \
    --seeds 42 --output runs/plantwild_experiments --pretrained
```

The latter group contains **reference integrations** and is not interchangeable with the exact training code used to obtain reported baseline paper metrics. In particular, the `dbm` and `h2t` modules are **approximate research examples**, not author-exact reimplementations. Replace them with your original method scripts for exact reproduction.

## 6. Evaluate and analyze

```bash
python evaluate.py --checkpoint runs/ce_rafa/best.pth \
    --manifest splits/cifar100lt_if100_seed42.json \
    --split test --output runs/ce_rafa/test_results

python scripts/profile_model.py --backbone resnet18 --classes 100 \
    --rank 16 --benchmark --device cuda

python scripts/router_diagnostics.py --checkpoint runs/ce_rafa/best.pth \
    --manifest splits/cifar100lt_if100_seed42.json \
    --output runs/router_reliability
```

`router_diagnostics.py` reports Spearman correlation between predicted rarity and **ground-truth class rarity (evaluation only)**, mean gate per shot group, few-to-head misrouting rate, and soft/hard/oracle diagnostic prediction accuracies. Ground-truth labels are never passed into the deployed inference path.

Outputs:

```text
runs/<experiment>/
    best.pth                  # Best validation-Macro-F1 checkpoint
    history.csv
    run_config.json
    training_summary.json
    group_assignments.csv
    test_results/
        metrics.json
        per_class.csv
        predictions.csv
```

## 7. Mean ± standard deviation (REAL runs only)

```bash
python scripts/aggregate_seeds.py --runs runs/cifar_baselines \
    --min-seeds 3 --output runs/cifar_mean_sd.csv
```

This script aggregates **actual independent seed runs**, using sample standard deviation (`ddof=1`). The script does not invent any SD values. Recalculate the mean and gain based on real repeated runs; previous illustrative SD tables should **not** be posted as measurements.

## 8. Code layout

```text
rafa/
    model.py            # The paper's core FC-SRR + gated low-rank adapter
    ensemble.py         # Project-adapted BBN/RIDE shared expert architectures
    variants.py         # Reference DBM/H2T integrations (not author exact)
    losses.py           # Balanced Softmax, Balanced Contrastive Learning
    data.py             # Reproducible CIFAR100-LT and natural ImageFolder splits
    metrics.py          # Many/Medium/Few and Macro-F1/accuracy
    utils.py
prepare_data.py         # Create a frozen manifest
train.py                # Train baseline or +RAFA
compare.py              # Paired comparisons
evaluate.py             # Evaluate checkpoints
scripts/
    aggregate_seeds.py
    profile_model.py
    router_diagnostics.py
tests/                   # Unit and smoke tests
configs/                 # Reference settings
```

## 9. Notes on research integrity

- Never use test labels to compute rarity, gate values, or update the network.
- Do not claim that `dbm_style_loss` or reference `H2TVariant` reproduces the original DBM/H2T published algorithms; the original specialized source code must be integrated for a strict reproduction.
- Store original seed, split and exact training recipe alongside all published results.
- When publishing benchmark tables, use genuine training-run means and measured standard deviations; do not copy synthetic placeholder SDs.
- No dataset files or trained weights are shipped.

## Citation

If you use this code, please cite the associated RAFA paper after its official publication. A BibTeX entry can be added after the paper receives stable bibliographic information.

## 10. Component ablation entry points

All ablations share the full RAFA network class and differ only in the
specified training/forward switches; these are code-ready experimental
configurations, not a claim that historical Table 4 values have been rerun:

```bash
# Full RAFA
python train.py --manifest splits/cifar100lt_if100_seed42.json --method ce --variant rafa --ablation full --output runs/ablation/full

# Router-only supervision, no feature correction
python train.py --manifest splits/cifar100lt_if100_seed42.json --method ce --variant rafa --ablation router_only --output runs/ablation/router_only

# Ungated or global-gated adapter without router auxiliary objective
python train.py --manifest splits/cifar100lt_if100_seed42.json --method ce --variant rafa --ablation ungated_adapter --output runs/ablation/ungated
python train.py --manifest splits/cifar100lt_if100_seed42.json --method ce --variant rafa --ablation global_gated --output runs/ablation/global_gate

# Disable adaptive gating (retain router supervision); disable auxiliary loss
python train.py --manifest splits/cifar100lt_if100_seed42.json --method ce --variant rafa --ablation no_gating --output runs/ablation/no_gate
python train.py --manifest splits/cifar100lt_if100_seed42.json --method ce --variant rafa --ablation no_aux --output runs/ablation/no_aux

# Hard vs soft router (training intervention)
python train.py --manifest splits/cifar100lt_if100_seed42.json --method ce --variant rafa --ablation hard_router --output runs/ablation/hard_router
```

Rank, temperature and auxiliary loss weight can also be varied with `--rank`,
`--router-temperature` and `--aux-weight`.

The reference implementation of a gate-off / ungated configuration keeps the
unused router weights allocated for controlled architecture toggles. For
parameter-matched comparisons, verify active and trainable parameter counts.
