# RAFA: Rarity-Aware Feature Adaptation for Long-Tailed Recognition

**A lightweight, plug-and-play feature adapter for CNN and Transformer classifiers.**

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


Other integrated modes: `bbn`, `ride`, `dbm`, `h2t` (see limitations below). Backbones: `resnet18`, `resnet34`, `resnet50`, `convnext_tiny`, `vit_b_16`, `swin_t`. TorchVision ViT uses image size 224.

Training uses AdamW + cosine learning-rate scheduling. Default epochs: 50; default LR: `1e-4` (ViT: `2e-5`), weight decay `1e-4`; seed 42. Model weights and metrics are saved to the supplied run folder.


## 5. Evaluate and analyze

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

