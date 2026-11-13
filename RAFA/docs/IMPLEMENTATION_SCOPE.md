# Reproducibility and implementation scope

**This repository is a consolidated release built from the RAFA manuscript and retrievable project code. It has not been verified to reproduce the paper's reported Tables 1--5 end to end.** Do not present any produced metric as a reproduced published result until the exact original protocol, splits, model initialization, hyperparameters, checkpoints and seeds have been matched.

## Verified against the available project implementation

- `rafa.model.RarityAwareFeatureAdapter`: feature LayerNorm, linear C-way router, training-count log-prior, probability-weighted rarity, monotonic sigmoid/Softplus gate, GELU low-rank correction, zero-initialized up-projection, residual output.
- Router auxiliary supervision is Balanced Softmax CE over **raw** routing logits `router_logits + log(n_c)`; forward/inference uses raw logits, not label nor test-set frequency information.
- Core CE/BS/BCL concepts and the original code's backbone and loss interface; however, original authors' BCL full recipe is **not** claimed.
- Train-only Many/Medium/Few grouping and shared split manifests.

## Adapted/reconstructed integrations

- `bbn`: project-style ResNet shared trunk and late branches, cumulative bilateral CE. Other backbones use feature-space expert heads. **Not exact original BBN architecture.**
- `ride`: project-style shared trunk and experts with diversity plus optional training-only stage-2 router. **Not exact original RIDE implementation.**
- `dbm`: **DBM-inspired approximate margin** loss, NOT the published official DBM training objective. Replace `rafa.variants.dbm_style_loss` with your actual original DBM implementation before comparing with paper results.
- `h2t`: **H2T-inspired approximate feature mixing** and two-stage training, NOT the original head-to-tail partial-map fusion. Replace `rafa.variants.H2TVariant` and associated training stage with your original H2T training code.

## Paper dataset and split fidelity

- CIFAR-100-LT: exponentially imbalanced class counts with factor 100, 10% in-training validation, balanced official test; train-only rarity priors. The integer rounding and exact seed/split manifest can affect results.
- Natural datasets: supports class folders or existing train/val/test subfolders. If no splits exist, creates deterministic stratified 72/8/20 splits. **Class labels and exact historical splits must be checked against the published experiments.**
- `shot_groups(..., 'threshold')`: Many >100, Medium 20--100, Few <20; `'knee'`: 3-piece linear fitting to log class counts.
- One run produces ONE accuracy value. Standard deviations require real independent repeats, not manufactured values. `scripts/aggregate_seeds.py` reports unbiased sample SD from real seed runs.

## What is explicitly NOT bundled

- Dataset image files (CIFAR, DLCPD-25, PlantWild v2): obtain from their owners and follow dataset terms.
- Full original 7 independent experiment repositories and exact pretrained checkpoints: not available in the uploaded/current accessible files.
- Fake benchmark tables or synthetic SDs.

## Publication checklist

- Freeze split manifest and identify train/val/test protocols in README.
- Reproduce each baseline and RAFA pair under identical settings and seeds.
- Use the original implementation/protocol for DBM and H2T comparisons.
- Run ≥3 independent seeds, report mean ± actual SD, and keep per-run logs.
- Include computational timings only if measured on recorded hardware.
- Examine router reliability using `scripts/router_diagnostics.py`.

## Component ablation interpretation

`--ablation router_only` retains the router training loss but sets residual
gating to zero. `ungated_adapter` bypasses the gate and disables the router
loss. `global_gated` learns a sample-independent gate. `no_gating` bypasses
gating but retains auxiliary router supervision. `no_aux` disables the router
loss. `hard_router` uses argmax rarity in the gate. In the reference toggles,
unused router layers may still be allocated, so report active/trainable
parameter counts carefully when evaluating fairness.

The reference `scripts/plot_fig3_from_workbooks.py` is the existing project
figure-generation script; it requires separately supplied Excel workbooks
(`IF100-results.xlsx`, `DLCPD-Results.xlsx`, `PlantWild-Results.xlsx`). Its
PlantWild ResNet-18 correction values are hard-coded based on a prior revision;
verify against final raw experiments before publication.
