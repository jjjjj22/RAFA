# Release validation (code smoke only)

Executed in a temporary CPU environment (PyTorch 2.10.0+cpu, TorchVision 0.25.0+cpu):

- Python `compileall` for all project source files: pass.
- 5 core/data unit tests: pass.
- Tiny synthetic folder dataset with three classes (7 images each) -> frozen split manifest: pass.
- `train.py` with `ce`, `bs`, `bcl`, `bbn`, `ride`, `dbm`, `h2t` on ResNet-18, one epoch (H2T one+one stages): forward/backprop/checkpoint/validation/test passed.
- `compare.py` with CE baseline and CE+RAFA: pass.
- `scripts/router_diagnostics.py` on synthetic checkpoint: pass; corrected constant-input correlation to null.
- `scripts/profile_model.py` for 100-class ResNet-18: 11,227,812 baseline vs 11,296,522 +RAFA parameters; delta 68,710; linear projection MACs 67,584/image.
- `--ablation global_gated` one-epoch smoke: pass.

These tests **do not validate the scientific fidelity of numerical results** on CIFAR-100-LT, DLCPD-25 or PlantWild v2. They are only software consistency checks. The actual original datasets, paper model checkpoints and full multi-seed experiments were not present.
