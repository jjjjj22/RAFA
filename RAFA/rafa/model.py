"""RAFA core, aligned with manuscript equations (1)--(9).

The forward pass uses *no* labels. Training-frequency buffers come exclusively
from the training split. Inference uses raw router logits and probability-
weighted rarity priors, NOT the Balanced-Softmax-adjusted training logits.
"""
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models

BACKBONES = ('resnet18', 'resnet34', 'resnet50', 'convnext_tiny', 'vit_b_16', 'swin_t')


def make_backbone(name: str, pretrained: bool = False):
    choices = {
        'resnet18': (models.resnet18, models.ResNet18_Weights.DEFAULT),
        'resnet34': (models.resnet34, models.ResNet34_Weights.DEFAULT),
        'resnet50': (models.resnet50, models.ResNet50_Weights.DEFAULT),
        'convnext_tiny': (models.convnext_tiny, models.ConvNeXt_Tiny_Weights.DEFAULT),
        'vit_b_16': (models.vit_b_16, models.ViT_B_16_Weights.DEFAULT),
        'swin_t': (models.swin_t, models.Swin_T_Weights.DEFAULT),
    }
    if name not in choices:
        raise ValueError(f'Unknown backbone: {name}. Supported: {tuple(choices)}')
    build, weights = choices[name]
    model = build(weights=weights if pretrained else None)
    if name.startswith('resnet'):
        d = model.fc.in_features
        model.fc = nn.Identity()
    elif name.startswith('convnext'):
        d = model.classifier[-1].in_features
        model.classifier[-1] = nn.Identity()
    elif name.startswith('vit_'):
        d = model.hidden_dim
        model.heads = nn.Identity()
    elif name == 'swin_t':
        d = model.head.in_features
        model.head = nn.Identity()
    return model, d


class RarityAwareFeatureAdapter(nn.Module):
    """Feature-conditioned soft rarity router + monotonic gate + low-rank adapter.

    rho_c = (log(n_max) - log(n_c)) / max(log(n_max)-log(n_min), eps)
    r(x) = sum_c softmax(router(LN(z))/tau)_c * rho_c
    gate(x) = sigmoid(b_g + softplus(s_g) * r(x))
    z' = z + gate(x) * W_up GELU(W_down LN(z))
    """
    def __init__(self, feature_dim, train_counts, rank=16, temperature=1.0,
                 gate_bias=-2.0, gate_raw_slope=1.85):
        super().__init__()
        counts = torch.as_tensor(train_counts, dtype=torch.float32)
        if counts.ndim != 1 or counts.numel() < 2 or (counts <= 0).any():
            raise ValueError('Training counts must be a positive vector of >=2 classes')
        if rank < 1 or temperature <= 0:
            raise ValueError('rank and temperature must be positive')
        ln = counts.log()
        rarity = (ln.max() - ln) / (ln.max() - ln.min()).clamp_min(1e-8)
        self.register_buffer('log_train_counts', ln)
        self.register_buffer('rarity_prior', rarity)
        self.temperature = float(temperature)
        self.gate_mode = 'adaptive'  # adaptive/global/identity/off
        self.route_mode = 'soft'     # soft/hard, for routing ablations
        self.norm = nn.LayerNorm(feature_dim)
        self.router = nn.Linear(feature_dim, len(counts))
        self.down = nn.Linear(feature_dim, rank, bias=False)
        self.up = nn.Linear(rank, feature_dim, bias=False)
        self.activation = nn.GELU()
        self.gate_bias = nn.Parameter(torch.tensor(float(gate_bias)))
        self.gate_raw_slope = nn.Parameter(torch.tensor(float(gate_raw_slope)))
        nn.init.zeros_(self.up.weight)

    def route(self, features):
        h = self.norm(features)
        logits = self.router(h)
        probs = F.softmax(logits.float()/self.temperature, dim=-1)
        rarity = (probs * self.rarity_prior[None]).sum(dim=-1)
        if self.route_mode == 'hard':
            rarity = self.rarity_prior[probs.argmax(dim=-1)]
        if self.gate_mode == 'adaptive':
            gate = torch.sigmoid(self.gate_bias + F.softplus(self.gate_raw_slope)*rarity)
        elif self.gate_mode == 'global':
            gate = torch.sigmoid(self.gate_bias).expand_as(rarity)
        elif self.gate_mode == 'identity':
            gate = torch.ones_like(rarity)
        elif self.gate_mode == 'off':
            gate = torch.zeros_like(rarity)
        else:
            raise ValueError(f'Unknown gate_mode: {self.gate_mode}')
        return h, logits, probs, rarity, gate

    def forward(self, features, return_aux=False, override_rarity=None):
        if features.ndim != 2:
            raise ValueError('RAFA expects a [batch, feature_dim] feature tensor')
        h, logits, probs, rarity, gate = self.route(features)
        if override_rarity is not None:
            # Diagnostic intervention only; never use labels here in deployed inference.
            gate = torch.sigmoid(self.gate_bias + F.softplus(self.gate_raw_slope)*override_rarity)
        residual = self.up(self.activation(self.down(h)))
        corrected = features + gate.to(residual.dtype)[:, None] * residual
        if return_aux:
            return corrected, {'router_logits': logits, 'probabilities': probs,
                               'rarity': rarity, 'gate': gate, 'residual': residual}
        return corrected

    def auxiliary_loss(self, logits, labels):
        return F.cross_entropy(logits.float() + self.log_train_counts[None], labels)


class RAFAClassifier(nn.Module):
    """Single-backbone model; use_rafa=False creates the paired baseline."""
    def __init__(self, backbone, train_counts, method='ce', use_rafa=False,
                 pretrained=False, rank=16, temperature=1.0, proj_dim=128):
        super().__init__()
        self.backbone, d = make_backbone(backbone, pretrained)
        self.classifier = nn.Linear(d, len(train_counts))
        self.projection = (nn.Sequential(nn.Linear(d, d), nn.ReLU(inplace=False),
                                         nn.Linear(d, proj_dim)) if method == 'bcl' else None)
        # Construct last: paired baseline/+RAFA begin with the same backbone/head initialization.
        self.adapter = (RarityAwareFeatureAdapter(d, train_counts, rank, temperature)
                        if use_rafa else None)
        self.method = method
        self.feature_dim = d
        self.use_rafa = use_rafa

    def encode(self, x, return_aux=False):
        z = self.backbone(x)
        if self.adapter is None:
            return (z, None) if return_aux else z
        return self.adapter(z, return_aux=True) if return_aux else self.adapter(z)

    def forward(self, x, return_aux=False):
        z, aux = self.encode(x, return_aux=True)
        logits = self.classifier(z)
        return (logits, aux) if return_aux else logits

    def projected_views(self, images):
        if self.projection is None:
            raise ValueError('Projected features are only available for BCL')
        return F.normalize(self.projection(self.encode(images)).float(), dim=-1)

    def class_centers(self):
        return F.normalize(self.projection(self.classifier.weight).float(), dim=-1)


def count_parameters(model):
    return sum(int(p.numel()) for p in model.parameters())
