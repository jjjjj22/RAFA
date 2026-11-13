"""Unified CE/BS/BCL losses. BCL follows the class-complement idea.

The BCL integration is research-compatible but not an exact reproduction of
all data augmentation, schedule and distributed training from the authors.
"""
import torch
from torch.nn import functional as F


def balanced_softmax(logits, targets, train_counts):
    counts = torch.as_tensor(train_counts, device=logits.device, dtype=torch.float32)
    return F.cross_entropy(logits.float() + counts.log()[None, :], targets)


def balanced_contrastive(views, labels, centers, temperature=0.07):
    z = F.normalize(views.float(), dim=-1)
    p = F.normalize(centers.float(), dim=-1)
    n, num_cls = z.size(0), p.size(0)
    all_z = torch.cat((z, p), dim=0)
    all_labels = torch.cat((labels, torch.arange(num_cls, device=labels.device)))
    logits = z @ all_z.T / temperature
    same = labels[:, None].eq(all_labels[None, :])
    self_mask = torch.zeros_like(same)
    self_mask[:, :n] = torch.eye(n, device=z.device, dtype=torch.bool)
    count = torch.bincount(labels, minlength=num_cls).float()
    divisors = (count[all_labels][None, :] + 1. - same.float()).clamp_min(1.)
    scaled = (logits - divisors.log()).masked_fill(self_mask, float('-inf'))
    log_denom = torch.logsumexp(scaled, dim=-1, keepdim=True)
    positive = same & ~self_mask
    log_prob = logits - log_denom
    return -(log_prob.masked_fill(~positive, 0.).sum(-1)/positive.sum(-1).clamp_min(1)).mean()
