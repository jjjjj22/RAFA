"""Train-only Many/Medium/Few assignment and honest macro metrics."""
import numpy as np


def shot_groups(train_counts, method='auto'):
    counts = np.asarray(train_counts, dtype=float)
    if counts.ndim != 1 or len(counts) < 3 or (counts <= 0).any():
        raise ValueError('Need >=3 classes with nonzero training counts')
    if method == 'auto':
        method = 'threshold' if len(counts) == 100 else 'knee'
    if method == 'threshold':
        return np.where(counts > 100, 'Many', np.where(counts >= 20, 'Medium', 'Few')).tolist()
    order = np.argsort(-counts, kind='stable')
    c = len(counts)
    if method == 'tertile' or counts.max()/counts.min() < 1.25:
        a, b = c//3, 2*c//3
    elif method == 'knee':
        # Piecewise-linear least squares on descending log class counts,
        # with a small, fixed tie-break favoring non-degenerate groups.
        y = np.log(counts[order])
        x = np.arange(c, dtype=float)
        pref = lambda q: np.r_[0., np.cumsum(q)]
        s = [pref(q) for q in (x, y, x*x, y*y, x*y)]
        def error(lo, hi):
            m = hi-lo
            xx, yy, x2, y2, xy = (q[hi]-q[lo] for q in s)
            sxx = x2-xx*xx/m
            syy = y2-yy*yy/m
            sxy = xy-xx*yy/m
            return max(0., syy-sxy*sxy/max(sxx,1e-12))
        minsz = 2 if c >= 6 else 1
        best = (float('inf'), c//3, 2*c//3)
        for i in range(minsz, c-2*minsz+1):
            left = error(0, i)
            for j in range(i+minsz, c-minsz+1):
                score = left+error(i,j)+error(j,c)+1e-8*((i-c/3)**2+(j-2*c/3)**2)
                if score < best[0]: best = score, i, j
        _, a, b = best
    else:
        raise ValueError(f'Unknown grouping {method}')
    labels = [''] * c
    for rank, cls in enumerate(order):
        labels[int(cls)] = 'Many' if rank < a else 'Medium' if rank < b else 'Few'
    return labels


def compute_metrics(truth, prediction, num_classes, groups):
    y = np.asarray(truth, dtype=int)
    p = np.asarray(prediction, dtype=int)
    if y.shape != p.shape or y.size == 0:
        raise ValueError('Mismatched/empty predictions')
    cm = np.bincount(y*num_classes+p, minlength=num_classes*num_classes).reshape(num_classes,num_classes)
    support = cm.sum(1)
    predicted = cm.sum(0)
    true_pos = np.diag(cm)
    recall = np.divide(true_pos, support, out=np.full(num_classes, np.nan), where=support>0)
    precision = np.divide(true_pos, predicted, out=np.zeros(num_classes), where=predicted>0)
    recall0 = np.nan_to_num(recall)
    f1 = np.divide(2*precision*recall0, precision+recall0, out=np.zeros(num_classes), where=(precision+recall0)>0)
    present = support>0
    result = {'overall_acc':float(100*true_pos.sum()/support.sum()),
              'macro_f1':float(100*f1[present].mean()),
              'macro_recall':float(100*np.nanmean(recall)),
              'num_samples':int(y.size), 'present_classes':int(present.sum())}
    for name in ['Many','Medium','Few']:
        ids = np.array([i for i,g in enumerate(groups) if g==name], dtype=int)
        vals = recall[ids]
        result[name.lower()+'_acc'] = float(100*np.nanmean(vals)) if np.isfinite(vals).any() else None
        result[name.lower()+'_classes'] = int(len(ids))
    per_class = [{'class_index':i,'group':groups[i],'support':int(support[i]),
                  'correct':int(true_pos[i]),'accuracy':float(100*recall[i]) if present[i] else None,
                  'precision':float(100*precision[i]),'f1':float(100*f1[i])} for i in range(num_classes)]
    return result, per_class
