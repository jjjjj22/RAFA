"""Offline Router reliability diagnostics; NEVER uses test labels to route the deployed model.

Works with single-backbone RAFA checkpoints (CE/BS/BCL/DBM/H2T). Evaluation
uses test labels only after forward inference to quantify routing bias.
"""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from scipy.stats import spearmanr
from rafa.data import load_manifest,make_loader
from rafa.utils import json_save,csv_save
from evaluate import load_for_evaluation


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--manifest',default=None)
    p.add_argument('--output',default='runs/router_diagnostics')
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--workers',type=int,default=0)
    p.add_argument('--device',default='auto')
    p.add_argument('--split',choices=('val','test'),default='test')
    a=p.parse_args()
    model,ckpt,device=load_for_evaluation(a.checkpoint,a.device)
    if model.adapter is None or not hasattr(model,'backbone'):
        raise ValueError('Requires a single-backbone RAFA checkpoint')
    manifest=load_manifest(a.manifest or ckpt['manifest'])
    loader=make_loader(manifest,a.split,ckpt['image_size'],a.batch_size,a.workers)
    prior=model.adapter.rarity_prior
    rows=[]
    model.eval()
    with torch.no_grad():
        for inputs,targets,names in loader:
            x=inputs.to(device)
            y=targets.to(device)
            z=model.backbone(x)
            h,logits,probs,rarity,gate=model.adapter.route(z)
            hard_class=probs.argmax(-1)
            oracle=prior[y]
            # Three counterfactual logits for diagnosis only. No gradient, no training.
            predicted={}
            for name,r in (('soft',None),('hard',prior[hard_class]),('oracle',oracle)):
                z_new=model.adapter(z,override_rarity=r)
                predicted[name]=model.classifier(z_new).argmax(-1)
            for i,(label,path) in enumerate(zip(targets.tolist(),names)):
                rows.append({'image':path,'true_class':label,
                             'true_group':ckpt['group_assignments'][label],
                             'true_rarity':float(oracle[i]),
                             'pred_rarity':float(rarity[i]),'gate':float(gate[i]),
                             'router_pred_class':int(hard_class[i]),
                             'router_correct':int(hard_class[i].item()==label),
                             **{f'{k}_correct':int(predicted[k][i].item()==label) for k in predicted}})
    true=np.asarray([r['true_rarity'] for r in rows])
    guess=np.asarray([r['pred_rarity'] for r in rows])
    corr=spearmanr(true,guess) if np.ptp(true)>0 and np.ptp(guess)>0 else None
    result={'n':len(rows),
            'spearman_rarity':float(corr.statistic) if corr is not None and np.isfinite(corr.statistic) else None,
            'spearman_p_value':float(corr.pvalue) if corr is not None and np.isfinite(corr.pvalue) else None,
            'router_top1_accuracy':float(np.mean([r['router_correct'] for r in rows]))}
    for name in ('Many','Medium','Few'):
        group=[r for r in rows if r['true_group']==name]
        if not group:continue
        result[name.lower()+'_n']=len(group)
        result[name.lower()+'_mean_rarity']=float(np.mean([r['pred_rarity'] for r in group]))
        result[name.lower()+'_mean_gate']=float(np.mean([r['gate'] for r in group]))
        if name=='Few':
            head_ids={i for i,g in enumerate(ckpt['group_assignments']) if g=='Many'}
            result['few_to_head_error_rate']=float(np.mean([r['router_pred_class'] in head_ids for r in group]))
    for name in ('soft','hard','oracle'):
        result[name+'_acc']=100*float(np.mean([r[name+'_correct'] for r in rows]))
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    json_save(out/'summary.json',result)
    csv_save(out/'per_sample.csv',rows)
    print(result)

if __name__=='__main__':main()
