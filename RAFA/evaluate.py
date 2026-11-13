"""Checkpoint evaluation and per-class metrics; no label-based gating in inference."""
import argparse
from pathlib import Path
import torch
from torch.nn import functional as F
from rafa.data import load_manifest,training_counts,make_loader
from rafa.metrics import compute_metrics
from rafa.utils import select_device,json_save,csv_save,torch_load,autocast


@torch.no_grad()
def evaluate_model(model,loader,device,num_classes,groups,amp=False,return_predictions=False):
    model.eval()
    true=[];pred=[];conf=[];paths=[]
    for images,labels,names in loader:
        if isinstance(images,(tuple,list)):images=images[0]
        images=images.to(device)
        with autocast(device,amp):
            logits=model(images)
        probabilities=F.softmax(logits.float(),dim=-1)
        confidence,indices=probabilities.max(-1)
        true.extend(labels.tolist());pred.extend(indices.cpu().tolist())
        conf.extend(confidence.cpu().tolist());paths.extend(names)
    result,per_class=compute_metrics(true,pred,num_classes,groups)
    if return_predictions:
        prediction_rows=[{'image':path,'true':t,'pred':p,'confidence':cf,'correct':int(t==p)}
                         for path,t,p,cf in zip(paths,true,pred,conf)]
        return result,per_class,prediction_rows
    return result,per_class


def load_for_evaluation(checkpoint,device='auto'):
    from types import SimpleNamespace
    from train import create_model  # lazy import avoids circular dependency
    ckpt=torch_load(checkpoint)
    args=SimpleNamespace(method=ckpt['method'],variant=ckpt['variant'],
          backbone=ckpt['backbone'],pretrained=False,
          rank=ckpt['rank'],router_temperature=ckpt['temperature'],
          num_experts=ckpt.get('num_experts',3),ablation=ckpt.get('ablation','full'))
    dev=select_device(device) if not isinstance(device,torch.device) else device
    model=create_model(args,ckpt['train_counts']).to(dev)
    model.load_state_dict(ckpt['state_dict'],strict=True)
    if ckpt['method']=='ride':model.routing_enabled=bool(ckpt.get('routing_enabled',False))
    model.eval()
    return model,ckpt,dev


def save_checkpoint_eval(checkpoint,manifest,split,out_dir,device='auto',batch_size=32,workers=0):
    model,ckpt,dev=load_for_evaluation(checkpoint,device)
    if manifest['classes']!=ckpt['classes'] or training_counts(manifest)!=ckpt['train_counts']:
        raise ValueError('Checkpoint and dataset manifest have different class counts/order')
    loader=make_loader(manifest,split,ckpt['image_size'],batch_size,workers,ckpt['seed'])
    result,per_class,predictions=evaluate_model(model,loader,dev,len(ckpt['classes']),
                              ckpt['group_assignments'],amp=dev.type=='cuda',return_predictions=True)
    result.update({'method':ckpt['method'],'variant':ckpt['variant'],'backbone':ckpt['backbone'],
                   'seed':ckpt['seed'],'split':split,'best_epoch':ckpt['best_epoch']})
    path=Path(out_dir);path.mkdir(parents=True,exist_ok=True)
    json_save(path/'metrics.json',result)
    csv_save(path/'per_class.csv',[
        {**row,'class_name':ckpt['classes'][row['class_index']],
         'train_count':ckpt['train_counts'][row['class_index']]}
        for row in per_class])
    csv_save(path/'predictions.csv',predictions)
    print('EVAL',split,' '.join(f'{k}={v:.3f}' for k,v in result.items()
                    if k in ('overall_acc','macro_f1','many_acc','medium_acc','few_acc') and v is not None),flush=True)
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--manifest',default=None)
    p.add_argument('--data-root-override',default=None)
    p.add_argument('--split',choices=('train','val','test'),default='test')
    p.add_argument('--output',default=None)
    p.add_argument('--device',default='auto')
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--workers',type=int,default=0)
    a=p.parse_args()
    checkpoint=Path(a.checkpoint)
    ckpt=torch_load(checkpoint)
    manifest=load_manifest(a.manifest or ckpt['manifest'],a.data_root_override)
    save_checkpoint_eval(checkpoint,manifest,a.split,a.output or checkpoint.parent/f'{a.split}_results',
                         a.device,a.batch_size,a.workers)

if __name__=='__main__':main()
