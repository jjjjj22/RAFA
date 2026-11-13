"""Unified RAFA training CLI. See README and docs/IMPLEMENTATION_SCOPE.md.

Usage:
  python train.py --manifest splits/cifar100lt.json --method ce --variant rafa --output runs/ce_rafa
  python compare.py --manifest splits/cifar100lt.json --methods ce,bs,bcl

Labels are used for supervised optimization ONLY. The RAFA forward route
never accepts the target labels.
"""
import argparse
from pathlib import Path
import math
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import WeightedRandomSampler

from rafa.data import load_manifest, training_counts, make_loader
from rafa.model import RAFAClassifier, count_parameters, BACKBONES
from rafa.ensemble import BBNModel, RIDEModel
from rafa.variants import DBMVariant, H2TVariant, dbm_style_loss
from rafa.losses import balanced_softmax, balanced_contrastive
from rafa.metrics import shot_groups
from rafa.utils import set_seed,select_device,json_save,csv_save,grad_scaler,autocast
from evaluate import evaluate_model, save_checkpoint_eval


def build_parser():
    p=argparse.ArgumentParser(description='Unified training of RAFA and paired baseline models')
    p.add_argument('--manifest',required=True,help='JSON from prepare_data.py')
    p.add_argument('--data-root-override',default=None)
    p.add_argument('--output',required=True)
    p.add_argument('--method',choices=('ce','bs','bcl','bbn','ride','dbm','h2t'),default='ce')
    p.add_argument('--variant',choices=('baseline','rafa'),default='rafa')
    p.add_argument('--backbone',choices=BACKBONES,default='resnet18')
    p.add_argument('--pretrained',action='store_true')
    p.add_argument('--epochs',type=int,default=50)
    p.add_argument('--h2t-stage1-epochs',type=int,default=30)
    p.add_argument('--h2t-stage2-epochs',type=int,default=20)
    p.add_argument('--h2t-mix-rho',type=float,default=.3)
    p.add_argument('--ride-router-epochs',type=int,default=0)
    p.add_argument('--num-experts',type=int,default=3)
    p.add_argument('--image-size',type=int,default=224)
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--vit-lr',type=float,default=2e-5)
    p.add_argument('--weight-decay',type=float,default=1e-4)
    p.add_argument('--aux-weight',type=float,default=.1)
    p.add_argument('--ablation',choices=('full','router_only','ungated_adapter','global_gated','no_gating','no_aux','hard_router'),default='full')
    p.add_argument('--rank',type=int,default=16)
    p.add_argument('--router-temperature',type=float,default=1.)
    p.add_argument('--group-method',choices=('auto','threshold','knee','tertile'),default='auto')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--workers',type=int,default=0)
    p.add_argument('--device',default='auto')
    p.add_argument('--no-amp',action='store_true')
    p.add_argument('--no-test',action='store_true')
    p.add_argument('--bcl-weight',type=float,default=1.)
    p.add_argument('--bcl-temperature',type=float,default=.07)
    p.add_argument('--ride-diversity-weight',type=float,default=.05)
    p.add_argument('--dbm-max-margin',type=float,default=.1)
    p.add_argument('--dbm-tau',type=float,default=.5)
    p.add_argument('--dbm-inst-weight',type=float,default=1.)
    return p


def create_model(args,counts,method=None,variant=None):
    method=method or args.method
    use_rafa=(variant or args.variant)=='rafa'
    kw=dict(backbone=args.backbone,pretrained=args.pretrained)
    if method=='bbn':
        model=BBNModel(args.backbone,counts,
                        pretrained=args.pretrained,use_rafa=use_rafa,
                        rank=args.rank,temperature=args.router_temperature)
    elif method=='ride':
        model=RIDEModel(args.backbone,counts,experts=args.num_experts,
                         pretrained=args.pretrained,use_rafa=use_rafa,
                         rank=args.rank,temperature=args.router_temperature)
    else:
        klass = DBMVariant if method=='dbm' else H2TVariant if method=='h2t' else RAFAClassifier
        model=klass(backbone=args.backbone,train_counts=counts,method=method,
                    use_rafa=use_rafa,pretrained=args.pretrained,rank=args.rank,
                    temperature=args.router_temperature)
    if use_rafa:
        gate_modes={'full':'adaptive','router_only':'off','ungated_adapter':'identity',
                    'global_gated':'global','no_gating':'identity','no_aux':'adaptive',
                    'hard_router':'adaptive'}
        model.adapter.gate_mode=gate_modes[args.ablation]
        if args.ablation=='hard_router': model.adapter.route_mode='hard'
    return model


def get_optimizer(model,args):
    rate=args.vit_lr if args.backbone.startswith('vit_') else args.lr
    return torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=rate,
                             weight_decay=args.weight_decay)


def _reverse_sampler(manifest):
    counts=np.asarray(training_counts(manifest),dtype=float)
    labels=np.asarray([c for _,c in manifest['train']],dtype=int)
    return WeightedRandomSampler(torch.as_tensor(1/counts[labels],dtype=torch.double),
                                 len(labels),replacement=True,generator=torch.Generator().manual_seed(manifest['seed']+13))


def _ride_diversity(logits,weight):
    if weight<=0:return logits.new_zeros(())
    p=F.softmax(logits.float()/2,dim=-1)
    q=F.log_softmax(logits.float()/2,dim=-1)
    pairs=[]
    for i in range(logits.shape[1]):
        for j in range(i+1,logits.shape[1]):
            kij=(p[:,i]*(q[:,i]-q[:,j])).sum(-1)
            kji=(p[:,j]*(q[:,j]-q[:,i])).sum(-1)
            pairs.append(.5*(kij+kji).mean())
    return -float(weight)*torch.stack(pairs).mean().clamp(max=10)


def _epoch(model,loader,optimizer,scaler,device,args,counts,epoch,
           reversed_loader=None,donor_loader=None,stage='regular'):
    model.train()
    if stage=='h2t_stage2':model.backbone.eval()
    iterator=iter(reversed_loader) if reversed_loader is not None else None
    donor_iter=iter(donor_loader) if donor_loader is not None else None
    running={k:0. for k in ('loss','classification','auxiliary')}
    samples=0
    for inputs,labels,_ in loader:
        if isinstance(inputs,(tuple,list)):
            x=inputs[0].to(device)
            x2=inputs[1].to(device)
            x3=inputs[2].to(device)
        else:
            x=inputs.to(device)
        labels=labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device,not args.no_amp):
            if args.method=='bbn':
                try:reverse,reverse_labels,_=next(iterator)
                except StopIteration:
                    iterator=iter(reversed_loader)
                    reverse,reverse_labels,_=next(iterator)
                reverse=reverse.to(device); reverse_labels=reverse_labels.to(device)
                alpha=1.-((epoch-1)/max(args.epochs-1,1))**2
                outputs,routes=model.forward_pair(x,reverse,alpha)
                classification=alpha*F.cross_entropy(outputs.float(),labels)+(1-alpha)*F.cross_entropy(outputs.float(),reverse_labels)
                auxiliary=classification.new_zeros(())
                if model.adapter is not None:
                    auxiliary=alpha*model.adapter.auxiliary_loss(routes[0]['router_logits'],labels)+(1-alpha)*model.adapter.auxiliary_loss(routes[1]['router_logits'],reverse_labels)
            elif args.method=='ride':
                experts,routes,_=model.expert_outputs(x)
                classification=torch.stack([F.cross_entropy(experts[:,i].float(),labels) for i in range(model.experts_count)]).mean()
                classification=classification+_ride_diversity(experts,args.ride_diversity_weight)
                auxiliary=classification.new_zeros(())
                if model.adapter is not None:
                    auxiliary=torch.stack([model.adapter.auxiliary_loss(r['router_logits'],labels) for r in routes]).mean()
            elif args.method=='h2t' and stage=='h2t_stage2':
                try:donor,_,_=next(donor_iter)
                except StopIteration:
                    donor_iter=iter(donor_loader)
                    donor,_,_=next(donor_iter)
                # All H2T donor data come from the training split.
                donor=donor.to(device)
                outputs,aux=model.mixed_logits(x,donor,rho=args.h2t_mix_rho)
                classification=F.cross_entropy(outputs.float(),labels)
                auxiliary=model.adapter.auxiliary_loss(aux['router_logits'],labels) if aux else classification.new_zeros(())
            else:
                outputs,aux=model(x,return_aux=True)
                if args.method=='bs':
                    classification=balanced_softmax(outputs,labels,counts)
                elif args.method=='dbm':
                    classification=dbm_style_loss(outputs,labels,counts,args.dbm_max_margin,
                                                   args.dbm_tau,args.dbm_inst_weight)
                else:
                    classification=F.cross_entropy(outputs.float(),labels)
                auxiliary=model.adapter.auxiliary_loss(aux['router_logits'],labels) if aux else classification.new_zeros(())
                if args.method=='bcl':
                    projections=torch.cat([model.projected_views(x2),model.projected_views(x3)],0)
                    y2=torch.cat([labels,labels],0)
                    classification=classification+args.bcl_weight*balanced_contrastive(
                        projections,y2,model.class_centers(),args.bcl_temperature)
            loss=classification+args.aux_weight*auxiliary
        if not torch.isfinite(loss):raise FloatingPointError(f'non-finite loss in {args.method}')
        scaler.scale(loss).backward()
        scaler.step(optimizer);scaler.update()
        b=len(labels);samples+=b
        for key,val in (('loss',loss),('classification',classification),('auxiliary',auxiliary)):
            running[key]+=float(val.detach())*b
    return {k:v/max(samples,1) for k,v in running.items()}


def _train_ride_router(model,loader,optimizer,device):
    # The binary target is constructed from *training* labels only.
    model.eval();model.router.train()
    total=0.;n=0
    for images,labels,_ in loader:
        images=images.to(device);labels=labels.to(device)
        with torch.no_grad():
            common=model.encoder.common(images)
            feats=model.encoder.route_feature(common).detach()
            out=[]
            for i,head in enumerate(model.classifiers):
                z=model.encoder.expert(common,i)
                if model.adapter is not None:z=model.adapter(z)
                out.append(head(z).argmax(-1))
            pred=torch.stack(out,dim=1)
            target=((pred[:,0]!=labels)&(pred[:,1:]==labels[:,None]).any(-1)).float()
        optimizer.zero_grad(set_to_none=True)
        gate=model.router(feats).squeeze(-1)
        loss=F.binary_cross_entropy_with_logits(gate.float(),target)
        loss.backward();optimizer.step()
        total+=float(loss.detach())*len(labels);n+=len(labels)
    return total/max(1,n)


def train(args):
    if args.epochs<=0 or args.batch_size<=0 or args.aux_weight<0:
        raise ValueError('Invalid training hyperparameters')
    if args.backbone.startswith('vit_') and args.image_size!=224:
        raise ValueError('Torchvision ViT requires image-size=224')
    if args.variant=='baseline' and args.ablation!='full':
        raise ValueError('Ablations only apply to --variant rafa')
    if args.ablation in ('ungated_adapter','global_gated','no_aux'):
        args.aux_weight=0.0  # Router objective is absent in these ablations
    set_seed(args.seed)
    manifest=load_manifest(args.manifest,args.data_root_override)
    counts=training_counts(manifest)
    group=shot_groups(counts,args.group_method)
    device=select_device(args.device)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    json_save(out/'run_config.json',vars(args))
    csv_save(out/'group_assignments.csv',[
        {'class_index':i,'class_name':name,'train_count':counts[i],'group':group[i]}
        for i,name in enumerate(manifest['classes'])])
    train_loader=make_loader(manifest,'train',args.image_size,args.batch_size,args.workers,
                             args.seed,bcl=args.method=='bcl')
    val_loader=make_loader(manifest,'val',args.image_size,args.batch_size,args.workers,args.seed)
    test_loader=make_loader(manifest,'test',args.image_size,args.batch_size,args.workers,args.seed)
    reverse_loader=(make_loader(manifest,'train',args.image_size,args.batch_size,args.workers,args.seed,
                                sampler=_reverse_sampler(manifest)) if args.method=='bbn' else None)
    donor_loader=(make_loader(manifest,'train',args.image_size,args.batch_size,args.workers,args.seed,
                              sampler=_reverse_sampler(manifest)) if args.method=='h2t' else None)
    model=create_model(args,counts).to(device)
    print(f'Method={args.method} variant={args.variant} backbone={args.backbone} parameters={count_parameters(model):,} device={device}',flush=True)
    history=[];best=-1.;best_epoch=0
    # Approximate H2T: stage 1 shared CE representation, stage 2 frozen backbone.
    if args.method=='h2t':
        stage1=H2TVariant(backbone=args.backbone,train_counts=counts,method='ce',
                          pretrained=args.pretrained,rank=args.rank,
                          temperature=args.router_temperature,use_rafa=False).to(device)
        opt=get_optimizer(stage1,args)
        sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=args.h2t_stage1_epochs)
        scale=grad_scaler(device,not args.no_amp)
        for epoch in range(1,args.h2t_stage1_epochs+1):
            # stage1 uses ordinary CE without RAFA
            previous=args.method;args.method='ce'
            stat=_epoch(stage1,train_loader,opt,scale,device,args,counts,epoch)
            args.method=previous
            sched.step()
            print(f'H2T stage 1 epoch {epoch}/{args.h2t_stage1_epochs} loss={stat["loss"]:.4f}',flush=True)
        # Copy backbone and head, leaving RAFA in its default initialized state.
        missing,unexpected=model.load_state_dict(stage1.state_dict(),strict=args.variant=='baseline')
        if args.variant=='rafa' and (unexpected or any(not k.startswith('adapter.') for k in missing)):
            raise RuntimeError(f'Unexpected H2T stage transfer keys: {missing} {unexpected}')
        model.freeze_backbone()
    optimizer=get_optimizer(model,args)
    actual_epochs=args.h2t_stage2_epochs if args.method=='h2t' else args.epochs
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=actual_epochs)
    scaler=grad_scaler(device,not args.no_amp)
    for epoch in range(1,actual_epochs+1):
        stat=_epoch(model,train_loader,optimizer,scaler,device,args,counts,epoch,
                    reversed_loader=reverse_loader,donor_loader=donor_loader,
                    stage='h2t_stage2' if args.method=='h2t' else 'regular')
        result,_=evaluate_model(model,val_loader,device,len(counts),group,amp=not args.no_amp)
        scheduler.step()
        record={'epoch':epoch,'train_loss':stat['loss'],'train_cls':stat['classification'],
                'train_router':stat['auxiliary'],**{'val_'+k:v for k,v in result.items()}}
        history.append(record);csv_save(out/'history.csv',history)
        print(f'Epoch {epoch:03}/{actual_epochs} valAcc={result["overall_acc"]:.2f} valF1={result["macro_f1"]:.2f} valFew={result["few_acc"]} ',flush=True)
        if result['macro_f1']>best:
            best=result['macro_f1'];best_epoch=epoch
            ckpt={'state_dict':model.state_dict(),'method':args.method,'variant':args.variant,
                  'backbone':args.backbone,'rank':args.rank,'temperature':args.router_temperature,
                  'num_experts':args.num_experts,'ablation':args.ablation,'classes':manifest['classes'],
                  'train_counts':counts,'group_assignments':group,'seed':args.seed,
                  'image_size':args.image_size,'best_epoch':epoch,'manifest':str(Path(args.manifest).resolve()),
                  'implementation_scope':'adapted' if args.method in ('bbn','ride','dbm','h2t') else 'original_rafa_integration'}
            torch.save(ckpt,out/'best.pth')
    if args.method=='ride' and args.ride_router_epochs>0:
        from rafa.utils import torch_load
        ckpt=torch_load(out/'best.pth')
        model.load_state_dict(ckpt['state_dict'])
        for param in model.parameters():param.requires_grad=False
        for param in model.router.parameters():param.requires_grad=True
        ropt=torch.optim.AdamW(model.router.parameters(),lr=args.lr)
        for i in range(args.ride_router_epochs):
            value=_train_ride_router(model,train_loader,ropt,device)
            print(f'RIDE router stage2 {i+1}: loss={value:.4f}',flush=True)
        model.routing_enabled=True
        ckpt['routing_enabled']=True
        ckpt['state_dict']=model.state_dict()
        torch.save(ckpt,out/'best.pth')
    json_save(out/'training_summary.json',{'best_epoch':best_epoch,'best_val_macro_f1':best,
                                            'parameters':count_parameters(model),'seed':args.seed})
    if not args.no_test:
        result=save_checkpoint_eval(out/'best.pth',manifest,'test',out/'test_results',
                                     device=device,batch_size=args.batch_size,workers=args.workers)
        return result
    return None


if __name__=='__main__':
    train(build_parser().parse_args())
