"""Project-adapted BBN and RIDE variants using the original RAFA core.

ResNets share layers conv1--layer3, with separate layer4 experts. For ViT,
ConvNeXt and Swin, expert heads are feature-space residual adapters. These
are unified integration variants, NOT bit-for-bit implementations of the
BBN/RIDE authors' full architectures or training protocols.
"""
from copy import deepcopy
import torch
from torch import nn
from torch.nn import functional as F
from .model import make_backbone, RarityAwareFeatureAdapter


class ResidualExpert(nn.Module):
    def __init__(self, d, rank=64):
        super().__init__()
        self.net=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,rank),nn.GELU(),nn.Linear(rank,d))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,x):return x+self.net(x)


class SharedLateBackbone(nn.Module):
    def __init__(self,backbone,experts,pretrained=False,expert_rank=64):
        super().__init__()
        full,dim=make_backbone(backbone,pretrained)
        self.dim=dim
        self.resnet=backbone.startswith('resnet')
        if self.resnet:
            self.shared=nn.Sequential(full.conv1,full.bn1,full.relu,full.maxpool,
                                      full.layer1,full.layer2,full.layer3)
            self.tails=nn.ModuleList([deepcopy(full.layer4) for _ in range(experts)])
            self.pool=nn.AdaptiveAvgPool2d(1)
            last=full.layer3[-1]
            self.router_dim=last.conv3.out_channels if hasattr(last,'conv3') else last.conv2.out_channels
        else:
            self.shared=full
            self.tails=nn.ModuleList([ResidualExpert(dim,expert_rank) for _ in range(experts)])
            self.router_dim=dim
    def common(self,images):return self.shared(images)
    def expert(self,common,i):
        x=self.tails[i](common)
        return self.pool(x).flatten(1) if self.resnet else x
    def route_feature(self,common):
        return F.adaptive_avg_pool2d(common,1).flatten(1) if self.resnet else common


class BBNModel(nn.Module):
    def __init__(self,backbone,counts,pretrained=False,use_rafa=False,rank=16,temperature=1.):
        super().__init__()
        self.encoder=SharedLateBackbone(backbone,2,pretrained)
        self.classifiers=nn.ModuleList([nn.Linear(self.encoder.dim,len(counts)) for _ in range(2)])
        self.adapter=RarityAwareFeatureAdapter(self.encoder.dim,counts,rank,temperature) if use_rafa else None
        self.use_rafa=use_rafa
    def branch(self,common,i):
        feat=self.encoder.expert(common,i)
        aux=None
        if self.adapter is not None:feat,aux=self.adapter(feat,return_aux=True)
        return self.classifiers[i](feat),aux
    def forward_pair(self,images_a,images_b,alpha):
        l1,a1=self.branch(self.encoder.common(images_a),0)
        l2,a2=self.branch(self.encoder.common(images_b),1)
        return alpha*l1+(1-alpha)*l2,(a1,a2)
    def forward(self,images):
        common=self.encoder.common(images)
        a,_=self.branch(common,0);b,_=self.branch(common,1)
        return (a+b)*.5


class RIDEModel(nn.Module):
    def __init__(self,backbone,counts,experts=3,pretrained=False,use_rafa=False,rank=16,temperature=1.):
        super().__init__()
        if experts<2:raise ValueError('RIDE experts must be >=2')
        self.encoder=SharedLateBackbone(backbone,experts,pretrained)
        self.classifiers=nn.ModuleList([nn.Linear(self.encoder.dim,len(counts)) for _ in range(experts)])
        self.adapter=RarityAwareFeatureAdapter(self.encoder.dim,counts,rank,temperature) if use_rafa else None
        self.router=nn.Sequential(nn.LayerNorm(self.encoder.router_dim),nn.Linear(self.encoder.router_dim,64),
                                  nn.ReLU(),nn.Linear(64,1))
        self.experts_count=experts
        self.use_rafa=use_rafa
        self.routing_enabled=False
    def expert_outputs(self,images):
        common=self.encoder.common(images)
        outputs=[];aux=[]
        for i,head in enumerate(self.classifiers):
            z=self.encoder.expert(common,i)
            route=None
            if self.adapter is not None:z,route=self.adapter(z,return_aux=True)
            outputs.append(head(z));aux.append(route)
        return torch.stack(outputs,dim=1),aux,self.router(self.encoder.route_feature(common)).squeeze(-1)
    def forward(self,images):
        logits,_,gate=self.expert_outputs(images)
        mean=logits.mean(1)
        if not self.routing_enabled:return mean
        return torch.where((torch.sigmoid(gate) >= 0.5)[:,None],mean,logits[:,0,:])
