"""Parameter count, theoretical RAFA linear MACs, optional MEASURED latency.

Timing depends on actual hardware and is disabled unless --benchmark is set.
"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from rafa.model import RAFAClassifier, count_parameters


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--backbone',default='resnet18')
    p.add_argument('--classes',type=int,default=100)
    p.add_argument('--rank',type=int,default=16)
    p.add_argument('--image-size',type=int,default=224)
    p.add_argument('--device',default='auto')
    p.add_argument('--benchmark',action='store_true')
    p.add_argument('--warmup',type=int,default=10)
    p.add_argument('--repeats',type=int,default=50)
    a=p.parse_args()
    counts=list(range(1,a.classes+1))
    models=[RAFAClassifier(a.backbone,counts,use_rafa=v,rank=a.rank) for v in (False,True)]
    base,rafa=map(count_parameters,models)
    d=models[-1].feature_dim
    extra_linear_macs=d*a.classes+2*d*a.rank
    print('backbone',a.backbone,'classes',a.classes,'features',d)
    print('Baseline params:',f'{base:,}','RAFA params:',f'{rafa:,}',
          'Added:',f'{rafa-base:,}',f'({100*(rafa-base)/base:.4f}%)')
    print('RAFA LINEAR projection MACs/image:',f'{extra_linear_macs:,}',
          '(excludes norm, GELU, Softmax, gate and elementwise ops)')
    if not a.benchmark:return
    dev=torch.device('cuda' if a.device=='auto' and torch.cuda.is_available() else 'cpu' if a.device=='auto' else a.device)
    x=torch.randn(1,3,a.image_size,a.image_size,device=dev)
    from time import perf_counter
    for variant,model in zip(['baseline','rafa'],models):
        model=model.to(dev).eval()
        with torch.inference_mode():
            for _ in range(a.warmup):model(x)
            if dev.type=='cuda':torch.cuda.synchronize()
            t0=perf_counter()
            for _ in range(a.repeats):model(x)
            if dev.type=='cuda':torch.cuda.synchronize()
        print('Measured latency:',variant, f'{(perf_counter()-t0)/a.repeats*1000:.3f} ms/image on {dev}')

if __name__=='__main__':main()
