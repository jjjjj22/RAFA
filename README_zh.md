# RAFA 长尾分类项目：中文快速使用说明

本项目包含 RAFA 主网络、CIFAR-100-LT 与农业类自然长尾数据集读取、训练、评估、成对方法对比、类别分组、真实多随机种子统计、模型复杂度和 Router 诊断。完整英文说明请参见 [README.md](README.md)。

## 一、核心网络

`rafa/model.py` 为论文 RAFA 的核心：训练集计算固定稀缺度先验、Soft Rarity Router、Softplus 保证正斜率的动态门控、低秩残差 Adapter、Balanced Softmax 路由辅助损失。**推理不使用测试标签，也不读取测试集类别统计。** 默认：`rank=16`、`tau=1.0`、`lambda_router=0.1`、`gate_bias=-2`、`raw_slope=1.85`。

## 二、安装

```bash
pip install -r requirements.txt
python -m unittest discover -s tests -v
```

## 三、创建数据划分（只做一次）

原始 CIFAR-100 Python 数据集目录需要包含 `train`、`test`、`meta`：

```bash
python prepare_data.py --dataset cifar100lt --root "D:/datasets/cifar-100-python" \
    --output splits/cifar100lt_seed42.json --imb-factor 100 --seed 42
```

DLCPD-25 或 PlantWild v2 每个类别单独一个目录：

```bash
python prepare_data.py --dataset folder --root "D:/datasets/PlantWildV2" \
    --output splits/plantwild_seed42.json --seed 42
```

也支持数据集已按 `train/class`、`val/class`、`test/class` 划分的形式。所有 Baseline 和 +RAFA 均使用同一个 JSON 划分文件。

## 四、训练与一键对比

```bash
python train.py --manifest splits/cifar100lt_seed42.json --method ce \
    --variant rafa --backbone resnet18 --pretrained --epochs 50 \
    --output runs/ce_rafa

python compare.py --manifest splits/cifar100lt_seed42.json \
    --methods ce,bs,bcl --backbones resnet18,vit_b_16 \
    --seeds 42,43,44 --pretrained --output runs/cifar_comparison
```

`--method` 可选 `ce/bs/bcl/bbn/ride/dbm/h2t`，但**BBN/RIDE 为适配版，DBM/H2T 为近似示范版**，这四者并非原论文作者逐位等价的实现，也不保证复现已有论文表格。正式公开论文复现结果前，需要接入你实际用来跑这些结果的完整原始训练代码。具体参见 `docs/IMPLEMENTATION_SCOPE.md`。

## 五、评估、统计、诊断

```bash
python evaluate.py --checkpoint runs/ce_rafa/best.pth \
    --manifest splits/cifar100lt_seed42.json

python scripts/aggregate_seeds.py --runs runs/cifar_comparison --min-seeds 3 \
    --output runs/mean_sd.csv

python scripts/profile_model.py --backbone resnet18 --classes 100

python scripts/router_diagnostics.py --checkpoint runs/ce_rafa/best.pth \
    --manifest splits/cifar100lt_seed42.json --output runs/router_diagnostic
```

其中 `router_diagnostics` 可计算真实类别稀缺度与 Router 预测稀缺度的 Spearman 相关性、不同组别平均门控、尾部错分至头部的比例，以及仅供离线诊断的 Soft/Hard/Oracle 结果。Oracle 使用标签仅限诊断，不进入正式模型推理。

## 六、GitHub 上传注意事项

不包含任何实际数据集图片、真实训练权重、API 密钥或模拟标准差。`runs/`、`data/`、`splits/`、`*.pth` 等已加入 `.gitignore`。正式论文的三张对比表和消融表仍需用保存的真实 checkpoint、日志和多随机种子结果核对。

**建议：** GitHub 仓库 README 标明这是一份整合后的 RAFA 研究代码，并在补齐原始 BBN/RIDE/DBM/H2T 项目与发表论文最终训练设置之后，将仓库标注为可复现版本。
