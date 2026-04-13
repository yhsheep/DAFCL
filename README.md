# DAFCL: Dual-Asynchronous Federated Continual Learning
This repository contains the official PyTorch implementation for the paper "DAFCL: Dual-Asynchronous Federated Continual Learning".

## Introduction
Federated Continual Learning (FCL) enables multiple clients to learn from evolving task streams without sharing raw data. However, most existing FCL methods assume synchronous training. In practical heterogeneous edge environments, clients differ in both communication/computation latency and task progression speeds.

We formalize this realistic setting as Dual-Asynchronous Federated Continual Learning, where the server receives updates that are simultaneously stale and semantically mismatched in task stages. DAFCL tackles the resulting semantic regression problem via:
1. Decoupled LoRA-Projector Architecture: Separates shared transferable knowledge and task-specific plasticity.

2. Semi-Dynamic Semantic Anchors: Provides a stable semantic coordinate system resistant to temporal drift.

3. Task-Adaptive Gated Aggregation (TAGA): Integrates asynchronous updates using a reliability-aware FIFO buffer mechanism.

## Project Structure
```text
.
├── core/
│   ├── client.py           
│   └── server.py           
├── data_loader/
│   └── continual_data.py   
├── models/
│   ├── backbone.py         
│   └── proxy_anchor_generator.py 
├── utils/
│   ├── args.py            
│   ├── simulator.py       
│   └── toolkit.py          
├── main.py                 
└── README.md
```

### Installation
1. Clone this repository:
```text
git clone [https://github.com/yhsheep/DAFCL.git](https://github.com/yhsheep/DAFCL.git)
cd DAFCL
```
2. Create a virtual environment and install the required dependencies:
```text
conda create -n dafcl python=3.9 -y
conda activate dafcl
pip install torch torchvision
pip install timm scikit-learn tqdm numpy pillow
```
### Data & Pretrained Weights Preparation

1. Pretrained Weights:
Download the ViT-B/16 ImageNet-21k pretrained weights and place them in the ./checkpoints/pretrained/ directory.

2.Datasets:
The datasets (CIFAR-10, CIFAR-100, MNIST) will be downloaded automatically by torchvision into the `./data` folder.
For the proxy dataset used in anchor initialization, please download Tiny-ImageNet-200 and extract it into `./data/tiny-imagenet-200/.`

## Quick Start
Run the main simulation with default parameters (CIFAR-100 on ViT backbone with 50 clients):

```text
python main.py \
    --project_name DAFCL_CIFAR100 \
    --dataset cifar100 \
    --model_name vit \
    --num_tasks 5 \
    --num_clients 50 \
    --active_ratio 0.1 \
    --global_rounds 200 \
    --adapter_dim 16 \
    --anchor_task_isolate
```

Advanced Configurations

You can easily tweak the dual-asynchronous severity by adjusting:

* `--max_delay`: Controls the maximum communication staleness (default: 10.0).

* `--delay_dist`: Staleness distribution (exponential or uniform).

* `--non_iid_beta`: Dirichlet concentration parameter for data heterogeneity (default: 0.5).

For more argument details, please refer to `utils/args.py`.

## Citation

If you find this code useful for your research, please consider citing our paper:
```text
@article{ren2025dafcl,
  title={DAFCL: Dual-Asynchronous Federated Continual Learning},
  author={Ren, Siqi and Ye, Hao and Cao, Jing and Tang, Zehui and Zhao, Shuai and Zeng, Shengke and Chen, Xiaofeng and Han, Song},
  journal={TBD},
  year={2025}
}
```

### Acknowledgments
This work was supported by the Hangzhou Key Research and Development Program, National Natural Science Foundation of China, and other grants.
