DAFCL: Dual-Asynchronous Federated Continual LearningThis repository contains the official PyTorch implementation for the paper "DAFCL: Dual-Asynchronous Federated Continual Learning".📖 IntroductionFederated Continual Learning (FCL) enables multiple clients to learn from evolving task streams without sharing raw data. However, most existing FCL methods assume synchronous training. In practical heterogeneous edge environments, clients differ in both communication/computation latency and task progression speeds.We formalize this realistic setting as Dual-Asynchronous Federated Continual Learning, where the server receives updates that are simultaneously stale and semantically mismatched in task stages. DAFCL tackles the resulting semantic regression problem via:Decoupled LoRA-Projector Architecture: Separates shared transferable knowledge and task-specific plasticity.Semi-Dynamic Semantic Anchors: Provides a stable semantic coordinate system resistant to temporal drift.Task-Adaptive Gated Aggregation (TAGA): Integrates asynchronous updates using a reliability-aware FIFO buffer mechanism.✨ Key FeaturesRealistic Async Simulator: Built-in event-driven simulator (utils/simulator.py) supporting exponential/uniform delay distributions and asynchronous task progression.Parameter-Efficient Tuning (PEFT): Implements decoupled LoRA tuning for diverse backbones (ViT, ResNet, ConvNeXt).Two-Stage Anchor-Guided Local Training: Effectively mitigates catastrophic forgetting at the client side.📂 Project Structure.
├── core/
│   ├── client.py               # Client-side 2-stage optimization (LoRA + Projector)
│   └── server.py               # Server-side reliability estimation & buffered aggregation
├── data_loader/
│   └── continual_data.py       # Dirichlet non-IID partitioning & task stream generation
├── models/
│   ├── backbone.py             # ViT/ResNet/ConvNeXt with Decoupled LoRA
│   └── proxy_anchor_generator.py # K-Means based anchor initialization
├── utils/
│   ├── args.py                 # Configuration and hyperparameters
│   ├── simulator.py            # Event-driven dual-asynchronous simulator
│   └── toolkit.py              # Helper functions & transforms
├── main.py                     # Entry point for the simulation
└── README.md
🛠️ InstallationClone this repository:git clone [https://github.com/yourusername/DAFCL.git](https://github.com/yourusername/DAFCL.git)
cd DAFCL
Create a virtual environment and install the required dependencies:conda create -n dafcl python=3.9 -y
conda activate dafcl
pip install torch torchvision
pip install timm scikit-learn tqdm numpy pillow
📦 Data & Pretrained Weights PreparationPretrained Weights:Download the ViT-B/16 ImageNet-21k pretrained weights and place them in the ./checkpoints/pretrained/ directory.mkdir -p checkpoints/pretrained
wget [https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz](https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz) # (Or applicable .pth file)
(Ensure the filename matches --pretrained_vit_path in args.py)Datasets:The datasets (CIFAR-10, CIFAR-100, MNIST) will be downloaded automatically by torchvision into the ./data folder.For the proxy dataset used in anchor initialization, please download Tiny-ImageNet-200 and extract it into ./data/tiny-imagenet-200/.🚀 Quick StartRun the main simulation with default parameters (CIFAR-100 on ViT backbone with 50 clients):python main.py \
    --project_name DAFCL_CIFAR100 \
    --dataset cifar100 \
    --model_name vit \
    --num_tasks 5 \
    --num_clients 50 \
    --active_ratio 0.1 \
    --global_rounds 200 \
    --adapter_dim 16 \
    --anchor_task_isolate
Advanced ConfigurationsYou can easily tweak the dual-asynchronous severity by adjusting:--max_delay: Controls the maximum communication staleness (default: 10.0).--delay_dist: Staleness distribution (exponential or uniform).--non_iid_beta: Dirichlet concentration parameter for data heterogeneity (default: 0.5).For more argument details, please refer to utils/args.py.📝 CitationIf you find this code useful for your research, please consider citing our paper:@article{ren2025dafcl,
  title={DAFCL: Dual-Asynchronous Federated Continual Learning},
  author={Ren, Siqi and Ye, Hao and Cao, Jing and Tang, Zehui and Zhao, Shuai and Zeng, Shengke and Chen, Xiaofeng and Han, Song},
  journal={TBD},
  year={2025}
}
🤝 AcknowledgmentsThis work was supported by the Hangzhou Key Research and Development Program, National Natural Science Foundation of China, and other grants.
