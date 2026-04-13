import argparse
import os
import torch

def get_args():
    parser = argparse.ArgumentParser(description="DAFCL: Dual-Asynchronous Federated Continual Learning")


    parser.add_argument('--project_name', type=str, default='DAFCL_Final_MultiModel_1024D')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output_dir', type=str, default='./checkpoints')

  
    parser.add_argument('--pretrained_vit_path', type=str, 
                        default='./checkpoints/pretrained/jx_vit_base_patch16_224_in21k-e5005f0a.pth',
                        help='Pretrained weights for ViT-B/16')


    parser.add_argument('--dataset', type=str, default='cifar10', 
                        choices=['mnist', 'cifar10', 'cifar100', 'imagenet-r'],
                        help='Name of the dataset')
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--img_size', type=int, default=224, help='Image size for backbone input')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--non_iid_beta', type=float, default=0.5, help='Dirichlet distribution alpha')
    parser.add_argument('--num_tasks', type=int, default=5, help='Total number of continual tasks')
    parser.add_argument('--class_order', type=str, default='random', choices=['random', 'sorted'])
    parser.add_argument('--pub_anchor_ratio', type=float, default=0.1, help='Ratio of public data for Phase 1')

 
    parser.add_argument('--model_name', type=str, default='vit', 
                        choices=['vit', 'resnet', 'convnext'],
                        help='Base model architecture')
    parser.add_argument('--backbone_type', type=str, default='vit_base_patch16_224',
                        help='Specific timm model string (e.g., resnet50, convnext_base)')
    parser.add_argument('--adapter_dim', type=int, default=16, help='LoRA rank (r)')
    parser.add_argument('--lora_task_strength', type=float, default=0.1, help='Global LoRA weight scale (0-1, smaller=stronger task isolation)')


    parser.add_argument('--num_clients', type=int, default=50)
    parser.add_argument('--local_epochs', type=int, default=4, help='2 for LoRA, 2 for Projector') 
    parser.add_argument('--global_rounds', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=64) 
    parser.add_argument('--active_ratio', type=float, default=0.1) 
    parser.add_argument('--window_size', type=int, default=3) 
    parser.add_argument('--delay_dist', type=str, default='exponential', choices=['exponential', 'uniform'])
    parser.add_argument('--max_delay', type=float, default=10.0)


    parser.add_argument('--lr', type=float, default=0.001) 
    parser.add_argument('--momentum', type=float, default=0.5, help='Server-side momentum (beta)')
    parser.add_argument('--weight_decay', type=float, default=1e-2) 


    parser.add_argument('--temp', type=float, default=0.05, help='Temperature for similarity')
    parser.add_argument('--alpha', type=float, default=0.3, help='Global update step size (eta)')
    parser.add_argument('--lambda_kd', type=float, default=1.0, help='Loss weight for compactness')
    

    parser.add_argument('--staleness_alpha', type=float, default=0.5)
    parser.add_argument('--sigma', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--gate_floor', type=float, default=0.1)


    parser.add_argument('--anchor_lr', type=float, default=0.01)
    parser.add_argument('--delta_threshold', type=float, default=0.7)
    parser.add_argument('--anchor_task_isolate', action='store_true', default=True, help='Isolate anchor offsets by task')

    parser.add_argument('--lambda_memory', type=float, default=0.5, help='Weight for old task memory loss')
    parser.add_argument('--memory_task_num', type=int, default=2, help='Number of past tasks to remember')

    args = parser.parse_args()


    dataset_classes = {
        'mnist': 10,
        'cifar10': 10,
        'cifar100': 100,
        'imagenet-r': 200
    }
    args.num_classes = dataset_classes.get(args.dataset.lower(), 100)
    
    if args.model_name == 'resnet' and 'resnet' not in args.backbone_type:
        args.backbone_type = 'resnet50'
    elif args.model_name == 'convnext' and 'convnext' not in args.backbone_type:
        args.backbone_type = 'convnext_base'
    elif args.model_name == 'vit' and 'vit' not in args.backbone_type:
        args.backbone_type = 'vit_base_patch16_224'

    return args
