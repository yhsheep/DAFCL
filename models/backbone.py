import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import os
import math


class LoRA_Linear(nn.Module):
    def __init__(self, original_linear, rank=16, alpha=32, num_tasks=5, lora_task_strength=0.3):
        super().__init__()
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.rank = rank
        self.scaling = alpha / rank
        self.num_tasks = num_tasks
        self.lora_task_strength = lora_task_strength  
        
        self.weight = nn.Parameter(original_linear.weight.data.clone())
        self.bias = nn.Parameter(original_linear.bias.data.clone()) if original_linear.bias is not None else None
        self.weight.requires_grad = False

        self.lora_A_global = nn.Parameter(torch.zeros(self.in_features, rank))
        self.lora_B_global = nn.Parameter(torch.zeros(rank, self.out_features))
        
        self.lora_As_task = nn.ParameterList([
            nn.Parameter(torch.zeros(self.in_features, rank)) for _ in range(num_tasks)
        ])
        self.lora_Bs_task = nn.ParameterList([
            nn.Parameter(torch.zeros(rank, self.out_features)) for _ in range(num_tasks)
        ])

        self.current_task_id = None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A_global, a=math.sqrt(5))
        self.lora_A_global.data *= self.lora_task_strength
        nn.init.zeros_(self.lora_B_global)
        for i in range(self.num_tasks):
            nn.init.kaiming_uniform_(self.lora_As_task[i], a=math.sqrt(5))
            nn.init.zeros_(self.lora_Bs_task[i])

    def forward(self, x):
        result = nn.functional.linear(x, self.weight, self.bias)
        result += (x @ self.lora_A_global @ self.lora_B_global) * self.scaling * self.lora_task_strength
        if self.current_task_id is not None and 0 <= self.current_task_id < self.num_tasks:
            result += (x @ self.lora_As_task[self.current_task_id] @ self.lora_Bs_task[self.current_task_id]) * self.scaling
        return result


class LoRA_Conv2d(nn.Module):
    def __init__(self, original_conv, rank=16, alpha=32, num_tasks=5, lora_task_strength=0.3):
        super().__init__()
        self.in_channels = original_conv.in_channels
        self.out_channels = original_conv.out_channels
        self.kernel_size = original_conv.kernel_size
        self.stride = original_conv.stride
        self.padding = original_conv.padding
        self.groups = original_conv.groups
        self.rank = rank
        self.scaling = alpha / rank
        self.num_tasks = num_tasks
        self.lora_task_strength = lora_task_strength
        
        self.weight = nn.Parameter(original_conv.weight.data.clone())
        self.bias = nn.Parameter(original_conv.bias.data.clone()) if original_conv.bias is not None else None
        self.weight.requires_grad = False

        self.lora_A_global = nn.Parameter(torch.zeros(self.out_channels, rank))
        self.lora_B_global = nn.Parameter(torch.zeros(rank, (self.in_channels // self.groups) * self.kernel_size[0] * self.kernel_size[1]))
        
        self.lora_As_task = nn.ParameterList([nn.Parameter(torch.zeros(self.out_channels, rank)) for _ in range(num_tasks)])
        self.lora_Bs_task = nn.ParameterList([nn.Parameter(torch.zeros(rank, (self.in_channels // self.groups) * self.kernel_size[0] * self.kernel_size[1])) for _ in range(num_tasks)])

        self.current_task_id = None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A_global, a=math.sqrt(5))
        self.lora_A_global.data *= self.lora_task_strength
        nn.init.zeros_(self.lora_B_global)
        for i in range(self.num_tasks):
            nn.init.kaiming_uniform_(self.lora_As_task[i], a=math.sqrt(5))
            nn.init.zeros_(self.lora_Bs_task[i])

    def forward(self, x):
        result = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, groups=self.groups)
        
        delta_w_g = (self.lora_A_global @ self.lora_B_global).view(self.out_channels, self.in_channels // self.groups, *self.kernel_size)
        result += F.conv2d(x, delta_w_g, None, self.stride, self.padding, groups=self.groups) * self.scaling * self.lora_task_strength

        if self.current_task_id is not None and 0 <= self.current_task_id < self.num_tasks:
            delta_w_t = (self.lora_As_task[self.current_task_id] @ self.lora_Bs_task[self.current_task_id]).view(self.out_channels, self.in_channels // self.groups, *self.kernel_size)
            result += F.conv2d(x, delta_w_t, None, self.stride, self.padding, groups=self.groups) * self.scaling
        return result


class FeatureProjector(nn.Module):
    def __init__(self, input_dim, output_dim=512, num_tasks=5):
        super().__init__()
        self.global_proj = nn.Linear(input_dim, output_dim)
        self.task_projs = nn.ModuleList([nn.Linear(input_dim, output_dim) for _ in range(num_tasks)])
        self._init_weights()

    def _init_weights(self):
        nn.init.orthogonal_(self.global_proj.weight, gain=1.0)
        for proj in self.task_projs:
            nn.init.zeros_(proj.weight)

    def forward(self, x, task_id):
        g_feat = self.global_proj(x)
        t_feat = self.task_projs[task_id](x) if task_id is not None else torch.zeros_like(g_feat)
        combined_feat = torch.cat([g_feat, t_feat], dim=-1)
        return F.normalize(combined_feat, p=2, dim=-1)


class FrozenBackbone(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.num_tasks = args.num_tasks
        self.lora_task_strength = args.lora_task_strength
        
        model_name_map = {
            'vit': 'vit_base_patch16_224',
            'resnet': 'resnet50',
            'convnext': 'convnext_base'
        }
        backbone_type = model_name_map.get(args.model_name, args.backbone_type)
        
        print(f"[Model] Loading {args.model_name}: {backbone_type} (LoRA task strength: {self.lora_task_strength})")
        self.backbone = timm.create_model(backbone_type, pretrained=False, num_classes=0)
        if os.path.exists(args.pretrained_vit_path):
            sd = torch.load(args.pretrained_vit_path, map_location='cpu')
            self.backbone.load_state_dict(sd, strict=False) 

        self._inject_lora(args.model_name)

        self.projector = FeatureProjector(
            input_dim=self.backbone.num_features, 
            output_dim=512, 
            num_tasks=self.num_tasks
        )
        self._freeze_all()

    def _inject_lora(self, model_name):
        for name, module in self.backbone.named_modules():
            if model_name == 'vit' and 'attn.qkv' in name:
                parent = dict(self.backbone.named_modules())[name.rsplit('.', 1)[0]]
                parent.qkv = LoRA_Linear(
                    module, 
                    self.args.adapter_dim, 
                    num_tasks=self.num_tasks,
                    lora_task_strength=self.lora_task_strength
                )
            elif model_name in ['resnet', 'convnext']:
                if isinstance(module, (nn.Linear, nn.Conv2d)):
                    if isinstance(module, nn.Linear) or (module.kernel_size[0] > 1):
                        parent_name, attr = name.rsplit('.', 1) if '.' in name else ('', name)
                        parent = dict(self.backbone.named_modules())[parent_name] if parent_name else self.backbone
                        wrapper = LoRA_Linear if isinstance(module, nn.Linear) else LoRA_Conv2d
                        setattr(parent, attr, wrapper(
                            module, 
                            self.args.adapter_dim, 
                            num_tasks=self.num_tasks,
                            lora_task_strength=self.lora_task_strength
                        ))

    def _freeze_all(self):
        for param in self.parameters():
            param.requires_grad = False
        for name, param in self.named_parameters():
            if 'lora' in name or 'projector' in name:
                param.requires_grad = True

    def forward(self, x, task_id=None):
        for module in self.modules():
            if isinstance(module, (LoRA_Linear, LoRA_Conv2d)):
                module.current_task_id = task_id
        
        feat = self.backbone(x)
        if len(feat.shape) == 4:
            feat = F.adaptive_avg_pool2d(feat, (1, 1)).flatten(1)
        
        return self.projector(feat, task_id)
