import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class Client:
    def __init__(self, client_id, args, device, model_placeholder, anchors):
       
        self.client_id = client_id
        self.args = args
        self.device = device
        self.model = None  
        self.anchors = anchors.to(device)
        self.old_task_protos = {}  

    def local_evaluate(self, loader, task_id):
        if loader is None: return 0.0
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                features = self.model(images, task_id=task_id)
                logits = features @ self.anchors.T / self.args.temp
                _, pred = torch.max(logits, 1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)
        return 100 * correct / total if total > 0 else 0.0

    def _compute_memory_loss(self, features, task_id):
        if task_id == 0 or not self.old_task_protos:
            return 0.0
        
        old_task_ids = [t for t in self.old_task_protos.keys() if t < task_id]
        old_task_ids = old_task_ids[-self.args.memory_task_num:] if old_task_ids else []
        if not old_task_ids:
            return 0.0
        
        memory_loss = 0.0
        for old_t in old_task_ids:
            old_protos = self.old_task_protos[old_t]
            old_labels = list(old_protos.keys())
            if not old_labels:
                continue
            sample_labels = np.random.choice(old_labels, min(10, len(old_labels)), replace=False)
            
            old_proto_tensor = torch.stack([old_protos[l].to(self.device) for l in sample_labels])
            sim_matrix = F.cosine_similarity(features.unsqueeze(1), old_proto_tensor.unsqueeze(0), dim=-1)
            memory_loss += (1 - sim_matrix.mean()).clamp(min=0)
        
        return memory_loss / len(old_task_ids) if old_task_ids else 0.0

    def train(self, model, task_loader, test_loader, global_model_state_dict, task_id, anchor_offsets=None, old_task_loaders=None):

        self.model = model
        self.model.load_state_dict(global_model_state_dict, strict=False)
        self.model.to(self.device)
        

        current_anchors = self.anchors
        if anchor_offsets is not None:
            if isinstance(anchor_offsets, dict):
                current_anchors = self.anchors + anchor_offsets.get(task_id, torch.zeros_like(self.anchors)).to(self.device)
            else:
                current_anchors = self.anchors + anchor_offsets.to(self.device)

        acc_before = self.local_evaluate(test_loader, task_id)
        half_epochs = max(1, self.args.local_epochs // 2)

        for name, p in self.model.named_parameters():
            if 'lora' in name and ('global' in name or f'.{task_id}.' in name or name.endswith(f'.{task_id}')):
                p.requires_grad = True
            else:
                p.requires_grad = False
        
        optimizer_lora = optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), 
                                    lr=self.args.lr, weight_decay=self.args.weight_decay)

        self.model.train()
        for _ in range(half_epochs):
            for images, labels in task_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer_lora.zero_grad()
                features = self.model(images, task_id=task_id)
                
                logits = features @ current_anchors.T / self.args.temp
                loss_ce = F.cross_entropy(logits, labels)
                
                target_anchors = current_anchors[labels]
                loss_compact = 1 - F.cosine_similarity(features, target_anchors).mean()
                
                loss_memory = self._compute_memory_loss(features, task_id)

                total_loss = loss_ce + self.args.lambda_kd * loss_compact + self.args.lambda_memory * loss_memory
                total_loss.backward()
                optimizer_lora.step()


        for name, p in self.model.named_parameters():
            if 'projector' in name and ('global' in name or f'.{task_id}.' in name or name.endswith(f'.{task_id}')):
                p.requires_grad = True
            else:
                p.requires_grad = False

        optimizer_proj = optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), 
                                    lr=self.args.lr, weight_decay=self.args.weight_decay)

        for _ in range(self.args.local_epochs - half_epochs):
            for images, labels in task_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer_proj.zero_grad()
                features = self.model(images, task_id=task_id)
                
                logits = features @ current_anchors.T / self.args.temp
                loss_ce = F.cross_entropy(logits, labels)
                
                target_anchors = current_anchors[labels]
                loss_compact = 1 - F.cosine_similarity(features, target_anchors).mean()
                loss_memory = self._compute_memory_loss(features, task_id)
                
                total_loss = loss_ce + self.args.lambda_kd * loss_compact + self.args.lambda_memory * loss_memory
                total_loss.backward()
                optimizer_proj.step()


        self.model.eval()
        local_protos = {}
        with torch.no_grad():
            class_features = {}
            for images, labels in task_loader:
                images = images.to(self.device)
                features = self.model(images, task_id=task_id)
                for feat, label in zip(features, labels):
                    l = label.item()
                    if l not in class_features: class_features[l] = []
                    class_features[l].append(feat)
            
            for l, feats in class_features.items():
                local_protos[l] = torch.stack(feats).mean(dim=0).cpu()
        

        self.old_task_protos[task_id] = local_protos

        acc_after = self.local_evaluate(test_loader, task_id)
        
        deltas = {k: v.cpu().clone() for k, v in self.model.state_dict().items() 
                  if ('lora' in k or 'projector' in k)}
                
        self.model = None
        
        return deltas, acc_before, acc_after, local_protos
