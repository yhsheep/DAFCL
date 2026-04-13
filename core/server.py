import torch
import torch.nn.functional as F
import numpy as np
from copy import deepcopy

class Server:
    def __init__(self, args, global_model, device):
        self.args = args
        self.global_model = global_model
        self.global_model.to(device)
        self.device = device
        
        self.global_round = 0
        self.fixed_anchors = None  
        self.anchor_offsets = None 
        self.task_class_map = {} 

        self.window_size = getattr(args, 'window_size', 5) 
        self.update_buffer = [] 
        self.server_momentum = getattr(args, 'momentum', 0.5) 

        self.global_velocities = {
            k: torch.zeros_like(v).to(device) 
            for k, v in self.global_model.state_dict().items() 
            if 'lora' in k or 'projector' in k
        }

    def set_anchors(self, anchors):
        self.fixed_anchors = anchors.to(self.device)
        if self.args.anchor_task_isolate:
            self.anchor_offsets = {
                t_id: torch.zeros_like(self.fixed_anchors).to(self.device) 
                for t_id in range(self.args.num_tasks)
            }
        else:
            self.anchor_offsets = torch.zeros_like(self.fixed_anchors).to(self.device)

    def set_task_map(self, tasks_list):
        for t_id, classes in enumerate(tasks_list):
            self.task_class_map[t_id] = torch.tensor(classes).to(self.device)

    def aggregate(self, client_weight, staleness, client_task_id, server_task_id, local_protos):
 
        if self.args.anchor_task_isolate:
            dynamic_anchors = self.fixed_anchors + self.anchor_offsets[client_task_id]
        else:
            dynamic_anchors = self.fixed_anchors + self.anchor_offsets
        
        align_scores = []
        for label, proto in local_protos.items():
            proto = proto.to(self.device).view(1, -1)
            anchor = dynamic_anchors[label].view(1, -1)
            align_scores.append(F.cosine_similarity(proto, anchor).item())
        
        avg_align = np.mean(align_scores) if align_scores else 0.5
        semantic_reliability = np.clip((1 + avg_align) / 2, 0, 1)

        time_decay = (staleness + 1) ** (-getattr(self.args, 'staleness_alpha', 0.5))
        stage_discrepancy = np.exp(-getattr(self.args, 'sigma', 1.0) * abs(server_task_id - client_task_id))
        
        omega_i = time_decay * stage_discrepancy * semantic_reliability
        
        gamma = getattr(self.args, 'gamma', 0.1)
        epsilon = getattr(self.args, 'gate_floor', 0.1)
        omega_g = epsilon + (1 - epsilon) * omega_i * np.exp(-gamma * abs(server_task_id - client_task_id))

        global_dict = self.global_model.state_dict()
        current_client_deltas = {}
        for k, v in client_weight.items():
            if k in global_dict:
                scale = omega_g if 'global' in k else omega_i
                delta = (v.to(self.device) - global_dict[k]) * scale
                current_client_deltas[k] = delta

        self.update_buffer.append(current_client_deltas)

        if len(self.update_buffer) >= self.window_size:
            for k in self.global_velocities.keys():
                deltas_in_window = [upd[k] for upd in self.update_buffer if k in upd]
                if not deltas_in_window: continue
                
                avg_delta = torch.stack(deltas_in_window).mean(dim=0)
                self.global_velocities[k] = self.server_momentum * self.global_velocities[k] + avg_delta
                global_dict[k] += self.args.alpha * self.global_velocities[k]
            
            self.global_model.load_state_dict(global_dict)
            self.global_round += 1
            self.update_buffer = []
            
            self._calibrate_anchors(local_protos, omega_i, client_task_id)

        return omega_g, avg_align

    def _calibrate_anchors(self, local_protos, omega_i, client_task_id):

        eta_a = getattr(self.args, 'anchor_lr', 0.01)
        threshold = getattr(self.args, 'delta_threshold', 0.7)
        
        if self.args.anchor_task_isolate:
            anchor_offsets = self.anchor_offsets[client_task_id]
        else:
            anchor_offsets = self.anchor_offsets
        
        for label, proto in local_protos.items():
            proto = proto.to(self.device)
            base_anchor = self.fixed_anchors[label]
            
            if F.cosine_similarity(proto.view(1,-1), base_anchor.view(1,-1)).item() > threshold:
                if self.args.anchor_task_isolate:
                    adyn = base_anchor + anchor_offsets[label]
                else:
                    adyn = base_anchor + anchor_offsets[label]
                target_diff = proto - adyn
                anchor_offsets[label] += eta_a * omega_i * target_diff
        
        if self.args.anchor_task_isolate:
            self.anchor_offsets[client_task_id] = anchor_offsets

    def test(self, data_loader, task_id):
        self.global_model.eval()
        if self.args.anchor_task_isolate:
            dynamic_anchors = self.fixed_anchors + self.anchor_offsets[task_id]
        else:
            dynamic_anchors = self.fixed_anchors + self.anchor_offsets
            
        correct, total = 0, 0
        
        with torch.no_grad():
            for images, labels in data_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                features = self.global_model(images, task_id=task_id)
                
                logits = features @ dynamic_anchors.T / self.args.temp
                _, pred = torch.max(logits, dim=1)
                
                correct += (pred == labels).sum().item()
                total += labels.size(0)
        return 100 * correct / total if total > 0 else 0.0
