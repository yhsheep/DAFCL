import torch
import os
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
import torch.nn.functional as F

class ProxyAnchorGenerator:
    def __init__(self, args, proxy_loader, backbone):
 
        self.args = args
        self.device = args.device
        self.proxy_loader = proxy_loader
        self.num_slots = args.num_classes 
        
        self.save_path = os.path.join(
            args.output_dir, 
            f"concat_anchors_{args.dataset}_{args.model_name}.pt"
        )
        self.backbone = backbone

        if os.path.exists(self.save_path):
            print(f"[System] Loading cached 1024D anchors from {self.save_path}...")
            self.anchors = torch.load(self.save_path, map_location=self.device)
        else:
            self.anchors = self._generate_slots()

    def _generate_slots(self):
        self.backbone.eval()
        all_features = []
        
        print(f"[Phase 1] Extracting 1024D features using {self.args.model_name}...")
        with torch.no_grad():
            for images, _ in tqdm(self.proxy_loader, desc="Feature Extraction"):
                images = images.to(self.device)
                
                features = self.backbone(images, task_id=None)
                
                features = F.normalize(features, p=2, dim=1)
                all_features.append(features.cpu())
        
        features_np = torch.cat(all_features, dim=0).numpy()
        
        print(f"[Phase 1] Running K-Means (K={self.num_slots}) for {self.args.dataset}...")
        kmeans = KMeans(
            n_clusters=self.num_slots, 
            random_state=42, 
            n_init=10,
            max_iter=300
        )
        cluster_centers = kmeans.fit(features_np).cluster_centers_
        
        final_anchors = torch.from_numpy(cluster_centers).float().to(self.device)
        final_anchors = F.normalize(final_anchors, p=2, dim=1)
        
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        torch.save(final_anchors, self.save_path)
        print(f"[System] 1024D anchors for {self.args.dataset} generated.")
        
        return final_anchors

    def get_anchors(self):
        return self.anchors