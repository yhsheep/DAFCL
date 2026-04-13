# main.py

import torch
import numpy as np
import os
import copy
import datetime
from tqdm import tqdm

from utils.args import get_args
from utils.toolkit import set_seed
from utils.simulator import AsyncSimulator
from data_loader.continual_data import ContinualDataManager
from models.backbone import FrozenBackbone 
from models.proxy_anchor_generator import ProxyAnchorGenerator 
from core.client import Client
from core.server import Server

class Logger:
    def __init__(self, log_dir, args): 
        self.args = args  
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_path = os.path.join(log_dir, f"log_DAFCL_Concat1024_{now}.txt")
        with open(self.log_path, 'w') as f:
            f.write(f"=== DAFCL: 1024D Concat & Multi-Model Adapt Mode Log: {now} ===\n")
            f.write(f"Dataset: {args.dataset} | Backbone: {args.model_name} | Num Tasks: {args.num_tasks}\n")
            f.write(f"LoRA Rank: {args.adapter_dim} | Temp: {args.temp} | LR: {args.lr}\n")

    def log(self, msg):
        print(msg)
        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')

def evaluate_all_tasks(server, data_manager, num_tasks):

    server.global_model.eval()
    accs = []
    for t_id in range(num_tasks):
        test_loader = data_manager.get_task_loader(t_id, mode='test_current')
        acc = server.test(test_loader, task_id=t_id)
        accs.append(acc)
    return accs, np.mean(accs)

def main():
    args = get_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    logger = Logger('./logs',args)
    
    data_manager = ContinualDataManager(args)
    
    logger.log(f"[System] Phase 1: Generating 1024D Anchors for {args.dataset} using {args.model_name}...")
    proxy_loader = data_manager.get_proxy_loader(batch_size=128)
    
    anchor_backbone = FrozenBackbone(args).to(device)
    anchor_gen = ProxyAnchorGenerator(args, proxy_loader, anchor_backbone)
    anchors = anchor_gen.get_anchors() 
    
    initial_projector_sd = copy.deepcopy(anchor_backbone.projector.state_dict())
    
    del anchor_backbone
    torch.cuda.empty_cache()

    global_model = FrozenBackbone(args).to(device)
    global_model.projector.load_state_dict(initial_projector_sd)
    
    server = Server(args, global_model, device)
    server.set_anchors(anchors)
    server.set_task_map(data_manager.tasks)
    
    worker_model = FrozenBackbone(args).to(device)
    worker_model.projector.load_state_dict(initial_projector_sd)
    
    logger.log(f"[System] Initializing {args.num_clients} clients for {args.dataset}...")
    clients = [Client(i, args, device, None, anchors) for i in range(args.num_clients)]
    
    simulator = AsyncSimulator(args)
    num_active = max(1, int(args.num_clients * args.active_ratio))
    total_events = args.global_rounds * args.window_size


    client_local_events = {i: 0 for i in range(args.num_clients)}
    
 
    events_per_client_per_task = max(1, total_events // (args.num_clients * args.num_tasks))
    logger.log(f"[System] Task-Switch Threshold: {events_per_client_per_task} events/task per client.")
  
    logger.log(f"[System] Bootstrapping {num_active} clients for Task 0...")
    for c_id in range(num_active):
        ideal_task_id = 0  
        t_loader = data_manager.get_task_loader(ideal_task_id, client_id=c_id, mode='train')
        
        res = clients[c_id].train(
            worker_model, 
            t_loader, 
            None, 
            server.global_model.state_dict(), 
            task_id=ideal_task_id,
            anchor_offsets=server.anchor_offsets if not args.anchor_task_isolate else server.anchor_offsets[ideal_task_id],
            old_task_loaders=None
        )
        simulator.register_event(c_id, res, 0, ideal_task_id)
        
        client_local_events[c_id] += 1

    logger.log(f"\n{'='*20} Dual-Async Training Start (Model: {args.model_name}) {'='*20}")
    logger.log(f"Anchor Task Isolation: {args.anchor_task_isolate} | LoRA Task Strength: {args.lora_task_strength}")
    logger.log(f"Memory Loss Weight: {args.lambda_memory} | Memory Task Num: {args.memory_task_num}")
    pbar = tqdm(total=args.global_rounds, desc="DAFCL Global Rounds")
    last_round = 0

    while server.global_round < args.global_rounds:
        event = simulator.get_next_event()
        if event is None: break
        
        c_id, (weights, acc_pre, acc_post, local_protos), start_round, client_task_id = event
        
        server_task_id = min(args.num_tasks - 1, int((simulator.processed_events / total_events) * args.num_tasks))
        

        staleness = server.global_round - start_round
        omega_g, semantic_sim = server.aggregate(weights, staleness, client_task_id, server_task_id, local_protos)
        
        if server.global_round > last_round:
            pbar.update(server.global_round - last_round)
            last_round = server.global_round
            
            task_accs, avg_acc = evaluate_all_tasks(server, data_manager, args.num_tasks)
            if args.anchor_task_isolate:
                offset_norm = np.mean([torch.norm(server.anchor_offsets[t]).item() for t in range(args.num_tasks)])
            else:
                offset_norm = torch.norm(server.anchor_offsets).item()
            
            logger.log(f"\n[EVENT] Round {server.global_round} | Events: {simulator.processed_events} | Client {c_id}")
            logger.log(f"  - Server Task: {server_task_id} | Client Task: {client_task_id}")
            logger.log(f"  - Semantic Sim: {semantic_sim:.4f} | Anchor Offset Norm: {offset_norm:.4f}")
            formatted_accs = [round(a, 1) for a in task_accs]
            logger.log(f"  - GLOBAL EVAL: Avg {avg_acc:.2f}% | Task Accs: {formatted_accs}")

        while len(simulator.in_flight_clients) < num_active:
            idle_list = [i for i in range(args.num_clients) if not simulator.is_client_busy(i)]
            if not idle_list: break
            
            next_c_id = np.random.choice(idle_list)
            

            ideal_task_id = min(args.num_tasks - 1, client_local_events[next_c_id] // events_per_client_per_task)

            
            t_loader = data_manager.get_task_loader(ideal_task_id, client_id=next_c_id, mode='train')
            
            old_task_loaders = {}
            if ideal_task_id > 0:
                for t in range(max(0, ideal_task_id - args.memory_task_num), ideal_task_id):
                    old_task_loaders[t] = data_manager.get_task_loader(t, client_id=next_c_id, mode='train')
            
            if args.anchor_task_isolate:
                anchor_offset = server.anchor_offsets[ideal_task_id]
            else:
                anchor_offset = server.anchor_offsets
            
            new_res = clients[next_c_id].train(
                worker_model, 
                t_loader, 
                None, 
                server.global_model.state_dict(), 
                ideal_task_id,
                anchor_offsets=server.anchor_offsets,  
                old_task_loaders=old_task_loaders     
            )
            
            simulator.register_event(next_c_id, new_res, server.global_round, ideal_task_id)
            
            client_local_events[next_c_id] += 1

    pbar.close()
    
    final_accs, final_avg = evaluate_all_tasks(server, data_manager, args.num_tasks)
    logger.log(f"\n{'='*20} Final Experiment Report {'='*20}")
    logger.log(f"Dataset: {args.dataset} | Backbone: {args.model_name}")
    logger.log(f"Anchor Task Isolation: {args.anchor_task_isolate} | LoRA Task Strength: {args.lora_task_strength}")
    logger.log(f"Memory Loss Weight: {args.lambda_memory} | Final Average Accuracy: {final_avg:.2f}%")
    logger.log(f"Final Per-Task Accuracy: {[round(a, 2) for a in final_accs]}")

if __name__ == "__main__":
    main()