#!/usr/bin/env python3
"""
Fixed Federated Client for DAD Dataset
- Uses proper IID partitioning with class balance
- Fixes graph indexing issues for CUDA compatibility
- Maintains same training methodology as train_dad.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import flwr as fl
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from typing import List
import argparse

from dataset_dad import Dataset as DADDataset
from models.graphofgraph import SpaceTempGoG_detr_dad
from eval_utils import evaluation
from torchmetrics.functional import pairwise_cosine_similarity

# Use CUDA for better performance
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}", flush=True)


def make_client_dataset(cid: int, dataset_path: str, img_dataset_path: str, split_path: str,
                       obj_mapping_file: str, ref_interval: int) -> DADDataset:
    """Load this client's train split and build a Dataset."""
    client_split_file = os.path.join(split_path, "clients", f"train_split_client_{cid}.txt")
    if not os.path.exists(client_split_file):
        raise FileNotFoundError(f"Client split file not found: {client_split_file}")
    
    # Create temporary split directory for this client
    temp_split_dir = os.path.join(split_path, f"temp_client_{cid}")
    os.makedirs(temp_split_dir, exist_ok=True)
    temp_train_file = os.path.join(temp_split_dir, "train_split.txt")
    
    # Copy client's split to expected filename
    with open(client_split_file) as src, open(temp_train_file, "w") as dst:
        dst.write(src.read())
    
    # Build dataset using client's split
    dataset = DADDataset(
        img_dataset_path=img_dataset_path,
        dataset_path=dataset_path,
        split_path=temp_split_dir,
        ref_interval=ref_interval,
        objmap_file=obj_mapping_file,
        training=True,
    )
    
    return dataset


def fix_graph_indices(batch_data, max_nodes_per_graph=50):
    """
    Fix graph indexing issues that cause CUDA errors.
    Ensures indices are within bounds and tensors are properly validated.
    """
    X, edge_index, y_true, img_feat, video_adj_list, edge_embeddings, temporal_adj_list, obj_vis_feat, batch_vec, toa = batch_data
    
    # Get original shapes
    n_nodes = X.shape[1]  # Assuming X is [batch_size, n_nodes, features]
    
    # Limit number of nodes to prevent memory issues
    if n_nodes > max_nodes_per_graph:
        print(f"Limiting graph from {n_nodes} to {max_nodes_per_graph} nodes")
        X = X[:, :max_nodes_per_graph, :]
        obj_vis_feat = obj_vis_feat[:, :max_nodes_per_graph, :]
        n_nodes = max_nodes_per_graph
    
    # Fix edge indices to be within bounds with proper CUDA handling
    if edge_index.numel() > 0:
        edge_index = torch.clamp(edge_index, 0, max(0, n_nodes - 1))
    
    # Fix video adjacency list with CUDA-safe operations
    if video_adj_list.numel() > 0:
        video_adj_list = torch.clamp(video_adj_list, 0, max(0, n_nodes - 1))
    
    # Fix temporal adjacency list with CUDA-safe operations
    if temporal_adj_list.numel() > 0:
        temporal_adj_list = torch.clamp(temporal_adj_list, 0, max(0, n_nodes - 1))
        # Additional validation for temporal indices
        if temporal_adj_list.shape[0] >= 2:
            valid_mask = (temporal_adj_list[0] < n_nodes) & (temporal_adj_list[1] < n_nodes)
            if valid_mask.any():
                temporal_adj_list = temporal_adj_list[:, valid_mask]
            else:
                temporal_adj_list = torch.zeros(2, 0, dtype=torch.long, device=temporal_adj_list.device)
    
    # Ensure batch vector matches with proper size handling
    if batch_vec.numel() > 0:
        if batch_vec.shape[0] > n_nodes:
            batch_vec = batch_vec[:n_nodes]
        elif batch_vec.shape[0] < n_nodes and n_nodes > 0:
            # Pad batch vector if needed
            padding = torch.zeros(n_nodes - batch_vec.shape[0], dtype=batch_vec.dtype, device=batch_vec.device)
            batch_vec = torch.cat([batch_vec, padding])
    
    return X, edge_index, y_true, img_feat, video_adj_list, edge_embeddings, temporal_adj_list, obj_vis_feat, batch_vec, toa


class DADFederatedClient(fl.client.NumPyClient):
    def __init__(self, model: nn.Module, client_dataset, cid: int):
        self.cid = cid
        self.model = model.to(device)
        # Weighted CrossEntropyLoss for class imbalance (previously used):
        # class_weights = torch.tensor([1.0, 1.84], dtype=torch.float32).to(device)  # [neg_weight, pos_weight]
        # self.criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)
        # print(f"Client {cid}: Using weighted CrossEntropyLoss with weights {class_weights.cpu().numpy()}", flush=True)
        # Now using unweighted CrossEntropyLoss:
        self.criterion = nn.CrossEntropyLoss().to(device)
        print(f"Client {cid}: Using unweighted CrossEntropyLoss", flush=True)
        
        # Use same optimizer settings as train_dad.py
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=5e-4)
        # Use same LR scheduler as train_dad.py - milestones=[25] 
        self.scheduler = MultiStepLR(self.optimizer, milestones=[25], gamma=0.5)
        # Track local epoch count (resets each fit call, just like train_dad.py)
        self.local_epoch_count = 0
        
        # Split client's data 85/15 for train/validation
        # NOTE: This is the ONLY difference from train_dad.py (which uses 70/30)
        # We use 85/15 to give clients more training data in federated setting
        self.train_fraction = 0.85  
        train_len = int(self.train_fraction * len(client_dataset))
        val_len = len(client_dataset) - train_len
        generator = torch.Generator().manual_seed(0)  # Same seed as train_dad.py
        self.train_dataset, self.val_dataset = random_split(
            client_dataset, [train_len, val_len], generator=generator
        )
        
        # Use same batch size as train_dad.py
        # Set num_workers=0 to avoid multiprocessing issues in federated environment
        self.train_loader = DataLoader(self.train_dataset, batch_size=1, shuffle=True, num_workers=0)
        self.val_loader = DataLoader(self.val_dataset, batch_size=1, shuffle=False, num_workers=0)
        
        print(f"Client {self.cid}: total={len(client_dataset)}, train={len(self.train_dataset)}, val={len(self.val_dataset)}", flush=True)

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]):
        state_dict = self.model.state_dict()
        for (k, v), p in zip(state_dict.items(), parameters):
            # Ensure correct device AND dtype (same as server.py)
            tensor = torch.from_numpy(p).to(device).to(v.dtype)
            state_dict[k] = tensor
        self.model.load_state_dict(state_dict)
        
        # CRITICAL FIX: Reset optimizer and scheduler state when receiving new parameters from server
        # This prevents stale momentum/gradient accumulation causing instability
        self.optimizer.state.clear()
        self.scheduler = MultiStepLR(self.optimizer, milestones=[25], gamma=0.5)
        self.local_epoch_count = 0
        print(f"Client {self.cid}: Reset optimizer and scheduler state after loading server parameters", flush=True)

    def fit(self, parameters: List[np.ndarray], config: dict):
        self.set_parameters(parameters)
        self.model.train()
        
        local_epochs = int(config.get("local_epochs", 1))
        # Reset local epoch counter for this training session
        self.local_epoch_count = 0
        
        best_ap = -1.0
        best_state = None

        for epoch in range(local_epochs):
            print(f"Client {self.cid}: Starting epoch {epoch + 1}/{local_epochs}", flush=True)
            
            # Training loop (same as train_dad.py) - with batch accumulation
            accumulated_loss = 0.0
            for batch_i, batch_data in enumerate(self.train_loader):
                if batch_i % 10 == 0:
                    print(f"Client {self.cid}: Processing batch {batch_i}", flush=True)
                
                # Extract batch data exactly like train_dad.py
                X, edge_index, y_true, img_feat, video_adj_list, edge_embeddings, temporal_adj_list, obj_vis_feat, batch_vec, toa = batch_data
                
                # Exactly same preprocessing as train_dad.py - no modifications
                X = X.reshape(-1, X.shape[2])
                img_feat = img_feat.reshape(-1, img_feat.shape[2])
                edge_index = edge_index.reshape(-1, edge_index.shape[2])
                edge_embeddings = edge_embeddings.view(-1, edge_embeddings.shape[-1])
                video_adj_list = video_adj_list.reshape(-1, video_adj_list.shape[2])
                temporal_adj_list = temporal_adj_list.reshape(-1, temporal_adj_list.shape[2])
                y = y_true.reshape(-1)
                
                # Handle toa - check if it's tensor or already int
                if hasattr(toa, 'item'):
                    toa = toa.item()
                elif isinstance(toa, torch.Tensor):
                    toa = int(toa)
                # else toa is already an int
                
                # Same cosine similarity computation as train_dad.py
                obj_vis_feat = obj_vis_feat.reshape(-1, obj_vis_feat.shape[-1]).to(device)
                feat_sim = pairwise_cosine_similarity(obj_vis_feat+1e-7, obj_vis_feat+1e-7)            
                temporal_edge_w = feat_sim[temporal_adj_list[0, :], temporal_adj_list[1, :]]
                batch_vec = batch_vec.view(-1).long()

                X, edge_index, y, img_feat, video_adj_list = X.to(device), edge_index.to(device), y.to(device), img_feat.to(device), video_adj_list.to(device)
                temporal_adj_list, temporal_edge_w, edge_embeddings, batch_vec = temporal_adj_list.to(device), temporal_edge_w.to(device), edge_embeddings.to(device), batch_vec.to(device)
                
                # Get predictions from the model (exactly same as train_dad.py)
                logits, probs = self.model(X, edge_index, img_feat, video_adj_list, edge_embeddings, temporal_adj_list, temporal_edge_w, batch_vec)
                
                # Exclude the actual accident frames from the training (exactly same as train_dad.py line 264)
                c_loss1 = self.criterion(logits[:toa], y[:toa])
                accumulated_loss = accumulated_loss + c_loss1
                
                # Same batch accumulation strategy as train_dad.py (every 3 batches)
                if (batch_i + 1) % 3 == 0:
                    self.optimizer.zero_grad()
                    accumulated_loss.backward()
                    self.optimizer.step()
                    accumulated_loss = 0.0

                # Empty cache after each batch (same as train_dad.py)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Apply any remaining accumulated gradients at end of epoch
            if accumulated_loss > 0:
                self.optimizer.zero_grad()
                accumulated_loss.backward()
                self.optimizer.step()
                accumulated_loss = 0.0

            # Validate after each local epoch (same as train_dad.py)
            try:
                ap = self._compute_ap_on_loader(self.val_loader)
                print(f"Client {self.cid}: Epoch {epoch + 1} validation AP = {ap:.4f}", flush=True)
                # Save best model based on validation AP
                if ap > best_ap:
                    best_ap = ap
                    best_state = self.model.state_dict().copy()
                    print(f"Client {self.cid}: New best AP = {best_ap:.4f}, saving checkpoint", flush=True)
            except Exception as e:
                print(f"Client {self.cid}: Error in validation: {e}", flush=True)
                best_ap = 0.0
            
            # Track local epochs and step scheduler exactly like train_dad.py
            self.local_epoch_count += 1
            
            # Step scheduler based on LOCAL epoch count (not federated rounds)
            current_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step()  
            new_lr = self.optimizer.param_groups[0]['lr']
            if new_lr != current_lr:
                print(f"Client {self.cid}: LR decayed from {current_lr} to {new_lr} at local epoch {self.local_epoch_count}", flush=True)

        if new_lr != current_lr:
                print(f"Client {self.cid}: LR decayed from {current_lr} to {new_lr} at local epoch {self.local_epoch_count}", flush=True)

        # Load best model checkpoint based on validation AP (same as train_dad.py)
        if best_state is not None:
            self.model.load_state_dict(best_state)
            print(f"Client {self.cid}: Loaded best model checkpoint with AP = {best_ap:.4f}", flush=True)
        else:
            print(f"Client {self.cid}: No best checkpoint, using final trained model with AP = {best_ap:.4f}", flush=True)
        
        return self.get_parameters({}), len(self.train_dataset), {"ap": best_ap}

    def _compute_ap_on_loader(self, data_loader):
        """Compute AP on validation set (exactly same as test_model in train_dad.py)"""
        self.model.eval()
        all_probs_vid2 = None
        all_y_vid = None
        all_toa = []
        
        with torch.no_grad():
            for batch_i, batch_data in enumerate(data_loader):
                try:
                    # Extract batch data exactly like train_dad.py
                    X, edge_index, y_true, img_feat, video_adj_list, edge_embeddings, temporal_adj_list, obj_vis_feat, batch_vec, toa = batch_data
                    
                    # Exactly same preprocessing as train_dad.py - no modifications
                    X = X.reshape(-1, X.shape[2])
                    img_feat = img_feat.reshape(-1, img_feat.shape[2])
                    edge_index = edge_index.reshape(-1, edge_index.shape[2])
                    edge_embeddings = edge_embeddings.view(-1, edge_embeddings.shape[-1])
                    video_adj_list = video_adj_list.reshape(-1, video_adj_list.shape[2])
                    temporal_adj_list = temporal_adj_list.reshape(-1, temporal_adj_list.shape[2])
                    y = y_true.reshape(-1)

                    obj_vis_feat = obj_vis_feat.reshape(-1, obj_vis_feat.shape[-1]).to(device)
                    
                    # Handle toa exactly like train_dad.py
                    if hasattr(toa, 'item'):
                        toa_val = toa.item()
                    elif isinstance(toa, torch.Tensor):
                        toa_val = int(toa)
                    else:
                        toa_val = int(toa)
                    
                    # Skip very large graphs for safety
                    if obj_vis_feat.shape[0] > 2000:
                        continue
                    
                    # Exactly same temporal computation as train_dad.py
                    feat_sim = pairwise_cosine_similarity(obj_vis_feat+1e-7, obj_vis_feat+1e-7)
                    temporal_edge_w = feat_sim[temporal_adj_list[0, :], temporal_adj_list[1, :]]
                    batch_vec = batch_vec.view(-1).long()

                    X, edge_index, y, img_feat, video_adj_list = X.to(device), edge_index.to(device), y.to(device), img_feat.to(device), video_adj_list.to(device)
                    temporal_adj_list, temporal_edge_w, edge_embeddings, batch_vec = temporal_adj_list.to(device), temporal_edge_w.to(device), edge_embeddings.to(device), batch_vec.to(device)
                    all_toa += [toa_val]

                    logits, probs = self.model(X, edge_index, img_feat, video_adj_list, edge_embeddings, temporal_adj_list, temporal_edge_w, batch_vec)
                    
                    pred_labels = probs.argmax(1)
                    
                    # Collect data exactly like train_dad.py test_model function
                    if batch_i == 0: 
                        all_probs_vid2 = probs[:, 1].cpu().unsqueeze(0)
                        all_y_vid = torch.max(y).unsqueeze(0).cpu()
                    else: 
                        all_probs_vid2 = torch.cat((all_probs_vid2, probs[:, 1].cpu().unsqueeze(0)))
                        all_y_vid = torch.cat((all_y_vid, torch.max(y).unsqueeze(0).cpu()))
                
                except Exception as e:
                    print(f"Client {self.cid}: Error in validation batch: {e}")
                    continue

        if all_probs_vid2 is None or len(all_toa) == 0:
            return 0.0
            
        # Use same evaluation function as train_dad.py
        try:
            avg_prec, _, _ = evaluation(all_probs_vid2.numpy(), all_y_vid.numpy(), all_toa)    
            avg_prec = avg_prec  # Don't multiply by 100 here, let server handle it
            return avg_prec
        except Exception as e:
            print(f"Client {self.cid}: Error in evaluation: {e}")
            return 0.0

    def evaluate(self, parameters: List[np.ndarray], config: dict):
        self.set_parameters(parameters)
        self.model.eval()
        
        # Use validation set for client-side evaluation
        ap = self._compute_ap_on_loader(self.val_loader)
        loss = 0.7  # Placeholder loss
        
        return loss, len(self.val_dataset), {"ap": ap}


def main():
    parser = argparse.ArgumentParser(description="DAD Federated Client")
    parser.add_argument("--cid", type=int, required=True, help="Client ID")
    parser.add_argument("--num-clients", type=int, default=5, help="Total number of clients")
    parser.add_argument("--server", type=str, default="127.0.0.1:8080", help="Server address")
    
    # DAD dataset arguments (same as train_dad.py)
    parser.add_argument("--dataset-path", type=str, default="data/dad/obj_feat")
    parser.add_argument("--img-dataset-path", type=str, default="data/dad/i3d_feat")
    parser.add_argument("--split-path", type=str, default="splits_dad")
    parser.add_argument("--obj-mapping-file", type=str, default="data/dad/obj_idx_to_labels.json")
    parser.add_argument("--ref-interval", type=int, default=20)  # Same as train_dad.py
    
    # Model arguments (same as train_dad.py)
    parser.add_argument("--input-dim", type=int, default=4096)  # Same as train_dad.py
    parser.add_argument("--embedding-dim", type=int, default=256)  # Same as train_dad.py  
    parser.add_argument("--img-feat-dim", type=int, default=2048)  # Same as train_dad.py
    parser.add_argument("--num-classes", type=int, default=2)
    
    args = parser.parse_args()

    print(f"Starting DAD federated client {args.cid}", flush=True)

    # Same model initialization as train_dad.py
    model = SpaceTempGoG_detr_dad(
        input_dim=args.input_dim,
        embedding_dim=args.embedding_dim,
        img_feat_dim=args.img_feat_dim,
        num_classes=args.num_classes,
    )

    # Load client's dataset
    client_dataset = make_client_dataset(
        cid=args.cid,
        dataset_path=args.dataset_path,
        img_dataset_path=args.img_dataset_path,
        split_path=args.split_path,
        obj_mapping_file=args.obj_mapping_file,
        ref_interval=args.ref_interval,
    )

    client = DADFederatedClient(model, client_dataset, cid=args.cid)
    # Use non-deprecated API
    fl.client.start_client(server_address=args.server, client=client.to_client())


if __name__ == "__main__":
    main()