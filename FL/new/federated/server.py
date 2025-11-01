import argparse
from typing import Callable, Dict, List, Tuple
import os
import sys

# Ensure project root is on sys.path for imports like `models` and `dataset_dad`
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from models.graphofgraph import SpaceTempGoG_detr_dad
from dataset_dad import Dataset as DADDataset
import torch.nn.functional as F
from eval_utils import evaluation
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc
from torchmetrics.functional import pairwise_cosine_similarity


def get_evaluate_fn(model: nn.Module, test_loader: DataLoader, device: torch.device, results_dir: str) -> Callable:
    """Return a central evaluation function for the server.
    
    Evaluates on global test_split.txt using same metrics as train_dad.py
    """
    # Weighted CrossEntropyLoss for class imbalance (previously used):
    # class_weights = torch.tensor([1.0, 1.84], dtype=torch.float32).to(device)
    # criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)
    # print(f"Server: Using weighted CrossEntropyLoss with weights {class_weights.cpu().numpy()}", flush=True)
    # Now using unweighted CrossEntropyLoss:
    criterion = nn.CrossEntropyLoss().to(device)
    print(f"Server: Using unweighted CrossEntropyLoss", flush=True)

    def evaluate(server_round: int, parameters: List[np.ndarray], config: Dict) -> Tuple[float, Dict[str, float]]:
        # Load parameters into model (same weight initialization as train_dad.py)
        state_dict = model.state_dict()
        for (k, v), p in zip(state_dict.items(), parameters):
            tensor = torch.from_numpy(p).to(device).to(v.dtype)  # Ensure correct device and dtype
            state_dict[k] = tensor
        model.load_state_dict(state_dict)
        model.eval()

        # For AP/mTTA/TTA@R80 metrics (exactly same as train_dad.py test_model)
        all_probs_vid2 = None
        all_pred = None
        all_y = None
        all_y_vid = None
        all_toa = []
        
        with torch.no_grad():
            for batch_i, (X, edge_index, y_true, img_feat, video_adj_list, edge_embeddings, temporal_adj_list, obj_vis_feat, batch_vec, toa) in enumerate(test_loader):
                # Same preprocessing as train_dad.py
                X = X.reshape(-1, X.shape[2])
                img_feat = img_feat.reshape(-1, img_feat.shape[2])
                edge_index = edge_index.reshape(-1, edge_index.shape[2])
                edge_embeddings = edge_embeddings.view(-1, edge_embeddings.shape[-1])
                video_adj_list = video_adj_list.reshape(-1, video_adj_list.shape[2])
                temporal_adj_list = temporal_adj_list.reshape(-1, temporal_adj_list.shape[2])
                y = y_true.reshape(-1)

                obj_vis_feat = obj_vis_feat.reshape(-1, obj_vis_feat.shape[-1]).to(device)
                # Exactly same temporal computation as train_dad.py and client.py
                feat_sim = pairwise_cosine_similarity(obj_vis_feat+1e-7, obj_vis_feat+1e-7)
                temporal_edge_w = feat_sim[temporal_adj_list[0, :], temporal_adj_list[1, :]]
                batch_vec = batch_vec.view(-1).long()

                X, edge_index, y, img_feat, video_adj_list = X.to(device), edge_index.to(device), y.to(device), img_feat.to(device), video_adj_list.to(device)
                temporal_adj_list, temporal_edge_w, edge_embeddings, batch_vec = temporal_adj_list.to(device), temporal_edge_w.to(device), edge_embeddings.to(device), batch_vec.to(device)

                logits, probs = model(X, edge_index, img_feat, video_adj_list, edge_embeddings, temporal_adj_list, temporal_edge_w, batch_vec)
                
                # EXACTLY same as train_dad.py: ALL frames for predictions and confusion matrix
                pred_labels = probs.argmax(1)  # ALL frames like train_dad.py

                # Debug: Print some statistics for first few batches
                if batch_i < 3:
                    pos_prob = probs[:, 1].cpu()
                    print(f"Batch {batch_i}: probs shape={probs.shape}, pos_prob range=[{pos_prob.min():.4f}, {pos_prob.max():.4f}]")
                    print(f"Batch {batch_i}: pred_labels unique={torch.unique(pred_labels).cpu().numpy()}, y unique={torch.unique(y).cpu().numpy()}")

                # Accumulate for AP/mTTA/TTA@R80 computation (same as train_dad.py)
                if batch_i == 0:
                    all_probs_vid2 = probs[:, 1].cpu().unsqueeze(0)
                    all_pred = pred_labels.cpu()  # ALL frames
                    all_y = y.cpu()  # ALL frame labels
                    all_y_vid = torch.max(y).unsqueeze(0).cpu()  # Video label from ALL frames
                else:
                    all_probs_vid2 = torch.cat((all_probs_vid2, probs[:, 1].cpu().unsqueeze(0)))
                    all_pred = torch.cat((all_pred, pred_labels.cpu()))  # ALL frames
                    all_y = torch.cat((all_y, y.cpu()))  # ALL frame labels  
                    all_y_vid = torch.cat((all_y_vid, torch.max(y).unsqueeze(0).cpu()))  # Video label from ALL frames
                
                all_toa.append(toa.item())  # Store toa as float like train_dad.py

                # Empty cache (same as train_dad.py)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        metrics: Dict[str, float] = {}

        # Compute same metrics as train_dad.py using eval_utils
        try:
            if all_probs_vid2 is not None and all_y_vid is not None:
                os.makedirs(os.path.join(results_dir, "federated"), exist_ok=True)
                all_probs_vid2_np = all_probs_vid2.numpy()
                all_pred_np = all_pred.numpy()
                all_y_np = all_y.numpy()
                all_y_vid_np = all_y_vid.numpy()

                # Use eval_utils.evaluation (same as train_dad.py)
                AP, mTTA, TTA_R80 = evaluation(all_probs_vid2_np, all_y_vid_np, all_toa)
                metrics.update({
                    "ap": float(AP),
                    "mtta": float(mTTA),
                    "tta_r80": float(TTA_R80),
                })

                # Confusion matrix and class-wise recall (same as train_dad.py)
                cf = confusion_matrix(all_y_np, all_pred_np)
                print(f"\n=== SERVER EVALUATION ROUND {server_round} ===", flush=True)
                print("Confusion Matrix:")
                print(cf, flush=True)
                class_recall = cf.diagonal() / cf.sum(axis=1)
                print("Class-wise recall:", np.round(class_recall, 3), flush=True)
                print("=" * 50, flush=True)
                if class_recall.size >= 2:
                    metrics.update({
                        "recall_neg": float(class_recall[0]),
                        "recall_pos": float(class_recall[1]),
                    })

                # Save PR curve (same as train_dad.py)
                y_scores = all_probs_vid2_np.flatten()
                y_true_flat = all_y_np.astype(np.int64)
                precision, recall, _ = precision_recall_curve(y_true_flat, y_scores)
                pr_auc = auc(recall, precision)
                metrics["pr_auc_frame"] = float(pr_auc)
                
                out_path = os.path.join(results_dir, "federated", f"pr_curve_round_{server_round}.png")
                plt.figure()
                plt.plot(recall, precision, label=f'PR AUC={pr_auc:.2f}')
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.title(f'Precision-Recall Curve (Round {server_round})')
                plt.legend()
                plt.savefig(out_path, bbox_inches='tight')
                plt.close()
                print(f"Saved PR curve: {out_path}")

        except Exception as e:
            print(f"[Server Eval] Metrics computation failed: {e}")

        # Return dummy loss (server evaluation focuses on metrics like train_dad.py)
        return 0.0, metrics

    return evaluate


def main():
    parser = argparse.ArgumentParser()
    # FL config
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--min-fit-clients", type=int, default=5)
    parser.add_argument("--min-available-clients", type=int, default=5)
    parser.add_argument("--fraction-fit", type=float, default=1.0)
    parser.add_argument("--server-address", type=str, default="0.0.0.0:8080")
    # Data/model config (same as train_dad.py)
    parser.add_argument("--dataset-path", type=str, default="data/dad/obj_feat")
    parser.add_argument("--img-dataset-path", type=str, default="data/dad/i3d_feat")
    parser.add_argument("--split-path", type=str, default="splits_dad")
    parser.add_argument("--obj-mapping-file", type=str, default="data/dad/obj_idx_to_labels.json")
    parser.add_argument("--test-batch-size", type=int, default=1)
    parser.add_argument("--input-dim", type=int, default=4096)
    parser.add_argument("--img-feat-dim", type=int, default=2048)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--ref-interval", type=int, default=20)
    parser.add_argument("--local-epochs", type=int, default=1, help="Number of local epochs per round")
    args = parser.parse_args()

    # Use CUDA for better performance
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Same model initialization as train_dad.py
    model = SpaceTempGoG_detr_dad(
        input_dim=args.input_dim,
        embedding_dim=args.embedding_dim,
        img_feat_dim=args.img_feat_dim,
        num_classes=args.num_classes,
    ).to(device)

    # Global test dataset (test_split.txt - same as train_dad.py)
    test_dataset = DADDataset(
        img_dataset_path=args.img_dataset_path,
        dataset_path=args.dataset_path,
        split_path=args.split_path,
        ref_interval=args.ref_interval,
        objmap_file=args.obj_mapping_file,
        training=False,
    )
    # Set num_workers=0 to avoid multiprocessing issues in federated environment
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=0)

    results_dir = "results"
    evaluate_fn = get_evaluate_fn(model, test_loader, device, results_dir)

    def on_fit_config_fn(server_round: int):
        return {
            "local_epochs": args.local_epochs,
        }

    # FedAvg strategy (same as flower1.ipynb)
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=args.fraction_fit,
        min_fit_clients=args.min_fit_clients,
        min_available_clients=args.min_available_clients,
        evaluate_fn=evaluate_fn,
        on_fit_config_fn=on_fit_config_fn,
    )

    print(f"Starting federated server with {args.rounds} rounds, {args.min_fit_clients} clients")
    fl.server.start_server(
        server_address=args.server_address,
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
    )


if __name__ == "__main__":
    main()