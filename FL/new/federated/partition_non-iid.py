#!/usr/bin/env python3
"""
Non-IID Data Partitioning for DAD Dataset based on Class Ratio
Creates a specified class imbalance (positive/negative ratio) across clients.
"""

import os
import numpy as np
import argparse
from collections import defaultdict
import math

# --- Shared Utility Functions (Copied from the original code) ---

def analyze_split_labels(split_file, data_path):
    """Analyze the class distribution in a split file"""
    with open(split_file) as f:
        samples = f.read().splitlines()
    
    positive_samples = []
    negative_samples = []
    
    print(f"Analyzing {len(samples)} samples from {split_file}...")
    
    for i, sample in enumerate(samples):
        if i % 100 == 0:
            print(f"Processed {i}/{len(samples)} samples")
            
        # Assuming "train" split file is in a path that contains "train"
        # and data is structured with a "training" or "testing" folder
        split_name = "training" if "train" in split_file else "testing"
        sample_path = os.path.join(data_path, split_name, sample)
        
        try:
            data = np.load(sample_path)
            label = int(data['labels'][1])  # Same logic as dataset_dad.py
            
            if label > 0:
                positive_samples.append(sample)
            else:
                negative_samples.append(sample)
                
        except Exception as e:
            # print(f"Warning: Could not load {sample}: {e}")
            continue
    
    print(f"Analysis complete:")
    print(f"  Positive (accident) samples: {len(positive_samples)}")
    print(f"  Negative (non-accident) samples: {len(negative_samples)}")
    print(f"  Total: {len(positive_samples) + len(negative_samples)}")
    print(f"  Class ratio: {len(positive_samples)/(len(positive_samples) + len(negative_samples)):.3f} positive")
    
    return positive_samples, negative_samples


def save_client_splits(client_data, output_dir, split_type="train"):
    """Save client partitions to files"""
    os.makedirs(output_dir, exist_ok=True)
    
    stats = {
        'total_samples': 0,
        'total_positive': 0,
        'total_negative': 0,
        'clients': []
    }
    
    for cid, data in enumerate(client_data):
        client_file = os.path.join(output_dir, f"{split_type}_non_iid_client_{cid}.txt")
        
        with open(client_file, 'w') as f:
            for sample in data['samples']:
                f.write(f"{sample}\n")
        
        stats['total_samples'] += data['total']
        stats['total_positive'] += data['pos_count'] 
        stats['total_negative'] += data['neg_count']
        stats['clients'].append({
            'client_id': cid,
            'total': data['total'],
            'positive': data['pos_count'],
            'negative': data['neg_count'],
            'pos_ratio': data['pos_count'] / data['total'] if data['total'] > 0 else 0
        })
        
        print(f"Saved Client {cid}: {client_file}")
    
    # Save statistics
    stats_file = os.path.join(output_dir, f"{split_type}_non_iid_stats.txt")
    with open(stats_file, 'w') as f:
        f.write(f"NON-IID {split_type.upper()} Split Statistics (Class Imbalance)\n")
        f.write(f"{'='*50}\n")
        f.write(f"Total samples: {stats['total_samples']}\n")
        f.write(f"Total positive: {stats['total_positive']}\n")
        f.write(f"Total negative: {stats['total_negative']}\n")
        
        if stats['total_samples'] > 0:
            f.write(f"Overall pos ratio: {stats['total_positive']/stats['total_samples']:.3f}\n\n")
        else:
            f.write("Overall pos ratio: 0.000\n\n")

        for client in stats['clients']:
            f.write(f"Client {client['client_id']}: {client['total']} samples "
                    f"({client['positive']} pos, {client['negative']} neg, "
                    f"ratio={client['pos_ratio']:.3f})\n")
    
    print(f"Statistics saved to: {stats_file}")
    return stats

# --- Non-IID Partitioning Function (The modified part) ---

def create_non_iid_clients_by_ratio(positive_samples, negative_samples, num_clients, target_ratios, seed=42):
    """
    Create Non-IID partitions by assigning a target positive class ratio 
    to each client. The total number of samples per client is kept equal 
    where possible, prioritizing the target ratios.
    """
    np.random.seed(seed)
    
    if len(target_ratios) != num_clients:
        raise ValueError("The number of target_ratios must equal num_clients.")
    
    # Shuffle both classes for an initial random assignment across clients
    pos_shuffled = positive_samples.copy()
    neg_shuffled = negative_samples.copy()
    np.random.shuffle(pos_shuffled)
    np.random.shuffle(neg_shuffled)
    
    total_samples = len(pos_shuffled) + len(neg_shuffled)
    
    # Calculate the total number of positive and negative samples required across all clients
    # based on the target ratios and the assumption of equal total size per client
    
    # 1. Total samples to be assigned to each client (approximately equal)
    samples_per_client = total_samples // num_clients
    
    client_data = []
    
    # The pools of samples available for assignment
    available_pos = pos_shuffled
    available_neg = neg_shuffled
    
    
    print("\n--- Non-IID Client Distribution ---")
    
    for cid in range(num_clients):
        ratio = target_ratios[cid]
        
        # Calculate the required number of positive and negative samples for this client
        # based on the target ratio and the planned total sample size for the client.
        
        # Required Positive Samples: pos_count = ratio * samples_per_client
        required_pos = math.floor(ratio * samples_per_client)
        # Required Negative Samples: neg_count = (1 - ratio) * samples_per_client
        required_neg = samples_per_client - required_pos
        
        # --- Assignment from available pools ---
        
        # Assign Positive Samples
        if len(available_pos) >= required_pos:
            client_pos = available_pos[:required_pos]
            available_pos = available_pos[required_pos:]
        else:
            # If not enough positive samples left, take all remaining
            client_pos = available_pos
            available_pos = []

        # Assign Negative Samples
        if len(available_neg) >= required_neg:
            client_neg = available_neg[:required_neg]
            available_neg = available_neg[required_neg:]
        else:
            # If not enough negative samples left, take all remaining
            client_neg = available_neg
            available_neg = []
            
        # The actual total number of samples for the client (may be less than samples_per_client 
        # if the pools ran out)
        client_total = len(client_pos) + len(client_neg)
        
        # Combine and shuffle client's data (IID within the client)
        client_samples = client_pos + client_neg
        np.random.shuffle(client_samples)
        
        # Calculate the actual achieved ratio for reporting
        actual_ratio = len(client_pos) / client_total if client_total > 0 else 0
        
        client_data.append({
            'samples': client_samples,
            'pos_count': len(client_pos),
            'neg_count': len(client_neg),
            'total': client_total
        })
        
        print(f"Client {cid}: Target Ratio={ratio:.2f}, Assigned Total={client_total}, "
              f"Actual Ratio={actual_ratio:.3f} ({len(client_pos)} pos, {len(client_neg)} neg)")
        
    # Handle any remaining samples (assign them to the last client, or distribute evenly)
    # For simplicity, we'll assign the remainder to the last client.
    if available_pos or available_neg:
        print("\nNote: Handling remaining unassigned samples...")
        last_client_id = num_clients - 1
        last_client_data = client_data[last_client_id]
        
        # Add remaining positive
        last_client_data['samples'].extend(available_pos)
        last_client_data['pos_count'] += len(available_pos)
        
        # Add remaining negative
        last_client_data['samples'].extend(available_neg)
        last_client_data['neg_count'] += len(available_neg)

        # Re-shuffle the final client's list
        np.random.shuffle(last_client_data['samples'])
        last_client_data['total'] = len(last_client_data['samples'])
        
        # Update reported stats for the last client
        last_client_total = last_client_data['total']
        last_client_ratio = last_client_data['pos_count'] / last_client_total if last_client_total > 0 else 0
        
        print(f"Added {len(available_pos)} pos and {len(available_neg)} neg to Client {last_client_id}.")
        print(f"Client {last_client_id} final total: {last_client_total}, final ratio: {last_client_ratio:.3f}")

    return client_data

# --- Main Execution Block ---

def main():
    parser = argparse.ArgumentParser(description="Create NON-IID partitions for DAD dataset based on class ratio")
    parser.add_argument("--num-clients", type=int, default=5, help="Number of clients")
    parser.add_argument("--data-path", type=str, default="data/dad/obj_feat", 
                        help="Path to DAD object features")
    parser.add_argument("--split-path", type=str, default="splits_dad", 
                        help="Path to original split files")
    parser.add_argument("--output-dir", type=str, default="splits_dad/non_iid_clients", 
                        help="Output directory for NON-IID client splits")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--analyze-only", action="store_true", 
                        help="Only analyze splits without creating partitions")
    
    args = parser.parse_args()
    
    # --- Define NON-IID Ratios ---
    # The requirement: ratio of positive class/negative class will be 0.1, 0.2, 0.3, 0.4, 0.5
    # The ratio is positive class / total samples.
    # Note: If the overall dataset positive ratio is low (e.g., 0.1), it might be mathematically
    # impossible to achieve a high target ratio (e.g., 0.5) for a client while maintaining equal total sizes.
    # The code below will try its best based on available samples.
    
    if args.num_clients != 5:
        print("Warning: Overriding num_clients to 5 to match the required target ratios.")
        args.num_clients = 5
        
    # The target ratios for Client 0, 1, 2, 3, 4
    target_ratios = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    print(f"Creating NON-IID partitions for {args.num_clients} clients with target ratios: {target_ratios}")
    print(f"Data path: {args.data_path}")
    print(f"Split path: {args.split_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Random seed: {args.seed}")
    print("-" * 60)
    
    # Analyze train split
    train_split_file = os.path.join(args.split_path, "train_split.txt")
    if not os.path.exists(train_split_file):
        print(f"Error: Train split file not found: {train_split_file}")
        return
    
    print("\n📊 ANALYZING TRAINING DATA")
    print("=" * 50)
    train_pos, train_neg = analyze_split_labels(train_split_file, args.data_path)
    
    if args.analyze_only:
        print("\nAnalysis complete. Use --analyze-only=False to create partitions.")
        return
    
    # Create NON-IID partitions for training data
    print("\n🔀 CREATING NON-IID PARTITIONS (Class Imbalance)")
    print("=" * 50)
    train_client_data = create_non_iid_clients_by_ratio(train_pos, train_neg, args.num_clients, target_ratios, args.seed)
    
    # Save partitions
    print("\n💾 SAVING CLIENT PARTITIONS")
    print("=" * 50)
    train_stats = save_client_splits(train_client_data, args.output_dir, "train")
    
    print("\n✅ NON-IID partitioning complete!")
    print(f"Created {args.num_clients} client partitions in {args.output_dir}")
    
    # Verify class balance
    print("\n📈 CLASS BALANCE VERIFICATION (Target vs. Actual)")
    print("=" * 50)
    for cid, client in enumerate(train_stats['clients']):
        target_ratio = target_ratios[cid]
        print(f"Client {client['client_id']}: Target={target_ratio:.2f}, Actual={client['pos_ratio']:.3f} positive ratio")


if __name__ == "__main__":
    main()
