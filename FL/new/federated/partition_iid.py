#!/usr/bin/env python3
"""
Improved IID Data Partitioning for DAD Dataset
Maintains class balance (positive/negative) across clients
"""

import os
import numpy as np
import argparse
from collections import defaultdict


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
            
        sample_path = os.path.join(data_path, "training" if "train" in split_file else "testing", sample)
        
        try:
            data = np.load(sample_path)
            label = int(data['labels'][1])  # Same logic as dataset_dad.py
            
            if label > 0:
                positive_samples.append(sample)
            else:
                negative_samples.append(sample)
                
        except Exception as e:
            print(f"Warning: Could not load {sample}: {e}")
            continue
    
    print(f"Analysis complete:")
    print(f"  Positive (accident) samples: {len(positive_samples)}")
    print(f"  Negative (non-accident) samples: {len(negative_samples)}")
    print(f"  Total: {len(positive_samples) + len(negative_samples)}")
    print(f"  Class ratio: {len(positive_samples)/(len(positive_samples) + len(negative_samples)):.3f} positive")
    
    return positive_samples, negative_samples


def create_iid_clients(positive_samples, negative_samples, num_clients, seed=42):
    """Create IID partitions maintaining class balance"""
    np.random.seed(seed)
    
    # Shuffle both classes
    pos_shuffled = positive_samples.copy()
    neg_shuffled = negative_samples.copy()
    np.random.shuffle(pos_shuffled)
    np.random.shuffle(neg_shuffled)
    
    # Calculate samples per client
    total_pos = len(pos_shuffled)
    total_neg = len(neg_shuffled)
    
    pos_per_client = total_pos // num_clients
    neg_per_client = total_neg // num_clients
    
    # Distribute samples
    client_data = []
    pos_start = 0
    neg_start = 0
    
    for cid in range(num_clients):
        # Handle remainder for last client
        if cid == num_clients - 1:
            client_pos = pos_shuffled[pos_start:]
            client_neg = neg_shuffled[neg_start:]
        else:
            client_pos = pos_shuffled[pos_start:pos_start + pos_per_client]
            client_neg = neg_shuffled[neg_start:neg_start + neg_per_client]
        
        # Combine and shuffle client's data
        client_samples = client_pos + client_neg
        np.random.shuffle(client_samples)
        
        client_data.append({
            'samples': client_samples,
            'pos_count': len(client_pos),
            'neg_count': len(client_neg),
            'total': len(client_samples)
        })
        
        pos_start += pos_per_client
        neg_start += neg_per_client
        
        print(f"Client {cid}: {len(client_samples)} total ({len(client_pos)} pos, {len(client_neg)} neg)")
    
    return client_data


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
        client_file = os.path.join(output_dir, f"{split_type}_split_client_{cid}.txt")
        
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
            'pos_ratio': data['pos_count'] / data['total']
        })
        
        print(f"Saved Client {cid}: {client_file}")
    
    # Save statistics
    stats_file = os.path.join(output_dir, f"{split_type}_split_stats.txt")
    with open(stats_file, 'w') as f:
        f.write(f"IID {split_type.upper()} Split Statistics\n")
        f.write(f"{'='*50}\n")
        f.write(f"Total samples: {stats['total_samples']}\n")
        f.write(f"Total positive: {stats['total_positive']}\n")
        f.write(f"Total negative: {stats['total_negative']}\n")
        f.write(f"Overall pos ratio: {stats['total_positive']/stats['total_samples']:.3f}\n\n")
        
        for client in stats['clients']:
            f.write(f"Client {client['client_id']}: {client['total']} samples "
                   f"({client['positive']} pos, {client['negative']} neg, "
                   f"ratio={client['pos_ratio']:.3f})\n")
    
    print(f"Statistics saved to: {stats_file}")
    return stats


def main():
    parser = argparse.ArgumentParser(description="Create IID partitions for DAD dataset")
    parser.add_argument("--num-clients", type=int, default=5, help="Number of clients")
    parser.add_argument("--data-path", type=str, default="data/dad/obj_feat", 
                       help="Path to DAD object features")
    parser.add_argument("--split-path", type=str, default="splits_dad", 
                       help="Path to split files")
    parser.add_argument("--output-dir", type=str, default="splits_dad/clients", 
                       help="Output directory for client splits")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--analyze-only", action="store_true", 
                       help="Only analyze splits without creating partitions")
    
    args = parser.parse_args()
    
    print(f"Creating IID partitions for {args.num_clients} clients")
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
    
    print("\\n📊 ANALYZING TRAINING DATA")
    print("=" * 50)
    train_pos, train_neg = analyze_split_labels(train_split_file, args.data_path)
    
    if args.analyze_only:
        print("\\nAnalysis complete. Use --analyze-only=False to create partitions.")
        return
    
    # Create IID partitions for training data
    print("\\n🔀 CREATING IID PARTITIONS")
    print("=" * 50)
    train_client_data = create_iid_clients(train_pos, train_neg, args.num_clients, args.seed)
    
    # Save partitions
    print("\\n💾 SAVING CLIENT PARTITIONS")
    print("=" * 50)
    train_stats = save_client_splits(train_client_data, args.output_dir, "train")
    
    print("\\n✅ IID partitioning complete!")
    print(f"Created {args.num_clients} client partitions in {args.output_dir}")
    
    # Verify class balance
    print("\\n📈 CLASS BALANCE VERIFICATION")
    print("=" * 50)
    for client in train_stats['clients']:
        print(f"Client {client['client_id']}: {client['pos_ratio']:.3f} positive ratio")


if __name__ == "__main__":
    main()