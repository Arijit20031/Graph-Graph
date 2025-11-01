import argparse
import os
from pathlib import Path
import random

# Partitions splits_dad/train_split.txt IID into num_clients shards 
# Only train_split.txt is distributed; test_split.txt remains global for server evaluation
# Saved under: splits_dad/clients/train_split_client_{cid}.txt

def load_list(path):
    with open(path) as f:
        return [l.strip() for l in f if l.strip()]


def save_list(path, items):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for it in items:
            f.write(f"{it}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-dir", default="splits_dad", type=str)
    parser.add_argument("--train-file", default="train_split.txt", type=str)
    parser.add_argument("--num-clients", default=5, type=int)
    parser.add_argument("--seed", default=0, type=int)
    args = parser.parse_args()

    random.seed(args.seed)

    # Only partition train_split.txt IID across clients
    train_path = Path(args.split_dir) / args.train_file
    train_list = load_list(str(train_path))
    
    # Shuffle and split IID
    random.shuffle(train_list)
    shards = [[] for _ in range(args.num_clients)]
    for i, item in enumerate(train_list):
        shards[i % args.num_clients].append(item)

    # Save client train splits
    out_dir = Path(args.split_dir) / "clients"
    out_dir.mkdir(parents=True, exist_ok=True)
    for cid in range(args.num_clients):
        save_list(str(out_dir / f"train_split_client_{cid}.txt"), shards[cid])
        print(f"Client {cid}: train={len(shards[cid])}")
    print(f"Wrote client train splits to {out_dir} for {args.num_clients} clients")


if __name__ == "__main__":
    main()
