# DAD Federated Learning Setup

This federated learning setup exactly mirrors the `train_dad.py` approach but distributes training across 5 clients.

## Key Features

- **Identical to train_dad.py**: Same model, optimizer, loss, metrics, and validation logic
- **5 clients**: Each gets IID split of `train_split.txt` 
- **70/30 split per client**: Each client splits their data 70% train, 30% validation (same seed as original)
- **AP-based checkpointing**: Best weights selected by Average Precision on local validation
- **Server evaluation**: Uses `eval_utils.evaluation` on global `test_split.txt` each round
- **Same initialization**: Model weights start identical to `train_dad.py`

## Usage

### 1. Partition data for 5 clients:
```bash
python federated/partition.py --num-clients 5
```

### 2. Start federated training:

**Option A: One launcher (recommended)**
```bash
python federated/launch_local.py --num-clients 5 --rounds 5 --local-epochs 1
```

**Option B: Manual (separate terminals)**

Terminal 1 (server):
```bash
python federated/server.py --rounds 5 --min-fit-clients 5 --min-available-clients 5 --local-epochs 1
```

Terminals 2-6 (clients):
```bash
python federated/client.py --cid 0 --num-clients 5 --server 127.0.0.1:8080
python federated/client.py --cid 1 --num-clients 5 --server 127.0.0.1:8080
python federated/client.py --cid 2 --num-clients 5 --server 127.0.0.1:8080
python federated/client.py --cid 3 --num-clients 5 --server 127.0.0.1:8080
python federated/client.py --cid 4 --num-clients 5 --server 127.0.0.1:8080
```

## How It Works

### Client Flow (per round):
1. Receive global weights from server
2. Load client's data shard from `splits_dad/clients/train_split_client_{cid}.txt`
3. Split 70/30 into local train/validation (same seed as `train_dad.py`)
4. Train for `local_epochs` using identical loop as `train_dad.py`:
   - Same batch size (1), optimizer (Adam), loss (CE on pre-accident frames)
   - Same temporal edge weight computation via cosine similarity
5. After each local epoch: evaluate on local validation, compute AP
6. Keep best checkpoint by AP (same logic as `train_dad.py`)
7. Send best weights to server

### Server Flow (per round):
1. Aggregate client weights using FedAvg
2. Evaluate on global `test_split.txt` using `eval_utils.evaluation`
3. Compute and log: AP, mTTA, TTA@R80, confusion matrix, class-wise recall
4. Save PR curve plot to `results/federated/pr_curve_round_{round}.png`

## Expected Output

```
Starting federated DAD training: 5 clients, 5 rounds
Client 0: total=X, train=Y, val=Z
...
Average Precision= 0.XXXX, mean Time to accident= X.XXXX
Recall@80%, Time to accident= X.XXXX
Confusion Matrix:
[[a b]
 [c d]]
Class-wise recall: [X.XXX Y.XXX]
Saved PR curve: results/federated/pr_curve_round_1.png
...
```

## Files

- `server.py`: FedAvg aggregation + global test evaluation using `eval_utils`
- `client.py`: 70/30 split + AP-based checkpointing + same training as `train_dad.py`
- `partition.py`: IID split of `train_split.txt` into 5 client files
- `launch_local.py`: Convenience launcher for local testing