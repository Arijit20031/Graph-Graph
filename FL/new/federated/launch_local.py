#!/usr/bin/env python3
"""
Launch script for DAD federated learning with fixed indexing
"""

import subprocess
import sys
import time
import argparse
import signal
import os


def run_server(args):
    """Start the federated server"""
    server_cmd = ["python", "federated/server.py",
        "--rounds", str(args.rounds),
        "--min-fit-clients", str(args.num_clients),
        "--min-available-clients", str(args.num_clients),
        "--server-address", args.server_address,
        "--local-epochs", str(args.local_epochs),
        "--input-dim", "4096",
        "--embedding-dim", "256", 
        "--img-feat-dim", "2048",
        "--num-classes", "2",
    ]
    
    print(f"Starting server: {' '.join(server_cmd)}")
    return subprocess.Popen(server_cmd)


def run_client(cid, args):
    """Start a federated client"""
    client_cmd = ["python", "federated/client.py",  # Use original client name
        "--cid", str(cid),
        "--num-clients", str(args.num_clients),
        "--server", args.server_address,
    ]
    
    print(f"Starting client {cid}: {' '.join(client_cmd)}")
    return subprocess.Popen(client_cmd)


def main():
    parser = argparse.ArgumentParser(description="Launch DAD federated learning")
    parser.add_argument("--num-clients", type=int, default=5, help="Number of clients")
    parser.add_argument("--rounds", type=int, default=5, help="Number of federated rounds")
    parser.add_argument("--local-epochs", type=int, default=1, help="Local epochs per round")
    parser.add_argument("--server-address", type=str, default="127.0.0.1:8080", help="Server address")
    
    args = parser.parse_args()
    
    print(f"Starting federated DAD training: {args.num_clients} clients, {args.rounds} rounds")
    
    processes = []
    
    try:
        # Start server
        server_process = run_server(args)
        processes.append(server_process)
        time.sleep(2)  # Give server time to start
        
        # Start clients
        client_processes = []
        for cid in range(args.num_clients):
            client_process = run_client(cid, args)
            client_processes.append(client_process)
            processes.append(client_process)
            time.sleep(2)  # Stagger client starts
        
        # Wait for server to complete
        server_process.wait()
        print("Federated training completed!")
        
    except KeyboardInterrupt:
        print("\\nInterrupting federated training...")
    finally:
        # Clean up all processes
        for process in processes:
            try:
                process.terminate()
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
            except Exception:
                pass


if __name__ == "__main__":
    main()