#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete Offline RL Pipeline for Hybrid PQC-TLS
===============================================
This script runs the complete pipeline:
1. Build offline RL dataset from handshake_raw.csv
2. Train all 5 RL algorithms (BC, CQL, IQL, BCQ, AWAC)
3. Evaluate against Rule-Based baseline (RL vs Rule-Based only)
4. Generate publication-quality figures
5. Generate LaTeX-ready tables for paper

Usage:
    python run_rl_pipeline.py [--data-path PATH] [--epochs N] [--skip-training]
"""

import os
import sys
import argparse
import shutil
import time
from datetime import datetime

# Add the hybrid_pqc_tls package to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hybrid_pqc_tls.rl_config import (
    DEFAULT_PATH_CONFIG,
    DEFAULT_TRAINING_CONFIG,
    TrainingConfig,
    PathConfig,
)
from hybrid_pqc_tls.rl_dataset_improved import build_improved_dataset
from hybrid_pqc_tls.rl_train import run_training
# Use v2 evaluation for RL vs Rule-Based comparison
from hybrid_pqc_tls.rl_evaluate_v2 import (
    run_comprehensive_evaluation,
    ComprehensiveEvaluator,
    PublicationVisualizer,
    generate_latex_tables,
)


def setup_data(upload_path: str, target_path: str) -> bool:
    """Copy uploaded data to expected location."""
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    
    if os.path.exists(upload_path):
        shutil.copy(upload_path, target_path)
        print(f"✓ Copied data from {upload_path} to {target_path}")
        return True
    elif os.path.exists(target_path):
        print(f"✓ Data already exists at {target_path}")
        return True
    else:
        print(f"✗ Data not found at {upload_path} or {target_path}")
        return False


def run_pipeline(
    data_path: str = None,
    num_epochs: int = 100,
    skip_training: bool = False,
    skip_dataset: bool = False,
) -> None:
    """
    Run the complete offline RL pipeline.
    
    Args:
        data_path: Path to handshake_raw.csv
        num_epochs: Number of training epochs
        skip_training: Skip training (use existing models)
        skip_dataset: Skip dataset building
    """
    start_time = time.time()
    print("="*70)
    print("OFFLINE RL PIPELINE FOR HYBRID PQC-TLS")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # Configuration
    path_config = DEFAULT_PATH_CONFIG
    training_config = TrainingConfig(
        num_epochs=num_epochs,
        batch_size=256,
        learning_rate=3e-4,
        eval_freq=10,
    )
    
    # ===== Step 1: Setup Data =====
    print("\n[Step 1/4] Setting up data...")
    
    upload_path = "/mnt/user-data/uploads/handshake_raw.csv"
    target_path = path_config.raw_data_csv
    
    if data_path:
        upload_path = data_path
    
    if not setup_data(upload_path, target_path):
        print("ERROR: Cannot proceed without data file.")
        return
    
    # ===== Step 2: Build Dataset =====
    print("\n[Step 2/4] Building improved offline RL dataset...")
    print("  (Creating behavioral dataset from evaluation grid)")
    
    if skip_dataset and os.path.exists(path_config.dataset_path):
        print(f"  Skipping (dataset exists at {path_config.dataset_path})")
    else:
        build_improved_dataset(
            csv_path=target_path,
            output_path=path_config.dataset_path,
            train_split=0.8,
            seed=42,
            target_size=10000,       # 10K behavioral samples
            behavior_epsilon=0.3,    # 30% exploration in behavior policy
        )
    
    # ===== Step 3: Train Models =====
    print("\n[Step 3/4] Training RL algorithms...")
    
    if skip_training:
        print("  Skipping training (--skip-training flag set)")
    else:
        run_training(
            data_path=path_config.dataset_path,
            algorithms=["BC", "CQL", "IQL", "BCQ", "AWAC"],
            config=training_config,
            verbose=True,
        )
    
    # ===== Step 4: Evaluate (RL vs Rule-Based Only) =====
    print("\n[Step 4/4] Evaluating RL vs Rule-Based...")
    print("  (Using comprehensive v2 evaluation)")
    
    results = run_comprehensive_evaluation(
        algorithms=["BC", "CQL", "IQL", "BCQ", "AWAC"],
        generate_figures=True,
    )
    
    # ===== Summary =====
    elapsed = time.time() - start_time
    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print("="*70)
    print(f"Total time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print(f"\nOutput locations:")
    print(f"  Dataset:     {path_config.dataset_path}")
    print(f"  Models:      {path_config.model_dir}/")
    print(f"  Evaluation:  {path_config.eval_dir}/")
    print(f"  Figures:     {path_config.figures_dir}/")
    print(f"  LaTeX:       {path_config.eval_dir}/latex/")
    
    # Print best model summary
    print("\n" + "="*70)
    print("BEST PERFORMING MODELS")
    print("="*70)
    
    if results:
        # Filter to RL methods only
        rl_results = {k: v for k, v in results.items() if k in ["BC", "CQL", "IQL", "BCQ", "AWAC"]}
        
        if rl_results and "Rule_Based" in results:
            baseline = results["Rule_Based"]
            
            print(f"\nRule-Based Baseline:")
            print(f"  Reward:  {baseline.mean_reward:.3f}")
            print(f"  Latency: {baseline.mean_latency:.1f} ms")
            print(f"  Wire:    {baseline.mean_wire:.0f} B")
            
            # Find best by reward improvement
            best_reward = max(rl_results.items(), key=lambda x: x[1].mean_reward)
            reward_imp = ((best_reward[1].mean_reward - baseline.mean_reward) / abs(baseline.mean_reward)) * 100
            
            print(f"\nBest RL Algorithm: {best_reward[0]}")
            print(f"  Reward:      {best_reward[1].mean_reward:.3f} ({reward_imp:+.1f}% vs Rule-Based)")
            print(f"  Latency:     {best_reward[1].mean_latency:.1f} ms")
            print(f"  Wire:        {best_reward[1].mean_wire:.0f} B")
            print(f"  Hybrid %:    {best_reward[1].hybrid_rate*100:.1f}%")
            print(f"  PQC %:       {best_reward[1].pqc_rate*100:.1f}%")
            print(f"  Violations:  {best_reward[1].violation_rate*100:.1f}%")
        
        if "Oracle" in results:
            oracle = results["Oracle"]
            print(f"\nOracle (Upper Bound):")
            print(f"  Reward:  {oracle.mean_reward:.3f}")
            print(f"  Latency: {oracle.mean_latency:.1f} ms")


def main():
    parser = argparse.ArgumentParser(
        description="Run the complete offline RL pipeline for Hybrid PQC-TLS"
    )
    parser.add_argument(
        "--data-path", "-d",
        type=str,
        default=None,
        help="Path to handshake_raw.csv (default: /mnt/user-data/uploads/handshake_raw.csv)"
    )
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)"
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training and use existing models"
    )
    parser.add_argument(
        "--skip-dataset",
        action="store_true",
        help="Skip dataset building if it already exists"
    )
    
    args = parser.parse_args()
    
    run_pipeline(
        data_path=args.data_path,
        num_epochs=args.epochs,
        skip_training=args.skip_training,
        skip_dataset=args.skip_dataset,
    )


if __name__ == "__main__":
    main()
