#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Offline RL Pipeline v2 - With Diverse Dataset
==============================================
Key changes:
1. Uses behavior_epsilon=0.6 (vs 0.3) for more diverse data
2. Adds minimum 2% coverage per action
3. Uses reward-proportional exploration

Expected outcome:
- BC, IQL, BCQ, AWAC should now differentiate
- CQL should still show conservative behavior (lower violations)
- Better demonstration of conservative policy learning benefits
"""

import os
import sys
import shutil
import time
from datetime import datetime

# Add paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the new dataset builder
from hybrid_pqc_tls.rl_dataset_improved_v2 import build_diverse_dataset   

# Import from the project
sys.path.insert(0, '/mnt/project')
from hybrid_pqc_tls.rl_config import (
    DEFAULT_PATH_CONFIG,
    DEFAULT_TRAINING_CONFIG,
    TrainingConfig,
    PathConfig,
)
from hybrid_pqc_tls.rl_train import OfflineRLTrainer
from hybrid_pqc_tls.rl_evaluate_v2 import (
    run_comprehensive_evaluation,
    ComprehensiveEvaluator,
    PublicationVisualizer,
    generate_latex_tables,
)


def run_pipeline_v2():
    """Run the complete pipeline with diverse dataset."""
    
    start_time = time.time()
    print("="*70)
    print("OFFLINE RL PIPELINE v2 - DIVERSE DATASET")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # Configuration
    path_config = DEFAULT_PATH_CONFIG
    training_config = TrainingConfig(
        num_epochs=100,
        batch_size=256,
        learning_rate=3e-4,
        eval_freq=10,
    )
    
    # ===== Step 1: Setup Data =====
    print("\n[Step 1/4] Setting up data...")
    
    # Check for uploaded data
    upload_path = "/mnt/user-data/uploads/handshake_raw.csv"
    target_path = path_config.raw_data_csv
    
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    
    if os.path.exists(upload_path):
        shutil.copy(upload_path, target_path)
        print(f"✓ Copied data from {upload_path}")
    elif not os.path.exists(target_path):
        print(f"✗ Data not found. Please provide handshake_raw.csv")
        return
    else:
        print(f"✓ Data already exists at {target_path}")
    
    # ===== Step 2: Build DIVERSE Dataset =====
    print("\n[Step 2/4] Building DIVERSE offline RL dataset...")
    print("  Key parameters:")
    print("    behavior_epsilon: 0.6 (was 0.3)")
    print("    min_action_coverage: 2%")
    print("    exploration_temperature: 2.0")
    
    build_diverse_dataset(
        csv_path=target_path,
        output_path=path_config.dataset_path,
        train_split=0.8,
        seed=42,
        target_size=10000,
        behavior_epsilon=0.6,        # INCREASED
        min_action_coverage=0.02,    # NEW: ensure all actions present
        exploration_temperature=2.0,  # NEW: reward-proportional exploration
    )
    
    # ===== Step 3: Train Models =====
    print("\n[Step 3/4] Training RL algorithms...")
    
    trainer = OfflineRLTrainer(config=training_config, path_config=path_config)
    trainer.load_data(path_config.dataset_path)
    
    history = trainer.train_all(
        algorithms=["BC", "CQL", "IQL", "BCQ", "AWAC"],
        save_models=True,
        verbose=True,
    )
    
    # ===== Step 4: Evaluate =====
    print("\n[Step 4/4] Evaluating RL vs Rule-Based...")
    
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
    
    # Print detailed comparison
    if results:
        print("\n" + "="*70)
        print("DETAILED COMPARISON")
        print("="*70)
        
        # Header
        print(f"\n{'Method':<12} {'Reward':>10} {'Latency':>10} {'Hybrid%':>8} {'PQC%':>8} {'Class%':>8} {'Viol%':>8}")
        print("-"*70)
        
        # Sort by reward
        sorted_methods = sorted(results.items(), key=lambda x: x[1].mean_reward, reverse=True)
        
        for name, m in sorted_methods:
            print(f"{name:<12} {m.mean_reward:>10.3f} {m.mean_latency:>10.1f} "
                  f"{m.hybrid_rate*100:>8.1f} {m.pqc_rate*100:>8.1f} "
                  f"{m.classical_rate*100:>8.1f} {m.violation_rate*100:>8.1f}")
        
        # Key findings
        print("\n" + "="*70)
        print("KEY FINDINGS FOR PAPER")
        print("="*70)
        
        if "Rule_Based" in results and "CQL" in results:
            rb = results["Rule_Based"]
            cql = results["CQL"]
            
            print(f"\n1. SAFETY COMPARISON (violation rate):")
            print(f"   Rule-Based: {rb.violation_rate*100:.1f}% violations")
            print(f"   CQL:        {cql.violation_rate*100:.1f}% violations")
            
            # Compare to other RL methods
            rl_methods = ["BC", "IQL", "BCQ", "AWAC"]
            rl_violations = [results[m].violation_rate*100 for m in rl_methods if m in results]
            if rl_violations:
                print(f"   Other RL:   {min(rl_violations):.1f}% - {max(rl_violations):.1f}% violations")
                
                if cql.violation_rate < min(results[m].violation_rate for m in rl_methods if m in results):
                    print(f"   → CQL is the MOST CONSERVATIVE RL method")
            
            print(f"\n2. EFFICIENCY COMPARISON (latency):")
            latency_improvement = (rb.mean_latency - cql.mean_latency) / rb.mean_latency * 100
            print(f"   Rule-Based: {rb.mean_latency:.1f} ms")
            print(f"   CQL:        {cql.mean_latency:.1f} ms ({latency_improvement:+.1f}%)")
            
            print(f"\n3. TRADEOFF ANALYSIS:")
            reward_diff = cql.mean_reward - rb.mean_reward
            print(f"   Reward difference: {reward_diff:+.3f}")
            if reward_diff < 0 and cql.violation_rate > rb.violation_rate:
                violation_cost = (cql.violation_rate - rb.violation_rate) * 100
                latency_gain = latency_improvement
                print(f"   Latency gain: {latency_gain:.1f}%")
                print(f"   Violation cost: {violation_cost:.1f}%")
                print(f"   → CQL trades {violation_cost:.1f}% more violations for {latency_gain:.1f}% lower latency")
    
    return results


if __name__ == "__main__":
    run_pipeline_v2()
