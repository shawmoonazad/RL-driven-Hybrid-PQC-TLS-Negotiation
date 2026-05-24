# -*- coding: utf-8 -*-
"""
Offline RL Training Pipeline for Hybrid PQC-TLS
===============================================
Trains all 5 offline RL algorithms (BC, CQL, IQL, BCQ, AWAC) and saves:
- Model checkpoints
- Training curves
- Intermediate evaluation metrics

Usage:
    python -m hybrid_pqc_tls.rl_train
"""

import os
import json
import time
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from torch.utils.data import DataLoader, TensorDataset

from .rl_config import (
    TrainingConfig,
    DEFAULT_TRAINING_CONFIG,
    PathConfig,
    DEFAULT_PATH_CONFIG,
    ACTION_LIST,
    NUM_ACTIONS,
    STATE_DIM,
)
from .rl_offline_dataset import OfflineRLDatasetBuilder, build_and_save_dataset
from .rl_models import (
    create_algorithm,
    get_available_algorithms,
    OfflineRLAlgorithm,
)


class OfflineRLTrainer:
    """
    Trainer for offline RL algorithms.
    
    Handles:
    - Data loading and batching
    - Training loop with logging
    - Periodic evaluation
    - Model checkpointing
    """
    
    def __init__(
        self,
        config: TrainingConfig = DEFAULT_TRAINING_CONFIG,
        path_config: PathConfig = DEFAULT_PATH_CONFIG,
    ):
        self.config = config
        self.path_config = path_config
        self.device = torch.device(config.device)
        
        # Set random seeds
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        
        # Data
        self.data: Optional[Dict[str, np.ndarray]] = None
        self.train_loader: Optional[DataLoader] = None
        self.test_data: Optional[Dict[str, torch.Tensor]] = None
        
        # Training history
        self.history: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    
    def load_data(self, data_path: Optional[str] = None) -> None:
        """Load and prepare the offline dataset."""
        if data_path is None:
            data_path = self.path_config.dataset_path
        
        # Check if dataset exists, if not build it
        if not os.path.exists(data_path):
            print(f"[Trainer] Dataset not found at {data_path}, building...")
            build_and_save_dataset()
        
        # Load dataset
        builder = OfflineRLDatasetBuilder()
        self.data = builder.load(data_path)
        
        # Prepare train/test splits
        train_idx = self.data["train_idx"]
        test_idx = self.data["test_idx"]
        
        # Training data as tensors
        S_train = torch.FloatTensor(self.data["S"][train_idx])
        A_train = torch.LongTensor(self.data["A"][train_idx])
        R_train = torch.FloatTensor(self.data["R"][train_idx])
        S_next_train = torch.FloatTensor(self.data["S_next"][train_idx])
        D_train = torch.FloatTensor(self.data["done"][train_idx])
        
        train_dataset = TensorDataset(S_train, A_train, R_train, S_next_train, D_train)
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=True,
        )
        
        # Test data
        self.test_data = {
            "S": torch.FloatTensor(self.data["S"][test_idx]).to(self.device),
            "A": torch.LongTensor(self.data["A"][test_idx]).to(self.device),
            "R": torch.FloatTensor(self.data["R"][test_idx]).to(self.device),
            "S_next": torch.FloatTensor(self.data["S_next"][test_idx]).to(self.device),
            "done": torch.FloatTensor(self.data["done"][test_idx]).to(self.device),
            "rtt": self.data["rtt"][test_idx],
            "latency": self.data["latency"][test_idx],
            "wire": self.data["wire"][test_idx],
        }
        
        print(f"[Trainer] Train batches: {len(self.train_loader)}, Test samples: {len(test_idx)}")
    
    def evaluate(
        self,
        algorithm: OfflineRLAlgorithm,
    ) -> Dict[str, float]:
        """
        Evaluate algorithm on test set.
        
        Returns metrics including:
        - Accuracy (matching dataset actions)
        - Average reward of selected actions
        - Policy distribution stats
        """
        S = self.test_data["S"]
        A_true = self.test_data["A"]
        R = self.test_data["R"]
        rtt = self.test_data["rtt"]
        latency = self.test_data["latency"]
        wire = self.test_data["wire"]
        
        # Get predicted actions
        predicted_actions = []
        for i in range(len(S)):
            state = S[i].cpu().numpy()
            action = algorithm.select_action(state, deterministic=True)
            predicted_actions.append(action)
        
        predicted_actions = np.array(predicted_actions)
        true_actions = A_true.cpu().numpy()
        
        # Accuracy
        accuracy = (predicted_actions == true_actions).mean()
        
        # Action distribution
        action_counts = np.bincount(predicted_actions, minlength=NUM_ACTIONS)
        action_dist = action_counts / len(predicted_actions)
        
        # Policy type usage
        hybrid_rate = sum(action_dist[i] for i, (p, l) in enumerate(ACTION_LIST) if p == "REQUIRE_HYBRID")
        pqc_rate = sum(action_dist[i] for i, (p, l) in enumerate(ACTION_LIST) if p == "PQC_ONLY")
        classical_rate = sum(action_dist[i] for i, (p, l) in enumerate(ACTION_LIST) if p == "CLASSICAL_ONLY")
        fallback_rate = sum(action_dist[i] for i, (p, l) in enumerate(ACTION_LIST) if p == "ALLOW_FALLBACK")
        
        # Security violations (level < 3)
        violation_rate = sum(action_dist[i] for i, (p, l) in enumerate(ACTION_LIST) if l < 3)
        
        # Level distribution
        level_dist = {1: 0, 3: 0, 5: 0}
        for i, (p, l) in enumerate(ACTION_LIST):
            level_dist[l] += action_dist[i]
        
        # Compute expected reward under predicted policy
        # (This is approximate - we use the reward from matching state-action pairs in dataset)
        # For true evaluation, we'd need to run handshakes
        
        return {
            "accuracy": accuracy,
            "hybrid_rate": hybrid_rate,
            "pqc_rate": pqc_rate,
            "classical_rate": classical_rate,
            "fallback_rate": fallback_rate,
            "violation_rate": violation_rate,
            "level_1_rate": level_dist[1],
            "level_3_rate": level_dist[3],
            "level_5_rate": level_dist[5],
        }
    
    def train_algorithm(
        self,
        algorithm: OfflineRLAlgorithm,
        num_epochs: Optional[int] = None,
        eval_freq: Optional[int] = None,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Train a single algorithm.
        
        Args:
            algorithm: The algorithm to train
            num_epochs: Number of training epochs
            eval_freq: Evaluation frequency (epochs)
            verbose: Whether to print progress
            
        Returns:
            Training history dictionary
        """
        if self.train_loader is None:
            raise RuntimeError("Data not loaded. Call load_data() first.")
        
        num_epochs = num_epochs or self.config.num_epochs
        eval_freq = eval_freq or self.config.eval_freq
        
        history = defaultdict(list)
        best_accuracy = 0.0
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_losses = defaultdict(list)
            
            # Training loop
            for batch in self.train_loader:
                states, actions, rewards, next_states, dones = [
                    x.to(self.device) for x in batch
                ]
                
                metrics = algorithm.train_step(states, actions, rewards, next_states, dones)
                
                for key, value in metrics.items():
                    epoch_losses[key].append(value)
            
            # Average epoch losses
            for key, values in epoch_losses.items():
                history[key].append(np.mean(values))
            
            # Periodic evaluation
            if (epoch + 1) % eval_freq == 0:
                eval_metrics = self.evaluate(algorithm)
                for key, value in eval_metrics.items():
                    history[f"eval_{key}"].append(value)
                
                if eval_metrics["accuracy"] > best_accuracy:
                    best_accuracy = eval_metrics["accuracy"]
                
                if verbose:
                    elapsed = time.time() - start_time
                    print(f"  Epoch {epoch+1:3d}/{num_epochs} | "
                          f"Loss: {history['loss'][-1]:.4f} | "
                          f"Acc: {eval_metrics['accuracy']:.3f} | "
                          f"Hybrid: {eval_metrics['hybrid_rate']:.2f} | "
                          f"PQC: {eval_metrics['pqc_rate']:.2f} | "
                          f"Viol: {eval_metrics['violation_rate']:.3f} | "
                          f"Time: {elapsed:.1f}s")
        
        return dict(history)
    
    def train_all(
        self,
        algorithms: Optional[List[str]] = None,
        save_models: bool = True,
        verbose: bool = True,
    ) -> Dict[str, Dict[str, List[float]]]:
        """
        Train all specified algorithms.
        
        Args:
            algorithms: List of algorithm names (default: all)
            save_models: Whether to save model checkpoints
            verbose: Whether to print progress
            
        Returns:
            Dictionary mapping algorithm name to training history
        """
        if algorithms is None:
            algorithms = get_available_algorithms()
        
        all_history = {}
        
        for algo_name in algorithms:
            print(f"\n{'='*60}")
            print(f"Training: {algo_name}")
            print(f"{'='*60}")
            
            # Create algorithm
            algorithm = create_algorithm(
                algo_name,
                config=self.config,
                device=self.config.device,
            )
            
            # Train
            history = self.train_algorithm(algorithm, verbose=verbose)
            all_history[algo_name] = history
            self.history[algo_name] = history
            
            # Save model
            if save_models:
                model_dir = self.path_config.model_dir
                os.makedirs(model_dir, exist_ok=True)
                model_path = os.path.join(model_dir, f"{algo_name.lower()}_model.pt")
                algorithm.save(model_path)
                print(f"  Model saved: {model_path}")
            
            # Final evaluation
            final_metrics = self.evaluate(algorithm)
            print(f"\n  Final Results for {algo_name}:")
            print(f"    Accuracy: {final_metrics['accuracy']:.4f}")
            print(f"    Hybrid Rate: {final_metrics['hybrid_rate']:.4f}")
            print(f"    PQC Rate: {final_metrics['pqc_rate']:.4f}")
            print(f"    Classical Rate: {final_metrics['classical_rate']:.4f}")
            print(f"    Violation Rate: {final_metrics['violation_rate']:.4f}")
        
        # Save training history
        history_path = os.path.join(self.path_config.log_dir, "training_history.json")
        os.makedirs(os.path.dirname(history_path), exist_ok=True)
        
        # Convert to serializable format
        serializable_history = {}
        for algo_name, hist in all_history.items():
            serializable_history[algo_name] = {
                k: [float(v) for v in vals] for k, vals in hist.items()
            }
        
        with open(history_path, "w") as f:
            json.dump(serializable_history, f, indent=2)
        print(f"\nTraining history saved: {history_path}")
        
        return all_history
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for all trained algorithms."""
        summary = {}
        
        for algo_name, history in self.history.items():
            # Get final evaluation metrics
            final_metrics = {}
            for key in history:
                if key.startswith("eval_") and len(history[key]) > 0:
                    final_metrics[key.replace("eval_", "")] = history[key][-1]
            
            # Get final loss
            if "loss" in history and len(history["loss"]) > 0:
                final_metrics["final_loss"] = history["loss"][-1]
            
            summary[algo_name] = final_metrics
        
        return summary


def run_training(
    data_path: Optional[str] = None,
    algorithms: Optional[List[str]] = None,
    config: Optional[TrainingConfig] = None,
    verbose: bool = True,
) -> Dict[str, Dict[str, List[float]]]:
    """
    Convenience function to run full training pipeline.
    
    Args:
        data_path: Path to dataset (optional)
        algorithms: List of algorithms to train (optional, default: all)
        config: Training configuration (optional)
        verbose: Whether to print progress
        
    Returns:
        Training history for all algorithms
    """
    if config is None:
        config = DEFAULT_TRAINING_CONFIG
    
    trainer = OfflineRLTrainer(config=config)
    trainer.load_data(data_path)
    
    history = trainer.train_all(algorithms=algorithms, verbose=verbose)
    
    # Print summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    
    summary = trainer.get_summary()
    for algo_name, metrics in summary.items():
        print(f"\n{algo_name}:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
    
    return history


if __name__ == "__main__":
    # First, copy the uploaded dataset to the expected location
    import shutil
    
    upload_path = "/mnt/user-data/uploads/handshake_raw.csv"
    target_dir = "results/eval_grid"
    target_path = f"{target_dir}/handshake_raw.csv"
    
    os.makedirs(target_dir, exist_ok=True)
    if os.path.exists(upload_path):
        shutil.copy(upload_path, target_path)
        print(f"Copied dataset from {upload_path} to {target_path}")
    
    # Run training
    run_training(verbose=True)
