# -*- coding: utf-8 -*-
"""
Offline RL Models for Hybrid PQC-TLS
====================================
Pure PyTorch implementations of 5 offline RL algorithms:

1. BC   - Behavioral Cloning (supervised baseline)
2. CQL  - Conservative Q-Learning (Kumar et al., 2020)
3. IQL  - Implicit Q-Learning (Kostrikov et al., 2021)
4. BCQ  - Batch-Constrained Q-Learning (Fujimoto et al., 2019)
5. AWAC - Advantage Weighted Actor-Critic (Nair et al., 2020)

All models use discrete action spaces (12 actions).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Optional, List
from abc import ABC, abstractmethod
from copy import deepcopy

from .rl_config import (
    STATE_DIM,
    NUM_ACTIONS,
    TrainingConfig,
    DEFAULT_TRAINING_CONFIG,
)


# ============================================================================
# Neural Network Building Blocks
# ============================================================================

class MLP(nn.Module):
    """Multi-layer perceptron with configurable architecture."""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int] = [256, 256],
        activation: nn.Module = nn.ReLU,
        output_activation: Optional[nn.Module] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(activation())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        if output_activation is not None:
            layers.append(output_activation())
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class QNetwork(nn.Module):
    """Q-network that outputs Q-values for all actions."""
    
    def __init__(
        self,
        state_dim: int = STATE_DIM,
        num_actions: int = NUM_ACTIONS,
        hidden_dims: List[int] = [256, 256],
        dropout: float = 0.0,
    ):
        super().__init__()
        self.net = MLP(
            input_dim=state_dim,
            output_dim=num_actions,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Returns Q-values for all actions: (batch, num_actions)"""
        return self.net(state)
    
    def get_q_value(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Get Q-value for specific actions: (batch,)"""
        q_values = self.forward(state)
        return q_values.gather(1, action.unsqueeze(1)).squeeze(1)


class PolicyNetwork(nn.Module):
    """Policy network that outputs action probabilities."""
    
    def __init__(
        self,
        state_dim: int = STATE_DIM,
        num_actions: int = NUM_ACTIONS,
        hidden_dims: List[int] = [256, 256],
        dropout: float = 0.0,
    ):
        super().__init__()
        self.net = MLP(
            input_dim=state_dim,
            output_dim=num_actions,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Returns action logits: (batch, num_actions)"""
        return self.net(state)
    
    def get_probs(self, state: torch.Tensor) -> torch.Tensor:
        """Returns action probabilities: (batch, num_actions)"""
        logits = self.forward(state)
        return F.softmax(logits, dim=-1)
    
    def get_log_probs(self, state: torch.Tensor) -> torch.Tensor:
        """Returns log action probabilities: (batch, num_actions)"""
        logits = self.forward(state)
        return F.log_softmax(logits, dim=-1)
    
    def sample(self, state: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Sample action from policy."""
        logits = self.forward(state)
        if deterministic:
            return logits.argmax(dim=-1)
        else:
            probs = F.softmax(logits, dim=-1)
            return torch.multinomial(probs, num_samples=1).squeeze(-1)


class ValueNetwork(nn.Module):
    """Value network V(s)."""
    
    def __init__(
        self,
        state_dim: int = STATE_DIM,
        hidden_dims: List[int] = [256, 256],
        dropout: float = 0.0,
    ):
        super().__init__()
        self.net = MLP(
            input_dim=state_dim,
            output_dim=1,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Returns state value: (batch,)"""
        return self.net(state).squeeze(-1)


# ============================================================================
# Base Class for All Algorithms
# ============================================================================

class OfflineRLAlgorithm(ABC):
    """Base class for offline RL algorithms."""
    
    def __init__(
        self,
        config: TrainingConfig = DEFAULT_TRAINING_CONFIG,
        device: str = "cpu",
    ):
        self.config = config
        self.device = torch.device(device)
        self.name = "BaseAlgorithm"
    
    @abstractmethod
    def train_step(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> Dict[str, float]:
        """Perform one training step."""
        pass
    
    @abstractmethod
    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = True,
    ) -> int:
        """Select action for a single state."""
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """Save model to disk."""
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """Load model from disk."""
        pass


# ============================================================================
# 1. Behavioral Cloning (BC) - Supervised Baseline
# ============================================================================

class BehavioralCloning(OfflineRLAlgorithm):
    """
    Behavioral Cloning: Learn to imitate the behavior policy via supervised learning.
    
    Loss: Cross-entropy between predicted action distribution and actual actions
    """
    
    def __init__(
        self,
        config: TrainingConfig = DEFAULT_TRAINING_CONFIG,
        device: str = "cpu",
    ):
        super().__init__(config, device)
        self.name = "BC"
        
        self.policy = PolicyNetwork(
            state_dim=STATE_DIM,
            num_actions=NUM_ACTIONS,
            hidden_dims=config.hidden_dims,
            dropout=config.dropout,
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    
    def train_step(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> Dict[str, float]:
        self.policy.train()
        
        logits = self.policy(states)
        loss = F.cross_entropy(logits, actions)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Compute accuracy
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            accuracy = (preds == actions).float().mean().item()
        
        return {"loss": loss.item(), "accuracy": accuracy}
    
    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = True,
    ) -> int:
        self.policy.eval()
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action = self.policy.sample(state_t, deterministic=deterministic)
            return action.item()
    
    def save(self, path: str) -> None:
        torch.save({
            "policy": self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, path)
    
    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])


# ============================================================================
# 2. Conservative Q-Learning (CQL)
# ============================================================================

class CQL(OfflineRLAlgorithm):
    """
    Conservative Q-Learning (CQL): Penalize Q-values for out-of-distribution actions.
    
    Loss = TD_loss + alpha * CQL_loss
    
    CQL_loss = E[logsumexp(Q(s,a))] - E[Q(s,a)] for a in dataset
    """
    
    def __init__(
        self,
        config: TrainingConfig = DEFAULT_TRAINING_CONFIG,
        device: str = "cpu",
    ):
        super().__init__(config, device)
        self.name = "CQL"
        
        # Q-networks (with target)
        self.q_net = QNetwork(
            state_dim=STATE_DIM,
            num_actions=NUM_ACTIONS,
            hidden_dims=config.hidden_dims,
            dropout=config.dropout,
        ).to(self.device)
        
        self.q_target = deepcopy(self.q_net)
        for p in self.q_target.parameters():
            p.requires_grad = False
        
        self.optimizer = torch.optim.Adam(
            self.q_net.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        self.cql_alpha = config.cql_alpha
        self.cql_temperature = config.cql_temperature
        self.gamma = config.gamma
        self.tau = config.tau
        self.update_counter = 0
        self.target_update_freq = config.target_update_freq
    
    def _soft_update_target(self):
        """Soft update target network."""
        for target_param, param in zip(self.q_target.parameters(), self.q_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def train_step(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> Dict[str, float]:
        self.q_net.train()
        
        # Current Q-values
        q_values = self.q_net(states)
        q_a = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Target Q-values (Double DQN style)
        with torch.no_grad():
            next_q_values = self.q_target(next_states)
            next_actions = self.q_net(next_states).argmax(dim=-1)
            next_q_a = next_q_values.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards + (1 - dones) * self.gamma * next_q_a
        
        # TD loss
        td_loss = F.mse_loss(q_a, target_q)
        
        # CQL loss: penalize high Q-values for all actions
        logsumexp_q = torch.logsumexp(q_values / self.cql_temperature, dim=-1).mean()
        dataset_q = q_a.mean()
        cql_loss = logsumexp_q - dataset_q
        
        # Total loss
        loss = td_loss + self.cql_alpha * cql_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self._soft_update_target()
        
        return {
            "loss": loss.item(),
            "td_loss": td_loss.item(),
            "cql_loss": cql_loss.item(),
            "q_mean": q_a.mean().item(),
        }
    
    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = True,
    ) -> int:
        self.q_net.eval()
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_net(state_t)
            return q_values.argmax(dim=-1).item()
    
    def save(self, path: str) -> None:
        torch.save({
            "q_net": self.q_net.state_dict(),
            "q_target": self.q_target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, path)
    
    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(checkpoint["q_net"])
        self.q_target.load_state_dict(checkpoint["q_target"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])


# ============================================================================
# 3. Implicit Q-Learning (IQL)
# ============================================================================

class IQL(OfflineRLAlgorithm):
    """
    Implicit Q-Learning (IQL): Avoid explicit policy constraint by using expectile regression.
    
    Key insight: Use expectile regression on V to approximate max over actions
    without needing to evaluate Q for out-of-distribution actions.
    """
    
    def __init__(
        self,
        config: TrainingConfig = DEFAULT_TRAINING_CONFIG,
        device: str = "cpu",
    ):
        super().__init__(config, device)
        self.name = "IQL"
        
        # Networks
        self.q_net = QNetwork(
            state_dim=STATE_DIM,
            num_actions=NUM_ACTIONS,
            hidden_dims=config.hidden_dims,
        ).to(self.device)
        
        self.v_net = ValueNetwork(
            state_dim=STATE_DIM,
            hidden_dims=config.hidden_dims,
        ).to(self.device)
        
        self.policy = PolicyNetwork(
            state_dim=STATE_DIM,
            num_actions=NUM_ACTIONS,
            hidden_dims=config.hidden_dims,
        ).to(self.device)
        
        self.q_target = deepcopy(self.q_net)
        for p in self.q_target.parameters():
            p.requires_grad = False
        
        self.q_optimizer = torch.optim.Adam(
            self.q_net.parameters(), lr=config.learning_rate
        )
        self.v_optimizer = torch.optim.Adam(
            self.v_net.parameters(), lr=config.learning_rate
        )
        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=config.learning_rate
        )
        
        self.tau_iql = config.iql_tau  # Expectile
        self.beta = config.iql_beta    # Inverse temperature
        self.gamma = config.gamma
        self.tau = config.tau
        self.update_counter = 0
        self.target_update_freq = config.target_update_freq
    
    def _expectile_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Asymmetric expectile loss."""
        diff = target - pred
        weight = torch.where(diff > 0, self.tau_iql, 1 - self.tau_iql)
        return (weight * (diff ** 2)).mean()
    
    def _soft_update_target(self):
        for target_param, param in zip(self.q_target.parameters(), self.q_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def train_step(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> Dict[str, float]:
        # ===== Value function update (expectile regression) =====
        with torch.no_grad():
            q_target_values = self.q_target(states)
            q_a = q_target_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        v = self.v_net(states)
        v_loss = self._expectile_loss(v, q_a)
        
        self.v_optimizer.zero_grad()
        v_loss.backward()
        self.v_optimizer.step()
        
        # ===== Q-function update =====
        with torch.no_grad():
            next_v = self.v_net(next_states)
            target_q = rewards + (1 - dones) * self.gamma * next_v
        
        q_values = self.q_net(states)
        q_a_current = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        q_loss = F.mse_loss(q_a_current, target_q)
        
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()
        
        # ===== Policy update (advantage-weighted) =====
        with torch.no_grad():
            q_for_policy = self.q_target(states)
            q_a_policy = q_for_policy.gather(1, actions.unsqueeze(1)).squeeze(1)
            v_for_policy = self.v_net(states)
            advantage = q_a_policy - v_for_policy
            weights = torch.exp(self.beta * advantage)
            weights = torch.clamp(weights, max=100.0)  # Clamp for stability
        
        log_probs = self.policy.get_log_probs(states)
        log_prob_a = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        policy_loss = -(weights * log_prob_a).mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        # Update target
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self._soft_update_target()
        
        return {
            "loss": (v_loss + q_loss + policy_loss).item(),
            "v_loss": v_loss.item(),
            "q_loss": q_loss.item(),
            "policy_loss": policy_loss.item(),
            "advantage_mean": advantage.mean().item(),
        }
    
    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = True,
    ) -> int:
        self.policy.eval()
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action = self.policy.sample(state_t, deterministic=deterministic)
            return action.item()
    
    def save(self, path: str) -> None:
        torch.save({
            "q_net": self.q_net.state_dict(),
            "q_target": self.q_target.state_dict(),
            "v_net": self.v_net.state_dict(),
            "policy": self.policy.state_dict(),
        }, path)
    
    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(checkpoint["q_net"])
        self.q_target.load_state_dict(checkpoint["q_target"])
        self.v_net.load_state_dict(checkpoint["v_net"])
        self.policy.load_state_dict(checkpoint["policy"])


# ============================================================================
# 4. Batch-Constrained Q-Learning (BCQ)
# ============================================================================

class BCQ(OfflineRLAlgorithm):
    """
    Batch-Constrained Q-Learning (BCQ): Restrict actions to those seen in the dataset.
    
    Uses a generative model (here: policy network) to filter actions,
    then selects the best Q-value among filtered actions.
    """
    
    def __init__(
        self,
        config: TrainingConfig = DEFAULT_TRAINING_CONFIG,
        device: str = "cpu",
    ):
        super().__init__(config, device)
        self.name = "BCQ"
        
        # Q-networks
        self.q_net = QNetwork(
            state_dim=STATE_DIM,
            num_actions=NUM_ACTIONS,
            hidden_dims=config.hidden_dims,
        ).to(self.device)
        
        self.q_target = deepcopy(self.q_net)
        for p in self.q_target.parameters():
            p.requires_grad = False
        
        # Behavior cloning policy (for action filtering)
        self.bc_policy = PolicyNetwork(
            state_dim=STATE_DIM,
            num_actions=NUM_ACTIONS,
            hidden_dims=config.hidden_dims,
        ).to(self.device)
        
        self.q_optimizer = torch.optim.Adam(
            self.q_net.parameters(), lr=config.learning_rate
        )
        self.bc_optimizer = torch.optim.Adam(
            self.bc_policy.parameters(), lr=config.learning_rate
        )
        
        self.threshold = config.bcq_threshold
        self.gamma = config.gamma
        self.tau = config.tau
        self.update_counter = 0
        self.target_update_freq = config.target_update_freq
    
    def _soft_update_target(self):
        for target_param, param in zip(self.q_target.parameters(), self.q_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def train_step(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> Dict[str, float]:
        # ===== BC policy update =====
        bc_logits = self.bc_policy(states)
        bc_loss = F.cross_entropy(bc_logits, actions)
        
        self.bc_optimizer.zero_grad()
        bc_loss.backward()
        self.bc_optimizer.step()
        
        # ===== Q-network update =====
        with torch.no_grad():
            # Get action probabilities from BC policy
            next_probs = self.bc_policy.get_probs(next_states)
            
            # Mask out low-probability actions
            max_probs = next_probs.max(dim=-1, keepdim=True)[0]
            mask = (next_probs >= self.threshold * max_probs).float()
            
            # Q-values for next states
            next_q = self.q_target(next_states)
            
            # Select best Q among allowed actions
            masked_q = next_q - 1e8 * (1 - mask)
            best_next_q = masked_q.max(dim=-1)[0]
            
            target_q = rewards + (1 - dones) * self.gamma * best_next_q
        
        q_values = self.q_net(states)
        q_a = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        q_loss = F.mse_loss(q_a, target_q)
        
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()
        
        # Update target
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self._soft_update_target()
        
        return {
            "loss": (bc_loss + q_loss).item(),
            "bc_loss": bc_loss.item(),
            "q_loss": q_loss.item(),
        }
    
    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = True,
    ) -> int:
        self.q_net.eval()
        self.bc_policy.eval()
        
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            # Get BC policy probabilities
            probs = self.bc_policy.get_probs(state_t)
            max_prob = probs.max()
            mask = (probs >= self.threshold * max_prob).float()
            
            # Get Q-values and mask
            q_values = self.q_net(state_t)
            masked_q = q_values - 1e8 * (1 - mask)
            
            return masked_q.argmax(dim=-1).item()
    
    def save(self, path: str) -> None:
        torch.save({
            "q_net": self.q_net.state_dict(),
            "q_target": self.q_target.state_dict(),
            "bc_policy": self.bc_policy.state_dict(),
        }, path)
    
    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(checkpoint["q_net"])
        self.q_target.load_state_dict(checkpoint["q_target"])
        self.bc_policy.load_state_dict(checkpoint["bc_policy"])


# ============================================================================
# 5. Advantage Weighted Actor-Critic (AWAC)
# ============================================================================

class AWAC(OfflineRLAlgorithm):
    """
    Advantage Weighted Actor-Critic (AWAC): Weight policy updates by advantage.
    
    Policy update: maximize E[exp(A(s,a)/lambda) * log pi(a|s)]
    where A(s,a) = Q(s,a) - V(s)
    """
    
    def __init__(
        self,
        config: TrainingConfig = DEFAULT_TRAINING_CONFIG,
        device: str = "cpu",
    ):
        super().__init__(config, device)
        self.name = "AWAC"
        
        # Actor (policy)
        self.policy = PolicyNetwork(
            state_dim=STATE_DIM,
            num_actions=NUM_ACTIONS,
            hidden_dims=config.hidden_dims,
        ).to(self.device)
        
        # Critic (Q-function)
        self.q_net = QNetwork(
            state_dim=STATE_DIM,
            num_actions=NUM_ACTIONS,
            hidden_dims=config.hidden_dims,
        ).to(self.device)
        
        self.q_target = deepcopy(self.q_net)
        for p in self.q_target.parameters():
            p.requires_grad = False
        
        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=config.learning_rate
        )
        self.q_optimizer = torch.optim.Adam(
            self.q_net.parameters(), lr=config.learning_rate
        )
        
        self.awac_lambda = config.awac_lambda
        self.gamma = config.gamma
        self.tau = config.tau
        self.update_counter = 0
        self.target_update_freq = config.target_update_freq
    
    def _soft_update_target(self):
        for target_param, param in zip(self.q_target.parameters(), self.q_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def train_step(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> Dict[str, float]:
        # ===== Critic update =====
        with torch.no_grad():
            next_probs = self.policy.get_probs(next_states)
            next_q = self.q_target(next_states)
            next_v = (next_probs * next_q).sum(dim=-1)
            target_q = rewards + (1 - dones) * self.gamma * next_v
        
        q_values = self.q_net(states)
        q_a = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        q_loss = F.mse_loss(q_a, target_q)
        
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()
        
        # ===== Actor update (advantage-weighted) =====
        with torch.no_grad():
            # Compute advantage
            q_for_adv = self.q_target(states)
            q_a_adv = q_for_adv.gather(1, actions.unsqueeze(1)).squeeze(1)
            
            # V = E_a[Q(s,a)] under current policy
            probs = self.policy.get_probs(states)
            v = (probs * q_for_adv).sum(dim=-1)
            
            advantage = q_a_adv - v
            weights = torch.exp(advantage / self.awac_lambda)
            weights = torch.clamp(weights, max=100.0)
        
        log_probs = self.policy.get_log_probs(states)
        log_prob_a = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        policy_loss = -(weights * log_prob_a).mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        # Update target
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self._soft_update_target()
        
        return {
            "loss": (q_loss + policy_loss).item(),
            "q_loss": q_loss.item(),
            "policy_loss": policy_loss.item(),
            "advantage_mean": advantage.mean().item(),
        }
    
    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = True,
    ) -> int:
        self.policy.eval()
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action = self.policy.sample(state_t, deterministic=deterministic)
            return action.item()
    
    def save(self, path: str) -> None:
        torch.save({
            "policy": self.policy.state_dict(),
            "q_net": self.q_net.state_dict(),
            "q_target": self.q_target.state_dict(),
        }, path)
    
    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy"])
        self.q_net.load_state_dict(checkpoint["q_net"])
        self.q_target.load_state_dict(checkpoint["q_target"])


# ============================================================================
# Model Factory
# ============================================================================

ALGORITHM_REGISTRY = {
    "BC": BehavioralCloning,
    "CQL": CQL,
    "IQL": IQL,
    "BCQ": BCQ,
    "AWAC": AWAC,
}


def create_algorithm(
    name: str,
    config: TrainingConfig = DEFAULT_TRAINING_CONFIG,
    device: str = "cpu",
) -> OfflineRLAlgorithm:
    """
    Factory function to create an algorithm by name.
    
    Args:
        name: Algorithm name (BC, CQL, IQL, BCQ, AWAC)
        config: Training configuration
        device: Device to use
        
    Returns:
        Instantiated algorithm
    """
    if name not in ALGORITHM_REGISTRY:
        raise ValueError(f"Unknown algorithm: {name}. Available: {list(ALGORITHM_REGISTRY.keys())}")
    
    return ALGORITHM_REGISTRY[name](config=config, device=device)


def get_available_algorithms() -> List[str]:
    """Return list of available algorithm names."""
    return list(ALGORITHM_REGISTRY.keys())


if __name__ == "__main__":
    print("Available Offline RL Algorithms:")
    for name in get_available_algorithms():
        algo = create_algorithm(name)
        print(f"  - {name}: {algo.__class__.__name__}")
