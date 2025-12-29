"""
Adversarial Training for Robustness
Implements FGSM, PGD, and TRADES for defending against attacks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable


class FGSM:
    """
    Fast Gradient Sign Method (FGSM)
    Single-step adversarial attack for training robustness
    """
    
    def __init__(self, epsilon: float = 0.01):
        self.epsilon = epsilon
    
    def generate_adversarial(
        self,
        model: nn.Module,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        loss_fn: Callable,
    ) -> torch.Tensor:
        """
        Generate adversarial embeddings using FGSM
        
        Args:
            model: Model to attack
            embeddings: (batch_size, seq_len, d_model)
            labels: (batch_size,) ground truth labels
            loss_fn: Loss function
        
        Returns:
            adv_embeddings: (batch_size, seq_len, d_model)
        """
        # Ensure embeddings require gradients
        embeddings = embeddings.detach()
        embeddings.requires_grad = True
        
        # Forward pass
        outputs = model.forward_from_embeddings(embeddings)
        loss = loss_fn(outputs, labels)
        
        # Backward pass
        model.zero_grad()
        loss.backward()
        
        # Get gradient
        grad = embeddings.grad.detach()
        
        # FGSM perturbation
        perturbation = self.epsilon * grad.sign()
        adv_embeddings = embeddings.detach() + perturbation
        
        return adv_embeddings


class PGD:
    """
    Projected Gradient Descent (PGD)
    Multi-step adversarial attack for stronger robustness
    """
    
    def __init__(
        self,
        epsilon: float = 0.01,
        alpha: float = 0.001,
        num_steps: int = 5,
        random_start: bool = True,
    ):
        self.epsilon = epsilon
        self.alpha = alpha  # Step size
        self.num_steps = num_steps
        self.random_start = random_start
    
    def generate_adversarial(
        self,
        model: nn.Module,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        loss_fn: Callable,
    ) -> torch.Tensor:
        """
        Generate adversarial embeddings using PGD
        
        Args:
            model: Model to attack
            embeddings: (batch_size, seq_len, d_model)
            labels: (batch_size,) ground truth labels
            loss_fn: Loss function
        
        Returns:
            adv_embeddings: (batch_size, seq_len, d_model)
        """
        adv_embeddings = embeddings.detach()
        
        # Random initialization
        if self.random_start:
            noise = torch.empty_like(adv_embeddings).uniform_(-self.epsilon, self.epsilon)
            adv_embeddings = adv_embeddings + noise
        
        # PGD iterations
        for _ in range(self.num_steps):
            adv_embeddings = adv_embeddings.detach()
            adv_embeddings.requires_grad = True
            
            # Forward pass
            outputs = model.forward_from_embeddings(adv_embeddings)
            loss = loss_fn(outputs, labels)
            
            # Backward pass
            model.zero_grad()
            loss.backward()
            
            # Get gradient
            grad = adv_embeddings.grad.detach()
            
            # PGD step
            adv_embeddings = adv_embeddings.detach() + self.alpha * grad.sign()
            
            # Project back to epsilon ball
            perturbation = torch.clamp(
                adv_embeddings - embeddings,
                min=-self.epsilon,
                max=self.epsilon
            )
            adv_embeddings = embeddings + perturbation
        
        return adv_embeddings.detach()


class TRADES:
    """
    TRADES (TRadeoff between Accuracy and robustness via adversarial training with DEfense Strength)
    Balances clean accuracy and adversarial robustness
    """
    
    def __init__(
        self,
        beta: float = 6.0,  # Trade-off parameter
        epsilon: float = 0.01,
        num_steps: int = 10,
        step_size: float = 0.001,
    ):
        self.beta = beta
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
    
    def compute_loss(
        self,
        model: nn.Module,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        loss_fn: Callable,
    ) -> torch.Tensor:
        """
        Compute TRADES loss
        
        Args:
            model: Model
            embeddings: (batch_size, seq_len, d_model)
            labels: (batch_size,)
            loss_fn: Loss function
        
        Returns:
            total_loss: TRADES loss (clean + robust term)
        """
        # Clean loss
        outputs = model.forward_from_embeddings(embeddings)
        clean_loss = loss_fn(outputs, labels)
        
        # Generate adversarial examples
        adv_embeddings = self._generate_adversarial(
            model, embeddings, labels, loss_fn
        )
        
        # Adversarial loss (KL divergence)
        with torch.no_grad():
            clean_logits = model.forward_from_embeddings(embeddings)
        
        adv_logits = model.forward_from_embeddings(adv_embeddings)
        
        robust_loss = F.kl_div(
            F.log_softmax(adv_logits, dim=-1),
            F.softmax(clean_logits, dim=-1),
            reduction='batchmean',
        )
        
        # Total loss
        total_loss = clean_loss + self.beta * robust_loss
        
        return total_loss
    
    def _generate_adversarial(
        self,
        model: nn.Module,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        loss_fn: Callable,
    ) -> torch.Tensor:
        """Generate adversarial examples using PGD"""
        adv_embeddings = embeddings.detach()
        
        for _ in range(self.num_steps):
            adv_embeddings = adv_embeddings.detach()
            adv_embeddings.requires_grad = True
            
            # Forward
            outputs = model.forward_from_embeddings(adv_embeddings)
            
            # KL divergence with clean output
            with torch.no_grad():
                clean_logits = model.forward_from_embeddings(embeddings)
            
            loss = F.kl_div(
                F.log_softmax(outputs, dim=-1),
                F.softmax(clean_logits, dim=-1),
                reduction='batchmean',
            )
            
            # Backward
            model.zero_grad()
            loss.backward()
            
            # Gradient ascent
            grad = adv_embeddings.grad.detach()
            adv_embeddings = adv_embeddings.detach() + self.step_size * grad.sign()
            
            # Project
            perturbation = torch.clamp(
                adv_embeddings - embeddings,
                min=-self.epsilon,
                max=self.epsilon
            )
            adv_embeddings = embeddings + perturbation
        
        return adv_embeddings.detach()


class AdversarialTrainer:
    """
    Adversarial Training Wrapper
    Integrates adversarial training into normal training loop
    """
    
    def __init__(
        self,
        method: str = 'fgsm',  # fgsm, pgd, trades
        epsilon: float = 0.01,
        start_epoch: int = 3,
        **kwargs
    ):
        self.method = method
        self.epsilon = epsilon
        self.start_epoch = start_epoch
        
        # Initialize adversarial method
        if method == 'fgsm':
            self.adv_method = FGSM(epsilon=epsilon)
        elif method == 'pgd':
            self.adv_method = PGD(
                epsilon=epsilon,
                alpha=kwargs.get('alpha', 0.001),
                num_steps=kwargs.get('num_steps', 5),
                random_start=kwargs.get('random_start', True),
            )
        elif method == 'trades':
            self.adv_method = TRADES(
                beta=kwargs.get('beta', 6.0),
                epsilon=epsilon,
                num_steps=kwargs.get('num_steps', 10),
                step_size=kwargs.get('step_size', 0.001),
            )
        else:
            raise ValueError(f"Unknown adversarial method: {method}")
    
    def training_step(
        self,
        model: nn.Module,
        batch: dict,
        loss_fn: Callable,
        current_epoch: int,
    ) -> torch.Tensor:
        """
        Adversarial training step
        
        Args:
            model: Model
            batch: Batch data
            loss_fn: Loss function
            current_epoch: Current training epoch
        
        Returns:
            loss: Total loss (clean + adversarial)
        """
        # Skip adversarial training in early epochs
        if current_epoch < self.start_epoch:
            outputs = model(**batch)
            return loss_fn(outputs, batch['labels'])
        
        # Get embeddings
        embeddings, pattern_scores = model.embedding(
            input_ids=batch['input_ids'],
            char_ids=batch.get('char_ids'),
            attention_mask=batch.get('attention_mask'),
        )
        
        # Clean loss
        outputs = model.forward_from_embeddings(embeddings, batch.get('attention_mask'))
        clean_loss = loss_fn(outputs, batch['labels'])
        
        # Generate adversarial examples
        if self.method == 'trades':
            # TRADES computes its own loss
            total_loss = self.adv_method.compute_loss(
                model, embeddings, batch['labels'], loss_fn
            )
        else:
            # FGSM/PGD: train on adversarial examples
            adv_embeddings = self.adv_method.generate_adversarial(
                model, embeddings, batch['labels'], loss_fn
            )
            
            adv_outputs = model.forward_from_embeddings(
                adv_embeddings, batch.get('attention_mask')
            )
            adv_loss = loss_fn(adv_outputs, batch['labels'])
            
            # Combined loss (50% clean, 50% adversarial)
            total_loss = 0.5 * clean_loss + 0.5 * adv_loss
        
        return total_loss

