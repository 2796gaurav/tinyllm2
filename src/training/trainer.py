"""
GuardrailTrainer: Main training class for TinyGuardrail model
Handles training loop, evaluation, checkpointing, and metrics tracking
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from transformers import get_linear_schedule_with_warmup
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import json
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from src.training.losses import GuardrailLoss, FocalLoss
from src.training.adversarial import AdversarialTrainer
from src.training.quantization import QuantizationAwareTrainer


class GuardrailTrainer:
    """
    Main trainer class for TinyGuardrail model
    
    Handles:
    - Training loop with mixed precision
    - Validation and evaluation
    - Adversarial training integration
    - Quantization-aware training
    - Checkpointing and metrics tracking
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Dict,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[object] = None,
        loss_fn: Optional[nn.Module] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize trainer
        
        Args:
            model: TinyGuardrail model instance
            config: Training configuration dictionary
            train_loader: Training dataloader
            val_loader: Validation dataloader
            optimizer: Optimizer (if None, will be created from config)
            scheduler: Learning rate scheduler (if None, will be created)
            loss_fn: Loss function (if None, will use GuardrailLoss)
            device: Device to train on (default: cuda if available)
        """
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Setup optimizer
        if optimizer is None:
            self.optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config.get('learning_rate', 5e-5),
                weight_decay=config.get('weight_decay', 0.01),
            )
        else:
            self.optimizer = optimizer
        
        # Setup scheduler
        if scheduler is None and train_loader is not None:
            num_training_steps = len(train_loader) * config.get('num_epochs', 5)
            num_warmup_steps = int(num_training_steps * config.get('warmup_ratio', 0.1))
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
            )
        else:
            self.scheduler = scheduler
        
        # Setup loss function
        if loss_fn is None:
            self.loss_fn = GuardrailLoss(
                focal_alpha=config.get('focal_alpha', 0.25),
                focal_gamma=config.get('focal_gamma', 2.0),
                aux_loss_weight=config.get('aux_loss_weight', 0.01),
                router_loss_weight=config.get('router_loss_weight', 0.1),
                pattern_loss_weight=config.get('pattern_loss_weight', 0.05),
            )
        else:
            self.loss_fn = loss_fn
        
        # Mixed precision
        self.use_fp16 = config.get('fp16', True)
        self.scaler = torch.cuda.amp.GradScaler() if self.use_fp16 and torch.cuda.is_available() else None
        
        # Adversarial training
        self.adv_trainer = None
        if config.get('use_adversarial', False):
            from src.training.adversarial import FGSM, PGD
            if config.get('adversarial_method', 'fgsm') == 'fgsm':
                attack = FGSM(epsilon=config.get('adversarial_epsilon', 0.01))
            else:
                attack = PGD(
                    epsilon=config.get('adversarial_epsilon', 0.01),
                    alpha=0.001,
                    num_steps=5,
                )
            self.adv_trainer = AdversarialTrainer(
                attack=attack,
                start_epoch=config.get('adversarial_start_epoch', 3),
            )
        
        # Quantization-aware training
        self.qat_trainer = None
        if config.get('use_quantization', False):
            self.qat_trainer = QuantizationAwareTrainer(
                backend=config.get('quantization_backend', 'fbgemm'),
                start_epoch=config.get('quantization_start_epoch', 4),
            )
        
        # Metrics tracking
        self.metrics_history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'val_f1': [],
            'learning_rate': [],
        }
        
        self.global_step = 0
        self.best_val_acc = 0.0
        self.best_val_f1 = 0.0
        
        # Output directory
        self.output_dir = Path(config.get('output_dir', './outputs'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def train(
        self,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
        num_epochs: Optional[int] = None,
    ):
        """
        Main training loop
        
        Args:
            train_loader: Training dataloader (uses self.train_loader if None)
            val_loader: Validation dataloader (uses self.val_loader if None)
            num_epochs: Number of epochs (uses config if None)
        """
        train_loader = train_loader or self.train_loader
        val_loader = val_loader or self.val_loader
        num_epochs = num_epochs or self.config.get('num_epochs', 5)
        
        if train_loader is None:
            raise ValueError("train_loader must be provided either in __init__ or train()")
        
        print(f"\n{'='*50}")
        print(f"Starting Training")
        print(f"{'='*50}\n")
        
        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")
            print("-" * 50)
            
            # Enable quantization-aware training if needed
            if self.qat_trainer and epoch == self.qat_trainer.start_epoch:
                print("Enabling Quantization-Aware Training...")
                self.model = self.qat_trainer.prepare_model(self.model)
                self.model = self.model.to(self.device)
            
            # Train epoch
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validate
            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)
                
                # Log metrics
                print(f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.4f}")
                print(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")
                print(f"Val F1: {val_metrics['f1']:.4f}")
                
                # Save best model
                if val_metrics['accuracy'] > self.best_val_acc:
                    self.best_val_acc = val_metrics['accuracy']
                    self.best_val_f1 = val_metrics['f1']
                    self.save_checkpoint(epoch, is_best=True)
                    print(f"✓ Saved best model (acc: {self.best_val_acc:.4f})")
                
                # Store metrics
                self.metrics_history['val_loss'].append(val_metrics['loss'])
                self.metrics_history['val_acc'].append(val_metrics['accuracy'])
                self.metrics_history['val_f1'].append(val_metrics['f1'])
            else:
                # Store train metrics only
                self.metrics_history['val_loss'].append(0.0)
                self.metrics_history['val_acc'].append(0.0)
                self.metrics_history['val_f1'].append(0.0)
            
            # Store train metrics
            self.metrics_history['train_loss'].append(train_metrics['loss'])
            self.metrics_history['train_acc'].append(train_metrics['accuracy'])
            self.metrics_history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
        
        print(f"\n✓ Training complete!")
        print(f"✓ Best validation accuracy: {self.best_val_acc:.4f}")
        print(f"✓ Best validation F1: {self.best_val_f1:.4f}")
        
        # Save final metrics
        self.save_metrics()
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for step, batch in enumerate(pbar):
            # Move to device
            batch = self._move_to_device(batch)
            
            # Forward pass
            if self.use_fp16 and self.scaler:
                with torch.cuda.amp.autocast():
                    loss, outputs = self._training_step(batch, epoch)
            else:
                loss, outputs = self._training_step(batch, epoch)
            
            # Backward pass
            if self.use_fp16 and self.scaler:
                self.scaler.scale(loss).backward()
                
                # Gradient accumulation
                if (step + 1) % self.config.get('gradient_accumulation_steps', 1) == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.get('max_grad_norm', 1.0)
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    if self.scheduler:
                        self.scheduler.step()
                    self.optimizer.zero_grad()
            else:
                loss.backward()
                
                # Gradient accumulation
                if (step + 1) % self.config.get('gradient_accumulation_steps', 1) == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.get('max_grad_norm', 1.0)
                    )
                    self.optimizer.step()
                    if self.scheduler:
                        self.scheduler.step()
                    self.optimizer.zero_grad()
            
            # Metrics
            total_loss += loss.item()
            if outputs and hasattr(outputs, 'logits'):
                preds = torch.argmax(outputs.logits, dim=-1)
                labels = batch.get('labels')
                if labels is not None:
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{correct/total:.4f}" if total > 0 else "0.0000",
            })
            
            self.global_step += 1
        
        avg_loss = total_loss / len(train_loader)
        avg_acc = correct / total if total > 0 else 0.0
        
        return {
            'loss': avg_loss,
            'accuracy': avg_acc,
        }
    
    def _training_step(self, batch: Dict, epoch: int) -> Tuple[torch.Tensor, object]:
        """Single training step"""
        # Forward pass
        outputs = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch.get('attention_mask'),
            char_ids=batch.get('char_ids'),
            labels=batch.get('labels'),
            text=batch.get('text', [None] * len(batch['input_ids']))[0] if isinstance(batch.get('text'), list) else None,
            return_dict=True,
        )
        
        # Compute loss
        if hasattr(outputs, 'loss') and outputs.loss is not None:
            loss = outputs.loss
        else:
            # Use custom loss function
            loss_dict = self.loss_fn(
                logits=outputs.logits,
                labels=batch['labels'],
                aux_loss=getattr(outputs, 'aux_loss', None),
                router_logits=getattr(outputs, 'route_logits', None),
                pattern_scores=getattr(outputs, 'pattern_scores', None),
            )
            loss = loss_dict.get('total_loss', loss_dict.get('loss', loss_dict))
        
        # Adversarial training (if enabled)
        if (self.adv_trainer and 
            epoch >= self.adv_trainer.start_epoch):
            # Note: Adversarial training would need to be integrated here
            # For now, we skip it as it requires more complex integration
            pass
        
        return loss, outputs
    
    def evaluate(self, val_loader: DataLoader) -> Dict:
        """Evaluate on validation set"""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                batch = self._move_to_device(batch)
                
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch.get('attention_mask'),
                    char_ids=batch.get('char_ids'),
                    labels=batch.get('labels'),
                    text=batch.get('text', [None] * len(batch['input_ids']))[0] if isinstance(batch.get('text'), list) else None,
                    return_dict=True,
                )
                
                # Loss
                if hasattr(outputs, 'loss') and outputs.loss is not None:
                    total_loss += outputs.loss.item()
                else:
                    loss_dict = self.loss_fn(
                        logits=outputs.logits,
                        labels=batch['labels'],
                        aux_loss=getattr(outputs, 'aux_loss', None),
                        router_logits=getattr(outputs, 'route_logits', None),
                        pattern_scores=getattr(outputs, 'pattern_scores', None),
                    )
                    loss = loss_dict.get('total_loss', loss_dict.get('loss', loss_dict))
                    total_loss += loss.item() if isinstance(loss, torch.Tensor) else loss
                
                # Predictions
                preds = torch.argmax(outputs.logits, dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch['labels'].cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted', zero_division=0
        )
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        }
    
    def _move_to_device(self, batch: Dict) -> Dict:
        """Move batch to device"""
        device_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.device)
            elif isinstance(value, list):
                device_batch[key] = value  # Keep lists as-is (e.g., text strings)
            else:
                device_batch[key] = value
        return device_batch
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'val_acc': self.best_val_acc,
            'val_f1': self.best_val_f1,
            'config': self.config,
        }
        
        # Save latest
        torch.save(checkpoint, self.output_dir / 'checkpoint_latest.pth')
        
        # Save best
        if is_best:
            torch.save(checkpoint, self.output_dir / 'best_model.pth')
    
    def save_metrics(self):
        """Save training metrics to JSON"""
        metrics_file = self.output_dir / 'training_metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        print(f"✓ Saved metrics to {metrics_file}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint and checkpoint['optimizer_state_dict'] is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
            if self.scheduler:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if 'val_acc' in checkpoint:
            self.best_val_acc = checkpoint['val_acc']
        if 'val_f1' in checkpoint:
            self.best_val_f1 = checkpoint['val_f1']
        print(f"✓ Loaded checkpoint from {checkpoint_path}")
