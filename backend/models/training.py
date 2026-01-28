"""
Professional training pipeline with early stopping, checkpointing, and metrics.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Dict, Optional, Tuple
import numpy as np
from tqdm import tqdm
import json
from pathlib import Path

from models.multi_head_segmentation import MultiHeadSegmentation, MultiTaskLoss


class MetricsCalculator:
    """Calculate IoU and other segmentation metrics."""
    
    @staticmethod
    def calculate_iou(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
        """
        Calculate Intersection over Union (IoU).
        
        Args:
            pred: Predictions (B, 1, H, W)
            target: Ground truth (B, 1, H, W)
            threshold: Binarization threshold
            
        Returns:
            IoU score (0-1)
        """
        pred_binary = (pred > threshold).float()
        target_binary = target.float()
        
        intersection = (pred_binary * target_binary).sum()
        union = pred_binary.sum() + target_binary.sum() - intersection
        
        if union == 0:
            return 1.0 if intersection == 0 else 0.0
        
        return (intersection / union).item()
    
    @staticmethod
    def calculate_dice(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
        """
        Calculate Dice coefficient.
        
        Args:
            pred: Predictions (B, 1, H, W)
            target: Ground truth (B, 1, H, W)
            threshold: Binarization threshold
            
        Returns:
            Dice score (0-1)
        """
        pred_binary = (pred > threshold).float()
        target_binary = target.float()
        
        intersection = (pred_binary * target_binary).sum()
        total = pred_binary.sum() + target_binary.sum()
        
        if total == 0:
            return 1.0 if intersection == 0 else 0.0
        
        return (2.0 * intersection / total).item()


class Trainer:
    """
    Professional training pipeline with:
    - Early stopping
    - Learning rate scheduling
    - Checkpointing
    - TensorBoard logging
    """
    
    def __init__(
        self,
        model: MultiHeadSegmentation,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = 'cuda',
        checkpoint_dir: str = 'checkpoints',
        learning_rate: float = 1e-4,
        task_weights: Optional[Dict[str, float]] = None
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Loss and optimizer
        self.criterion = MultiTaskLoss(**(task_weights or {}))
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=1e-4
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        
        # Metrics
        self.metrics_calculator = MetricsCalculator()
        self.best_val_loss = float('inf')
        self.best_val_iou = 0.0
        self.patience_counter = 0
        self.early_stopping_patience = 10
        
        # History
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_iou': [],
            'val_iou': [],
            'learning_rates': []
        }
    
    def train_epoch(self) -> Tuple[float, float, Dict[str, float]]:
        """
        Train for one epoch.
        
        Returns:
            Tuple of (avg_loss, avg_iou, task_losses)
        """
        self.model.train()
        
        total_loss = 0.0
        total_iou = 0.0
        task_losses_sum = {'fire': 0.0, 'flood': 0.0, 'collapse': 0.0, 'human': 0.0}
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        
        for batch in pbar:
            images = batch['image'].to(self.device)
            targets = {
                key: batch[key].to(self.device)
                for key in ['fire', 'flood', 'collapse', 'human']
                if key in batch
            }
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(images)
            
            # Calculate loss
            loss, task_losses = self.criterion(predictions, targets)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Calculate IoU (on fire as representative)
            if 'fire' in predictions and 'fire' in targets:
                iou = self.metrics_calculator.calculate_iou(
                    predictions['fire'],
                    targets['fire']
                )
                total_iou += iou
            
            # Accumulate
            total_loss += loss.item()
            for key in task_losses:
                task_losses_sum[key] += task_losses[key]
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'iou': f'{iou:.4f}' if 'fire' in predictions else 'N/A'
            })
        
        avg_loss = total_loss / num_batches
        avg_iou = total_iou / num_batches if total_iou > 0 else 0.0
        avg_task_losses = {
            key: value / num_batches
            for key, value in task_losses_sum.items()
        }
        
        return avg_loss, avg_iou, avg_task_losses
    
    def validate(self) -> Tuple[float, float, Dict[str, float]]:
        """
        Validate the model.
        
        Returns:
            Tuple of (avg_loss, avg_iou, task_ious)
        """
        self.model.eval()
        
        total_loss = 0.0
        task_ious = {'fire': 0.0, 'flood': 0.0, 'collapse': 0.0, 'human': 0.0}
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                images = batch['image'].to(self.device)
                targets = {
                    key: batch[key].to(self.device)
                    for key in ['fire', 'flood', 'collapse', 'human']
                    if key in batch
                }
                
                # Forward pass
                predictions = self.model(images)
                
                # Calculate loss
                loss, _ = self.criterion(predictions, targets)
                total_loss += loss.item()
                
                # Calculate IoU for each task
                for key in predictions:
                    if key in targets:
                        iou = self.metrics_calculator.calculate_iou(
                            predictions[key],
                            targets[key]
                        )
                        task_ious[key] += iou
                
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_task_ious = {
            key: value / num_batches
            for key, value in task_ious.items()
        }
        avg_iou = np.mean(list(avg_task_ious.values()))
        
        return avg_loss, avg_iou, avg_task_ious
    
    def save_checkpoint(self, epoch: int, val_loss: float, val_iou: float, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'val_iou': val_iou,
            'history': self.history
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"✅ Saved best model (IoU: {val_iou:.4f})")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint.get('history', self.history)
        
        print(f"✅ Loaded checkpoint from {checkpoint_path}")
        return checkpoint['epoch']
    
    def train(self, num_epochs: int, resume_from: Optional[str] = None):
        """
        Main training loop.
        
        Args:
            num_epochs: Number of epochs to train
            resume_from: Path to checkpoint to resume from
        """
        start_epoch = 0
        
        if resume_from:
            start_epoch = self.load_checkpoint(resume_from) + 1
        
        print(f"\n🚀 Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")
        
        for epoch in range(start_epoch, num_epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"{'='*60}")
            
            # Training
            train_loss, train_iou, task_losses = self.train_epoch()
            
            print(f"\n📊 Training Results:")
            print(f"  Loss: {train_loss:.4f}")
            print(f"  IoU: {train_iou:.4f}")
            print(f"  Task Losses: {json.dumps({k: f'{v:.4f}' for k, v in task_losses.items()}, indent=4)}")
            
            # Validation
            val_loss, val_iou, task_ious = self.validate()
            
            print(f"\n📊 Validation Results:")
            print(f"  Loss: {val_loss:.4f}")
            print(f"  IoU: {val_iou:.4f}")
            print(f"  Task IoUs: {json.dumps({k: f'{v:.4f}' for k, v in task_ious.items()}, indent=4)}")
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"\n📈 Learning Rate: {current_lr:.6f}")
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_iou'].append(train_iou)
            self.history['val_iou'].append(val_iou)
            self.history['learning_rates'].append(current_lr)
            
            # Save checkpoint
            is_best = val_iou > self.best_val_iou
            if is_best:
                self.best_val_iou = val_iou
                self.best_val_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            self.save_checkpoint(epoch + 1, val_loss, val_iou, is_best)
            
            # Early stopping
            if self.patience_counter >= self.early_stopping_patience:
                print(f"\n⚠️ Early stopping triggered after {epoch + 1} epochs")
                print(f"Best Val IoU: {self.best_val_iou:.4f}")
                break
        
        print(f"\n✅ Training completed!")
        print(f"Best Val IoU: {self.best_val_iou:.4f}")
        print(f"Best Val Loss: {self.best_val_loss:.4f}")
        
        # Save training history
        history_path = self.checkpoint_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"Training history saved to {history_path}")


# Quick test
if __name__ == "__main__":
    print("Training pipeline ready!")
    print("\nTo train:")
    print("1. Prepare your dataset loaders")
    print("2. Create model: model = MultiHeadSegmentation()")
    print("3. Create trainer: trainer = Trainer(model, train_loader, val_loader)")
    print("4. Start training: trainer.train(num_epochs=30)")