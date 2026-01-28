"""
Multi-Head Semantic Segmentation Model for RescueAI Vision System
Architecture: Shared Encoder + 4 Task-Specific Heads
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from typing import Dict, Tuple


class MultiHeadSegmentation(nn.Module):
    """
    Professional multi-task segmentation model for disaster analysis.
    """
    
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: str = "imagenet",
        input_channels: int = 3,
        output_size: Tuple[int, int] = (256, 256)
    ):
        super().__init__()
        
        self.output_size = output_size
        
        # Shared encoder (backbone) - pretrained on ImageNet
        self.encoder = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=input_channels,
            classes=1,
            activation=None
        )
        
        # Get encoder output channels
        if encoder_name == "resnet34":
            encoder_channels = 512
        elif encoder_name == "efficientnet-b0":
            encoder_channels = 320
        else:
            encoder_channels = 512
        
        # Task-specific segmentation heads
        self.fire_head = self._make_segmentation_head(encoder_channels)
        self.flood_head = self._make_segmentation_head(encoder_channels)
        self.collapse_head = self._make_segmentation_head(encoder_channels)
        self.human_head = self._make_segmentation_head(encoder_channels)
        
        # Output activations
        self.sigmoid = nn.Sigmoid()
        
    def _make_segmentation_head(self, in_channels: int) -> nn.Module:
        """Creates a segmentation head."""
        return nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 1, kernel_size=1)
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through shared encoder and all heads.
        
        Args:
            x: Input tensor (B, 3, H, W)
            
        Returns:
            Dictionary with segmentation maps
        """
        # Get encoder features
        features = self.encoder.encoder(x)
        
        # Get the last feature map (deepest layer)
        if isinstance(features, (list, tuple)):
            decoder_output = features[-1]
        else:
            decoder_output = features
        
        # Pass through each head
        fire_logits = self.fire_head(decoder_output)
        flood_logits = self.flood_head(decoder_output)
        collapse_logits = self.collapse_head(decoder_output)
        human_logits = self.human_head(decoder_output)
        
        # Resize to target output size
        fire_map = torch.nn.functional.interpolate(
            fire_logits, size=self.output_size, mode='bilinear', align_corners=False
        )
        flood_map = torch.nn.functional.interpolate(
            flood_logits, size=self.output_size, mode='bilinear', align_corners=False
        )
        collapse_map = torch.nn.functional.interpolate(
            collapse_logits, size=self.output_size, mode='bilinear', align_corners=False
        )
        human_map = torch.nn.functional.interpolate(
            human_logits, size=self.output_size, mode='bilinear', align_corners=False
        )
        
        # Apply sigmoid for probability outputs
        return {
            'fire': self.sigmoid(fire_map),
            'flood': self.sigmoid(flood_map),
            'collapse': self.sigmoid(collapse_map),
            'human': self.sigmoid(human_map)
        }
    
    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> Dict[str, torch.Tensor]:
        """Inference with thresholding."""
        with torch.no_grad():
            outputs = self.forward(x)
            predictions = {
                key: (value > threshold).float()
                for key, value in outputs.items()
            }
            return predictions
    
    def get_confidence_score(self, outputs: Dict[str, torch.Tensor]) -> float:
        """Calculate overall confidence score."""
        confidences = []
        for key, value in outputs.items():
            max_probs = value.max(dim=-1)[0].max(dim=-1)[0]
            confidences.append(max_probs.mean().item())
        return sum(confidences) / len(confidences)


class DiceLoss(nn.Module):
    """Dice Loss for segmentation tasks."""
    
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        
        intersection = (pred * target).sum()
        dice_coef = (2.0 * intersection + self.smooth) / (
            pred.sum() + target.sum() + self.smooth
        )
        
        return 1.0 - dice_coef


class CombinedLoss(nn.Module):
    """Combined BCE + Dice Loss."""
    
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCELoss()
        self.dice = DiceLoss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


class MultiTaskLoss(nn.Module):
    """Multi-task loss with task-specific weighting."""
    
    def __init__(
        self,
        fire_weight: float = 1.0,
        flood_weight: float = 1.0,
        collapse_weight: float = 1.0,
        human_weight: float = 1.5
    ):
        super().__init__()
        self.weights = {
            'fire': fire_weight,
            'flood': flood_weight,
            'collapse': collapse_weight,
            'human': human_weight
        }
        self.criterion = CombinedLoss()
        self.mse = nn.MSELoss()
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        masks: Dict[str, torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        total_loss = 0.0
        task_losses = {}
        
        for task in ['fire', 'flood', 'collapse']:
            if task in targets:
                pred = predictions[task]
                target = targets[task]
                
                if masks and task in masks:
                    mask = masks[task]
                    pred = pred * mask
                    target = target * mask
                
                loss = self.criterion(pred, target)
                weighted_loss = self.weights[task] * loss
                
                total_loss += weighted_loss
                task_losses[task] = loss.item()
        
        if 'human' in targets:
            pred = predictions['human']
            target = targets['human']
            
            if masks and 'human' in masks:
                mask = masks['human']
                pred = pred * mask
                target = target * mask
            
            loss = self.mse(pred, target)
            weighted_loss = self.weights['human'] * loss
            
            total_loss += weighted_loss
            task_losses['human'] = loss.item()
        
        return total_loss, task_losses


if __name__ == "__main__":
    print("Testing Multi-Head Segmentation Model...")
    model = MultiHeadSegmentation()
    x = torch.randn(2, 3, 256, 256)
    outputs = model(x)
    
    print("\nModel outputs:")
    for key, value in outputs.items():
        print(f"  {key}: {value.shape}")
    
    confidence = model.get_confidence_score(outputs)
    print(f"\nConfidence: {confidence:.3f}")
    print("\n✅ Model architecture ready!")