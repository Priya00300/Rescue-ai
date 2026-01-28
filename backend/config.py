# ============================================================================
# config.py - Configuration Management
# ============================================================================

from dataclasses import dataclass
from typing import Tuple


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    
    # Data
    data_dir: str = "datasets"
    image_size: Tuple[int, int] = (256, 256)
    batch_size: int = 8
    num_workers: int = 4
    
    # Model
    encoder_name: str = "resnet34"  # or "efficientnet-b0"
    encoder_weights: str = "imagenet"
    
    # Training
    num_epochs: int = 30
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    
    # Task weights
    fire_weight: float = 1.0
    flood_weight: float = 1.0
    collapse_weight: float = 1.0
    human_weight: float = 1.5  # Higher priority
    
    # Optimization
    early_stopping_patience: int = 10
    lr_scheduler_patience: int = 5
    lr_scheduler_factor: float = 0.5
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_frequency: int = 5  # Save every N epochs
    
    # Device
    device: str = "cuda"  # or "cpu"
    
    # Logging
    log_dir: str = "logs"
    tensorboard: bool = True


@dataclass
class InferenceConfig:
    """Inference configuration."""
    
    checkpoint_path: str = "checkpoints/best_model.pth"
    image_size: Tuple[int, int] = (256, 256)
    device: str = "cuda"
    
    # Thresholds
    fire_threshold: float = 0.5
    flood_threshold: float = 0.5
    collapse_threshold: float = 0.5
    human_threshold: float = 0.6
    
    # Grid conversion
    grid_width: int = 30
    grid_height: int = 25