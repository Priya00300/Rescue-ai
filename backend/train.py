"""
Complete training script for RescueAI Vision System
"""

import torch
import argparse
from pathlib import Path
import sys

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from models.multi_head_segmentation import MultiHeadSegmentation
from models.dataset_loader import DatasetBuilder
from models.training import Trainer


def train(
    data_dir='../datasets',
    batch_size=4,
    num_epochs=30,
    learning_rate=1e-4,
    encoder_name='resnet34',
    device='cuda'
):
    """
    Train the vision model.
    """
    print("=" * 60)
    print("RescueAI Vision System - Training")
    print("=" * 60)
    print()
    
    print("📋 Configuration:")
    print(f"  data_dir: {data_dir}")
    print(f"  batch_size: {batch_size}")
    print(f"  num_epochs: {num_epochs}")
    print(f"  learning_rate: {learning_rate}")
    print(f"  encoder: {encoder_name}")
    print(f"  device: {device}")
    print()
    
    # Create directories
    Path('checkpoints').mkdir(exist_ok=True)
    Path('logs').mkdir(exist_ok=True)
    
    # Set device
    if device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, using CPU")
        device = 'cpu'
    
    device = torch.device(device)
    print(f"🖥️  Using device: {device}")
    print()
    
    # Build dataloaders
    print("📂 Loading datasets...")
    try:
        train_loader, val_loader = DatasetBuilder.build_loaders(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=0,  # Use 0 for Windows
            image_size=(256, 256)
        )
        
        print(f"  ✅ Train samples: {len(train_loader.dataset)}")
        print(f"  ✅ Val samples: {len(val_loader.dataset)}")
        print(f"  ✅ Train batches: {len(train_loader)}")
        print(f"  ✅ Val batches: {len(val_loader)}")
        print()
    except Exception as e:
        print(f"❌ Error loading datasets: {e}")
        print("\nMake sure datasets are organized in:")
        print(f"  {Path(data_dir).absolute()}")
        return
    
    # Build model
    print("🏗️  Building model...")
    model = MultiHeadSegmentation(
        encoder_name=encoder_name,
        encoder_weights='imagenet',
        output_size=(256, 256)
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"  Architecture: Multi-Head Segmentation")
    print(f"  Encoder: {encoder_name}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print()
    
    # Create trainer
    print("🚀 Initializing trainer...")
    trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    device=str(device),
    checkpoint_dir='checkpoints',
    learning_rate=learning_rate
)
    print()
    
    # Start training
    print("=" * 60)
    print("Starting Training")
    print("=" * 60)
    print()
    
    try:
        trainer.train(num_epochs=num_epochs)
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Summary
    print()
    print("=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"  Best Val IoU: {trainer.best_val_iou:.4f}")
    print(f"  Best Val Loss: {trainer.best_val_loss:.4f}")
    print(f"  Model saved to: checkpoints/best_model.pth")
    print()


def main():
    """Main entry point with CLI arguments."""
    parser = argparse.ArgumentParser(description="RescueAI Vision System Training")
    
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval'],
                       help='Training or evaluation mode')
    parser.add_argument('--data_dir', type=str, default='../datasets',
                       help='Path to dataset directory')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--encoder', type=str, default='resnet34',
                       choices=['resnet34', 'efficientnet-b0'],
                       help='Encoder architecture')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cuda', 'cpu'],
                       help='Device to use')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            encoder_name=args.encoder,
            device=args.device
        )
    else:
        print("Evaluation mode not implemented yet")


if __name__ == "__main__":
    main()