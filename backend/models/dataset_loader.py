"""
Multi-source dataset loader with weak supervision support.
Handles:
- Fire datasets (FireNet, FLAME)
- Flood datasets (FloodNet, Sen1Floods11)
- Collapse datasets (xView2, RescueNet)
- Human detection (CrowdHuman, VisDrone)
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import json
from PIL import Image

from utils.preprocessing import DisasterAugmentations, HeatmapGenerator


class MultiTaskDisasterDataset(Dataset):
    """
    Multi-task dataset with weak supervision.
    Not all images have all labels - uses ignore_index for missing tasks.
    """
    
    def __init__(
        self,
        data_dir: str,
        mode: str = 'train',
        image_size: Tuple[int, int] = (256, 256),
        augment: bool = True
    ):
        """
        Args:
            data_dir: Root directory with subdirs: fire/, flood/, collapse/, human/
            mode: 'train', 'val', or 'test'
            image_size: Target (height, width)
            augment: Apply augmentations
        """
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.image_size = image_size
        
        # Setup transforms
        if augment and mode == 'train':
            self.transform = DisasterAugmentations.get_train_transforms(image_size)
        else:
            self.transform = DisasterAugmentations.get_val_transforms(image_size)
        
        # Load dataset index
        self.samples = self._load_dataset_index()
        
        print(f"Loaded {len(self.samples)} samples for {mode}")
    
    def _load_dataset_index(self) -> List[Dict]:
        """
        Load dataset index with available labels.
        
        Expected structure:
        data_dir/
        ├── fire/
        │   ├── images/
        │   └── masks/
        ├── flood/
        │   ├── images/
        │   └── masks/
        ├── collapse/
        │   ├── images/
        │   └── masks/
        └── human/
            ├── images/
            └── annotations.json  # Bounding boxes
        """
        samples = []
        
        # Fire samples
        fire_dir = self.data_dir / 'fire' / self.mode
        if fire_dir.exists():
            samples.extend(self._load_task_samples(fire_dir, 'fire'))
        
        # Flood samples
        flood_dir = self.data_dir / 'flood' / self.mode
        if flood_dir.exists():
            samples.extend(self._load_task_samples(flood_dir, 'flood'))
        
        # Collapse samples
        collapse_dir = self.data_dir / 'collapse' / self.mode
        if collapse_dir.exists():
            samples.extend(self._load_task_samples(collapse_dir, 'collapse'))
        
        # Human samples
        human_dir = self.data_dir / 'human' / self.mode
        if human_dir.exists():
            samples.extend(self._load_human_samples(human_dir))
        
        return samples
    
    def _load_task_samples(self, task_dir: Path, task_name: str) -> List[Dict]:
        """Load samples for segmentation tasks (fire, flood, collapse)."""
        samples = []
        
        image_dir = task_dir / 'images'
        mask_dir = task_dir / 'masks'
        
        if not image_dir.exists():
            return samples
        
        for img_path in image_dir.glob('*.jpg'):
            # Find corresponding mask
            mask_path = mask_dir / f"{img_path.stem}.png"
            
            if mask_path.exists():
                samples.append({
                    'image_path': str(img_path),
                    'masks': {task_name: str(mask_path)},
                    'available_tasks': [task_name]
                })
        
        return samples
    
    def _load_human_samples(self, human_dir: Path) -> List[Dict]:
        """Load samples with human bounding boxes."""
        samples = []
        
        image_dir = human_dir / 'images'
        anno_file = human_dir / 'annotations.json'
        
        if not anno_file.exists():
            return samples
        
        # Load annotations
        with open(anno_file, 'r') as f:
            annotations = json.load(f)
        
        for img_name, bboxes in annotations.items():
            img_path = image_dir / img_name
            
            if img_path.exists():
                samples.append({
                    'image_path': str(img_path),
                    'bboxes': bboxes,  # List of [x1, y1, x2, y2]
                    'available_tasks': ['human']
                })
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a sample with all available labels.
        
        Returns:
            Dictionary with:
            - image: Tensor (3, H, W)
            - fire: Tensor (1, H, W) or None
            - flood: Tensor (1, H, W) or None
            - collapse: Tensor (1, H, W) or None
            - human: Tensor (1, H, W) or None
            - mask: Tensor (4, H, W) indicating valid labels
        """
        sample = self.samples[idx]
        
        # Load image
        image = cv2.imread(sample['image_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Initialize masks
        masks_dict = {}
        valid_mask = {
            'fire': False,
            'flood': False,
            'collapse': False,
            'human': False
        }
        
        # Load segmentation masks
        if 'masks' in sample:
            for task, mask_path in sample['masks'].items():
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    masks_dict[task] = mask
                    valid_mask[task] = True
        
        # Load human bounding boxes and convert to heatmap
        if 'bboxes' in sample:
            bboxes = np.array(sample['bboxes'])
            heatmap = HeatmapGenerator.bboxes_to_heatmap(
                bboxes,
                image.shape[:2]
            )
            masks_dict['human'] = (heatmap * 255).astype(np.uint8)
            valid_mask['human'] = True
        
        # Apply transforms
        # Prepare masks for albumentations
        mask_list = []
        mask_keys = []
        for key in ['fire', 'flood', 'collapse', 'human']:
            if key in masks_dict:
                mask_list.append(masks_dict[key])
                mask_keys.append(key)
        
        if mask_list:
            transformed = self.transform(
                image=image,
                masks=mask_list
            )
            
            image_tensor = transformed['image']
            
            # Reconstruct masks
            output = {'image': image_tensor}
            
            for i, key in enumerate(mask_keys):
                mask_tensor = transformed['masks'][i]
                
                # Convert to tensor if needed
                if not isinstance(mask_tensor, torch.Tensor):
                    mask_tensor = torch.from_numpy(mask_tensor)
                
                # Normalize to 0-1
                mask_tensor = mask_tensor.float() / 255.0
                
                # Add channel dimension if needed
                if mask_tensor.ndim == 2:
                    mask_tensor = mask_tensor.unsqueeze(0)
                
                output[key] = mask_tensor
            
            # Add valid mask indicators
            for key in ['fire', 'flood', 'collapse', 'human']:
                if key not in output:
                    # Create dummy mask
                    output[key] = torch.zeros(1, self.image_size[0], self.image_size[1])
                    output[f'{key}_valid'] = torch.tensor(0.0)
                else:
                    output[f'{key}_valid'] = torch.tensor(1.0)
            
            return output
        
        else:
            # Image only (no labels)
            transformed = self.transform(image=image)
            
            output = {'image': transformed['image']}
            
            for key in ['fire', 'flood', 'collapse', 'human']:
                output[key] = torch.zeros(1, self.image_size[0], self.image_size[1])
                output[f'{key}_valid'] = torch.tensor(0.0)
            
            return output


class DatasetBuilder:
    """
    Builds training and validation dataloaders.
    """
    
    @staticmethod
    def build_loaders(
        data_dir: str,
        batch_size: int = 8,
        num_workers: int = 4,
        image_size: Tuple[int, int] = (256, 256)
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Build train and validation dataloaders.
        
        Args:
            data_dir: Root data directory
            batch_size: Batch size
            num_workers: Number of worker processes
            image_size: Target image size
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        # Create datasets
        train_dataset = MultiTaskDisasterDataset(
            data_dir=data_dir,
            mode='train',
            image_size=image_size,
            augment=True
        )
        
        val_dataset = MultiTaskDisasterDataset(
            data_dir=data_dir,
            mode='val',
            image_size=image_size,
            augment=False
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        return train_loader, val_loader


# Dataset preparation guide
DATASET_GUIDE = """
# Dataset Preparation Guide

## Directory Structure

datasets/
├── fire/
│   ├── train/
│   │   ├── images/
│   │   └── masks/
│   └── val/
│       ├── images/
│       └── masks/
├── flood/
│   ├── train/
│   │   ├── images/
│   │   └── masks/
│   └── val/
│       ├── images/
│       └── masks/
├── collapse/
│   ├── train/
│   │   ├── images/
│   │   └── masks/
│   └── val/
│       ├── images/
│       └── masks/
└── human/
    ├── train/
    │   ├── images/
    │   └── annotations.json
    └── val/
        ├── images/
        └── annotations.json

## Mask Format

- Binary masks (0=background, 255=foreground)
- PNG format
- Same dimensions as input image

## Annotations Format (Human Detection)

{
  "image_001.jpg": [
    [x1, y1, x2, y2],  // Bounding box 1
    [x1, y1, x2, y2]   // Bounding box 2
  ],
  "image_002.jpg": [
    [x1, y1, x2, y2]
  ]
}

## Dataset Sources

### Fire & Smoke
- FireNet: https://github.com/OlafenwaMoses/FireNET
- FLAME: https://github.com/AiForMankind/wildfire-smoke-dataset

### Flood
- FloodNet: https://github.com/BinaLab/FloodNet-Supervised_v1.0
- Sen1Floods11: https://github.com/cloudtostreet/Sen1Floods11

### Collapsed Buildings
- xView2: https://xview2.org/
- RescueNet: https://github.com/BinaLab/RescueNet

### Human Presence
- CrowdHuman: https://www.crowdhuman.org/
- VisDrone: https://github.com/VisDrone/VisDrone-Dataset
"""


if __name__ == "__main__":
    print(DATASET_GUIDE)
    
    print("\n" + "="*60)
    print("Dataset Loader Ready!")
    print("="*60)
    
    print("\nTo use:")
    print("1. Organize datasets as shown above")
    print("2. Create loaders:")
    print("   train_loader, val_loader = DatasetBuilder.build_loaders('datasets/')")
    print("3. Start training!")