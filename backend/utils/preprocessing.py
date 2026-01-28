"""
Professional data preprocessing and augmentation for disaster imagery.
Simulates real-world conditions: smoke, motion blur, lighting variations.
"""

import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Dict, Tuple, Optional
import torch


class DisasterAugmentations:
    """
    Augmentation pipeline specifically designed for disaster imagery.
    Simulates realistic conditions: drone footage, smoke, poor lighting.
    """
    
    @staticmethod
    def get_train_transforms(image_size: Tuple[int, int] = (256, 256)) -> A.Compose:
        """
        Training augmentations with realistic disaster conditions.
        
        Args:
            image_size: Target (height, width)
            
        Returns:
            Albumentations composition
        """
        return A.Compose([
            # Resize
            A.Resize(height=image_size[0], width=image_size[1]),
            
            # Geometric transforms (drone movement)
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),  # Slight rotation
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.2,
                rotate_limit=15,
                p=0.5
            ),
            
            # Lighting conditions (day/night, smoke)
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=0.7
            ),
            A.RandomGamma(gamma_limit=(70, 130), p=0.5),
            A.CLAHE(clip_limit=4.0, p=0.3),  # Enhance local contrast
            
            # Weather & atmospheric effects
            A.GaussianBlur(blur_limit=(3, 7), p=0.4),  # Smoke/haze
            A.MotionBlur(blur_limit=7, p=0.3),  # Drone movement
            A.GaussNoise(var_limit=(10, 50), p=0.3),  # Camera noise
            
            # Color variations (fire glow, water reflection)
            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=0.5
            ),
            
            # Occlusions (debris, smoke)
            A.CoarseDropout(
                max_holes=8,
                max_height=32,
                max_width=32,
                fill_value=0,
                p=0.3
            ),
            
            # Normalization (ImageNet stats)
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0
            ),
            
            ToTensorV2()
        ])
    
    @staticmethod
    def get_val_transforms(image_size: Tuple[int, int] = (256, 256)) -> A.Compose:
        """
        Validation transforms (no augmentation).
        
        Args:
            image_size: Target (height, width)
            
        Returns:
            Albumentations composition
        """
        return A.Compose([
            A.Resize(height=image_size[0], width=image_size[1]),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0
            ),
            ToTensorV2()
        ])
    
    @staticmethod
    def get_test_transforms(image_size: Tuple[int, int] = (256, 256)) -> A.Compose:
        """
        Test transforms (identical to validation).
        
        Args:
            image_size: Target (height, width)
            
        Returns:
            Albumentations composition
        """
        return DisasterAugmentations.get_val_transforms(image_size)


class ImagePreprocessor:
    """
    Handles all image preprocessing for the vision model.
    Converts raw images to model-ready tensors.
    """
    
    def __init__(self, image_size: Tuple[int, int] = (256, 256)):
        self.image_size = image_size
        self.train_transform = DisasterAugmentations.get_train_transforms(image_size)
        self.val_transform = DisasterAugmentations.get_val_transforms(image_size)
    
    def preprocess_image(
        self,
        image: np.ndarray,
        mode: str = 'val'
    ) -> torch.Tensor:
        """
        Preprocess a single image.
        
        Args:
            image: NumPy array (H, W, 3) in BGR or RGB
            mode: 'train' or 'val'
            
        Returns:
            Preprocessed tensor (1, 3, H, W)
        """
        # Convert BGR to RGB if needed
        if image.shape[-1] == 3 and self._is_bgr(image):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        transform = self.train_transform if mode == 'train' else self.val_transform
        transformed = transform(image=image)
        tensor = transformed['image']
        
        # Add batch dimension
        return tensor.unsqueeze(0)
    
    def preprocess_batch(
        self,
        images: np.ndarray,
        masks: Optional[Dict[str, np.ndarray]] = None,
        mode: str = 'train'
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Preprocess a batch of images with optional masks.
        
        Args:
            images: NumPy array (B, H, W, 3)
            masks: Dictionary of masks {task: (B, H, W)}
            mode: 'train' or 'val'
            
        Returns:
            Tuple of (image_batch, mask_batch)
        """
        transform = self.train_transform if mode == 'train' else self.val_transform
        
        processed_images = []
        processed_masks = {key: [] for key in (masks or {})}
        
        for i in range(len(images)):
            image = images[i]
            
            # Convert BGR to RGB if needed
            if self._is_bgr(image):
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            if masks:
                # Apply same transforms to image and masks
                mask_dict = {key: masks[key][i] for key in masks}
                
                transformed = transform(
                    image=image,
                    masks=list(mask_dict.values())
                )
                
                processed_images.append(transformed['image'])
                
                # Reconstruct masks
                for idx, key in enumerate(mask_dict.keys()):
                    mask = transformed['masks'][idx]
                    if isinstance(mask, np.ndarray):
                        mask = torch.from_numpy(mask)
                    processed_masks[key].append(mask)
            else:
                transformed = transform(image=image)
                processed_images.append(transformed['image'])
        
        # Stack into batches
        image_batch = torch.stack(processed_images)
        
        if masks:
            mask_batch = {
                key: torch.stack(processed_masks[key])
                for key in processed_masks
            }
            return image_batch, mask_batch
        
        return image_batch, None
    
    @staticmethod
    def _is_bgr(image: np.ndarray) -> bool:
        """
        Heuristic to detect if image is BGR (OpenCV format).
        Checks if blue channel has higher values than red.
        """
        if image.shape[-1] != 3:
            return False
        
        # Sample center region
        h, w = image.shape[:2]
        center = image[h//4:3*h//4, w//4:3*w//4]
        
        blue_mean = center[:, :, 0].mean()
        red_mean = center[:, :, 2].mean()
        
        # BGR if blue > red (common in outdoor images)
        return blue_mean > red_mean * 1.1


class VideoPreprocessor:
    """
    Handles video frame extraction and preprocessing.
    """
    
    def __init__(self, frame_rate: int = 2, image_size: Tuple[int, int] = (256, 256)):
        """
        Args:
            frame_rate: Extract N frames per second
            image_size: Target frame size
        """
        self.frame_rate = frame_rate
        self.image_preprocessor = ImagePreprocessor(image_size)
    
    def extract_frames(
        self,
        video_path: str,
        max_frames: Optional[int] = None
    ) -> np.ndarray:
        """
        Extract frames from video.
        
        Args:
            video_path: Path to video file
            max_frames: Maximum frames to extract (None = all)
            
        Returns:
            NumPy array (N, H, W, 3) of frames
        """
        cap = cv2.VideoCapture(video_path)
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = max(1, int(fps / self.frame_rate))
        
        frames = []
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extract every Nth frame
            if frame_idx % frame_interval == 0:
                frames.append(frame)
                
                if max_frames and len(frames) >= max_frames:
                    break
            
            frame_idx += 1
        
        cap.release()
        
        return np.array(frames)
    
    def preprocess_video(
        self,
        video_path: str,
        max_frames: Optional[int] = None
    ) -> torch.Tensor:
        """
        Extract and preprocess video frames.
        
        Args:
            video_path: Path to video file
            max_frames: Maximum frames to process
            
        Returns:
            Tensor (N, 3, H, W) of preprocessed frames
        """
        frames = self.extract_frames(video_path, max_frames)
        
        preprocessed = []
        for frame in frames:
            tensor = self.image_preprocessor.preprocess_image(frame, mode='val')
            preprocessed.append(tensor.squeeze(0))
        
        return torch.stack(preprocessed)


class HeatmapGenerator:
    """
    Converts bounding boxes to Gaussian heatmaps for human presence.
    This is CRITICAL for spatial intelligence.
    """
    
    @staticmethod
    def generate_gaussian_heatmap(
        bbox: Tuple[int, int, int, int],
        image_size: Tuple[int, int],
        sigma: float = 10.0
    ) -> np.ndarray:
        """
        Generate Gaussian heatmap from bounding box.
        
        Args:
            bbox: (x1, y1, x2, y2) coordinates
            image_size: (height, width)
            sigma: Gaussian spread
            
        Returns:
            Heatmap (H, W) with values 0-1
        """
        height, width = image_size
        heatmap = np.zeros((height, width), dtype=np.float32)
        
        # Calculate center
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        
        # Generate Gaussian
        for y in range(height):
            for x in range(width):
                dist = np.sqrt((x - cx)**2 + (y - cy)**2)
                heatmap[y, x] = np.exp(-(dist**2) / (2 * sigma**2))
        
        return heatmap
    
    @staticmethod
    def bboxes_to_heatmap(
        bboxes: np.ndarray,
        image_size: Tuple[int, int],
        sigma: float = 10.0
    ) -> np.ndarray:
        """
        Convert multiple bounding boxes to combined heatmap.
        
        Args:
            bboxes: Array of (N, 4) bounding boxes
            image_size: (height, width)
            sigma: Gaussian spread
            
        Returns:
            Combined heatmap (H, W)
        """
        heatmap = np.zeros(image_size, dtype=np.float32)
        
        for bbox in bboxes:
            single_heatmap = HeatmapGenerator.generate_gaussian_heatmap(
                bbox, image_size, sigma
            )
            heatmap = np.maximum(heatmap, single_heatmap)
        
        return heatmap


# Test preprocessing
if __name__ == "__main__":
    print("Testing preprocessing pipeline...")
    
    # Create dummy image
    image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    
    # Test preprocessor
    preprocessor = ImagePreprocessor(image_size=(256, 256))
    tensor = preprocessor.preprocess_image(image, mode='val')
    
    print(f"\nInput shape: {image.shape}")
    print(f"Output shape: {tensor.shape}")
    print(f"Value range: [{tensor.min():.3f}, {tensor.max():.3f}]")
    
    # Test heatmap generation
    bboxes = np.array([
        [100, 100, 150, 150],
        [200, 200, 250, 250]
    ])
    
    heatmap = HeatmapGenerator.bboxes_to_heatmap(bboxes, (256, 256))
    print(f"\nHeatmap shape: {heatmap.shape}")
    print(f"Heatmap range: [{heatmap.min():.3f}, {heatmap.max():.3f}]")
    
    print("\n✅ Preprocessing pipeline ready!")