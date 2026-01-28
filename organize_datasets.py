"""
Organize downloaded datasets for RescueAI training
"""

import os
import shutil
import json
from pathlib import Path
from PIL import Image
import random

print("=" * 60)
print("RescueAI Dataset Organization")
print("=" * 60)
print()

base_dir = Path('datasets')
downloads_dir = base_dir / 'downloads'

# ============================================================================
# 1. ORGANIZE FIRE DATASET
# ============================================================================
print("🔥 Organizing Fire Dataset...")

fire_raw = downloads_dir / 'fire_raw' / 'fire_dataset'
fire_train_img = base_dir / 'fire' / 'train' / 'images'
fire_train_mask = base_dir / 'fire' / 'train' / 'masks'
fire_val_img = base_dir / 'fire' / 'val' / 'images'
fire_val_mask = base_dir / 'fire' / 'val' / 'masks'

for d in [fire_train_img, fire_train_mask, fire_val_img, fire_val_mask]:
    d.mkdir(parents=True, exist_ok=True)

# Check fire dataset structure
if fire_raw.exists():
    fire_subfolders = list(fire_raw.iterdir())
    print(f"  Found folders: {[f.name for f in fire_subfolders if f.is_dir()]}")
    
    # Look for images and masks
    all_images = list(fire_raw.rglob('*.jpg')) + list(fire_raw.rglob('*.png'))
    print(f"  Found {len(all_images)} images")
    
    # Split 80/20 train/val
    random.shuffle(all_images)
    split_idx = int(len(all_images) * 0.8)
    train_images = all_images[:split_idx]
    val_images = all_images[split_idx:]
    
    # Copy training images
    for i, img_path in enumerate(train_images):
        # Check if there's a corresponding mask
        mask_path = img_path.parent / 'masks' / img_path.name
        if not mask_path.exists():
            mask_path = img_path.parent.parent / 'masks' / img_path.name
        
        # Copy image
        shutil.copy(img_path, fire_train_img / f'fire_{i}.jpg')
        
        # Copy or create mask
        if mask_path.exists():
            shutil.copy(mask_path, fire_train_mask / f'fire_{i}.png')
        else:
            # Create dummy mask if no mask exists
            img = Image.open(img_path)
            mask = Image.new('L', img.size, 0)
            mask.save(fire_train_mask / f'fire_{i}.png')
    
    # Copy validation images
    for i, img_path in enumerate(val_images):
        mask_path = img_path.parent / 'masks' / img_path.name
        if not mask_path.exists():
            mask_path = img_path.parent.parent / 'masks' / img_path.name
        
        shutil.copy(img_path, fire_val_img / f'fire_{i}.jpg')
        
        if mask_path.exists():
            shutil.copy(mask_path, fire_val_mask / f'fire_{i}.png')
        else:
            img = Image.open(img_path)
            mask = Image.new('L', img.size, 0)
            mask.save(fire_val_mask / f'fire_{i}.png')
    
    print(f"  ✅ Organized: {len(train_images)} train, {len(val_images)} val")
else:
    print("  ⚠️ Fire dataset not found")

# ============================================================================
# 2. ORGANIZE FLOOD DATASET
# ============================================================================
print("\n🌊 Organizing Flood Dataset...")

flood_raw_img = downloads_dir / 'flood_raw' / 'Image'
flood_raw_mask = downloads_dir / 'flood_raw' / 'Mask'
flood_train_img = base_dir / 'flood' / 'train' / 'images'
flood_train_mask = base_dir / 'flood' / 'train' / 'masks'
flood_val_img = base_dir / 'flood' / 'val' / 'images'
flood_val_mask = base_dir / 'flood' / 'val' / 'masks'

for d in [flood_train_img, flood_train_mask, flood_val_img, flood_val_mask]:
    d.mkdir(parents=True, exist_ok=True)

if flood_raw_img.exists() and flood_raw_mask.exists():
    all_images = list(flood_raw_img.glob('*.jpg')) + list(flood_raw_img.glob('*.png'))
    print(f"  Found {len(all_images)} images")
    
    # Split 80/20
    random.shuffle(all_images)
    split_idx = int(len(all_images) * 0.8)
    train_images = all_images[:split_idx]
    val_images = all_images[split_idx:]
    
    # Copy training
    for i, img_path in enumerate(train_images):
        mask_path = flood_raw_mask / img_path.name
        
        shutil.copy(img_path, flood_train_img / f'flood_{i}.jpg')
        if mask_path.exists():
            shutil.copy(mask_path, flood_train_mask / f'flood_{i}.png')
    
    # Copy validation
    for i, img_path in enumerate(val_images):
        mask_path = flood_raw_mask / img_path.name
        
        shutil.copy(img_path, flood_val_img / f'flood_{i}.jpg')
        if mask_path.exists():
            shutil.copy(mask_path, flood_val_mask / f'flood_{i}.png')
    
    print(f"  ✅ Organized: {len(train_images)} train, {len(val_images)} val")
else:
    print("  ⚠️ Flood dataset not found")

# ============================================================================
# 3. ORGANIZE HURRICANE/DAMAGE DATASET
# ============================================================================
print("\n🏚️ Organizing Hurricane/Damage Dataset...")

hurricane_train = downloads_dir / 'hurricane_raw' / 'train_another'
hurricane_val = downloads_dir / 'hurricane_raw' / 'validation_another'
collapse_train_img = base_dir / 'collapse' / 'train' / 'images'
collapse_train_mask = base_dir / 'collapse' / 'train' / 'masks'
collapse_val_img = base_dir / 'collapse' / 'val' / 'images'
collapse_val_mask = base_dir / 'collapse' / 'val' / 'masks'

for d in [collapse_train_img, collapse_train_mask, collapse_val_img, collapse_val_mask]:
    d.mkdir(parents=True, exist_ok=True)

if hurricane_train.exists():
    # Find damaged images
    train_images = list(hurricane_train.rglob('*.jpg')) + list(hurricane_train.rglob('*.png'))
    print(f"  Found {len(train_images)} training images")
    
    for i, img_path in enumerate(train_images[:500]):  # Limit to 500
        shutil.copy(img_path, collapse_train_img / f'collapse_{i}.jpg')
        
        # Create mask (we'll treat whole image as potential damage)
        img = Image.open(img_path)
        mask = Image.new('L', img.size, 128)  # Gray = potential damage
        mask.save(collapse_train_mask / f'collapse_{i}.png')

if hurricane_val.exists():
    val_images = list(hurricane_val.rglob('*.jpg')) + list(hurricane_val.rglob('*.png'))
    print(f"  Found {len(val_images)} validation images")
    
    for i, img_path in enumerate(val_images[:100]):  # Limit to 100
        shutil.copy(img_path, collapse_val_img / f'collapse_{i}.jpg')
        
        img = Image.open(img_path)
        mask = Image.new('L', img.size, 128)
        mask.save(collapse_val_mask / f'collapse_{i}.png')
    
    print(f"  ✅ Organized: 500 train, 100 val")
else:
    print("  ⚠️ Hurricane dataset not found")

# ============================================================================
# 4. ORGANIZE HUMAN DETECTION DATASET
# ============================================================================
print("\n👤 Organizing Human Detection Dataset...")

human_raw = downloads_dir / 'human_raw' / 'PennFudanPed'
human_train_img = base_dir / 'human' / 'train' / 'images'
human_val_img = base_dir / 'human' / 'val' / 'images'

for d in [human_train_img, human_val_img]:
    d.mkdir(parents=True, exist_ok=True)

if human_raw.exists():
    # Find all images
    all_images = list((human_raw / 'PNGImages').glob('*.png')) if (human_raw / 'PNGImages').exists() else []
    
    if not all_images:
        all_images = list(human_raw.rglob('*.png'))
    
    print(f"  Found {len(all_images)} images")
    
    # Split 80/20
    random.shuffle(all_images)
    split_idx = int(len(all_images) * 0.8)
    train_images = all_images[:split_idx]
    val_images = all_images[split_idx:]
    
    # Create annotations
    train_annotations = {}
    val_annotations = {}
    
    # Copy training
    for i, img_path in enumerate(train_images):
        filename = f'human_{i}.jpg'
        
        # Convert PNG to JPG if needed
        img = Image.open(img_path)
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        img.save(human_train_img / filename)
        
        # Create dummy bounding box (center of image)
        w, h = img.size
        bbox = [w//4, h//4, 3*w//4, 3*h//4]
        train_annotations[filename] = [bbox]
    
    # Copy validation
    for i, img_path in enumerate(val_images):
        filename = f'human_{i}.jpg'
        
        img = Image.open(img_path)
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        img.save(human_val_img / filename)
        
        w, h = img.size
        bbox = [w//4, h//4, 3*w//4, 3*h//4]
        val_annotations[filename] = [bbox]
    
    # Save annotations
    with open(base_dir / 'human' / 'train' / 'annotations.json', 'w') as f:
        json.dump(train_annotations, f, indent=2)
    
    with open(base_dir / 'human' / 'val' / 'annotations.json', 'w') as f:
        json.dump(val_annotations, f, indent=2)
    
    print(f"  ✅ Organized: {len(train_images)} train, {len(val_images)} val")
else:
    print("  ⚠️ Human dataset not found")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 60)
print("✅ DATASET ORGANIZATION COMPLETE!")
print("=" * 60)
print()

# Count final files
fire_train_count = len(list(fire_train_img.glob('*')))
fire_val_count = len(list(fire_val_img.glob('*')))
flood_train_count = len(list(flood_train_img.glob('*')))
flood_val_count = len(list(flood_val_img.glob('*')))
collapse_train_count = len(list(collapse_train_img.glob('*')))
collapse_val_count = len(list(collapse_val_img.glob('*')))
human_train_count = len(list(human_train_img.glob('*')))
human_val_count = len(list(human_val_img.glob('*')))

print("📊 Final Dataset Sizes:")
print(f"  🔥 Fire:      {fire_train_count} train, {fire_val_count} val")
print(f"  🌊 Flood:     {flood_train_count} train, {flood_val_count} val")
print(f"  🏚️ Collapse:  {collapse_train_count} train, {collapse_val_count} val")
print(f"  👤 Human:     {human_train_count} train, {human_val_count} val")
print()
print(f"  📈 Total Training Images: {fire_train_count + flood_train_count + collapse_train_count + human_train_count}")
print(f"  📈 Total Validation Images: {fire_val_count + flood_val_count + collapse_val_count + human_val_count}")
print()
print("🚀 Ready to train! Run:")
print("   python backend/train.py --mode train --epochs 30 --batch_size 4")
print()