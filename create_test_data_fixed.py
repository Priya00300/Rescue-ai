import cv2
import numpy as np
import json
from pathlib import Path
import os

print('Creating test dataset...')

# Create images for each split
for split in ['train', 'val']:
    print(f'Creating {split} data...')
    
    # Fire dataset
    for i in range(5):
        fire_img_path = f'datasets/fire/{split}/images/fire_{i}.jpg'
        fire_mask_path = f'datasets/fire/{split}/masks/fire_{i}.png'
        
        img = np.random.randint(100, 255, (256, 256, 3), dtype=np.uint8)
        img[:, :, 2] = np.random.randint(200, 255, (256, 256), dtype=np.uint8)
        cv2.imwrite(fire_img_path, img)
        
        mask = np.zeros((256, 256), dtype=np.uint8)
        mask[50:150, 50:150] = 255
        cv2.imwrite(fire_mask_path, mask)
    
    # Flood dataset
    for i in range(5):
        flood_img_path = f'datasets/flood/{split}/images/flood_{i}.jpg'
        flood_mask_path = f'datasets/flood/{split}/masks/flood_{i}.png'
        
        img = np.random.randint(100, 255, (256, 256, 3), dtype=np.uint8)
        img[:, :, 0] = np.random.randint(200, 255, (256, 256), dtype=np.uint8)
        cv2.imwrite(flood_img_path, img)
        
        mask = np.zeros((256, 256), dtype=np.uint8)
        mask[100:200, 100:200] = 255
        cv2.imwrite(flood_mask_path, mask)
    
    # Collapse dataset
    for i in range(5):
        collapse_img_path = f'datasets/collapse/{split}/images/collapse_{i}.jpg'
        collapse_mask_path = f'datasets/collapse/{split}/masks/collapse_{i}.png'
        
        img = np.random.randint(50, 150, (256, 256, 3), dtype=np.uint8)
        cv2.imwrite(collapse_img_path, img)
        
        mask = np.zeros((256, 256), dtype=np.uint8)
        mask[80:120, 80:120] = 255
        cv2.imwrite(collapse_mask_path, mask)
    
    # Human dataset
    annotations = {}
    for i in range(5):
        human_img_path = f'datasets/human/{split}/images/human_{i}.jpg'
        
        img = np.random.randint(100, 200, (256, 256, 3), dtype=np.uint8)
        cv2.imwrite(human_img_path, img)
        
        annotations[f'human_{i}.jpg'] = [[50, 50, 100, 100], [150, 150, 200, 200]]
    
    # Save annotations
    anno_path = f'datasets/human/{split}/annotations.json'
    with open(anno_path, 'w') as f:
        json.dump(annotations, f, indent=2)
    
    print(f'   {split} data created')

print('')
print(' Test dataset created successfully!')
print('   - Fire: 10 images (5 train + 5 val)')
print('   - Flood: 10 images (5 train + 5 val)')
print('   - Collapse: 10 images (5 train + 5 val)')
print('   - Human: 10 images (5 train + 5 val)')
