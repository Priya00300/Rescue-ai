import cv2
import numpy as np
import json
from pathlib import Path

print('Creating test dataset...')

for split in ['train', 'val']:
    for i in range(5):
        # Fire
        fire_img = Path(f'datasets/fire/{split}/images/fire_{i}.jpg')
        fire_mask = Path(f'datasets/fire/{split}/masks/fire_{i}.png')
        img = np.random.randint(100, 255, (256, 256, 3), dtype=np.uint8)
        img[:, :, 2] = np.random.randint(200, 255, (256, 256))
        cv2.imwrite(str(fire_img), img)
        mask = np.zeros((256, 256), dtype=np.uint8)
        mask[50:150, 50:150] = 255
        cv2.imwrite(str(fire_mask), mask)
        
        # Flood
        flood_img = Path(f'datasets/flood/{split}/images/flood_{i}.jpg')
        flood_mask = Path(f'datasets/flood/{split}/masks/flood_{i}.png')
        img = np.random.randint(100, 255, (256, 256, 3), dtype=np.uint8)
        img[:, :, 0] = np.random.randint(200, 255, (256, 256))
        cv2.imwrite(str(flood_img), img)
        mask = np.zeros((256, 256), dtype=np.uint8)
        mask[100:200, 100:200] = 255
        cv2.imwrite(str(flood_mask), mask)
        
        # Collapse
        collapse_img = Path(f'datasets/collapse/{split}/images/collapse_{i}.jpg')
        collapse_mask = Path(f'datasets/collapse/{split}/masks/collapse_{i}.png')
        img = np.random.randint(50, 150, (256, 256, 3), dtype=np.uint8)
        cv2.imwrite(str(collapse_img), img)
        mask = np.zeros((256, 256), dtype=np.uint8)
        mask[80:120, 80:120] = 255
        cv2.imwrite(str(collapse_mask), mask)

# Human dataset
for split in ['train', 'val']:
    annotations = {}
    for i in range(5):
        human_img = Path(f'datasets/human/{split}/images/human_{i}.jpg')
        img = np.random.randint(100, 200, (256, 256, 3), dtype=np.uint8)
        cv2.imwrite(str(human_img), img)
        annotations[f'human_{i}.jpg'] = [[50, 50, 100, 100], [150, 150, 200, 200]]
    
    with open(f'datasets/human/{split}/annotations.json', 'w') as f:
        json.dump(annotations, f, indent=2)

print('✅ Test dataset created!')
