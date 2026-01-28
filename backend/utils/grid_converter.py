"""
Converts vision model outputs to RescueAI grid format.
This is the bridge between computer vision and pathfinding.
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple


class VisionToGridConverter:
    """
    Converts segmentation maps to RescueAI grid structure.
    """
    
    def __init__(
        self,
        grid_width: int = 30,
        grid_height: int = 25,
        fire_threshold: float = 0.5,
        flood_threshold: float = 0.5,
        collapse_threshold: float = 0.5,
        human_threshold: float = 0.6
    ):
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.thresholds = {
            'fire': fire_threshold,
            'flood': flood_threshold,
            'collapse': collapse_threshold,
            'human': human_threshold
        }
    
    def convert(
        self,
        fire_map: np.ndarray,
        flood_map: np.ndarray,
        collapse_map: np.ndarray,
        human_map: np.ndarray
    ) -> Dict:
        """
        Convert vision maps to complete RescueAI scenario.
        """
        # Resize all maps to grid size
        fire_grid = self._resize_map(fire_map)
        flood_grid = self._resize_map(flood_map)
        collapse_grid = self._resize_map(collapse_map)
        human_grid = self._resize_map(human_map)
        
        # Create grid cells
        cells = self._create_cells(fire_grid, flood_grid, collapse_grid)
        
        # Extract victims
        victims = self._extract_victims(human_grid, cells)
        
        # Calculate safe drone positions
        drone_positions = self._find_safe_drone_positions(cells, num_drones=4)
        
        # Determine scenario type
        scenario_type = self._determine_scenario_type(fire_grid, flood_grid, collapse_grid)
        
        return {
            'grid': {
                'width': self.grid_width,
                'height': self.grid_height,
                'cellSize': 25,
                'cells': cells
            },
            'victims': victims,
            'dronePositions': drone_positions,
            'scenarioType': scenario_type,
            'confidence': 0.85,
            'metadata': {
                'fire_coverage': float(np.mean(fire_grid > self.thresholds['fire'])),
                'flood_coverage': float(np.mean(flood_grid > self.thresholds['flood'])),
                'collapse_coverage': float(np.mean(collapse_grid > self.thresholds['collapse'])),
                'victim_count': len(victims)
            }
        }
    
    def _resize_map(self, map_array: np.ndarray) -> np.ndarray:
        """Resize map to grid dimensions."""
        return cv2.resize(
            map_array,
            (self.grid_width, self.grid_height),
            interpolation=cv2.INTER_LINEAR
        )
    
    def _create_cells(
        self,
        fire_grid: np.ndarray,
        flood_grid: np.ndarray,
        collapse_grid: np.ndarray
    ) -> List[List[Dict]]:
        """Create grid cells with obstacle types."""
        cells = []
        
        for y in range(self.grid_height):
            row = []
            for x in range(self.grid_width):
                cell_type = 'empty'
                cost = 1.0
                is_walkable = True
                
                # Determine cell type (priority order)
                if fire_grid[y, x] > self.thresholds['fire']:
                    cell_type = 'fire'
                    cost = float('inf')
                    is_walkable = False
                elif flood_grid[y, x] > self.thresholds['flood']:
                    cell_type = 'water'
                    cost = float('inf')
                    is_walkable = False
                elif collapse_grid[y, x] > self.thresholds['collapse']:
                    cell_type = 'obstacle'
                    cost = float('inf')
                    is_walkable = False
                else:
                    # Risk-aware cost for safe cells
                    risk = (
                        fire_grid[y, x] * 2.0 +
                        flood_grid[y, x] * 1.5 +
                        collapse_grid[y, x] * 1.0
                    ) / 4.5
                    
                    cost = 1.0 + risk * 4.0
                
                row.append({
                    'position': {'x': x, 'y': y},
                    'type': cell_type,
                    'cost': cost,
                    'isWalkable': is_walkable
                })
            
            cells.append(row)
        
        return cells
    
    def _extract_victims(
        self,
        human_grid: np.ndarray,
        cells: List[List[Dict]]
    ) -> List[Dict]:
        """Extract victim positions from human presence map."""
        victim_positions = []
        
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                if human_grid[y, x] > self.thresholds['human']:
                    # Check if position is walkable
                    cell = cells[y][x]
                    if not cell['isWalkable']:
                        continue
                    
                    # Check if this is a local maximum
                    is_local_max = True
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            if dy == 0 and dx == 0:
                                continue
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < self.grid_height and 0 <= nx < self.grid_width:
                                if human_grid[ny, nx] > human_grid[y, x]:
                                    is_local_max = False
                                    break
                        if not is_local_max:
                            break
                    
                    if is_local_max:
                        probability = float(human_grid[y, x])
                        priority = self._calculate_victim_priority(probability, x, y, cells)
                        
                        victim_positions.append({
                            'id': f'v{len(victim_positions) + 1}',
                            'position': {'x': int(x), 'y': int(y)},
                            'isRescued': False,
                            'priority': priority,
                            'detectedAt': 0
                        })
        
        # Sort by priority and limit to 10
        victim_positions.sort(key=lambda v: v['priority'], reverse=True)
        return victim_positions[:10]
    
    def _calculate_victim_priority(
        self,
        probability: float,
        x: int,
        y: int,
        cells: List[List[Dict]]
    ) -> int:
        """Calculate victim priority (1-5)."""
        base_priority = int(probability * 3) + 1
        
        # Check surrounding hazards
        hazard_bonus = 0
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < self.grid_height and 0 <= nx < self.grid_width:
                    cell = cells[ny][nx]
                    if cell['type'] in ['fire', 'water']:
                        hazard_bonus = 1
                        break
        
        priority = min(5, base_priority + hazard_bonus)
        return priority
    
    def _find_safe_drone_positions(
        self,
        cells: List[List[Dict]],
        num_drones: int = 4
    ) -> List[Dict]:
        """Find safe starting positions for drones."""
        candidates = [
            (1, 1),
            (self.grid_width - 2, 1),
            (1, self.grid_height - 2),
            (self.grid_width - 2, self.grid_height - 2),
            (self.grid_width // 2, 1),
            (1, self.grid_height // 2),
            (self.grid_width - 2, self.grid_height // 2),
            (self.grid_width // 2, self.grid_height - 2)
        ]
        
        safe_positions = []
        for x, y in candidates:
            if cells[y][x]['isWalkable'] and cells[y][x]['cost'] < 2.0:
                safe_positions.append({'x': x, 'y': y})
        
        return safe_positions[:num_drones]
    
    def _determine_scenario_type(
        self,
        fire_grid: np.ndarray,
        flood_grid: np.ndarray,
        collapse_grid: np.ndarray
    ) -> str:
        """Determine primary scenario type."""
        fire_coverage = np.mean(fire_grid > self.thresholds['fire'])
        flood_coverage = np.mean(flood_grid > self.thresholds['flood'])
        collapse_coverage = np.mean(collapse_grid > self.thresholds['collapse'])
        
        if fire_coverage > flood_coverage and fire_coverage > collapse_coverage:
            return 'wildfire'
        elif flood_coverage > collapse_coverage:
            return 'flood'
        else:
            return 'earthquake'