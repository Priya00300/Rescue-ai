"""
Converts vision model outputs to RescueAI grid format.
This is the bridge between computer vision and pathfinding.
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple
from scipy.ndimage import label, maximum_filter


class VisionToGridConverter:
    """
    Converts segmentation maps to RescueAI grid structure.
    Handles:
    - Fire/flood/collapse → obstacle placement
    - Human presence → victim detection
    - Risk-aware cost assignment
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
        """
        Args:
            grid_width: Target grid width
            grid_height: Target grid height
            fire_threshold: Fire detection threshold
            flood_threshold: Flood detection threshold
            collapse_threshold: Collapse detection threshold
            human_threshold: Human presence threshold
        """
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
        
        Args:
            fire_map: Fire probability map (H, W)
            flood_map: Flood probability map (H, W)
            collapse_map: Collapse probability map (H, W)
            human_map: Human presence map (H, W)
            
        Returns:
            Dictionary with grid, victims, and metadata
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
                cost = 999999  # Use large number instead of infinity
                is_walkable = False
            elif flood_grid[y, x] > self.thresholds['flood']:
                cell_type = 'water'
                cost = 999999
                is_walkable = False
            elif collapse_grid[y, x] > self.thresholds['collapse']:
                cell_type = 'obstacle'
                cost = 999999
                is_walkable = False
            else:
                # Risk-aware cost for safe cells
                risk = (
                    fire_grid[y, x] * 2.0 +
                    flood_grid[y, x] * 1.5 +
                    collapse_grid[y, x] * 1.0
                ) / 4.5
                
                cost = float(1.0 + risk * 4.0)  # Ensure it's a regular float
            
            row.append({
                'position': {'x': int(x), 'y': int(y)},
                'type': cell_type,
                'cost': float(cost),  # Convert to regular float
                'isWalkable': bool(is_walkable)  # Convert to regular bool
            })
        
        cells.append(row)
    
    return cells
    
    def _extract_victims(
        self,
        human_grid: np.ndarray,
        cells: List[List[Dict]]
    ) -> List[Dict]:
        """
        Extract victim positions from human presence map.
        Uses local maxima detection to avoid clustering.
        """
        # Apply threshold
        human_binary = human_grid > self.thresholds['human']
        
        # Find local maxima (victims are peaks in heatmap)
        local_max = maximum_filter(human_grid, size=3) == human_grid
        human_peaks = human_binary & local_max
        
        # Get coordinates
        victim_coords = np.argwhere(human_peaks)
        
        victims = []
        victim_id = 1
        
        for coord in victim_coords:
            y, x = coord
            
            # Check if position is walkable (not in obstacle)
            cell = cells[y][x]
            if not cell['isWalkable']:
                continue
            
            # Calculate priority based on probability
            probability = float(human_grid[y, x])
            priority = self._calculate_victim_priority(probability, x, y, cells)
            
            victims.append({
                'id': f'v{victim_id}',
                'position': {'x': int(x), 'y': int(y)},
                'isRescued': False,
                'priority': priority,
                'detectedAt': 0  # Will be set by frontend
            })
            
            victim_id += 1
        
        # Sort by priority (high to low) and limit
        victims.sort(key=lambda v: v['priority'], reverse=True)
        return victims[:10]  # Max 10 victims
    
    def _calculate_victim_priority(
        self,
        probability: float,
        x: int,
        y: int,
        cells: List[List[Dict]]
    ) -> int:
        """
        Calculate victim priority (1-5) based on:
        - Detection probability
        - Proximity to hazards
        """
        # Base priority from probability
        base_priority = int(probability * 3) + 1  # 1-4
        
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
        """
        Find safe starting positions for drones.
        Places drones in corners or safe zones.
        """
        # Candidate positions (corners + center edges)
        candidates = [
            (1, 1),  # Top-left
            (self.grid_width - 2, 1),  # Top-right
            (1, self.grid_height - 2),  # Bottom-left
            (self.grid_width - 2, self.grid_height - 2),  # Bottom-right
            (self.grid_width // 2, 1),  # Top-center
            (1, self.grid_height // 2),  # Left-center
            (self.grid_width - 2, self.grid_height // 2),  # Right-center
            (self.grid_width // 2, self.grid_height - 2)  # Bottom-center
        ]
        
        # Filter safe positions
        safe_positions = []
        for x, y in candidates:
            if cells[y][x]['isWalkable'] and cells[y][x]['cost'] < 2.0:
                safe_positions.append({'x': x, 'y': y})
        
        # Return requested number or all safe positions
        return safe_positions[:num_drones] if len(safe_positions) >= num_drones else safe_positions
    
    def _determine_scenario_type(
        self,
        fire_grid: np.ndarray,
        flood_grid: np.ndarray,
        collapse_grid: np.ndarray
    ) -> str:
        """
        Determine primary scenario type based on dominant hazard.
        """
        fire_coverage = np.mean(fire_grid > self.thresholds['fire'])
        flood_coverage = np.mean(flood_grid > self.thresholds['flood'])
        collapse_coverage = np.mean(collapse_grid > self.thresholds['collapse'])
        
        if fire_coverage > flood_coverage and fire_coverage > collapse_coverage:
            return 'wildfire'
        elif flood_coverage > collapse_coverage:
            return 'flood'
        else:
            return 'earthquake'
    
    def export_to_json(self, scenario: Dict) -> str:
        """
        Export scenario to JSON string for frontend.
        
        Args:
            scenario: Scenario dictionary from convert()
            
        Returns:
            JSON string
        """
        import json
        
        # Convert numpy types to native Python types
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            return obj
        
        scenario_clean = convert_types(scenario)
        return json.dumps(scenario_clean, indent=2)


class RiskAwarePathCostCalculator:
    """
    Calculates path costs incorporating vision-based risk.
    Used by A* for risk-aware pathfinding.
    """
    
    @staticmethod
    def calculate_cell_cost(
        base_cost: float,
        fire_prob: float,
        flood_prob: float,
        collapse_prob: float
    ) -> float:
        """
        Calculate final cell cost incorporating all risks.
        
        Args:
            base_cost: Base movement cost (1.0 for empty)
            fire_prob: Fire probability (0-1)
            flood_prob: Flood probability (0-1)
            collapse_prob: Collapse probability (0-1)
            
        Returns:
            Final path cost
        """
        if base_cost == float('inf'):
            return float('inf')
        
        # Weight different hazards
        fire_weight = 3.0  # Fire is most dangerous
        flood_weight = 2.0
        collapse_weight = 1.5
        
        risk_cost = (
            fire_prob * fire_weight +
            flood_prob * flood_weight +
            collapse_prob * collapse_weight
        )
        
        # Final cost: base + risk component
        return base_cost * (1.0 + risk_cost)


# Test converter
if __name__ == "__main__":
    print("Testing Vision-to-Grid Converter...")
    
    # Create synthetic maps
    H, W = 256, 256
    
    fire_map = np.random.rand(H, W) * 0.3
    fire_map[50:100, 50:100] = 0.8  # Fire region
    
    flood_map = np.random.rand(H, W) * 0.2
    flood_map[150:200, 150:200] = 0.9  # Flood region
    
    collapse_map = np.random.rand(H, W) * 0.3
    
    human_map = np.zeros((H, W))
    human_map[120, 80] = 0.9  # Victim 1
    human_map[180, 180] = 0.85  # Victim 2
    
    # Convert
    converter = VisionToGridConverter(grid_width=30, grid_height=25)
    scenario = converter.convert(fire_map, flood_map, collapse_map, human_map)
    
    print(f"\n✅ Scenario generated:")
    print(f"  Type: {scenario['scenarioType']}")
    print(f"  Victims: {len(scenario['victims'])}")
    print(f"  Drone positions: {len(scenario['dronePositions'])}")
    print(f"  Metadata: {scenario['metadata']}")
    
    # Export
    json_output = converter.export_to_json(scenario)
    print(f"\n📄 JSON output: {len(json_output)} characters")