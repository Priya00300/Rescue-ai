// src/utils/pathfinding.ts

import { Grid, Cell, Position } from '../types';
import { getCell, getNeighbors, manhattanDistance, isPositionEqual } from './gridUtils';

class PriorityQueue {
  private items: Cell[] = [];
  
  enqueue(cell: Cell): void {
    this.items.push(cell);
    this.items.sort((a, b) => (a.f || 0) - (b.f || 0));
  }
  
  dequeue(): Cell | undefined {
    return this.items.shift();
  }
  
  isEmpty(): boolean {
    return this.items.length === 0;
  }
  
  contains(cell: Cell): boolean {
    return this.items.some(item => 
      isPositionEqual(item.position, cell.position)
    );
  }
}

export const findPath = (
  grid: Grid, 
  start: Position, 
  goal: Position
): Position[] | null => {
  
  const startCell = getCell(grid, start.x, start.y);
  const goalCell = getCell(grid, goal.x, goal.y);
  
  if (!startCell || !goalCell || !startCell.isWalkable || !goalCell.isWalkable) {
    return null;
  }
  
  for (let y = 0; y < grid.height; y++) {
    for (let x = 0; x < grid.width; x++) {
      grid.cells[y][x].g = Infinity;
      grid.cells[y][x].h = 0;
      grid.cells[y][x].f = Infinity;
      grid.cells[y][x].parent = undefined;
    }
  }
  
  startCell.g = 0;
  startCell.h = manhattanDistance(start, goal);
  startCell.f = startCell.h;
  
  const openSet = new PriorityQueue();
  const closedSet: Set<string> = new Set();
  
  openSet.enqueue(startCell);
  
  while (!openSet.isEmpty()) {
    const current = openSet.dequeue()!;
    const currentKey = `${current.position.x},${current.position.y}`;
    
    if (isPositionEqual(current.position, goal)) {
      return reconstructPath(current);
    }
    
    closedSet.add(currentKey);
    
    const neighbors = getNeighbors(grid, current);
    
    for (const neighbor of neighbors) {
      const neighborKey = `${neighbor.position.x},${neighbor.position.y}`;
      
      if (closedSet.has(neighborKey)) continue;
      
      const tentativeG = current.g! + neighbor.cost;
      
      if (tentativeG < neighbor.g!) {
        neighbor.parent = current;
        neighbor.g = tentativeG;
        neighbor.h = manhattanDistance(neighbor.position, goal);
        neighbor.f = neighbor.g + neighbor.h;
        
        if (!openSet.contains(neighbor)) {
          openSet.enqueue(neighbor);
        }
      }
    }
  }
  
  return null;
};

const reconstructPath = (goalCell: Cell): Position[] => {
  const path: Position[] = [];
  let current: Cell | undefined = goalCell;
  
  while (current) {
    path.unshift(current.position);
    current = current.parent;
  }
  
  return path;
};

export const findMultiplePaths = (
  grid: Grid,
  starts: Position[],
  goals: Position[]
): (Position[] | null)[] => {
  const paths: (Position[] | null)[] = [];
  
  for (let i = 0; i < starts.length; i++) {
    const path = findPath(grid, starts[i], goals[i]);
    paths.push(path);
  }
  
  return paths;
};