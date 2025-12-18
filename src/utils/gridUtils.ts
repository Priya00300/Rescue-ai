// src/utils/gridUtils.ts

import { Grid, Cell, CellType, Position } from '../types';

export const createEmptyGrid = (width: number, height: number, cellSize: number = 30): Grid => {
  const cells: Cell[][] = [];
  
  for (let y = 0; y < height; y++) {
    cells[y] = [];
    for (let x = 0; x < width; x++) {
      cells[y][x] = {
        position: { x, y },
        type: CellType.EMPTY,
        cost: 1,
        isWalkable: true
      };
    }
  }
  
  return { width, height, cellSize, cells };
};

export const setCellType = (grid: Grid, x: number, y: number, type: CellType): void => {
  if (x < 0 || x >= grid.width || y < 0 || y >= grid.height) return;
  
  grid.cells[y][x].type = type;
  
  switch (type) {
    case CellType.OBSTACLE:
    case CellType.FIRE:
      grid.cells[y][x].isWalkable = false;
      grid.cells[y][x].cost = Infinity;
      break;
    case CellType.WATER:
      grid.cells[y][x].isWalkable = false;
      grid.cells[y][x].cost = Infinity;
      break;
    case CellType.PATH:
      grid.cells[y][x].isWalkable = true;
      grid.cells[y][x].cost = 5;
      break;
    default:
      grid.cells[y][x].isWalkable = true;
      grid.cells[y][x].cost = 1;
  }
};

export const getCell = (grid: Grid, x: number, y: number): Cell | null => {
  if (x < 0 || x >= grid.width || y < 0 || y >= grid.height) return null;
  return grid.cells[y][x];
};

export const getNeighbors = (grid: Grid, cell: Cell): Cell[] => {
  const { x, y } = cell.position;
  const neighbors: Cell[] = [];
  
  const directions = [
    { dx: 0, dy: -1 },
    { dx: 0, dy: 1 },
    { dx: -1, dy: 0 },
    { dx: 1, dy: 0 }
  ];
  
  for (const { dx, dy } of directions) {
    const neighbor = getCell(grid, x + dx, y + dy);
    if (neighbor && neighbor.isWalkable) {
      neighbors.push(neighbor);
    }
  }
  
  return neighbors;
};

export const manhattanDistance = (a: Position, b: Position): number => {
  return Math.abs(a.x - b.x) + Math.abs(a.y - b.y);
};

export const euclideanDistance = (a: Position, b: Position): number => {
  return Math.sqrt(Math.pow(a.x - b.x, 2) + Math.pow(a.y - b.y, 2));
};

export const isPositionEqual = (a: Position, b: Position): boolean => {
  return a.x === b.x && a.y === b.y;
};

export const getCellColor = (type: CellType): string => {
  const colors: Record<CellType, string> = {
    [CellType.EMPTY]: '#1e293b',
    [CellType.OBSTACLE]: '#475569',
    [CellType.FIRE]: '#ef4444',
    [CellType.WATER]: '#3b82f6',
    [CellType.VICTIM]: '#eab308',
    [CellType.DRONE]: '#10b981',
    [CellType.PATH]: '#8b5cf6',
    [CellType.SAFE_ZONE]: '#22c55e'
  };
  return colors[type];
};

export const clearPath = (grid: Grid): void => {
  for (let y = 0; y < grid.height; y++) {
    for (let x = 0; x < grid.width; x++) {
      if (grid.cells[y][x].type === CellType.PATH) {
        grid.cells[y][x].type = CellType.EMPTY;
        grid.cells[y][x].cost = 1;
      }
    }
  }
};