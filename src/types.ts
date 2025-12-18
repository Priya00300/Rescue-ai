// src/types.ts

export interface Position {
  x: number;
  y: number;
}

export enum CellType {
  EMPTY = 'empty',
  OBSTACLE = 'obstacle',
  FIRE = 'fire',
  WATER = 'water',
  VICTIM = 'victim',
  DRONE = 'drone',
  PATH = 'path',
  SAFE_ZONE = 'safe_zone'
}

export interface Cell {
  position: Position;
  type: CellType;
  cost: number;
  isWalkable: boolean;
  g?: number;
  h?: number;
  f?: number;
  parent?: Cell;
}

export interface Grid {
  width: number;
  height: number;
  cellSize: number;
  cells: Cell[][];
}

export enum DroneStatus {
  IDLE = 'idle',
  MOVING = 'moving',
  RESCUING = 'rescuing',
  RETURNING = 'returning',
  CHARGING = 'charging'
}

export interface Drone {
  id: string;
  position: Position;
  targetPosition: Position | null;
  path: Position[];
  status: DroneStatus;
  battery: number;
  speed: number;
  rescuedVictims: number;
  color: string;
}

export interface Victim {
  id: string;
  position: Position;
  isRescued: boolean;
  priority: number;
  detectedAt: number;
}

export interface Obstacle {
  position: Position;
  type: 'building' | 'fire' | 'water' | 'debris';
  radius: number;
}

export enum ScenarioType {
  WILDFIRE = 'wildfire',
  FLOOD = 'flood',
  EARTHQUAKE = 'earthquake',
  SEARCH_RESCUE = 'search_rescue'
}

export interface Scenario {
  type: ScenarioType;
  name: string;
  description: string;
  gridSize: { width: number; height: number };
  drones: number;
  victims: number;
  obstacles: Obstacle[];
}

export interface SimulationStats {
  victimsRescued: number;
  totalVictims: number;
  averageResponseTime: number;
  dronesActive: number;
  pathsCalculated: number;
  totalDistance: number;
  efficiency: number;
}