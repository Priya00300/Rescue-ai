// src/utils/autoSimulator.ts

import { Grid, Drone, Victim, DroneStatus } from '../types';
import { findPath } from './pathfinding';
import { DataCollector } from './dataCollector';

export class AutoSimulator {
  
  static async runSimulation(
    scenarioKey: string,
    grid: Grid,
    drones: Drone[],
    victims: Victim[]
  ): Promise<boolean> {
    
    const missionId = DataCollector.startMission(
      scenarioKey,
      grid,
      victims,
      drones
    );
    
    const startTime = Date.now();
    const currentDrones = [...drones];
    const currentVictims = [...victims];
    
    while (currentVictims.some(v => !v.isRescued)) {
      const idleDrones = currentDrones.filter(d => d.status === DroneStatus.IDLE);
      const unrescuedVictims = currentVictims.filter(v => !v.isRescued);
      
      if (idleDrones.length === 0 || unrescuedVictims.length === 0) break;
      
      const sortedVictims = [...unrescuedVictims].sort((a, b) => b.priority - a.priority);
      
      for (const victim of sortedVictims) {
        const availableDrone = idleDrones.find(d => d.status === DroneStatus.IDLE);
        if (!availableDrone) break;
        
        const path = findPath(grid, availableDrone.position, victim.position);
        
        if (path) {
          availableDrone.status = DroneStatus.MOVING;
          availableDrone.position = victim.position;
          availableDrone.rescuedVictims++;
          victim.isRescued = true;
          availableDrone.status = DroneStatus.IDLE;
        }
      }
    }
    
    const totalTime = Math.floor((Date.now() - startTime) / 10);
    
    DataCollector.completeMission(
      missionId,
      totalTime,
      currentVictims,
      currentDrones
    );
    
    return currentVictims.every(v => v.isRescued);
  }
  
  static async runMultipleSimulations(
    count: number,
    scenarioConfigs: Array<{
      scenarioKey: string;
      grid: Grid;
      drones: Drone[];
      victims: Victim[];
    }>,
    onProgress?: (current: number, total: number) => void
  ): Promise<void> {
    
    for (let i = 0; i < count; i++) {
      const config = scenarioConfigs[i % scenarioConfigs.length];
      
      await this.runSimulation(
        config.scenarioKey,
        config.grid,
        config.drones,
        config.victims
      );
      
      if (onProgress) {
        onProgress(i + 1, count);
      }
      
      await new Promise(resolve => setTimeout(resolve, 10));
    }
  }
}