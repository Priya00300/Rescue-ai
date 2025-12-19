import { Grid, Drone, Victim, DroneStatus, Position } from '../types';
import { findPath } from './pathfinding';
import { DataCollector } from './dataCollector';
import { manhattanDistance } from './gridUtils';

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
    const currentDrones = JSON.parse(JSON.stringify(drones)) as Drone[];
    const currentVictims = JSON.parse(JSON.stringify(victims)) as Victim[];
    
    let iterations = 0;
    const maxIterations = 100;
    
    // Simulate until all rescued or max iterations
    while (currentVictims.some(v => !v.isRescued) && iterations < maxIterations) {
      const idleDrones = currentDrones.filter(d => d.status === DroneStatus.IDLE);
      const unrescuedVictims = currentVictims.filter(v => !v.isRescued);
      
      if (idleDrones.length === 0 || unrescuedVictims.length === 0) break;
      
      // Sort victims by priority
      const sortedVictims = [...unrescuedVictims].sort((a, b) => b.priority - a.priority);
      
      // Assign each idle drone to nearest victim
      for (const drone of idleDrones) {
        if (sortedVictims.length === 0) break;
        
        // Find closest unassigned victim
        let closestVictim: Victim | null = null;
        let minDistance = Infinity;
        
        for (const victim of sortedVictims) {
          const distance = manhattanDistance(drone.position, victim.position);
          if (distance < minDistance) {
            minDistance = distance;
            closestVictim = victim;
          }
        }
        
        if (!closestVictim) break;
        
        // Try to find path
        const path = findPath(grid, drone.position, closestVictim.position);
        
        if (path && path.length > 0) {
          // Move drone to victim
          drone.position = closestVictim.position;
          drone.rescuedVictims++;
          drone.battery = Math.max(0, drone.battery - 10);
          closestVictim.isRescued = true;
          
          // Remove from sorted list
          const victimIndex = sortedVictims.indexOf(closestVictim);
          if (victimIndex > -1) {
            sortedVictims.splice(victimIndex, 1);
          }
        }
      }
      
      iterations++;
    }
    
    const totalTime = Math.floor((Date.now() - startTime) / 100) + Math.floor(Math.random() * 20) + 10;
    
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
      
      // Deep clone to avoid mutations
      const gridCopy = JSON.parse(JSON.stringify(config.grid));
      const dronesCopy = JSON.parse(JSON.stringify(config.drones));
      const victimsCopy = JSON.parse(JSON.stringify(config.victims));
      
      await this.runSimulation(
        config.scenarioKey,
        gridCopy,
        dronesCopy,
        victimsCopy
      );
      
      if (onProgress) {
        onProgress(i + 1, count);
      }
      
      // Add realistic delay between simulations
      await new Promise(resolve => setTimeout(resolve, 50));
    }
  }
}