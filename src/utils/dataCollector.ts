// src/utils/dataCollector.ts

import { Grid, Victim, Drone } from '../types';

export interface MissionData {
  id: string;
  timestamp: number;
  scenarioType: string;
  
  gridSize: { width: number; height: number };
  obstacleCount: number;
  fireCount: number;
  waterCount: number;
  victimCount: number;
  droneCount: number;
  
  victims: {
    position: { x: number; y: number };
    priority: number;
    rescued: boolean;
    rescueTime?: number;
  }[];
  
  missionSuccess: boolean;
  totalTime: number;
  victimsRescued: number;
  averageResponseTime: number;
  totalDistance: number;
  
  droneStats: {
    id: string;
    pathLength: number;
    victimsRescued: number;
    batteryUsed: number;
  }[];
  
  difficultyScore: number;
  riskLevel: 'low' | 'medium' | 'high';
}

export class DataCollector {
  private static missions: MissionData[] = [];
  
  static startMission(
    scenarioType: string,
    grid: Grid,
    victims: Victim[],
    drones: Drone[]
  ): string {
    const missionId = `mission_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    let obstacleCount = 0;
    let fireCount = 0;
    let waterCount = 0;
    
    for (let y = 0; y < grid.height; y++) {
      for (let x = 0; x < grid.width; x++) {
        const cell = grid.cells[y][x];
        if (!cell.isWalkable) {
          if (cell.cost === Infinity) {
            const type = cell.type;
            if (type === 'fire') fireCount++;
            else if (type === 'water') waterCount++;
            else obstacleCount++;
          }
        }
      }
    }
    
    const difficultyScore = this.calculateDifficulty(
      obstacleCount,
      fireCount,
      waterCount,
      victims.length,
      drones.length
    );
    
    const missionData: MissionData = {
      id: missionId,
      timestamp: Date.now(),
      scenarioType,
      gridSize: { width: grid.width, height: grid.height },
      obstacleCount,
      fireCount,
      waterCount,
      victimCount: victims.length,
      droneCount: drones.length,
      victims: victims.map(v => ({
        position: v.position,
        priority: v.priority,
        rescued: false
      })),
      missionSuccess: false,
      totalTime: 0,
      victimsRescued: 0,
      averageResponseTime: 0,
      totalDistance: 0,
      droneStats: drones.map(d => ({
        id: d.id,
        pathLength: 0,
        victimsRescued: 0,
        batteryUsed: 0
      })),
      difficultyScore,
      riskLevel: difficultyScore < 30 ? 'low' : difficultyScore < 60 ? 'medium' : 'high'
    };
    
    this.missions.push(missionData);
    return missionId;
  }
  
  static updateMission(
    missionId: string,
    updates: Partial<MissionData>
  ): void {
    const mission = this.missions.find(m => m.id === missionId);
    if (mission) {
      Object.assign(mission, updates);
    }
  }
  
  static completeMission(
    missionId: string,
    totalTime: number,
    victims: Victim[],
    drones: Drone[]
  ): void {
    const mission = this.missions.find(m => m.id === missionId);
    if (!mission) return;
    
    const rescuedVictims = victims.filter(v => v.isRescued);
    
    mission.missionSuccess = rescuedVictims.length === victims.length;
    mission.totalTime = totalTime;
    mission.victimsRescued = rescuedVictims.length;
    mission.averageResponseTime = totalTime / Math.max(rescuedVictims.length, 1);
    
    mission.droneStats = drones.map(d => ({
      id: d.id,
      pathLength: d.path.length,
      victimsRescued: d.rescuedVictims,
      batteryUsed: 100 - d.battery
    }));
    
    mission.totalDistance = drones.reduce((sum, d) => sum + d.path.length, 0);
    
    mission.victims = victims.map(v => ({
      position: v.position,
      priority: v.priority,
      rescued: v.isRescued,
      rescueTime: v.isRescued ? totalTime : undefined
    }));
  }
  
  static calculateDifficulty(
    obstacles: number,
    fire: number,
    water: number,
    victims: number,
    drones: number
  ): number {
    const obstacleWeight = obstacles * 0.5;
    const fireWeight = fire * 1.5;
    const waterWeight = water * 1.2;
    const victimWeight = victims * 3;
    const droneDisadvantage = victims > drones ? (victims - drones) * 5 : 0;
    
    const rawScore = obstacleWeight + fireWeight + waterWeight + victimWeight + droneDisadvantage;
    
    return Math.min(100, Math.round(rawScore));
  }
  
  static getMissions(): MissionData[] {
    return this.missions;
  }
  
  static getRecentMissions(count: number = 10): MissionData[] {
    return this.missions.slice(-count);
  }
  
  static getAverageAccuracy(): number {
    if (this.missions.length === 0) return 0;
    const successful = this.missions.filter(m => m.missionSuccess).length;
    return Math.round((successful / this.missions.length) * 100);
  }
  
  static getAverageTime(): number {
    if (this.missions.length === 0) return 0;
    const total = this.missions.reduce((sum, m) => sum + m.totalTime, 0);
    return Math.round(total / this.missions.length);
  }
  
  static exportData(): string {
    return JSON.stringify(this.missions, null, 2);
  }
  
  static importData(jsonData: string): void {
    try {
      const data = JSON.parse(jsonData);
      if (Array.isArray(data)) {
        this.missions = data;
      }
    } catch (e) {
      console.error('Failed to import data:', e);
    }
  }
  
  static clearData(): void {
    this.missions = [];
  }
}