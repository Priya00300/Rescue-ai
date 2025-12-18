// src/utils/swarmCommunication.ts

import { Position, Victim } from '../types';

export interface SwarmMessage {
  id: string;
  senderId: string;
  timestamp: number;
  type: 'victim_found' | 'obstacle_detected' | 'path_blocked' | 'rescue_complete' | 'assistance_needed';
  data: any;
  priority: number;
}

export interface SharedKnowledge {
  discoveredVictims: Map<string, { position: Position; priority: number; discoveredBy: string; timestamp: number }>;
  discoveredObstacles: Map<string, { position: Position; type: string; discoveredBy: string; timestamp: number }>;
  completedRescues: Set<string>;
  activeAssignments: Map<string, string>;
}

export class SwarmCommunication {
  private static knowledge: SharedKnowledge = {
    discoveredVictims: new Map(),
    discoveredObstacles: new Map(),
    completedRescues: new Set(),
    activeAssignments: new Map()
  };
  
  private static messageQueue: SwarmMessage[] = [];
  private static messageHistory: SwarmMessage[] = [];
  
  static reset(): void {
    this.knowledge = {
      discoveredVictims: new Map(),
      discoveredObstacles: new Map(),
      completedRescues: new Set(),
      activeAssignments: new Map()
    };
    this.messageQueue = [];
    this.messageHistory = [];
  }
  
  static broadcastVictimFound(
    droneId: string,
    victim: Victim
  ): void {
    const message: SwarmMessage = {
      id: `msg_${Date.now()}_${Math.random()}`,
      senderId: droneId,
      timestamp: Date.now(),
      type: 'victim_found',
      data: { victim },
      priority: victim.priority
    };
    
    this.messageQueue.push(message);
    this.messageHistory.push(message);
    
    const key = `${victim.position.x},${victim.position.y}`;
    this.knowledge.discoveredVictims.set(key, {
      position: victim.position,
      priority: victim.priority,
      discoveredBy: droneId,
      timestamp: Date.now()
    });
  }
  
  static broadcastObstacleDetected(
    droneId: string,
    position: Position,
    type: string
  ): void {
    const message: SwarmMessage = {
      id: `msg_${Date.now()}_${Math.random()}`,
      senderId: droneId,
      timestamp: Date.now(),
      type: 'obstacle_detected',
      data: { position, type },
      priority: 3
    };
    
    this.messageQueue.push(message);
    this.messageHistory.push(message);
    
    const key = `${position.x},${position.y}`;
    this.knowledge.discoveredObstacles.set(key, {
      position,
      type,
      discoveredBy: droneId,
      timestamp: Date.now()
    });
  }
  
  static broadcastRescueComplete(
    droneId: string,
    victimId: string
  ): void {
    const message: SwarmMessage = {
      id: `msg_${Date.now()}_${Math.random()}`,
      senderId: droneId,
      timestamp: Date.now(),
      type: 'rescue_complete',
      data: { victimId },
      priority: 5
    };
    
    this.messageQueue.push(message);
    this.messageHistory.push(message);
    
    this.knowledge.completedRescues.add(victimId);
    this.knowledge.activeAssignments.delete(droneId);
  }
  
  static assignVictim(droneId: string, victimId: string): void {
    this.knowledge.activeAssignments.set(droneId, victimId);
  }
  
  static isVictimAssigned(victimId: string): boolean {
    return Array.from(this.knowledge.activeAssignments.values()).includes(victimId);
  }
  
  static getUnassignedVictims(allVictims: Victim[]): Victim[] {
    return allVictims.filter(v => 
      !v.isRescued && 
      !this.isVictimAssigned(v.id) &&
      !this.knowledge.completedRescues.has(v.id)
    );
  }
  
  static getMessages(count: number = 5): SwarmMessage[] {
    return this.messageHistory.slice(-count).reverse();
  }
  
  static getKnowledge(): SharedKnowledge {
    return this.knowledge;
  }
  
  static getSwarmStats() {
    return {
      victimsDiscovered: this.knowledge.discoveredVictims.size,
      obstaclesDetected: this.knowledge.discoveredObstacles.size,
      rescuesCompleted: this.knowledge.completedRescues.size,
      activeAssignments: this.knowledge.activeAssignments.size,
      messagesSent: this.messageHistory.length
    };
  }
}