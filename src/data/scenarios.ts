// src/data/scenarios.ts

import { Victim, Position } from '../types';

export interface ScenarioConfig {
  name: string;
  description: string;
  icon: string;
  gridSize: { width: number; height: number };
  obstacles: { positions: Position[]; type: 'fire' | 'water' | 'obstacle' }[];
  victims: Victim[];
  dronePositions: Position[];
  difficulty: 'Easy' | 'Medium' | 'Hard';
}

export const scenarios: Record<string, ScenarioConfig> = {
  wildfire: {
    name: 'California Wildfire',
    description: 'Rapidly spreading wildfire threatening residential areas. Multiple civilians trapped.',
    icon: '🔥',
    gridSize: { width: 30, height: 25 },
    difficulty: 'Hard',
    dronePositions: [
      { x: 2, y: 2 },
      { x: 27, y: 2 },
      { x: 2, y: 22 },
      { x: 27, y: 22 }
    ],
    obstacles: [
      {
        type: 'fire',
        positions: [
          // Vertical fire wall
          { x: 14, y: 10 }, { x: 14, y: 11 }, { x: 14, y: 12 }, { x: 14, y: 13 }, { x: 14, y: 14 }, { x: 14, y: 15 }, { x: 14, y: 16 }, { x: 14, y: 17 },
          { x: 15, y: 10 }, { x: 15, y: 11 }, { x: 15, y: 12 }, { x: 15, y: 13 }, { x: 15, y: 14 }, { x: 15, y: 15 }, { x: 15, y: 16 }, { x: 15, y: 17 },
          // Spreading fire patches
          { x: 8, y: 8 }, { x: 9, y: 8 }, { x: 8, y: 9 }, { x: 9, y: 9 },
          { x: 22, y: 18 }, { x: 23, y: 18 }, { x: 22, y: 19 }, { x: 23, y: 19 },
        ]
      },
      {
        type: 'obstacle',
        positions: [
          // Buildings
          { x: 8, y: 13 }, { x: 9, y: 13 }, { x: 10, y: 13 }, { x: 11, y: 13 }, { x: 12, y: 13 },
          { x: 18, y: 10 }, { x: 19, y: 10 }, { x: 20, y: 10 }, { x: 21, y: 10 }, { x: 22, y: 10 },
        ]
      }
    ],
    victims: [
      { id: 'v1', position: { x: 15, y: 5 }, isRescued: false, priority: 5, detectedAt: Date.now() },
      { id: 'v2', position: { x: 8, y: 12 }, isRescued: false, priority: 4, detectedAt: Date.now() },
      { id: 'v3', position: { x: 22, y: 15 }, isRescued: false, priority: 5, detectedAt: Date.now() },
      { id: 'v4', position: { x: 12, y: 20 }, isRescued: false, priority: 3, detectedAt: Date.now() },
      { id: 'v5', position: { x: 25, y: 8 }, isRescued: false, priority: 4, detectedAt: Date.now() },
      { id: 'v6', position: { x: 5, y: 18 }, isRescued: false, priority: 2, detectedAt: Date.now() },
    ]
  },

  flood: {
    name: 'Urban Flood Emergency',
    description: 'Flash flooding in city center. Roads submerged, people stranded on rooftops.',
    icon: '🌊',
    gridSize: { width: 30, height: 25 },
    difficulty: 'Medium',
    dronePositions: [
      { x: 1, y: 1 },
      { x: 28, y: 1 },
      { x: 1, y: 23 },
      { x: 28, y: 23 }
    ],
    obstacles: [
      {
        type: 'water',
        positions: [
          // Large flooded area 1
          { x: 6, y: 8 }, { x: 7, y: 8 }, { x: 8, y: 8 }, { x: 9, y: 8 }, { x: 10, y: 8 },
          { x: 6, y: 9 }, { x: 7, y: 9 }, { x: 8, y: 9 }, { x: 9, y: 9 }, { x: 10, y: 9 },
          { x: 6, y: 10 }, { x: 7, y: 10 }, { x: 8, y: 10 }, { x: 9, y: 10 }, { x: 10, y: 10 },
          { x: 6, y: 11 }, { x: 7, y: 11 }, { x: 8, y: 11 }, { x: 9, y: 11 }, { x: 10, y: 11 },
          { x: 6, y: 12 }, { x: 7, y: 12 }, { x: 8, y: 12 }, { x: 9, y: 12 }, { x: 10, y: 12 },
          
          // Large flooded area 2
          { x: 18, y: 14 }, { x: 19, y: 14 }, { x: 20, y: 14 }, { x: 21, y: 14 }, { x: 22, y: 14 },
          { x: 18, y: 15 }, { x: 19, y: 15 }, { x: 20, y: 15 }, { x: 21, y: 15 }, { x: 22, y: 15 },
          { x: 18, y: 16 }, { x: 19, y: 16 }, { x: 20, y: 16 }, { x: 21, y: 16 }, { x: 22, y: 16 },
          { x: 18, y: 17 }, { x: 19, y: 17 }, { x: 20, y: 17 }, { x: 21, y: 17 }, { x: 22, y: 17 },
        ]
      },
      {
        type: 'obstacle',
        positions: [
          // Buildings (safe zones/rooftops)
          { x: 12, y: 10 }, { x: 13, y: 10 }, { x: 12, y: 11 }, { x: 13, y: 11 },
          { x: 24, y: 12 }, { x: 25, y: 12 }, { x: 24, y: 13 }, { x: 25, y: 13 },
        ]
      }
    ],
    victims: [
      { id: 'v1', position: { x: 5, y: 7 }, isRescued: false, priority: 5, detectedAt: Date.now() },
      { id: 'v2', position: { x: 11, y: 12 }, isRescued: false, priority: 4, detectedAt: Date.now() },
      { id: 'v3', position: { x: 17, y: 15 }, isRescued: false, priority: 5, detectedAt: Date.now() },
      { id: 'v4', position: { x: 23, y: 18 }, isRescued: false, priority: 3, detectedAt: Date.now() },
      { id: 'v5', position: { x: 14, y: 6 }, isRescued: false, priority: 4, detectedAt: Date.now() },
    ]
  },

  earthquake: {
    name: 'Earthquake Aftermath',
    description: 'Major earthquake collapsed buildings. Search and rescue in unstable areas.',
    icon: '🏚',
    gridSize: { width: 30, height: 25 },
    difficulty: 'Easy',
    dronePositions: [
      { x: 2, y: 2 },
      { x: 27, y: 2 },
      { x: 2, y: 22 },
      { x: 27, y: 22 }
    ],
    obstacles: [
      {
        type: 'obstacle',
        positions: [
          // Collapsed building 1
          { x: 8, y: 8 }, { x: 9, y: 8 }, { x: 10, y: 8 },
          { x: 8, y: 9 }, { x: 9, y: 9 }, { x: 10, y: 9 },
          { x: 8, y: 10 }, { x: 9, y: 10 }, { x: 10, y: 10 },
          
          // Collapsed building 2
          { x: 18, y: 12 }, { x: 19, y: 12 }, { x: 20, y: 12 }, { x: 21, y: 12 },
          { x: 18, y: 13 }, { x: 19, y: 13 }, { x: 20, y: 13 }, { x: 21, y: 13 },
          { x: 18, y: 14 }, { x: 19, y: 14 }, { x: 20, y: 14 }, { x: 21, y: 14 },
          
          // Debris scattered
          { x: 5, y: 15 }, { x: 6, y: 15 },
          { x: 14, y: 18 }, { x: 15, y: 18 },
          { x: 24, y: 9 }, { x: 25, y: 9 },
        ]
      }
    ],
    victims: [
      { id: 'v1', position: { x: 7, y: 7 }, isRescued: false, priority: 5, detectedAt: Date.now() },
      { id: 'v2', position: { x: 11, y: 9 }, isRescued: false, priority: 4, detectedAt: Date.now() },
      { id: 'v3', position: { x: 22, y: 15 }, isRescued: false, priority: 5, detectedAt: Date.now() },
      { id: 'v4', position: { x: 16, y: 19 }, isRescued: false, priority: 3, detectedAt: Date.now() },
    ]
  }
};