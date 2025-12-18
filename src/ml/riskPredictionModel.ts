// src/ml/riskPredictionModel.ts

import { Grid, CellType } from '../types';
import { DataCollector, MissionData } from '../utils/dataCollector';

export interface RiskPrediction {
  difficultyScore: number;
  estimatedTime: number;
  recommendedDrones: number;
  successProbability: number;
  riskLevel: 'low' | 'medium' | 'high' | 'extreme';
  confidence: number;
}

export class RiskPredictionModel {
  private trainingData: MissionData[] = [];
  
  train(missions: MissionData[]): void {
    this.trainingData = missions;
  }
  
  predict(
    grid: Grid,
    victimCount: number,
    droneCount: number
  ): RiskPrediction {
    let obstacleCount = 0;
    let fireCount = 0;
    let waterCount = 0;
    
    for (let y = 0; y < grid.height; y++) {
      for (let x = 0; x < grid.width; x++) {
        const cell = grid.cells[y][x];
        if (cell.type === CellType.OBSTACLE) obstacleCount++;
        else if (cell.type === CellType.FIRE) fireCount++;
        else if (cell.type === CellType.WATER) waterCount++;
      }
    }
    
    const difficultyScore = DataCollector.calculateDifficulty(
      obstacleCount,
      fireCount,
      waterCount,
      victimCount,
      droneCount
    );
    
    let estimatedTime = 30;
    let successProbability = 0.85;
    let confidence = 0.5;
    
    if (this.trainingData.length > 0) {
      const similarMissions = this.trainingData.filter(m => 
        Math.abs(m.difficultyScore - difficultyScore) < 15 &&
        Math.abs(m.victimCount - victimCount) <= 2
      );
      
      if (similarMissions.length > 0) {
        estimatedTime = Math.round(
          similarMissions.reduce((sum, m) => sum + m.totalTime, 0) / similarMissions.length
        );
        
        const successCount = similarMissions.filter(m => m.missionSuccess).length;
        successProbability = successCount / similarMissions.length;
        
        confidence = Math.min(0.95, 0.5 + (similarMissions.length * 0.1));
      }
    }
    
    const recommendedDrones = Math.min(6, Math.max(2, Math.ceil(victimCount / 2)));
    
    let riskLevel: 'low' | 'medium' | 'high' | 'extreme';
    if (difficultyScore < 30) riskLevel = 'low';
    else if (difficultyScore < 50) riskLevel = 'medium';
    else if (difficultyScore < 75) riskLevel = 'high';
    else riskLevel = 'extreme';
    
    return {
      difficultyScore,
      estimatedTime,
      recommendedDrones,
      successProbability: Math.round(successProbability * 100) / 100,
      riskLevel,
      confidence: Math.round(confidence * 100) / 100
    };
  }
  
  getImprovementMetrics(): {
    missionsAnalyzed: number;
    averageAccuracy: number;
    totalLearningTime: number;
  } {
    return {
      missionsAnalyzed: this.trainingData.length,
      averageAccuracy: DataCollector.getAverageAccuracy(),
      totalLearningTime: this.trainingData.reduce((sum, m) => sum + m.totalTime, 0)
    };
  }
}