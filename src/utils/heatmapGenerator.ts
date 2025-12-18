// src/utils/heatmapGenerator.ts

import { Grid, Position, CellType } from '../types';

export interface HeatmapData {
  grid: number[][];
  maxValue: number;
  minValue: number;
}

export class HeatmapGenerator {
  
  static generateRiskHeatmap(grid: Grid): HeatmapData {
    const heatmap: number[][] = Array(grid.height).fill(0).map(() => Array(grid.width).fill(0));
    
    for (let y = 0; y < grid.height; y++) {
      for (let x = 0; x < grid.width; x++) {
        const cell = grid.cells[y][x];
        
        let risk = 0;
        
        if (cell.type === CellType.FIRE) {
          risk = 1.0;
          this.spreadHeat(heatmap, x, y, 0.7, 3, grid);
        } else if (cell.type === CellType.WATER) {
          risk = 0.8;
          this.spreadHeat(heatmap, x, y, 0.5, 2, grid);
        } else if (cell.type === CellType.OBSTACLE) {
          risk = 0.6;
          this.spreadHeat(heatmap, x, y, 0.3, 2, grid);
        }
        
        heatmap[y][x] = Math.max(heatmap[y][x], risk);
      }
    }
    
    return {
      grid: heatmap,
      maxValue: 1.0,
      minValue: 0
    };
  }
  
  static generateVictimProbabilityHeatmap(
    grid: Grid,
    knownVictims: Position[]
  ): HeatmapData {
    const heatmap: number[][] = Array(grid.height).fill(0).map(() => Array(grid.width).fill(0));
    
    for (let y = 0; y < grid.height; y++) {
      for (let x = 0; x < grid.width; x++) {
        const cell = grid.cells[y][x];
        
        if (!cell.isWalkable) {
          heatmap[y][x] = 0;
          continue;
        }
        
        let maxProb = 0.1;
        
        knownVictims.forEach(victim => {
          const distance = Math.sqrt(
            Math.pow(x - victim.x, 2) + Math.pow(y - victim.y, 2)
          );
          const prob = Math.max(0, 1 - (distance / 10));
          maxProb = Math.max(maxProb, prob);
        });
        
        if (cell.type === CellType.EMPTY) {
          maxProb += 0.2;
        }
        
        heatmap[y][x] = Math.min(1, maxProb);
      }
    }
    
    return {
      grid: heatmap,
      maxValue: 1.0,
      minValue: 0
    };
  }
  
  private static spreadHeat(
    heatmap: number[][],
    centerX: number,
    centerY: number,
    intensity: number,
    radius: number,
    grid: Grid
  ): void {
    for (let dy = -radius; dy <= radius; dy++) {
      for (let dx = -radius; dx <= radius; dx++) {
        const x = centerX + dx;
        const y = centerY + dy;
        
        if (x < 0 || x >= grid.width || y < 0 || y >= grid.height) continue;
        
        const distance = Math.sqrt(dx * dx + dy * dy);
        if (distance <= radius) {
          const falloff = 1 - (distance / radius);
          const heat = intensity * falloff;
          heatmap[y][x] = Math.max(heatmap[y][x], heat);
        }
      }
    }
  }
  
  static getHeatmapColor(value: number, alpha: number = 0.5): string {
    if (value < 0.5) {
      const ratio = value * 2;
      const r = Math.round(ratio * 255);
      const g = Math.round(ratio * 255);
      const b = 255;
      return `rgba(${r}, ${g}, ${b}, ${alpha})`;
    } else {
      const ratio = (value - 0.5) * 2;
      const r = 255;
      const g = Math.round((1 - ratio) * 255);
      const b = 0;
      return `rgba(${r}, ${g}, ${b}, ${alpha})`;
    }
  }
}