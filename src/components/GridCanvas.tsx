// src/components/GridCanvas.tsx - Enhanced with Heatmap

import { useRef, useEffect } from 'react';
import { Grid, Position, Drone, Victim } from '../types';
import { getCellColor } from '../utils/gridUtils';
import { HeatmapData, HeatmapGenerator } from '../utils/heatmapGenerator';

interface GridCanvasProps {
  grid: Grid;
  drones: Drone[];
  victims: Victim[];
  onCellClick?: (position: Position) => void;
  showRiskHeatmap?: boolean;
  showVictimHeatmap?: boolean;
}

export const GridCanvas = ({ 
  grid, 
  drones, 
  victims,
  onCellClick,
  showRiskHeatmap = false,
  showVictimHeatmap = false
}: GridCanvasProps) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Draw base grid cells
    for (let y = 0; y < grid.height; y++) {
      for (let x = 0; x < grid.width; x++) {
        const cell = grid.cells[y][x];
        const cellX = x * grid.cellSize;
        const cellY = y * grid.cellSize;
        
        ctx.fillStyle = getCellColor(cell.type);
        ctx.fillRect(cellX, cellY, grid.cellSize, grid.cellSize);
        
        ctx.strokeStyle = '#334155';
        ctx.lineWidth = 0.5;
        ctx.strokeRect(cellX, cellY, grid.cellSize, grid.cellSize);
      }
    }
    
    // Draw risk heatmap overlay
    if (showRiskHeatmap) {
      const riskHeatmap = HeatmapGenerator.generateRiskHeatmap(grid);
      drawHeatmapOverlay(ctx, riskHeatmap, grid);
    }
    
    // Draw victim probability heatmap
    if (showVictimHeatmap) {
      const knownVictims = victims.filter(v => !v.isRescued).map(v => v.position);
      const victimHeatmap = HeatmapGenerator.generateVictimProbabilityHeatmap(grid, knownVictims);
      drawHeatmapOverlay(ctx, victimHeatmap, grid, 0.4);
    }
    
    // Draw victims
    victims.forEach(victim => {
      if (!victim.isRescued) {
        const x = victim.position.x * grid.cellSize + grid.cellSize / 2;
        const y = victim.position.y * grid.cellSize + grid.cellSize / 2;
        
        ctx.fillStyle = '#eab308';
        ctx.beginPath();
        ctx.arc(x, y, grid.cellSize * 0.3, 0, Math.PI * 2);
        ctx.fill();
        
        ctx.fillStyle = 'white';
        ctx.font = `${grid.cellSize * 0.4}px Arial`;
        ctx.textAlign = 'center' as CanvasTextAlign;
        ctx.textBaseline = 'middle';
        ctx.fillText(victim.priority.toString(), x, y);
      }
    });
    
    // Draw drones
    drones.forEach(drone => {
      const x = drone.position.x * grid.cellSize + grid.cellSize / 2;
      const y = drone.position.y * grid.cellSize + grid.cellSize / 2;
      
      ctx.fillStyle = drone.color;
      ctx.beginPath();
      ctx.arc(x, y, grid.cellSize * 0.35, 0, Math.PI * 2);
      ctx.fill();
      
      ctx.strokeStyle = 'white';
      ctx.lineWidth = 2;
      ctx.stroke();
      
      if (drone.path.length > 0) {
        ctx.strokeStyle = drone.color;
        ctx.lineWidth = 2;
        ctx.globalAlpha = 0.5;
        ctx.beginPath();
        
        const startX = drone.position.x * grid.cellSize + grid.cellSize / 2;
        const startY = drone.position.y * grid.cellSize + grid.cellSize / 2;
        ctx.moveTo(startX, startY);
        
        drone.path.forEach(pos => {
          const pathX = pos.x * grid.cellSize + grid.cellSize / 2;
          const pathY = pos.y * grid.cellSize + grid.cellSize / 2;
          ctx.lineTo(pathX, pathY);
        });
        
        ctx.stroke();
        ctx.globalAlpha = 1;
      }
    });
    
  }, [grid, drones, victims, showRiskHeatmap, showVictimHeatmap]);
  
  const drawHeatmapOverlay = (
    ctx: CanvasRenderingContext2D,
    heatmap: HeatmapData,
    grid: Grid,
    alpha: number = 0.4
  ) => {
    for (let y = 0; y < grid.height; y++) {
      for (let x = 0; x < grid.width; x++) {
        const value = heatmap.grid[y][x];
        if (value > 0.1) {
          const cellX = x * grid.cellSize;
          const cellY = y * grid.cellSize;
          
          ctx.fillStyle = HeatmapGenerator.getHeatmapColor(value, alpha);
          ctx.fillRect(cellX, cellY, grid.cellSize, grid.cellSize);
        }
      }
    }
  };
  
  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!onCellClick) return;
    
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const rect = canvas.getBoundingClientRect();
    const x = Math.floor((e.clientX - rect.left) / grid.cellSize);
    const y = Math.floor((e.clientY - rect.top) / grid.cellSize);
    
    if (x >= 0 && x < grid.width && y >= 0 && y < grid.height) {
      onCellClick({ x, y });
    }
  };
  
  return (
    <canvas
      ref={canvasRef}
      width={grid.width * grid.cellSize}
      height={grid.height * grid.cellSize}
      style={{
        border: '2px solid #475569',
        borderRadius: '0.5rem',
        cursor: 'crosshair'
      }}
      onClick={handleCanvasClick}
    />
  );
};