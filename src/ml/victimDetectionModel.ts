// src/ml/victimDetectionModel.ts

import * as tf from '@tensorflow/tfjs';
import { Grid, CellType } from '../types';
import { MissionData } from '../utils/dataCollector';

export class VictimDetectionModel {
  private model: tf.LayersModel | null = null;
  private isTraining: boolean = false;
  private trainingProgress: number = 0;
  
  private gridToTensor(grid: Grid): number[][] {
    const tensor: number[][] = [];
    
    for (let y = 0; y < grid.height; y++) {
      const row: number[] = [];
      for (let x = 0; x < grid.width; x++) {
        const cell = grid.cells[y][x];
        
        let value = 0;
        if (cell.type === CellType.EMPTY) value = 0;
        else if (cell.type === CellType.OBSTACLE) value = 0.3;
        else if (cell.type === CellType.FIRE) value = 0.7;
        else if (cell.type === CellType.WATER) value = 0.5;
        else if (cell.type === CellType.VICTIM) value = 1.0;
        
        row.push(value);
      }
      tensor.push(row);
    }
    
    return tensor;
  }
  
private async prepareTrainingData(missions: MissionData[]): Promise<{ inputs: tf.Tensor; outputs: tf.Tensor }> {
  const inputs: number[][][] = [];
  const outputs: number[][][] = [];
  
  // Filter missions to only use those with matching grid size
  const validMissions = missions.filter(m => 
    m.gridSize.height === 25 && m.gridSize.width === 30
  );
  
  if (validMissions.length < 5) {
    throw new Error(`Not enough valid missions. Need 25x30 grids, found only ${validMissions.length} valid missions out of ${missions.length}`);
  }
  
  console.log(`Using ${validMissions.length} valid missions (filtered from ${missions.length})`);
  
  validMissions.forEach(mission => {
    const height = 25;
    const width = 30;
    
    // Create input grid
    const inputGrid: number[][] = Array(height).fill(0).map(() => Array(width).fill(0));
    
    // Calculate obstacle density
    const density = (mission.obstacleCount + mission.fireCount) / (height * width);
    
    // Fill input grid with obstacle probabilities
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        inputGrid[y][x] = Math.random() < density ? 0.5 : 0;
      }
    }
    
    // Create output grid
    const outputGrid: number[][] = Array(height).fill(0).map(() => Array(width).fill(0));
    
    // Mark victim positions
    mission.victims.forEach(victim => {
      const pos = victim.position;
      if (pos && pos.y < height && pos.x < width) {
        // Normalize priority to 0-1 range
        outputGrid[pos.y][pos.x] = Math.min(1, Math.max(0, victim.priority / 5));
      }
    });
    
    inputs.push(inputGrid);
    outputs.push(outputGrid);
  });
  
  console.log(`Training data shapes: inputs[${inputs.length}][25][30], outputs[${outputs.length}][25][30]`);
  
  // Create tensors with explicit types
  const inputTensor = tf.tensor3d(inputs, [inputs.length, 25, 30]);
  const outputTensor = tf.tensor3d(outputs, [outputs.length, 25, 30]);
  
  return {
    inputs: inputTensor,
    outputs: outputTensor
  };
}
  
async buildModel(): Promise<void> {
  if (this.model) {
    this.model.dispose();
  }
  
  this.model = tf.sequential({
    layers: [
      // Input: [25, 30, 1]
      tf.layers.conv2d({
        inputShape: [25, 30, 1],
        filters: 16,
        kernelSize: 3,
        activation: 'relu',
        padding: 'same'
      }),
      
      // Don't use pooling - just more conv layers
      tf.layers.conv2d({
        filters: 32,
        kernelSize: 3,
        activation: 'relu',
        padding: 'same'
      }),
      
      tf.layers.conv2d({
        filters: 16,
        kernelSize: 3,
        activation: 'relu',
        padding: 'same'
      }),
      
      // Output: [25, 30, 1]
      tf.layers.conv2d({
        filters: 1,
        kernelSize: 1,
        activation: 'sigmoid',
        padding: 'same'
      })
    ]
  });
  
  this.model.compile({
    optimizer: tf.train.adam(0.001),
    loss: 'binaryCrossentropy',
    metrics: ['accuracy']
  });
  
  console.log('Model built with input shape: [25, 30, 1] (no pooling)');
}
async train(missions: MissionData[], epochs: number = 50): Promise<void> {
  if (missions.length < 5) {
    console.log('Need at least 5 missions to train');
    return;
  }
  
  // ALWAYS rebuild model before training to ensure correct shape
  console.log('Building fresh model...');
  await this.buildModel();
  
  // Check if model was created
  if (!this.model) {
    throw new Error('Failed to build model');
  }
  
  this.isTraining = true;
  this.trainingProgress = 0;
  
  let inputs: tf.Tensor | null = null;
  let outputs: tf.Tensor | null = null;
  let reshapedInputs: tf.Tensor | null = null;
  let reshapedOutputs: tf.Tensor | null = null;
  
  try {
    const trainingData = await this.prepareTrainingData(missions);
    inputs = trainingData.inputs;
    outputs = trainingData.outputs;
    
    console.log('Input shape:', inputs.shape);
    console.log('Output shape:', outputs.shape);
    
    // Reshape to [batch, 25, 30, 1]
    reshapedInputs = inputs.reshape([missions.length, 25, 30, 1]);
    reshapedOutputs = outputs.reshape([missions.length, 25, 30, 1]);
    
    console.log('Reshaped input:', reshapedInputs.shape);
    console.log('Reshaped output:', reshapedOutputs.shape);
    
    await this.model.fit(reshapedInputs, reshapedOutputs, {
      epochs,
      batchSize: 2,
      validationSplit: 0.2,
      shuffle: true,
      callbacks: {
        onEpochEnd: (epoch: number, logs: any) => {
          this.trainingProgress = ((epoch + 1) / epochs) * 100;
          console.log(`Epoch ${epoch + 1}/${epochs} - Loss: ${logs?.loss?.toFixed(4)}`);
        }
      }
    });
    
    this.isTraining = false;
    this.trainingProgress = 100;
    console.log('Training completed successfully');
    
  } catch (error) {
    console.error('Training error:', error);
    this.isTraining = false;
    this.trainingProgress = 0;
    throw error;
  } finally {
    // Safely dispose all tensors
    const tensors = [inputs, outputs, reshapedInputs, reshapedOutputs];
    for (const tensor of tensors) {
      if (tensor && !tensor.isDisposed) {
        tensor.dispose();
      }
    }
  }
}
  
  async predict(grid: Grid): Promise<number[][]> {
  if (!this.model) {
    return Array(grid.height).fill(0).map(() => Array(grid.width).fill(0.1));
  }
  
  try {
    const gridTensor = this.gridToTensor(grid);
    
    // Flatten to 1D array
    const flatData: number[] = [];
    for (let y = 0; y < grid.height; y++) {
      for (let x = 0; x < grid.width; x++) {
        flatData.push(gridTensor[y][x]);
      }
    }
    
    // Create 4D tensor: [batch=1, height, width, channels=1]
    const inputTensor = tf.tensor4d(flatData, [1, grid.height, grid.width, 1]);
    
    // Predict
    const prediction = this.model.predict(inputTensor) as tf.Tensor;
    
    // Get shape and reshape to 2D
    const shape = prediction.shape;
    const height = typeof shape[1] === 'number' ? shape[1] : grid.height;
    const width = typeof shape[2] === 'number' ? shape[2] : grid.width;
    
    const squeezed = prediction.reshape([height, width]);
    
    // Convert to array
    const predictionArray = await squeezed.array() as number[][];
    
    // Cleanup
    inputTensor.dispose();
    prediction.dispose();
    squeezed.dispose();
    
    return predictionArray;
  } catch (error) {
    console.error('Prediction error:', error);
    return Array(grid.height).fill(0).map(() => Array(grid.width).fill(0.1));
  }
}
  
  getTrainingProgress(): number {
    return this.trainingProgress;
  }
  
  isModelTraining(): boolean {
    return this.isTraining;
  }
  
  isModelReady(): boolean {
    return this.model !== null && !this.isTraining;
  }
  
  async saveModel(): Promise<void> {
    if (this.model) {
      await this.model.save('localstorage://victim-detection-model');
    }
  }
  
  async loadModel(): Promise<boolean> {
    try {
      this.model = await tf.loadLayersModel('localstorage://victim-detection-model');
      return true;
    } catch (e) {
      console.log('No saved model found');
      return false;
    }
  }
}