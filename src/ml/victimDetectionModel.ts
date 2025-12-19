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
  
private async prepareTrainingData(missions: MissionData[]) {
  const inputs: number[][][] = [];
  const outputs: number[][][] = [];
  
  missions.forEach(mission => {
    // Use actual grid size from mission
    const height = mission.gridSize.height;
    const width = mission.gridSize.width;
    
    const inputGrid: number[][] = Array(height).fill(0).map(() => Array(width).fill(0));
    
    // Mark obstacles based on mission data
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        // Add some variation based on obstacle/fire counts
        const density = (mission.obstacleCount + mission.fireCount) / (height * width);
        inputGrid[y][x] = Math.random() < density ? 0.5 : 0;
      }
    }
    
    const outputGrid: number[][] = Array(height).fill(0).map(() => Array(width).fill(0));
    
    mission.victims.forEach(victim => {
      const pos = victim.position;
      if (pos && pos.y < height && pos.x < width) {
        outputGrid[pos.y][pos.x] = victim.priority / 5;
      }
    });
    
    inputs.push(inputGrid);
    outputs.push(outputGrid);
  });
  
  return {
    inputs: tf.tensor3d(inputs, [inputs.length, 25, 30]),  // Make sure this matches!
    outputs: tf.tensor3d(outputs, [outputs.length, 25, 30])
  };
}
  
async buildModel(): Promise<void> {
  this.model = tf.sequential({
    layers: [
      tf.layers.conv2d({
        inputShape: [25, 30, 1],  // CHANGE FROM [25, 30, 1] - match grid size!
        filters: 16,
        kernelSize: 3,
        activation: 'relu',
        padding: 'same'
      }),
        
        tf.layers.maxPooling2d({
          poolSize: [2, 2]
        }),
        
        tf.layers.conv2d({
          filters: 32,
          kernelSize: 3,
          activation: 'relu',
          padding: 'same'
        }),
        
        tf.layers.upSampling2d({
          size: [2, 2]
        }),
        
        tf.layers.conv2d({
          filters: 16,
          kernelSize: 3,
          activation: 'relu',
          padding: 'same'
        }),
        
        tf.layers.conv2d({
          filters: 1,
          kernelSize: 1,
          activation: 'sigmoid'
        })
      ]
    });
    
    this.model.compile({
      optimizer: tf.train.adam(0.001),
      loss: 'binaryCrossentropy',
      metrics: ['accuracy']
    });
  }
  
async train(missions: MissionData[], epochs: number = 50): Promise<void> {
  if (!this.model) {
    await this.buildModel();
  }
  
  if (missions.length < 5) {
    console.log('Need at least 5 missions to train');
    return;
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
    
    // Reshape to match model input: [batch, 25, 30, 1]
    reshapedInputs = inputs.reshape([missions.length, 25, 30, 1]);
    reshapedOutputs = outputs.reshape([missions.length, 25, 30, 1]);
    
    await this.model!.fit(reshapedInputs, reshapedOutputs, {
      epochs,
      batchSize: Math.min(4, missions.length),
      validationSplit: 0.2,
      shuffle: true,
      callbacks: {
        onEpochEnd: (epoch: number, logs: tf.Logs | undefined) => {
          this.trainingProgress = ((epoch + 1) / epochs) * 100;
          const loss = logs?.loss?.toFixed(4) || 'N/A';
          const valLoss = logs?.val_loss?.toFixed(4) || 'N/A';
          console.log(`Epoch ${epoch + 1}/${epochs} - Loss: ${loss} - Val Loss: ${valLoss}`);
        }
      }
    });
    
    console.log('Training completed successfully');
    
  } catch (error) {
    console.error('Training failed:', error);
    // Re-throw if you want calling code to handle it
    throw new Error(`Training failed: ${error}`);
  } finally {
    // Cleanup tensors
    [inputs, outputs, reshapedInputs, reshapedOutputs].forEach(tensor => {
      if (tensor && !tensor.isDisposed) {
        tensor.dispose();
      }
    });
    
    this.isTraining = false;
    this.trainingProgress = 100;
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