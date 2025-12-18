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
      const inputGrid: number[][] = Array(25).fill(0).map(() => Array(30).fill(0));
      
      for (let y = 0; y < 25; y++) {
        for (let x = 0; x < 30; x++) {
          inputGrid[y][x] = Math.random() < 0.1 ? 0.5 : 0;
        }
      }
      
      const outputGrid: number[][] = Array(25).fill(0).map(() => Array(30).fill(0));
      
      mission.victims.forEach(victim => {
        const pos = victim.position;
        if (pos && pos.y < 25 && pos.x < 30) {
          outputGrid[pos.y][pos.x] = victim.priority / 5;
        }
      });
      
      inputs.push(inputGrid);
      outputs.push(outputGrid);
    });
    
    return {
      inputs: tf.tensor3d(inputs, [inputs.length, 25, 30]),
      outputs: tf.tensor3d(outputs, [outputs.length, 25, 30])
    };
  }
  
  async buildModel(): Promise<void> {
    this.model = tf.sequential({
      layers: [
        tf.layers.conv2d({
          inputShape: [25, 30, 1],
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
    
    const { inputs, outputs } = await this.prepareTrainingData(missions);
    
    const reshapedInputs = inputs.reshape([missions.length, 25, 30, 1]);
    const reshapedOutputs = outputs.reshape([missions.length, 25, 30, 1]);
    
    await this.model!.fit(reshapedInputs, reshapedOutputs, {
      epochs,
      batchSize: 4,
      validationSplit: 0.2,
      callbacks: {
        onEpochEnd: (epoch: number, logs: any) => {
          this.trainingProgress = ((epoch + 1) / epochs) * 100;
          console.log(`Epoch ${epoch + 1}/${epochs} - Loss: ${logs?.loss.toFixed(4)}`);
        }
      }
    });
    
    this.isTraining = false;
    this.trainingProgress = 100;
    
    inputs.dispose();
    outputs.dispose();
    reshapedInputs.dispose();
    reshapedOutputs.dispose();
  }
  
  async predict(grid: Grid): Promise<number[][]> {
  if (!this.model) {
    return Array(grid.height).fill(0).map(() => Array(grid.width).fill(0.1));
  }
  
  try {
    const gridTensor = this.gridToTensor(grid);
    
    // Create proper 4D tensor for CNN input
    const inputArray = [gridTensor];
    const inputTensor = tf.tensor4d(inputArray as number[][][][], [1, grid.height, grid.width, 1]);
    
    const prediction = this.model.predict(inputTensor) as tf.Tensor4D;
    
    // Squeeze to remove batch and channel dimensions
    const squeezed = prediction.squeeze([0, 3]) as tf.Tensor2D;
    
    // Get the array data
    const predictionArray = await squeezed.array();
    
    // Clean up tensors
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