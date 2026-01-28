// src/App.tsx - Complete ML-Powered RescueAI System

import { useState, useEffect, useRef } from 'react';
import { GridCanvas } from './components/GridCanvas';
import { MLDashboard } from './components/MLDashboard';
import { Grid, Drone, Victim, Position, CellType, DroneStatus } from './types';
import { createEmptyGrid, setCellType, clearPath, manhattanDistance } from './utils/gridUtils';
import { findPath } from './utils/pathfinding';
import { scenarios } from './data/scenarios';
import { DataCollector } from './utils/dataCollector';
import { SwarmCommunication } from './utils/swarmCommunication';
import { VictimDetectionModel } from './ml/victimDetectionModel';
import { RiskPredictionModel } from './ml/riskPredictionModel';
import { AutoSimulator } from './utils/autoSimulator';
import { VisionUploader } from './components/VisionUploader';

function App() {
  const [currentScenario, setCurrentScenario] = useState<string>('wildfire');
  const [grid, setGrid] = useState<Grid>(() => createEmptyGrid(30, 25, 25));
  const [drones, setDrones] = useState<Drone[]>([]);
  const [victims, setVictims] = useState<Victim[]>([]);
  const [isSimulating, setIsSimulating] = useState(false);
  const [selectedTool, setSelectedTool] = useState<CellType>(CellType.OBSTACLE);
  const [missionTime, setMissionTime] = useState(0);
  const [startTime, setStartTime] = useState<number | null>(null);
  const [currentMissionId, setCurrentMissionId] = useState<string | null>(null);
  
  // ML States
  const [showVisionUploader, setShowVisionUploader] = useState(false);
  const [victimDetectionModel] = useState(() => new VictimDetectionModel());
  const [riskPredictionModel] = useState(() => new RiskPredictionModel());
  const [isTrainingModel, setIsTrainingModel] = useState(false);
  const [modelReady, setModelReady] = useState(false);
  const [showRiskHeatmap, setShowRiskHeatmap] = useState(false);
  const [showVictimHeatmap, setShowVictimHeatmap] = useState(false);
  const [riskPrediction, setRiskPrediction] = useState<any>(null);
  const [autoSimProgress, setAutoSimProgress] = useState<{running: boolean, current: number, total: number}>({
    running: false,
    current: 0,
    total: 0
  });

  // Alert guard flag to prevent duplicate alerts
  const alertShownRef = useRef(false);

  const loadScenario = (scenarioKey: string) => {
    alertShownRef.current = false; // Reset flag
    const scenario = scenarios[scenarioKey];
    setCurrentScenario(scenarioKey);
    setIsSimulating(false);
    setMissionTime(0);
    setStartTime(null);
    setCurrentMissionId(null);
    SwarmCommunication.reset();
    
    const newGrid = createEmptyGrid(scenario.gridSize.width, scenario.gridSize.height, 25);
    
    scenario.obstacles.forEach(obstacleGroup => {
      obstacleGroup.positions.forEach(pos => {
        let cellType: CellType;
        switch (obstacleGroup.type) {
          case 'fire': cellType = CellType.FIRE; break;
          case 'water': cellType = CellType.WATER; break;
          case 'obstacle': cellType = CellType.OBSTACLE; break;
        }
        setCellType(newGrid, pos.x, pos.y, cellType);
      });
    });
    
    setGrid(newGrid);
    
    const colors = ['#10b981', '#3b82f6', '#f59e0b', '#8b5cf6'];
    const newDrones = scenario.dronePositions.map((pos, idx) => ({
      id: `drone-${idx + 1}`,
      position: pos,
      targetPosition: null,
      path: [],
      status: DroneStatus.IDLE,
      battery: 100,
      speed: 2,
      rescuedVictims: 0,
      color: colors[idx % colors.length]
    }));
    setDrones(newDrones);
    
    setVictims(scenario.victims.map(v => ({ ...v, detectedAt: Date.now() })));
    
    // Generate risk prediction
    const prediction = riskPredictionModel.predict(newGrid, scenario.victims.length, newDrones.length);
    setRiskPrediction(prediction);
  };

  useEffect(() => {
    loadScenario('wildfire');
    
    // Try to load saved model
    victimDetectionModel.loadModel().then(loaded => {
      if (loaded) setModelReady(true);
    });
  }, []);

  useEffect(() => {
    if (isSimulating && startTime) {
      const interval = setInterval(() => {
        setMissionTime(Math.floor((Date.now() - startTime) / 1000));
      }, 1000);
      return () => clearInterval(interval);
    }
  }, [isSimulating, startTime]);

  const handleCellClick = (position: Position) => {
    if (isSimulating) return;
    const newGrid = { ...grid };
    setCellType(newGrid, position.x, position.y, selectedTool);
    setGrid(newGrid);
  };

const assignVictimsToDrones = () => {
  const availableVictims = SwarmCommunication.getUnassignedVictims(victims);
  const idleDrones = drones.filter(d => d.status === DroneStatus.IDLE);
  
  const sortedVictims = [...availableVictims].sort((a, b) => b.priority - a.priority);
  const assignments: { drone: Drone; victim: Victim; distance: number }[] = [];
  
  for (const victim of sortedVictims) {
    let closestDrone: Drone | null = null;
    let minDistance = Infinity;
    
    for (const drone of idleDrones) {
      // Skip if already assigned
      const isAssigned = assignments.find(a => a.drone.id === drone.id);
      if (isAssigned) continue;
      
      const distance = manhattanDistance(drone.position, victim.position);
      if (distance < minDistance) {
        minDistance = distance;
        closestDrone = drone;
      }
    }
    
    // Only proceed if we found a drone
    if (closestDrone !== null) {
      assignments.push({ 
        drone: closestDrone, 
        victim, 
        distance: minDistance 
      });
      SwarmCommunication.assignVictim(closestDrone.id, victim.id);
    }
  }
  
  return assignments;
};

  const startMission = () => {
    alertShownRef.current = false; // Reset flag when starting new mission
    setIsSimulating(true);
    setStartTime(Date.now());
    setMissionTime(0);
    
    // Start data collection
    const missionId = DataCollector.startMission(currentScenario, grid, victims, drones);
    setCurrentMissionId(missionId);
    
    const assignments = assignVictimsToDrones();
    
    if (assignments.length === 0) {
      alert('All victims rescued or no available drones!');
      setIsSimulating(false);
      return;
    }
    
    clearPath(grid);
    const newDrones = [...drones];
    
    assignments.forEach(({ drone, victim }) => {
      const path = findPath(grid, drone.position, victim.position);
      
      if (path) {
        path.forEach(pos => {
          if (grid.cells[pos.y][pos.x].type === CellType.EMPTY) {
            setCellType(grid, pos.x, pos.y, CellType.PATH);
          }
        });
        
        const droneIndex = newDrones.findIndex(d => d.id === drone.id);
        newDrones[droneIndex] = {
          ...drone,
          path,
          targetPosition: victim.position,
          status: DroneStatus.MOVING
        };
        
        // Broadcast to swarm
        SwarmCommunication.broadcastVictimFound(drone.id, victim);
        
        simulateDroneMovement(path, drone.id, victim.id);
      }
    });
    
    setDrones(newDrones);
    setGrid({ ...grid });
  };

  const simulateDroneMovement = (path: Position[], droneId: string, victimId: string) => {
  let currentIndex = 0;
  
  const interval = setInterval(() => {
    if (currentIndex >= path.length) {
      clearInterval(interval);
      
      // Broadcast rescue complete
      SwarmCommunication.broadcastRescueComplete(droneId, victimId);
      
      setVictims(prev => prev.map(v => 
        v.id === victimId ? { ...v, isRescued: true } : v
      ));
      
      setDrones(prev => {
        const updated = prev.map(d => 
          d.id === droneId ? { 
            ...d, 
            status: DroneStatus.IDLE,
            rescuedVictims: d.rescuedVictims + 1,
            path: [],
            battery: Math.max(0, d.battery - 10)
          } : d
        );
        
        // Check mission status ONLY after a short delay to let state settle
        setTimeout(() => {
          // Re-check current victims state
          setVictims(currentVictims => {
            const allRescued = currentVictims.every(v => v.isRescued);
            const anyDroneMoving = updated.some(d => d.status === DroneStatus.MOVING);
            
            if (allRescued && !anyDroneMoving && !alertShownRef.current) {
              // Set flag IMMEDIATELY to prevent duplicates
              alertShownRef.current = true;
              
              setIsSimulating(false);
              
              // Capture the CURRENT mission time
              const finalTime = missionTime > 0 ? missionTime : Math.floor((Date.now() - (startTime || Date.now())) / 1000);
              
              if (currentMissionId && startTime) {
                DataCollector.completeMission(currentMissionId, finalTime, currentVictims, updated);
                const missions = DataCollector.getMissions();
                riskPredictionModel.train(missions);
              }
              
              alert(`🎉 Mission Complete! All ${currentVictims.length} victims rescued in ${finalTime} seconds!`);
            } else if (!allRescued && !anyDroneMoving) {
              // Some victims left and no drones moving - assign more
              const idleDrones = updated.filter(d => d.status === DroneStatus.IDLE);
              if (idleDrones.length > 0) {
                startMission();
              }
            }
            
            return currentVictims;
          });
        }, 500); // Wait 500ms for all states to update
        
        return updated;
      });
      
      return;
    }
    
    const newPos = path[currentIndex];
    setDrones(prev => prev.map(d => 
      d.id === droneId ? { ...d, position: newPos } : d
    ));
    
    currentIndex++;
  }, 200);
};

  const handleTrainModel = async () => {
    const missions = DataCollector.getMissions();
    
    if (missions.length < 10) {
      alert('Need at least 10 missions to train the model. Run auto-simulations first!');
      return;
    }
    
    setIsTrainingModel(true);
    
    try {
      await victimDetectionModel.train(missions, 30);
      await victimDetectionModel.saveModel();
      setModelReady(true);
      alert('✅ Model trained successfully! AI predictions now active.');
    } catch (error) {
      console.error('Training failed:', error);
      alert('Training failed. Check console for details.');
    } finally {
      setIsTrainingModel(false);
    }
  };

  const handleRunAutoSimulations = async () => {
    setAutoSimProgress({ running: true, current: 0, total: 50 });
    
    const scenarioConfigs = Object.entries(scenarios).map(([key, scenario]) => {
      const newGrid = createEmptyGrid(scenario.gridSize.width, scenario.gridSize.height, 25);
      
      scenario.obstacles.forEach(obstacleGroup => {
        obstacleGroup.positions.forEach(pos => {
          let cellType: CellType;
          switch (obstacleGroup.type) {
            case 'fire': cellType = CellType.FIRE; break;
            case 'water': cellType = CellType.WATER; break;
            case 'obstacle': cellType = CellType.OBSTACLE; break;
          }
          setCellType(newGrid, pos.x, pos.y, cellType);
        });
      });
      
      const colors = ['#10b981', '#3b82f6', '#f59e0b', '#8b5cf6'];
      const newDrones = scenario.dronePositions.map((pos, idx) => ({
        id: `drone-${idx + 1}`,
        position: pos,
        targetPosition: null,
        path: [],
        status: DroneStatus.IDLE,
        battery: 100,
        speed: 2,
        rescuedVictims: 0,
        color: colors[idx % colors.length]
      }));
      
      return {
        scenarioKey: key,
        grid: newGrid,
        drones: newDrones,
        victims: scenario.victims.map(v => ({ ...v, detectedAt: Date.now() }))
      };
    });
    
    await AutoSimulator.runMultipleSimulations(
      50,
      scenarioConfigs,
      (current, total) => {
        setAutoSimProgress({ running: true, current, total });
      }
    );
    
    setAutoSimProgress({ running: false, current: 50, total: 50 });
    
    // Retrain models with new data
    const missions = DataCollector.getMissions();
    riskPredictionModel.train(missions);
    
    alert(`✅ Completed 50 simulations! Collected ${missions.length} total missions for training.`);
  };

  const activeDrones = drones.filter(d => d.status === DroneStatus.MOVING).length;
  const totalRescued = drones.reduce((sum, d) => sum + d.rescuedVictims, 0);
  const remainingVictims = victims.filter(v => !v.isRescued).length;
  const scenario = scenarios[currentScenario];
  const swarmStats = SwarmCommunication.getSwarmStats();
  const missions = DataCollector.getMissions();

  const styles = {
    container: { minHeight: '100vh', padding: '2rem', color: 'white' },
    header: { marginBottom: '2rem' },
    title: {
      fontSize: '3rem',
      fontWeight: 'bold',
      marginBottom: '0.5rem',
      background: 'linear-gradient(to right, #34d399, #22d3ee)',
      WebkitBackgroundClip: 'text',
      WebkitTextFillColor: 'transparent',
      backgroundClip: 'text'
    },
    subtitle: { color: '#94a3b8', fontSize: '1.125rem' },
    grid: { display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '1.5rem', maxWidth: '1600px' },
    card: {
      background: 'rgba(30, 41, 59, 0.5)',
      backdropFilter: 'blur(10px)',
      borderRadius: '1rem',
      padding: '1.5rem',
      border: '1px solid #334155'
    },
    buttonPrimary: {
      flex: 1,
      background: isSimulating ? '#475569' : '#059669',
      color: 'white',
      padding: '0.75rem 1.5rem',
      borderRadius: '0.5rem',
      fontWeight: '600',
      border: 'none',
      cursor: isSimulating ? 'not-allowed' : 'pointer',
      fontSize: '1rem'
    },
    buttonSecondary: {
      background: '#475569',
      color: 'white',
      padding: '0.75rem 1.5rem',
      borderRadius: '0.5rem',
      fontWeight: '600',
      border: 'none',
      cursor: 'pointer',
      fontSize: '1rem'
    },
    scenarioButton: (active: boolean) => ({
      padding: '1rem',
      borderRadius: '0.5rem',
      border: active ? '2px solid #10b981' : '2px solid #475569',
      background: active ? 'rgba(16, 185, 129, 0.1)' : 'transparent',
      cursor: 'pointer',
      transition: 'all 0.2s',
      textAlign: 'left' as const
    }),
    toolButton: (isSelected: boolean) => ({
      padding: '0.75rem',
      borderRadius: '0.5rem',
      border: isSelected ? '2px solid #10b981' : '2px solid #475569',
      background: isSelected ? '#334155' : 'transparent',
      cursor: 'pointer',
      transition: 'all 0.2s'
    }),
    colorBox: (color: string) => ({
      width: '100%',
      height: '2rem',
      borderRadius: '0.25rem',
      background: color,
      marginBottom: '0.5rem'
    }),
    statRow: {
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'center',
      marginBottom: '0.75rem'
    },
    badge: (color: string) => ({
      display: 'inline-block',
      padding: '0.25rem 0.75rem',
      borderRadius: '1rem',
      fontSize: '0.875rem',
      fontWeight: '600',
      background: color
    }),
    toggleButton: (active: boolean) => ({
      padding: '0.5rem 1rem',
      borderRadius: '0.5rem',
      border: active ? '2px solid #8b5cf6' : '2px solid #475569',
      background: active ? 'rgba(139, 92, 246, 0.2)' : 'transparent',
      color: 'white',
      cursor: 'pointer',
      fontSize: '0.875rem',
      fontWeight: '600',
      marginRight: '0.5rem'
    })
  };

  return (
    <div style={styles.container}>
      <div style={{ maxWidth: '1600px', margin: '0 auto' }}>
        <div style={styles.header}>
          <h1 style={styles.title}>🚁 RescueAI - Intelligent Swarm System</h1>
          <p style={styles.subtitle}>ML-Powered Multi-Drone Coordination with Predictive Analytics</p>
        </div>

        {autoSimProgress.running && (
          <div style={{
            ...styles.card,
            marginBottom: '1rem',
            background: 'rgba(139, 92, 246, 0.2)',
            border: '2px solid #8b5cf6'
          }}>
            <div style={{ fontSize: '1rem', fontWeight: 'bold', marginBottom: '0.5rem' }}>
              🤖 Running Auto-Simulations: {autoSimProgress.current}/{autoSimProgress.total}
            </div>
            <div style={{ width: '100%', height: '8px', background: '#334155', borderRadius: '4px' }}>
              <div style={{
                width: `${(autoSimProgress.current / autoSimProgress.total) * 100}%`,
                height: '100%',
                background: 'linear-gradient(to right, #8b5cf6, #ec4899)',
                borderRadius: '4px',
                transition: 'width 0.3s ease'
              }} />
            </div>
          </div>
        )}

        <div style={styles.grid}>
          <div>
            <div style={styles.card}>
              <div style={{ marginBottom: '1rem' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '1rem' }}>
                  <div>
                    <div style={{ fontSize: '1.5rem', fontWeight: 'bold', marginBottom: '0.25rem' }}>
                      {scenario.icon} {scenario.name}
                    </div>
                    <div style={{ fontSize: '0.875rem', color: '#94a3b8', marginBottom: '0.5rem' }}>
                      {scenario.description}
                    </div>
                    <div>
                      <span style={styles.badge(isSimulating ? '#059669' : '#475569')}>
                        {isSimulating ? '🟢 ACTIVE' : '⚪ STANDBY'}
                      </span>
                      {isSimulating && (
                        <span style={{ ...styles.badge('#1e293b'), marginLeft: '0.5rem' }}>
                          ⏱️ {missionTime}s
                        </span>
                      )}
                      <span style={{ ...styles.badge(
                        scenario.difficulty === 'Easy' ? '#22c55e' : 
                        scenario.difficulty === 'Medium' ? '#f59e0b' : '#ef4444'
                      ), marginLeft: '0.5rem' }}>
                        {scenario.difficulty}
                      </span>
                    </div>
                  </div>
                  
                  {riskPrediction && (
                    <div style={{
                      background: 'rgba(139, 92, 246, 0.1)',
                      border: '1px solid #8b5cf6',
                      borderRadius: '0.5rem',
                      padding: '0.75rem',
                      minWidth: '200px'
                    }}>
                      <div style={{ fontSize: '0.75rem', color: '#94a3b8', marginBottom: '0.25rem' }}>
                        🤖 AI PREDICTION
                      </div>
                      <div style={{ fontSize: '0.875rem', marginBottom: '0.25rem' }}>
                        Difficulty: <strong>{riskPrediction.difficultyScore}/100</strong>
                      </div>
                      <div style={{ fontSize: '0.875rem', marginBottom: '0.25rem' }}>
                        Est. Time: <strong>{riskPrediction.estimatedTime}s</strong>
                      </div>
                      <div style={{ fontSize: '0.875rem' }}>
                        Success Rate: <strong>{Math.round(riskPrediction.successProbability * 100)}%</strong>
                      </div>
                      <div style={{ fontSize: '0.75rem', color: '#94a3b8', marginTop: '0.5rem' }}>
                        Confidence: {Math.round(riskPrediction.confidence * 100)}%
                      </div>
                    </div>
                  )}
                </div>
                
                <div style={{ marginBottom: '0.5rem' }}>
                  <button
                    onClick={() => setShowRiskHeatmap(!showRiskHeatmap)}
                    style={styles.toggleButton(showRiskHeatmap)}
                  >
                    {showRiskHeatmap ? '✓' : ''} Risk Heatmap
                  </button>
                  <button
                    onClick={() => setShowVictimHeatmap(!showVictimHeatmap)}
                    style={styles.toggleButton(showVictimHeatmap)}
                    disabled={!modelReady}
                  >
                    {showVictimHeatmap ? '✓' : ''} AI Predictions
                  </button>
                </div>
              </div>
              
              <GridCanvas
                grid={grid}
                drones={drones}
                victims={victims}
                onCellClick={handleCellClick}
                showRiskHeatmap={showRiskHeatmap}
                showVictimHeatmap={showVictimHeatmap && modelReady}
              />
              
              <div style={{ marginTop: '1.5rem', display: 'flex', gap: '1rem' }}>
                <button onClick={startMission} disabled={isSimulating || remainingVictims === 0} style={styles.buttonPrimary}>
                  {isSimulating ? '🚁 Mission In Progress...' : remainingVictims === 0 ? '✅ All Rescued!' : '🚀 Deploy Drone Fleet'}
                </button>
                
                <button onClick={() => loadScenario(currentScenario)} style={styles.buttonSecondary}>
                  🔄 Reset
                </button>
              </div>
              
              {swarmStats.messagesSent > 0 && (
                <div style={{ marginTop: '1rem', padding: '0.75rem', background: 'rgba(16, 185, 129, 0.1)', borderRadius: '0.5rem', border: '1px solid #10b981' }}>
                  <div style={{ fontSize: '0.875rem', fontWeight: 'bold', marginBottom: '0.5rem', color: '#10b981' }}>
                    📡 Swarm Network Activity
                  </div>
                  <div style={{ fontSize: '0.75rem', color: '#94a3b8' }}>
                    Messages: {swarmStats.messagesSent} | Victims Discovered: {swarmStats.victimsDiscovered} | Active Assignments: {swarmStats.activeAssignments}
                  </div>
                </div>
              )}
            </div>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
            <MLDashboard
              isTraining={isTrainingModel}
              trainingProgress={victimDetectionModel.getTrainingProgress()}
              modelReady={modelReady}
              missionsCollected={missions.length}
              averageAccuracy={DataCollector.getAverageAccuracy()}
              onTrainModel={handleTrainModel}
              onRunAutoSimulations={handleRunAutoSimulations}
            />
{/* Vision System Upload */}
<div style={styles.card}>
  <button
    onClick={() => setShowVisionUploader(!showVisionUploader)}
    style={{
      ...styles.buttonPrimary,
      width: '100%',
      marginBottom: '1rem',
      background: showVisionUploader ? '#475569' : 'linear-gradient(to right, #8b5cf6, #ec4899)'
    }}
  >
    {showVisionUploader ? '📷 Hide Vision Upload' : '🤖 Upload Real Disaster Image'}
  </button>
  
  {showVisionUploader && (
    <VisionUploader
      onScenarioGenerated={(scenario) => {
        console.log('Vision scenario received:', scenario);
        
        // Update grid with vision data
        const newGrid = {
          ...grid,
          cells: scenario.grid.cells
        };
        setGrid(newGrid);
        
        // Set victims from AI detection
        setVictims(scenario.victims);
        
        // Update drone positions to safe zones
        const newDrones = drones.map((drone, idx) => {
          if (scenario.dronePositions[idx]) {
            return {
              ...drone,
              position: scenario.dronePositions[idx],
              status: 'idle' as any
            };
          }
          return drone;
        });
        setDrones(newDrones);
        
        // Update scenario info
        setCurrentScenario(scenario.scenarioType);
        
        // Show success message
        alert(`✅ Vision Scenario Loaded!\n\nType: ${scenario.scenarioType}\nVictims Detected: ${scenario.victims.length}\nConfidence: ${Math.round(scenario.confidence * 100)}%\n\nClick "Deploy Drone Fleet" to start rescue!`);
      }}
      apiUrl="http://localhost:8000"
    />
  )}
</div>
            <div style={styles.card}>
              <h3 style={{ fontSize: '1.25rem', fontWeight: 'bold', marginBottom: '1rem' }}>🌍 Emergency Scenarios</h3>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                {Object.entries(scenarios).map(([key, s]) => (
                  <button
                    key={key}
                    onClick={() => !isSimulating && loadScenario(key)}
                    style={styles.scenarioButton(currentScenario === key)}
                    disabled={isSimulating}
                  >
                    <div style={{ fontSize: '1.5rem', marginBottom: '0.25rem' }}>{s.icon}</div>
                    <div style={{ fontWeight: '600', marginBottom: '0.25rem', color: 'white' }}>
                      {s.name}
                    </div>
                    <div style={{ fontSize: '0.75rem', color: '#94a3b8' }}>
                      {s.victims.length} victims • {s.difficulty}
                    </div>
                  </button>
                ))}
              </div>
            </div>

            <div style={styles.card}>
              <h3 style={{ fontSize: '1.25rem', fontWeight: 'bold', marginBottom: '1rem' }}>📊 Mission Analytics</h3>
              <div>
                <div style={styles.statRow}>
                  <span style={{ color: '#94a3b8' }}>Active Drones:</span>
                  <span style={{ fontWeight: 'bold', color: '#10b981' }}>{activeDrones}/{drones.length}</span>
                </div>
                <div style={styles.statRow}>
                  <span style={{ color: '#94a3b8' }}>Victims Remaining:</span>
                  <span style={{ fontWeight: 'bold', color: '#eab308' }}>{remainingVictims}</span>
                </div>
                <div style={styles.statRow}>
                  <span style={{ color: '#94a3b8' }}>Rescued:</span>
                  <span style={{ fontWeight: 'bold', color: '#22c55e' }}>{totalRescued}</span>
                </div>
                <div style={styles.statRow}>
                  <span style={{ color: '#94a3b8' }}>Mission Time:</span>
                  <span style={{ fontWeight: 'bold', color: '#3b82f6' }}>{missionTime}s</span>
                </div>
                <div style={styles.statRow}>
                  <span style={{ color: '#94a3b8' }}>Success Rate:</span>
                  <span style={{ fontWeight: 'bold', color: '#8b5cf6' }}>
                    {victims.length > 0 ? Math.round((totalRescued / victims.length) * 100) : 0}%
                  </span>
                </div>
              </div>
            </div>

            <div style={styles.card}>
              <h3 style={{ fontSize: '1.25rem', fontWeight: 'bold', marginBottom: '1rem' }}>🚁 Drone Fleet</h3>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                {drones.map(drone => (
                  <div key={drone.id} style={{
                    padding: '0.75rem',
                    background: 'rgba(15, 23, 42, 0.5)',
                    borderRadius: '0.5rem',
                    border: `2px solid ${drone.color}`
                  }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                      <span style={{ fontWeight: '600', color: drone.color, fontSize: '0.875rem' }}>
                        {drone.id.toUpperCase()}
                      </span>
                      <span style={{
                        fontSize: '0.75rem',
                        padding: '0.125rem 0.5rem',
                        borderRadius: '0.25rem',
                        background: drone.status === DroneStatus.MOVING ? '#059669' : '#475569',
                        color: 'white'
                      }}>
                        {drone.status === DroneStatus.MOVING ? 'ACTIVE' : 'IDLE'}
                      </span>
                    </div>
                    <div style={{ fontSize: '0.75rem', color: '#94a3b8' }}>
                      Battery: {drone.battery}% | Rescued: {drone.rescuedVictims}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;