// src/components/MLDashboard.tsx

interface MLDashboardProps {
  isTraining: boolean;
  trainingProgress: number;
  modelReady: boolean;
  missionsCollected: number;
  averageAccuracy: number;
  onTrainModel: () => void;
  onRunAutoSimulations: () => void;
}

export const MLDashboard = ({
  isTraining,
  trainingProgress,
  modelReady,
  missionsCollected,
  averageAccuracy,
  onTrainModel,
  onRunAutoSimulations
}: MLDashboardProps) => {
  
  const styles = {
    card: {
      background: 'rgba(30, 41, 59, 0.5)',
      backdropFilter: 'blur(10px)',
      borderRadius: '1rem',
      padding: '1.5rem',
      border: '1px solid #334155'
    },
    statRow: {
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'center',
      marginBottom: '0.75rem'
    },
    button: (disabled: boolean) => ({
      width: '100%',
      background: disabled ? '#475569' : '#8b5cf6',
      color: 'white',
      padding: '0.75rem 1rem',
      borderRadius: '0.5rem',
      fontWeight: '600',
      border: 'none',
      cursor: disabled ? 'not-allowed' : 'pointer',
      fontSize: '0.875rem',
      marginBottom: '0.5rem'
    }),
    progressBar: {
      width: '100%',
      height: '8px',
      background: '#334155',
      borderRadius: '4px',
      overflow: 'hidden',
      marginTop: '0.5rem'
    },
    progressFill: {
      height: '100%',
      background: 'linear-gradient(to right, #8b5cf6, #ec4899)',
      transition: 'width 0.3s ease'
    }
  };
  
  return (
    <div style={styles.card}>
      <h3 style={{ fontSize: '1.25rem', fontWeight: 'bold', marginBottom: '1rem' }}>
        🤖 AI Learning System
      </h3>
      
      <div>
        <div style={styles.statRow}>
          <span style={{ color: '#94a3b8', fontSize: '0.875rem' }}>Model Status:</span>
          <span style={{ 
            fontWeight: 'bold', 
            color: modelReady ? '#22c55e' : '#94a3b8',
            fontSize: '0.875rem'
          }}>
            {isTraining ? 'Training...' : modelReady ? '✓ Ready' : 'Not Trained'}
          </span>
        </div>
        
        <div style={styles.statRow}>
          <span style={{ color: '#94a3b8', fontSize: '0.875rem' }}>Training Data:</span>
          <span style={{ fontWeight: 'bold', color: '#10b981', fontSize: '0.875rem' }}>
            {missionsCollected} missions
          </span>
        </div>
        
        <div style={styles.statRow}>
          <span style={{ color: '#94a3b8', fontSize: '0.875rem' }}>Prediction Accuracy:</span>
          <span style={{ fontWeight: 'bold', color: '#3b82f6', fontSize: '0.875rem' }}>
            {averageAccuracy}%
          </span>
        </div>
        
        {isTraining && (
          <div>
            <div style={{ fontSize: '0.875rem', color: '#94a3b8', marginTop: '1rem' }}>
              Training Progress: {Math.round(trainingProgress)}%
            </div>
            <div style={styles.progressBar}>
              <div style={{
                ...styles.progressFill,
                width: `${trainingProgress}%`
              }} />
            </div>
          </div>
        )}
        
        <div style={{ marginTop: '1.5rem' }}>
          <button
            onClick={onRunAutoSimulations}
            disabled={isTraining}
            style={styles.button(isTraining)}
          >
            🔄 Run 50 Auto-Simulations
          </button>
          
          <button
            onClick={onTrainModel}
            disabled={isTraining || missionsCollected < 10}
            style={styles.button(isTraining || missionsCollected < 10)}
          >
            {isTraining ? '⏳ Training Model...' : '🧠 Train AI Model'}
          </button>
          
          {missionsCollected < 10 && (
            <div style={{ 
              fontSize: '0.75rem', 
              color: '#f59e0b', 
              marginTop: '0.5rem',
              textAlign: 'center'
            }}>
              Need {10 - missionsCollected} more missions to train
            </div>
          )}
        </div>
      </div>
    </div>
  );
};