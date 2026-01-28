// src/components/VisionUploader.tsx
import { useState } from 'react';

interface VisionUploaderProps {
  onScenarioGenerated: (scenario: any) => void;
  apiUrl?: string;
}

export function VisionUploader({ onScenarioGenerated, apiUrl = 'http://localhost:8000' }: VisionUploaderProps) {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (!selectedFile) return;

    setFile(selectedFile);
    setError(null);
    setSuccess(null);

    const reader = new FileReader();
    reader.onload = (event) => {
      setPreview(event.target?.result as string);
    };
    reader.readAsDataURL(selectedFile);
  };

  const generateScenario = async () => {
    if (!file) return;

    setIsAnalyzing(true);
    setError(null);
    setSuccess(null);

    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('grid_width', '30');
      formData.append('grid_height', '25');
      formData.append('threshold', '0.5');

      const response = await fetch(`${apiUrl}/grid-from-image`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Failed: ${response.statusText}`);
      }

      const scenario = await response.json();
      
      setSuccess(`✅ Scenario generated! Found ${scenario.victims.length} victims (Confidence: ${Math.round(scenario.confidence * 100)}%)`);
      onScenarioGenerated(scenario);

    } catch (err) {
      setError(err instanceof Error ? err.message : 'Analysis failed. Is the backend running?');
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div style={{
      background: 'rgba(30, 41, 59, 0.5)',
      backdropFilter: 'blur(10px)',
      borderRadius: '1rem',
      padding: '1.5rem',
      border: '1px solid #334155',
      marginBottom: '1.5rem'
    }}>
      <h3 style={{
        fontSize: '1.25rem',
        fontWeight: 'bold',
        marginBottom: '1rem',
        color: 'white'
      }}>
        📷 Vision System - Upload Real Image
      </h3>

      <label htmlFor="file-upload">
        <div style={{
          border: `2px dashed ${file ? '#10b981' : '#475569'}`,
          borderRadius: '0.5rem',
          padding: '2rem',
          textAlign: 'center',
          cursor: 'pointer',
          background: file ? 'rgba(16, 185, 129, 0.05)' : 'transparent',
          marginBottom: '1rem',
          transition: 'all 0.2s'
        }}>
          <div style={{ fontSize: '2rem', marginBottom: '0.5rem' }}>📁</div>
          <div style={{ color: '#94a3b8', fontSize: '0.875rem' }}>
            {file ? file.name : 'Click to upload disaster image'}
          </div>
          <div style={{ color: '#64748b', fontSize: '0.75rem', marginTop: '0.5rem' }}>
            JPG, PNG (Fire, Flood, Earthquake images)
          </div>
        </div>
      </label>

      <input
        id="file-upload"
        type="file"
        accept="image/*"
        onChange={handleFileSelect}
        style={{ display: 'none' }}
      />

      {preview && (
        <img 
          src={preview} 
          alt="Preview" 
          style={{
            width: '100%',
            maxHeight: '300px',
            objectFit: 'contain',
            borderRadius: '0.5rem',
            marginBottom: '1rem',
            border: '1px solid #334155'
          }}
        />
      )}

      {file && (
        <button
          onClick={generateScenario}
          disabled={isAnalyzing}
          style={{
            width: '100%',
            background: isAnalyzing ? '#475569' : 'linear-gradient(to right, #8b5cf6, #ec4899)',
            color: 'white',
            padding: '0.75rem 1.5rem',
            borderRadius: '0.5rem',
            fontWeight: '600',
            border: 'none',
            cursor: isAnalyzing ? 'not-allowed' : 'pointer',
            fontSize: '1rem',
            marginBottom: '0.5rem'
          }}
        >
          {isAnalyzing ? '🔄 Analyzing Image & Generating...' : '🎮 Generate Mission from Image'}
        </button>
      )}

      {error && (
        <div style={{
          background: 'rgba(239, 68, 68, 0.1)',
          border: '1px solid #ef4444',
          borderRadius: '0.5rem',
          padding: '0.75rem',
          color: '#fca5a5',
          fontSize: '0.875rem'
        }}>
          ⚠️ {error}
        </div>
      )}

      {success && (
        <div style={{
          background: 'rgba(16, 185, 129, 0.1)',
          border: '1px solid #10b981',
          borderRadius: '0.5rem',
          padding: '0.75rem',
          color: '#10b981',
          fontSize: '0.875rem'
        }}>
          {success}
        </div>
      )}
    </div>
  );
}