"""
FastAPI server for RescueAI Vision System.
Provides endpoints for image and video analysis.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, List, Optional
import torch
import numpy as np
import cv2
import io
from PIL import Image
import base64

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.multi_head_segmentation import MultiHeadSegmentation
from utils.preprocessing import ImagePreprocessor, VideoPreprocessor


app = FastAPI(
    title="RescueAI Vision API",
    description="Multi-task segmentation for disaster response",
    version="1.0.0"
)

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global model instance
model: Optional[MultiHeadSegmentation] = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
preprocessor = ImagePreprocessor(image_size=(256, 256))
video_preprocessor = VideoPreprocessor(frame_rate=2, image_size=(256, 256))


class AnalysisResponse(BaseModel):
    """Response model for image/video analysis."""
    fire_map: List[List[float]]
    flood_map: List[List[float]]
    collapse_map: List[List[float]]
    human_presence_map: List[List[float]]
    confidence: float
    metadata: Dict


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    device: str


@app.on_event("startup")
async def load_model():
    """Load the trained model on startup."""
    global model
    
    try:
        print("Loading RescueAI Vision Model...")
        
        model = MultiHeadSegmentation(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            output_size=(256, 256)
        )
        
        # Try to load trained weights
        checkpoint_path = "checkpoints/best_model.pth"
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Loaded trained model from {checkpoint_path}")
        except FileNotFoundError:
            print("⚠️ No trained checkpoint found, using pretrained encoder only")
        
        model = model.to(device)
        model.eval()
        
        print(f"✅ Model loaded successfully on {device}")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        model = None


@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if model is not None else "model_not_loaded",
        model_loaded=model is not None,
        device=str(device)
    )


@app.post("/analyze-image", response_model=AnalysisResponse)
async def analyze_image(file: UploadFile = File(...)):
    """
    Analyze a single image for disaster elements.
    
    Args:
        file: Uploaded image file (JPG, PNG)
        
    Returns:
        Segmentation maps and metadata
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        original_size = image.shape[:2]
        
        # Preprocess
        input_tensor = preprocessor.preprocess_image(image, mode='val')
        input_tensor = input_tensor.to(device)
        
        # Inference
        with torch.no_grad():
            predictions = model(input_tensor)
        
        # Convert to numpy arrays
        fire_map = predictions['fire'].cpu().numpy()[0, 0].tolist()
        flood_map = predictions['flood'].cpu().numpy()[0, 0].tolist()
        collapse_map = predictions['collapse'].cpu().numpy()[0, 0].tolist()
        human_map = predictions['human'].cpu().numpy()[0, 0].tolist()
        
        # Calculate confidence
        confidence = model.get_confidence_score(predictions)
        
        # Metadata
        metadata = {
            "original_size": {
                "height": original_size[0],
                "width": original_size[1]
            },
            "output_size": {
                "height": 256,
                "width": 256
            },
            "device": str(device),
            "filename": file.filename
        }
        
        return AnalysisResponse(
            fire_map=fire_map,
            flood_map=flood_map,
            collapse_map=collapse_map,
            human_presence_map=human_map,
            confidence=confidence,
            metadata=metadata
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/analyze-video")
async def analyze_video(file: UploadFile = File(...), max_frames: int = 30):
    """
    Analyze video for disaster elements with temporal aggregation.
    
    Args:
        file: Uploaded video file (MP4, AVI)
        max_frames: Maximum frames to analyze
        
    Returns:
        Aggregated segmentation maps
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Save video temporarily
        temp_path = f"/tmp/{file.filename}"
        contents = await file.read()
        
        with open(temp_path, "wb") as f:
            f.write(contents)
        
        # Extract and preprocess frames
        frames_tensor = video_preprocessor.preprocess_video(temp_path, max_frames)
        frames_tensor = frames_tensor.to(device)
        
        # Analyze each frame
        all_predictions = {
            'fire': [],
            'flood': [],
            'collapse': [],
            'human': []
        }
        
        with torch.no_grad():
            for i in range(len(frames_tensor)):
                frame = frames_tensor[i:i+1]
                predictions = model(frame)
                
                for key in all_predictions:
                    all_predictions[key].append(
                        predictions[key].cpu().numpy()[0, 0]
                    )
        
        # Temporal aggregation (average over time)
        aggregated = {
            key: np.mean(np.array(values), axis=0).tolist()
            for key, values in all_predictions.items()
        }
        
        # Calculate overall confidence
        confidence = np.mean([
            model.get_confidence_score({
                k: torch.tensor(all_predictions[k][i]).unsqueeze(0).unsqueeze(0)
                for k in all_predictions
            })
            for i in range(len(frames_tensor))
        ])
        
        metadata = {
            "frames_analyzed": len(frames_tensor),
            "device": str(device),
            "filename": file.filename
        }
        
        return AnalysisResponse(
            fire_map=aggregated['fire'],
            flood_map=aggregated['flood'],
            collapse_map=aggregated['collapse'],
            human_presence_map=aggregated['human'],
            confidence=float(confidence),
            metadata=metadata
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video analysis failed: {str(e)}")
def make_json_safe(obj):
    """Convert numpy types and handle infinity/NaN values."""
    import math
    import numpy as np
    
    if isinstance(obj, dict):
        return {key: make_json_safe(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [make_json_safe(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        val = float(obj)
        if math.isinf(val) or math.isnan(val):
            return 999999  # Replace inf/nan with large number
        return val
    elif isinstance(obj, np.ndarray):
        return make_json_safe(obj.tolist())
    elif isinstance(obj, (float, int)):
        if isinstance(obj, float) and (math.isinf(obj) or math.isnan(obj)):
            return 999999
        return obj
    else:
        return obj
    
@app.post("/grid-from-image")
async def generate_grid(
    file: UploadFile = File(...),
    grid_width: int = 30,
    grid_height: int = 25,
    threshold: float = 0.5
):
    """
    Convert image analysis to RescueAI grid format.
    """
    print("=" * 60)
    print("📥 GRID GENERATION REQUEST")
    print("=" * 60)
    
    if model is None:
        print("❌ Model not loaded")
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Read file
        print(f"📁 File: {file.filename}")
        contents = await file.read()
        print(f"📦 Size: {len(contents)} bytes")
        
        # Decode image
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            print("❌ Failed to decode image")
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        print(f"✅ Image shape: {image.shape}")
        original_size = image.shape[:2]
        
        # Preprocess
        print("🔄 Preprocessing...")
        input_tensor = preprocessor.preprocess_image(image, mode='val')
        input_tensor = input_tensor.to(device)
        print(f"✅ Tensor shape: {input_tensor.shape}")
        
        # Inference
        print("🤖 Running model inference...")
        with torch.no_grad():
            predictions = model(input_tensor)
        print("✅ Predictions generated")
        
        # Convert to numpy
        print("📊 Converting predictions to numpy...")
        fire_map = predictions['fire'].cpu().numpy()[0, 0]
        flood_map = predictions['flood'].cpu().numpy()[0, 0]
        collapse_map = predictions['collapse'].cpu().numpy()[0, 0]
        human_map = predictions['human'].cpu().numpy()[0, 0]
        print("✅ Maps extracted")
        
        # Import converter
        print("🔄 Importing grid converter...")
        from utils.grid_converter import VisionToGridConverter
        print("✅ Converter imported")
        
        # Convert to grid
        print("🗺️ Converting to grid...")
        converter = VisionToGridConverter(
            grid_width=grid_width,
            grid_height=grid_height,
            fire_threshold=threshold,
            flood_threshold=threshold,
            collapse_threshold=threshold,
            human_threshold=threshold + 0.1
        )
        
        scenario = converter.convert(fire_map, flood_map, collapse_map, human_map)
        print("✅ Grid conversion complete")
        
        print(f"📊 Results:")
        print(f"   Victims: {len(scenario['victims'])}")
        print(f"   Drones: {len(scenario['dronePositions'])}")
        print(f"   Type: {scenario['scenarioType']}")
        print("=" * 60)
        
        safe_scenario = make_json_safe(scenario)
        return safe_scenario
        
    except Exception as e:
        print("=" * 60)
        print(f"❌ ERROR: {str(e)}")
        print(f"❌ Type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        print("=" * 60)
        raise HTTPException(status_code=500, detail=f"Grid generation failed: {str(e)}")

@app.get("/model-info")
async def get_model_info():
    """Get information about the loaded model."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            "architecture": "Multi-Head Semantic Segmentation",
            "encoder": "ResNet34",
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "device": str(device),
            "output_size": list(model.output_size),
            "tasks": ["fire", "flood", "collapse", "human_presence"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting RescueAI Vision Server...")
    print(f"Device: {device}")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )