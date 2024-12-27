# main.py (for Render)

import os
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import savgol_filter, butter, filtfilt
import io
from pydantic import BaseModel
from typing import List, Dict
import soundfile as sf

app = FastAPI(title="Uroflowmetry API")

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionResponse(BaseModel):
    flow_rate: List[float]
    parameters: Dict[str, str]
    time: List[float]

# Add a simple HTML form for testing
@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
        <head>
            <title>Uroflowmetry API</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .container { max-width: 800px; margin: 0 auto; }
                .form-group { margin-bottom: 20px; }
                .btn { padding: 10px 20px; background-color: #007bff; color: white; border: none; cursor: pointer; }
                .response { margin-top: 20px; padding: 20px; background-color: #f8f9fa; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Uroflowmetry API Test Interface</h1>
                <form action="/predict/" method="post" enctype="multipart/form-data">
                    <div class="form-group">
                        <label>Select audio file:</label><br>
                        <input type="file" name="file" accept=".wav,.mp3">
                    </div>
                    <button type="submit" class="btn">Process Audio</button>
                </form>
                <div class="response" id="response"></div>
            </div>
        </body>
    </html>
    """

# Your existing functions here
def calculate_rms(signal, frame_length, hop_length):
    """Calculate RMS manually without using librosa"""
    try:
        signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
        pad_length = frame_length - 1
        padded_signal = np.pad(signal, (pad_length // 2, pad_length - pad_length // 2))
        n_frames = 1 + (len(signal) - frame_length) // hop_length
        frames = np.zeros((n_frames, frame_length))
        
        for i in range(n_frames):
            start = i * hop_length
            frames[i] = padded_signal[start:start + frame_length]
        
        rms = np.sqrt(np.mean(frames ** 2 + 1e-10, axis=1))
        rms = np.nan_to_num(rms, nan=0.0, posinf=0.0, neginf=0.0)
        return rms
    except Exception as e:
        raise ValueError(f"RMS calculation failed: {str(e)}")

def apply_bandpass_filter(signal, sr, lowcut=20, highcut=2000):
    """Apply bandpass filter to the audio signal"""
    try:
        signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
        nyquist = sr / 2
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(4, [low, high], btype='band')
        filtered = filtfilt(b, a, signal)
        filtered = np.nan_to_num(filtered, nan=0.0, posinf=0.0, neginf=0.0)
        return filtered
    except Exception as e:
        raise ValueError(f"Bandpass filter failed: {str(e)}")

def process_audio(audio_bytes):
    """Process audio file and extract features"""
    try:
        with io.BytesIO(audio_bytes) as audio_io:
            y, sr = sf.read(audio_io)
            if len(y.shape) > 1:
                y = np.mean(y, axis=1)
        
        if len(y) == 0:
            raise ValueError("No audio data found")
            
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        y_filtered = apply_bandpass_filter(y, sr)
        
        frame_length = int(sr * 0.1)
        hop_length = int(frame_length / 2)
        rms = calculate_rms(y_filtered, frame_length, hop_length)
        
        if len(rms) == 0:
            raise ValueError("No RMS values calculated")
        
        if len(rms) > 3:
            window_length = min(15, len(rms)-2 if len(rms) % 2 == 0 else len(rms)-1)
            if window_length > 2:
                rms_smoothed = savgol_filter(rms, window_length, 3)
            else:
                rms_smoothed = rms
        else:
            rms_smoothed = rms
            
        rms_smoothed = np.nan_to_num(rms_smoothed, nan=0.0, posinf=0.0, neginf=0.0)
        rms_smoothed = rms_smoothed + 1e-10
        
        scaler = MinMaxScaler(feature_range=(0, 50))
        flow_rate = scaler.fit_transform(rms_smoothed.reshape(-1, 1)).flatten()
        
        duration = len(y) / sr
        time = np.linspace(0, duration, len(flow_rate))
        
        flow_rate = np.nan_to_num(flow_rate, nan=0.0, posinf=50.0, neginf=0.0)
        time = np.nan_to_num(time, nan=0.0, posinf=duration, neginf=0.0)
        
        if not np.all(np.isfinite(flow_rate)) or not np.all(np.isfinite(time)):
            raise ValueError("Invalid values in processed data")
            
        return flow_rate, time
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audio processing failed: {str(e)}")

def calculate_parameters(time, flow_rate):
    """Calculate uroflowmetry parameters"""
    try:
        flow_rate = np.nan_to_num(flow_rate, nan=0.0, posinf=50.0, neginf=0.0)
        time = np.nan_to_num(time, nan=0.0)
        
        max_flow = float(np.max(flow_rate))
        avg_flow = float(np.mean(flow_rate))
        voiding_duration = float(time[-1])
        voided_volume = float(np.trapz(flow_rate, time))
        time_to_max = float(time[np.argmax(flow_rate)])
        
        idx_2s = np.where(time >= 2.0)[0]
        flow_at_2s = float(flow_rate[idx_2s[0]]) if len(idx_2s) > 0 else 0.0
        acceleration = flow_at_2s / 2.0 if flow_at_2s > 0 else 0.0
        
        max_flow = min(max_flow, 50.0)
        avg_flow = min(avg_flow, 50.0)
        voiding_duration = min(voiding_duration, 300.0)
        voided_volume = min(voided_volume, 1000.0)
        
        return {
            "Maximum Flow Rate": f"{max_flow:.2f} ml/s",
            "Average Flow Rate": f"{avg_flow:.2f} ml/s",
            "Voiding Duration": f"{voiding_duration:.2f} s",
            "Voided Volume": f"{voided_volume:.2f} ml",
            "Time to Max Flow": f"{time_to_max:.2f} s",
            "Flow at 2 Seconds": f"{flow_at_2s:.2f} ml/s",
            "Acceleration": f"{acceleration:.2f} ml/s²"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Parameter calculation failed: {str(e)}")

@app.post("/predict/", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """Handle audio file upload and generate predictions"""
    try:
        if not file:
            raise HTTPException(status_code=400, detail="No file uploaded")
        
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file")
        
        flow_rate, time = process_audio(contents)
        
        flow_rate = np.nan_to_num(flow_rate, nan=0.0, posinf=50.0, neginf=0.0)
        time = np.nan_to_num(time, nan=0.0)
        
        if len(flow_rate) == 0 or len(time) == 0:
            raise HTTPException(status_code=500, detail="No data generated from audio processing")
        
        parameters = calculate_parameters(time, flow_rate)
        
        return PredictionResponse(
            flow_rate=flow_rate.tolist(),
            parameters=parameters,
            time=time.tolist()
        )
        
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": str(np.datetime64('now'))
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)