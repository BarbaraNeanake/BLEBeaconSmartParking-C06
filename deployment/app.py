from fastapi import FastAPI, UploadFile, HTTPException, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse
import paho.mqtt.client as mqtt
import ssl
import time
import os
import io
import logging
from typing import List, Dict, Optional
import numpy as np
import cv2
from PIL import Image
from pydantic import BaseModel
from huggingface_hub import hf_hub_download
import json
from datetime import datetime
from pathlib import Path

# Import SPARK inference engine
from spark_engine import create_engine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="SPARK Backend - Car Detection & IoT",
    description="Combined car detection API and IoT sensor backend",
    version="1.0.0"
)

latest_image: bytes = None

class AlarmPayload(BaseModel):
    sensor_id: str

class TestDataPayload(BaseModel):
    data: str

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# --- Configuration (from environment variables or Hugging Face Secrets) ---
MQTT_BROKER_HOST = os.environ.get("MQTT_BROKER_HOST")
MQTT_USERNAME = os.environ.get("MQTT_USERNAME")
MQTT_PASSWORD = os.environ.get("MQTT_PASSWORD")
MQTT_BROKER_PORT = 8883
SENSOR_DATA_TOPIC = "#"

# Hugging Face Model Configuration
HF_REPO_ID = os.environ.get("HF_REPO_ID", "danishritonga/SPARK-car-detector")
HF_MODEL_FILENAME = os.environ.get("HF_MODEL_FILENAME", "best_model.npz")
HF_CONFIG_FILENAME = os.environ.get("HF_CONFIG_FILENAME", "config.json")
HF_TOKEN = os.environ.get("HF_TOKEN", None)

# Local cache directory for downloaded models
MODEL_CACHE_DIR = os.environ.get("MODEL_CACHE_DIR", "/app/models_cache")

# JSON storage file for post-test endpoint
# Use /data directory for persistent storage in Hugging Face Spaces
JSON_STORAGE_FILE = os.environ.get("JSON_STORAGE_FILE", "/data/stored_data.json")

# Global inference engine
inference_engine = None

# In-memory list to store the last 20 messages
g_messages: List[Dict] = []

# --- MQTT Callbacks ---
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("✅ [Backend] Connected to HiveMQ Cloud!")
        client.subscribe(SENSOR_DATA_TOPIC)
        print(f"👂 [Backend] Subscribed to topic: {SENSOR_DATA_TOPIC}")
    else:
        print(f"❌ [Backend] Failed to connect, return code {rc}")

def on_message(client, userdata, msg):
    """
    This function stores the incoming message in the g_messages list.
    """
    device_id = msg.topic.split('/')[-1]
    payload = msg.payload.decode()
    
    print(f"📩 [Backend] Received from '{device_id}': {payload}")
    
    log_entry = {
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        "device_id": device_id,
        "payload": payload
    }
    
    # Add the newest message to the start of the list
    g_messages.insert(0, log_entry)
    
    # Keep the list trimmed to the last 20 messages
    if len(g_messages) > 20:
        g_messages.pop()

# --- MQTT Client Setup ---
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1, client_id="fastapi-test-backend")
client.on_connect = on_connect
client.on_message = on_message

# Set credentials and enable TLS for a secure connection
client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
client.tls_set(tls_version=ssl.PROTOCOL_TLS)

@app.on_event("startup")
async def startup_event():
    """Connect to MQTT and initialize inference engine when the app starts."""
    global inference_engine
    
    # Ensure persistent data directory exists
    data_dir = Path(JSON_STORAGE_FILE).parent
    os.makedirs(data_dir, exist_ok=True)
    logger.info(f"✅ Data directory ready: {data_dir}")
    
    # Initialize MQTT
    if not all([MQTT_BROKER_HOST, MQTT_USERNAME, MQTT_PASSWORD]):
        logger.warning("❌ MQTT credentials not set in environment variables!")
    else:
        client.connect(MQTT_BROKER_HOST, MQTT_BROKER_PORT, 60)
        client.loop_start()
        logger.info("✅ MQTT connected")
    
    # Download model from Hugging Face
    try:
        os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
        
        logger.info(f"📥 Downloading model from Hugging Face: {HF_REPO_ID}/{HF_MODEL_FILENAME}")
        model_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_MODEL_FILENAME,
            cache_dir=MODEL_CACHE_DIR,
            token=HF_TOKEN
        )
        
        logger.info(f"📥 Downloading config from Hugging Face: {HF_REPO_ID}/{HF_CONFIG_FILENAME}")
        config_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_CONFIG_FILENAME,
            cache_dir=MODEL_CACHE_DIR,
            token=HF_TOKEN
        )
        
        logger.info(f"✅ Model downloaded to: {model_path}")
        logger.info(f"✅ Config downloaded to: {config_path}")
        
    except Exception as e:
        logger.error(f"❌ Failed to download model from Hugging Face: {e}")
        raise
    
    # Initialize inference engine with downloaded model
    try:
        inference_engine = create_engine(
            model_path=model_path,
            config_path=config_path,
            input_size=416,
            backend="resnet34"
        )
        logger.info(f"✅ SPARK inference engine initialized with model: {model_path}")
    except Exception as e:
        logger.error(f"❌ Failed to initialize inference engine: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Disconnect from MQTT when the app shuts down."""
    client.loop_stop()
    client.disconnect()

# --- API Endpoints ---
@app.get("/")
async def root():
    return {
        "status": "ok", 
        "message": "SPARK Backend - Car Detection & IoT",
        "version": "1.0.0",
        "features": {
            "car_detection": "/detect - POST image for car detection",
            "sensor_logs": "/logs - GET latest sensor messages",
            "alarm_trigger": "/pelanggaran - POST to trigger sensor alarm",
            "health_check": "/health - API health status"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    mqtt_status = "connected" if client.is_connected() else "disconnected"
    
    return {
        "status": "healthy", 
        "message": "API is running",
        "mqtt_status": mqtt_status,
        "model_loaded": inference_engine is not None,
        "huggingface_repo": HF_REPO_ID,
        "recent_messages": len(g_messages)
    }

@app.post("/detect")
async def detect_cars(file: UploadFile = File(...)):
    """
    Car detection endpoint using SPARK inference engine
    """
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read and convert image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to BGR format for OpenCV (model expects BGR)
        image_array = np.array(image, dtype=np.uint8)
        image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        
        # Run inference with SPARK engine
        start_time = time.time()
        detections = inference_engine.predict(image_bgr)
        stats = inference_engine.get_stats()
        
        # Convert NumPy types to native Python types for JSON serialization
        detections_serializable = []
        for det in detections:
            det_dict = {
                "bbox": [float(x) for x in det.get('bbox', [])],
                "confidence": float(det.get('confidence', 0)),
                "class": int(det.get('class', 0)),
                "class_name": str(det.get('class_name', 'unknown'))
            }
            detections_serializable.append(det_dict)
        
        result = {
            "success": True,
            "detections": detections_serializable,
            "inference_time": float(stats.get("avg_time", 0.0)),
            "image_shape": [int(x) for x in image_array.shape],
            "num_detections": len(detections_serializable)
        }
        
        logger.info(f"Detected {len(detections)} cars in {stats.get('avg_time', 0.0):.3f}s")
        
        return result
        
    except Exception as e:
        logger.error(f"Car detection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

@app.get("/logs", summary="Get the latest sensor messages")
async def get_logs() -> List[Dict]:
    """Returns the most recent messages stored in memory."""
    return g_messages

@app.post("/pelanggaran")
async def trigger_buzzer(sensor_payload: AlarmPayload):
    """
    Endpoint to trigger a buzzer on a specific sensor device.
    Publishes a message to the corresponding MQTT topic.
    """
    sensor_id = sensor_payload.sensor_id
    alarm_topic = f"violation/{sensor_id}"
    payload_to_send = "True"

    result = client.publish(alarm_topic, payload_to_send)
    
    # Use the globally defined MQTT client to publish the message
    if result.rc == mqtt.MQTT_ERR_SUCCESS:
        print(f"⬆️  [Backend] Alarm signal sent to topic: {alarm_topic}")
        return {"status": "success", "topic": alarm_topic, "message": f"Alarm signal sent to sensor {sensor_id}."}
    else:
        print(f"❌ [Backend] Failed to send alarm signal to topic: {alarm_topic}")
        return {"status": "error", "message": "Failed to send MQTT message."}

@app.post("/post-test")
async def store_test_data(payload: TestDataPayload):
    """
    Store string data to a JSON file. Appends to existing data if file exists.
    """
    try:
        file_path = Path(JSON_STORAGE_FILE)
        
        # Add timestamp to the payload
        entry = {
            "timestamp": datetime.now().isoformat(),
            "data": payload.data
        }
        
        # Read existing data if file exists
        if file_path.exists():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:  # Check if file is not empty
                        existing_data = json.loads(content)
                        if not isinstance(existing_data, list):
                            existing_data = [existing_data]
                    else:
                        # File exists but is empty
                        existing_data = []
            except json.JSONDecodeError:
                # File is corrupted, start fresh
                logger.warning(f"⚠️ JSON file corrupted, starting fresh")
                existing_data = []
        else:
            existing_data = []
        
        # Append new entry
        existing_data.append(entry)
        
        # Write back to file
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Data stored successfully. Total entries: {len(existing_data)}")
        
        return {
            "status": "success",
            "message": "Data stored successfully",
            "total_entries": len(existing_data),
            "file": str(file_path)
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to store data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to store data: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)