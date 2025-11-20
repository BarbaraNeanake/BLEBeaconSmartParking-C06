from fastapi import FastAPI, UploadFile, HTTPException, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse
import paho.mqtt.client as mqtt
import ssl
import time
import os
import io
import logging
from typing import List, Dict, Optional, Union
import numpy as np
import cv2
from PIL import Image
from pydantic import BaseModel
from huggingface_hub import hf_hub_download
import json
from datetime import datetime
from pathlib import Path

# Import SPARK modules
from spark_engine import create_engine
from spark_db import DatabaseManager
from spark_recon import (
    SlotRegionManager, 
    hogging_detection,
    preprocess_uploaded_image,
    serialize_detections,
    fetch_relevant_slots,
    handle_hogging_violations,
    correct_false_positives
)
from spark_mqtt import MQTTManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="SPARK - Smart Parking System",
)

latest_image: bytes = None

class AlarmPayload(BaseModel):
    sensor_id: int

class ParkingStatusRequest(BaseModel):
    slot_id: int

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Configuration
MQTT_BROKER_HOST = os.environ.get("MQTT_BROKER_HOST", "broker.hivemq.com")
MQTT_BROKER_PORT = 1883  # Public broker uses non-TLS port
SENSOR_DATA_TOPIC = "SPARK_C06/#"

# Parking Slot Debouncing Configuration
DEBOUNCE_WINDOW_MS = 2000  # 2 seconds - wait time before confirming state change
COOLDOWN_MS = 5000         # 5 seconds - minimum time between DB updates
OCCUPIED_KEYWORD = "True"
AVAILABLE_KEYWORD = "False"

# Slot Regions Configuration
SLOT_REGIONS_FILE = os.environ.get("SLOT_REGIONS_FILE", "./slot_regions.json")

# Hugging Face Model Configuration
HF_REPO_ID = os.environ.get("HF_REPO_ID", "danishritonga/SPARK-car-detector")
HF_MODEL_FILENAME = os.environ.get("HF_MODEL_FILENAME", "best_model.npz")
HF_CONFIG_FILENAME = os.environ.get("HF_CONFIG_FILENAME", "config.json")
HF_ANCHOR_FILENAME = os.environ.get("HF_ANCHOR_FILENAME", "anchors.npy")
HF_TOKEN = os.environ.get("HF_TOKEN", None)

# Local cache directory for downloaded models
MODEL_CACHE_DIR = os.environ.get("MODEL_CACHE_DIR", "/app/models_cache")

# Neon DB Configuration
NEON_DB_HOST = os.environ.get("NEON_DB_HOST")
NEON_DB_NAME = os.environ.get("NEON_DB_NAME")
NEON_DB_USER = os.environ.get("NEON_DB_USER")
NEON_DB_PASSWORD = os.environ.get("NEON_DB_PASSWORD")
NEON_DB_PORT = os.environ.get("NEON_DB_PORT", "5432")

# Global managers
inference_engine = None
db_manager = DatabaseManager()
slot_region_manager = SlotRegionManager()
mqtt_manager = MQTTManager(
    debounce_window_ms=DEBOUNCE_WINDOW_MS,
    cooldown_ms=COOLDOWN_MS,
    occupied_keyword=OCCUPIED_KEYWORD,
    available_keyword=AVAILABLE_KEYWORD
)

# MQTT Client Setup
# Callback wrapper to schedule async DB updates
def schedule_db_update(slot_id: str, is_occupied: bool):
    import asyncio
    try:
        loop = asyncio.get_event_loop()
        asyncio.create_task(db_manager.update_slot_status(slot_id, is_occupied))
    except RuntimeError:
        # No event loop running, this shouldn't happen in FastAPI but handle it
        logger.warning(f"⚠️ No event loop available for DB update: slot {slot_id}")

mqtt_manager.set_db_callback(schedule_db_update)
client = mqtt_manager.create_client(
    broker_host=MQTT_BROKER_HOST,
    broker_port=MQTT_BROKER_PORT,
    sensor_topic=SENSOR_DATA_TOPIC
)


@app.on_event("startup")
async def startup_event():
    """Connect to MQTT, initialize database pool, and initialize inference engine when the app starts."""
    global inference_engine
    
    # Load slot regions for hogging detection
    slot_region_manager.load(SLOT_REGIONS_FILE)
    
    # Initialize Neon DB connection pool
    if all([NEON_DB_HOST, NEON_DB_NAME, NEON_DB_USER, NEON_DB_PASSWORD]):
        db_manager.initialize(
            host=NEON_DB_HOST,
            database=NEON_DB_NAME,
            user=NEON_DB_USER,
            password=NEON_DB_PASSWORD,
            port=NEON_DB_PORT,
            min_conn=1,
            max_conn=10
        )
    else:
        logger.warning("⚠️ Neon DB credentials not fully configured, database features disabled")
        logger.warning(f"   Missing: {', '.join([k for k, v in {'NEON_DB_HOST': NEON_DB_HOST, 'NEON_DB_NAME': NEON_DB_NAME, 'NEON_DB_USER': NEON_DB_USER, 'NEON_DB_PASSWORD': NEON_DB_PASSWORD}.items() if not v])}")
    
    # Initialize MQTT
    if not MQTT_BROKER_HOST:
        logger.warning("❌ MQTT broker host not set, using default: broker.hivemq.com")
    
    try:
        client.connect(MQTT_BROKER_HOST, MQTT_BROKER_PORT, 60)
        client.loop_start()
        logger.info(f"✅ MQTT connected to {MQTT_BROKER_HOST}:{MQTT_BROKER_PORT}")
    except Exception as e:
        logger.error(f"❌ Failed to connect to MQTT broker: {e}")
    
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
        
        logger.info(f"📥 Downloading anchors from Hugging Face: {HF_REPO_ID}/{HF_ANCHOR_FILENAME}")
        anchor_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_ANCHOR_FILENAME,
            cache_dir=MODEL_CACHE_DIR,
            token=HF_TOKEN
        )
        
        logger.info(f"✅ Model downloaded to: {model_path}")
        logger.info(f"✅ Config downloaded to: {config_path}")
        logger.info(f"✅ Anchors downloaded to: {anchor_path}")
        
    except Exception as e:
        logger.error(f"❌ Failed to download model from Hugging Face: {e}")
        raise
    
    # Initialize inference engine with downloaded model
    try:
        inference_engine = create_engine(
            model_path=model_path,
            config_path=config_path,
            input_size=416,
            backend="resnet34",
            anchor_path=anchor_path
        )
        logger.info(f"✅ SPARK inference engine initialized with model: {model_path}")
    except Exception as e:
        logger.error(f"❌ Failed to initialize inference engine: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Disconnect from MQTT and close database pool when the app shuts down."""
    # Disconnect MQTT
    client.loop_stop()
    client.disconnect()
    logger.info("✅ MQTT disconnected")
    
    # Close database connection pool
    db_manager.close()

# --- API Endpoints ---
@app.get("/")
async def root():
    return {
        "status": "ok", 
        "message": "SPARK Backend - Car Detection & IoT",
        "features": {
            "car_detection": "/detect - POST image for car detection with hogging detection",
            "parking_slots": "/parking-slots - GET all parking slot statuses (MQTT-based)",
            "slot_regions": "/slot-regions - GET/POST slot region definitions",
            "sensor_logs": "/logs - GET latest sensor messages",
            "alarm_trigger": "/pelanggaran - POST to trigger sensor alarm",
            "health_check": "/health - API health status"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    mqtt_status = "connected" if client.is_connected() else "disconnected"
    db_status = "connected" if db_manager.is_connected() else "not configured"
    
    # Test DB connection if pool exists
    if db_manager.is_connected():
        try:
            test_conn = db_manager.pool.getconn()
            db_manager.pool.putconn(test_conn)
            db_status = "connected"
        except Exception as e:
            db_status = f"error: {str(e)}"
    
    return {
        "status": "healthy", 
        "message": "API is running",
        "mqtt_status": mqtt_status,
        "database_status": db_status,
        "model_loaded": inference_engine is not None,
        "huggingface_repo": HF_REPO_ID,
        "recent_messages": len(mqtt_manager.get_messages()),
        "parking_slots_tracked": len(mqtt_manager.get_parking_slots())
    }

@app.post("/detect")
async def detect_cars(file: UploadFile = File(...)):
    """
    Car detection endpoint using SPARK inference engine.
    Handles hogging detection and database validation.
    """
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Preprocess image
        image_bytes = await file.read()
        image_bgr = preprocess_uploaded_image(image_bytes)
        
        # Run inference
        detections = inference_engine.predict(image_bgr)
        stats = inference_engine.get_stats()
        detections_serializable = serialize_detections(detections)
        
        logger.info(f"Detected {len(detections)} cars in {stats.get('avg_time', 0.0):.3f}s")
        
        # Hogging detection workflow
        slot_regions = slot_region_manager.get()
        if not slot_regions:
            return {"success": True}
        
        # Stage 1: Fetch relevant slots from database
        slot_data, relevant_slot_regions, occupied_slots = await fetch_relevant_slots(
            db_manager, slot_regions
        )
        
        if not slot_data or not relevant_slot_regions:
            return {"success": True}
        
        logger.info(f"🔍 Stage 2: Analyzing {len(relevant_slot_regions)} slots ({len(occupied_slots)} occupied)")
        
        # Stage 3: Run hogging detection
        hogging_analysis = hogging_detection(detections_serializable, relevant_slot_regions)
        logger.info(f"🎯 Stage 3: {hogging_analysis['total_violations']} violations found")
        
        # Handle cases
        if hogging_analysis["total_violations"] > 0:
            await handle_hogging_violations(
                hogging_analysis["violations"], 
                slot_data, 
                db_manager
            )
        
        await correct_false_positives(
            occupied_slots, 
            hogging_analysis.get('slot_occupancy', {}), 
            db_manager
        )
        
        logger.info(f"✅ Case validation complete: {len(occupied_slots)} slots checked")
        
        return {"success": True}
        
    except Exception as e:
        logger.error(f"Car detection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

@app.get("/logs", summary="Get the latest sensor messages")
async def get_logs() -> List[Dict]:
    """Returns the most recent messages stored in memory."""
    return mqtt_manager.get_messages()

@app.post("/findCars")
async def find_cars(file: UploadFile = File(...)):
    """
    Car detection endpoint that returns image with bounding boxes.
    
    Returns:
        Image with bounding boxes drawn around detected cars
    """
    try:
        # More lenient content type check
        if file.content_type and not file.content_type.startswith('image/'):
            logger.warning(f"Content type is {file.content_type}, proceeding anyway")
    
    except Exception as validation_error:
        logger.error(f"Validation error: {validation_error}")
        raise HTTPException(status_code=400, detail=f"Invalid file: {str(validation_error)}")
    
    try:
        # Preprocess image
        image_bytes = await file.read()
        image_bgr = preprocess_uploaded_image(image_bytes)
        
        # Run inference
        detections = inference_engine.predict(image_bgr)
        
        # Draw bounding boxes on image
        output_image = image_bgr.copy()
        for det in detections:
            bbox = det.get('bbox', [])
            confidence = det.get('confidence', 0)
            class_name = det.get('class_name', 'unknown')
            
            if len(bbox) == 4:
                x1, y1, x2, y2 = map(int, bbox)
                
                # Draw rectangle
                cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw label
                label = f"{class_name}: {confidence:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(output_image, (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0], y1), (0, 255, 0), -1)
                cv2.putText(output_image, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # Convert BGR to RGB for PIL
        output_image_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(output_image_rgb)
        
        # Convert to bytes
        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format='JPEG')
        img_byte_arr.seek(0)
        
        logger.info(f"Detected {len(detections)} cars with bounding boxes drawn")
        
        return Response(content=img_byte_arr.getvalue(), media_type="image/jpeg")
        
    except Exception as e:
        logger.error(f"Car detection with bounding boxes failed: {e}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

@app.post("/parkingStatus")
async def get_parking_status(request: ParkingStatusRequest):
    slot_status = await db_manager.get_slot_status(request.slot_id)
    
    if slot_status is None:
        return None
    
    return slot_status["status"]

@app.post("/pelanggaran")
async def trigger_buzzer(sensor_payload: AlarmPayload):
    slotID = sensor_payload.sensor_id
    alarm_topic = f"SPARK_C06/ping/{slotID}"
    payload_to_send = "True"

    result = client.publish(alarm_topic, payload_to_send)
    
    # Use the globally defined MQTT client to publish the message
    if result.rc == mqtt.MQTT_ERR_SUCCESS:
        print(f"⬆️  [Backend] Alarm signal sent to topic: {alarm_topic}")
        return {"status": "success", "topic": alarm_topic, "message": f"Alarm signal sent to sensor {slotID}."}
    else:
        print(f"❌ [Backend] Failed to send alarm signal to topic: {alarm_topic}")
        return {"status": "error", "message": "Failed to send MQTT message."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)