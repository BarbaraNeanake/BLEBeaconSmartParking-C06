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

# Import SPARK modules
from spark_engine import create_engine
from spark_db import DatabaseManager
from spark_recon import SlotRegionManager, hogging_detection
from spark_mqtt import MQTTManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="SPARK - Smart Parking System",
)

latest_image: bytes = None

class AlarmPayload(BaseModel):
    sensor_id: str

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# --- Configuration (from environment variables or Hugging Face Secrets) ---
MQTT_BROKER_HOST = os.environ.get("MQTT_BROKER_HOST", "broker.hivemq.com")
MQTT_BROKER_PORT = 1883  # Public broker uses non-TLS port
SENSOR_DATA_TOPIC = "SPARK_C06/#"

# Parking Slot Debouncing Configuration
DEBOUNCE_WINDOW_MS = 2000  # 2 seconds - wait time before confirming state change
COOLDOWN_MS = 5000         # 5 seconds - minimum time between DB updates
OCCUPIED_KEYWORD = "True"  # Keyword for occupied slot
AVAILABLE_KEYWORD = "False"  # Keyword for available slot

# Slot Regions Configuration (for hogging detection)
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

# --- MQTT Client Setup ---
mqtt_manager.set_db_callback(lambda slot_id, is_occupied: db_manager.update_slot_status(slot_id, is_occupied))
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
        "version": "1.0.0",
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
    Includes slot hogging detection if slot regions are configured.
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
        
        # Three-stage hogging detection workflow
        hogging_analysis = None
        db_status = None
        slot_data = None
        verification_results = None
        slot_regions = slot_region_manager.get()
        
        if slot_regions:
            # STAGE 1: Read ONLY occupied slots + neighbors from database
            slot_data = await db_manager.get_occupied_slots_with_neighbors()
            
            if not slot_data:
                logger.info("ℹ️ No occupied slots found, skipping hogging detection")
            else:
                occupied_count = sum(1 for s in slot_data.values() if s["query_reason"] == "occupied")
                neighbor_count = sum(1 for s in slot_data.values() if s["query_reason"] == "neighbor")
                logger.info(f"📊 Stage 1 (Optimized): Retrieved {occupied_count} occupied + {neighbor_count} neighbor slots")
            
            # Extract just status for backward compatibility
            db_status = {slot_id: data["status"] for slot_id, data in slot_data.items()}
            
            # STAGE 2: Filter slot regions to only those we queried from DB
            relevant_slot_regions = {
                slot_id: slot_regions[slot_id] 
                for slot_id in slot_data.keys() 
                if slot_id in slot_regions
            }
            
            occupied_slots = [slot_id for slot_id, data in slot_data.items() if data["status"] == 'occupied']
            logger.info(f"🔍 Stage 2: Analyzing {len(relevant_slot_regions)} relevant slots ({len(occupied_slots)} occupied)")
            
            # Build verification report
            verification_results = {
                "db_occupied_slots": occupied_slots,
                "total_db_occupied": len(occupied_slots),
                "total_analyzed_slots": len(relevant_slot_regions),
                "total_detections": len(detections_serializable),
                "optimization": f"Queried {len(slot_data)} slots instead of all slots"
            }
            
            # STAGE 3: Run hogging detection ONLY on relevant slots
            # Only analyze detections if we have both relevant slot regions and detections
            if len(detections_serializable) > 0 and len(relevant_slot_regions) > 0:
                hogging_analysis = hogging_detection(detections_serializable, relevant_slot_regions)
                
                logger.info(f"🎯 Stage 3: Hogging analysis complete - {hogging_analysis['total_violations']} violations found")
                
                if hogging_analysis["total_violations"] > 0:
                    logger.warning(f"⚠️ Detected {hogging_analysis['total_violations']} slot hogging violations!")
                    for violation in hogging_analysis["violations"]:
                        slots_str = ", ".join(violation['occupied_slots'])
                        logger.warning(f"   🚗 Car (conf: {violation['confidence']:.2f}) hogging slots: {slots_str}")
                        
                        # Find a userid from the occupied slots (skip userid=0)
                        violator_userid = None
                        violator_slot = None
                        for slot_id in violation['occupied_slots']:
                            slot_info = slot_data.get(slot_id, {})
                            userid = slot_info.get('userid', 0)
                            if userid > 0:
                                violator_userid = userid
                                violator_slot = slot_id
                                break
                        
                        # Insert violation into pelanggaran table (checks for duplicates automatically)
                        if violator_userid and violator_slot:
                            inserted = await db_manager.insert_pelanggaran(
                                userid=violator_userid,
                                nomor=violator_slot,
                                jenis_pelanggaran="Slot Hogging"
                            )
                            if inserted:
                                logger.info(f"📝 Logged NEW violation for userid={violator_userid} at slot {violator_slot}")
                            else:
                                logger.info(f"⏭️ Violation already exists for userid={violator_userid} at slot {violator_slot}")
                        else:
                            logger.warning(f"⚠️ No valid userid found in hogging slots: {slots_str}")
            else:
                logger.info("ℹ️ No detections or no relevant slots, skipping hogging analysis")
        
        logger.info(f"Detected {len(detections)} cars in {stats.get('avg_time', 0.0):.3f}s")
        
        # Return simple success response
        return {"success": True}
        
    except Exception as e:
        logger.error(f"Car detection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

@app.get("/logs", summary="Get the latest sensor messages")
async def get_logs() -> List[Dict]:
    """Returns the most recent messages stored in memory."""
    return mqtt_manager.get_messages()

@app.get("/parking-slots", summary="Get current parking slot occupancy status")
async def get_parking_slots_api():
    """
    Returns the current occupancy status of all parking slots with debouncing info.
    Data is received from MQTT topic: SPARK_C06/isOccupied/{slotID}
    """
    parking_slots = mqtt_manager.get_parking_slots()
    return {
        "status": "success",
        "total_slots": len(parking_slots),
        "slots": [slot.to_dict() for slot in parking_slots.values()],
        "summary": {
            "occupied": sum(1 for slot in parking_slots.values() if slot.is_occupied),
            "available": sum(1 for slot in parking_slots.values() if not slot.is_occupied),
            "pending": sum(1 for slot in parking_slots.values() if slot.pending_value is not None)
        },
        "config": {
            "debounce_window_ms": DEBOUNCE_WINDOW_MS,
            "cooldown_ms": COOLDOWN_MS,
            "occupied_keyword": OCCUPIED_KEYWORD,
            "available_keyword": AVAILABLE_KEYWORD
        },
        "performance": {
            "total_messages": sum(slot.message_count for slot in parking_slots.values())
        }
    }

@app.get("/parking-slots/{slot_id}", summary="Get specific parking slot status")
async def get_parking_slot(slot_id: str):
    """
    Returns the occupancy status and debouncing info of a specific parking slot.
    """
    parking_slots = mqtt_manager.get_parking_slots()
    if slot_id not in parking_slots:
        raise HTTPException(status_code=404, detail=f"Slot {slot_id} not found")
    
    slot = parking_slots[slot_id]
    return {
        "status": "success",
        "slot": slot.to_dict()
    }

@app.get("/parkingStatus/{slot_id}", summary="Get parking slot occupancy status from database")
async def get_parking_status(slot_id: str):
    """
    Returns the current occupancy status of a specific parking slot from the database.
    This reflects the actual database state updated via MQTT messages.
    """
    slot_status = await db_manager.get_slot_status(slot_id)
    
    if slot_status is None:
        raise HTTPException(status_code=404, detail=f"Slot {slot_id} not found in database")
    
    return {
        "slot_id": slot_status["nomor"],
        "status": slot_status["status"],
        "is_occupied": slot_status["status"] == "occupied",
        "userid": slot_status["userid"]
    }

@app.post("/pelanggaran")
async def trigger_buzzer(sensor_payload: AlarmPayload):
    """
    Endpoint to trigger a buzzer on a specific sensor device.
    Publishes a message to the corresponding MQTT topic.
    """
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



@app.post("/slot-regions")
async def update_slot_regions(regions: Dict[str, Dict]):
    """
    Update parking slot region definitions.
    
    Expected payload:
    {
        "A01": {
            "corners": [[100, 200], [300, 200], [300, 400], [100, 400]],
            "name": "Slot A01"
        },
        "A02": { ... }
    }
    """
    try:
        # Validate structure
        for slot_id, region_data in regions.items():
            if 'corners' not in region_data:
                raise HTTPException(status_code=400, detail=f"Slot {slot_id} missing 'corners' field")
            if len(region_data['corners']) < 3:
                raise HTTPException(status_code=400, detail=f"Slot {slot_id} must have at least 3 corners")
        
        # Save to file using slot region manager
        slot_region_manager.save(regions, SLOT_REGIONS_FILE)
        
        return {
            "status": "success",
            "message": f"Updated {len(regions)} slot regions",
            "slots": list(regions.keys()),
            "file": SLOT_REGIONS_FILE
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to update slot regions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/slot-regions")
async def get_slot_regions_api():
    """
    Get current parking slot region definitions.
    """
    slot_regions = slot_region_manager.get()
    return {
        "status": "success",
        "regions": slot_regions,
        "total_slots": len(slot_regions),
        "detection_method": "x_axis_bounds",
        "config": {
            "file": SLOT_REGIONS_FILE,
            "note": "Using X-axis bounds method for hogging detection. Can be changed to other methods later."
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)