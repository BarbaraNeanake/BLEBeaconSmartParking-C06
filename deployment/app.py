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
import psycopg2
from psycopg2 import pool

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
MQTT_BROKER_HOST = os.environ.get("MQTT_BROKER_HOST", "broker.hivemq.com")
MQTT_BROKER_PORT = 1883  # Public broker uses non-TLS port
SENSOR_DATA_TOPIC = "SPARK_C06/#"

# Parking Slot Debouncing Configuration
DEBOUNCE_WINDOW_MS = 2000  # 2 seconds - wait time before confirming state change
COOLDOWN_MS = 5000         # 5 seconds - minimum time between DB updates
OCCUPIED_KEYWORD = "True"  # Keyword for occupied slot
AVAILABLE_KEYWORD = "False"  # Keyword for available slot

# Slot Regions Configuration (for hogging detection)
SLOT_REGIONS_FILE = os.environ.get("SLOT_REGIONS_FILE", "/data/slot_regions.json")

# Hugging Face Model Configuration
HF_REPO_ID = os.environ.get("HF_REPO_ID", "danishritonga/SPARK-car-detector")
HF_MODEL_FILENAME = os.environ.get("HF_MODEL_FILENAME", "best_model.npz")
HF_CONFIG_FILENAME = os.environ.get("HF_CONFIG_FILENAME", "config.json")
HF_TOKEN = os.environ.get("HF_TOKEN", None)

# Local cache directory for downloaded models
MODEL_CACHE_DIR = os.environ.get("MODEL_CACHE_DIR", "/app/models_cache")

# JSON storage file for post-test endpoint
JSON_STORAGE_FILE = os.environ.get("JSON_STORAGE_FILE", "/data/stored_data.json")

# Neon DB Configuration
NEON_DB_HOST = os.environ.get("NEON_DB_HOST")
NEON_DB_NAME = os.environ.get("NEON_DB_NAME")
NEON_DB_USER = os.environ.get("NEON_DB_USER")
NEON_DB_PASSWORD = os.environ.get("NEON_DB_PASSWORD")
NEON_DB_PORT = os.environ.get("NEON_DB_PORT", "5432")

# Global inference engine
inference_engine = None

# Global database connection pool
db_pool = None

# In-memory list to store the last 20 messages
g_messages: List[Dict] = []

# --- Parking Slot Data Structure ---
class ParkingSlot:
    def __init__(self, slot_id: str):
        self.slot_id = slot_id
        self.is_occupied = False
        self.last_value = None          # Last sensor reading (True/False)
        self.last_change_time = None    # When value last changed
        self.last_db_update = None      # When we last wrote to DB
        self.pending_value = None       # Value waiting for debounce
        self.message_count = 0
    
    def to_dict(self):
        return {
            "slot_id": self.slot_id,
            "is_occupied": self.is_occupied,
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "last_update": self.last_change_time.isoformat() if self.last_change_time else None,
            "message_count": self.message_count,
            "pending_change": self.pending_value if self.pending_value is not None else "none"
        }

# In-memory storage for parking slot objects
parking_slots: Dict[str, ParkingSlot] = {}

# In-memory storage for slot region definitions (for hogging detection)
slot_regions: Dict[str, Dict] = {}

# --- Slot Region Management Functions ---
def load_slot_regions():
    """
    Load parking slot region definitions from JSON file.
    
    Expected format:
    {
        "A01": {
            "corners": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
            "name": "Slot A01"
        },
        "A02": { ... }
    }
    """
    global slot_regions
    
    try:
        file_path = Path(SLOT_REGIONS_FILE)
        if file_path.exists():
            with open(file_path, 'r') as f:
                slot_regions = json.load(f)
            logger.info(f"✅ Loaded {len(slot_regions)} slot regions from {file_path}")
        else:
            logger.warning(f"⚠️ Slot regions file not found: {file_path}")
            slot_regions = {}
    except Exception as e:
        logger.error(f"❌ Failed to load slot regions: {e}")
        slot_regions = {}


def hogging_detection(detections: List[Dict], slot_regions: Dict[str, Dict]) -> Dict:
    """
    Detect if any car is hogging multiple parking slots using X-axis bounds.
    Simple method: checks if car's x_min and x_max fall outside any single slot's x bounds.
    
    Args:
        detections: List of detection dicts with 'bbox' key [x_min, y_min, x_max, y_max]
        slot_regions: Dict of {slot_id: {corners: [[x,y], ...], name: ...}}
    
    Returns:
        Dict with hogging analysis:
        {
            "violations": [...],
            "slot_occupancy": {...},
            "total_violations": int,
            "method": "x_axis_bounds"
        }
    
    Note: This is a simple method with limitations:
    - Ignores Y-axis (depth/row information)
    - May give false positives for multi-row parking
    - Best for single-row, side-by-side parking slots
    """
    violations = []
    slot_occupancy = {}
    
    # Initialize slot occupancy
    for slot_id in slot_regions.keys():
        slot_occupancy[slot_id] = {
            "is_occupied": False,
            "car_count": 0,
            "cars": [],
            "is_violation": False
        }
    
    # Calculate X-bounds for each slot
    slot_x_bounds = {}
    for slot_id, slot_data in slot_regions.items():
        corners = slot_data.get('corners', [])
        if len(corners) < 3:
            logger.warning(f"⚠️ Slot {slot_id}: Invalid corners (need at least 3)")
            continue
        
        x_coords = [c[0] for c in corners]
        slot_x_bounds[slot_id] = {
            "x_min": min(x_coords),
            "x_max": max(x_coords)
        }
    
    # Check each detected car
    for detection in detections:
        car_bbox = detection.get('bbox', [])
        if len(car_bbox) != 4:
            continue
        
        x_min_car, y_min_car, x_max_car, y_max_car = car_bbox
        occupied_slots = []
        slot_overlaps = {}
        
        # Check which slots this car overlaps with (X-axis only)
        for slot_id, bounds in slot_x_bounds.items():
            x_min_slot = bounds["x_min"]
            x_max_slot = bounds["x_max"]
            
            # Check if car's X-range overlaps with slot's X-range
            # Overlap exists if: NOT (car completely left of slot OR car completely right of slot)
            x_overlap = not (x_max_car < x_min_slot or x_min_car > x_max_slot)
            
            if x_overlap:
                # Calculate overlap amount (for debugging/analysis)
                overlap_start = max(x_min_car, x_min_slot)
                overlap_end = min(x_max_car, x_max_slot)
                overlap_width = overlap_end - overlap_start
                car_width = x_max_car - x_min_car
                overlap_percentage = overlap_width / car_width if car_width > 0 else 0
                
                occupied_slots.append(slot_id)
                slot_overlaps[slot_id] = {
                    "overlap_percentage": float(overlap_percentage),
                    "car_x_range": [float(x_min_car), float(x_max_car)],
                    "slot_x_range": [float(x_min_slot), float(x_max_slot)]
                }
                
                # Update slot occupancy
                slot_occupancy[slot_id]["is_occupied"] = True
                slot_occupancy[slot_id]["car_count"] += 1
                slot_occupancy[slot_id]["cars"].append({
                    "bbox": car_bbox,
                    "confidence": float(detection.get('confidence', 0)),
                    "overlap_percentage": float(overlap_percentage)
                })
        
        # If car occupies multiple slots, it's a violation (hogging)
        if len(occupied_slots) > 1:
            violation = {
                "car_bbox": car_bbox,
                "confidence": float(detection.get('confidence', 0)),
                "occupied_slots": occupied_slots,
                "num_slots": len(occupied_slots),
                "slot_overlaps": slot_overlaps,
                "violation_type": "slot_hogging",
                "severity": "high" if len(occupied_slots) >= 3 else "medium"
            }
            violations.append(violation)
            
            # Mark all affected slots as violations
            for slot_id in occupied_slots:
                slot_occupancy[slot_id]["is_violation"] = True
    
    return {
        "violations": violations,
        "slot_occupancy": slot_occupancy,
        "total_violations": len(violations),
        "affected_slots": [slot_id for slot_id, data in slot_occupancy.items() if data["is_violation"]],
        "total_slots": len(slot_regions),
        "method": "x_axis_bounds",
        "note": "Simple X-axis method. Limitations: ignores Y-axis, may have false positives in multi-row parking."
    }

# --- Database Helper Functions ---
async def trigger_db_update(slot: ParkingSlot, is_occupied: bool):
    """
    Placeholder for database update with rate limiting.
    
    Args:
        slot: The ParkingSlot object
        is_occupied: True if slot is occupied, False if available
    """
    current_time = datetime.now()
    
    # Rate limiting: Don't spam DB
    if slot.last_db_update:
        elapsed_ms = (current_time - slot.last_db_update).total_seconds() * 1000
        if elapsed_ms < 500:  # Max 2 updates per second per slot
            logger.debug(f"🕐 DB update throttled for slot {slot.slot_id}")
            return
    
    slot.last_db_update = current_time
    logger.info(f"🔄 [PLACEHOLDER] Would update DB: Slot {slot.slot_id} → {'OCCUPIED' if is_occupied else 'AVAILABLE'}")


async def get_occupancy_status() -> Dict[str, Dict[str, any]]:
    """
    Retrieve all parking slot statuses and userids from Neon DB.
    
    Returns:
        Dict of {slot_id: {"status": str, "userid": int}}
        Returns empty dict if DB is not configured or query fails
    """
    global db_pool
    
    # Check if DB pool is initialized
    if db_pool is None:
        logger.warning("⚠️ DB pool not initialized, cannot fetch slot status")
        return {}
    
    try:
        # Get connection from pool
        conn = db_pool.getconn()
        cursor = conn.cursor()
        
        # Query all parking slots with their status and userid
        cursor.execute(
            "SELECT nomor, status, userid FROM parking_slots ORDER BY nomor"
        )
        rows = cursor.fetchall()
        cursor.close()
        
        # Return connection to pool
        db_pool.putconn(conn)
        
        # Build dictionary with status and userid
        slot_data = {
            row[0]: {
                "status": row[1].lower(),
                "userid": row[2] if row[2] is not None else 0
            }
            for row in rows
        }
        logger.info(f"✅ Retrieved {len(slot_data)} slot statuses from Neon DB")
        
        return slot_data
        
    except Exception as e:
        logger.error(f"❌ Failed to retrieve slot status from Neon DB: {e}")
        # Try to return connection to pool even on error
        try:
            if 'conn' in locals():
                db_pool.putconn(conn)
        except:
            pass
        return {}


async def insert_pelanggaran(userid: int, nomor: str, jenis_pelanggaran: str) -> bool:
    """
    Insert a violation record into the pelanggaran table.
    
    Args:
        userid: The user ID who committed the violation
        nomor: The parking slot number (nomor from parking table)
        jenis_pelanggaran: Type of violation (e.g., "Slot Hogging")
    
    Returns:
        True if successful, False otherwise
    """
    global db_pool
    
    # Check if DB pool is initialized
    if db_pool is None:
        logger.warning("⚠️ DB pool not initialized, cannot insert violation")
        return False
    
    # Skip if userid is 0 (empty slot)
    if userid == 0:
        logger.info(f"ℹ️ Skipping violation insert for empty slot (userid=0)")
        return False
    
    try:
        # Get connection from pool
        conn = db_pool.getconn()
        cursor = conn.cursor()
        
        # Insert violation record
        cursor.execute(
            "INSERT INTO pelanggaran (userid, nomor, jenis_pelanggaran) VALUES (%s, %s, %s)",
            (userid, nomor, jenis_pelanggaran)
        )
        conn.commit()
        cursor.close()
        
        # Return connection to pool
        db_pool.putconn(conn)
        
        logger.info(f"✅ Inserted violation: userid={userid}, nomor={nomor}, type={jenis_pelanggaran}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to insert violation: {e}")
        # Try to return connection to pool even on error
        try:
            if 'conn' in locals():
                db_pool.putconn(conn)
        except:
            pass
        return False


# --- Simple Debouncing Logic ---
def handle_simple_debounce(slot: ParkingSlot, new_value: bool) -> bool:
    """
    Simple time-window debouncing:
    1. If value is same as last stable value → ignore (duplicate)
    2. If value changed → wait DEBOUNCE_WINDOW_MS before accepting
    3. If value stays stable for DEBOUNCE_WINDOW_MS → update DB
    4. Enforce COOLDOWN_MS between DB updates
    
    Args:
        slot: The ParkingSlot object
        new_value: New occupancy value (True/False)
    
    Returns:
        True if DB should be updated, False otherwise
    """
    current_time = datetime.now()
    slot.message_count += 1
    
    # CASE 1: First message ever
    if slot.last_value is None:
        slot.last_value = new_value
        slot.last_change_time = current_time
        slot.is_occupied = new_value
        logger.info(f"✨ Slot {slot.slot_id}: Initial value = {new_value}")
        return True  # Update DB immediately on first detection
    
    # CASE 2: Same value as before (duplicate/stable)
    if new_value == slot.last_value:
        # Check if we have a pending different value
        if slot.pending_value is not None and slot.pending_value != new_value:
            # Value changed back before debounce completed - cancel pending
            logger.info(f"🔄 Slot {slot.slot_id}: Pending {slot.pending_value} cancelled (reverted to {new_value})")
            slot.pending_value = None
        
        # No action needed - already at this state
        logger.debug(f"⏭️ Slot {slot.slot_id}: Duplicate {new_value} ignored")
        return False
    
    # CASE 3: Value changed from last stable value
    if slot.pending_value is None:
        # New change detected - start debounce timer
        slot.pending_value = new_value
        slot.last_change_time = current_time
        logger.info(f"⏳ Slot {slot.slot_id}: Value change {slot.last_value} → {new_value} (debouncing...)")
        return False  # Don't update yet, wait for debounce
    
    # CASE 4: We have a pending value - check if debounce period elapsed
    if slot.pending_value == new_value:
        elapsed_ms = (current_time - slot.last_change_time).total_seconds() * 1000
        
        if elapsed_ms >= DEBOUNCE_WINDOW_MS:
            # Debounce period passed - confirm the change
            
            # Check cooldown period
            if slot.last_db_update:
                cooldown_elapsed = (current_time - slot.last_db_update).total_seconds() * 1000
                if cooldown_elapsed < COOLDOWN_MS:
                    remaining = COOLDOWN_MS - cooldown_elapsed
                    logger.debug(f"🛡️ Slot {slot.slot_id}: In cooldown ({remaining:.0f}ms remaining)")
                    return False
            
            # Accept the change
            logger.info(f"✅ Slot {slot.slot_id}: Change confirmed {slot.last_value} → {new_value} (stable for {elapsed_ms:.0f}ms)")
            slot.last_value = new_value
            slot.is_occupied = new_value
            slot.pending_value = None
            return True  # Update DB
        else:
            # Still waiting for debounce
            remaining = DEBOUNCE_WINDOW_MS - elapsed_ms
            logger.debug(f"⏳ Slot {slot.slot_id}: Debouncing... ({remaining:.0f}ms remaining)")
            return False
    else:
        # Pending value changed again - restart debounce
        logger.info(f"🔄 Slot {slot.slot_id}: Value changed again to {new_value} (restarting debounce)")
        slot.pending_value = new_value
        slot.last_change_time = current_time
        return False


# --- MQTT Message Handlers ---
def handle_parking_slot_occupancy(topic: str, payload: str):
    """
    Handle parking slot occupancy with simple time-window debouncing.
    Uses OCCUPIED_KEYWORD and AVAILABLE_KEYWORD for parsing.
    """
    slot_id = topic.split('/')[-1]
    
    try:
        # Parse occupancy status using configured keywords
        is_occupied = payload.strip() == OCCUPIED_KEYWORD
        
        # Validate payload (must be either OCCUPIED_KEYWORD or AVAILABLE_KEYWORD)
        if payload.strip() not in [OCCUPIED_KEYWORD, AVAILABLE_KEYWORD]:
            logger.warning(f"⚠️ Slot {slot_id}: Invalid payload '{payload}' (expected '{OCCUPIED_KEYWORD}' or '{AVAILABLE_KEYWORD}')")
            return
        
        # Get or create slot
        if slot_id not in parking_slots:
            parking_slots[slot_id] = ParkingSlot(slot_id)
            logger.info(f"✨ New parking slot registered: {slot_id}")
        
        slot = parking_slots[slot_id]
        
        # Apply simple debouncing
        should_update_db = handle_simple_debounce(slot, is_occupied)
        
        # Trigger DB update if needed
        if should_update_db:
            import asyncio
            try:
                asyncio.create_task(trigger_db_update(slot, is_occupied))
            except RuntimeError:
                # Event loop not running, log placeholder
                logger.info(f"🔄 [PLACEHOLDER] Would update DB: Slot {slot_id} → {is_occupied}")
        
    except Exception as e:
        logger.error(f"❌ Error processing slot {slot_id}: {e}")


def log_message_to_memory(topic: str, payload: str):
    """
    Store message in the g_messages list (keeps last 20 messages).
    Only logs non-parking messages to reduce spam.
    """
    # Only log non-parking messages (reduce spam)
    if not topic.startswith("SPARK_C06/isOccupied/"):
        device_id = topic.split('/')[-1]
        log_entry = {
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "topic": topic,
            "device_id": device_id,
            "payload": payload
        }
        g_messages.insert(0, log_entry)
        if len(g_messages) > 20:
            g_messages.pop()


# --- MQTT Callbacks ---
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("✅ [Backend] Connected to HiveMQ Public Broker!")
        client.subscribe(SENSOR_DATA_TOPIC)
        print(f"👂 [Backend] Subscribed to topic: {SENSOR_DATA_TOPIC}")
    else:
        print(f"❌ [Backend] Failed to connect, return code {rc}")


def on_message(client, userdata, msg):
    """
    Main MQTT message router - delegates to specific handlers based on topic.
    Implements simple debouncing for parking slot occupancy messages.
    """
    topic = msg.topic
    payload = msg.payload.decode().strip()
    
    # Only print non-parking messages to reduce spam
    if not topic.startswith("SPARK_C06/isOccupied/"):
        print(f"📩 [Backend] Received from '{topic}': {payload}")
    
    # Route to specific handlers based on topic
    if topic.startswith("SPARK_C06/isOccupied/"):
        handle_parking_slot_occupancy(topic, payload)
    
    # Add more topic handlers here as needed
    # elif topic.startswith("SPARK_C06/temperature/"):
    #     handle_temperature_update(topic, payload)
    # elif topic.startswith("SPARK_C06/ping/"):
    #     handle_ping_response(topic, payload)
    
    # Log significant messages
    log_message_to_memory(topic, payload)

# --- MQTT Client Setup ---
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1, client_id="fastapi-test-backend")
client.on_connect = on_connect
client.on_message = on_message


@app.on_event("startup")
async def startup_event():
    """Connect to MQTT, initialize database pool, and initialize inference engine when the app starts."""
    global inference_engine, db_pool
    
    # Ensure persistent data directory exists
    data_dir = Path(JSON_STORAGE_FILE).parent
    os.makedirs(data_dir, exist_ok=True)
    logger.info(f"✅ Data directory ready: {data_dir}")
    
    # Load slot regions for hogging detection
    load_slot_regions()
    
    # Initialize Neon DB connection pool
    if all([NEON_DB_HOST, NEON_DB_NAME, NEON_DB_USER, NEON_DB_PASSWORD]):
        try:
            db_pool = psycopg2.pool.SimpleConnectionPool(
                minconn=1,
                maxconn=10,
                host=NEON_DB_HOST,
                database=NEON_DB_NAME,
                user=NEON_DB_USER,
                password=NEON_DB_PASSWORD,
                port=NEON_DB_PORT,
                sslmode='require'
            )
            logger.info(f"✅ Neon DB connection pool initialized (host: {NEON_DB_HOST})")
            
            # Test connection
            test_conn = db_pool.getconn()
            test_conn.close()
            db_pool.putconn(test_conn)
            logger.info(f"✅ Neon DB connection test successful")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Neon DB connection pool: {e}")
            db_pool = None
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
    """Disconnect from MQTT and close database pool when the app shuts down."""
    global db_pool
    
    # Disconnect MQTT
    client.loop_stop()
    client.disconnect()
    logger.info("✅ MQTT disconnected")
    
    # Close database connection pool
    if db_pool is not None:
        try:
            db_pool.closeall()
            logger.info("✅ Neon DB connection pool closed")
        except Exception as e:
            logger.error(f"❌ Error closing DB pool: {e}")

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
    db_status = "connected" if db_pool is not None else "not configured"
    
    # Test DB connection if pool exists
    if db_pool is not None:
        try:
            test_conn = db_pool.getconn()
            db_pool.putconn(test_conn)
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
        "recent_messages": len(g_messages),
        "parking_slots_tracked": len(parking_slots)
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
        
        if slot_regions:
            # STAGE 1: Read slot status and userid from database
            slot_data = await get_occupancy_status()
            logger.info(f"📊 Stage 1: Retrieved {len(slot_data)} slot statuses from DB")
            
            # Extract just status for backward compatibility
            db_status = {slot_id: data["status"] for slot_id, data in slot_data.items()}
            
            # STAGE 2: Verify DB status with model detections
            # Filter detections to only those in slots marked as 'occupied' in DB
            occupied_slots = [slot_id for slot_id, status in db_status.items() if status == 'occupied']
            logger.info(f"🔍 Stage 2: Verifying {len(occupied_slots)} occupied slots against model detections")
            
            # Build verification report
            verification_results = {
                "db_occupied_slots": occupied_slots,
                "total_db_occupied": len(occupied_slots),
                "total_detections": len(detections_serializable),
                "note": "Comparing DB status with model detections for occupied slots"
            }
            
            # STAGE 3: Run hogging detection on verified occupied slots
            # Only analyze detections if we have both slot regions and detections
            if len(detections_serializable) > 0:
                hogging_analysis = hogging_detection(detections_serializable, slot_regions)
                
                # Cross-reference with DB status
                hogging_analysis["db_verification"] = {
                    "db_occupied_slots": occupied_slots,
                    "slots_with_detected_cars": list(hogging_analysis["slot_occupancy"].keys()),
                    "mismatches": []
                }
                
                # Find mismatches: DB says occupied but model says not, or vice versa
                for slot_id in slot_regions.keys():
                    db_says_occupied = db_status.get(slot_id, 'available') == 'occupied'
                    model_says_occupied = hogging_analysis["slot_occupancy"].get(slot_id, {}).get("is_occupied", False)
                    
                    if db_says_occupied != model_says_occupied:
                        hogging_analysis["db_verification"]["mismatches"].append({
                            "slot_id": slot_id,
                            "db_status": "occupied" if db_says_occupied else "available",
                            "model_status": "occupied" if model_says_occupied else "available",
                            "discrepancy_type": "false_positive" if model_says_occupied else "false_negative"
                        })
                
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
                        
                        # Insert violation into pelanggaran table
                        if violator_userid and violator_slot:
                            await insert_pelanggaran(
                                userid=violator_userid,
                                nomor=violator_slot,
                                jenis_pelanggaran="Slot Hogging"
                            )
                            logger.info(f"📝 Logged violation for userid={violator_userid} at slot {violator_slot}")
                        else:
                            logger.warning(f"⚠️ No valid userid found in hogging slots: {slots_str}")
                
                if hogging_analysis["db_verification"]["mismatches"]:
                    logger.warning(f"⚠️ Found {len(hogging_analysis['db_verification']['mismatches'])} DB/model mismatches")
            else:
                logger.info("ℹ️ No detections found, skipping hogging analysis")
        
        result = {
            "success": True,
            "detections": detections_serializable,
            "inference_time": float(stats.get("avg_time", 0.0)),
            "image_shape": [int(x) for x in image_array.shape],
            "num_detections": len(detections_serializable),
            "db_status": db_status,
            "slot_data": slot_data if slot_regions else None,
            "verification_results": verification_results,
            "hogging_analysis": hogging_analysis
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

@app.get("/parking-slots", summary="Get current parking slot occupancy status")
async def get_parking_slots():
    """
    Returns the current occupancy status of all parking slots with debouncing info.
    Data is received from MQTT topic: SPARK_C06/isOccupied/{slotID}
    """
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
    if slot_id not in parking_slots:
        raise HTTPException(status_code=404, detail=f"Slot {slot_id} not found")
    
    slot = parking_slots[slot_id]
    return {
        "status": "success",
        "slot": slot.to_dict()
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

@app.get("/get-test")
async def get_test_data():
    """
    Retrieve all stored data from the JSON file.
    """
    try:
        file_path = Path(JSON_STORAGE_FILE)
        
        if not file_path.exists():
            return {
                "status": "success",
                "message": "No data file found",
                "data": [],
                "total_entries": 0,
                "file": str(file_path)
            }
        
        # Read the file
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            
        if not content:
            return {
                "status": "success",
                "message": "File exists but is empty",
                "data": [],
                "total_entries": 0,
                "file": str(file_path)
            }
        
        # Parse JSON
        data = json.loads(content)
        if not isinstance(data, list):
            data = [data]
        
        return {
            "status": "success",
            "message": "Data retrieved successfully",
            "data": data,
            "total_entries": len(data),
            "file": str(file_path)
        }
        
    except json.JSONDecodeError as e:
        logger.error(f"❌ Failed to parse JSON: {e}")
        return {
            "status": "error",
            "message": f"JSON file is corrupted: {str(e)}",
            "data": [],
            "total_entries": 0,
            "file": str(file_path)
        }
    except Exception as e:
        logger.error(f"❌ Failed to read data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to read data: {str(e)}")

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
    global slot_regions
    
    try:
        # Validate structure
        for slot_id, region_data in regions.items():
            if 'corners' not in region_data:
                raise HTTPException(status_code=400, detail=f"Slot {slot_id} missing 'corners' field")
            if len(region_data['corners']) < 3:
                raise HTTPException(status_code=400, detail=f"Slot {slot_id} must have at least 3 corners")
        
        # Save to file
        file_path = Path(SLOT_REGIONS_FILE)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w') as f:
            json.dump(regions, f, indent=2)
        
        # Update in-memory
        slot_regions = regions
        
        logger.info(f"✅ Updated {len(regions)} slot regions")
        
        return {
            "status": "success",
            "message": f"Updated {len(regions)} slot regions",
            "slots": list(regions.keys()),
            "file": str(file_path)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to update slot regions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/slot-regions")
async def get_slot_regions():
    """
    Get current parking slot region definitions.
    """
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