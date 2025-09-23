from fastapi import FastAPI
import paho.mqtt.client as mqtt
import ssl
import time
import os
from typing import List, Dict
from pydantic import BaseModel


app = FastAPI(
    title = "CAPS Backend"
)

class AlarmPayload(BaseModel):
    sensor_id: str

# --- Configuration (from Hugging Face Secrets) ---
MQTT_BROKER_HOST = os.environ.get("MQTT_BROKER_HOST")
MQTT_USERNAME = os.environ.get("MQTT_USERNAME")
MQTT_PASSWORD = os.environ.get("MQTT_PASSWORD")
MQTT_BROKER_PORT = 8883
SENSOR_DATA_TOPIC = "sensor/+/data"

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
    """Connect to MQTT when the app starts."""
    if not all([MQTT_BROKER_HOST, MQTT_USERNAME, MQTT_PASSWORD]):
        print("❌ MQTT credentials not set in Space secrets!")
        return
    client.connect(MQTT_BROKER_HOST, MQTT_BROKER_PORT, 60)
    client.loop_start()

@app.on_event("shutdown")
async def shutdown_event():
    """Disconnect from MQTT when the app shuts down."""
    client.loop_stop()
    client.disconnect()

# --- API Endpoints ---
@app.get("/")
async def root():
    return {"status": "ok", "message": "Backend is running."}

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
