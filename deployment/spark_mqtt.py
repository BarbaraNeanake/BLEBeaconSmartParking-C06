"""
SPARK MQTT Module
Handles MQTT communication, parking slot tracking, and debouncing logic.
"""

import logging
import time
from typing import Dict, List, Callable, Optional
from datetime import datetime
import paho.mqtt.client as mqtt

logger = logging.getLogger(__name__)


class ParkingSlot:
    """Represents a parking slot with debouncing state machine."""
    
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


class MQTTManager:
    """Manages MQTT connections and parking slot tracking."""
    
    def __init__(self, debounce_window_ms: int = 2000, cooldown_ms: int = 5000,
                 occupied_keyword: str = "True", available_keyword: str = "False"):
        self.debounce_window_ms = debounce_window_ms
        self.cooldown_ms = cooldown_ms
        self.occupied_keyword = occupied_keyword
        self.available_keyword = available_keyword
        self.parking_slots: Dict[str, ParkingSlot] = {}
        self.messages: List[Dict] = []
        self.db_update_callback: Optional[Callable] = None
        
        logger.info(f"✅ MQTT manager initialized: debounce={debounce_window_ms}ms, cooldown={cooldown_ms}ms")
    
    def get_parking_slots(self) -> Dict[str, ParkingSlot]:
        """Get the parking slots dictionary."""
        return self.parking_slots
    
    def get_messages(self) -> List[Dict]:
        """Get the message log."""
        return self.messages
    
    def set_db_callback(self, callback: Callable):
        """Set the database update callback."""
        self.db_update_callback = callback
    
    def handle_simple_debounce(self, slot: ParkingSlot, new_value: bool) -> bool:
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
            
            if elapsed_ms >= self.debounce_window_ms:
                # Debounce period passed - confirm the change
                
                # Check cooldown period
                if slot.last_db_update:
                    cooldown_elapsed = (current_time - slot.last_db_update).total_seconds() * 1000
                    if cooldown_elapsed < self.cooldown_ms:
                        remaining = self.cooldown_ms - cooldown_elapsed
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
                remaining = self.debounce_window_ms - elapsed_ms
                logger.debug(f"⏳ Slot {slot.slot_id}: Debouncing... ({remaining:.0f}ms remaining)")
                return False
        else:
            # Pending value changed again - restart debounce
            logger.info(f"🔄 Slot {slot.slot_id}: Value changed again to {new_value} (restarting debounce)")
            slot.pending_value = new_value
            slot.last_change_time = current_time
            return False


    def handle_parking_slot_occupancy(self, topic: str, payload: str):
        """
        Handle parking slot occupancy with simple time-window debouncing.
        Uses OCCUPIED_KEYWORD and AVAILABLE_KEYWORD for parsing.
        
        Args:
            topic: MQTT topic
            payload: MQTT payload
        """
        slot_id = topic.split('/')[-1]
        
        try:
            # Parse occupancy status using configured keywords
            is_occupied = payload.strip() == self.occupied_keyword
            
            # Validate payload (must be either OCCUPIED_KEYWORD or AVAILABLE_KEYWORD)
            if payload.strip() not in [self.occupied_keyword, self.available_keyword]:
                logger.warning(f"⚠️ Slot {slot_id}: Invalid payload '{payload}' (expected '{self.occupied_keyword}' or '{self.available_keyword}')")
                return
            
            # Get or create slot
            if slot_id not in self.parking_slots:
                self.parking_slots[slot_id] = ParkingSlot(slot_id)
                logger.info(f"✨ New parking slot registered: {slot_id}")
            
            slot = self.parking_slots[slot_id]
            
            # Update slot state directly (debouncing disabled)
            slot.last_value = is_occupied
            slot.is_occupied = is_occupied
            slot.last_change_time = datetime.now()
            
            # Apply simple debouncing (DISABLED - keeping for future use)
            # should_update_db = self.handle_simple_debounce(slot, is_occupied)
            should_update_db = True  # Always trigger update without debouncing
            
            # Trigger DB update callback if needed
            if should_update_db and self.db_update_callback:
                import asyncio
                try:
                    # Call the database update function directly with slot_id
                    asyncio.create_task(self.db_update_callback(slot_id, is_occupied))
                except RuntimeError:
                    # Event loop not running, log placeholder
                    logger.info(f"🔄 [PLACEHOLDER] Would update DB: Slot {slot_id} → {is_occupied}")
            
        except Exception as e:
            logger.error(f"❌ Error processing slot {slot_id}: {e}")


    def log_message_to_memory(self, topic: str, payload: str):
        """
        Store message in the messages list (keeps last 20 messages).
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
            self.messages.insert(0, log_entry)
            if len(self.messages) > 20:
                self.messages.pop()


    def _on_connect(self, client, userdata, flags, rc):
        """MQTT connection callback."""
        if rc == 0:
            print("✅ [Backend] Connected to HiveMQ Public Broker!")
            sensor_topic = userdata.get('sensor_topic', 'SPARK_C06/#')
            client.subscribe(sensor_topic)
            print(f"👂 [Backend] Subscribed to topic: {sensor_topic}")
        else:
            print(f"❌ [Backend] Failed to connect, return code {rc}")


    def _on_message(self, client, userdata, msg):
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
            self.handle_parking_slot_occupancy(topic, payload)
        
        # Add more topic handlers here as needed
        # elif topic.startswith("SPARK_C06/temperature/"):
        #     handle_temperature_update(topic, payload)
        # elif topic.startswith("SPARK_C06/ping/"):
        #     handle_ping_response(topic, payload)
        
        # Log significant messages
        self.log_message_to_memory(topic, payload)


    def create_client(self, broker_host: str, broker_port: int, sensor_topic: str) -> mqtt.Client:
        """
        Create and configure MQTT client.
        
        Args:
            broker_host: MQTT broker hostname
            broker_port: MQTT broker port
            sensor_topic: Topic to subscribe to
        
        Returns:
            Configured MQTT client
        """
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1, client_id="fastapi-test-backend")
        
        # Set userdata with configuration
        client.user_data_set({'sensor_topic': sensor_topic})
        
        # Bind callbacks to this instance
        client.on_connect = lambda c, u, f, rc: self._on_connect(c, u, f, rc)
        client.on_message = lambda c, u, m: self._on_message(c, u, m)
        
        return client
