"""
SPARK Database Module
Handles all PostgreSQL database operations for parking slot management.
"""

import logging
from typing import Dict, Optional
from datetime import datetime
import psycopg2
from psycopg2 import pool

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages PostgreSQL database connections and operations."""
    
    def __init__(self):
        self.pool: Optional[pool.SimpleConnectionPool] = None
    
    def initialize(self, host: str, database: str, user: str, password: str, 
                   port: str = "5432", min_conn: int = 1, max_conn: int = 10):
        """
        Initialize PostgreSQL connection pool for Neon DB.
        
        Args:
            host: Database host
            database: Database name
            user: Database user
            password: Database password
            port: Database port (default: 5432)
            min_conn: Minimum connections in pool (default: 1)
            max_conn: Maximum connections in pool (default: 10)
        
        Returns:
            Self for method chaining
        """
        try:
            self.pool = psycopg2.pool.SimpleConnectionPool(
                minconn=min_conn,
                maxconn=max_conn,
                host=host,
                database=database,
                user=user,
                password=password,
                port=port,
                sslmode='require'
            )
            logger.info(f"✅ Neon DB connection pool initialized (host: {host})")
            
            # Test connection
            test_conn = self.pool.getconn()
            test_conn.close()
            self.pool.putconn(test_conn)
            logger.info(f"✅ Neon DB connection test successful")
            
            return self
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Neon DB connection pool: {e}")
            self.pool = None
            return self
    
    def close(self):
        """Close the database connection pool."""
        if self.pool is not None:
            try:
                self.pool.closeall()
                logger.info("✅ Neon DB connection pool closed")
            except Exception as e:
                logger.error(f"❌ Error closing DB pool: {e}")
    
    def is_connected(self) -> bool:
        """Check if database pool is initialized."""
        return self.pool is not None


    async def update_slot_status(self, slot_id: str, is_occupied: bool) -> bool:
        """
        Update parking slot status in the database.
        
        Args:
            slot_id: The parking slot ID (nomor)
            is_occupied: True if slot is occupied, False if available
        
        Returns:
            True if successful, False otherwise
        """
        if self.pool is None:
            logger.warning("⚠️ DB pool not initialized, cannot update slot status")
            return False
        
        try:
            # Get connection from pool
            conn = self.pool.getconn()
            cursor = conn.cursor()
            
            # Update slot status in parking table
            status = 'occupied' if is_occupied else 'available'
            cursor.execute(
                "UPDATE parking SET status = %s WHERE nomor = %s",
                (status, slot_id)
            )
            conn.commit()
            cursor.close()
            
            # Return connection to pool
            self.pool.putconn(conn)
            
            logger.info(f"✅ Updated slot {slot_id} status to {status}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to update slot status: {e}")
            # Try to return connection to pool even on error
            try:
                if 'conn' in locals():
                    self.pool.putconn(conn)
            except:
                pass
            return False
    
    async def get_slot_status(self, slot_id: str) -> dict:
        """
        Get the current status of a parking slot.
        
        Args:
            slot_id: The parking slot ID (nomor)
        
        Returns:
            Dict with slot status or None if not found
        """
        if self.pool is None:
            logger.warning("⚠️ DB pool not initialized, cannot get slot status")
            return None
        
        try:
            # Get connection from pool
            conn = self.pool.getconn()
            cursor = conn.cursor()
            
            # Query slot status
            cursor.execute(
                "SELECT nomor, status FROM parking WHERE nomor = %s",
                (slot_id,)
            )
            row = cursor.fetchone()
            cursor.close()
            
            # Return connection to pool
            self.pool.putconn(conn)
            
            if row:
                return {
                    "nomor": row[0],
                    "status": row[1]
                }
            else:
                logger.warning(f"⚠️ Slot {slot_id} not found in database")
                return None
            
        except Exception as e:
            logger.error(f"❌ Failed to get slot status: {e}")
            # Try to return connection to pool even on error
            try:
                if 'conn' in locals():
                    self.pool.putconn(conn)
            except:
                pass
            return None
    
    async def trigger_update(self, slot, is_occupied: bool):
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


    async def get_occupied_slots_with_neighbors(self) -> Dict[str, Dict[str, any]]:
        # Check if DB pool is initialized
        if self.pool is None:
            logger.warning("⚠️ DB pool not initialized, cannot fetch slot status")
            return {}
        
        try:
            # Get connection from pool
            conn = self.pool.getconn()
            cursor = conn.cursor()
            
            # STEP 1: Get all occupied slots
            cursor.execute(
                "SELECT nomor, status, userid FROM parking_slots WHERE status = 'occupied' ORDER BY nomor"
            )
            occupied_rows = cursor.fetchall()
        
            if not occupied_rows:
                logger.info("ℹ️ No occupied slots found in database")
                cursor.close()
                self.pool.putconn(conn)
                return {}
            
            # STEP 2: Calculate neighbor slots for each occupied slot
            # For integer slot IDs: neighbors are simply slot_id-1 and slot_id+1
            occupied_slot_ids = [row[0] for row in occupied_rows]
            neighbor_ids = set()
            
            for slot_id in occupied_slot_ids:
                try:
                    # Convert to int and get left/right neighbors
                    slot_num = int(slot_id)
                    left_neighbor = str(slot_num - 1)
                    right_neighbor = str(slot_num + 1)
                    
                    neighbor_ids.add(left_neighbor)
                    neighbor_ids.add(right_neighbor)
                except (ValueError, TypeError):
                    logger.warning(f"⚠️ Skipping non-integer slot_id: {slot_id}")
                    continue
            
            # Remove occupied slots from neighbor list (avoid duplicates)
            neighbor_ids = neighbor_ids - set(occupied_slot_ids)
            
            # STEP 3: Query neighbor slots
            neighbor_rows = []
            if neighbor_ids:
                placeholders = ','.join(['%s'] * len(neighbor_ids))
                cursor.execute(
                    f"SELECT nomor, status, userid FROM parking_slots WHERE nomor IN ({placeholders})",
                    tuple(neighbor_ids)
                )
                neighbor_rows = cursor.fetchall()
            
            cursor.close()
            self.pool.putconn(conn)
            
            # STEP 4: Combine occupied + neighbor data
            slot_data = {}
            
            # Add occupied slots
            for row in occupied_rows:
                slot_data[row[0]] = {
                    "status": row[1].lower(),
                    "userid": row[2] if row[2] is not None else 0,
                    "query_reason": "occupied"
                }
            
            # Add neighbor slots
            for row in neighbor_rows:
                slot_data[row[0]] = {
                    "status": row[1].lower(),
                    "userid": row[2] if row[2] is not None else 0,
                    "query_reason": "neighbor"
                }
            
            logger.info(f"✅ Retrieved {len(occupied_rows)} occupied slots + {len(neighbor_rows)} neighbors from DB")
            
            return slot_data
            
        except Exception as e:
            logger.error(f"❌ Failed to retrieve slot status from Neon DB: {e}")
            # Try to return connection to pool even on error
            try:
                if 'conn' in locals():
                    self.pool.putconn(conn)
            except:
                pass
            return {}


    async def check_existing_violation(self, userid: int, nomor: str) -> bool:
        # Check if DB pool is initialized
        if self.pool is None:
            logger.warning("⚠️ DB pool not initialized, cannot check existing violation")
            return False
        
        try:
            # Get connection from pool
            conn = self.pool.getconn()
            cursor = conn.cursor()
            
            # Check if violation exists for this userid and slot
            # We check recent violations (same day) to avoid false positives from old sessions
            cursor.execute(
                """SELECT COUNT(*) FROM pelanggaran 
                   WHERE userid = %s AND nomor = %s 
                   AND DATE(created_at) = CURRENT_DATE""",
                (userid, nomor)
            )
            count = cursor.fetchone()[0]
            cursor.close()
            
            # Return connection to pool
            self.pool.putconn(conn)
            
            exists = count > 0
            if exists:
                logger.info(f"ℹ️ Violation already logged today for userid={userid}, nomor={nomor}")
            
            return exists
            
        except Exception as e:
            logger.error(f"❌ Failed to check existing violation: {e}")
            # Try to return connection to pool even on error
            try:
                if 'conn' in locals():
                    self.pool.putconn(conn)
            except:
                pass
            return False


    async def insert_pelanggaran(self, userid: int, nomor: str, jenis_pelanggaran: str) -> bool:
        """
        Insert a violation record into the pelanggaran table.
        Checks for existing violations first to prevent duplicates.
        
        Args:
            userid: The user ID who committed the violation
            nomor: The parking slot number (nomor from parking table)
            jenis_pelanggaran: Type of violation (e.g., "Slot Hogging")
        
        Returns:
            True if successful, False otherwise
        """
        # Check if DB pool is initialized
        if self.pool is None:
            logger.warning("⚠️ DB pool not initialized, cannot insert violation")
            return False
        
        # Skip if userid is 0 (empty slot)
        if userid == 0:
            logger.info(f"ℹ️ Skipping violation insert for empty slot (userid=0)")
            return False
        
        # Check if violation already exists for this user and slot
        if await self.check_existing_violation(userid, nomor):
            logger.info(f"⏭️ Skipping duplicate violation for userid={userid}, nomor={nomor}")
            return False
        
        try:
            # Get connection from pool
            conn = self.pool.getconn()
            cursor = conn.cursor()
            
            # Insert violation record
            cursor.execute(
                "INSERT INTO pelanggaran (userid, nomor, jenis_pelanggaran) VALUES (%s, %s, %s)",
                (userid, nomor, jenis_pelanggaran)
            )
            conn.commit()
            cursor.close()
            
            # Return connection to pool
            self.pool.putconn(conn)
            
            logger.info(f"✅ Inserted violation: userid={userid}, nomor={nomor}, type={jenis_pelanggaran}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to insert violation: {e}")
            # Try to return connection to pool even on error
            try:
                if 'conn' in locals():
                    self.pool.putconn(conn)
            except:
                pass
            return False
