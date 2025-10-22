# app/utils/mac_manager.py
from sqlalchemy.orm import Session
from ..models import AuthorizedDevice  # ✅ 正確
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class MACManager:
    def __init__(self, db: Session):
        self.db = db
    
    def add_device(self, mac_address: str, device_name: str, user_name: str, notes: str = None) -> bool:
        try:
            mac_address = self._normalize_mac(mac_address)
            
            existing = self.db.query(AuthorizedDevice).filter(
                AuthorizedDevice.mac_address == mac_address
            ).first()
            
            if existing:
                if not existing.is_active:
                    existing.is_active = True
                    existing.device_name = device_name
                    existing.user_name = user_name
                    existing.notes = notes
                    existing.registered_date = datetime.now()
                    self.db.commit()
                    logger.info(f"Reactivated device: {mac_address}")
                    return True
                else:
                    logger.warning(f"Device already exists and active: {mac_address}")
                    return False
            
            new_device = AuthorizedDevice(
                mac_address=mac_address,
                device_name=device_name,
                user_name=user_name,
                notes=notes
            )
            
            self.db.add(new_device)
            self.db.commit()
            logger.info(f"Added new device: {mac_address} ({device_name})")
            return True
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to add device {mac_address}: {e}")
            return False
    
    def remove_device(self, mac_address: str) -> bool:
        try:
            mac_address = self._normalize_mac(mac_address)
            
            device = self.db.query(AuthorizedDevice).filter(
                AuthorizedDevice.mac_address == mac_address
            ).first()
            
            if device:
                device.is_active = False
                self.db.commit()
                logger.info(f"Deactivated device: {mac_address}")
                return True
            else:
                logger.warning(f"Device not found: {mac_address}")
                return False
                
        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to remove device {mac_address}: {e}")
            return False
    
    def list_devices(self, active_only: bool = True):
        query = self.db.query(AuthorizedDevice)
        if active_only:
            query = query.filter(AuthorizedDevice.is_active == True)
        
        return query.all()
    
    def get_device(self, mac_address: str):
        mac_address = self._normalize_mac(mac_address)
        return self.db.query(AuthorizedDevice).filter(
            AuthorizedDevice.mac_address == mac_address
        ).first()
    
    def _normalize_mac(self, mac_address: str) -> str:
        return mac_address.upper().replace('-', ':')