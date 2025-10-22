# app/auth.py
from fastapi import HTTPException, Header, Depends, Request
from sqlalchemy.orm import Session
import hashlib
import hmac
import os
import json  # 🆕 新增匯入
import logging
from .database import SessionLocal
from .models import AuthorizedDevice
from . import models  # 🆕 新增匯入
from datetime import datetime

logger = logging.getLogger(__name__)

# 兼容性設置
COMPATIBILITY_MODE = os.getenv("COMPATIBILITY_MODE", "true").lower() == "true"
DEFAULT_ALLOWED_IPS = os.getenv("DEFAULT_ALLOWED_IPS", "").split(",") if os.getenv("DEFAULT_ALLOWED_IPS") else []

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class DeviceAuthenticator:
    def __init__(self, db: Session):
        self.db = db
        self.secret_key = os.getenv("AUTH_SECRET_KEY", "your-default-secret-key")
    
        # 🆕 從環境變數讀取門檻值
        self.fingerprint_enabled = os.getenv("FINGERPRINT_ENABLED", "true").lower() == "true"
        self.high_risk_threshold = float(os.getenv("HIGH_RISK_THRESHOLD", "0.7"))
        self.medium_risk_threshold = float(os.getenv("MEDIUM_RISK_THRESHOLD", "0.9"))
        
    def check_device_fingerprint(self, device_data: dict) -> dict:
        """檢查設備指紋是否異常"""
        
        # 檢查是否啟用指紋功能
        if not self.fingerprint_enabled:
            return {"status": "disabled", "risk_level": "low", "message": "指紋檢查已停用"}
        
        # 1. 生成設備指紋
        fingerprint = self._generate_fingerprint(device_data)
        device_id = device_data.get('device_id', 'unknown')
        
        # 2. 查詢歷史指紋記錄
        historical_fingerprints = self._get_device_history(device_id)
        
        # 3. 異常偵測邏輯
        if not historical_fingerprints:
            # 新設備，建立基線
            self._save_fingerprint(device_id, fingerprint, device_data)
            return {
                "status": "new_device", 
                "risk_level": "low", 
                "message": "新設備已記錄",
                "fingerprint": fingerprint,
                "similarity_score": 1.0
            }
        
        # 4. 計算相似度
        similarity_score = self._calculate_similarity(fingerprint, historical_fingerprints[-1])
        
        # 5. 判斷異常等級（使用環境變數門檻值）
        if similarity_score > self.medium_risk_threshold:
            risk_level = "low"
            message = "設備正常"
        elif similarity_score > self.high_risk_threshold:
            risk_level = "medium" 
            message = "設備有輕微變化"
        else:
            risk_level = "high"
            message = "設備指紋異常，可能為偽造設備"
        
        # 6. 更新指紋記錄
        if risk_level != "high":
            self._save_fingerprint(device_id, fingerprint, device_data)
        
        return {
            "status": "verified",
            "risk_level": risk_level,
            "similarity_score": similarity_score,
            "message": message,
            "fingerprint": fingerprint
        }
    
    def _generate_fingerprint(self, device_data: dict) -> str:
        """生成設備指紋（使用現有資料）"""
        # 使用你已經收集的硬體資訊
        fingerprint_data = {
            "gpu_model": device_data.get('gpu_model', 'unknown'),
            "os_type": device_data.get('os_type', 'unknown'),
            "os_version": device_data.get('os_version', 'unknown'),
            "agent_version": device_data.get('agent_version', 'unknown'),
            # 可以加入更多硬體特徵
        }
        
        # 生成hash指紋
        fingerprint_str = json.dumps(fingerprint_data, sort_keys=True)
        return hashlib.sha256(fingerprint_str.encode()).hexdigest()[:16]  # 取前16位
    
    def _get_device_history(self, device_id: str) -> list:
        """取得設備歷史指紋記錄"""
        # 查詢最近的指紋記錄（可以存在現有資料表中）
        try:
            recent_records = self.db.query(models.EnergyRaw)\
                .filter(models.EnergyRaw.device_id == device_id)\
                .order_by(models.EnergyRaw.timestamp_utc.desc())\
                .limit(5).all()
            
            fingerprints = []
            for record in recent_records:
                if hasattr(record, 'device_fingerprint') and record.device_fingerprint:
                    fingerprints.append(record.device_fingerprint)
            
            return fingerprints
        except Exception as e:
            logger.error(f"Failed to get device history: {str(e)}")
            return []
    
    def _calculate_similarity(self, new_fingerprint: str, old_fingerprint: str) -> float:
        """計算指紋相似度（簡化版）"""
        # 簡單的字元相似度計算
        if new_fingerprint == old_fingerprint:
            return 1.0
        
        # 計算hamming距離
        different_chars = sum(c1 != c2 for c1, c2 in zip(new_fingerprint, old_fingerprint))
        similarity = 1.0 - (different_chars / len(new_fingerprint))
        
        return max(0.0, similarity)
    
    def _save_fingerprint(self, device_id: str, fingerprint: str, device_data: dict):
        """儲存設備指紋（可以加到現有資料中）"""
        # 這裡可以在現有的EnergyRaw表中加一個fingerprint欄位
        # 或者只在記憶體中暫存最近的指紋記錄
        logger.info(f"Device {device_id} fingerprint updated: {fingerprint}")
    
    def is_device_authorized(self, mac_address: str) -> bool:
        """檢查設備是否被授權"""
        if not mac_address:
            return False
            
        mac_address = self._normalize_mac(mac_address)
        
        device = self.db.query(AuthorizedDevice).filter(
            AuthorizedDevice.mac_address == mac_address,
            AuthorizedDevice.is_active == True
        ).first()
        
        if device:
            device.last_seen = datetime.now()
            self.db.commit()
            logger.info(f"Authorized device accessed: {mac_address}")
            return True
        
        logger.warning(f"Unauthorized device attempted access: {mac_address}")
        return False
    
    def verify_certificate(self, mac_address: str, certificate: str) -> bool:
        """驗證設備憑證"""
        if not mac_address or not certificate:
            return False
        
        mac_address = self._normalize_mac(mac_address)
        expected_cert = hmac.new(
            self.secret_key.encode(), 
            mac_address.encode(), 
            hashlib.sha256
        ).hexdigest()
        
        return certificate == expected_cert
    
    def _normalize_mac(self, mac_address: str) -> str:
        """標準化 MAC 地址格式"""
        return mac_address.upper().replace('-', ':')

# 兼容性認證依賴
async def verify_device_auth_compatible(
    request: Request,
    mac_address: str = Header(None, alias="MAC-Address"),
    device_certificate: str = Header(None, alias="Device-Certificate"),
    db: Session = Depends(get_db)
):
    """兼容舊版 Agent 的認證中間件（移除白名單強制要求）"""
    
    # 模式 1：新版 Agent (有完整認證 Headers)
    if mac_address and device_certificate:
        logger.info("Using new authentication method")
        authenticator = DeviceAuthenticator(db)
        
        # 🆕 改為可選的白名單檢查（不強制）
        is_whitelisted = authenticator.is_device_authorized(mac_address)
        
        # 驗證憑證
        if not authenticator.verify_certificate(mac_address, device_certificate):
            raise HTTPException(status_code=401, detail="Invalid device certificate")
        
        # 🆕 返回白名單狀態，但不阻止非白名單設備
        return {
            "mac_address": mac_address, 
            "authenticated": True, 
            "method": "whitelist_auth" if is_whitelisted else "fingerprint_auth",
            "whitelisted": is_whitelisted
        }
    
    # 模式 2：兼容模式 (舊版 Agent) - 完全依賴指紋
    elif COMPATIBILITY_MODE:
        logger.warning("Using compatibility mode for legacy agent")
        client_ip = request.client.host
        
        # 🆕 移除IP白名單檢查，直接允許
        logger.info(f"Legacy agent allowed in compatibility mode: {client_ip}")
        return {
            "mac_address": f"legacy-{client_ip}", 
            "authenticated": True, 
            "method": "legacy_mode",
            "whitelisted": False
        }
    
    # 模式 3：嚴格模式 (拒絕舊版)
    else:
        raise HTTPException(
            status_code=401, 
            detail="Missing authentication headers. Please upgrade your agent."
        )