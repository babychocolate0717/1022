# app/schemas.py - 更新版本

from pydantic import BaseModel, validator
from datetime import datetime
from typing import Optional
from .models import QuotaPeriod, QuotaScope

class EnergyData(BaseModel):
    # 核心能耗數據欄位
    timestamp_utc: str
    gpu_model: str
    gpu_usage_percent: float
    gpu_power_watt: float
    cpu_power_watt: float
    memory_used_mb: float
    disk_read_mb_s: float
    disk_write_mb_s: float
    system_power_watt: float
    device_id: str
    user_id: str
    agent_version: str
    os_type: str
    os_version: str
    location: str

    #額度管理
    collection_interval_sec: Optional[int] = 60
    
    # 🔧 設備指紋欄位（可選，用於安全檢測）
    cpu_model: Optional[str] = None
    cpu_count: Optional[int] = None
    total_memory: Optional[int] = None
    disk_partitions: Optional[int] = None
    network_interfaces: Optional[int] = None
    platform_machine: Optional[str] = None
    platform_architecture: Optional[str] = None
    
    # 🔧 清洗相關欄位（可選，由 cleaning-api 添加）
    is_anomaly: Optional[bool] = False 
    anomaly_reason: Optional[str] = None
    confidence_score: Optional[float] = None  # 🆕 支援這個欄位
    
    # 資料驗證
    @validator('gpu_usage_percent')
    def validate_gpu_usage(cls, v):
        if not 0 <= v <= 100:
            raise ValueError('GPU usage must be between 0 and 100')
        return v
    
    @validator('gpu_power_watt', 'cpu_power_watt')
    def validate_component_power(cls, v):
        if not 0 <= v <= 1000:
            raise ValueError('Component power consumption must be between 0 and 1000W')
        return v
    
    @validator('system_power_watt')
    def validate_system_power(cls, v):
        # 合理的系統功耗上限
        if not 0 <= v <= 1500:
            raise ValueError('System power consumption must be between 0 and 1500W')
        return v
    
    @validator('memory_used_mb')
    def validate_memory(cls, v):
        if not 0 <= v <= 128000:
            raise ValueError('Memory usage must be between 0 and 128GB')
        return v
    
    @validator('confidence_score')
    def validate_confidence_score(cls, v):
        if v is not None and not 0 <= v <= 1:
            raise ValueError('Confidence score must be between 0 and 1')
        return v

# 設備管理 schemas
class DeviceCreate(BaseModel):
    mac_address: str
    device_name: str
    user_name: str
    notes: Optional[str] = None

class DeviceResponse(BaseModel):
    mac_address: str
    device_name: str
    user_name: str
    registered_date: datetime
    last_seen: Optional[datetime]
    is_active: bool
    notes: Optional[str]
    
    class Config:
        from_attributes = True

# 設備指紋相關 schemas
class DeviceFingerprintResponse(BaseModel):
    risk_level: str  # "low", "medium", "high"
    similarity_score: float
    message: str
    is_suspicious: bool

class FingerprintCheckResult(BaseModel):
    fingerprint_check: DeviceFingerprintResponse
    status: str

# 用於 API 回應的資料庫原始記錄模型
class EnergyRawResponse(BaseModel):
    id: int
    timestamp_utc: datetime
    device_id: str
    user_id: str
    raw_data: Optional[dict] = None
    mac_address: Optional[str] = None
    is_cleaned: Optional[bool] = None
    risk_level: Optional[str] = None
    device_fingerprint: Optional[str] = None

    class Config:
        from_attributes = True


# ==========================================================================
# 步驟三：新增碳排放額度管理 Schemas
# ==========================================================================

class CarbonQuotaBase(BaseModel):
    scope_type: QuotaScope = QuotaScope.user
    scope_id: str
    period: QuotaPeriod
    limit_co2_kg: float
    notes: Optional[str] = None
    is_active: bool = True

class CarbonQuotaCreate(CarbonQuotaBase):
    pass

class CarbonQuotaResponse(CarbonQuotaBase):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True # 舊版 Pydantic 寫 orm_mode = True

class CarbonQuotaUsageResponse(BaseModel):
    scope_type: QuotaScope
    scope_id: str
    period_type: QuotaPeriod
    period_key: str
    used_co2_kg: float
    limit_co2_kg: Optional[float] = None
    remaining_co2_kg: Optional[float] = None

    class Config:
        from_attributes = True