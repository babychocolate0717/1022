# app/models.py - 簡化版本（移除碳排放欄位）

from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey, Enum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from sqlalchemy import UniqueConstraint
import enum

Base = declarative_base()

class EnergyRaw(Base):
    """原始能耗數據表（Agent 直接寫入）"""
    __tablename__ = "energy_raw"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp_utc = Column(DateTime, nullable=False, index=True)
    
    # 硬體資訊
    gpu_model = Column(String)
    gpu_usage_percent = Column(Float)
    gpu_power_watt = Column(Float)
    cpu_power_watt = Column(Float)
    memory_used_mb = Column(Float)
    disk_read_mb_s = Column(Float)
    disk_write_mb_s = Column(Float)
    system_power_watt = Column(Float)
    
    # 設備資訊
    device_id = Column(String, index=True)
    user_id = Column(String)
    agent_version = Column(String)
    os_type = Column(String)
    os_version = Column(String)
    location = Column(String)
    
    # 🆕 設備指紋欄位（用於安全檢測）
    cpu_model = Column(String)
    cpu_count = Column(Integer)
    total_memory = Column(Integer)
    disk_partitions = Column(Integer)
    network_interfaces = Column(Integer)
    platform_machine = Column(String)
    platform_architecture = Column(String)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class EnergyCleaned(Base):
    """清洗後的能耗數據表"""
    __tablename__ = "energy_cleaned"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp_utc = Column(DateTime(timezone=True), nullable=False, index=True)
    
    # 清洗後的硬體數據
    gpu_model = Column(String)
    gpu_usage_percent = Column(Float)
    gpu_power_watt = Column(Float)
    cpu_power_watt = Column(Float)
    memory_used_mb = Column(Float)
    disk_read_mb_s = Column(Float)
    disk_write_mb_s = Column(Float)
    system_power_watt = Column(Float)
    
    # 設備資訊
    device_id = Column(String, index=True)
    user_id = Column(String)
    agent_version = Column(String)
    os_type = Column(String)
    os_version = Column(String)
    location = Column(String)
    
    # 清洗狀態
    is_anomaly = Column(Boolean, default=False)
    anomaly_reason = Column(String, default="No anomaly detected.")
    confidence_score = Column(Float)  # 數據置信度
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())

# 🔄 新增：碳排放計算結果表（由微服務寫入）
class CarbonEmissions(Base):
    """碳排放計算結果表（由碳排計算微服務寫入）"""
    __tablename__ = "carbon_emissions"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # 關聯到清洗後的數據
    energy_cleaned_id = Column(Integer, index=True)  # 關聯到 EnergyCleaned
    device_id = Column(String, index=True)
    timestamp_utc = Column(String, nullable=False)
    
    # 原始功耗數據（來源）
    cpu_power_watt = Column(Float)
    gpu_power_watt = Column(Float)
    system_power_watt = Column(Float)
    
    # 碳排放計算結果
    cpu_co2_kg = Column(Float)
    gpu_co2_kg = Column(Float)
    system_co2_kg = Column(Float)
    total_co2_kg = Column(Float)
    
    # 計算參數
    emission_factor = Column(Float)  # 使用的排放係數
    calculation_interval_seconds = Column(Integer)
    calculation_method = Column(String)  # 計算方法說明
    
    # 累積統計（可選）
    daily_cumulative_co2_kg = Column(Float)
    monthly_cumulative_co2_kg = Column(Float)
    
    calculated_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # 索引優化
    __table_args__ = (
        {'comment': '碳排放計算結果表，由微服務計算並寫入'}
    )

# 設備授權模型
class AuthorizedDevice(Base):
    __tablename__ = "authorized_devices"
    
    id = Column(Integer, primary_key=True, index=True)
    mac_address = Column(String, unique=True, index=True, nullable=False)
    device_name = Column(String, nullable=False)
    user_name = Column(String, nullable=False)
    registered_date = Column(DateTime(timezone=True), server_default=func.now())
    last_seen = Column(DateTime(timezone=True))
    is_active = Column(Boolean, default=True)
    notes = Column(Text)

# 設備指紋模型
class DeviceFingerprint(Base):
    __tablename__ = "device_fingerprints"
    
    id = Column(Integer, primary_key=True, index=True)
    mac_address = Column(String, index=True, nullable=False)
    device_id = Column(String, index=True)
    
    # 硬體指紋
    cpu_model = Column(String)
    cpu_count = Column(Integer)
    total_memory = Column(Integer)
    disk_partitions = Column(Integer)
    network_interfaces = Column(Integer)
    platform_machine = Column(String)
    platform_architecture = Column(String)
    
    # 指紋 hash
    fingerprint_hash = Column(String, index=True)
    
    # 風險評估
    risk_score = Column(Float, default=0.0)
    is_suspicious = Column(Boolean, default=False)
    
    first_seen = Column(DateTime(timezone=True), server_default=func.now())
    last_seen = Column(DateTime(timezone=True), server_default=func.now())
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())

# 🔄 碳排放分析統計表
class CarbonAnalytics(Base):
    """碳排放分析統計表"""
    __tablename__ = "carbon_analytics"
    
    id = Column(Integer, primary_key=True, index=True)
    device_id = Column(String, index=True)
    
    # 分析週期
    period_start = Column(DateTime(timezone=True))
    period_end = Column(DateTime(timezone=True))
    period_type = Column(String)  # 'daily', 'weekly', 'monthly'
    
    # 能耗統計
    total_energy_kwh = Column(Float)
    average_power_watt = Column(Float)
    peak_power_watt = Column(Float)
    
    # 碳排放統計
    total_co2_kg = Column(Float)
    daily_average_co2_kg = Column(Float)
    co2_per_kwh = Column(Float)
    
    # 組件分析
    cpu_contribution_percent = Column(Float)
    gpu_contribution_percent = Column(Float)
    
    # 效率指標
    efficiency_score = Column(Float)
    
    generated_at = Column(DateTime(timezone=True), server_default=func.now())

# ==========================================================================
# 步驟一：新增碳排放額度管理模型
# ==========================================================================

class QuotaPeriod(str, enum.Enum):
    """定義額度計算的週期"""
    daily = "daily"
    monthly = "monthly"
    yearly = "yearly"

class QuotaScope(str, enum.Enum):
    """定義額度應用的範圍"""
    global_ = "global"  # 使用 global_ 避免與 Python 關鍵字衝突
    user = "user"
    device = "device"


class CarbonQuota(Base):
    """
    碳排放額度設定表
    (用來儲存規則，例如：User 'A' 每月限額 500kg)
    """
    __tablename__ = "carbon_quotas"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # 額度應用的範圍 (例如 'user', 'device' 或是 'global')
    scope_type = Column(Enum(QuotaScope), nullable=False, default=QuotaScope.user)  # 改成小寫 user
    
    # 範圍的具體 ID (例如 user_id 或是 device_id)
    scope_id = Column(String, nullable=False, index=True, default="default")
    
    # 額度的週期 (daily, monthly, yearly)
    period = Column(Enum(QuotaPeriod), nullable=False)
    
    # 碳排放限制 (單位: kg CO2)
    limit_co2_kg = Column(Float, nullable=False)
    
    # 規則是否啟用
    is_active = Column(Boolean, default=True)
    
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    __table_args__ = (
        # 確保同一個範圍和週期只有一條規則
        UniqueConstraint('scope_type', 'scope_id', 'period', name='_scope_period_uc'),
    )


class CarbonQuotaUsage(Base):
    """
    碳排放額度累計使用量表
    (用來追蹤實際用量，例如：User 'A' 在 2025-10-18 已使用 10.5kg)
    """
    __tablename__ = "carbon_quota_usage"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # 對應的範圍 (例如 'user')
    scope_type = Column(Enum(QuotaScope), nullable=False)
    
    # 對應的 ID (例如 user_id)
    scope_id = Column(String, nullable=False, index=True)
    
    # 額度的週期
    period_type = Column(Enum(QuotaPeriod), nullable=False)
    
    # 追蹤的週期金鑰 (例如: '2025-10-18', '2025-10', '2025')
    period_key = Column(String, nullable=False, index=True)
    
    # 已使用的碳排放量 (單位: kg CO2)
    used_co2_kg = Column(Float, default=0.0)
    
    last_updated = Column(DateTime(timezone=True), onupdate=func.now())

    __table_args__ = (
        # 確保追蹤的唯一性
        UniqueConstraint('scope_type', 'scope_id', 'period_type', 'period_key', name='_usage_tracking_uc'),
    )