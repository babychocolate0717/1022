-- init.sql - 放在 ingestion-api/init.sql

-- 創建 energy_raw 表
CREATE TABLE IF NOT EXISTS energy_raw (
    id SERIAL PRIMARY KEY,
    timestamp_utc VARCHAR NOT NULL,
    gpu_model VARCHAR,
    gpu_usage_percent FLOAT,
    gpu_power_watt FLOAT,
    cpu_power_watt FLOAT,
    memory_used_mb FLOAT,
    disk_read_mb_s FLOAT,
    disk_write_mb_s FLOAT,
    system_power_watt FLOAT,
    device_id VARCHAR,
    user_id VARCHAR,
    agent_version VARCHAR,
    os_type VARCHAR,
    os_version VARCHAR,
    location VARCHAR,
    cpu_model VARCHAR,
    cpu_count INTEGER,
    total_memory BIGINT,
    disk_partitions INTEGER,
    network_interfaces INTEGER,
    platform_machine VARCHAR,
    platform_architecture VARCHAR,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 創建 energy_cleaned 表
CREATE TABLE IF NOT EXISTS energy_cleaned (
    id SERIAL PRIMARY KEY,
    timestamp_utc VARCHAR NOT NULL,
    gpu_model VARCHAR,
    gpu_usage_percent FLOAT,
    gpu_power_watt FLOAT,
    cpu_power_watt FLOAT,
    memory_used_mb FLOAT,
    disk_read_mb_s FLOAT,
    disk_write_mb_s FLOAT,
    system_power_watt FLOAT,
    device_id VARCHAR,
    user_id VARCHAR,
    agent_version VARCHAR,
    os_type VARCHAR,
    os_version VARCHAR,
    location VARCHAR,
    is_anomaly BOOLEAN DEFAULT FALSE,
    anomaly_reason VARCHAR,
    confidence_score FLOAT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 創建 authorized_devices 表
CREATE TABLE IF NOT EXISTS authorized_devices (
    id SERIAL PRIMARY KEY,
    mac_address VARCHAR UNIQUE NOT NULL,
    device_name VARCHAR NOT NULL,
    user_name VARCHAR NOT NULL,
    registered_date TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_seen TIMESTAMP WITH TIME ZONE,
    is_active BOOLEAN DEFAULT TRUE,
    notes TEXT
);

-- 創建 device_fingerprints 表
CREATE TABLE IF NOT EXISTS device_fingerprints (
    id SERIAL PRIMARY KEY,
    mac_address VARCHAR NOT NULL,
    device_id VARCHAR,
    cpu_model VARCHAR,
    cpu_count INTEGER,
    total_memory BIGINT,
    disk_partitions INTEGER,
    network_interfaces INTEGER,
    platform_machine VARCHAR,
    platform_architecture VARCHAR,
    fingerprint_hash VARCHAR,
    risk_score FLOAT DEFAULT 0.0,
    is_suspicious BOOLEAN DEFAULT FALSE,
    first_seen TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_seen TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 創建索引
CREATE INDEX IF NOT EXISTS idx_energy_raw_device_id ON energy_raw(device_id);
CREATE INDEX IF NOT EXISTS idx_energy_cleaned_device_id ON energy_cleaned(device_id);
CREATE INDEX IF NOT EXISTS idx_authorized_devices_mac ON authorized_devices(mac_address);
CREATE INDEX IF NOT EXISTS idx_device_fingerprints_mac ON device_fingerprints(mac_address);
CREATE INDEX IF NOT EXISTS idx_device_fingerprints_hash ON device_fingerprints(fingerprint_hash);