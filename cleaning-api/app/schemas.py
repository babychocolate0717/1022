from pydantic import BaseModel

class RawEnergyData(BaseModel):
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


## cleaning-api/app/cleaning.py
def clean_energy_data(data: dict) -> dict:
    cleaned = data.copy()

    # 替換 unknown GPU 型號
    if cleaned.get("gpu_model", "").lower() == "unknown":
        cleaned["gpu_model"] = "Generic GPU"

    # 數值欄位轉換
    float_fields = [
        "gpu_usage_percent", "gpu_power_watt", "cpu_power_watt",
        "memory_used_mb", "disk_read_mb_s", "disk_write_mb_s", "system_power_watt"
    ]
    for field in float_fields:
        try:
            cleaned[field] = float(cleaned.get(field, 0))
        except (ValueError, TypeError):
            cleaned[field] = 0.0

    return cleaned
