# \project-root_1015\ingestion-api\app\main.py

from fastapi import FastAPI, Depends, HTTPException, Request
from sqlalchemy.orm import Session
from . import models, schemas, auth
from .database import SessionLocal, engine, Base
from .auth import verify_device_auth_compatible, get_db
from .utils.mac_manager import MACManager
import requests
import logging
from datetime import datetime, timezone
from typing import List
from sqlalchemy import text, func, distinct
from .models import CarbonQuota, CarbonQuotaUsage, QuotaPeriod, QuotaScope
from .schemas import CarbonQuotaCreate, CarbonQuotaResponse, CarbonQuotaUsageResponse
from sqlalchemy.dialects.postgresql import insert as pg_insert # 用於 UPSERT
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Enum, and_


# --- Prometheus 監控套件 ---
from starlette.middleware import Middleware
from starlette_exporter import PrometheusMiddleware, handle_metrics

# --- 建立 FastAPI App 並加入 Prometheus Middleware ---
app = FastAPI(
    title="Energy Data Ingestion API",
    version="1.3.0-final",
    middleware=[
        Middleware(PrometheusMiddleware)
    ]
)
app.add_route("/metrics", handle_metrics) # 自動產生 Prometheus 指標

# --- 設定日誌 ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 資料庫初始化 ---
logger.info("開始建立資料表...")
Base.metadata.create_all(bind=engine)
logger.info("資料表建立完成")

#定義碳排放因子
CARBON_EMISSION_FACTOR_EF = 0.474 # (kgCO2e/kWh) 根據您的公式
#新增：碳排用量更新輔助函數
def update_carbon_usage(
    db: Session, 
    scope_id: str, 
    co2_kg: float, 
    timestamp: datetime,
    scope_type: QuotaScope = QuotaScope.user # 預設以 User 為範圍
):
    """
    更新指定範圍 (例如 User) 的每日、每月、每年的碳排用量。
    使用 UPSERT (INSERT ... ON CONFLICT) 確保高效能。
    """
    if co2_kg <= 0:
        return

    # 定義不同週期的 'period_key'
    periods = [
        (QuotaPeriod.daily, now.strftime('%Y-%m-%d')),
        (QuotaPeriod.monthly, now.strftime('%Y-%m')),
        (QuotaPeriod.yearly, now.strftime('%Y')),
    ]

    try:
        for period_type, period_key in periods:

            # 準備 UPSERT 語句
            stmt = pg_insert(CarbonQuotaUsage).values(
                scope_type=scope_type,
                scope_id=scope_id,
                period_type=period_type,
                period_key=period_key,
                used_co2_kg=co2_kg,
                last_updated=datetime.now(timezone.utc) # 確保時區一致
            )

            # 定義衝突時的處理方式：累加 used_co2_kg
            stmt = stmt.on_conflict_do_update(
                index_elements=['scope_type', 'scope_id', 'period_type', 'period_key'],
                set_=dict(
                    used_co2_kg=CarbonQuotaUsage.used_co2_kg + stmt.excluded.used_co2_kg,
                    last_updated=stmt.excluded.last_updated
                )
            )

            db.execute(stmt)

        logger.info(f"Updated carbon usage for {scope_type.value} '{scope_id}': +{co2_kg:.6f} kg")

    except Exception as e:
        logger.error(f"Failed to update carbon usage for '{scope_id}': {str(e)}")
        # 這裡不應中斷主資料流程，僅記錄錯誤
        pass



@app.get("/")
async def root():
    return {
        "message": "Energy Data Ingestion API",
        "version": "1.3.0-final",
        "features": ["MAC Authentication", "Device Fingerprint", "Device Management", "Health Monitoring", "Prometheus Metrics"]
    }

# --- ✅ 最終修正版的 Health Check ---
@app.get("/health")
async def health_check(db: Session = Depends(get_db)):
    """健康檢查端點（只檢查自身與資料庫）"""
    try:
        db.execute(text("SELECT 1"))
        return {
            "status": "healthy",
            "database": "connected",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail="Service unhealthy")

@app.post("/ingest")
def ingest(
    request: Request,
    data: schemas.EnergyData,
    db: Session = Depends(get_db),
    auth: dict = Depends(verify_device_auth_compatible)
):
    """接收能耗資料並進行處理（直接使用 Agent 的 AI 清洗結果）"""
    logger.info(f"Received data from device: {auth['mac_address']} (method: {auth['method']})")

    # --- 🔽 [NEW] 定義需要轉換的欄位及其類型 🔽 ---
    numeric_fields = {
        # Integers
        "cpu_count": int,
        "total_memory": int,
        "disk_partitions": int,
        "network_interfaces": int,
        # Floats
        "gpu_usage_percent": float,
        "gpu_power_watt": float,
        "cpu_power_watt": float,
        "memory_used_mb": float,
        "disk_read_mb_s": float,
        "disk_write_mb_s": float,
        "system_power_watt": float,
        "confidence_score": float, # From EnergyCleaned
        "similarity_score": float, # From potential security checks
        "risk_score": float # From potential security checks
    }
    # --- 🔼 [NEW] 結束 🔼 ---

    try:
        raw_data = data.dict()
        unsupported_fields = ['device_fingerprint', 'fingerprint_hash', 'risk_score'] # risk_score 在 numeric_fields 處理
        for field in unsupported_fields:
            if field != 'risk_score': # 保留 risk_score 給後面轉換
                 raw_data.pop(field, None)

        raw_supported_fields = {
            "timestamp_utc", "gpu_model", "gpu_usage_percent", "gpu_power_watt",
            "cpu_power_watt", "memory_used_mb", "disk_read_mb_s", "disk_write_mb_s",
            "system_power_watt", "device_id", "user_id", "agent_version",
            "os_type", "os_version", "location", "cpu_model", "cpu_count",
            "total_memory", "disk_partitions", "network_interfaces",
            "platform_machine", "platform_architecture",
            # Include fields for cleaning and numeric conversion
            "is_anomaly", "anomaly_reason", "confidence_score", "similarity_score", "risk_score"
        }
        # 先不過濾 None，因為類型轉換需要處理 None
        raw_filtered = {k: v for k, v in raw_data.items() if k in raw_supported_fields}

        # --- 🔽 [FIX 1 & NEW] 轉換 timestamp 和數字欄位 (for EnergyRaw) 🔽 ---
        if 'timestamp_utc' in raw_filtered and raw_filtered['timestamp_utc'] is not None:
            try:
                raw_filtered['timestamp_utc'] = datetime.fromisoformat(str(raw_filtered['timestamp_utc']).replace('Z', '+00:00'))
            except Exception as e:
                logger.error(f"無法解析 raw timestamp: {raw_filtered['timestamp_utc']} - 錯誤: {e}")
                del raw_filtered['timestamp_utc'] # 解析失敗則移除

        for field, target_type in numeric_fields.items():
            if field in raw_filtered and raw_filtered[field] is not None:
                try:
                    # 先轉成字串再轉目標類型，增加彈性
                    raw_filtered[field] = target_type(str(raw_filtered[field]))
                except (ValueError, TypeError) as e:
                    logger.warning(f"無法將 raw field '{field}' ({raw_filtered[field]}) 轉換為 {target_type.__name__}: {e}. 設為 None.")
                    raw_filtered[field] = None # 轉換失敗設為 None
        # --- 🔼 [FIX 1 & NEW] 結束 🔼 ---

        # 現在才過濾掉 None (轉換失敗的欄位會變 None)
        raw_insert_data = {k: v for k, v in raw_filtered.items() if k in models.EnergyRaw.__table__.columns and v is not None}

        raw_record = models.EnergyRaw(**raw_insert_data)
        db.add(raw_record)
        db.flush() # flush 以便後續可能需要 raw_record.id (雖然目前沒用到)

        try:
            logger.info("📊 使用 Agent 的 AI 清洗結果（不呼叫 cleaning-api）")
            energy_cleaned_fields = {
                "timestamp_utc", "gpu_model", "gpu_usage_percent", "gpu_power_watt",
                "cpu_power_watt", "memory_used_mb", "disk_read_mb_s", "disk_write_mb_s",
                "system_power_watt", "device_id", "user_id", "agent_version",
                "os_type", "os_version", "location", "is_anomaly", "anomaly_reason",
                "confidence_score" # 確保包含 confidence_score
            }
            # 從原始過濾後的 raw_filtered 開始，因為它已經做了類型轉換
            cleaned_filtered = {k: v for k, v in raw_filtered.items() if k in energy_cleaned_fields}

            # --- 🔽 [FIX 2 - Timestamp 已在上面處理過] 確保 cleaned_filtered 中的 timestamp 是 datetime 物件 🔽 ---
            # (不需要重複轉換 timestamp，直接使用 raw_filtered 的結果)
            if 'timestamp_utc' not in cleaned_filtered and 'timestamp_utc' in raw_filtered:
                 cleaned_filtered['timestamp_utc'] = raw_filtered['timestamp_utc'] # 從 raw 複製過來
            elif 'timestamp_utc' not in cleaned_filtered:
                 logger.warning(f"Cleaned data for {data.device_id} is missing timestamp.")
                 # 可以選擇 raise 錯誤或使用預設值，這裡先跳過
                 raise ValueError("Cleaned data missing timestamp after processing.")


            # --- 🔼 [FIX 2] 結束 🔼 ---

            # --- 🔽 [NEW] 確保 cleaned_filtered 中的數字欄位類型正確 🔽 ---
            # (不需要重複轉換，直接使用 raw_filtered 的結果)
            for field, target_type in numeric_fields.items():
                if field in energy_cleaned_fields: # 只處理 cleaned 表有的欄位
                    if field not in cleaned_filtered and field in raw_filtered:
                         cleaned_filtered[field] = raw_filtered[field] # 從 raw 複製
                    elif field in cleaned_filtered and not isinstance(cleaned_filtered[field], target_type) and cleaned_filtered[field] is not None:
                         # 如果不知為何類型又錯了，再嘗試轉一次 (理論上不應發生)
                         logger.warning(f"Retrying conversion for cleaned field '{field}'")
                         try:
                             cleaned_filtered[field] = target_type(str(cleaned_filtered[field]))
                         except (ValueError, TypeError):
                             cleaned_filtered[field] = None

            # --- 🔼 [NEW] 結束 🔼 ---


            if "is_anomaly" not in cleaned_filtered:
                cleaned_filtered["is_anomaly"] = False
            elif cleaned_filtered["is_anomaly"] is None: # Handle potential None from agent
                cleaned_filtered["is_anomaly"] = False

            if "anomaly_reason" not in cleaned_filtered:
                cleaned_filtered["anomaly_reason"] = None

            # 過濾掉 None 值 和 不在 EnergyCleaned 模型中的欄位
            cleaned_insert_data = {k: v for k, v in cleaned_filtered.items() if k in models.EnergyCleaned.__table__.columns and v is not None}


            # --- 🔽 [FIX] 檢查 cleaned_insert_data 是否缺少必要欄位 (例如 timestamp_utc) 🔽 ---
            if 'timestamp_utc' not in cleaned_insert_data:
                logger.error(f"FATAL: timestamp_utc missing before creating EnergyCleaned record for {data.device_id}. Data: {cleaned_filtered}")
                raise ValueError("timestamp_utc is missing for EnergyCleaned record")
            if 'device_id' not in cleaned_insert_data:
                 logger.error(f"FATAL: device_id missing before creating EnergyCleaned record for {data.device_id}. Data: {cleaned_filtered}")
                 raise ValueError("device_id is missing for EnergyCleaned record")
            # --- 🔼 [FIX] 結束 🔼 ---


            cleaned_record = models.EnergyCleaned(**cleaned_insert_data)
            db.add(cleaned_record)

            # --- 🔽 [NEW] 更新碳排放額度 (如果您的 main.py 有這段邏輯) 🔽 ---
            # 假設您有 update_carbon_usage 函數，並且需要 co2_kg
            # 您需要在這裡計算 co2_kg 或從 cleaned_insert_data 取得相關功耗數據
            # total_power_watt = cleaned_insert_data.get('system_power_watt', 0) or \
            #                    (cleaned_insert_data.get('cpu_power_watt', 0) + cleaned_insert_data.get('gpu_power_watt', 0))
            # if total_power_watt > 0:
            #      # 假設每秒計算一次，且有排放因子 EMISSION_FACTOR
            #      # co2_kg = calculate_co2(total_power_watt, interval_seconds=1, emission_factor=EMISSION_FACTOR)
            #      # update_carbon_usage(db, scope_type=QuotaScope.DEVICE, scope_id=data.device_id, co2_kg=co2_kg)
            #      # update_carbon_usage(db, scope_type=QuotaScope.USER, scope_id=cleaned_insert_data.get('user_id', 'unknown'), co2_kg=co2_kg)
            #      pass # 暫時跳過，因為計算邏輯不在此處
            # --- 🔼 [NEW] 結束 🔼 ---

# ... (前一行是 db.add(cleaned_record)) ...

            # ==========================================================
            # 🆕 3️⃣ 碳排放計算 (整合到您的新版 main.py)
            # ==========================================================

            # 從已清洗的資料中獲取功率數據
            system_power_watt = cleaned_insert_data.get('system_power_watt', 0.0) or 0.0
            cpu_power_watt = cleaned_insert_data.get('cpu_power_watt', 0.0) or 0.0
            gpu_power_watt = cleaned_insert_data.get('gpu_power_watt', 0.0)

            # 檢查是否有有效的功率數據
            if system_power_watt > 0 or cpu_power_watt > 0 or gpu_power_watt > 0:
                logger.info(f"🌍 開始計算來自 {data.device_id} 的碳排放")

                # 您的檔案頂部已經定義了 CARBON_EMISSION_FACTOR_EF = 0.474
                # 假設 Agent 回報數據的固定時間間隔（秒）
                CALCULATION_INTERVAL_SECONDS = 10  

                # 將功率 (W) 轉換為能源消耗 (kWh): Energy(kWh) = (Power(W) * Time(s)) / (1000 * 3600)
                conversion_factor = CALCULATION_INTERVAL_SECONDS / 3600000.0
                system_energy_kwh = system_power_watt * conversion_factor
                cpu_energy_kwh = cpu_power_watt * conversion_factor
                gpu_energy_kwh = gpu_power_watt * conversion_factor

                # 計算 CO2(kg) = Energy(KWh) X EF(kg/kWh)
                system_co2_kg = system_energy_kwh * CARBON_EMISSION_FACTOR_EF
                cpu_co2_kg = cpu_energy_kwh * CARBON_EMISSION_FACTOR_EF
                gpu_co2_kg = gpu_energy_kwh * CARBON_EMISSION_FACTOR_EF

                total_co2_kg = system_co2_kg # 暫時以 system power 為總量

                # (A) 寫入 CarbonEmissions (用於 Grafana 時序圖)
                try:
                    # 確保 timestamp_utc 是 datetime 物件
                    ts_utc = cleaned_insert_data.get('timestamp_utc')
                    if isinstance(ts_utc, str):
                        ts_utc = datetime.fromisoformat(ts_utc.replace('Z', '+00:00'))

                    if ts_utc: # 確保有時間戳才寫入
                        carbon_record = models.CarbonEmissions(
                            energy_cleaned_id=cleaned_record.id,
                            device_id=data.device_id,
                            timestamp_utc=ts_utc, # 使用已轉換的 datetime
                            cpu_power_watt=cpu_power_watt,
                            gpu_power_watt=gpu_power_watt,
                            system_power_watt=system_power_watt,
                            cpu_co2_kg=cpu_co2_kg,
                            gpu_co2_kg=gpu_co2_kg,
                            system_co2_kg=system_co2_kg,
                            total_co2_kg=total_co2_kg,
                            emission_factor=CARBON_EMISSION_FACTOR_EF,
                            calculation_interval_seconds=CALCULATION_INTERVAL_SECONDS,
                            calculation_method=f"Energy(kWh) * EF({CARBON_EMISSION_FACTOR_EF})"
                        )
                        db.add(carbon_record)
                        logger.info(f"🌍 成功計算碳排放 (CarbonEmissions): {total_co2_kg:.6f} kg CO2")
                    else:
                        logger.warning("CarbonEmissions 寫入失敗：缺少 timestamp_utc")

                except Exception as e:
                    logger.error(f"寫入 CarbonEmissions 失敗: {e}")

                # (B) 更新 CarbonQuotaUsage (用於配額統計)
                # 呼叫您檔案中既有的 update_carbon_usage 函數
                user_id = cleaned_insert_data.get('user_id', 'unknown')
                if total_co2_kg > 0 and user_id != 'unknown':
                    update_carbon_usage(
                        db=db,
                        scope_id=user_id,
                        co2_kg=total_co2_kg,
                        timestamp=cleaned_insert_data.get('timestamp_utc', datetime.now(timezone.utc)),
                        scope_type=QuotaScope.user
                    )

            else:
                logger.info(f"無有效功率數據，跳過碳計算 for {data.device_id}")

            # ==========================================================
            # 🆕 碳計算結束
            # ==========================================================

            db.commit() # Commit 包含 raw, cleaned, carbon_emissions, 和 quota_usage

            logger.info(f"✅ Successfully processed data from {data.device_id}")

        except Exception as processing_error:
                    db.rollback() # <--- 出錯時，撤銷所有資料庫寫入
                    logger.error(f"⚠️ 內部處理失敗 (processing_error) for {data.device_id}: {str(processing_error)}")
                    import traceback
                    logger.error(traceback.format_exc())

        except Exception as cleaning_error:
            db.rollback() # 如果 cleaning 或 commit 失敗，回滾 raw 記錄
            logger.error(f"❌ Failed during cleaning/commit for {data.device_id}: {str(cleaning_error)}")
            # 重新 raise 錯誤，以便外層 try-except 捕捉並回傳 500
            raise cleaning_error

        response_data = {"status": "success", "device": data.device_id, "auth_method": auth['method']}
        if 'fingerprint_check' in auth:
            response_data["fingerprint_check"] = auth['fingerprint_check']

        return response_data

    except Exception as e:
        db.rollback() # 確保任何部分失敗都回滾
        logger.error(f"❌ Failed to process data from {data.device_id}: {str(e)}")
        # 這裡的 detail 會顯示給使用者，保持簡潔
        raise HTTPException(status_code=500, detail=f"Processing failed: {type(e).__name__}") # 只顯示錯誤類型

# ==========================================================================
# 管理端點 - 安全存取版本
# ==========================================================================

@app.get("/admin/dashboard")
async def get_dashboard(db: Session = Depends(get_db)):
    """取得後台總覽資訊"""
    try:
        # 基本統計
        total_records = db.query(models.EnergyRaw).count()
        unique_devices = db.query(func.count(distinct(models.EnergyRaw.device_id))).scalar()
        
        # 今日統計
        today = datetime.now().date()
        today_records = db.query(models.EnergyRaw).filter(
            models.EnergyRaw.timestamp_utc >= today
        ).count()
        
        # 風險等級統計（安全檢查）
        try:
            risk_stats = db.query(
                models.EnergyRaw.risk_level,
                func.count(models.EnergyRaw.risk_level)
            ).filter(
                models.EnergyRaw.risk_level.isnot(None)
            ).group_by(models.EnergyRaw.risk_level).all()
            
            risk_summary = {level: count for level, count in risk_stats}
        except:
            risk_summary = {}
        
        # 白名單設備統計
        try:
            whitelisted_devices = db.query(models.AuthorizedDevice).filter(
                models.AuthorizedDevice.is_active == True
            ).count()
        except:
            whitelisted_devices = 0
        
        return {
            "total_records": total_records,
            "unique_devices": unique_devices,
            "records_today": today_records,
            "risk_summary": risk_summary,
            "whitelisted_devices": whitelisted_devices,
            "last_updated": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Dashboard query failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Dashboard error: {str(e)}")

@app.get("/admin/device-ids")
async def get_device_ids(db: Session = Depends(get_db)):
    """取得所有設備ID列表（已優化 N+1 查詢）"""
    try:
        # 步驟 1: 建立一個子查詢，使用 ROW_NUMBER() 
        # 根據 device_id 分組，並依照 timestamp_utc 降冪排序
        subq = db.query(
            models.EnergyRaw,
            func.row_number().over(
                partition_by=models.EnergyRaw.device_id,
                order_by=models.EnergyRaw.timestamp_utc.desc()
            ).label('rn')
        ).subquery('latest_records_subquery')

        # 步驟 2: 只查詢 rn = 1 (即每個分組中的最新一筆) 的記錄
        latest_records = db.query(subq).filter(
            subq.c.rn == 1
        ).all()

        # 步驟 3: 格式化輸出
        id_list = []
        for row in latest_records:
            # 由於 'row' 現在是子查詢的結果，我們需要透過 .c 屬性來存取欄位
            id_list.append({
                "device_id": row.device_id,
                "user_id": getattr(row, 'user_id', 'Unknown'),
                "last_seen": row.timestamp_utc,
                "risk_level": getattr(row, 'risk_level', 'unknown'),
                "gpu_model": getattr(row, 'gpu_model', 'Unknown'),
                "os_type": getattr(row, 'os_type', 'Unknown'),
                "similarity_score": getattr(row, 'similarity_score', 0.0)
            })
        
        return {
            "device_ids": id_list,
            "total_count": len(id_list)
        }
    except Exception as e:
        logger.error(f"Device IDs query failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query error: {str(e)}")

@app.get("/admin/devices-simple")
async def get_devices_simple(db: Session = Depends(get_db)):
    """取得所有設備的簡化列表"""
    try:
        # 取得最近的記錄並去重
        devices = db.query(models.EnergyRaw).order_by(
            models.EnergyRaw.timestamp_utc.desc()
        ).limit(200).all()
        
        # 去重並取得每個設備的最新記錄
        device_dict = {}
        for device in devices:
            if device.device_id not in device_dict:
                device_dict[device.device_id] = device
        
        device_list = []
        for device_id, device in device_dict.items():
            device_info = {
                "device_id": device.device_id,
                "user_id": getattr(device, 'user_id', 'Unknown'),
                "gpu_model": getattr(device, 'gpu_model', 'Unknown'),
                "os_type": getattr(device, 'os_type', 'Unknown'),
                "os_version": getattr(device, 'os_version', 'Unknown'),
                "agent_version": getattr(device, 'agent_version', 'Unknown'),
                "location": getattr(device, 'location', 'Unknown'),
                "last_seen": device.timestamp_utc,
                "risk_level": getattr(device, 'risk_level', 'unknown'),
                "device_fingerprint": getattr(device, 'device_fingerprint', 'N/A'),
                "similarity_score": getattr(device, 'similarity_score', 0.0),
                "cpu_power": getattr(device, 'cpu_power_watt', 0.0),
                "gpu_power": getattr(device, 'gpu_power_watt', 0.0),
                "system_power": getattr(device, 'system_power_watt', 0.0)
            }
            device_list.append(device_info)
        
        return {
            "devices": device_list,
            "total_count": len(device_list)
        }
    except Exception as e:
        logger.error(f"Devices query failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query error: {str(e)}")

@app.get("/admin/device/{device_id}")
async def get_device_simple_details(device_id: str, db: Session = Depends(get_db)):
    """取得特定設備的詳細記錄（簡化版）"""
    try:
        # 取得設備最近10筆記錄
        records = db.query(models.EnergyRaw).filter(
            models.EnergyRaw.device_id == device_id
        ).order_by(models.EnergyRaw.timestamp_utc.desc()).limit(10).all()
        
        if not records:
            raise HTTPException(status_code=404, detail="Device not found")
        
        # 統計資訊
        total_records = db.query(models.EnergyRaw).filter(
            models.EnergyRaw.device_id == device_id
        ).count()
        
        latest_record = records[0]
        
        return {
            "device_info": {
                "device_id": device_id,
                "user_id": getattr(latest_record, 'user_id', 'Unknown'),
                "gpu_model": getattr(latest_record, 'gpu_model', 'Unknown'),
                "os_type": getattr(latest_record, 'os_type', 'Unknown'),
                "os_version": getattr(latest_record, 'os_version', 'Unknown'),
                "agent_version": getattr(latest_record, 'agent_version', 'Unknown'),
                "location": getattr(latest_record, 'location', 'Unknown'),
                "first_seen": records[-1].timestamp_utc,
                "last_seen": latest_record.timestamp_utc
            },
            "statistics": {
                "total_records": total_records
            },
            "fingerprint_history": [
                {
                    "timestamp": r.timestamp_utc,
                    "fingerprint": getattr(r, 'device_fingerprint', 'N/A'),
                    "risk_level": getattr(r, 'risk_level', 'unknown'),
                    "similarity_score": getattr(r, 'similarity_score', 0.0)
                } for r in records if getattr(r, 'device_fingerprint', None)
            ],
            "recent_records": [
                {
                    "timestamp": r.timestamp_utc,
                    "cpu_power": getattr(r, 'cpu_power_watt', 0.0),
                    "gpu_power": getattr(r, 'gpu_power_watt', 0.0),
                    "system_power": getattr(r, 'system_power_watt', 0.0),
                    "risk_level": getattr(r, 'risk_level', 'unknown'),
                    "similarity_score": getattr(r, 'similarity_score', 0.0)
                } for r in records
            ]
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Device details query failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query error: {str(e)}")

@app.get("/admin/high-risk")
async def get_high_risk_simple(db: Session = Depends(get_db)):
    """取得高風險設備列表（簡化版）"""
    try:
        high_risk_devices = db.query(models.EnergyRaw).filter(
            models.EnergyRaw.risk_level == "high"
        ).order_by(models.EnergyRaw.timestamp_utc.desc()).limit(20).all()
        
        devices = []
        for device in high_risk_devices:
            devices.append({
                "device_id": device.device_id,
                "user_id": getattr(device, 'user_id', 'Unknown'),
                "timestamp": device.timestamp_utc,
                "risk_level": getattr(device, 'risk_level', 'unknown'),
                "similarity_score": getattr(device, 'similarity_score', 0.0),
                "device_fingerprint": getattr(device, 'device_fingerprint', 'N/A'),
                "gpu_model": getattr(device, 'gpu_model', 'Unknown')
            })
        
        return {
            "high_risk_devices": devices,
            "count": len(devices)
        }
    except Exception as e:
        logger.error(f"High risk devices query failed: {str(e)}")
        return {
            "high_risk_devices": [],
            "count": 0,
            "error": str(e)
        }

# ==========================================================================
# 原有的設備管理端點（白名單相關）
# ==========================================================================

@app.get("/admin/devices", response_model=List[schemas.DeviceResponse])
async def list_devices(db: Session = Depends(get_db)):
    """列出所有授權設備"""
    manager = MACManager(db)
    return manager.list_devices()

@app.post("/admin/devices")
async def add_device(device_data: schemas.DeviceCreate, db: Session = Depends(get_db)):
    """新增設備到白名單"""
    manager = MACManager(db)
    success = manager.add_device(
        device_data.mac_address,
        device_data.device_name,
        device_data.user_name,
        device_data.notes
    )
    
    if success:
        return {"status": "success", "message": "Device added to whitelist"}
    else:
        raise HTTPException(status_code=400, detail="Failed to add device or device already exists")

@app.delete("/admin/devices/{mac_address}")
async def remove_device(mac_address: str, db: Session = Depends(get_db)):
    """從白名單移除設備"""
    manager = MACManager(db)
    success = manager.remove_device(mac_address)
    
    if success:
        return {"status": "success", "message": "Device removed from whitelist"}
    else:
        raise HTTPException(status_code=404, detail="Device not found")

@app.get("/admin/devices/{mac_address}", response_model=schemas.DeviceResponse)
async def get_device_info(mac_address: str, db: Session = Depends(get_db)):
    """取得設備詳細資訊"""
    manager = MACManager(db)
    device = manager.get_device(mac_address)
    
    if device:
        return device
    else:
        raise HTTPException(status_code=404, detail="Device not found")

# ==========================================================================
# 系統監控端點
# ==========================================================================

@app.get("/metrics")
async def get_metrics(db: Session = Depends(get_db)):
    """取得系統指標"""
    try:
        today = datetime.now().date()
        
        raw_count = db.query(models.EnergyRaw).filter(
            models.EnergyRaw.timestamp_utc >= today
        ).count()
        
        cleaned_count = db.query(models.EnergyCleaned).filter(
            models.EnergyCleaned.timestamp_utc >= today
        ).count()
        
        try:
            active_devices = db.query(models.AuthorizedDevice).filter(
                models.AuthorizedDevice.is_active == True
            ).count()
        except:
            active_devices = 0
        
        # 異常設備統計
        try:
            high_risk_count = db.query(models.EnergyRaw).filter(
                models.EnergyRaw.timestamp_utc >=today ,
                models.EnergyRaw.risk_level == "high"
            ).count()
            
            medium_risk_count = db.query(models.EnergyRaw).filter(
                models.EnergyRaw.timestamp_utc >= today,
                models.EnergyRaw.risk_level == "medium"
            ).count()
        except:
            high_risk_count = 0
            medium_risk_count = 0
        
        return {
            "records_today": {
                "raw": raw_count,
                "cleaned": cleaned_count,
                "success_rate": f"{(cleaned_count/raw_count*100):.1f}%" if raw_count > 0 else "0%"
            },
            "active_devices": active_devices,
            "security_status": {
                "high_risk_devices": high_risk_count,
                "medium_risk_devices": medium_risk_count,
                "total_anomalies": high_risk_count + medium_risk_count
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Metrics collection failed: {str(e)}")
        return {"error": "Unable to collect metrics"}
    
#  GET /data/ 
@app.get("/data/", response_model=list[schemas.EnergyRawResponse])
def read_data(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """
    讀取所有儲存的原始能源數據，並使用正確的資料庫欄位名稱。
    """
    db_records = db.query(models.EnergyRaw).order_by(models.EnergyRaw.id.desc()).offset(skip).limit(limit).all()

    result = []
    for record in db_records:
        # 建構包含所有實際欄位的字典
        raw_data_dict = {
            "timestamp_utc": record.timestamp_utc,
            "gpu_model": getattr(record, 'gpu_model', None),
            "gpu_usage_percent": getattr(record, 'gpu_usage_percent', None),
            "gpu_power_watt": getattr(record, 'gpu_power_watt', None),
            "cpu_power_watt": getattr(record, 'cpu_power_watt', None),
            "memory_used_mb": getattr(record, 'memory_used_mb', None),
            "disk_read_mb_s": getattr(record, 'disk_read_mb_s', None),
            "disk_write_mb_s": getattr(record, 'disk_write_mb_s', None),
            "system_power_watt": getattr(record, 'system_power_watt', None),
            "device_id": record.device_id,
            "user_id": getattr(record, 'user_id', None),
            "agent_version": getattr(record, 'agent_version', None),
            "os_type": getattr(record, 'os_type', None),
            "os_version": getattr(record, 'os_version', None),
            "location": getattr(record, 'location', None),
            "cpu_model": getattr(record, 'cpu_model', None),
            "cpu_count": getattr(record, 'cpu_count', None),
            "total_memory": getattr(record, 'total_memory', None),
            "disk_partitions": getattr(record, 'disk_partitions', None),
            "network_interfaces": getattr(record, 'network_interfaces', None),
            "platform_machine": getattr(record, 'platform_machine', None),
            "platform_architecture": getattr(record, 'platform_architecture', None)
        }
        
        result.append({
            "id": record.id,
            "timestamp_utc": record.timestamp_utc,
            "device_id": record.device_id,
            "user_id": getattr(record, 'user_id', None),
            "raw_data": raw_data_dict,  # 現在包含所有原始數據
            "mac_address": getattr(record, 'mac_address', None),
            "is_cleaned": getattr(record, 'is_cleaned', False),
            "risk_level": getattr(record, 'risk_level', None),
            "device_fingerprint": getattr(record, 'device_fingerprint', None)
        })

    return result

# ==========================================================================
# 新增管理碳排額度的 API 端點
# ==========================================================================

@app.post("/admin/quotas", response_model=CarbonQuotaResponse)
def create_quota(quota: CarbonQuotaCreate, db: Session = Depends(get_db)):
    """建立一條新的碳排額度規則"""

    # 檢查是否已存在
    db_quota = db.query(CarbonQuota).filter(
        CarbonQuota.scope_type == quota.scope_type,
        CarbonQuota.scope_id == quota.scope_id,
        CarbonQuota.period == quota.period
    ).first()

    if db_quota:
        raise HTTPException(status_code=400, detail="Quota rule for this scope and period already exists")

    db_quota = CarbonQuota(**quota.model_dump())
    db.add(db_quota)
    db.commit()
    db.refresh(db_quota)
    return db_quota

@app.get("/admin/quotas/{scope_type}/{scope_id}", response_model=List[CarbonQuotaResponse])
def get_quotas_for_scope(scope_type: QuotaScope, scope_id: str, db: Session = Depends(get_db)):
    """取得特定範圍（例如某個 User）的所有額度規則"""
    quotas = db.query(CarbonQuota).filter(
        CarbonQuota.scope_type == scope_type,
        CarbonQuota.scope_id == scope_id
    ).all()
    return quotas

@app.delete("/admin/quotas/{quota_id}")
def delete_quota(quota_id: int, db: Session = Depends(get_db)):
    """刪除一條額度規則"""
    db_quota = db.query(CarbonQuota).filter(CarbonQuota.id == quota_id).first()
    if not db_quota:
        raise HTTPException(status_code=404, detail="Quota rule not found")

    db.delete(db_quota)
    db.commit()
    return {"status": "success", "message": f"Quota rule {quota_id} deleted"}

@app.get("/admin/usage/{scope_type}/{scope_id}", response_model=List[CarbonQuotaUsageResponse])
def get_usage_for_scope(scope_type: QuotaScope, scope_id: str, db: Session = Depends(get_db)):
    """
    取得特定範圍（例如某個 User）目前的碳排用量（日/月/年）
    並計算剩餘額度
    """
    now = datetime.now(timezone.utc)
    period_keys = {
        QuotaPeriod.daily: now.strftime('%Y-%m-%d'),
        QuotaPeriod.monthly: now.strftime('%Y-%m'),
        QuotaPeriod.yearly: now.strftime('%Y')
    }

    results = []

    # 1. 取得所有額度規則
    quotas = db.query(CarbonQuota).filter(
        CarbonQuota.scope_type == scope_type,
        CarbonQuota.scope_id == scope_id,
        CarbonQuota.is_active == True
    ).all()

    # 轉換為字典以便快速查找
    quota_limits = {q.period: q.limit_co2_kg for q in quotas}

    # 2. 取得目前的用量
    usages = db.query(CarbonQuotaUsage).filter(
        CarbonQuotaUsage.scope_type == scope_type,
        CarbonQuotaUsage.scope_id == scope_id,
        and_(
            CarbonQuotaUsage.period_key.in_(period_keys.values())
        )
    ).all()

    # 3. 組合結果
    for period_type, period_key in period_keys.items():

        # 查找對應的用量
        usage_record = next(
            (u for u in usages if u.period_type == period_type and u.period_key == period_key),
            None # 如果找不到，表示本週期還沒有用量
        )

        used_kg = usage_record.used_co2_kg if usage_record else 0.0
        limit_kg = quota_limits.get(period_type) # 取得設定的限制

        remaining_kg = None
        if limit_kg is not None:
            remaining_kg = limit_kg - used_kg

        results.append(
            CarbonQuotaUsageResponse(
                scope_type=scope_type,
                scope_id=scope_id,
                period_type=period_type,
                period_key=period_key,
                used_co2_kg=round(used_kg, 6),
                limit_co2_kg=limit_kg,
                remaining_co2_kg=round(remaining_kg, 6) if remaining_kg is not None else None
            )
        )

    return results

