import os, asyncio, datetime as dt
import json
from pathlib import Path
from typing import Dict, Any, List
from urllib.parse import urlparse
import socket 

import numpy as np
import pandas as pd
import joblib
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from sqlalchemy import create_engine, text
from tensorflow.keras.models import load_model
from pydantic import BaseModel 
from sqlalchemy.exc import OperationalError

# ---------- env & config ----------
load_dotenv()

# 核心設定
DATABASE_URL = os.getenv("DATABASE_URL") or os.getenv("DB_URL")
LOOKBACK_MIN = int(os.getenv("BATCH_LOOKBACK_MINUTES", "720"))
STEP_MIN     = int(os.getenv("STEP_MINUTES", "1"))
MODEL_VERSION= os.getenv("MODEL_VERSION", "lstm_v1")
RUN_INTERVAL = int(os.getenv("RUN_INTERVAL_SECONDS", "60"))
EF           = float(os.getenv("EF", "0.502"))
WINDOW       = int(os.getenv("WINDOW", "72"))
DELTA_HR     = STEP_MIN / 60.0

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL / DB_URL 未設定")

# ----------------------------------------------------------------------------------
# *** 關鍵修正：解決 主機環境 無法識別 'db' 的問題 ***
# ----------------------------------------------------------------------------------
if 'db:5432' in DATABASE_URL:
    try:
        socket.gethostbyname('db')
    except socket.error:
        DATABASE_URL = DATABASE_URL.replace("db:5432", "localhost:5433")
        print(f"⚠️ Swapping DB host to {DATABASE_URL.split('@')[-1]} for host environment.")
# ----------------------------------------------------------------------------------

u = urlparse(DATABASE_URL)
print("Using DATABASE_URL:", f"{u.scheme}://{u.username}:******@{u.hostname}:{u.port}{u.path}")
print(f"STEP_MIN={STEP_MIN}, WINDOW={WINDOW}, LOOKBACK_MIN={LOOKBACK_MIN}, EF={EF}, MODEL_VERSION={MODEL_VERSION}")

# 初始化 DB 引擎
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

# ----------------------------------------------------------------------------------
# *** 修正 1：宣告 Model 與 Scaler 為全域變數，但不載入 ***
# ----------------------------------------------------------------------------------
model = None
scaler = None
# ----------------------------------------------------------------------------------

# --- Pydantic Model (新增用於預測的數據模型) ---
class PredictMetric(BaseModel):
    timestamp: dt.datetime
    predicted_power_w: float 
    predicted_co2_kg: float 
    strategy: Dict[str, Any]

# --- 輔助函式定義 (Function Definitions) ---

def floor_to_step(dt_obj: dt.datetime, step_min: int) -> dt.datetime:
    """將 datetime 物件向下取整到 STEP_MIN 的倍數分鐘 (精確對齊)"""
    if step_min <= 0: return dt_obj
        
    delta = dt.timedelta(minutes=dt_obj.minute % step_min, 
                         seconds=dt_obj.second, 
                         microseconds=dt_obj.microsecond)
    return dt_obj - delta

def get_power_thresholds() -> Dict[str, float]:
    """從資料庫計算用電量 P20, P80 門檻 (簡化版，實際應查詢 DB)"""
    return {"p20": 100.0, "p80": 400.0} 

def recommend_strategy(pred_w: float, band_thresholds: dict) -> Dict[str, Any]:
    """根據預測功耗回傳結構化的策略建議 (字典/JSON)"""
    p80 = band_thresholds.get("p80", 400.0)
    p20 = band_thresholds.get("p20", 100.0)

    if pred_w >= p80:
        return {"load_level": "HIGH", "summary": "高負載預測：建議立即採取節能措施。", 
                "recommendations": ["限制 GPU 功耗", "批次任務重新排程"]}
    elif pred_w <= p20:
        return {"load_level": "LOW", "summary": "低功耗預測：適合執行耗時任務。", 
                "recommendations": ["開始執行模型訓練或數據備份"]}
    else:
        return {"load_level": "MID", "summary": "中等功耗預測：持續監控即可。", 
                "recommendations": ["維持正常監控"]}


def ensure_pred_table(engine):
    """同步函式：確保預測結果表存在 (在異步環境中需用 to_thread 呼叫)"""
    sql = text("""
    CREATE TABLE IF NOT EXISTS carbon_emissions (
      timestamp_from      timestamptz NOT NULL,
      timestamp_to        timestamptz NOT NULL,
      horizon_steps       integer     NOT NULL,
      predicted_power_w   double precision NOT NULL,
      predicted_co2_kg    double precision NOT NULL,
      model_version       text        NOT NULL,
      recommended_strategy jsonb,
      created_at          timestamptz NOT NULL DEFAULT now(),
      PRIMARY KEY (timestamp_to, model_version)
    );
    """)
    with engine.begin() as conn:
        conn.execute(sql)

# ----------------------------------------------------------------------------------
# *** 修正 2：新增同步模型載入函式 (供 on_startup 呼叫) ***
# ----------------------------------------------------------------------------------
def _load_model_sync():
    """同步載入模型與 Scaler"""
    models_dir = Path(__file__).resolve().parents[1] / "models"
    keras_path = models_dir / "lstm_carbon_model.keras"
    h5_path    = models_dir / "lstm_carbon_model.h5"
    scaler_path= models_dir / "scaler_power.pkl"

    model_path = keras_path if keras_path.exists() else h5_path
    if not model_path.exists():
        raise FileNotFoundError(f"找不到模型檔：{keras_path} 或 {h5_path}")

    if not scaler_path.exists():
        raise FileNotFoundError(f"找不到 scaler 檔：{scaler_path}")

    print(f"Loading model: {model_path.name}")
    
    # 這是同步操作
    loaded_model = load_model(model_path, compile=False)
    loaded_scaler = joblib.load(scaler_path)
    
    return loaded_model, loaded_scaler
# ----------------------------------------------------------------------------------

# ---------- data access & preprocessing (多變量輸入) ----------
def fetch_power_series(end_ts: dt.datetime, minutes: int) -> pd.DataFrame:
    # 檢查模型是否已載入
    if model is None or scaler is None:
        # 如果模型還沒載入就呼叫，則拋出錯誤 (通常只發生在啟動時的極短時間)
        raise RuntimeError("Model service is not fully initialized yet.")
        
    """
    從 energy_cleaned 取回數據，重採樣，並新增時間特徵 (hour, dayofweek)。
    """
    start_ts = end_ts - dt.timedelta(minutes=minutes)

    sql = text("""
        SELECT
          (timestamp_utc)::timestamptz AS ts,
          system_power_watt
        FROM energy_cleaned
        WHERE timestamp_utc IS NOT NULL
          AND system_power_watt IS NOT NULL
          AND (timestamp_utc)::timestamptz >  :start
          AND (timestamp_utc)::timestamptz <= :end
        ORDER BY (timestamp_utc)::timestamptz
    """)

    with engine.connect() as conn:
        raw = pd.read_sql(sql, conn, params={"start": start_ts, "end": end_ts}, parse_dates=["ts"])

    if raw.empty:
        return raw.rename(columns={"ts": "timestamp"})

    df = raw.rename(columns={"ts": "timestamp"}).set_index("timestamp").sort_index()
    rule = f"{STEP_MIN}min"
    df = df.resample(rule).mean()
    df["system_power_watt"] = df["system_power_watt"].ffill().bfill()
    df = df.dropna(subset=["system_power_watt"]).reset_index()
    
    df['hour'] = df['timestamp'].dt.hour
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    
    df['system_power_watt'] = df['system_power_watt'].astype(float)
    df['hour'] = df['hour'].astype(float)
    df['dayofweek'] = df['dayofweek'].astype(float)

    return df

def predict_next_power_w(df: pd.DataFrame) -> float:
    if model is None or scaler is None:
        raise RuntimeError("Model service is not fully initialized yet.")
        
    """
    取最後 WINDOW 步的多變量特徵，經 scaler → LSTM → inverse_transform，得到下一步功率(W)。
    """
    features = df[['system_power_watt', 'hour', 'dayofweek']].values.astype(float)

    if len(features) < WINDOW:
        raise ValueError(f"需要至少 {WINDOW} 筆資料（當前 {len(features)}），請增加 LOOKBACK_MIN 或降低 WINDOW。")

    last_window = features[-WINDOW:]
    last_scaled = scaler.transform(last_window).reshape(1, WINDOW, last_window.shape[1])
    
    y_scaled    = model.predict(last_scaled, verbose=0)
    
    temp_array = np.zeros((y_scaled.shape[0], scaler.n_features_in_))
    temp_array[:, 0] = y_scaled[:, 0]
    
    y_watt = scaler.inverse_transform(temp_array)[:, 0][0]

    return float(y_watt)

def upsert_carbon_emission(ts_from, ts_to, steps, pw, co2, strategy: Dict[str, Any]):
    """
    將預測結果、CO2 和策略寫入 carbon_emissions 表格。
    """
    strategy_json = json.dumps(strategy) 
    
    sql = text("""
        INSERT INTO carbon_emissions
        (timestamp_from, timestamp_to, horizon_steps, predicted_power_w, predicted_co2_kg, model_version, recommended_strategy)
        VALUES (:from, :to, :h, :pw, :co2, :mv, :strategy)
        ON CONFLICT (timestamp_to, model_version) DO UPDATE
        SET predicted_power_w = EXCLUDED.predicted_power_w,
            predicted_co2_kg  = EXCLUDED.predicted_co2_kg,
            recommended_strategy = EXCLUDED.recommended_strategy,
            timestamp_from    = EXCLUDED.timestamp_from,
            horizon_steps     = EXCLUDED.horizon_steps;
    """)
    with engine.begin() as conn:
        conn.execute(sql, {
            "from": ts_from, "to": ts_to, "h": steps,
            "pw": pw, "co2": co2, "mv": MODEL_VERSION,
            "strategy": strategy_json
        })

# ---------- background loop ----------
async def loop_job():
    while True:
        # 關鍵修正：使用 floor_to_step 確保時間戳記對齊到整分鐘
        now_raw = dt.datetime.utcnow()
        ts_to = floor_to_step(now_raw + dt.timedelta(minutes=STEP_MIN), STEP_MIN)
        ts_from = ts_to - dt.timedelta(minutes=STEP_MIN)
        
        try:
            df = fetch_power_series(ts_from, LOOKBACK_MIN) # 抓取數據至上一個整分鐘
            
            if df.empty:
                print(f"[{ts_to.isoformat()}Z] Job error: No data in lookback window ({LOOKBACK_MIN} min).")
            else:
                pred_power_w = predict_next_power_w(df)
                kWh = (pred_power_w / 1000.0) * DELTA_HR
                co2 = kWh * EF
                
                thresholds = get_power_thresholds()
                strategy = recommend_strategy(pred_power_w, thresholds)

                # 寫入操作是同步的，使用 asyncio.to_thread 確保不阻塞事件循環
                await asyncio.to_thread(upsert_carbon_emission, ts_from, ts_to, 1, pred_power_w, co2, strategy)

                print(f"[{ts_to.isoformat()}Z] Pred={pred_power_w:.2f} W | kWh={kWh:.6f} | CO2={co2:.6f} kg | Strategy: {strategy['summary']}")
        
        except RuntimeError as e:
            # 捕獲模型未初始化的錯誤
            print(f"[{dt.datetime.utcnow().isoformat()}Z] Job Warning: {e}")
        except ValueError as e:
            print(f"[{dt.datetime.utcnow().isoformat()}Z] Job error (Data/Predict): {e}")
        except OperationalError as e:
            # 捕獲 DB 連線錯誤，並嘗試重連
            print(f"[{dt.datetime.utcnow().isoformat()}Z] DB Operational Error: {e}")
            await asyncio.sleep(5) # 短暫等待後重試
        except Exception as e:
            print(f"[{dt.datetime.utcnow().isoformat()}Z] Job error (General): {repr(e)}")

        await asyncio.sleep(RUN_INTERVAL)

# ---------- FastAPI endpoints ----------
app = FastAPI(title="Prediction API (LSTM → Carbon)")

@app.on_event("startup")
async def on_startup():
    global model, scaler
    # 關鍵修正：將同步的 DB 表格檢查移到異步啟動事件中執行
    await asyncio.to_thread(ensure_pred_table, engine)

    # ----------------------------------------------------------------------------------
    # *** 修正 3：在異步啟動事件中載入模型，不阻塞事件循環 ***
    # ----------------------------------------------------------------------------------
    try:
        model, scaler = await asyncio.to_thread(_load_model_sync)
        print("INFO: Model and Scaler loaded successfully.")
    except FileNotFoundError as e:
        print(f"FATAL: Model initialization failed: {e}")
        # 如果模型載入失敗，這裡應該拋出錯誤，阻止應用程式運行
        # 由於 uvicorn 的行為，我們不拋出錯誤，但模型會保持 None，後續調用會報錯
    # ----------------------------------------------------------------------------------

    # 開始背景任務迴圈
    asyncio.create_task(loop_job())

@app.get("/health")
def health():
    return {
        "status": "ok" if model is not None else "model_unloaded",
        "model_version": MODEL_VERSION,
        "step_minutes": STEP_MIN,
        "window": WINDOW,
        "lookback_min": LOOKBACK_MIN
    }

@app.post("/run-once", response_model=PredictMetric)
def run_once():
    # 關鍵修正：確保手動執行時，時間戳記也對齊
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model service not ready yet. Please wait for startup.")
        
    now_raw = dt.datetime.utcnow()
    ts_to = floor_to_step(now_raw + dt.timedelta(minutes=STEP_MIN), STEP_MIN)
    ts_from = ts_to - dt.timedelta(minutes=STEP_MIN)
    
    df = fetch_power_series(ts_from, LOOKBACK_MIN)
    if df.empty:
        raise HTTPException(400, detail=f"no data in last {LOOKBACK_MIN} minutes")

    pred_power_w = predict_next_power_w(df)
    kWh = (pred_power_w / 1000.0) * DELTA_HR
    co2 = kWh * EF

    thresholds = get_power_thresholds()
    strategy = recommend_strategy(pred_power_w, thresholds)
    
    # 寫入操作是同步的，在 run_once 路由中運行是 OK 的 (不阻塞主循環)
    upsert_carbon_emission(ts_from, ts_to, 1, pred_power_w, co2, strategy) 

    return PredictMetric(
        timestamp=ts_to,
        predicted_power_w=pred_power_w,
        predicted_co2_kg=co2,
        strategy=strategy
    )