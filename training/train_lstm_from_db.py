import os
import random
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from urllib.parse import urlparse
from datetime import datetime, timezone

# ========= 隨機種子（可重現） =========
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

# ========= 先載入 .env，再取環境變數 =========
load_dotenv() 

# 載入環境變數
DB_URL = os.getenv("DB_URL") or os.getenv("DATABASE_URL")
STEP_MIN = int(os.getenv("STEP_MINUTES", "1"))
# 修正 1：定義 WINDOW 實驗範圍
WINDOW_LIST = [60, 90, 120, 180, 300] 
WINDOW = int(os.getenv("WINDOW", "60"))
EPOCHS = int(os.getenv("EPOCHS", "500"))
BATCH = int(os.getenv("BATCH", "128"))

# 可選訓練期間參數
TRAIN_DAYS = os.getenv("TRAIN_DAYS")
TRAIN_START_UTC = os.getenv("TRAIN_START_UTC")
TRAIN_END_UTC = os.getenv("TRAIN_END_UTC")

# ----------------------------------------------------------------------
# *** 關鍵修正：解決 DB_URL 主機名問題 (使 Windows 主機可以連線) ***
# ----------------------------------------------------------------------
if DB_URL and 'db:5432' in DB_URL:
    DB_URL = DB_URL.replace("db:5432", "localhost:5433")
    print("WARNING: DB_URL host swapped to localhost:5433 for local execution.")
# ----------------------------------------------------------------------

if not DB_URL:
    raise RuntimeError("DB_URL 未設定，請檢查 .env 檔案。")
u = urlparse(DB_URL)
print("Using DB_URL:", f"{u.scheme}://{u.username}:******@{u.hostname}:{u.port}{u.path}")
print(f"STEP_MIN={STEP_MIN}, EPOCHS={EPOCHS}, BATCH={BATCH}") # 移除 WINDOW 打印，因為 WINDOWs 是迭代的

# ========= 欄位名稱 =========
TIME_COL = "timestamp"
PWR_COL = "system_power_watt"
N_FEATURES = 4 # 4 個特徵 (power_w, hour, dayofweek, lag_24h)

# ========= SQL（讀取資料） =========
where_extra, params = [], {}
if TRAIN_DAYS and TRAIN_DAYS.isdigit():
    where_extra.append("(timestamp_utc)::timestamptz >= now() - interval :train_days")
    params["train_days"] = f"{int(TRAIN_DAYS)} days"
if TRAIN_START_UTC:
    where_extra.append("(timestamp_utc)::timestamptz >= :start_utc")
    params["start_utc"] = TRAIN_START_UTC
if TRAIN_END_UTC:
    where_extra.append("(timestamp_utc)::timestamptz <= :end_utc")
    params["end_utc"] = TRAIN_END_UTC

where_clause = (" AND " + " AND ".join(where_extra) + "\n") if where_extra else ""

SQL = f"""
SELECT
  (timestamp_utc)::timestamptz AS {TIME_COL},
  {PWR_COL}
FROM energy_cleaned
WHERE timestamp_utc IS NOT NULL
  AND {PWR_COL} IS NOT NULL
{where_clause}ORDER BY (timestamp_utc)::timestamptz
""".strip()
print("DEBUG SQL:\n", SQL)

# ========= 讀取資料並進行基礎清洗 (數據只讀取一次) =========
engine = create_engine(DB_URL, pool_pre_ping=True)
df_raw = pd.read_sql(text(SQL), engine, params=params, parse_dates=[TIME_COL])
if df_raw.empty:
    raise RuntimeError("energy_cleaned 查無資料（符合條件為 0）。")

print(f"Loaded rows: {len(df_raw)} | time span: {df_raw[TIME_COL].min()} → {df_raw[TIME_COL].max()}")

# ========= 數據準備函式 (在循環內調用) =========

def prepare_data(df_input, window_size):
    """執行特徵工程、對齊、切分和正規化"""
    df = df_input.set_index(TIME_COL).sort_index().copy()
    df = df.resample(f"{STEP_MIN}min").mean()
    df[PWR_COL] = df[PWR_COL].ffill().bfill()
    
    # 尖峰平滑
    q1, q3 = df[PWR_COL].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    df[PWR_COL] = df[PWR_COL].clip(lower, upper)
    df[PWR_COL] = df[PWR_COL].rolling(window=3, center=True, min_periods=1).median()

    # >>> 特徵工程 <<<
    df.index = pd.to_datetime(df.index)
    df['hour'] = df.index.hour.astype(float)
    df['dayofweek'] = df.index.dayofweek.astype(float)
    
    # Day-Ahead Lag 特徵 (1440 分鐘)
    LAG_MINUTES = 24 * 60 
    df['power_lag_24h'] = df[PWR_COL].shift(LAG_MINUTES)
    df = df.dropna()

    features = df[[PWR_COL, 'hour', 'dayofweek', 'power_lag_24h']].astype(float).values

    if len(features) < window_size + 2:
        # 自動調整 WINDOW 邏輯 (由於是實驗，直接跳過或報錯)
        raise RuntimeError(f"數據量不足以建立 WINDOW={window_size} 的序列。")

    # Scaling (Scaler 使用所有 N_FEATURES=4)
    scaler = MinMaxScaler()
    split_idx = int(len(features) * 0.8)
    features_tr, features_val = features[:split_idx], features[split_idx:]
    
    scaler.fit(features_tr)
    features_tr_scaled  = scaler.transform(features_tr)
    features_val_scaled = scaler.transform(features_val)
    
    # 建立序列
    X_tr, y_tr   = make_sequences(features_tr_scaled, window_size, N_FEATURES)
    X_val, y_val = make_sequences(features_val_scaled, window_size, N_FEATURES)

    if len(X_tr) == 0 or len(X_val) == 0:
        raise RuntimeError("切分後無法形成序列，請調整 WINDOW。")
    
    return X_tr, y_tr, X_val, y_val, scaler, len(features), len(X_tr), len(X_val)

def make_sequences(arr, window, n_features):
    X, y = [], []
    for i in range(len(arr) - window - 1):
        X.append(arr[i:i+window])
        y.append(arr[i+window, 0]) 
    return np.array(X).reshape(-1, window, n_features), np.array(y).reshape(-1, 1)


def train_and_evaluate(window_size, df_full, out_dir):
    """執行單次訓練、評估並儲存結果"""
    print(f"--- 正在測試 WINDOW = {window_size} ---")
    try:
        # 1. 準備數據
        X_tr, y_tr, X_val, y_val, scaler, n_raw, n_tr, n_val = prepare_data(df_full, window_size)
        
        # 2. 建模 (Stacked LSTM)
        model = Sequential([
            LSTM(128, input_shape=(window_size, N_FEATURES), return_sequences=True), 
            Dropout(0.2),
            LSTM(64, return_sequences=False),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer="adam", loss="mse")

        # 3. 訓練（增加 Patience）
        es  = EarlyStopping(patience=15, restore_best_weights=True)
        rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-5, verbose=0)

        history = model.fit(
            X_tr, y_tr, validation_data=(X_val, y_val),
            epochs=EPOCHS, batch_size=BATCH, callbacks=[es, rlr], verbose=0
        )
        
        # 4. 評估
        y_pred = model.predict(X_val, verbose=0)
        y_temp = np.zeros((len(y_val), N_FEATURES)); y_pred_temp = np.zeros((len(y_pred), N_FEATURES))
        y_temp[:, 0] = y_val.flatten(); y_pred_temp[:, 0] = y_pred.flatten()
        y_val_w = scaler.inverse_transform(y_temp)[:, 0]
        y_pred_w = scaler.inverse_transform(y_pred_temp)[:, 0]

        rmse = float(np.sqrt(mean_squared_error(y_val_w, y_pred_w)))
        mape = float(mean_absolute_percentage_error(y_val_w, y_pred_w) * 100)
        best_val_loss = float(np.min(history.history["val_loss"]))
        
        # 5. 儲存最佳模型 (在實驗中，只在最佳運行時儲存)
        # 這裡我們只輸出日誌，不覆蓋主模型

        return rmse, mape, best_val_loss, n_tr, n_val, model, scaler

    except RuntimeError as e:
        print(f"--- WINDOW {window_size} 失敗: {e} ---")
        return None, None, None, None, None, None, None


# ----------------------------------------------------
# *** 實驗控制中心 (腳本運行入口) ***
# ----------------------------------------------------
if __name__ == "__main__":
    # 確保輸出目錄存在
    out_dir = Path("/models")
    if not out_dir.exists():
        out_dir = Path(__file__).resolve().parents[1] / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 執行所有訓練 (數據只讀取一次)
    df_input = df_raw.copy()
    
    BEST_RMSE = float('inf')
    BEST_WINDOW = 0
    FINAL_RESULTS = []

    print("\n\n======== 啟動 WINDOW 敏感度實驗 ========")
    for W in WINDOW_LIST:
        rmse, mape, val_loss, n_tr, n_val, model_run, scaler_run = train_and_evaluate(W, df_input, out_dir)
        
        if rmse is not None:
            FINAL_RESULTS.append({
                'window': W, 
                'rmse_w': f"{rmse:.3f}", 
                'mape_percent': f"{mape:.2f}%", 
                'n_train': n_tr,
                'n_val': n_val
            })
            
            if rmse < BEST_RMSE:
                BEST_RMSE = rmse
                BEST_WINDOW = W
                
                # 儲存最佳運行結果的模型和 Scaler (覆蓋主部署檔案)
                model_run.save(out_dir / "lstm_carbon_model.keras") 
                joblib.dump(scaler_run, out_dir / "scaler_power.pkl")
                print(f"🏆 發現最佳性能！模型已儲存 (RMSE: {BEST_RMSE:.3f}W)")
            
            print(f"✅ WINDOW {W} 完成。RMSE: {rmse:.3f}W")

    # 輸出最終總結報告
    print("\n\n======== 實驗總結報告 ========")
    results_df = pd.DataFrame(FINAL_RESULTS)

    if not results_df.empty:
        print(f"總結測試的 WINDOWs: {WINDOW_LIST}")
        print(f"🏆 最終最佳 WINDOW: {BEST_WINDOW} 分鐘 (RMSE: {BEST_RMSE:.3f}W)")
        
        # 寫入 CSV 檔案
        results_df.to_csv(out_dir / "window_sensitivity_results.csv", index=False)
        print(f"📝 實驗結果已儲存至: {out_dir / 'window_sensitivity_results.csv'}")
    else:
        print("⚠️ 無法運行任何 WINDOW 尺寸。數據量過少。")