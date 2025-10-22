import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
import joblib
import os
import datetime

# --- 1. 設定常量與路徑 ---
FLOAT_FIELDS = [
    "gpu_usage_percent", "gpu_power_watt", "cpu_power_watt",
    "memory_used_mb", "disk_read_mb_s", "disk_write_mb_s", "system_power_watt"
]

# 自動生成唯一的 ID 作為版本標識 (日期_時間)
MODEL_ID = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") 
print(f"Starting training for unique Model ID: {MODEL_ID}")

# 設定模型輸出路徑
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "models", "cleaning_models") 

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


# --- 2. 數據載入函數 ---
def load_historical_data():
    """
    載入歷史數據用於訓練。
    
    【重要】請根據您的實際情況修改此函數：
    1. 從 CSV 檔案讀取
    2. 從資料庫查詢
    3. 從 API 獲取
    
    示例：
    df = pd.read_csv('historical_energy_data.csv')
    return df[FLOAT_FIELDS]
    """
    
    # === 以下是模擬數據，請替換為真實數據載入邏輯 ===
    np.random.seed(42)
    n_samples = 5000  # 增加訓練樣本數
    
    data = {
        # 模擬真實的功耗分布
        "cpu_power_watt": np.random.normal(25, 8, n_samples),      # 平均25W，標準差8W
        "gpu_power_watt": np.random.normal(15, 10, n_samples),     # 平均15W，標準差10W
        "system_power_watt": np.random.normal(150, 30, n_samples), # 平均150W，標準差30W
        "gpu_usage_percent": np.random.beta(2, 5, n_samples) * 100, # 偏低使用率分布
        "memory_used_mb": np.random.normal(8000, 2000, n_samples),
        "disk_read_mb_s": np.random.exponential(5, n_samples),     # 偶爾高峰的分布
        "disk_write_mb_s": np.random.exponential(3, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # 確保數值在合理範圍內
    df['cpu_power_watt'] = df['cpu_power_watt'].clip(5, 125)
    df['gpu_power_watt'] = df['gpu_power_watt'].clip(0, 350)
    df['system_power_watt'] = df['system_power_watt'].clip(50, 500)
    df['gpu_usage_percent'] = df['gpu_usage_percent'].clip(0, 100)
    df['memory_used_mb'] = df['memory_used_mb'].clip(1000, 32000)
    df['disk_read_mb_s'] = df['disk_read_mb_s'].clip(0, 500)
    df['disk_write_mb_s'] = df['disk_write_mb_s'].clip(0, 500)
    
    # 模擬一些缺失值 (約5%)
    for field in FLOAT_FIELDS:
        missing_indices = np.random.choice(df.index, size=int(len(df) * 0.05), replace=False)
        df.loc[missing_indices, field] = np.nan
    
    # 注入少量真實異常值 (約2%)
    anomaly_indices = np.random.choice(df.index, size=int(len(df) * 0.02), replace=False)
    df.loc[anomaly_indices, 'cpu_power_watt'] = np.random.uniform(150, 200, len(anomaly_indices))
    df.loc[anomaly_indices, 'system_power_watt'] = np.random.uniform(600, 800, len(anomaly_indices))
    
    print(f"Loaded {len(df)} training samples")
    print(f"Missing values: {df.isnull().sum().sum()} ({df.isnull().sum().sum() / df.size * 100:.2f}%)")
    
    return df


# --- 3. 訓練流程 ---
def train_and_save_models(contamination_rate=0.03):
    """
    訓練並儲存 AI 清洗模型
    
    參數:
        contamination_rate: 異常值比例 (預設 0.03 = 3%)
                           - 0.01-0.02: 非常嚴格，適合穩定環境
                           - 0.03-0.05: 平衡設定，適合一般使用
                           - 0.05-0.10: 寬鬆設定，適合多變環境
    """
    print(f"\n{'='*60}")
    print(f"Training Configuration:")
    print(f"  - Contamination Rate: {contamination_rate*100:.1f}%")
    print(f"  - Model ID: {MODEL_ID}")
    print(f"{'='*60}\n")
    
    # 載入訓練數據
    historical_data = load_historical_data()
    data_for_training = historical_data[FLOAT_FIELDS].copy()
    
    # 移除完全無效的樣本
    data_clean_for_fit = data_for_training.dropna()
    
    if data_clean_for_fit.empty:
        print("❌ Error: Training data is too sparse or empty after dropping NaNs.")
        return
    
    print(f"Training samples after cleaning: {len(data_clean_for_fit)}")
    
    # --- 訓練 Scaler (用於標準化) ---
    print("\n[1/3] Training StandardScaler...")
    scaler = StandardScaler()
    scaler.fit(data_clean_for_fit) 
    scaler_path = os.path.join(OUTPUT_DIR, f'scaler_{MODEL_ID}.joblib')
    joblib.dump(scaler, scaler_path)
    print(f"  ✓ Saved: {scaler_path}")
    
    # --- 訓練 M_A: 異常值偵測 (IsolationForest) ---
    print("\n[2/3] Training Anomaly Detection Model (IsolationForest)...")
    anomaly_model = IsolationForest(
        contamination=contamination_rate,  # 調整敏感度
        random_state=42,
        n_estimators=150,      # 增加樹的數量提高穩定性
        max_samples='auto',    # 自動選擇樣本數
        max_features=1.0,      # 使用所有特徵
        bootstrap=False,       # 不使用自助法採樣
        n_jobs=-1             # 使用所有 CPU 核心
    )
    anomaly_model.fit(data_clean_for_fit)
    
    # 評估模型效果
    predictions = anomaly_model.predict(data_clean_for_fit)
    anomaly_count = np.sum(predictions == -1)
    anomaly_ratio = anomaly_count / len(predictions)
    
    print(f"  ✓ Training complete")
    print(f"  - Detected anomalies: {anomaly_count}/{len(predictions)} ({anomaly_ratio*100:.2f}%)")
    print(f"  - Expected rate: {contamination_rate*100:.1f}%")
    
    if abs(anomaly_ratio - contamination_rate) > 0.02:
        print(f"  ⚠️  Warning: Actual anomaly rate differs significantly from expected")
    
    anomaly_path = os.path.join(OUTPUT_DIR, f'anomaly_model_{MODEL_ID}.joblib')
    joblib.dump(anomaly_model, anomaly_path)
    print(f"  ✓ Saved: {anomaly_path}")

    # --- 訓練 M_I: 智慧填補 (KNNImputer) ---
    print("\n[3/3] Training Imputation Model (KNNImputer)...")
    imputer_model = KNNImputer(
        n_neighbors=5,
        weights='distance',  # 使用距離加權
        metric='nan_euclidean'
    )
    
    # 先標準化再訓練填補器
    data_scaled = scaler.transform(data_for_training.fillna(data_for_training.mean())) 
    imputer_model.fit(data_scaled) 
    
    imputer_path = os.path.join(OUTPUT_DIR, f'imputer_model_{MODEL_ID}.joblib')
    joblib.dump(imputer_model, imputer_path)
    print(f"  ✓ Saved: {imputer_path}")
    
    # --- 訓練完成總結 ---
    print(f"\n{'='*60}")
    print(f"✅ Training Complete!")
    print(f"{'='*60}")
    print(f"\n📋 Model Summary:")
    print(f"  Model ID: {MODEL_ID}")
    print(f"  Contamination Rate: {contamination_rate*100:.1f}%")
    print(f"  Training Samples: {len(data_clean_for_fit)}")
    print(f"  Detected Anomalies: {anomaly_count} ({anomaly_ratio*100:.2f}%)")
    print(f"\n📁 Saved Models:")
    print(f"  - {scaler_path}")
    print(f"  - {anomaly_path}")
    print(f"  - {imputer_path}")
    print(f"\n🔧 Next Step:")
    print(f"  Update your agent's ACTIVE_MODEL_ID to: {MODEL_ID}")
    print(f"\n  In integrated_agent.py, change:")
    print(f"    ACTIVE_MODEL_ID = \"{MODEL_ID}\"")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # === 調整此參數來控制敏感度 ===
    # 建議值：
    #   - 0.02: 非常嚴格，只標記極端異常
    #   - 0.03: 平衡設定（推薦）
    #   - 0.05: 寬鬆設定
    #   - 0.08: 非常寬鬆
    
    CONTAMINATION_RATE = 0.03  # 從 0.0001 改為 0.03
    
    train_and_save_models(contamination_rate=CONTAMINATION_RATE)