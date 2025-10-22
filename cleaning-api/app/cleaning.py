# cleaning-api/app/cleaning.py (完整整合程式碼)

import pandas as pd
import numpy as np
from typing import Optional

# 假設的硬體極限/規則 (Hard Rules Filter)
POWER_LIMITS = {
    "gpu_power_watt": (0.0, 350.0), 
    "cpu_power_watt": (0.0, 150.0),
    "system_power_watt": (0.0, 500.0)
}
FLOAT_FIELDS = [
    "gpu_usage_percent", "gpu_power_watt", "cpu_power_watt",
    "memory_used_mb", "disk_read_mb_s", "disk_write_mb_s", "system_power_watt"
]


# 函式簽名更新：加入 scaler 參數
def clean_energy_data(data: dict, anomaly_model=None, imputer_model=None, scaler=None) -> dict:
    cleaned = data.copy()

    # ================= 1. 原始固定規則：類別資料清洗 =================
    if cleaned.get("gpu_model", "").lower() == "unknown":
        cleaned["gpu_model"] = "Generic GPU"
    
    data_df = pd.DataFrame([cleaned])
    
    # ================= 2. 數值準備：型別轉換與標記 NaN =================
    for field in FLOAT_FIELDS:
        # errors='coerce' 將無法轉換的值設為 NaN，取代原有的 0.0 補值
        data_df[field] = pd.to_numeric(data_df.get(field), errors='coerce')
        if data_df[field].isnull().iloc[0]:
            cleaned[field] = np.nan

    cleaned['is_ai_spike'] = False

    if anomaly_model is not None and imputer_model is not None and scaler is not None:
        try:
            # ================= 階段 A：硬性規則過濾（識別物理錯誤）=================
            for field, (min_val, max_val) in POWER_LIMITS.items():
                if field in data_df.columns:
                    is_physical_error = (data_df[field] < min_val) | (data_df[field] > max_val)
                    if is_physical_error.iloc[0]:
                        data_df.loc[0, field] = np.nan # 確定錯誤的數值設為 NaN
            
            numeric_data = data_df[FLOAT_FIELDS]
            
            # ================= 階段 B：AI 異常值偵測 (M_A) - 僅標記真實高峰 =================
            # 準備 M_A 輸入：用均值填補 NaN (僅用於 M_A 評分)
            numeric_data_filled_for_ma = numeric_data.fillna(numeric_data.mean().fillna(0.0)) 
            
            is_anomaly = anomaly_model.predict(numeric_data_filled_for_ma)
            
            # Record-Level Flagging：如果 AI 偵測為異常，標記為 Spike (保留數值)
            if is_anomaly[0] == -1:
                cleaned['is_ai_spike'] = True
            
            # ================= 階段 C：智慧填補 (M_I) - 修復 NaN =================
            numeric_data_scaled = scaler.transform(numeric_data)
            imputed_array_scaled = imputer_model.transform(numeric_data_scaled)
            imputed_array = scaler.inverse_transform(imputed_array_scaled) # 反標準化
            
            numeric_data_filled_final = pd.DataFrame(imputed_array, columns=FLOAT_FIELDS, index=numeric_data.index)
            
            # 更新 cleaned 字典 (只有錯誤值被填補了)
            for field in FLOAT_FIELDS:
                cleaned[field] = float(numeric_data_filled_final.loc[0, field])

        except Exception as e:
            # AI 模組出錯時的回退邏輯
            print(f"AI cleaning failed: {e}. Falling back to original 0.0 filling.")
            cleaned['is_ai_spike'] = False
            # 最終補救：將所有的 NaN 值補為 0.0
            for field in FLOAT_FIELDS:
                if pd.isna(cleaned.get(field)):
                     cleaned[field] = 0.0
            
    else:
        # 如果 AI 模型未載入，執行原始程式碼的 0.0 補值邏輯
        for field in FLOAT_FIELDS:
            if pd.isna(cleaned.get(field)):
                 cleaned[field] = 0.0

    return cleaned