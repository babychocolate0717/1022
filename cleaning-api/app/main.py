from fastapi import FastAPI
from app.schemas import RawEnergyData
from app.cleaning import clean_energy_data
import joblib
import os
# from os import environ # 不再需要導入 environ

app = FastAPI()

# ==========================================================
# *** 核心修正：將模型 ID 直接設定在程式碼中 (手動版本控制) ***
#
# 【使用說明】: 當您訓練好新模型後，只需將此變數替換為新的 ID 即可。
# ==========================================================
# 請將 'default' 替換為您的實際模型 ID (例如: '20251012_143920')
ACTIVE_MODEL_ID = "20251012_150520" 

# --- 模型載入邏輯 ---

MODEL_ID = ACTIVE_MODEL_ID

# 假設 models/ 在上兩層目錄
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(BASE_DIR, "models", "cleaning_models") 

# 構造版本控制後的檔案名稱
ANOMALY_MODEL_PATH = os.path.join(MODEL_DIR, f"anomaly_model_{MODEL_ID}.joblib")
IMPUTER_MODEL_PATH = os.path.join(MODEL_DIR, f"imputer_model_{MODEL_ID}.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, f"scaler_{MODEL_ID}.joblib") 


# 載入模型
try:
    if MODEL_ID == "default":
        # 如果仍使用預設值，視為找不到模型
        raise FileNotFoundError 
        
    ANOMALY_MODEL = joblib.load(ANOMALY_MODEL_PATH)
    IMPUTER_MODEL = joblib.load(IMPUTER_MODEL_PATH)
    SCALER = joblib.load(SCALER_PATH) 
    print(f"AI Cleaning Models (ID: {MODEL_ID}) Loaded Successfully.")
except FileNotFoundError:
    ANOMALY_MODEL = None
    IMPUTER_MODEL = None
    SCALER = None
    print(f"Warning: AI Model files for ID {MODEL_ID} not found. Using basic cleaning only.")


@app.post("/clean")
def clean_endpoint(data: RawEnergyData):
    cleaned = clean_energy_data(
        data.dict(), 
        anomaly_model=ANOMALY_MODEL, 
        imputer_model=IMPUTER_MODEL,
        scaler=SCALER 
    )
    return {"cleaned_data": cleaned}