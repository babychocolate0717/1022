# 將此段代碼替換 integrated_agent.py 中的模型路徑配置部分
# (從 "# --- AI 清洗配置" 到 "SCALER_PATH = ..." 的部分)

# --- AI 清洗配置 (修正版) ---
def resource_path(relative_path):
    """ 獲取資源的絕對路徑，適用於開發環境和 PyInstaller 打包後 """
    try:
        # PyInstaller 建立一個暫存資料夾，並將路徑儲存在 _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        # 在開發環境中，使用目前的檔案路徑
        base_path = os.path.dirname(os.path.abspath(__file__))

    return os.path.join(base_path, relative_path)

# 【重要】請根據您訓練腳本的輸出，手動設定此處的 ID！
ACTIVE_MODEL_ID = "20251012_164538"

# 定義模型路徑 (多重檢查機制)
# 優先順序：
# 1. 打包後內嵌: _MEIPASS/models/cleaning_models
# 2. 打包後外部: exe同目錄/models/cleaning_models
# 3. 開發環境: agent/../models/cleaning_models

MODEL_DIR = None
possible_paths = []

try:
    # 路徑 1: 打包後內嵌
    if hasattr(sys, '_MEIPASS'):
        path1 = os.path.join(sys._MEIPASS, "models", "cleaning_models")
        possible_paths.append(("內嵌路徑", path1))
    
    # 路徑 2: 打包後外部 (與 exe 同目錄)
    exe_dir = os.path.dirname(os.path.abspath(sys.executable if getattr(sys, 'frozen', False) else __file__))
    path2 = os.path.join(exe_dir, "models", "cleaning_models")
    possible_paths.append(("外部路徑", path2))
    
    # 路徑 3: 開發環境
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path3 = os.path.join(os.path.dirname(script_dir), "models", "cleaning_models")
    possible_paths.append(("開發路徑", path3))
    
    # 嘗試每個路徑
    for name, path in possible_paths:
        if os.path.exists(path):
            test_file = os.path.join(path, f"anomaly_model_{ACTIVE_MODEL_ID}.joblib")
            if os.path.exists(test_file):
                MODEL_DIR = path
                print(f"✅ 使用 {name}: {MODEL_DIR}")
                break
            else:
                print(f"⚠️ {name} 存在但缺少模型檔案: {path}")
        else:
            print(f"⚠️ {name} 不存在: {path}")
    
    if MODEL_DIR is None:
        print(f"❌ 所有路徑都找不到模型檔案")
        MODEL_DIR = path2  # 使用外部路徑作為預設值
        
except Exception as e:
    print(f"❌ 路徑檢查異常: {e}")
    MODEL_DIR = os.path.join("..", "models", "cleaning_models")

ANOMALY_MODEL_PATH = os.path.join(MODEL_DIR, f"anomaly_model_{ACTIVE_MODEL_ID}.joblib")
IMPUTER_MODEL_PATH = os.path.join(MODEL_DIR, f"imputer_model_{ACTIVE_MODEL_ID}.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, f"scaler_{ACTIVE_MODEL_ID}.joblib")

# 硬性規則限制 (與 cleaning-api 的邏輯保持一致)
POWER_LIMITS = {
    "cpu_power_watt": (0.0, 150.0), 
    "gpu_power_watt": (0.0, 350.0),
    "system_power_watt": (0.0, 500.0) 
}
FLOAT_FIELDS = [
    "gpu_usage_percent", "gpu_power_watt", "cpu_power_watt",
    "memory_used_mb", "disk_read_mb_s", "disk_write_mb_s", "system_power_watt"
]

# 載入 AI 模型
ANOMALY_MODEL = None
IMPUTER_MODEL = None
SCALER = None

print(f"🔍 檢查模型路徑...")
print(f"   MODEL_DIR: {MODEL_DIR}")
print(f"   ACTIVE_MODEL_ID: {ACTIVE_MODEL_ID}")
print(f"   Anomaly model path: {ANOMALY_MODEL_PATH}")
print(f"   檔案是否存在: {os.path.exists(ANOMALY_MODEL_PATH)}")

try:
    if ACTIVE_MODEL_ID == "YOUR_MODEL_ID_HERE":
        print("⚠️ Model ID 尚未設定")
        raise FileNotFoundError("Model ID not configured")
    
    if not os.path.exists(ANOMALY_MODEL_PATH):
        print(f"⚠️ 模型檔案不存在: {ANOMALY_MODEL_PATH}")
        print(f"   當前工作目錄: {os.getcwd()}")
        print(f"   嘗試列出 MODEL_DIR 內容...")
        if os.path.exists(MODEL_DIR):
            print(f"   MODEL_DIR 內容: {os.listdir(MODEL_DIR)}")
        else:
            print(f"   MODEL_DIR 不存在！")
        raise FileNotFoundError(f"Model file not found: {ANOMALY_MODEL_PATH}")
    
    print("📦 載入 Anomaly Model...")
    ANOMALY_MODEL = joblib.load(ANOMALY_MODEL_PATH)
    print("📦 載入 Imputer Model...")
    IMPUTER_MODEL = joblib.load(IMPUTER_MODEL_PATH)
    print("📦 載入 Scaler...")
    SCALER = joblib.load(SCALER_PATH)
    
    print(f"✅ AI Models (ID: {ACTIVE_MODEL_ID}) Loaded Successfully for Local Cleaning.")
    
except FileNotFoundError as e:
    print(f"⚠️ WARNING: Model files not found: {e}")
    print(f"   Using basic rules only.")
except Exception as e:
    print(f"⚠️ WARNING: Failed to load AI models: {type(e).__name__}: {e}")
    print(f"   Model path: {ANOMALY_MODEL_PATH}")
    print(f"   Using basic rules only.")
    import traceback
    traceback.print_exc()