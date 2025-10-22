@echo off
rem 嘗試將主控台字碼頁設定為 UTF-8 以支援中文字元
chcp 65001 > nul

echo ========================================
echo  Energy Monitor Agent 打包工具
echo ========================================
echo.

REM 檢查 Python 是否安裝
python --version >nul 2>&1
if errorlevel 1 (
    echo [錯誤] 找不到 Python，請先安裝 Python 3.8+
    pause
    exit /b 1
)

echo [1/5] 檢查必要套件...
pip show pyinstaller >nul 2>&1
if errorlevel 1 (
    echo [安裝] PyInstaller 未安裝，正在安裝...
    pip install pyinstaller
)
pip show pythonnet >nul 2>&1
if errorlevel 1 (
    echo [安裝] pythonnet 未安裝，正在安裝...
    pip install pythonnet
)

echo.
echo [2/5] 清理舊的打包檔案...
if exist "dist" rmdir /s /q dist
if exist "build" rmdir /s /q build
if exist "*.spec" del /q *.spec

echo.
echo [3/5] 檢查必要檔案是否存在...
if not exist "config.yaml" (
    echo [錯誤] 找不到 config.yaml！
    echo 請確保 config.yaml 與此腳本在同一目錄。
    pause
    exit /b 1
)
if not exist "LibreHardwareMonitorLib.dll" (
    echo [錯誤] 找不到 LibreHardwareMonitorLib.dll！
    echo 請確保 LibreHardwareMonitorLib.dll 與此腳本在同一目錄。
    pause
    exit /b 1
)

REM 檢查模型檔案
if not exist "..\models\cleaning_models\anomaly_model_20251012_164538.joblib" (
    echo [錯誤] 找不到 AI 模型檔案！
    echo 請確保以下檔案存在：
    echo    ..\models\cleaning_models\anomaly_model_20251012_164538.joblib
    echo    ..\models\cleaning_models\imputer_model_20251012_164538.joblib
    echo    ..\models\cleaning_models\scaler_20251012_164538.joblib
    pause
    exit /b 1
)

echo [成功] 所有必要檔案都已找到。
echo.

if not exist "icon.ico" (
    echo [提示] 找不到 icon.ico,將使用預設圖示。
    set ICON_PARAM=
) else (
    set ICON_PARAM=--icon=icon.ico
)

echo [4/5] 開始打包(這可能需要幾分鐘)...

REM 明確列出所有模型檔案
set MODEL_ID=20251012_164538

echo [打包] 配置檔...
echo [打包] DLL 檔案...
echo [打包] AI 模型檔案...

pyinstaller --onefile ^
    --add-data "config.yaml;." ^
    --add-data "LibreHardwareMonitorLib.dll;." ^
    --add-data "..\models\cleaning_models\anomaly_model_%MODEL_ID%.joblib;models\cleaning_models" ^
    --add-data "..\models\cleaning_models\imputer_model_%MODEL_ID%.joblib;models\cleaning_models" ^
    --add-data "..\models\cleaning_models\scaler_%MODEL_ID%.joblib;models\cleaning_models" ^
    --hidden-import="sklearn.ensemble._iforest" ^
    --hidden-import="sklearn.impute" ^
    --name "EnergyMonitorAgent" ^
    --console ^
    %ICON_PARAM% ^
    integrated_agent.py

echo [檢查] 驗證打包的模型檔案...

echo.
echo [5/5] 打包程序結束。
if exist "dist\EnergyMonitorAgent.exe" (
    echo [成功] EnergyMonitorAgent.exe 已成功建立於 'dist' 資料夾中。
    
    echo.
    echo [部署] 正在複製必要檔案到 dist 資料夾...
    
    if exist "config.yaml" (
        copy /Y "config.yaml" "dist\config.yaml" >nul
        echo    [OK] config.yaml
    )
    
    if not exist "dist\agent_logs" mkdir "dist\agent_logs"
    echo    [OK] agent_logs 資料夾
    
    echo.
    echo ========================================
    echo [完成] 可執行文件已準備就緒！
    echo ========================================
    echo   執行檔: dist\EnergyMonitorAgent.exe
    echo   配置檔: dist\config.yaml (可編輯^)
    echo   日誌目錄: dist\agent_logs
    echo   AI 模型: 已內建
    echo ========================================
) else (
    echo ========================================
    echo [失敗] 打包失敗，請檢查上方的錯誤訊息。
    echo ========================================
)
echo.
pause