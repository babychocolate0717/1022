@echo off
echo [%date% %time%] Starting Energy Monitor Agent...

:: Change directory to the script's location
cd /d "%~dp0"

:: Create logs directory if it does not exist
if not exist "logs" mkdir logs

echo [%date% %time%] Agent startup initiated >> logs\startup.log

:: Execute the agent script using the system's python
:: All output will be redirected to log files
python -u "integrated_agent.py" >> logs\agent_output.log 2>> logs\agent_error.log

echo [%date% %time%] Agent process ended.
pause