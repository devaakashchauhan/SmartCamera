@echo off
title Smart Camera Web Dashboard
echo ========================================================
echo  Starting Smart Camera Dashboard & API Server
echo  Access at: http://127.0.0.1:8000
echo ========================================================
.\venv\Scripts\python.exe -m uvicorn app.main:app --app-dir backend --reload --host 127.0.0.1 --port 8000
pause
