@echo off
title PlantDiseaseAI
cd /d "%~dp0backend"
start "" "http://127.0.0.1:5000"
"C:\Users\vdbn1\AppData\Local\Programs\Python\Python311\python.exe" app.py
pause
