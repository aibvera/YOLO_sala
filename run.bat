@echo off
REM Cambia al directorio donde está el entorno virtual y el script
cd C:\Users\aleja\OneDrive\Alejandro\Proyectos\YOLO_sala

REM Activa el entorno virtual
call venv\Scripts\activate.bat

REM Ejecuta el script
python Wifi_camera.py

REM Pausa de 5 segundos
timeout /t 5 /nobreak >nul

REM Desactiva el entorno virtual
deactivate
