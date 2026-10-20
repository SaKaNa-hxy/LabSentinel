@echo off
setlocal

cd /d "%~dp0"
set "ROOT=%~dp0"
set "PYTHON=%ROOT%.venv\Scripts\python.exe"
set "URL=http://127.0.0.1:8001/"
set "LOG=%ROOT%start_error.log"

if not exist "%PYTHON%" (
    echo [ERROR] Python environment not found: %PYTHON%
    pause
    exit /b 1
)

if not exist "%ROOT%app\backend\main.py" (
    echo [ERROR] Backend entry not found: %ROOT%app\backend\main.py
    pause
    exit /b 1
)

echo [INFO] Starting LabSentinel...
echo [INFO] Frontend URL: %URL%
echo [INFO] Close this window to stop the backend.
echo [INFO] Log file: %LOG%

echo [%date% %time%] start.bat launched > "%LOG%"
start "" /b powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "$url='%URL%'; for($i=0;$i -lt 30;$i++){try{$r=Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 2;if($r.StatusCode -ge 200){Start-Process $url;exit}}catch{};Start-Sleep -Seconds 1}"

"%PYTHON%" -B -m uvicorn app.backend.main:app --host 127.0.0.1 --port 8001 >> "%LOG%" 2>&1
set "EXIT_CODE=%errorlevel%"

echo.
if not "%EXIT_CODE%"=="0" (
    echo [ERROR] Backend failed. Exit code: %EXIT_CODE%
    echo [ERROR] See log file: %LOG%
) else (
    echo [INFO] Backend stopped.
)
echo.
pause
endlocal
