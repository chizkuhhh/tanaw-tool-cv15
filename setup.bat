@echo off
echo ========================================
echo   Annotation Review Tool - Setup
echo ========================================
echo.

REM Check Python 3.9 exists
py -3.9 --version >nul 2>&1
IF ERRORLEVEL 1 (
    echo ERROR: Python 3.9 is not installed.
    echo Please install Python 3.9.x from:
    echo https://www.python.org/downloads/release/python-390/
    pause
    exit /b
)

echo Python 3.9 detected
echo.

echo Creating virtual environment: venv
py -3.9 -m venv venv

echo Activating virtual environment...
call venv\Scripts\activate

echo Upgrading pip...
python -m pip install --upgrade pip

echo Installing dependencies from requirements.txt...
pip install -r requirements.txt

echo.
echo ========================================
echo         Setup Complete!
echo ========================================
echo.
echo You can now run the tool using run.bat
echo.
pause