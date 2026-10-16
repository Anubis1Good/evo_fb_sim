@echo off
chcp 65001 >nul
title Составы команд

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [i] Создаю виртуальное окружение .venv
    python -m venv .venv
    if errorlevel 1 (
        echo [!] Не удалось создать venv. Проверьте, что Python добавлен в PATH.
        pause
        exit /b 1
    )
)

call ".venv\Scripts\activate.bat"

python -c "import streamlit" 2>nul
if errorlevel 1 (
    echo [i] Устанавливаю зависимости в .venv ...
    python -m pip install --upgrade pip
    python -m pip install pandas numpy openpyxl streamlit plotly
)

echo.
echo [i] Запускаю Streamlit на http://localhost:8501
echo [i] Остановить: Ctrl+C.
echo.
streamlit run app.py

echo.
echo [i] Приложение остановлено.
pause