@echo off
REM Start Ollama with custom model path
REM Edit the MODEL_PATH variable below to your desired path

set MODEL_PATH=C:\Users\%USERNAME%\Documents\LLM_Models
set OLLAMA_MODELS=%MODEL_PATH%

echo Starting Ollama with custom model path: %MODEL_PATH%
echo.
echo To change the model path, edit this file and modify the MODEL_PATH variable
echo.

ollama serve

pause
