#!/bin/bash
# Start Ollama with custom model path
# Edit the MODEL_PATH variable below to your desired path

MODEL_PATH="$HOME/Documents/LLM_Models"
export OLLAMA_MODELS="$MODEL_PATH"

echo "Starting Ollama with custom model path: $MODEL_PATH"
echo ""
echo "To change the model path, edit this file and modify the MODEL_PATH variable"
echo ""

ollama serve
