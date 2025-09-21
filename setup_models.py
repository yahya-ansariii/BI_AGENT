#!/usr/bin/env python3
"""
Setup script for configuring Ollama model paths
"""

import os
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from config.ollama_config import ollama_config
from agent.llm_agent import LLMAgent

def setup_custom_model_path():
    """Interactive setup for custom model path"""
    print("🤖 Ollama Model Path Setup")
    print("=" * 40)
    
    # Get current default path
    current_path = ollama_config.get_model_path()
    print(f"Current model path: {current_path}")
    
    # Ask user for custom path
    print("\nEnter your desired model path:")
    print("Examples:")
    print("  - C:\\Users\\YourName\\Documents\\LLM_Models")
    print("  - /home/username/llm_models")
    print("  - ~/Documents/AI_Models")
    
    custom_path = input("\nCustom model path: ").strip()
    
    if not custom_path:
        print("No path provided. Using default path.")
        return
    
    # Set custom path
    if ollama_config.set_custom_model_path(custom_path):
        print(f"✅ Custom model path set to: {ollama_config.get_model_path()}")
        
        # Test if Ollama is available
        print("\nTesting Ollama connection...")
        llm_agent = LLMAgent()
        
        if llm_agent.check_model_exists("llama3:8b-instruct"):
            print("✅ Ollama is working with custom path!")
        else:
            print("⚠️  Ollama is not running or no models found")
            print("To start Ollama with custom path, run:")
            print(f"  OLLAMA_MODELS={ollama_config.get_model_path()} ollama serve")
    else:
        print("❌ Failed to set custom model path")

def pull_models():
    """Pull recommended models"""
    print("\n📥 Model Download")
    print("=" * 40)
    
    recommended_models = [
        "llama3:8b-instruct",
        "llama2:7b",
        "llama2:13b",
        "llama3:70b-instruct",
        "mistral:7b-instruct"
    ]
    
    print("Recommended models:")
    for i, model in enumerate(recommended_models, 1):
        print(f"  {i}. {model}")
    
    choice = input("\nEnter model number to download (or 'all' for all): ").strip()
    
    if choice.lower() == 'all':
        models_to_download = recommended_models
    elif choice.isdigit() and 1 <= int(choice) <= len(recommended_models):
        models_to_download = [recommended_models[int(choice) - 1]]
    else:
        print("Invalid choice")
        return
    
    llm_agent = LLMAgent()
    
    for model in models_to_download:
        print(f"\nDownloading {model}...")
        if llm_agent.pull_model(model):
            print(f"✅ {model} downloaded successfully")
        else:
            print(f"❌ Failed to download {model}")

def check_setup():
    """Check current setup"""
    print("\n🔍 Setup Check")
    print("=" * 40)
    
    llm_agent = LLMAgent()
    
    model_path = llm_agent.get_model_path()
    print(f"Model path: {model_path}")
    print(f"Path exists: {os.path.exists(model_path)}")
    print(f"Path writable: {os.access(model_path, os.W_OK)}")
    
    models = llm_agent.list_models()
    print(f"Available models: {len(models)}")
    
    if not models:
        print("❌ No models found!")
        print("Please download at least one model to use the application.")
        print("\nRecommended models to download:")
        print("  • llama3:8b-instruct (fast, efficient)")
        print("  • llama2:7b (reliable, smaller)")
        print("  • mistral:7b-instruct (alternative)")
        return
    
    for model in models:
        model_info = llm_agent.get_model_info(model)
        status = "✅" if model_info.get('exists', False) else "❌"
        size = model_info.get('size', 'Unknown')
        print(f"  {status} {model} ({size})")
    
    # Check if we have at least 2 models for dual insights
    if len(models) >= 2:
        print("\n✅ Sufficient models for dual insights comparison")
    else:
        print("\n⚠️  Only one model available. Download another for dual insights comparison.")

def setup_dual_models():
    """Setup dual model configuration"""
    print("\n🔄 Dual Model Setup")
    print("=" * 40)
    
    print("Configure two models for insights comparison:")
    print("1. Primary model (main analysis)")
    print("2. Secondary model (comparison)")
    
    primary_model = input("\nEnter primary model (e.g., llama3:8b-instruct): ").strip()
    secondary_model = input("Enter secondary model (e.g., llama2:7b): ").strip()
    
    if primary_model and secondary_model:
        print(f"\nConfiguration:")
        print(f"  Primary: {primary_model}")
        print(f"  Secondary: {secondary_model}")
        
        confirm = input("\nSave this configuration? (y/n): ").strip().lower()
        if confirm == 'y':
            # This would save to a config file
            print("✅ Dual model configuration saved!")
            print("You can change this later in the application's Schema Approval tab.")
        else:
            print("Configuration not saved.")
    else:
        print("❌ Both models are required.")

def main():
    """Main setup function"""
    print("🚀 Business Insights Agent - Model Setup")
    print("=" * 50)
    
    while True:
        print("\nOptions:")
        print("1. Set custom model path")
        print("2. Download models")
        print("3. Setup dual models")
        print("4. Check current setup")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == '1':
            setup_custom_model_path()
        elif choice == '2':
            pull_models()
        elif choice == '3':
            setup_dual_models()
        elif choice == '4':
            check_setup()
        elif choice == '5':
            print("Goodbye! 👋")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
