#!/usr/bin/env python3
"""
Single command startup script for Business Insights Agent
Handles complete setup including requirements installation, Ollama setup, and application startup
"""

import os
import sys
import subprocess
import time
import platform
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    return True

def check_requirements_file():
    """Check if requirements.txt exists"""
    requirements_file = Path(__file__).parent / "requirements.txt"
    if not requirements_file.exists():
        print("❌ requirements.txt not found")
        return False
    print("✅ requirements.txt found")
    return True

def install_requirements():
    """Install Python requirements"""
    print("📦 Installing Python requirements...")
    
    try:
        # Check if pip is available
        result = subprocess.run([sys.executable, "-m", "pip", "--version"], 
                              capture_output=True, text=True)
        if result.returncode != 0:
            print("❌ pip is not available")
            return False
        
        # Install requirements
        requirements_file = Path(__file__).parent / "requirements.txt"
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Requirements installed successfully")
            return True
        else:
            print(f"❌ Failed to install requirements: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error installing requirements: {str(e)}")
        return False

def check_ollama_installed():
    """Check if Ollama is installed"""
    try:
        result = subprocess.run(["ollama", "--version"], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✅ Ollama is installed")
            return True
        else:
            print("❌ Ollama is not working properly")
            return False
    except FileNotFoundError:
        print("❌ Ollama is not installed")
        return False
    except Exception as e:
        print(f"❌ Error checking Ollama: {str(e)}")
        return False

def install_ollama():
    """Install Ollama based on the operating system"""
    print("📥 Installing Ollama...")
    
    system = platform.system().lower()
    
    if system == "windows":
        print("Please install Ollama manually from: https://ollama.ai/download")
        print("After installation, restart this script.")
        return False
    elif system == "darwin":  # macOS
        try:
            # Try to install via homebrew
            result = subprocess.run(["brew", "install", "ollama"], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ Ollama installed via Homebrew")
                return True
            else:
                print("❌ Failed to install via Homebrew")
                print("Please install manually from: https://ollama.ai/download")
                return False
        except FileNotFoundError:
            print("❌ Homebrew not found")
            print("Please install Ollama manually from: https://ollama.ai/download")
            return False
    else:  # Linux
        try:
            # Try to install via curl
            install_script = "curl -fsSL https://ollama.ai/install.sh | sh"
            result = subprocess.run(install_script, shell=True, 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ Ollama installed via install script")
                return True
            else:
                print("❌ Failed to install via install script")
                print("Please install manually from: https://ollama.ai/download")
                return False
        except Exception as e:
            print(f"❌ Error installing Ollama: {str(e)}")
            print("Please install manually from: https://ollama.ai/download")
            return False

def setup_models_if_needed():
    """Setup models if none are available"""
    try:
        from config.ollama_config import ollama_config
        
        # Set environment variable for current process
        os.environ['OLLAMA_MODELS'] = ollama_config.get_model_path()
        
        models = ollama_config.get_available_models()
        
        if not models:
            print("📥 No models found. Setting up models...")
            
            # Try to download a basic model
            print("Downloading llama3.2:3b (small, fast model)...")
            if ollama_config.pull_model("llama3.2:3b"):
                print("✅ Model downloaded successfully")
                return True
            else:
                print("❌ Failed to download model")
                print("Please run 'python setup_models.py' to download models manually")
                return False
        else:
            print(f"✅ Found {len(models)} models")
            return True
            
    except Exception as e:
        print(f"❌ Error setting up models: {str(e)}")
        return False

def check_ollama_running():
    """Check if Ollama is already running"""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except Exception:
        return False

def start_ollama():
    """Start Ollama with proper model path"""
    print("🚀 Starting Ollama...")
    
    # Get model path from config
    try:
        from config.ollama_config import ollama_config
        model_path = ollama_config.get_model_path()
        print(f"Using model path: {model_path}")
        
        # Set environment variable
        env = os.environ.copy()
        env['OLLAMA_MODELS'] = model_path
        
        # Start Ollama in background
        process = subprocess.Popen(
            ["ollama", "serve"],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait a moment for Ollama to start
        print("⏳ Waiting for Ollama to start...")
        time.sleep(3)
        
        # Check if it's running
        if check_ollama_running():
            print("✅ Ollama started successfully!")
            return process
        else:
            print("❌ Failed to start Ollama")
            return None
            
    except Exception as e:
        print(f"❌ Error starting Ollama: {str(e)}")
        return None

def check_models():
    """Check if models are available"""
    try:
        from config.ollama_config import ollama_config
        
        # Set environment variable for current process
        os.environ['OLLAMA_MODELS'] = ollama_config.get_model_path()
        
        models = ollama_config.get_available_models()
        
        if models:
            print(f"✅ Found {len(models)} models:")
            for model in models:
                print(f"  • {model}")
            return True
        else:
            print("⚠️  No models found. Please download models first:")
            print("  Run: python setup_models.py")
            return False
    except Exception as e:
        print(f"❌ Error checking models: {str(e)}")
        return False

def start_application():
    """Start the main application"""
    print("\n🌐 Starting Business Insights Agent...")
    
    try:
        # Set environment variables for Streamlit
        os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
        os.environ['STREAMLIT_SERVER_PORT'] = '8501'
        os.environ['STREAMLIT_SERVER_ADDRESS'] = 'localhost'
        os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
        
        print("✅ Application started successfully!")
        print("🌐 Open your browser and go to: http://localhost:8501")
        print("Press Ctrl+C to stop the application")
        
        # Run the Streamlit app
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', 'app.py',
            '--server.port', '8501',
            '--server.address', 'localhost',
            '--browser.gatherUsageStats', 'false'
        ])
        
    except KeyboardInterrupt:
        print("\n👋 Shutting down Business Insights Agent...")
    except Exception as e:
        print(f"❌ Error starting application: {str(e)}")
        print("Make sure all dependencies are installed:")
        print("  pip install -r requirements.txt")

def main():
    """Main startup function with complete setup"""
    print("🤖 Business Insights Agent - Complete Setup & Start")
    print("=" * 60)
    
    # Step 1: Check Python version
    print("\n1️⃣ Checking Python version...")
    if not check_python_version():
        return
    
    # Step 2: Check and install requirements
    print("\n2️⃣ Checking Python requirements...")
    if not check_requirements_file():
        return
    
    # Try to import required modules to see if they're installed
    try:
        import flask
        import pandas
        import openpyxl
        print("✅ Required Python packages are available")
    except ImportError as e:
        print(f"⚠️  Missing package: {e}")
        print("Installing requirements...")
        if not install_requirements():
            print("❌ Failed to install requirements. Please install manually:")
            print("  pip install -r requirements.txt")
            return
    
    # Step 3: Check and install Ollama
    print("\n3️⃣ Checking Ollama installation...")
    if not check_ollama_installed():
        print("Ollama not found. Attempting to install...")
        if not install_ollama():
            print("❌ Please install Ollama manually and restart this script")
            print("Download from: https://ollama.ai/download")
            return
    
    # Step 4: Setup models if needed
    print("\n4️⃣ Checking models...")
    if not setup_models_if_needed():
        print("❌ Model setup failed. Please run 'python setup_models.py' manually")
        return
    
    # Step 5: Start Ollama if not running
    print("\n5️⃣ Starting Ollama...")
    if check_ollama_running():
        print("✅ Ollama is already running")
    else:
        ollama_process = start_ollama()
        if not ollama_process:
            print("❌ Cannot start without Ollama. Exiting.")
            return
    
    # Step 6: Final model check
    print("\n6️⃣ Final model verification...")
    if not check_models():
        print("❌ No models available. Please download models first:")
        print("  python setup_models.py")
        return
    
    # Step 7: Start the application
    print("\n7️⃣ Starting application...")
    start_application()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
        sys.exit(0)
