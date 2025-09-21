#!/usr/bin/env python3
"""
Installation script for Business Insights Agent
Checks dependencies and sets up the environment
"""

import sys
import subprocess
import os
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    return True

def check_dependencies():
    """Check if required dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        ('streamlit', 'Streamlit'),
        ('pandas', 'Pandas'),
        ('duckdb', 'DuckDB'),
        ('matplotlib', 'Matplotlib'),
        ('plotly', 'Plotly'),
        ('seaborn', 'Seaborn'),
        ('openpyxl', 'OpenPyXL'),
        ('numpy', 'NumPy'),
        ('requests', 'Requests'),
        ('psutil', 'PSUtil'),
    ]
    
    all_good = True
    
    for package, name in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {name}")
        except ImportError:
            print(f"   ❌ {name} - Required package missing")
            all_good = False
    
    return all_good

def install_requirements():
    """Install required packages"""
    print("📦 Installing requirements...")
    
    try:
        result = subprocess.run([
            sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'
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

def check_ollama():
    """Check if Ollama is available"""
    print("🤖 Checking Ollama...")
    
    try:
        result = subprocess.run(['ollama', '--version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✅ Ollama is installed")
            return True
        else:
            print("❌ Ollama is not working properly")
            return False
    except FileNotFoundError:
        print("❌ Ollama is not installed")
        print("Please install Ollama from: https://ollama.ai/download")
        return False
    except Exception as e:
        print(f"❌ Error checking Ollama: {str(e)}")
        return False

def create_directories():
    """Create necessary directories"""
    print("📁 Creating directories...")
    
    directories = ['data', 'logs', 'schema_store', 'demo_data']
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    
    print("✅ Directories created")

def test_imports():
    """Test if all required packages can be imported"""
    print("🧪 Testing imports...")
    
    required_packages = [
        ('streamlit', 'Streamlit'),
        ('pandas', 'Pandas'),
        ('duckdb', 'DuckDB'),
        ('matplotlib', 'Matplotlib'),
        ('plotly', 'Plotly'),
        ('seaborn', 'Seaborn'),
        ('openpyxl', 'OpenPyXL'),
        ('numpy', 'NumPy'),
        ('requests', 'Requests'),
        ('psutil', 'PSUtil'),
    ]
    
    all_good = True
    
    for package, name in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {name}")
        except ImportError:
            print(f"   ❌ {name} - Required package missing")
            all_good = False
    
    return all_good

def main():
    """Main installation process"""
    print("🚀 Business Insights Agent - Installation")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check dependencies
    if not check_dependencies():
        print("\n📦 Installing missing dependencies...")
        if not install_requirements():
            print("❌ Failed to install dependencies")
            sys.exit(1)
    
    # Test imports after installation
    if not test_imports():
        print("❌ Some packages still missing after installation")
        sys.exit(1)
    
    # Check Ollama (optional)
    ollama_available = check_ollama()
    
    # Create directories
    create_directories()
    
    print("\n🎉 Installation completed successfully!")
    
    if not ollama_available:
        print("\n⚠️  Ollama not found - AI features will be limited")
        print("To enable AI features:")
        print("   1. Install Ollama from: https://ollama.ai/download")
        print("   2. Pull a model: ollama pull llama2")
    else:
        print("\n✅ Ollama detected - AI features available")
    
    print("\n📚 For more information, see README.md")

if __name__ == "__main__":
    main()