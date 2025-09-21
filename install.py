#!/usr/bin/env python3
"""
Installation script for Business Insights Agent
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stdout:
            print(f"STDOUT: {e.stdout}")
        if e.stderr:
            print(f"STDERR: {e.stderr}")
        return False

def check_python_version():
    """Check Python version"""
    print("🐍 Checking Python version...")
    if sys.version_info < (3, 10):
        print(f"❌ Python 3.10+ required, found {sys.version}")
        return False
    print(f"✅ Python {sys.version.split()[0]} is compatible")
    return True

def install_requirements():
    """Install Python requirements"""
    if not Path("requirements.txt").exists():
        print("❌ requirements.txt not found")
        return False
    
    return run_command(
        f"{sys.executable} -m pip install -r requirements.txt",
        "Installing Python dependencies"
    )

def create_directories():
    """Create necessary directories"""
    print("📁 Creating directories...")
    directories = ['schemas', 'data', 'logs', 'exports']
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"   ✅ Created {directory}/")
    
    return True

def check_ollama():
    """Check if Ollama is available"""
    print("🤖 Checking Ollama...")
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print("✅ Ollama is running")
            return True
    except:
        pass
    
    print("⚠️  Ollama not detected")
    print("💡 To enable AI features:")
    print("   1. Install Ollama from https://ollama.ai/")
    print("   2. Start Ollama: ollama serve")
    print("   3. Pull a model: ollama pull llama2")
    return False

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
    ]
    
    optional_packages = [
        ('langchain', 'LangChain'),
        ('ollama', 'Ollama'),
    ]
    
    all_good = True
    
    for package, name in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {name}")
        except ImportError:
            print(f"   ❌ {name} - Required package missing")
            all_good = False
    
    for package, name in optional_packages:
        try:
            __import__(package)
            print(f"   ✅ {name} (optional)")
        except ImportError:
            print(f"   ⚠️  {name} (optional) - AI features will be limited")
    
    return all_good

def main():
    """Main installation process"""
    print("🚀 Business Insights Agent - Installation")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        print("\n❌ Installation failed: Python version incompatible")
        sys.exit(1)
    
    # Create directories
    if not create_directories():
        print("\n❌ Installation failed: Could not create directories")
        sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        print("\n❌ Installation failed: Could not install dependencies")
        sys.exit(1)
    
    # Test imports
    if not test_imports():
        print("\n❌ Installation failed: Required packages not available")
        sys.exit(1)
    
    # Check Ollama (optional)
    ollama_available = check_ollama()
    
    print("\n" + "=" * 50)
    print("🎉 Installation completed successfully!")
    print("\n📋 Next steps:")
    print("   1. Run the application: python run.py")
    print("   2. Or use Streamlit directly: streamlit run app.py")
    
    if not ollama_available:
        print("\n⚠️  Note: AI features require Ollama to be installed and running")
    
    print("\n🌐 The application will be available at: http://localhost:8501")
    print("\n📚 For more information, see README.md")

if __name__ == "__main__":
    main()
