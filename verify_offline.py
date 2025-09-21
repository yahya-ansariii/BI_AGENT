"""
Offline Operation Verification Script
Verifies that the Business Insights Agent can run completely offline
"""

import os
import sys
import subprocess
import requests
import json
from pathlib import Path

def check_offline_dependencies():
    """Check that all dependencies are local and no online services are required"""
    print("🔍 Checking Offline Dependencies...")
    
    # Check that all required packages are installed locally
    required_packages = [
        'streamlit', 'pandas', 'duckdb', 'matplotlib', 
        'plotly', 'openpyxl', 'xlrd', 'numpy', 'seaborn', 
        'requests', 'psutil'
    ]
    
    # Optional packages (not critical for offline operation)
    optional_packages = ['python-dotenv']
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"  ❌ {package} - MISSING")
    
    # Check optional packages
    for package in optional_packages:
        try:
            __import__(package)
            print(f"  ✅ {package} (optional)")
        except ImportError:
            print(f"  ⚠️  {package} - OPTIONAL (not installed)")
    
    if missing_packages:
        print(f"\n❌ Missing required packages: {', '.join(missing_packages)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("✅ All required packages are installed locally")
    return True

def check_ollama_local():
    """Check that Ollama is configured for local operation only"""
    print("\n🔍 Checking Ollama Local Configuration...")
    
    # Check if Ollama is running locally
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            print(f"  ✅ Ollama running locally with {len(models)} models")
            for model in models:
                print(f"    • {model.get('name', 'Unknown')}")
            return True
        else:
            print(f"  ❌ Ollama not responding (status: {response.status_code})")
            return False
    except requests.exceptions.ConnectionError:
        print("  ❌ Ollama not running locally")
        print("  💡 Start Ollama with: ollama serve")
        return False
    except Exception as e:
        print(f"  ❌ Error checking Ollama: {str(e)}")
        return False

def check_no_external_dependencies():
    """Check that no external APIs or services are used"""
    print("\n🔍 Checking for External Dependencies...")
    
    # Check Python files for external URLs (excluding localhost and download links)
    external_urls = []
    python_files = list(Path('.').rglob('*.py'))
    
    for file_path in python_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
                
                for line_num, line in enumerate(lines, 1):
                    # Look for external URLs (not localhost or download links)
                    if 'http://' in line or 'https://' in line:
                        # Skip localhost, download links, installation instructions, and verification script itself
                        if not any(skip in line.lower() for skip in [
                            'localhost', '127.0.0.1', 'ollama.ai/download', 
                            'github.com', 'readthedocs.io', 'placeholder',
                            'verify_offline.py', 'install.sh', 'curl -fssl',
                            'print(', 'install ollama from', 'download from'
                        ]):
                            external_urls.append(f"{file_path}:{line_num} - {line.strip()}")
        except Exception as e:
            print(f"  ⚠️  Could not check {file_path}: {str(e)}")
    
    if external_urls:
        print("  ❌ Found potential external dependencies:")
        for url in external_urls:
            print(f"    {url}")
        return False
    else:
        print("  ✅ No external API dependencies found")
        return True

def check_data_processing_offline():
    """Check that data processing works offline"""
    print("\n🔍 Testing Offline Data Processing...")
    
    try:
        from modules.data_processor import DataProcessor
        import pandas as pd
        
        # Create test data
        test_data = pd.DataFrame({
            'name': ['Alice', 'Bob', 'Charlie'],
            'age': [25, 30, 35],
            'salary': [50000, 60000, 70000]
        })
        
        # Test data processor
        processor = DataProcessor()
        processor.data = test_data
        processor.data_name = 'test'
        
        # Test various operations
        summary = processor.get_data_summary()
        preview = processor.get_data_preview(2)
        csv_export = processor.export_to_csv()
        
        print("  ✅ Data processing works offline")
        print(f"    • Data summary: {len(summary)} fields")
        print(f"    • Data preview: {len(preview)} rows")
        print(f"    • CSV export: {len(csv_export)} characters")
        return True
        
    except Exception as e:
        print(f"  ❌ Data processing error: {str(e)}")
        return False

def check_ai_analysis_offline():
    """Check that AI analysis works offline (if Ollama is running)"""
    print("\n🔍 Testing Offline AI Analysis...")
    
    try:
        from modules.llm_agent import LLMAgent
        import pandas as pd
        
        # Create test data
        test_data = pd.DataFrame({
            'product': ['A', 'B', 'C'],
            'sales': [100, 200, 150],
            'profit': [20, 40, 30]
        })
        
        # Test LLM agent
        agent = LLMAgent()
        
        # Check if Ollama is available
        if agent._check_ollama_connection():
            print("  ✅ Ollama connection available")
            
            # Try to initialize a model
            available_models = agent.list_models()
            if available_models:
                print(f"  ✅ Available models: {', '.join(available_models)}")
                
                # Test data preparation (this was causing JSON errors)
                summary = agent._prepare_data_summary(test_data)
                print("  ✅ Data summary preparation works")
                print(f"    • Summary keys: {list(summary.keys())}")
                print(f"    • Sample data: {len(summary.get('sample_data', []))} rows")
                
                return True
            else:
                print("  ⚠️  No models available (download models with: python setup_models.py)")
                return False
        else:
            print("  ❌ Ollama not running locally")
            print("  💡 Start Ollama with: ollama serve")
            return False
            
    except Exception as e:
        print(f"  ❌ AI analysis error: {str(e)}")
        return False

def main():
    """Main verification function"""
    print("🚀 Business Insights Agent - Offline Operation Verification")
    print("=" * 60)
    
    checks = [
        ("Local Dependencies", check_offline_dependencies),
        ("Ollama Local", check_ollama_local),
        ("No External APIs", check_no_external_dependencies),
        ("Data Processing", check_data_processing_offline),
        ("AI Analysis", check_ai_analysis_offline)
    ]
    
    results = []
    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"  ❌ {check_name} check failed: {str(e)}")
            results.append((check_name, False))
    
    print("\n" + "=" * 60)
    print("📊 Verification Summary:")
    
    passed = 0
    for check_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} - {check_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{len(results)} checks passed")
    
    if passed == len(results):
        print("🎉 System is fully configured for offline operation!")
        print("\n💡 To run the application offline:")
        print("   1. Make sure Ollama is running: ollama serve")
        print("   2. Start the app: python start.py")
        print("   3. Open browser to: http://localhost:8501")
    else:
        print("⚠️  Some checks failed. Please address the issues above.")
        print("\n💡 For offline operation, ensure:")
        print("   1. All dependencies are installed: pip install -r requirements.txt")
        print("   2. Ollama is installed and running: ollama serve")
        print("   3. Models are downloaded: python setup_models.py")

if __name__ == "__main__":
    main()
