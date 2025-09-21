#!/usr/bin/env python3
"""
Test script to verify all setup components work correctly
"""

import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

def test_setup():
    """Test all setup components"""
    print("🧪 Testing Business Insights Agent Setup")
    print("=" * 50)
    
    try:
        from start import (
            check_python_version, 
            check_requirements_file, 
            check_ollama_installed, 
            setup_models_if_needed,
            check_ollama_running,
            check_models
        )
        
        # Test each component
        tests = [
            ("Python Version", check_python_version),
            ("Requirements File", check_requirements_file),
            ("Ollama Installation", check_ollama_installed),
            ("Model Setup", setup_models_if_needed),
            ("Ollama Running", check_ollama_running),
            ("Models Available", check_models)
        ]
        
        results = []
        for test_name, test_func in tests:
            print(f"\n🔍 Testing {test_name}...")
            try:
                result = test_func()
                results.append((test_name, result))
                status = "✅ PASS" if result else "❌ FAIL"
                print(f"   {status}")
            except Exception as e:
                results.append((test_name, False))
                print(f"   ❌ ERROR: {str(e)}")
        
        # Summary
        print("\n" + "=" * 50)
        print("📊 Test Summary:")
        passed = sum(1 for _, result in results if result)
        total = len(results)
        
        for test_name, result in results:
            status = "✅" if result else "❌"
            print(f"   {status} {test_name}")
        
        print(f"\n🎯 Overall: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 All tests passed! The application is ready to run.")
            print("   Run: python start.py")
        else:
            print("⚠️  Some tests failed. Please check the issues above.")
            
        return passed == total
        
    except Exception as e:
        print(f"❌ Critical error during testing: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_setup()
    sys.exit(0 if success else 1)
