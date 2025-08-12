#!/usr/bin/env python3
"""
Test script to verify all imports and basic functionality work
Run this before deployment to catch any issues early
"""

import sys
import os

def test_basic_imports():
    """Test essential imports"""
    print("🧪 Testing basic imports...")
    
    try:
        import streamlit as st
        print("✅ Streamlit imported successfully")
    except ImportError as e:
        print(f"❌ Streamlit import failed: {e}")
        return False
    
    try:
        import pandas as pd
        import numpy as np
        import plotly.express as px
        print("✅ Data analysis libraries imported successfully")
    except ImportError as e:
        print(f"❌ Data analysis libraries import failed: {e}")
        return False
    
    try:
        import requests
        print("✅ Requests imported successfully")
    except ImportError as e:
        print(f"❌ Requests import failed: {e}")
        return False
    
    return True

def test_ai_libraries():
    """Test AI provider libraries - REQUIRED for business research functionality"""
    print("\n🤖 Testing AI libraries (REQUIRED for business research)...")
    
    all_good = True
    required_libs = [
        ('openai', 'OpenAI API library'),
        ('tavily', 'Tavily search API library'), 
        ('groq', 'Groq AI API library')
    ]
    
    for lib_name, description in required_libs:
        try:
            __import__(lib_name)
            print(f"✅ {description} available")
        except ImportError:
            print(f"❌ {description} NOT available - REQUIRED for business research")
            all_good = False
    
    if not all_good:
        print("\n🚨 CRITICAL: Missing required AI libraries!")
        print("📝 To fix, run: pip install openai tavily-python groq")
        print("🔑 You'll also need API keys configured in .env or Streamlit secrets")
    else:
        print("\n✅ All required AI libraries available for business research!")
    
    return all_good

def test_file_structure():
    """Test that required files exist"""
    print("\n📁 Testing file structure...")
    
    required_files = [
        'ai_csv_analyzer.py',
        'requirements.txt',
        '.streamlit/config.toml',
        '.gitignore'
    ]
    
    all_good = True
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file} exists")
        else:
            print(f"❌ {file} missing")
            all_good = False
    
    return all_good

def test_main_app_syntax():
    """Test that main app file has no syntax errors"""
    print("\n🔍 Testing main app syntax...")
    
    try:
        with open('ai_csv_analyzer.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Try to compile the code
        compile(content, 'ai_csv_analyzer.py', 'exec')
        print("✅ ai_csv_analyzer.py has no syntax errors")
        return True
    except SyntaxError as e:
        print(f"❌ Syntax error in ai_csv_analyzer.py: {e}")
        print(f"   Line {e.lineno}: {e.text}")
        return False
    except Exception as e:
        print(f"❌ Error reading ai_csv_analyzer.py: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Pre-deployment Test Suite")
    print("=" * 40)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("AI Libraries", test_ai_libraries),
        ("File Structure", test_file_structure),
        ("App Syntax", test_main_app_syntax)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🔄 Running {test_name} test...")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 40)
    print("📊 TEST SUMMARY")
    print("=" * 40)
    
    passed = 0
    for test_name, result in results:
        if result:
            print(f"✅ {test_name}: PASSED")
            passed += 1
        else:
            print(f"❌ {test_name}: FAILED")
    
    total = len(results)
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Ready for deployment.")
        return True
    else:
        print("⚠️ Some tests failed. Please fix issues before deploying.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
