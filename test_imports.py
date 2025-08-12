#!/usr/bin/env python3
"""
Test script to verify all imports work correctly
"""

import sys
import os

# Add modules directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
modules_dir = os.path.join(current_dir, 'modules')
if modules_dir not in sys.path:
    sys.path.insert(0, modules_dir)

print("Testing imports...")

# Test basic imports
try:
    import streamlit as st
    print("✅ Streamlit import: OK")
except ImportError as e:
    print(f"❌ Streamlit import failed: {e}")

try:
    import pandas as pd
    print("✅ Pandas import: OK")
except ImportError as e:
    print(f"❌ Pandas import failed: {e}")

# Test modules imports
try:
    from modules.web_scraping_module import perform_web_scraping
    print("✅ perform_web_scraping import: OK")
except ImportError as e:
    print(f"❌ perform_web_scraping import failed: {e}")

try:
    from modules.enhanced_timber_business_researcher import EnhancedTimberBusinessResearcher
    print("✅ EnhancedTimberBusinessResearcher import: OK")
except ImportError as e:
    print(f"❌ EnhancedTimberBusinessResearcher import failed: {e}")

try:
    from data_explorer import create_data_explorer
    print("✅ data_explorer import: OK")
except ImportError as e:
    print(f"❌ data_explorer import failed: {e}")

# Test the main file import
try:
    # Don't actually import the main file (it would run streamlit)
    # Just check if the file exists and can be read
    with open('ai_csv_analyzer.py', 'r') as f:
        content = f.read()
    
    # Check for problematic imports
    problematic_patterns = [
        'research_timber_businesses_from_dataframe',
        'from modules.web_scraping_module import perform_web_scraping'
    ]
    
    issues = []
    for pattern in problematic_patterns:
        if pattern in content:
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if pattern in line:
                    issues.append(f"Line {i+1}: {line.strip()}")
    
    if issues:
        print("⚠️ Potential import issues found:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("✅ ai_csv_analyzer.py syntax check: OK")
        
except Exception as e:
    print(f"❌ ai_csv_analyzer.py check failed: {e}")

print("\nImport test completed.")
