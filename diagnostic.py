import streamlit as st
import os

# Diagnostic startup app to identify deployment issues
st.title("🔬 Deployment Diagnostic Tool")

# Test 1: Basic imports
st.subheader("📦 Testing Basic Imports")
try:
    import pandas as pd
    import numpy as np
    import plotly.express as px
    st.success("✅ Basic data libraries imported successfully")
except Exception as e:
    st.error(f"❌ Basic imports failed: {e}")

# Test 2: Check file structure (safely)
st.subheader("📁 Checking File Structure")
try:
    files_to_check = [
        'ai_csv_analyzer.py',
        'requirements.txt',
        '.streamlit/config.toml'
    ]
    
    for file in files_to_check:
        if os.path.exists(file):
            st.success(f"✅ {file} exists")
        else:
            st.error(f"❌ {file} missing")
except Exception as e:
    st.error(f"❌ File check error: {e}")

# Test 3: Check modules directory (safely)
st.subheader("🔧 Checking Modules Directory")
try:
    if os.path.exists('modules'):
        st.success("✅ modules directory exists")
        try:
            module_files = os.listdir('modules')
            st.write("Module files found:", module_files)
        except Exception as e:
            st.warning(f"⚠️ Could not list module files: {e}")
    else:
        st.warning("⚠️ modules directory not found")
except Exception as e:
    st.error(f"❌ Module check error: {e}")

# Test 4: Test AI library imports
st.subheader("🤖 Testing AI Libraries")
ai_libs = ['openai', 'tavily', 'groq']
for lib in ai_libs:
    try:
        __import__(lib)
        st.success(f"✅ {lib} imported successfully")
    except ImportError:
        st.warning(f"⚠️ {lib} not available")
    except Exception as e:
        st.error(f"❌ {lib} import error: {e}")

# Test 5: Environment variables
st.subheader("🔑 Environment Variables")
try:
    env_vars = ['OPENAI_API_KEY', 'TAVILY_API_KEY', 'GROQ_API_KEY']
    for var in env_vars:
        value = os.getenv(var)
        if value:
            st.success(f"✅ {var} is set")
        else:
            st.info(f"ℹ️ {var} not set")
except Exception as e:
    st.error(f"❌ Environment check error: {e}")

st.markdown("---")
st.write("🎯 **Next Step:** Once all tests pass, we can deploy the full application.")
