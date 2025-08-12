import streamlit as st

st.title("🔬 Simple Diagnostic")
st.success("✅ Streamlit is working!")

try:
    import pandas as pd
    st.success("✅ Pandas imported")
except Exception as e:
    st.error(f"❌ Pandas error: {e}")

try:
    import numpy as np
    st.success("✅ Numpy imported")
except Exception as e:
    st.error(f"❌ Numpy error: {e}")

try:
    import plotly.express as px
    st.success("✅ Plotly imported")
except Exception as e:
    st.error(f"❌ Plotly error: {e}")

st.write("🎉 Basic diagnostic complete!")
