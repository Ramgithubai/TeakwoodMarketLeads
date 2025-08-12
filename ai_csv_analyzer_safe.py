import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from datetime import datetime
import os
import sys

# Page configuration
st.set_page_config(
    page_title="AI-Powered CSV Data Analyzer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Safe startup message
st.title("🤖 AI-Powered CSV Data Analyzer")
st.write("🚀 **Deployment Status:** Successfully started!")

# Check deployment environment
if st.checkbox("🔍 Show deployment info"):
    st.write("**Python version:**", sys.version)
    st.write("**Working directory:**", os.getcwd())
    st.write("**Available files:**", os.listdir('.'))

# Core functionality - CSV Analysis
st.subheader("📁 Upload Your Data")
uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=['csv', 'xlsx'])

if uploaded_file is not None:
    try:
        # Smart file loading
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        st.success(f"✅ Loaded {len(df):,} rows × {len(df.columns)} columns")
        
        # Tabs for different views
        tab1, tab2, tab3 = st.tabs(["📊 Overview", "🔍 Data Explorer", "📈 Visualizations"])
        
        with tab1:
            # Basic overview
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Rows", f"{len(df):,}")
            with col2:
                st.metric("Columns", len(df.columns))
            with col3:
                numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
                st.metric("Numeric Cols", numeric_cols)
            with col4:
                text_cols = len(df.select_dtypes(include=['object']).columns)
                st.metric("Text Cols", text_cols)
            
            st.subheader("📋 Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Column info
            st.subheader("📝 Column Information")
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes,
                'Non-Null Count': df.count(),
                'Null Count': df.isnull().sum()
            })
            st.dataframe(col_info, use_container_width=True)
        
        with tab2:
            # Data explorer
            st.subheader("🔍 Filter and Explore")
            
            # Simple filters
            columns = df.columns.tolist()
            
            # Text search
            search_term = st.text_input("🔍 Search in all columns:")
            if search_term:
                mask = df.astype(str).apply(lambda x: x.str.contains(search_term, case=False, na=False)).any(axis=1)
                filtered_df = df[mask]
                st.info(f"Found {len(filtered_df)} rows containing '{search_term}'")
            else:
                filtered_df = df
            
            # Column filter
            if st.checkbox("Filter by column value"):
                filter_col = st.selectbox("Select column:", columns)
                unique_values = df[filter_col].dropna().unique()[:100]  # Limit to 100 values
                if len(unique_values) <= 50:
                    selected_values = st.multiselect(f"Select {filter_col} values:", unique_values)
                    if selected_values:
                        filtered_df = filtered_df[filtered_df[filter_col].isin(selected_values)]
                else:
                    filter_value = st.text_input(f"Enter {filter_col} value to filter:")
                    if filter_value:
                        filtered_df = filtered_df[filtered_df[filter_col].astype(str).str.contains(filter_value, case=False, na=False)]
            
            # Display filtered data
            st.write(f"**Showing {len(filtered_df):,} of {len(df):,} rows**")
            st.dataframe(filtered_df.head(100), use_container_width=True)
            
            # Download filtered data
            if len(filtered_df) < len(df):
                csv = filtered_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Filtered Data",
                    csv,
                    f"filtered_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv"
                )
        
        with tab3:
            # Visualizations
            st.subheader("📈 Quick Visualizations")
            
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            text_columns = df.select_dtypes(include=['object']).columns.tolist()
            
            if numeric_columns:
                st.write("**Numeric Column Distribution**")
                selected_numeric = st.selectbox("Select numeric column:", numeric_columns)
                if selected_numeric:
                    fig = px.histogram(df, x=selected_numeric, title=f"Distribution of {selected_numeric}")
                    st.plotly_chart(fig, use_container_width=True)
            
            if text_columns:
                st.write("**Text Column Analysis**")
                selected_text = st.selectbox("Select text column:", text_columns)
                if selected_text:
                    value_counts = df[selected_text].value_counts().head(10)
                    fig = px.bar(x=value_counts.index, y=value_counts.values, 
                               title=f"Top 10 values in {selected_text}")
                    fig.update_xaxis(title=selected_text)
                    fig.update_yaxis(title="Count")
                    st.plotly_chart(fig, use_container_width=True)
            
            # Correlation heatmap for numeric data
            if len(numeric_columns) > 1:
                st.write("**Correlation Heatmap**")
                corr_matrix = df[numeric_columns].corr()
                fig = px.imshow(corr_matrix, 
                              title="Correlation Matrix",
                              aspect="auto",
                              color_continuous_scale="RdBu")
                st.plotly_chart(fig, use_container_width=True)
        
        # Download section
        st.markdown("---")
        st.subheader("📥 Download Options")
        col1, col2 = st.columns(2)
        with col1:
            csv = df.to_csv(index=False)
            st.download_button(
                "📄 Download as CSV",
                csv,
                f"data_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv"
            )
        with col2:
            # Basic statistics summary
            if numeric_columns:
                summary = df[numeric_columns].describe().to_csv()
                st.download_button(
                    "📊 Download Statistics Summary",
                    summary,
                    f"statistics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv"
                )
        
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        st.info("Please check that your file is a valid CSV or Excel file.")

# Footer
st.markdown("---")
st.info("🎯 **Status:** Core functionality is working! Advanced AI features and business research will be added in future updates.")

# API key status (if available)
with st.expander("🔧 Advanced Features Status"):
    st.write("**AI Chat Features:**")
    ai_keys = ['OPENAI_API_KEY', 'GROQ_API_KEY', 'ANTHROPIC_API_KEY']
    ai_available = any(os.getenv(key) for key in ai_keys)
    if ai_available:
        st.success("✅ AI providers configured - chat features available")
    else:
        st.info("ℹ️ Configure AI API keys in Streamlit secrets for chat features")
    
    st.write("**Business Research Features:**")
    research_available = os.getenv('TAVILY_API_KEY') and ai_available
    if research_available:
        st.success("✅ Business research APIs configured")
    else:
        st.info("ℹ️ Configure TAVILY_API_KEY + AI provider for business research")
