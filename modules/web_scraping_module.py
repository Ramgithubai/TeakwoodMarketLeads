"""
Web Scraping Module for Business Contact Research
Streamlit Cloud Compatible Version
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys
import asyncio
import warnings


def perform_web_scraping(filtered_df):
    """
    Perform web scraping of business contact information from filtered data
    Streamlit Cloud Compatible Version
    """
    
    # Suppress warnings
    warnings.filterwarnings('ignore')
    
    # Initialize session state variables to prevent resets
    if 'research_interface_open' not in st.session_state:
        st.session_state.research_interface_open = False
    if 'selected_business_column' not in st.session_state:
        st.session_state.selected_business_column = None
    if 'max_businesses_to_research' not in st.session_state:
        st.session_state.max_businesses_to_research = 5
    if 'business_range_from' not in st.session_state:
        st.session_state.business_range_from = 1
    if 'business_range_to' not in st.session_state:
        st.session_state.business_range_to = 5
    if 'api_test_completed' not in st.session_state:
        st.session_state.api_test_completed = False
    if 'api_test_result' not in st.session_state:
        st.session_state.api_test_result = None
    
    # Session state for research results
    if 'research_results' not in st.session_state:
        st.session_state.research_results = None
    if 'run_enhanced_research' not in st.session_state:
        st.session_state.run_enhanced_research = False
    if 'enhanced_research_list' not in st.session_state:
        st.session_state.enhanced_research_list = []
    if 'interface_mode' not in st.session_state:
        st.session_state.interface_mode = 'initial'
    if 'show_enhanced_selection' not in st.session_state:
        st.session_state.show_enhanced_selection = False
    
    # Check if DataFrame is empty
    if len(filtered_df) == 0:
        st.error("❌ No data to scrape. Please adjust your filters.")
        return
    
    # Set interface as open
    st.session_state.research_interface_open = True
    
    # Find suitable columns for business names
    potential_name_columns = []
    for col in filtered_df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in ['consignee', 'name', 'company', 'business', 'shipper', 'supplier']):
            potential_name_columns.append(col)
    
    if not potential_name_columns:
        st.error("❌ No suitable business name columns found. Need columns like 'Consignee Name', 'Company Name', etc.")
        st.session_state.research_interface_open = False
        return
    
    # Business name column selection
    st.write("🏷️ **Select Business Name Column:**")
    
    # Set default if not already set
    if st.session_state.selected_business_column is None or st.session_state.selected_business_column not in potential_name_columns:
        st.session_state.selected_business_column = potential_name_columns[0]
    
    selected_column = st.selectbox(
        "Choose the column containing business names:",
        potential_name_columns,
        index=potential_name_columns.index(st.session_state.selected_business_column),
        help="Select the column that contains the business names you want to research",
        key="business_name_column_selector_stable"
    )
    
    # Update session state when selection changes
    if selected_column != st.session_state.selected_business_column:
        st.session_state.selected_business_column = selected_column
        st.rerun()
    
    # Check unique business count
    unique_businesses = filtered_df[selected_column].dropna().nunique()
    if unique_businesses == 0:
        st.error(f"❌ No business names found in column '{selected_column}'")
        st.session_state.research_interface_open = False
        return
    
    st.info(f"📊 Found {unique_businesses} unique businesses to research in '{selected_column}'")
    
    # Research limit selection
    max_limit = min(20, unique_businesses)
    
    # Ensure session state values don't exceed current max_limit
    st.session_state.business_range_from = min(st.session_state.business_range_from, max_limit)
    st.session_state.business_range_to = min(st.session_state.business_range_to, max_limit)
    
    st.write("🎯 **Business Research Range:**")
    col_from, col_to = st.columns(2)
    
    with col_from:
        range_from = st.number_input(
            "From:",
            min_value=1,
            max_value=max_limit,
            value=st.session_state.business_range_from,
            help="Starting business number",
            key="business_range_from_input"
        )
    
    with col_to:
        range_to = st.number_input(
            "To:",
            min_value=range_from,
            max_value=max_limit,
            value=max(st.session_state.business_range_to, range_from),
            help="Ending business number",
            key="business_range_to_input"
        )
    
    # Calculate number of businesses to research
    max_businesses = range_to - range_from + 1
    
    # Update session state
    st.session_state.business_range_from = range_from
    st.session_state.business_range_to = range_to
    
    # Show summary
    st.info(f"📊 Will research businesses {range_from} to {range_to} ({max_businesses} total businesses)")
    
    # Cost estimation
    standard_cost = max_businesses * 0.03
    enhanced_cost = max_businesses * 0.05
    st.warning(f"💰 **Estimated API Cost:** Standard ~${standard_cost:.2f} | Enhanced ~${enhanced_cost:.2f}")
    
    # API Configuration check
    st.write("🔧 **API Configuration:**")
    
    # Check environment variables
    try:
        # Try to load environment variables
        from dotenv import load_dotenv
        load_dotenv(override=True)
    except ImportError:
        # dotenv not available (e.g., on Streamlit Cloud)
        pass
    
    # Get API keys
    openai_key = os.getenv('OPENAI_API_KEY') or st.secrets.get('OPENAI_API_KEY', '')
    tavily_key = os.getenv('TAVILY_API_KEY') or st.secrets.get('TAVILY_API_KEY', '')
    groq_key = os.getenv('GROQ_API_KEY') or st.secrets.get('GROQ_API_KEY', '')
    
    # Key validation function
    def is_valid_key(key, key_type):
        if not key or key.strip() == '':
            return False, "Key is empty or missing"
        if key.strip() in ['your_openai_key_here', 'your_tavily_key_here', 'your_groq_key_here', 'sk-...', 'tvly-...', 'gsk_...']:
            return False, "Key is a placeholder value"
        if key_type == 'openai' and not key.startswith('sk-'):
            return False, "OpenAI key should start with 'sk-'"
        if key_type == 'tavily' and not key.startswith('tvly-'):
            return False, "Tavily key should start with 'tvly-'"
        if key_type == 'groq' and not key.startswith('gsk_'):
            return False, "Groq key should start with 'gsk_'"
        return True, "Key format is valid"
    
    openai_valid, openai_reason = is_valid_key(openai_key, 'openai')
    tavily_valid, tavily_reason = is_valid_key(tavily_key, 'tavily')
    groq_valid, groq_reason = is_valid_key(groq_key, 'groq')
    
    # Display API status
    col_api1, col_api2, col_api3 = st.columns(3)
    
    with col_api1:
        if openai_valid:
            st.success("✅ OpenAI API Key: Configured")
            masked_key = f"{openai_key[:10]}...{openai_key[-4:]}" if len(openai_key) > 14 else f"{openai_key[:6]}..."
            st.caption(f"Key: {masked_key}")
        else:
            st.error(f"❌ OpenAI API Key: {openai_reason}")
            st.caption("Add OPENAI_API_KEY to secrets")
    
    with col_api2:
        if tavily_valid:
            st.success("✅ Tavily API Key: Configured")
            masked_key = f"{tavily_key[:10]}...{tavily_key[-4:]}" if len(tavily_key) > 14 else f"{tavily_key[:6]}..."
            st.caption(f"Key: {masked_key}")
        else:
            st.error(f"❌ Tavily API Key: {tavily_reason}")
            st.caption("Add TAVILY_API_KEY to secrets")
            
    with col_api3:
        if groq_valid:
            st.success("✅ Groq API Key: Configured")
            masked_key = f"{groq_key[:10]}...{groq_key[-4:]}" if len(groq_key) > 14 else f"{groq_key[:6]}..."
            st.caption(f"Key: {masked_key}")
        else:
            st.error(f"❌ Groq API Key: {groq_reason}")
            st.caption("Add GROQ_API_KEY to secrets (optional)")
    
    # Show setup instructions if keys are invalid
    if not openai_valid or not tavily_valid:
        st.warning("⚠️ **API Keys Required**: Please configure OpenAI and Tavily API keys in Streamlit secrets.")
        
        with st.expander("📝 Setup Instructions", expanded=False):
            st.markdown("""
            **To set up API keys in Streamlit Cloud:**
            
            1. **Go to your Streamlit Cloud app settings**
            2. **Navigate to Secrets section**
            3. **Add your API keys:**
               ```toml
               OPENAI_API_KEY = "sk-your_actual_openai_key_here"
               TAVILY_API_KEY = "tvly-your_actual_tavily_key_here"
               GROQ_API_KEY = "gsk_your_groq_key_here"
               ```
            4. **Restart the app**
            5. **Get API keys from:**
               - [OpenAI API Keys](https://platform.openai.com/api-keys)
               - [Tavily API](https://tavily.com)
               - [Groq API](https://console.groq.com/)
            """)
    
    # Test API connectivity
    both_apis_configured = openai_valid and tavily_valid
    
    if both_apis_configured:
        st.info("🟢 **API keys configured!** You can proceed with web scraping.")
        
        # Test API connection button
        col_test, col_status = st.columns([1, 2])
        
        with col_test:
            if st.button("🧪 Test API Connection", 
                        help="Test if APIs are working correctly", 
                        key="test_api_button_stable"):
                st.session_state.api_test_completed = False
                
                with st.spinner("Testing API connections..."):
                    try:
                        # Try to import the business researcher
                        try:
                            from streamlit_business_researcher import StreamlitBusinessResearcher
                            
                            # Test API connectivity
                            test_researcher = StreamlitBusinessResearcher()
                            api_ok, api_message = test_researcher.test_apis()
                            
                            st.session_state.api_test_completed = True
                            st.session_state.api_test_result = (api_ok, api_message)
                            
                        except ImportError as ie:
                            st.session_state.api_test_completed = True
                            st.session_state.api_test_result = (False, f"Business researcher module not available: {ie}")
                            
                    except Exception as e:
                        st.session_state.api_test_completed = True
                        st.session_state.api_test_result = (False, f"Test Error: {str(e)}")
        
        with col_status:
            # Display test results from session state
            if st.session_state.api_test_completed and st.session_state.api_test_result:
                api_ok, api_message = st.session_state.api_test_result
                if api_ok:
                    st.success(f"✅ API Test Successful: {api_message}")
                else:
                    st.error(f"❌ API Test Failed: {api_message}")
    
    # Research start section
    st.markdown("---")
    
    # Research button
    button_disabled = not both_apis_configured
    
    col_button1, col_info = st.columns([1, 3])
    
    with col_button1:
        start_research = st.button(
            f"🚀 Start Research ({max_businesses} businesses)",
            type="primary",
            disabled=button_disabled,
            help="Standard business research using web sources",
            key="start_research_button_stable"
        )
    
    with col_info:
        if button_disabled:
            st.info("⚠️ Configure API keys first")
        else:
            st.info(f"Ready to research {max_businesses} businesses")
    
    # Handle research execution
    if start_research and both_apis_configured:
        # Create a placeholder for the research process
        research_container = st.container()
        
        with research_container:
            st.info("🔄 Starting business research...")
            
            # Create progress indicators
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Try to import and run research
                try:
                    from streamlit_business_researcher import research_businesses_from_dataframe
                    
                    status_text.info("🚀 Initializing research system...")
                    progress_bar.progress(10)
                    
                    # Execute research
                    with st.spinner("Researching businesses using AI web scraping..."):
                        
                        async def run_research():
                            return await research_businesses_from_dataframe(
                                df=filtered_df,
                                consignee_column=selected_column,
                                max_businesses=max_businesses
                            )
                        
                        status_text.info("🔍 Starting business research process...")
                        progress_bar.progress(30)
                        
                        # Execute the research
                        try:
                            results_df, summary, csv_filename = asyncio.run(run_research())
                            progress_bar.progress(90)
                            status_text.success("✅ Research completed successfully!")
                            
                            # Display results if successful
                            if results_df is not None and not results_df.empty:
                                progress_bar.progress(100)
                                
                                # Display summary
                                st.success("🎉 **Research Summary:**")
                                col_sum1, col_sum2, col_sum3, col_sum4 = st.columns(4)
                                
                                with col_sum1:
                                    st.metric("Total Processed", summary['total_processed'])
                                with col_sum2:
                                    st.metric("Successful", summary['successful'])
                                with col_sum3:
                                    st.metric("Manual Required", summary['manual_required'])
                                with col_sum4:
                                    st.metric("Success Rate", f"{summary['success_rate']:.1f}%")
                                
                                # Store research results in session state
                                st.session_state.research_results = results_df
                                
                                # Display results table
                                st.subheader("📊 Research Results")
                                st.dataframe(results_df, use_container_width=True, height=400)
                                
                                # Download section
                                st.subheader("📥 Download Research Results")
                                
                                csv_data = results_df.to_csv(index=False)
                                
                                col_down1, col_down2 = st.columns(2)
                                with col_down1:
                                    st.download_button(
                                        label="📄 Download Research Results CSV",
                                        data=csv_data,
                                        file_name=f"business_research_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                        mime="text/csv"
                                    )
                                
                                with col_down2:
                                    # Create enhanced dataset
                                    if 'business_name' in results_df.columns:
                                        try:
                                            # Deduplicate results
                                            results_df_unique = results_df.drop_duplicates(subset=['business_name'], keep='first')
                                            
                                            research_mapping = results_df_unique.set_index('business_name')[['phone', 'email', 'website', 'address']].to_dict('index')
                                            
                                            enhanced_df = filtered_df.copy()
                                            enhanced_df['research_phone'] = enhanced_df[selected_column].map(lambda x: research_mapping.get(x, {}).get('phone', ''))
                                            enhanced_df['research_email'] = enhanced_df[selected_column].map(lambda x: research_mapping.get(x, {}).get('email', ''))
                                            enhanced_df['research_website'] = enhanced_df[selected_column].map(lambda x: research_mapping.get(x, {}).get('website', ''))
                                            enhanced_df['research_address'] = enhanced_df[selected_column].map(lambda x: research_mapping.get(x, {}).get('address', ''))
                                            
                                            enhanced_csv = enhanced_df.to_csv(index=False)
                                            
                                            st.download_button(
                                                label="🔗 Download Enhanced Dataset",
                                                data=enhanced_csv,
                                                file_name=f"enhanced_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                                mime="text/csv",
                                                help="Original data + research results combined"
                                            )
                                            
                                        except Exception as e:
                                            st.warning(f"Could not create enhanced dataset: {e}")
                                
                                # Success message
                                st.balloons()
                                st.success(f"🎉 Successfully researched {summary['successful']} businesses!")
                                
                                if summary['manual_required'] > 0:
                                    st.info(f"🔍 {summary['manual_required']} businesses require manual research")
                            
                            else:
                                st.warning("⚠️ Research completed but no results were found.")
                                st.info("This might be due to API rate limits, no search results, or connection problems.")
                        
                        except Exception as e:
                            error_msg = str(e)
                            progress_bar.progress(0)
                            status_text.error("❌ Research failed")
                            
                            st.error(f"❌ Research Error: {error_msg}")
                            
                            # Categorize errors for better user guidance
                            if "API" in error_msg or "key" in error_msg.lower():
                                st.error("🔑 API key issue. Please check your secrets configuration.")
                            elif "billing" in error_msg.lower() or "quota" in error_msg.lower():
                                st.error("💳 API billing/quota issue. Please check your API account.")
                            elif "connection" in error_msg.lower():
                                st.error("🌐 Connection error. Please check your internet connection.")
                            else:
                                st.error("⚠️ Please check your configuration and try again.")
                            
                            # Debug information
                            with st.expander("🔍 Debug Information", expanded=False):
                                st.code(f"Full error: {error_msg}")
                
                except ImportError as ie:
                    st.error(f"❌ Could not import business researcher: {str(ie)}")
                    st.error("📁 Business research modules not available in this deployment.")
                    st.info("This is a common issue in Streamlit Cloud deployments.")
                    
                    # Create fallback research functionality
                    st.subheader("📊 Simulated Research Results")
                    st.info("Creating sample results for demonstration purposes...")
                    
                    # Create sample data
                    unique_businesses_list = filtered_df[selected_column].dropna().unique()[:max_businesses]
                    
                    sample_data = []
                    for business in unique_businesses_list:
                        sample_data.append({
                            'business_name': business,
                            'phone': 'Contact research not available',
                            'email': 'Contact research not available',
                            'website': 'Contact research not available',
                            'address': 'Contact research not available',
                            'status': 'demo_mode'
                        })
                    
                    sample_df = pd.DataFrame(sample_data)
                    
                    st.dataframe(sample_df, use_container_width=True)
                    
                    st.warning("⚠️ This is demo data. To enable real research, ensure all required modules are available.")
                    
            except Exception as e:
                st.error(f"❌ Unexpected error during research: {str(e)}")
                st.error("🔄 Please check your configuration and try again.")
                
                with st.expander("🔍 Debug Information", expanded=False):
                    st.code(f"Error details: {str(e)}")
                    st.code(f"Error type: {type(e).__name__}")


# Support functions for backward compatibility
def display_research_results_with_selection(results_df):
    """Display research results with selection interface"""
    if results_df is None or results_df.empty:
        st.warning("No research results to display")
        return
    
    st.subheader("📊 Research Results")
    st.dataframe(results_df, use_container_width=True, height=400)


async def research_selected_businesses_enhanced(selected_business_names):
    """Placeholder for enhanced research functionality"""
    st.error("❌ Enhanced research functionality not available in this deployment.")
    return None, None, None
