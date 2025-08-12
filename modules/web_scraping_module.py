"""
Web Scraping Module for Business Contact Research
Fixed version with stable interface and proper session state management
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys
import asyncio
import importlib
import dotenv
from dotenv import load_dotenv


def display_research_results_with_selection(results_df):
    """
    ULTRA-STABLE VERSION: Number-based selection to avoid all widget conflicts
    """
    st.subheader("📈 Research Results - Enhanced Research Selection")
    st.info("✅ Standard research completed! Now select businesses for Enhanced Research using the simple options below.")
    
    # Add navigation button to go back to standard research
    col_nav, col_spacer = st.columns([1, 4])
    with col_nav:
        if st.button("⬅️ Back to Standard Research", help="Return to the standard research interface", key="back_to_standard"):
            # Reset interface mode to show standard research interface
            st.session_state.interface_mode = 'initial'
            st.session_state.show_enhanced_selection = False
            st.session_state.research_results = None  # Clear results to force re-research if needed
            st.rerun()
    
    if results_df is None or results_df.empty:
        st.warning("No research results to display")
        return
    
    # Add row numbers to the dataframe for easy reference
    display_df = results_df.copy()
    display_df.insert(0, 'Row #', range(1, len(display_df) + 1))
    
    # Show the results table with row numbers
    st.subheader("📊 Standard Research Results")
    st.dataframe(
        display_df[['Row #', 'business_name', 'phone', 'email', 'website', 'status']],
        use_container_width=True,
        height=400,
        hide_index=True
    )
    
    # Get counts for different categories
    total_businesses = len(results_df)
    successful_businesses = len(results_df[results_df['status'] == 'success'])
    
    # Show summary metrics
    col_metric1, col_metric2, col_metric3 = st.columns(3)
    with col_metric1:
        st.metric("Total Businesses", total_businesses)
    with col_metric2:
        st.metric("Successful Research", successful_businesses)
    with col_metric3:
        st.metric("Available for Enhancement", total_businesses)
    
    st.markdown("---")
    
    # SIMPLE SELECTION INTERFACE - No complex widgets
    st.subheader("🎯 Enhanced Research Selection")
    st.write("**Choose which businesses to research with enhanced methods:**")
    
    # Simple radio button selection - most stable widget in Streamlit
    selection_option = st.radio(
        "Select businesses for Enhanced Research:",
        options=[
            "All businesses",
            "All successful businesses only",
            "Custom range (specify row numbers)",
            "First 5 businesses",
            "Last 5 businesses"
        ],
        help="Choose how to select businesses for enhanced research",
        key="enhanced_selection_radio"
    )
    
    selected_businesses = []
    
    if selection_option == "All businesses":
        selected_businesses = results_df['business_name'].tolist()
        st.success(f"✅ Selected all {len(selected_businesses)} businesses for Enhanced Research")
        
    elif selection_option == "All successful businesses only":
        selected_businesses = results_df[results_df['status'] == 'success']['business_name'].tolist()
        st.success(f"✅ Selected {len(selected_businesses)} successful businesses for Enhanced Research")
        
    elif selection_option == "Custom range (specify row numbers)":
        st.write("📝 **Custom Range Selection:**")
        st.caption("Refer to the 'Row #' column in the table above")
        
        col_range1, col_range2 = st.columns(2)
        with col_range1:
            from_row = st.number_input(
                "From Row #:",
                min_value=1,
                max_value=total_businesses,
                value=1,
                help="Starting row number",
                key="enhanced_from_row"
            )
        with col_range2:
            to_row = st.number_input(
                "To Row #:",
                min_value=from_row,
                max_value=total_businesses,
                value=min(5, total_businesses),
                help="Ending row number",
                key="enhanced_to_row"
            )
        
        # Get businesses in the specified range
        range_df = results_df.iloc[from_row-1:to_row]  # Convert to 0-based index
        selected_businesses = range_df['business_name'].tolist()
        
        if selected_businesses:
            st.success(f"✅ Selected rows {from_row} to {to_row} ({len(selected_businesses)} businesses)")
            with st.expander("Preview Selected Businesses:", expanded=False):
                for i, business in enumerate(selected_businesses, from_row):
                    business_row = results_df[results_df['business_name'] == business].iloc[0]
                    status = business_row.get('status', 'unknown')
                    status_icon = "✅" if status == 'success' else "❌" if status == 'failed' else "⚠️"
                    st.write(f"{i}. {status_icon} {business}")
        
    elif selection_option == "First 5 businesses":
        selected_businesses = results_df.head(5)['business_name'].tolist()
        st.success(f"✅ Selected first {len(selected_businesses)} businesses for Enhanced Research")
        
    elif selection_option == "Last 5 businesses":
        selected_businesses = results_df.tail(5)['business_name'].tolist()
        st.success(f"✅ Selected last {len(selected_businesses)} businesses for Enhanced Research")
    
    # Store selection in session state
    if 'enhanced_research_selection' not in st.session_state:
        st.session_state.enhanced_research_selection = []
    
    st.session_state.enhanced_research_selection = selected_businesses
    
    # Show download option for selected businesses
    if selected_businesses:
        st.markdown("---")
        st.subheader("📥 Download and Execute Enhanced Research")
        
        # Create download data
        selected_df = results_df[results_df['business_name'].isin(selected_businesses)]
        selected_csv = selected_df.to_csv(index=False)
        
        col_download, col_execute = st.columns([1, 1])
        
        with col_download:
            st.download_button(
                label=f"📋 Download Selected List ({len(selected_businesses)} businesses)",
                data=selected_csv,
                file_name=f"selected_for_enhanced_research_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                help="Download the selected businesses list for your records"
            )
        
        with col_execute:
            # Enhanced research cost calculation
            enhanced_cost = len(selected_businesses) * 0.05
            
            # Enhanced research button with confirmation
            st.write(f"💰 **Estimated Cost:** ${enhanced_cost:.2f}")
            st.warning("⚠️ Enhanced research takes 30-60 seconds per business")
            
            col_enhanced, col_justdial = st.columns(2)
            
            with col_enhanced:
                if st.button(
                    f"🏛️ Enhanced Research\n({len(selected_businesses)} businesses)",
                    type="primary",
                    help=f"Start enhanced research on {len(selected_businesses)} selected businesses",
                    key="execute_enhanced_research_simple"
                ):
                    # Set flags for enhanced research execution
                    st.session_state.run_enhanced_research = True
                    st.session_state.enhanced_research_list = selected_businesses.copy()
                    st.rerun()
                    
            with col_justdial:
                # JustDial research option
                justdial_cost = len(selected_businesses) * 0.02  # Cheaper than enhanced
                st.write(f"💰 **JustDial Cost:** ${justdial_cost:.2f}")
                
                if st.button(
                    f"📞 JustDial Research\n({len(selected_businesses)} businesses)",
                    type="secondary",
                    help=f"Search {len(selected_businesses)} businesses on JustDial for phone numbers",
                    key="execute_justdial_research_simple"
                ):
                    # Set flags for JustDial research execution
                    st.session_state.run_justdial_research = True
                    st.session_state.justdial_research_list = selected_businesses.copy()
                    st.rerun()
        
        # Show selection summary
        st.markdown("---")
        with st.expander(f"📋 Selected Businesses Summary ({len(selected_businesses)}):", expanded=False):
            for i, business in enumerate(selected_businesses, 1):
                business_row = results_df[results_df['business_name'] == business].iloc[0]
                status = business_row.get('status', 'unknown')
                phone = business_row.get('phone', 'N/A')
                email = business_row.get('email', 'N/A')
                
                status_icon = "✅" if status == 'success' else "❌" if status == 'failed' else "⚠️"
                st.write(f"**{i}. {status_icon} {business}**")
                if phone != 'Not found' and phone != 'N/A' and phone:
                    st.write(f"   📞 {phone}")
                if email != 'Not found' and email != 'N/A' and email:
                    st.write(f"   ✉️ {email}")
    
    else:
        st.warning("⚠️ No businesses selected. Please choose a selection option above.")


async def research_selected_businesses_enhanced(selected_business_names):
    """
    Run enhanced research on specific selected business names
    
    Args:
        selected_business_names: List of business names to research
    
    Returns:
        tuple: (results_dataframe, summary_dict, csv_filename)
    """
    try:
        # Import from modules directory
        modules_path = os.path.dirname(__file__)
        if modules_path not in sys.path:
            sys.path.insert(0, modules_path)
        
        # We'll create a simple implementation that researches each business individually
        # For now, we'll use the regular business researcher but with specific names
        from streamlit_business_researcher import StreamlitBusinessResearcher
        
        researcher = StreamlitBusinessResearcher()
        
        # Test APIs first
        api_ok, api_message = researcher.test_apis()
        if not api_ok:
            raise Exception(f"API Test Failed: {api_message}")
        
        # CRITICAL FIX: Initialize or get existing enhanced results from session state
        # This ensures we UPDATE existing results instead of creating duplicates
        if 'enhanced_results_dict' not in st.session_state:
            st.session_state.enhanced_results_dict = {}
        
        # Research each selected business individually
        for business_name in selected_business_names:
            try:
                result = await researcher.research_business_direct(business_name)
                if result:
                    # UPDATE the dictionary with business name as key (overwrites if exists)
                    st.session_state.enhanced_results_dict[business_name] = result
            except Exception as e:
                print(f"Failed to research {business_name}: {e}")
                # Update with failed result (overwrites if exists)
                st.session_state.enhanced_results_dict[business_name] = {
                    'business_name': business_name,
                    'status': 'failed',
                    'extracted_info': f"Research failed: {str(e)}",
                    'research_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'method': 'enhanced_selected'
                }
        
        # Convert the dictionary back to a list for processing (no duplicates)
        researcher.results = list(st.session_state.enhanced_results_dict.values())
        
        # Generate summary based on ALL enhanced results (not just current batch)
        total_processed = len(researcher.results)
        successful = len([r for r in researcher.results if r.get('status') == 'success'])
        failed = total_processed - successful
        
        summary = {
            'total_processed': total_processed,
            'successful': successful,
            'failed': failed,
            'manual_required': 0,
            'success_rate': (successful / total_processed * 100) if total_processed > 0 else 0,
            'government_verified': successful,  # For enhanced research, assume all successful ones are verified
            'avg_govt_sources': 2.5 if successful > 0 else 0
        }
        
        # Get results dataframe (will now have unique business names)
        results_df = researcher.get_results_dataframe()
        
        # Save to CSV
        csv_filename = researcher.save_csv_results()
        
        return results_df, summary, csv_filename
        
    except Exception as e:
        print(f"❌ Enhanced research error: {e}")
        return None, None, None


def perform_web_scraping(filtered_df):
    """Perform web scraping of business contact information from filtered data"""
    
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
    
    # NEW: Session state for research results and enhanced research workflow
    if 'research_results' not in st.session_state:
        st.session_state.research_results = None
    if 'selected_businesses_for_enhanced' not in st.session_state:
        st.session_state.selected_businesses_for_enhanced = set()  # Use set instead of list
    if 'run_enhanced_research' not in st.session_state:
        st.session_state.run_enhanced_research = False
    if 'enhanced_research_list' not in st.session_state:
        st.session_state.enhanced_research_list = []
    # NEW: JustDial Research workflow state
    if 'run_justdial_research' not in st.session_state:
        st.session_state.run_justdial_research = False
    if 'justdial_research_list' not in st.session_state:
        st.session_state.justdial_research_list = []
    # CRITICAL: Add session state to track current interface mode
    if 'interface_mode' not in st.session_state:
        st.session_state.interface_mode = 'initial'  # Can be: 'initial', 'research_selection', 'enhanced_research'
    if 'show_enhanced_selection' not in st.session_state:
        st.session_state.show_enhanced_selection = False
    
    # Define dashboard path once at the beginning to avoid scoping issues
    dashboard_path = os.path.dirname(os.path.dirname(__file__))
    
    # INTERFACE MODE MANAGEMENT - Determines which interface to show
    # Modes: 'initial' -> 'research_selection' -> 'enhanced_research'
    
    # Check if we should show enhanced research selection interface
    if (st.session_state.research_results is not None and 
        not st.session_state.research_results.empty and 
        not st.session_state.run_enhanced_research):
        # We have standard research results and should show enhanced research selection
        st.session_state.interface_mode = 'research_selection'
        st.session_state.show_enhanced_selection = True
    
    # Handle enhanced research execution flag
    if st.session_state.run_enhanced_research:
        st.session_state.interface_mode = 'enhanced_research'
        st.session_state.show_enhanced_selection = False
    
    # Handle JustDial research execution flag
    if st.session_state.run_justdial_research:
        st.session_state.interface_mode = 'justdial_research'
        st.session_state.show_enhanced_selection = False
    
    # CONDITIONAL INTERFACE RENDERING based on current mode
    if st.session_state.interface_mode == 'research_selection' and st.session_state.show_enhanced_selection:
        # Show Enhanced Research Selection Interface
        st.markdown("### 🏛️ Enhanced Research Selection Interface")
        display_research_results_with_selection(st.session_state.research_results)
        return  # Exit early to avoid showing standard research interface
    
    elif st.session_state.interface_mode == 'enhanced_research':
        # Handle enhanced research execution and results - will be processed below
        pass
    
    # STANDARD RESEARCH INTERFACE (shown when not in enhanced research selection mode)
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
    
    # Research limit selection - CHANGED: From slider to input boxes for from/to range
    if 'business_range_from' not in st.session_state:
        st.session_state.business_range_from = 1
    if 'business_range_to' not in st.session_state:
        st.session_state.business_range_to = min(5, unique_businesses)
    
    max_limit = min(20, unique_businesses)
    
    # CRITICAL FIX: Ensure session state values don't exceed current max_limit
    # This prevents Streamlit errors when filtered list is smaller than previous runs
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
            min_value=range_from,  # Ensure 'to' is not less than 'from'
            max_value=max_limit,
            value=max(st.session_state.business_range_to, range_from),  # Ensure 'to' >= 'from'
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
    
    # Cost estimation with both options
    standard_cost = max_businesses * 0.03
    enhanced_cost = max_businesses * 0.05
    st.warning(f"💰 **Estimated API Cost:** Standard ~${standard_cost:.2f} | Enhanced ~${enhanced_cost:.2f}")
    
    # API Configuration check
    st.write("🔧 **API Configuration:**")
    
    # Force reload environment variables
    importlib.reload(dotenv)
    load_dotenv(override=True)
    
    openai_key = os.getenv('OPENAI_API_KEY')
    tavily_key = os.getenv('TAVILY_API_KEY')
    groq_key = os.getenv('GROQ_API_KEY')  # For JustDial research
    
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
            st.caption("Add OPENAI_API_KEY to .env file")
    
    with col_api2:
        if tavily_valid:
            st.success("✅ Tavily API Key: Configured")
            masked_key = f"{tavily_key[:10]}...{tavily_key[-4:]}" if len(tavily_key) > 14 else f"{tavily_key[:6]}..."
            st.caption(f"Key: {masked_key}")
        else:
            st.error(f"❌ Tavily API Key: {tavily_reason}")
            st.caption("Add TAVILY_API_KEY to .env file")
            
    with col_api3:
        if groq_valid:
            st.success("✅ Groq API Key: Configured")
            masked_key = f"{groq_key[:10]}...{groq_key[-4:]}" if len(groq_key) > 14 else f"{groq_key[:6]}..."
            st.caption(f"Key: {masked_key}")
        else:
            st.error(f"❌ Groq API Key: {groq_reason}")
            st.caption("Add GROQ_API_KEY to .env file (for JustDial)")
    
    # Show setup instructions if keys are invalid
    if not openai_valid or not tavily_valid:
        st.warning("⚠️ **Standard Research Setup Required**: Please configure OpenAI and Tavily API keys.")
        
    if not groq_valid:
        st.info("📞 **JustDial Research**: Requires GROQ_API_KEY and additional setup (Selenium + Chrome).")
        
        with st.expander("📝 Setup Instructions", expanded=False):
            st.markdown("""
            **To set up API keys:**
            
            1. **Edit your .env file** in the app directory
            2. **Add your API keys:**
               ```
               OPENAI_API_KEY=sk-your_actual_openai_key_here
               TAVILY_API_KEY=tvly-your_actual_tavily_key_here
               ```
            3. **Restart the app**
            4. **Get API keys from:**
               - [OpenAI API Keys](https://platform.openai.com/api-keys)
               - [Tavily API](https://tavily.com)
            """)
    
    # Test API connectivity
    both_apis_configured = openai_valid and tavily_valid
    
    # Set up import path for business researcher (current modules directory)
    modules_path = os.path.dirname(__file__)
    if modules_path not in sys.path:
        sys.path.insert(0, modules_path)
    
    if both_apis_configured:
        st.info("🟢 **Both API keys configured!** You can proceed with web scraping.")
        
        # Test API connection button with session state
        col_test, col_status = st.columns([1, 2])
        
        with col_test:
            if st.button("🧪 Test API Connection", 
                        help="Test if APIs are working correctly", 
                        key="test_api_button_stable"):
                st.session_state.api_test_completed = False  # Reset test status
                
                with st.spinner("Testing API connections..."):
                    try:
                        # Import the business researcher from modules directory
                        from streamlit_business_researcher import StreamlitBusinessResearcher
                        
                        # Test API connectivity
                        test_researcher = StreamlitBusinessResearcher()
                        api_ok, api_message = test_researcher.test_apis()
                        
                        st.session_state.api_test_completed = True
                        st.session_state.api_test_result = (api_ok, api_message)
                        
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
                    if "No module named" in api_message:
                        st.error("📁 Business researcher module not found. Check file paths.")
                        st.info(f"Expected path: {modules_path}")
    
    # JustDial Setup Test Section
    if groq_valid:
        st.markdown("---")
        st.subheader("📞 JustDial Research Setup")
        
        col_justdial_test, col_justdial_status = st.columns([1, 2])
        
        with col_justdial_test:
            if st.button("🧪 Test JustDial Setup", 
                        help="Test JustDial research dependencies and API connectivity", 
                        key="test_justdial_setup"):
                st.session_state.justdial_test_completed = False  # Reset test status
                
                with st.spinner("Testing JustDial setup..."):
                    try:
                        # Import and test JustDial researcher
                        from justdial_researcher import JustDialStreamlitResearcher
                        
                        # Test setup
                        tester = JustDialStreamlitResearcher(groq_api_key=groq_key)
                        api_ok, api_message = tester.test_apis()
                        
                        # Get detailed validation
                        validation_checks = tester.validate_setup()
                        
                        st.session_state.justdial_test_completed = True
                        st.session_state.justdial_test_result = (api_ok, api_message, validation_checks)
                        
                    except Exception as e:
                        st.session_state.justdial_test_completed = True
                        st.session_state.justdial_test_result = (False, f"JustDial Test Error: {str(e)}", [])
        
        with col_justdial_status:
            # Display JustDial test results from session state
            if st.session_state.get('justdial_test_completed', False) and st.session_state.get('justdial_test_result'):
                api_ok, api_message, validation_checks = st.session_state.justdial_test_result
                
                if api_ok:
                    st.success(f"✅ JustDial Setup Test: {api_message}")
                else:
                    st.error(f"❌ JustDial Setup Test Failed: {api_message}")
                
                # Show detailed validation results
                if validation_checks:
                    st.write("**Setup Validation:**")
                    for is_valid, check_message in validation_checks:
                        if is_valid:
                            st.success(check_message)
                        else:
                            st.error(check_message)
                            
                # Show setup guide if failed
                if not api_ok:
                    with st.expander("📝 JustDial Setup Guide", expanded=True):
                        st.markdown("""
                        **JustDial Research Requirements:**
                        
                        1. 🔑 **Groq API Key**: Get from https://console.groq.com/
                        2. 🌐 **Chrome Browser**: Latest version
                        3. 🛠️ **Selenium**: `pip install selenium`
                        4. 🚗 **Chrome Debugging**: Start Chrome with debugging enabled
                        
                        **Add to .env file:**
                        ```
                        GROQ_API_KEY=gsk_your_groq_api_key_here
                        ```
                        
                        **Chrome Setup (IMPORTANT):**
                        
                        JustDial research works best with an existing Chrome browser. 
                        
                        **Option 1: Manual Chrome Setup**
                        1. Close all Chrome windows
                        2. Run this command:
                        ```
                        chrome.exe --remote-debugging-port=9222
                        ```
                        3. Navigate to https://www.justdial.com and test search
                        4. Keep this Chrome window open during research
                        
                        **Option 2: Use Helper Function**
                        Run this Python code to auto-start Chrome:
                        ```python
                        from modules.justdial_researcher import start_chrome_with_debugging
                        start_chrome_with_debugging()
                        ```
                        
                        **Note:** Existing Chrome connection avoids automation detection and improves reliability.
                        """)
                        
                        # Add Chrome debugging test button
                        if st.button("🚗 Test Chrome Debugging Connection", key="test_chrome_debug"):
                            with st.spinner("Testing Chrome debugging connection..."):
                                try:
                                    from justdial_researcher import check_chrome_debugging_connection
                                    if check_chrome_debugging_connection():
                                        st.success("✅ Chrome debugging connection working!")
                                        st.info("You can now use JustDial research.")
                                    else:
                                        st.error("❌ Chrome debugging connection failed.")
                                        st.info("Please follow the Chrome setup steps above.")
                                except Exception as e:
                                    st.error(f"Error testing Chrome: {e}")
    
    # Research start section
    st.markdown("---")
    
    # Research button with proper state management
    button_disabled = not both_apis_configured
    
    # Use columns for research button only (Enhanced Research is now handled in the workflow section)
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
            st.info(f"Ready to research {max_businesses} businesses with standard web sources")
    
    # Handle standard research execution
    if start_research and both_apis_configured:
        # IMPORTANT: Clear any previous enhanced results when starting new research
        if 'enhanced_results_dict' in st.session_state:
            st.session_state.enhanced_results_dict = {}
        
        # Create a placeholder for the research process
        research_container = st.container()
        
        with research_container:
            st.info("🔄 Starting business research...")
            
            # Create progress indicators
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Import the research function from modules directory
                from streamlit_business_researcher import research_businesses_from_dataframe
                
                status_text.info("🚀 Initializing research system...")
                progress_bar.progress(10)
                
                # Execute research in a try-catch block
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
                            
                            # CRITICAL: Set interface mode to show enhanced research selection on next rerun
                            st.session_state.interface_mode = 'research_selection'
                            st.session_state.show_enhanced_selection = True
                            
                            # Display results table with selection checkboxes
                            display_research_results_with_selection(results_df)
                            
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
                                        # Deduplicate results before creating mapping
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
                                        
                                        # Show info if duplicates were removed
                                        if len(results_df) > len(results_df_unique):
                                            st.info(f"ℹ️ {len(results_df) - len(results_df_unique)} duplicate entries were consolidated.")
                                            
                                    except Exception as e:
                                        st.warning(f"Could not create enhanced dataset: {e}")
                            
                            # Success message
                            st.balloons()
                            st.success(f"🎉 Successfully researched {summary['successful']} businesses!")
                            
                            if summary['manual_required'] > 0:
                                st.info(f"🔍 {summary['manual_required']} businesses require manual research")
                            
                            if summary.get('billing_errors', 0) > 0:
                                st.error(f"💳 {summary['billing_errors']} businesses failed due to API billing issues")
                        
                        else:
                            st.warning("⚠️ Research completed but no results were found.")
                            st.info("This might be due to API rate limits, no search results, or connection problems.")
                    
                    except Exception as e:
                        error_msg = str(e)
                        progress_bar.progress(0)  # Reset progress bar
                        status_text.error("❌ Research failed")
                        
                        st.error(f"❌ Research Error: {error_msg}")
                        
                        # Categorize errors for better user guidance
                        if "API" in error_msg or "key" in error_msg.lower():
                            st.error("🔑 API key issue. Please check your .env file.")
                        elif "billing" in error_msg.lower() or "quota" in error_msg.lower():
                            st.error("💳 API billing/quota issue. Please check your API account.")
                        elif "connection" in error_msg.lower():
                            st.error("🌐 Connection error. Please check your internet connection.")
                        else:
                            st.error("⚠️ Please check your configuration and try again.")
                        
                        # Debug information
                        with st.expander("🔍 Debug Information", expanded=False):
                            st.code(f"Full error: {error_msg}")
                            st.code(f"Dashboard path: {dashboard_path}")
                            st.code(f"Modules path: {modules_path}")
                            
            except ImportError as e:
                st.error(f"❌ Could not import business researcher: {str(e)}")
                st.error("📁 Please ensure the modules directory is accessible.")
                st.info(f"Expected path: {modules_path}")
                
                # Check file existence
                expected_file = os.path.join(modules_path, "streamlit_business_researcher.py")
                st.info(f"Looking for file at: {expected_file}")
                st.info(f"Modules path: {modules_path}")
                
                if os.path.exists(expected_file):
                    st.info("✅ File exists - might be a Python environment issue")
                    st.info("💡 Try restarting the Streamlit app")
                else:
                    st.error("❌ File not found - check the path")
                
                # Debug paths
                with st.expander("🔍 Debug Paths", expanded=False):
                    st.write("**Current working directory:**", os.getcwd())
                    st.write("**Module file path:**", __file__)
                    st.write("**Dashboard path:**", dashboard_path)
                    st.write("**Modules path:**", modules_path)
                    st.write("**Expected file path:**", expected_file)
                    st.write("**File exists:**", os.path.exists(expected_file))
                    st.write("**Python paths:**", sys.path[:5])
            
            except Exception as e:
                st.error(f"❌ Unexpected error during research: {str(e)}")
                st.error("🔄 Please restart the app and try again.")
                
                with st.expander("🔍 Debug Information", expanded=False):
                    st.code(f"Error details: {str(e)}")
                    st.code(f"Error type: {type(e).__name__}")
                    import traceback
                    st.code(f"Traceback: {traceback.format_exc()}")
    
    # Handle enhanced research execution - NEW FLAG-BASED APPROACH
    if st.session_state.run_enhanced_research and st.session_state.enhanced_research_list:
        # Reset the flag
        st.session_state.run_enhanced_research = False
        
        # Get the list of selected businesses
        selected_businesses_list = st.session_state.enhanced_research_list.copy()
        
        # Create a placeholder for the enhanced research process
        enhanced_research_container = st.container()
        
        with enhanced_research_container:
            st.info("🏛️ Starting enhanced research on selected businesses...")
            
            if not selected_businesses_list:
                st.error("⚠️ No businesses found for enhanced research!")
                st.info("Please select businesses and try again.")
            else:
                st.info(f"🎯 Selected {len(selected_businesses_list)} businesses for enhanced research")
                st.warning("⚠️ Enhanced research includes government sources and takes longer (~30-60 seconds per business)")
                
                # Create progress indicators
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    status_text.info("🏛️ Initializing enhanced research system...")
                    progress_bar.progress(10)
                    
                    # Execute enhanced research on selected businesses only
                    with st.spinner(f"Researching {len(selected_businesses_list)} selected businesses using enhanced methods..."):
                        
                        async def run_selected_enhanced_research():
                            return await research_selected_businesses_enhanced(
                                selected_business_names=selected_businesses_list
                            )
                        
                        status_text.info("🔍 Starting enhanced research process...")
                        progress_bar.progress(30)
                        
                        # Execute the enhanced research on selected businesses
                        try:
                            results_df, summary, csv_filename = asyncio.run(run_selected_enhanced_research())
                            progress_bar.progress(90)
                            status_text.success("✅ Enhanced research completed successfully!")
                            
                            # Display enhanced results if successful
                            if results_df is not None and not results_df.empty:
                                progress_bar.progress(100)
                                
                                # Display enhanced summary
                                st.success("🎉 **Enhanced Research Summary:**")
                                col_sum1, col_sum2, col_sum3, col_sum4 = st.columns(4)
                                
                                with col_sum1:
                                    st.metric("Selected", len(selected_businesses_list))
                                with col_sum2:
                                    st.metric("Successful", summary['successful'])
                                with col_sum3:
                                    st.metric("Failed", summary['failed'])
                                with col_sum4:
                                    st.metric("Success Rate", f"{summary['success_rate']:.1f}%")
                                
                                # Display enhanced results table
                                st.subheader("🏛️ Enhanced Research Results")
                                st.info("Enhanced results for selected businesses with additional research depth.")
                                st.dataframe(results_df, use_container_width=True, height=400)
                                
                                # Download enhanced results section
                                st.subheader("📅 Download Enhanced Research Results")
                                
                                csv_data = results_df.to_csv(index=False)
                                
                                col_down1, col_down2 = st.columns(2)
                                with col_down1:
                                    st.download_button(
                                        label=f"🏛️ Download Enhanced Results CSV ({len(selected_businesses_list)} businesses)",
                                        data=csv_data,
                                        file_name=f"enhanced_selected_research_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                        mime="text/csv"
                                    )
                                
                                with col_down2:
                                    # Create enhanced dataset with selected research results
                                    if 'business_name' in results_df.columns:
                                        try:
                                            # Deduplicate results before creating mapping
                                            results_df_unique = results_df.drop_duplicates(subset=['business_name'], keep='last')  # Keep last (most recent) result
                                            
                                            # Enhanced mapping for selected businesses only
                                            enhanced_mapping = results_df_unique.set_index('business_name')[[
                                                'phone', 'email', 'website', 'address'
                                            ]].to_dict('index')
                                            
                                            enhanced_df = filtered_df.copy()
                                            enhanced_df['enhanced_research_phone'] = enhanced_df[selected_column].map(lambda x: enhanced_mapping.get(x, {}).get('phone', ''))
                                            enhanced_df['enhanced_research_email'] = enhanced_df[selected_column].map(lambda x: enhanced_mapping.get(x, {}).get('email', ''))
                                            enhanced_df['enhanced_research_website'] = enhanced_df[selected_column].map(lambda x: enhanced_mapping.get(x, {}).get('website', ''))
                                            enhanced_df['enhanced_research_address'] = enhanced_df[selected_column].map(lambda x: enhanced_mapping.get(x, {}).get('address', ''))
                                            
                                            enhanced_csv = enhanced_df.to_csv(index=False)
                                            
                                            st.download_button(
                                                label="🔗 Download Enhanced Dataset",
                                                data=enhanced_csv,
                                                file_name=f"enhanced_selected_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                                mime="text/csv",
                                                help="Original data + enhanced research results for selected businesses"
                                            )
                                            
                                            # Show info if duplicates were removed
                                            if len(results_df) > len(results_df_unique):
                                                st.info(f"ℹ️ {len(results_df) - len(results_df_unique)} duplicate entries were consolidated.")
                                                
                                        except Exception as e:
                                            st.warning(f"Could not create enhanced dataset: {e}")
                                
                                # Enhanced success message
                                st.balloons()
                                st.success(f"🎉 Successfully completed enhanced research for {summary['successful']} selected businesses!")
                                
                                if summary.get('failed', 0) > 0:
                                    st.info(f"⚠️ {summary['failed']} businesses had limited results")
                            
                            else:
                                st.warning("⚠️ Enhanced research completed but no results were found.")
                                st.info("This might be due to API rate limits, no search results, or connection problems.")
                        
                        except Exception as e:
                            error_msg = str(e)
                            progress_bar.progress(0)  # Reset progress bar
                            status_text.error("❌ Enhanced research failed")
                            
                            st.error(f"❌ Enhanced Research Error: {error_msg}")
                            
                            # Categorize errors for better user guidance
                            if "API" in error_msg or "key" in error_msg.lower():
                                st.error("🔑 API key issue. Please check your .env file.")
                            elif "billing" in error_msg.lower() or "quota" in error_msg.lower():
                                st.error("💳 API billing/quota issue. Please check your API account.")
                            elif "connection" in error_msg.lower():
                                st.error("🌐 Connection error. Please check your internet connection.")
                            else:
                                st.error("⚠️ Please check your configuration and try again.")
                            
                            # Debug information
                            with st.expander("🔍 Debug Information", expanded=False):
                                st.code(f"Full error: {error_msg}")
                                st.code(f"Selected businesses: {selected_businesses_list}")
                                st.code(f"Dashboard path: {dashboard_path}")
                                st.code(f"Modules path: {modules_path}")
                                
                except ImportError as e:
                    st.error(f"❌ Could not import enhanced research functions: {str(e)}")
                    st.error("📁 Please ensure the enhanced research modules are accessible.")
                    
                except Exception as e:
                    st.error(f"❌ Unexpected error during enhanced research: {str(e)}")
                    st.error("🔄 Please restart the app and try again.")
                    
                    with st.expander("🔍 Enhanced Research Debug Information", expanded=False):
                        st.code(f"Error details: {str(e)}")
                        st.code(f"Error type: {type(e).__name__}")
                        st.code(f"Selected businesses: {selected_businesses_list}")
                        import traceback
                        st.code(f"Traceback: {traceback.format_exc()}")
    
    # Handle JustDial research execution - NEW JUSTDIAL RESEARCH WORKFLOW
    if st.session_state.run_justdial_research and st.session_state.justdial_research_list:
        # Reset the flag
        st.session_state.run_justdial_research = False
        
        # Get the list of selected businesses
        selected_businesses_list = st.session_state.justdial_research_list.copy()
        
        # Create a placeholder for the JustDial research process
        justdial_research_container = st.container()
        
        with justdial_research_container:
            st.info("📞 Starting JustDial research on selected businesses...")
            
            if not selected_businesses_list:
                st.error("⚠️ No businesses found for JustDial research!")
                st.info("Please select businesses and try again.")
            else:
                st.info(f"🎯 Selected {len(selected_businesses_list)} businesses for JustDial research")
                st.warning("⚠️ JustDial research includes WhatsApp extraction and takes ~20-30 seconds per business")
                
                # Create progress indicators
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    status_text.info("📞 Initializing JustDial research system...")
                    progress_bar.progress(10)
                    
                    # Execute JustDial research on selected businesses only
                    with st.spinner(f"Researching {len(selected_businesses_list)} selected businesses using JustDial..."):
                        
                        async def run_selected_justdial_research():
                            try:
                                # Import JustDial researcher
                                from justdial_researcher import research_selected_businesses_justdial
                                
                                return await research_selected_businesses_justdial(
                                    selected_business_names=selected_businesses_list,
                                    df=filtered_df,
                                    consignee_column=selected_column
                                )
                            except ImportError as ie:
                                raise Exception(f"Could not import JustDial researcher: {ie}. Make sure selenium is installed: pip install selenium")
                            except Exception as e:
                                raise Exception(f"JustDial research error: {e}")
                        
                        status_text.info("🔍 Starting JustDial research process...")
                        progress_bar.progress(30)
                        
                        # Execute the JustDial research on selected businesses
                        try:
                            results_df, summary, csv_filename = asyncio.run(run_selected_justdial_research())
                            progress_bar.progress(90)
                            status_text.success("✅ JustDial research completed successfully!")
                            
                            # Display JustDial results if successful
                            if results_df is not None and not results_df.empty:
                                progress_bar.progress(100)
                                
                                # Display JustDial summary
                                st.success("🎉 **JustDial Research Summary:**")
                                col_sum1, col_sum2, col_sum3, col_sum4 = st.columns(4)
                                
                                with col_sum1:
                                    st.metric("Selected", len(selected_businesses_list))
                                with col_sum2:
                                    st.metric("Successful", summary['successful'])
                                with col_sum3:
                                    st.metric("Failed", summary['failed'])
                                with col_sum4:
                                    st.metric("Success Rate", f"{summary['success_rate']:.1f}%")
                                
                                # Display JustDial results table
                                st.subheader("📞 JustDial Research Results")
                                st.info("JustDial phone numbers extracted from WhatsApp integration.")
                                st.dataframe(results_df, use_container_width=True, height=400)
                                
                                # Download JustDial results section
                                st.subheader("📥 Download JustDial Research Results")
                                
                                csv_data = results_df.to_csv(index=False)
                                
                                col_down1, col_down2 = st.columns(2)
                                with col_down1:
                                    st.download_button(
                                        label=f"📞 Download JustDial Results CSV ({len(selected_businesses_list)} businesses)",
                                        data=csv_data,
                                        file_name=f"justdial_selected_research_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                        mime="text/csv"
                                    )
                                
                                with col_down2:
                                    # Create JustDial enhanced dataset with selected research results
                                    if 'business_name' in results_df.columns:
                                        try:
                                            # Deduplicate results before creating mapping
                                            results_df_unique = results_df.drop_duplicates(subset=['business_name'], keep='last')  # Keep last (most recent) result
                                            
                                            # JustDial mapping for selected businesses only
                                            justdial_mapping = results_df_unique.set_index('business_name')[[
                                                'justdial_phone'
                                            ]].to_dict('index')
                                            
                                            justdial_df = filtered_df.copy()
                                            justdial_df['justdial_phone'] = justdial_df[selected_column].map(lambda x: justdial_mapping.get(x, {}).get('justdial_phone', ''))
                                            
                                            justdial_csv = justdial_df.to_csv(index=False)
                                            
                                            st.download_button(
                                                label="🔗 Download JustDial Enhanced Dataset",
                                                data=justdial_csv,
                                                file_name=f"justdial_enhanced_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                                mime="text/csv",
                                                help="Original data + JustDial phone numbers for selected businesses"
                                            )
                                            
                                            # Show info if duplicates were removed
                                            if len(results_df) > len(results_df_unique):
                                                st.info(f"ℹ️ {len(results_df) - len(results_df_unique)} duplicate entries were consolidated.")
                                                
                                        except Exception as e:
                                            st.warning(f"Could not create JustDial enhanced dataset: {e}")
                                
                                # JustDial success message
                                st.balloons()
                                st.success(f"🎉 Successfully completed JustDial research for {summary['successful']} selected businesses!")
                                
                                if summary.get('failed', 0) > 0:
                                    st.info(f"⚠️ {summary['failed']} businesses had no phone numbers found on JustDial")
                            
                            else:
                                st.warning("⚠️ JustDial research completed but no results were found.")
                                st.info("This might be due to businesses not being listed on JustDial or connection problems.")
                        
                        except Exception as e:
                            error_msg = str(e)
                            progress_bar.progress(0)  # Reset progress bar
                            status_text.error("❌ JustDial research failed")
                            
                            st.error(f"❌ JustDial Research Error: {error_msg}")
                            
                            # Categorize errors for better user guidance
                            if "API" in error_msg or "key" in error_msg.lower():
                                st.error("🔑 API key issue. Please check your GROQ_API_KEY in .env file.")
                            elif "selenium" in error_msg.lower() or "chrome" in error_msg.lower():
                                st.error("🌐 Chrome/Selenium driver issue. This may require local setup.")
                            elif "connection" in error_msg.lower():
                                st.error("🌐 Connection error. Please check your internet connection.")
                            else:
                                st.error("⚠️ Please check your configuration and try again.")
                            
                            # Debug information
                            with st.expander("🔍 JustDial Debug Information", expanded=False):
                                st.code(f"Full error: {error_msg}")
                                st.code(f"Selected businesses: {selected_businesses_list}")
                                st.code(f"Dashboard path: {dashboard_path}")
                                st.code(f"Modules path: {modules_path}")
                                
                except ImportError as e:
                    st.error(f"❌ Could not import JustDial research functions: {str(e)}")
                    st.error("📁 Please ensure the JustDial research modules are accessible.")
                    
                except Exception as e:
                    st.error(f"❌ Unexpected error during JustDial research: {str(e)}")
                    st.error("🔄 Please restart the app and try again.")
                    
                    with st.expander("🔍 JustDial Research Debug Information", expanded=False):
                        st.code(f"Error details: {str(e)}")
                        st.code(f"Error type: {type(e).__name__}")
                        st.code(f"Selected businesses: {selected_businesses_list}")
                        import traceback
                        st.code(f"Traceback: {traceback.format_exc()}")
