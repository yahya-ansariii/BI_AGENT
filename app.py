"""
Business Insights Agent - Main Streamlit Application
A local AI-powered business intelligence tool for Excel data analysis
"""

import streamlit as st
import pandas as pd
import duckdb
import json
import os
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

from modules.data_processor import DataProcessor
from modules.visualizer import Visualizer
from modules.llm_agent import LLMAgent
from modules.schema_manager import SchemaManager

# Page configuration
st.set_page_config(
    page_title="Business Insights Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stButton > button {
        background-color: #1f77b4;
        color: white;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main application function"""
    
    # Initialize session state
    if 'data_processor' not in st.session_state:
        st.session_state.data_processor = DataProcessor()
    if 'visualizer' not in st.session_state:
        st.session_state.visualizer = Visualizer()
    if 'llm_agent' not in st.session_state:
        st.session_state.llm_agent = LLMAgent()
    if 'schema_manager' not in st.session_state:
        st.session_state.schema_manager = SchemaManager()
    
    # Header
    st.markdown('<h1 class="main-header">📊 Business Insights Agent</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # Data source selection
        st.subheader("Data Source")
        data_source = st.selectbox(
            "Choose data source:",
            ["Upload Excel File", "Sample Data", "Connect to Database"]
        )
        
        # LLM Configuration
        st.subheader("AI Configuration")
        model_name = st.selectbox(
            "Select LLM Model:",
            ["llama2", "codellama", "mistral", "phi"]
        )
        
        if st.button("🔄 Initialize AI Agent"):
            with st.spinner("Initializing AI agent..."):
                try:
                    st.session_state.llm_agent.initialize_model(model_name)
                    st.success(f"✅ {model_name} model initialized!")
                except Exception as e:
                    st.error(f"❌ Error initializing model: {str(e)}")
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Dashboard", "🔍 Data Explorer", "🤖 AI Insights", "⚙️ Settings"])
    
    with tab1:
        dashboard_tab()
    
    with tab2:
        data_explorer_tab()
    
    with tab3:
        ai_insights_tab()
    
    with tab4:
        settings_tab()

def dashboard_tab():
    """Dashboard tab with key metrics and visualizations"""
    st.header("📈 Business Dashboard")
    
    # Check if data is loaded
    if not st.session_state.data_processor.has_data():
        st.warning("⚠️ No data loaded. Please upload an Excel file or use sample data.")
        return
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", st.session_state.data_processor.get_record_count())
    
    with col2:
        st.metric("Data Columns", st.session_state.data_processor.get_column_count())
    
    with col3:
        st.metric("Memory Usage", f"{st.session_state.data_processor.get_memory_usage():.2f} MB")
    
    with col4:
        st.metric("Data Quality", f"{st.session_state.data_processor.get_data_quality_score():.1f}%")
    
    st.markdown("---")
    
    # Quick visualizations
    st.subheader("📊 Quick Insights")
    
    # Get data summary
    summary = st.session_state.data_processor.get_data_summary()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Data Types Distribution")
        type_counts = summary['dtypes'].value_counts()
        fig = px.pie(values=type_counts.values, names=type_counts.index, title="Column Types")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Missing Values")
        missing_data = summary['missing_values']
        if missing_data.sum() > 0:
            fig = px.bar(x=missing_data.index, y=missing_data.values, title="Missing Values per Column")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("✅ No missing values found!")

def data_explorer_tab():
    """Data explorer tab for detailed data analysis"""
    st.header("🔍 Data Explorer")
    
    if not st.session_state.data_processor.has_data():
        st.warning("⚠️ No data loaded. Please upload an Excel file or use sample data.")
        return
    
    # Data upload section
    st.subheader("📁 Upload Data")
    uploaded_file = st.file_uploader(
        "Choose an Excel file",
        type=['xlsx', 'xls'],
        help="Upload your business data in Excel format"
    )
    
    if uploaded_file is not None:
        try:
            with st.spinner("Processing uploaded file..."):
                st.session_state.data_processor.load_excel_data(uploaded_file)
            st.success("✅ Data loaded successfully!")
        except Exception as e:
            st.error(f"❌ Error loading data: {str(e)}")
    
    # Sample data option
    if st.button("📊 Load Sample Data"):
        with st.spinner("Loading sample data..."):
            st.session_state.data_processor.load_sample_data()
        st.success("✅ Sample data loaded!")
    
    # Data preview
    if st.session_state.data_processor.has_data():
        st.subheader("📋 Data Preview")
        
        # Display options
        col1, col2 = st.columns([3, 1])
        with col1:
            show_rows = st.slider("Number of rows to display", 5, 100, 10)
        with col2:
            show_columns = st.multiselect(
                "Select columns to display",
                st.session_state.data_processor.get_column_names(),
                default=st.session_state.data_processor.get_column_names()[:5]
            )
        
        # Display data
        data_preview = st.session_state.data_processor.get_data_preview(show_rows, show_columns)
        st.dataframe(data_preview, use_container_width=True)
        
        # Data statistics
        st.subheader("📊 Data Statistics")
        st.dataframe(st.session_state.data_processor.get_descriptive_stats())

def ai_insights_tab():
    """AI-powered insights tab"""
    st.header("🤖 AI-Powered Insights")
    
    if not st.session_state.data_processor.has_data():
        st.warning("⚠️ No data loaded. Please upload an Excel file or use sample data.")
        return
    
    # Query input
    st.subheader("💬 Ask Questions About Your Data")
    
    user_query = st.text_area(
        "Enter your question:",
        placeholder="e.g., What are the top 5 products by sales? Show me trends in customer acquisition over time.",
        height=100
    )
    
    col1, col2 = st.columns([3, 1])
    with col1:
        if st.button("🔍 Generate Insights", type="primary"):
            if user_query:
                with st.spinner("AI is analyzing your data..."):
                    try:
                        response = st.session_state.llm_agent.analyze_data(
                            user_query, 
                            st.session_state.data_processor.get_data()
                        )
                        st.success("✅ Analysis complete!")
                        st.write(response)
                    except Exception as e:
                        st.error(f"❌ Error generating insights: {str(e)}")
            else:
                st.warning("Please enter a question first.")
    
    with col2:
        if st.button("🎯 Quick Analysis"):
            quick_insights = st.session_state.llm_agent.get_quick_insights(
                st.session_state.data_processor.get_data()
            )
            st.write(quick_insights)
    
    # Pre-defined analysis templates
    st.subheader("📋 Analysis Templates")
    
    template_col1, template_col2 = st.columns(2)
    
    with template_col1:
        if st.button("📈 Sales Analysis"):
            analysis = st.session_state.llm_agent.sales_analysis(
                st.session_state.data_processor.get_data()
            )
            st.write(analysis)
    
    with template_col2:
        if st.button("👥 Customer Analysis"):
            analysis = st.session_state.llm_agent.customer_analysis(
                st.session_state.data_processor.get_data()
            )
            st.write(analysis)

def settings_tab():
    """Settings and configuration tab"""
    st.header("⚙️ Settings & Configuration")
    
    # Schema management
    st.subheader("🗂️ Data Schema Management")
    
    if st.button("📋 View Current Schema"):
        schema = st.session_state.schema_manager.get_schema()
        st.json(schema)
    
    if st.button("🔄 Update Schema"):
        st.session_state.schema_manager.update_schema_from_data(
            st.session_state.data_processor.get_data()
        )
        st.success("✅ Schema updated!")
    
    # Export options
    st.subheader("📤 Export Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("💾 Export Data as CSV"):
            if st.session_state.data_processor.has_data():
                csv_data = st.session_state.data_processor.export_to_csv()
                st.download_button(
                    label="Download CSV",
                    data=csv_data,
                    file_name="business_data.csv",
                    mime="text/csv"
                )
    
    with col2:
        if st.button("📊 Export Insights as JSON"):
            insights = st.session_state.llm_agent.get_insights_history()
            st.download_button(
                label="Download Insights",
                data=json.dumps(insights, indent=2),
                file_name="business_insights.json",
                mime="application/json"
            )
    
    # System information
    st.subheader("ℹ️ System Information")
    
    info_col1, info_col2 = st.columns(2)
    
    with info_col1:
        st.metric("Python Version", "3.10+")
        st.metric("Streamlit Version", "1.28.1")
        st.metric("DuckDB Version", "0.9.2")
    
    with info_col2:
        st.metric("Pandas Version", "2.1.3")
        st.metric("LangChain Version", "0.0.350")
        st.metric("Plotly Version", "5.17.0")

if __name__ == "__main__":
    main()
