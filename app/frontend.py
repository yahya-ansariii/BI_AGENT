"""
Frontend Application
Streamlit interface for Business Insights Agent
"""

import streamlit as st
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from connectors.excel_connector import ExcelConnector
from connectors.snowflake_connector import SnowflakeConnector
from connectors.ga_connector import GoogleAnalyticsConnector
from connectors.oncore_connector import OncoreConnector
from agent.sql_generator import SQLGenerator
from agent.insights import InsightsGenerator
from agent.llm_agent import LLMAgent

# Page configuration
st.set_page_config(
    page_title="Business Insights Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'schema' not in st.session_state:
    st.session_state.schema = {}
if 'relationships' not in st.session_state:
    st.session_state.relationships = []
if 'excel_connector' not in st.session_state:
    st.session_state.excel_connector = ExcelConnector()
if 'sql_generator' not in st.session_state:
    st.session_state.sql_generator = SQLGenerator()
if 'insights_generator' not in st.session_state:
    st.session_state.insights_generator = InsightsGenerator()

def load_demo_schema():
    """Load demo schema from Excel files"""
    try:
        # Load sales schema
        sales_schema = st.session_state.excel_connector.get_schema('demo_data/sales.xlsx')
        # Load web_traffic schema
        traffic_schema = st.session_state.excel_connector.get_schema('demo_data/web_traffic.xlsx')
        
        # Combine schemas
        demo_schema = {
            'sales': sales_schema.get('Sheet1', []),
            'web_traffic': traffic_schema.get('Sheet1', [])
        }
        
        return demo_schema
    except Exception as e:
        st.error(f"Error loading demo schema: {str(e)}")
        return {}

def save_schema(schema, filepath='schema_store/schema.json'):
    """Save schema to JSON file"""
    try:
        os.makedirs('schema_store', exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(schema, f, indent=2)
        return True
    except Exception as e:
        st.error(f"Error saving schema: {str(e)}")
        return False

def save_relationships(relationships, filepath='schema_store/relationships.json'):
    """Save relationships to JSON file"""
    try:
        os.makedirs('schema_store', exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(relationships, f, indent=2)
        return True
    except Exception as e:
        st.error(f"Error saving relationships: {str(e)}")
        return False

def load_relationships(filepath='schema_store/relationships.json'):
    """Load relationships from JSON file"""
    try:
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                return json.load(f)
        return []
    except Exception as e:
        st.error(f"Error loading relationships: {str(e)}")
        return []

def create_chart(data, chart_type, x_col, y_col=None):
    """Create chart based on data and type"""
    try:
        if chart_type == "bar_chart":
            fig = px.bar(data, x=x_col, y=y_col, title=f"{y_col} by {x_col}")
        elif chart_type == "line_chart":
            fig = px.line(data, x=x_col, y=y_col, title=f"{y_col} over {x_col}")
        elif chart_type == "scatter_plot":
            fig = px.scatter(data, x=x_col, y=y_col, title=f"{y_col} vs {x_col}")
        else:
            fig = px.bar(data, x=x_col, y=y_col, title=f"Data Visualization")
        
        return fig
    except Exception as e:
        st.error(f"Error creating chart: {str(e)}")
        return None

def excel_data_source_tab():
    """Excel data source configuration"""
    st.subheader("📊 Excel Files Configuration")
    
    # File upload
    uploaded_files = st.file_uploader(
        "Upload Excel files",
        type=['xlsx', 'xls'],
        accept_multiple_files=True,
        help="Upload one or more Excel files to analyze"
    )
    
    if uploaded_files:
        st.success(f"Uploaded {len(uploaded_files)} file(s)")
        
        # Process each file
        for i, file in enumerate(uploaded_files):
            with st.expander(f"File {i+1}: {file.name}"):
                try:
                    # Get schema
                    schema = st.session_state.excel_connector.get_schema(file)
                    if schema:
                        st.write("Schema:")
                        for sheet_name, columns in schema.items():
                            st.write(f"**{sheet_name}**:")
                            col_df = pd.DataFrame(columns)
                            st.dataframe(col_df, use_container_width=True)
                    else:
                        st.error("Failed to read schema")
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")
    
    # Demo data option
    if st.button("Use Demo Data"):
        if os.path.exists('demo_data/sales.xlsx') and os.path.exists('demo_data/web_traffic.xlsx'):
            st.success("Demo data available!")
            st.info("Demo files: sales.xlsx, web_traffic.xlsx")
        else:
            st.warning("Demo data not found. Run 'python create_demo_data.py' first.")

def snowflake_data_source_tab():
    """Snowflake data source configuration"""
    st.subheader("❄️ Snowflake Database Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Connection Settings**")
        account = st.text_input("Account", placeholder="your-account.snowflakecomputing.com")
        user = st.text_input("Username", placeholder="your_username")
        password = st.text_input("Password", type="password")
        warehouse = st.text_input("Warehouse", placeholder="COMPUTE_WH")
        database = st.text_input("Database", placeholder="YOUR_DATABASE")
        schema = st.text_input("Schema", value="PUBLIC")
        role = st.text_input("Role (optional)", placeholder="ACCOUNTADMIN")
    
    with col2:
        st.write("**Connection Test**")
        if st.button("Test Connection"):
            if all([account, user, password, warehouse, database]):
                with st.spinner("Testing connection..."):
                    try:
                        snowflake_conn = SnowflakeConnector()
                        if snowflake_conn.connect(account, user, password, warehouse, database, schema, role):
                            st.success("✅ Connection successful!")
                            
                            # Get tables
                            tables = snowflake_conn.get_tables(database, schema)
                            if tables:
                                st.write("**Available Tables:**")
                                for table in tables[:10]:  # Show first 10
                                    st.write(f"• {table['table_name']}")
                            else:
                                st.warning("No tables found")
                        else:
                            st.error("❌ Connection failed")
                    except Exception as e:
                        st.error(f"Connection error: {str(e)}")
            else:
                st.warning("Please fill in all required fields")
    
    # Save connection settings
    if st.button("Save Snowflake Configuration"):
        config = {
            "account": account,
            "user": user,
            "warehouse": warehouse,
            "database": database,
            "schema": schema,
            "role": role
        }
        # Save to session state or file
        st.session_state.snowflake_config = config
        st.success("Configuration saved!")

def ga_data_source_tab():
    """Google Analytics data source configuration"""
    st.subheader("📈 Google Analytics Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**GA4 Settings**")
        property_id = st.text_input("Property ID", placeholder="123456789")
        credentials_path = st.text_input("Credentials JSON Path", placeholder="path/to/credentials.json")
        
        st.write("**Date Range**")
        start_date = st.date_input("Start Date", value=pd.Timestamp.now() - pd.Timedelta(days=30))
        end_date = st.date_input("End Date", value=pd.Timestamp.now())
    
    with col2:
        st.write("**Data Types**")
        data_types = st.multiselect(
            "Select data to fetch:",
            ["Website Traffic", "Page Performance", "Ecommerce Data", "Audience Data"],
            default=["Website Traffic"]
        )
        
        st.write("**Connection Test**")
        if st.button("Test GA Connection"):
            if property_id and credentials_path:
                with st.spinner("Testing connection..."):
                    try:
                        ga_conn = GoogleAnalyticsConnector()
                        if ga_conn.connect(credentials_path, property_id):
                            st.success("✅ Connection successful!")
                        else:
                            st.error("❌ Connection failed")
                    except Exception as e:
                        st.error(f"Connection error: {str(e)}")
            else:
                st.warning("Please provide Property ID and Credentials path")
    
    # Fetch data
    if st.button("Fetch GA Data"):
        if property_id and credentials_path:
            with st.spinner("Fetching data..."):
                try:
                    ga_conn = GoogleAnalyticsConnector()
                    if ga_conn.connect(credentials_path, property_id):
                        start_str = start_date.strftime("%Y-%m-%d")
                        end_str = end_date.strftime("%Y-%m-%d")
                        
                        if "Website Traffic" in data_types:
                            traffic_data = ga_conn.get_website_traffic(start_str, end_str)
                            if not traffic_data.empty:
                                st.success(f"Fetched {len(traffic_data)} traffic records")
                                st.dataframe(traffic_data.head(), use_container_width=True)
                except Exception as e:
                    st.error(f"Error fetching data: {str(e)}")
        else:
            st.warning("Please configure connection first")

def oncore_data_source_tab():
    """Oncore data source configuration"""
    st.subheader("🏥 Oncore System Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Connection Settings**")
        base_url = st.text_input("Base URL", placeholder="https://your-oncore-instance.com")
        api_key = st.text_input("API Key", type="password")
    
    with col2:
        st.write("**Connection Test**")
        if st.button("Test Oncore Connection"):
            if base_url and api_key:
                with st.spinner("Testing connection..."):
                    try:
                        oncore_conn = OncoreConnector()
                        if oncore_conn.connect(base_url, api_key):
                            st.success("✅ Connection successful!")
                            
                            # Get data quality report
                            quality_report = oncore_conn.get_data_quality_report()
                            if quality_report:
                                st.write("**Data Quality Report:**")
                                st.json(quality_report)
                        else:
                            st.error("❌ Connection failed")
                    except Exception as e:
                        st.error(f"Connection error: {str(e)}")
            else:
                st.warning("Please provide Base URL and API Key")
    
    # Data export options
    if st.button("Export Oncore Data"):
        if base_url and api_key:
            with st.spinner("Exporting data..."):
                try:
                    oncore_conn = OncoreConnector()
                    if oncore_conn.connect(base_url, api_key):
                        # Export different data types
                        protocols = oncore_conn.get_protocols(limit=100)
                        if not protocols.empty:
                            st.success(f"Exported {len(protocols)} protocols")
                            st.dataframe(protocols.head(), use_container_width=True)
                except Exception as e:
                    st.error(f"Error exporting data: {str(e)}")
        else:
            st.warning("Please configure connection first")

def multiple_sources_tab():
    """Multiple data sources configuration"""
    st.subheader("🔗 Multiple Data Sources")
    
    st.write("Configure and combine data from multiple sources:")
    
    # Source selection
    selected_sources = st.multiselect(
        "Select data sources to combine:",
        ["Excel Files", "Snowflake", "Google Analytics", "Oncore"],
        default=["Excel Files"]
    )
    
    if selected_sources:
        st.write("**Selected Sources:**")
        for source in selected_sources:
            st.write(f"• {source}")
        
        # Data integration options
        st.write("**Integration Options:**")
        integration_method = st.selectbox(
            "How to combine data:",
            ["Manual Joins", "Automatic Key Detection", "Time-based Alignment"]
        )
        
        if st.button("Configure Integration"):
            st.info(f"Integration method: {integration_method}")
            st.write("This would configure how data from different sources is combined.")
            st.write("For now, each source can be used independently in the 'Ask Questions' tab.")

def main():
    """Main application function"""
    
    st.title("📊 Business Insights Agent")
    st.markdown("---")
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Data Sources", "Schema Approval", "Ask Questions"])
    
    with tab1:
        st.header("🔗 Data Sources Configuration")
        
        # Data source selection
        data_source = st.selectbox(
            "Select Data Source Type:",
            ["Excel Files", "Snowflake Database", "Google Analytics", "Oncore System", "Multiple Sources"]
        )
        
        if data_source == "Excel Files":
            excel_data_source_tab()
        elif data_source == "Snowflake Database":
            snowflake_data_source_tab()
        elif data_source == "Google Analytics":
            ga_data_source_tab()
        elif data_source == "Oncore System":
            oncore_data_source_tab()
        elif data_source == "Multiple Sources":
            multiple_sources_tab()
    
    with tab2:
        st.header("🗂️ Schema Approval")
        
        # Model Configuration Section
        st.subheader("🤖 Model Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            model_path = st.text_input(
                "Custom Model Path",
                value="~/Documents/LLM_Models",
                help="Path where Ollama models will be stored"
            )
            
            if st.button("Set Model Path"):
                if st.session_state.llm_agent.set_model_path(model_path):
                    st.success(f"Model path set to: {st.session_state.llm_agent.get_model_path()}")
                else:
                    st.error("Failed to set model path")
        
        with col2:
            current_path = st.session_state.llm_agent.get_model_path()
            st.info(f"Current model path: {current_path}")
            
            if st.button("Check Available Models"):
                models = st.session_state.llm_agent.list_models()
                if models:
                    st.write("Available models:")
                    for model in models:
                        st.write(f"• {model}")
                else:
                    st.warning("No models found or Ollama not running")
        
        st.markdown("---")
        
        # Load demo schema
        if st.button("Load Demo Schema"):
            with st.spinner("Loading demo schema..."):
                st.session_state.schema = load_demo_schema()
                st.success("Demo schema loaded!")
        
        # Display current schema
        if st.session_state.schema:
            st.subheader("Current Schema")
            
            for table_name, columns in st.session_state.schema.items():
                with st.expander(f"Table: {table_name}"):
                    col_df = pd.DataFrame(columns)
                    st.dataframe(col_df, use_container_width=True)
            
            # Schema relationships section
            st.subheader("Define Relationships")
            st.markdown("Define relationships between tables (e.g., sales.date = web_traffic.date)")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                from_table = st.selectbox("From Table", list(st.session_state.schema.keys()))
            
            with col2:
                from_columns = [col['name'] for col in st.session_state.schema[from_table]]
                from_column = st.selectbox("From Column", from_columns)
            
            with col3:
                to_table = st.selectbox("To Table", list(st.session_state.schema.keys()))
            
            with col4:
                to_columns = [col['name'] for col in st.session_state.schema[to_table]]
                to_column = st.selectbox("To Column", to_columns)
            
            if st.button("Add Relationship"):
                relationship = {
                    "from_table": from_table,
                    "from_column": from_column,
                    "to_table": to_table,
                    "to_column": to_column
                }
                st.session_state.relationships.append(relationship)
                st.success(f"Added relationship: {from_table}.{from_column} = {to_table}.{to_column}")
            
            # Display current relationships
            if st.session_state.relationships:
                st.subheader("Current Relationships")
                for i, rel in enumerate(st.session_state.relationships):
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.write(f"{rel['from_table']}.{rel['from_column']} = {rel['to_table']}.{rel['to_column']}")
                    with col2:
                        if st.button("Remove", key=f"remove_{i}"):
                            st.session_state.relationships.pop(i)
                            st.rerun()
            
            # Save schema and relationships
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Save Schema"):
                    if save_schema(st.session_state.schema):
                        st.success("Schema saved!")
            
            with col2:
                if st.button("Save Relationships"):
                    if save_relationships(st.session_state.relationships):
                        st.success("Relationships saved!")
    
    with tab3:
        st.header("❓ Ask Questions")
        
        # Data source status
        st.subheader("Data Source Status")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            excel_status = "✅ Configured" if st.session_state.schema else "❌ Not configured"
            st.metric("Excel Files", excel_status)
        
        with col2:
            snowflake_status = "✅ Configured" if "snowflake_config" in st.session_state else "❌ Not configured"
            st.metric("Snowflake", snowflake_status)
        
        with col3:
            ga_status = "✅ Configured" if "ga_config" in st.session_state else "❌ Not configured"
            st.metric("Google Analytics", ga_status)
        
        with col4:
            oncore_status = "✅ Configured" if "oncore_config" in st.session_state else "❌ Not configured"
            st.metric("Oncore", oncore_status)
        
        st.markdown("---")
        
        # Data source selection for querying
        st.subheader("Select Data Source")
        query_source = st.selectbox(
            "Choose data source to query:",
            ["Excel Files", "Snowflake", "Google Analytics", "Oncore"],
            help="Select the data source you want to query"
        )
        
        # Check if schema is loaded or data source is configured
        if query_source == "Excel Files" and not st.session_state.schema:
            st.warning("Please load a schema first in the Schema Approval tab.")
            return
        elif query_source in ["Snowflake", "Google Analytics", "Oncore"]:
            if query_source == "Snowflake" and "snowflake_config" not in st.session_state:
                st.warning("Please configure Snowflake connection first in the Data Sources tab.")
                return
            elif query_source == "Google Analytics" and "ga_config" not in st.session_state:
                st.warning("Please configure Google Analytics connection first in the Data Sources tab.")
                return
            elif query_source == "Oncore" and "oncore_config" not in st.session_state:
                st.warning("Please configure Oncore connection first in the Data Sources tab.")
                return
        
        # Load relationships
        st.session_state.relationships = load_relationships()
        
        # Question input
        question = st.text_area(
            "Enter your question:",
            placeholder="e.g., Total sales by region, Compare sales and visits by date",
            height=100
        )
        
        if st.button("Generate Analysis", type="primary"):
            if question:
                with st.spinner("Generating analysis..."):
                    try:
                        # Generate SQL
                        sql = st.session_state.sql_generator.generate_sql(
                            question, 
                            st.session_state.schema, 
                            st.session_state.relationships
                        )
                        
                        if sql:
                            st.subheader("Generated SQL")
                            st.code(sql, language="sql")
                            
                            # Execute SQL based on data source
                            st.subheader("Query Results")
                            
                            result_df = None
                            
                            if query_source == "Excel Files":
                                # Try both demo files
                                for file_path in ['demo_data/sales.xlsx', 'demo_data/web_traffic.xlsx']:
                                    if os.path.exists(file_path):
                                        temp_result = st.session_state.excel_connector.run_query(file_path, sql)
                                        if not temp_result.empty:
                                            result_df = temp_result
                                            break
                            
                            elif query_source == "Snowflake":
                                # Execute SQL on Snowflake
                                try:
                                    snowflake_conn = SnowflakeConnector()
                                    config = st.session_state.snowflake_config
                                    if snowflake_conn.connect(
                                        config["account"], config["user"], config["password"],
                                        config["warehouse"], config["database"], 
                                        config["schema"], config.get("role")
                                    ):
                                        result_df = snowflake_conn.execute_query(sql)
                                except Exception as e:
                                    st.error(f"Snowflake query error: {str(e)}")
                            
                            elif query_source == "Google Analytics":
                                st.info("GA queries are handled through the GA connector's specific methods. Use the Data Sources tab to fetch GA data.")
                                return
                            
                            elif query_source == "Oncore":
                                st.info("Oncore queries are handled through the Oncore connector's specific methods. Use the Data Sources tab to export Oncore data.")
                                return
                            
                            if result_df is not None and not result_df.empty:
                                st.dataframe(result_df, use_container_width=True)
                                
                                # Generate insights
                                st.subheader("Insights")
                                insights = st.session_state.insights_generator.generate_insights(
                                    question, sql, result_df
                                )
                                st.write(insights)
                                
                                # Generate chart
                                st.subheader("Visualization")
                                
                                # Determine chart type
                                date_cols = result_df.select_dtypes(include=['datetime64']).columns
                                categorical_cols = result_df.select_dtypes(include=['object']).columns
                                numeric_cols = result_df.select_dtypes(include=['number']).columns
                                
                                if len(date_cols) > 0 and len(numeric_cols) > 0:
                                    # Line chart for time series
                                    fig = create_chart(result_df, "line_chart", date_cols[0], numeric_cols[0])
                                elif len(categorical_cols) > 0 and len(numeric_cols) > 0:
                                    # Bar chart for categorical data
                                    fig = create_chart(result_df, "bar_chart", categorical_cols[0], numeric_cols[0])
                                elif len(numeric_cols) >= 2:
                                    # Scatter plot for two numeric variables
                                    fig = create_chart(result_df, "scatter_plot", numeric_cols[0], numeric_cols[1])
                                else:
                                    # Default bar chart
                                    if len(result_df.columns) >= 2:
                                        fig = create_chart(result_df, "bar_chart", result_df.columns[0], result_df.columns[1])
                                    else:
                                        st.info("Not enough data for visualization")
                                        fig = None
                                
                                if fig:
                                    st.plotly_chart(fig, use_container_width=True)
                                
                            else:
                                st.warning("No results returned from query")
                        else:
                            st.error("Failed to generate SQL query")
                            
                    except Exception as e:
                        st.error(f"Error generating analysis: {str(e)}")
            else:
                st.warning("Please enter a question first")

if __name__ == "__main__":
    main()
