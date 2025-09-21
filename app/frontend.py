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

def main():
    """Main application function"""
    
    st.title("📊 Business Insights Agent")
    st.markdown("---")
    
    # Create tabs
    tab1, tab2 = st.tabs(["Schema Approval", "Ask Questions"])
    
    with tab1:
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
    
    with tab2:
        st.header("❓ Ask Questions")
        
        # Check if schema is loaded
        if not st.session_state.schema:
            st.warning("Please load a schema first in the Schema Approval tab.")
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
                            
                            # Execute SQL
                            st.subheader("Query Results")
                            
                            # Try both demo files
                            result_df = None
                            for file_path in ['demo_data/sales.xlsx', 'demo_data/web_traffic.xlsx']:
                                if os.path.exists(file_path):
                                    temp_result = st.session_state.excel_connector.run_query(file_path, sql)
                                    if not temp_result.empty:
                                        result_df = temp_result
                                        break
                            
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
