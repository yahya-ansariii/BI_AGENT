"""
Business Insights Agent - Modern Streamlit Application
A local AI-powered business intelligence tool with advanced data analysis capabilities
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
from datetime import datetime
import io
import re
from typing import Dict, List, Optional, Tuple

# PDF and Word document generation
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

try:
    from docx import Document
    from docx.shared import Inches
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

# Optional import for ER diagrams
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

from modules.data_processor import DataProcessor
from modules.visualizer import Visualizer
from modules.llm_agent import LLMAgent
from modules.schema_manager import SchemaManager

# Import Ollama configuration
try:
    from config.ollama_config import ollama_config
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    ollama_config = None

# Page configuration
    st.set_page_config(
        page_title="BI Agent",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

# Modern CSS for minimalist design
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .main {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1a1a1a;
        text-align: center;
        margin-bottom: 0.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .sub-header {
        font-size: 1.1rem;
        color: #6b7280;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #f8fafc;
        padding: 8px;
        border-radius: 12px;
        margin-bottom: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 500;
        color: #6b7280;
        transition: all 0.2s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #ffffff;
        color: #1a1a1a;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Card Styling */
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        text-align: center;
        transition: transform 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 15px -3px rgba(0, 0, 0, 0.1);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1a1a1a;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 0.875rem;
        color: #6b7280;
        font-weight: 500;
    }
    
    /* Status Indicators */
    .status-success {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        font-weight: 500;
        margin: 1rem 0;
    }
    
    .status-warning {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        font-weight: 500;
        margin: 1rem 0;
    }
    
    .status-error {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        font-weight: 500;
        margin: 1rem 0;
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        font-size: 0.875rem;
        transition: all 0.2s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    
    /* File Upload Styling */
    .uploadedFile {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        border: 2px dashed #cbd5e1;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8fafc 0%, #ffffff 100%);
    }
    
    /* Data Table Styling */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Chart Container */
    .chart-container {
        background: white;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    /* ER Diagram Container */
    .er-diagram {
        background: white;
        border-radius: 12px;
        padding: 2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        text-align: center;
    }
    
    /* Loading Spinner */
    .stSpinner > div {
        border-top-color: #667eea;
    }
</style>
""", unsafe_allow_html=True)

def get_available_models():
    """Get list of available Ollama models"""
    if not OLLAMA_AVAILABLE:
        return []
    
    try:
        return ollama_config.get_available_models()
    except Exception as e:
        st.error(f"Error getting available models: {str(e)}")
        return []

def download_model(model_name: str):
    """Download an Ollama model"""
    with st.spinner(f"Downloading {model_name}..."):
        try:
            import subprocess
            result = subprocess.run(
                ["ollama", "pull", model_name],
                capture_output=True,
                text=True,
                timeout=300
            )
            if result.returncode == 0:
                st.success(f"✅ {model_name} downloaded successfully!")
                st.rerun()
            else:
                st.error(f"❌ Failed to download {model_name}: {result.stderr}")
        except subprocess.TimeoutExpired:
            st.error("❌ Download timeout. Large models may take longer.")
        except Exception as e:
            st.error(f"❌ Error downloading model: {str(e)}")

def safe_dataframe_display(df, width='stretch'):
    """Display DataFrame with proper data type handling to avoid Arrow serialization issues"""
    try:
        # Create a copy and convert problematic data types
        display_df = df.copy()
        for col in display_df.columns:
            if display_df[col].dtype == 'object':
                # Convert object columns to string, handling mixed types
                display_df[col] = display_df[col].astype(str)
            elif 'datetime' in str(display_df[col].dtype):
                # Convert datetime columns to string
                display_df[col] = display_df[col].dt.strftime('%Y-%m-%d %H:%M:%S')
            elif 'Timestamp' in str(display_df[col].dtype):
                # Handle pandas Timestamp objects
                display_df[col] = display_df[col].astype(str)
        
        return st.dataframe(display_df, width=width)
    except Exception as e:
        st.error(f"Error displaying data: {str(e)}")
        # Fallback: display as text
        st.text(str(df.head()))

def validate_table_name(name: str) -> Tuple[bool, str]:
    """Validate table name for SQL compatibility"""
    if not name or not name.strip():
        return False, "Table name cannot be empty"
    
    name = name.strip()
    
    # Check for SQL reserved words
    sql_reserved = {
        'select', 'from', 'where', 'insert', 'update', 'delete', 'create', 'drop', 'alter',
        'table', 'database', 'index', 'view', 'procedure', 'function', 'trigger', 'constraint',
        'primary', 'foreign', 'key', 'unique', 'check', 'default', 'null', 'not', 'and', 'or',
        'as', 'in', 'like', 'between', 'is', 'exists', 'all', 'any', 'some', 'union', 'intersect',
        'except', 'order', 'group', 'having', 'limit', 'offset', 'distinct', 'top', 'case',
        'when', 'then', 'else', 'end', 'if', 'while', 'for', 'do', 'begin', 'end', 'return',
        'break', 'continue', 'goto', 'declare', 'set', 'exec', 'execute', 'sp_', 'xp_'
    }
    
    if name.lower() in sql_reserved:
        return False, f"'{name}' is a SQL reserved word"
    
    # Check for valid characters (alphanumeric and underscore only)
    if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', name):
        return False, "Table name must start with letter or underscore and contain only letters, numbers, and underscores"
    
    # Check length
    if len(name) > 128:
        return False, "Table name must be 128 characters or less"
    
    return True, "Valid table name"

def check_duplicate_table_name(name: str, existing_tables: Dict) -> Tuple[bool, str]:
    """Check if table name already exists"""
    if name in existing_tables:
        return False, f"Table '{name}' already exists"
    return True, "Table name is available"

def validate_data_entry(df, edited_data):
    """Validate data entry in Excel-like editor"""
    validation_errors = []
    
    for col in df.columns:
        col_type = df[col].dtype
        
        for idx, value in enumerate(edited_data[col]):
            if pd.isna(value) or value == '' or value is None:
                continue  # Skip empty values
                
            # Validate based on column type
            if 'int' in str(col_type) or 'float' in str(col_type):
                try:
                    float(value)
                except (ValueError, TypeError):
                    validation_errors.append(f"Row {idx+1}, Column '{col}': '{value}' is not a valid number")
            
            elif 'datetime' in str(col_type):
                try:
                    pd.to_datetime(value)
                except (ValueError, TypeError):
                    validation_errors.append(f"Row {idx+1}, Column '{col}': '{value}' is not a valid date")
            
            elif col_type == 'object':
                # For text columns, check for reasonable length
                if len(str(value)) > 1000:
                    validation_errors.append(f"Row {idx+1}, Column '{col}': Text too long (max 1000 characters)")
    
    return validation_errors

def create_table_on_enter():
    """Handle Enter key press to create table when all columns are filled"""
    # This function will be called when the last column name is entered
    st.session_state.auto_create_table = True
    st.rerun()

def generate_pdf_report(insight_data: Dict, filename: str) -> bytes:
    """Generate PDF report using ReportLab"""
    if not REPORTLAB_AVAILABLE:
        raise ImportError("ReportLab not available. Install with: pip install reportlab")
    
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
    
    # Get styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=TA_CENTER,
        textColor=colors.HexColor('#667eea')
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        spaceAfter=12,
        textColor=colors.HexColor('#333333')
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=11,
        spaceAfter=6
    )
    
    code_style = ParagraphStyle(
        'CustomCode',
        parent=styles['Code'],
        fontSize=9,
        leftIndent=20,
        rightIndent=20,
        spaceAfter=12,
        backgroundColor=colors.HexColor('#f8f9fa')
    )
    
    # Build content
    story = []
    
    # Title
    story.append(Paragraph("🤖 AI Analysis Report", title_style))
    story.append(Spacer(1, 12))
    
    # Metadata
    story.append(Paragraph(f"<b>Generated:</b> {insight_data['timestamp']}", normal_style))
    story.append(Paragraph(f"<b>Data Shape:</b> {insight_data['data_shape'][0]} rows × {insight_data['data_shape'][1]} columns", normal_style))
    story.append(Spacer(1, 20))
    
    # Question
    story.append(Paragraph("📝 Question", heading_style))
    story.append(Paragraph(insight_data['question'], normal_style))
    story.append(Spacer(1, 20))
    
    # SQL Query
    story.append(Paragraph("🔍 SQL Query", heading_style))
    story.append(Paragraph(insight_data['sql_query'], code_style))
    story.append(Spacer(1, 20))
    
    # AI Analysis
    story.append(Paragraph("🧠 AI Analysis", heading_style))
    
    # Split analysis into paragraphs
    analysis_paragraphs = insight_data['analysis'].split('\n\n')
    for para in analysis_paragraphs:
        if para.strip():
            story.append(Paragraph(para.strip(), normal_style))
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

def generate_word_report(insight_data: Dict) -> bytes:
    """Generate Word document report"""
    if not DOCX_AVAILABLE:
        raise ImportError("python-docx not available. Install with: pip install python-docx")
    
    doc = Document()
    
    # Title
    title = doc.add_heading('🤖 AI Analysis Report', 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    # Metadata
    doc.add_paragraph(f"Generated: {insight_data['timestamp']}")
    doc.add_paragraph(f"Data Shape: {insight_data['data_shape'][0]} rows × {insight_data['data_shape'][1]} columns")
    doc.add_paragraph("")
    
    # Question
    doc.add_heading('📝 Question', level=1)
    doc.add_paragraph(insight_data['question'])
    
    # SQL Query
    doc.add_heading('🔍 SQL Query', level=1)
    sql_para = doc.add_paragraph(insight_data['sql_query'])
    sql_para.style = 'Code'
    
    # AI Analysis
    doc.add_heading('🧠 AI Analysis', level=1)
    
    # Split analysis into paragraphs
    analysis_paragraphs = insight_data['analysis'].split('\n\n')
    for para in analysis_paragraphs:
        if para.strip():
            doc.add_paragraph(para.strip())
    
    # Save to bytes
    buffer = io.BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer.getvalue()

def initialize_session_state():
    """Initialize all session state variables"""
    if 'data_processor' not in st.session_state:
        st.session_state.data_processor = DataProcessor()
    if 'visualizer' not in st.session_state:
        st.session_state.visualizer = Visualizer()
    if 'llm_agent' not in st.session_state:
        st.session_state.llm_agent = LLMAgent()
    if 'schema_manager' not in st.session_state:
        st.session_state.schema_manager = SchemaManager()
    if 'loaded_tables' not in st.session_state:
        st.session_state.loaded_tables = {}
    if 'selected_columns' not in st.session_state:
        st.session_state.selected_columns = {}
    if 'table_relationships' not in st.session_state:
        st.session_state.table_relationships = []
    if 'ai_analysis_results' not in st.session_state:
        st.session_state.ai_analysis_results = {}

def load_multi_sheet_excel(file_upload):
    """Load all sheets from Excel file"""
    try:
        # Read the file into memory first
        file_content = file_upload.read()
        
        # Create ExcelFile object from bytes
        excel_file = pd.ExcelFile(io.BytesIO(file_content))
        sheet_names = excel_file.sheet_names
        
        loaded_tables = {}
        for sheet_name in sheet_names:
            try:
                # Read each sheet from the ExcelFile object
                df = excel_file.parse(sheet_name)
                if not df.empty:
                    loaded_tables[sheet_name] = df
            except Exception as e:
                st.warning(f"Could not load sheet '{sheet_name}': {str(e)}")
        
        return loaded_tables
    except Exception as e:
        st.error(f"Error loading Excel file: {str(e)}")
        return {}

def create_er_diagram(tables, relationships):
    """Create a beautiful Entity Relationship diagram using NetworkX and Matplotlib"""
    if not NETWORKX_AVAILABLE:
        st.warning("NetworkX not available. Install with: pip install networkx")
        return None
    
    try:
        G = nx.DiGraph()
        
        # Add nodes for tables with metadata
        for table_name, df in tables.items():
            G.add_node(table_name, node_type='table', columns=df.columns.tolist(), rows=df.shape[0])
        
        # Add edges for relationships
        for rel in relationships:
            if rel['source_table'] in tables and rel['target_table'] in tables:
                G.add_edge(rel['source_table'], rel['target_table'], 
                          label=f"{rel['source_column']} → {rel['target_column']}",
                          rel_type=rel.get('relationship_type', 'One-to-Many'))
        
        # Calculate optimal figure size based on number of tables
        num_tables = len(tables)
        if num_tables <= 3:
            figsize = (12, 8)
        elif num_tables <= 6:
            figsize = (16, 10)
        else:
            figsize = (20, 12)
        
        # Create the plot with responsive sizing
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor('white')
        ax.set_facecolor('#fafafa')
        
        # Create layout with better spacing
        pos = nx.spring_layout(G, k=3, iterations=100, seed=42)
        
        # Title
        ax.set_title("📊 Entity Relationship Diagram", fontsize=18, fontweight='bold', 
                    color='#2c3e50', pad=20)
        
        # Draw nodes with beautiful styling
        node_colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
        for i, (node, data) in enumerate(G.nodes(data=True)):
            if data['node_type'] == 'table':
                color = node_colors[i % len(node_colors)]
                nx.draw_networkx_nodes(G, pos, 
                                      nodelist=[node],
                                      node_color=color, 
                                      node_size=3000, 
                                      alpha=0.9,
                                      ax=ax)
                
                # Add table info as text
                x, y = pos[node]
                ax.text(x, y, f"{node}\n({data['rows']} rows)\n{len(data['columns'])} cols", 
                       ha='center', va='center', fontsize=9, fontweight='bold', 
                       color='white', bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.8))
        
        # Draw edges with beautiful styling
        edge_colors = ['#7f8c8d' for _ in G.edges()]
        nx.draw_networkx_edges(G, pos, 
                              edge_color=edge_colors, 
                              arrows=True, 
                              arrowsize=20,
                              alpha=0.7,
                              width=2,
                              arrowstyle='->',
                              ax=ax)
        
        # Draw edge labels with better positioning
        edge_labels = nx.get_edge_attributes(G, 'label')
        for edge, label in edge_labels.items():
            x1, y1 = pos[edge[0]]
            x2, y2 = pos[edge[1]]
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mid_x, mid_y, label, ha='center', va='center', 
                   fontsize=8, fontweight='bold', color='#2c3e50',
                   bbox=dict(boxstyle="round,pad=0.1", facecolor='white', alpha=0.8, edgecolor='#bdc3c7'))
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#3498db', 
                      markersize=12, label='Tables'),
            plt.Line2D([0], [0], color='#7f8c8d', linewidth=2, label='Relationships')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        # Remove axes and add subtle grid
        ax.axis('off')
        ax.grid(True, alpha=0.1)
        
        # Ensure the diagram fits properly
        plt.tight_layout(pad=1.0)
        return fig
        
    except Exception as e:
        st.error(f"Error creating ER diagram: {str(e)}")
        return None

def main():
    """Main application function"""
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">📊 BI Agent</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Data Analysis</p>', unsafe_allow_html=True)
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📁 Data Load", 
        "🔍 Data Explorer", 
        "🔗 Relationship Builder", 
        "🤖 AI Analysis", 
        "💻 Custom SQL", 
        "📄 Insights", 
        "⚙️ Settings"
    ])
    
    with tab1:
        data_load_tab()
    
    with tab2:
        data_explorer_tab()
    
    with tab3:
        relationship_builder_tab()
    
    with tab4:
        ai_analysis_tab()
    
    with tab5:
        custom_sql_tab()
    
    with tab6:
        insights_tab()
    
    with tab7:
        settings_tab()

def data_load_tab():
    """Data loading tab with AI model initialization first"""
    st.header("📁 Data Load & AI Setup")
    
    # AI Model Configuration - REQUIRED FIRST STEP
    st.subheader("🤖 Step 1: AI Model Setup (Required)")
    
    # Check if model is already initialized
    model_initialized = hasattr(st.session_state, 'llm_agent') and st.session_state.llm_agent.is_initialized()
    
    if model_initialized:
        st.success(f"✅ AI Model Ready: {st.session_state.llm_agent.model_name}")
        st.markdown("---")
    else:
        st.warning("⚠️ **AI Model Required**: Please initialize an AI model before loading data.")
        
        if OLLAMA_AVAILABLE:
            available_models = get_available_models()
            
            if available_models:
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    model_name = st.selectbox(
                        "Select AI Model:",
                        available_models,
                        help="Choose from your available Ollama models",
                        key="data_load_model_selector"
                    )
                
                with col2:
                    if st.button("🔄 Initialize Model", key="data_load_init_model_btn", type="primary"):
                        with st.spinner("Initializing AI model..."):
                            try:
                                success = st.session_state.llm_agent.initialize_model(model_name)
                                if success:
                                    st.success(f"✅ {model_name} initialized!")
                                    st.rerun()
                                else:
                                    st.error(f"❌ Failed to initialize {model_name}")
                            except Exception as e:
                                st.error(f"❌ Error: {str(e)}")
            else:
                st.warning("⚠️ No AI models found. Please download a model to continue.")
                
                # Model download section with popular models
                st.markdown("### 📥 Download AI Model")
                
                # Popular models section
                st.markdown("**Popular Models:**")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("📥 Download Llama2", key="download_llama2_btn"):
                        download_model("llama2")
                
                with col2:
                    if st.button("📥 Download Mistral", key="download_mistral_btn"):
                        download_model("mistral")
                
                with col3:
                    if st.button("📥 Download CodeLlama", key="download_codellama_btn"):
                        download_model("codellama")
                
                # Custom model download
                st.markdown("**Or download a custom model:**")
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    model_name = st.text_input(
                        "Model Name:",
                        placeholder="e.g., llama2:7b, mistral:7b, codellama:7b",
                        help="Enter the name of the model you want to download",
                        key="download_model_name"
                    )
                
                with col2:
                    if st.button("📥 Download", key="download_custom_model_btn"):
                        if model_name:
                            download_model(model_name)
                        else:
                            st.warning("⚠️ Please enter a model name.")
                
                # Help section
                st.markdown("---")
                st.markdown("### 💡 Need Help?")
                st.markdown("""
                **To download models manually:**
                1. Open terminal/command prompt
                2. Run: `ollama pull llama2`
                3. Refresh this page
                
                **Popular model sizes:**
                - `llama2` (3.8GB) - Good for general use
                - `llama2:7b` (3.8GB) - Smaller, faster
                - `mistral` (4.1GB) - Great for analysis
                - `codellama` (3.8GB) - Good for SQL generation
                """)
        else:
            st.error("❌ Ollama not available. Please install Ollama first.")
            st.markdown("""
            **Installation Guide:**
            1. Visit: https://ollama.ai/download
            2. Download and install Ollama
            3. Restart this application
            """)
        
        # Disable data loading if model not initialized
        st.markdown("---")
        st.markdown('<div class="status-warning">⚠️ <strong>Data loading is disabled</strong> until AI model is initialized.</div>', unsafe_allow_html=True)
        return  # Exit early if model not initialized
    
    # Data Loading Section - Only shown if model is initialized
    st.subheader("📊 Step 2: Load Your Data")
    
    # File upload section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📤 Upload Data Files")
        uploaded_file = st.file_uploader(
            "Choose a data file",
            type=['xlsx', 'xls', 'csv', 'xlsm', 'xlsb', 'odf', 'ods', 'odt'],
            help="Upload your business data in Excel, CSV, or other supported formats",
            key="main_upload"
        )
        
        if uploaded_file is not None:
            file_extension = uploaded_file.name.lower().split('.')[-1]
            
            if file_extension in ['xlsx', 'xls', 'xlsm', 'xlsb']:
                # Multi-sheet Excel file
                with st.spinner("Loading all sheets..."):
                    loaded_tables = load_multi_sheet_excel(uploaded_file)
                    if loaded_tables:
                        st.session_state.loaded_tables = loaded_tables
                        st.success(f"✅ Loaded {len(loaded_tables)} sheets successfully!")
                        
                        # Show loaded sheets
                        for sheet_name, df in loaded_tables.items():
                            st.write(f"**{sheet_name}**: {df.shape[0]} rows × {df.shape[1]} columns")
            else:
                # Single file (CSV, etc.)
                try:
                    with st.spinner("Processing file..."):
                        st.session_state.data_processor.load_excel_data(uploaded_file)
                    st.success("✅ Data loaded successfully!")
                except Exception as e:
                    st.error(f"❌ Error loading data: {str(e)}")
    
    with col2:
        st.subheader("📊 Sample Data")
        st.markdown("Try the application with sample business data")
        
        if st.button("📊 Load Sample Data", key="sample_data"):
            with st.spinner("Loading sample data..."):
                if st.session_state.data_processor.load_sample_data():
                    # Get the loaded tables from the data processor
                    loaded_tables = st.session_state.data_processor.get_loaded_tables()
                    if loaded_tables:
                        st.session_state.loaded_tables = loaded_tables
                        st.success(f"✅ Sample data loaded! {len(loaded_tables)} tables available.")
                    else:
                        st.success("✅ Sample data loaded!")
                else:
                    st.error("❌ Failed to load sample data")
    
    # PowerBI-style table creation
    st.markdown("---")
    st.subheader("🛠️ Create Custom Table (PowerBI Style)")
    
    with st.expander("➕ Enter Data", expanded=True):
        # Single row interface like PowerBI
        col1, col2 = st.columns([2, 1])
        
        with col1:
            table_name = st.text_input(
                "Table Name:",
                placeholder="e.g., sales_data, customer_info",
                help="Enter a valid SQL table name",
                key="powerbi_table_name"
            )
        
        with col2:
            if st.button("🆕 Create Table", key="powerbi_create_btn", type="primary"):
                if table_name:
                    # Validate table name
                    is_valid, message = validate_table_name(table_name)
                    if not is_valid:
                        st.error(f"❌ {message}")
                    else:
                        # Check for duplicates
                        all_tables = {}
                        if hasattr(st.session_state, 'loaded_tables'):
                            all_tables.update(st.session_state.loaded_tables)
                        if hasattr(st.session_state, 'data_processor') and st.session_state.data_processor.has_data():
                            all_tables['main'] = st.session_state.data_processor.get_data()
                        
                        is_unique, dup_message = check_duplicate_table_name(table_name, all_tables)
                        if not is_unique:
                            st.error(f"❌ {dup_message}")
                        else:
                            # Create empty table with default columns
                            data = {
                                'Column1': pd.Series(dtype='object'),
                                'Column2': pd.Series(dtype='object'),
                                'Column3': pd.Series(dtype='object')
                            }
                            
                            new_df = pd.DataFrame(data)
                            
                            # Initialize loaded_tables if not exists
                            if 'loaded_tables' not in st.session_state:
                                st.session_state.loaded_tables = {}
                            
                            st.session_state.loaded_tables[table_name] = new_df
                            st.session_state.current_table = table_name
                            st.session_state[f"editing_table_{table_name}"] = True
                            
                            st.success(f"✅ Table '{table_name}' created! You can now edit the data directly.")
                            st.rerun()
                else:
                    st.warning("⚠️ Please enter a table name.")
        
        # Show instructions
        st.info("💡 **PowerBI Style**: Create a table with default columns, then edit the data directly in the table below.")
    
    # Show loaded data summary
    if st.session_state.loaded_tables or st.session_state.data_processor.has_data():
        st.markdown("---")
        st.subheader("📋 Table Management")
        
        # Display all tables
        all_tables = {}
        if hasattr(st.session_state, 'loaded_tables'):
            all_tables.update(st.session_state.loaded_tables)
        if hasattr(st.session_state, 'data_processor') and st.session_state.data_processor.has_data():
            all_tables['main'] = st.session_state.data_processor.get_data()
        
        if all_tables:
            for table_name, df in all_tables.items():
                with st.expander(f"📊 {table_name} ({df.shape[0]} rows × {df.shape[1]} columns)", expanded=False):
                    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
                    
                    with col1:
                        st.markdown(f"**Columns:** {', '.join(df.columns.tolist())}")
                    
                    with col2:
                        if st.button(f"✏️ Edit Data", key=f"edit_data_{table_name}"):
                            st.session_state[f"editing_table_{table_name}"] = True
                            st.rerun()
                    
                    with col3:
                        if st.button(f"🏷️ Rename", key=f"rename_{table_name}"):
                            st.session_state[f"renaming_{table_name}"] = True
                            st.rerun()
                    
                    with col4:
                        if st.button(f"🗑️ Delete", key=f"delete_{table_name}"):
                            if table_name in st.session_state.loaded_tables:
                                del st.session_state.loaded_tables[table_name]
                            elif table_name == 'main':
                                st.session_state.data_processor.clear_data()
                            st.rerun()
                    
                    # Rename functionality
                    if st.session_state.get(f"renaming_{table_name}", False):
                        new_name = st.text_input(f"New name for '{table_name}':", value=table_name)
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("✅ Confirm", key=f"confirm_rename_{table_name}"):
                                if new_name and new_name != table_name:
                                    is_valid, message = validate_table_name(new_name)
                                    if is_valid:
                                        is_unique, dup_message = check_duplicate_table_name(new_name, all_tables)
                                        if is_unique:
                                            # Rename the table
                                            if table_name in st.session_state.loaded_tables:
                                                st.session_state.loaded_tables[new_name] = st.session_state.loaded_tables.pop(table_name)
                                            elif table_name == 'main':
                                                # For main table, we need to handle differently
                                                st.session_state.loaded_tables[new_name] = st.session_state.data_processor.get_data()
                                                st.session_state.data_processor.clear_data()
                                            
                                            st.session_state.current_table = new_name
                                            st.success(f"✅ Table renamed to '{new_name}'")
                                            st.rerun()
                                        else:
                                            st.error(f"❌ {dup_message}")
                                    else:
                                        st.error(f"❌ {message}")
                                else:
                                    st.error("❌ New name cannot be empty or same as current name")
                        with col2:
                            if st.button("❌ Cancel", key=f"cancel_rename_{table_name}"):
                                st.session_state[f"renaming_{table_name}"] = False
                                st.rerun()
                    
                    # Data preview
                    st.markdown("**Data Preview:**")
                    safe_dataframe_display(df.head(10), width='stretch')
    
    # Excel-like data entry interface for all tables
    # Get all available tables (including sample data and Excel files)
    all_available_tables = {}
    if st.session_state.loaded_tables:
        all_available_tables.update(st.session_state.loaded_tables)
    if st.session_state.data_processor.has_data():
        all_available_tables['main'] = st.session_state.data_processor.get_data()
    
    if all_available_tables:
        for table_name, df in all_available_tables.items():
            if st.session_state.get(f"editing_table_{table_name}", False):
                st.markdown("---")
                st.subheader(f"📊 Excel-like Data Editor: {table_name}")
                
                # Use Streamlit's data editor for true Excel-like experience
                st.markdown("**Edit data directly in the table below (like Excel):**")
                
                # Create a copy for editing
                edited_df = df.copy()
                
                # Add empty rows for new data entry
                empty_rows = pd.DataFrame(index=range(5), columns=df.columns)
                empty_rows = empty_rows.fillna('')  # Fill with empty strings for editing
                edited_df = pd.concat([edited_df, empty_rows], ignore_index=True)
                
                # Use Streamlit's data editor
                edited_data = st.data_editor(
                    edited_df,
                    num_rows="dynamic",
                    width='stretch',
                    key=f"data_editor_{table_name}",
                    column_config={
                        col: st.column_config.TextColumn(
                            col,
                            help=f"Edit {col} values directly",
                            width="medium"
                        ) for col in df.columns
                    }
                )
                
                # Action buttons
                col1, col2, col3 = st.columns([1, 1, 1])
                
                with col1:
                    if st.button("💾 Save Changes", key=f"save_data_{table_name}", type="primary"):
                        # Validate data before saving
                        validation_errors = validate_data_entry(df, edited_data)
                        
                        if validation_errors:
                            st.error("❌ Data validation failed:")
                            for error in validation_errors:
                                st.error(f"• {error}")
                        else:
                            # Remove empty rows and save
                            # Filter out rows where all values are empty
                            non_empty_rows = edited_data[~(edited_data == '').all(axis=1)]
                            if not non_empty_rows.empty:
                                # Convert data types properly
                                for col in non_empty_rows.columns:
                                    original_type = df[col].dtype
                                    if 'int' in str(original_type) or 'float' in str(original_type):
                                        non_empty_rows[col] = pd.to_numeric(non_empty_rows[col], errors='coerce')
                                    elif 'datetime' in str(original_type):
                                        non_empty_rows[col] = pd.to_datetime(non_empty_rows[col], errors='coerce')
                                
                                # Save to appropriate location
                                if table_name in st.session_state.loaded_tables:
                                    st.session_state.loaded_tables[table_name] = non_empty_rows
                                elif table_name == 'main':
                                    st.session_state.data_processor.data = non_empty_rows
                                    st.session_state.data_processor._register_data_in_duckdb()
                                
                                st.success("✅ Data saved successfully!")
                                st.rerun()
                            else:
                                st.warning("⚠️ No data to save.")
                
                with col2:
                    if st.button("🔄 Reset", key=f"reset_data_{table_name}"):
                        st.rerun()
                
                with col3:
                    if st.button("✅ Finish Editing", key=f"finish_edit_{table_name}"):
                        st.session_state[f"editing_table_{table_name}"] = False
                        st.rerun()

def data_explorer_tab():
    """Data exploration tab with column selection"""
    st.header("🔍 Explore Your Data")
    
    if not st.session_state.loaded_tables and not st.session_state.data_processor.has_data():
        st.markdown('<div class="status-warning">⚠️ No data loaded. Please go to the Data Load tab to upload your data.</div>', unsafe_allow_html=True)
        return
    
    # Table selection - include both loaded tables and main data
    all_available_tables = {}
    if st.session_state.loaded_tables:
        all_available_tables.update(st.session_state.loaded_tables)
    if st.session_state.data_processor.has_data():
        all_available_tables['main'] = st.session_state.data_processor.get_data()
    
    if all_available_tables:
        selected_table = st.selectbox(
            "Select Table to Explore:",
            list(all_available_tables.keys()),
            key="table_selector"
        )
        
        if selected_table:
            current_data = all_available_tables[selected_table]
            st.session_state.current_table = selected_table
    else:
        current_data = None
    
    if current_data is not None and not current_data.empty:
        # Column selection
        st.subheader("📋 Select Columns for Analysis")
        
        all_columns = list(current_data.columns)
        selected_columns = st.multiselect(
            "Choose columns to analyze:",
            all_columns,
            default=all_columns[:5] if len(all_columns) > 5 else all_columns,
            key=f"column_selector_{st.session_state.current_table}"
        )
        
        if selected_columns:
            st.session_state.selected_columns[st.session_state.current_table] = selected_columns
            filtered_data = current_data[selected_columns]
            
            # Data overview
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Selected Columns", len(selected_columns))
            with col2:
                st.metric("Total Rows", len(filtered_data))
            with col3:
                st.metric("Memory Usage", f"{filtered_data.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
            with col4:
                missing_pct = (filtered_data.isnull().sum().sum() / (len(filtered_data) * len(selected_columns))) * 100
                st.metric("Missing Data", f"{missing_pct:.1f}%")
            
            # Data preview
            st.subheader("👀 Data Preview")
            show_rows = st.slider("Rows to display", 5, 100, 10)
            safe_dataframe_display(filtered_data.head(show_rows), width='stretch')
            
            # Data analysis
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Data Types")
                type_counts = filtered_data.dtypes.value_counts()
                type_names = [str(dtype) for dtype in type_counts.index]
                fig = px.pie(values=type_counts.values, names=type_names, title="Column Types")
                st.plotly_chart(fig, width='stretch')
            
            with col2:
                st.subheader("❓ Missing Values")
                missing_data = filtered_data.isnull().sum()
                if missing_data.sum() > 0:
                    fig = px.bar(x=missing_data.index, y=missing_data.values, title="Missing Values per Column")
                    st.plotly_chart(fig, width='stretch')
                else:
                    st.markdown('<div class="status-success">✅ No missing values found!</div>', unsafe_allow_html=True)
            
            # Statistical summary
            st.subheader("📈 Statistical Summary")
            safe_dataframe_display(filtered_data.describe(), width='stretch')

def relationship_builder_tab():
    """Relationship builder tab with ER diagram"""
    st.header("🔗 Build Table Relationships")
    
    if not st.session_state.loaded_tables and not st.session_state.data_processor.has_data():
        st.markdown('<div class="status-warning">⚠️ No data loaded. Please load data first.</div>', unsafe_allow_html=True)
        return
    
    # Get all available tables
    all_tables = {}
    if st.session_state.loaded_tables:
        all_tables.update(st.session_state.loaded_tables)
    if st.session_state.data_processor.has_data():
        all_tables["main"] = st.session_state.data_processor.get_data()
    
    if not all_tables:
        st.markdown('<div class="status-warning">⚠️ No tables available for relationship building.</div>', unsafe_allow_html=True)
        return
    
    # Relationship builder interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🔗 Create Relationship")
        
        # AI-powered relationship detection with progress and cancel
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if st.button("🤖 Auto-Detect Relationships", key="ai_detect_relationships_btn", type="primary"):
                if hasattr(st.session_state, 'llm_agent') and st.session_state.llm_agent.is_initialized():
                    st.session_state[f"ai_detecting_relationships"] = True
                    st.rerun()
                else:
                    st.warning("⚠️ Please initialize an AI model in the Data Load tab first.")
        
        with col2:
            if st.session_state.get(f"ai_detecting_relationships", False):
                if st.button("❌ Cancel Detection", key="cancel_ai_detection_btn"):
                    st.session_state[f"ai_detecting_relationships"] = False
                    st.rerun()
        
        # Show progress and run detection
        if st.session_state.get(f"ai_detecting_relationships", False):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                status_text.text("🔍 Analyzing table structures...")
                progress_bar.progress(20)
                
                status_text.text("🤖 AI is analyzing relationships...")
                progress_bar.progress(50)
                
                # Print to terminal for debugging
                print("Starting AI relationship detection...")
                detected_relationships = st.session_state.llm_agent.detect_relationships(all_tables)
                print(f"AI detection completed. Found {len(detected_relationships) if detected_relationships else 0} relationships.")
                
                progress_bar.progress(80)
                status_text.text("📊 Processing results...")
                
                if detected_relationships:
                    new_relationships = 0
                    for rel in detected_relationships:
                        # Add to existing relationships if not already present
                        rel_exists = any(
                            r['source_table'] == rel['source_table'] and 
                            r['target_table'] == rel['target_table'] and
                            r['source_column'] == rel['source_column'] and
                            r['target_column'] == rel['target_column']
                            for r in st.session_state.table_relationships
                        )
                        if not rel_exists:
                            st.session_state.table_relationships.append({
                                "source_table": rel['source_table'],
                                "target_table": rel['target_table'],
                                "source_column": rel['source_column'],
                                "target_column": rel['target_column'],
                                "type": rel.get('relationship_type', 'One-to-Many'),
                                "confidence": rel.get('confidence', 'Medium')
                            })
                            new_relationships += 1
                    
                    progress_bar.progress(100)
                    status_text.text(f"✅ AI detected {new_relationships} new relationships!")
                    st.success(f"✅ AI detected {new_relationships} new relationships!")
                    st.session_state[f"ai_detecting_relationships"] = False
                    st.rerun()
                else:
                    progress_bar.progress(100)
                    status_text.text("🤖 No relationships detected.")
                    st.warning("🤖 AI couldn't detect any obvious relationships.")
                    st.session_state[f"ai_detecting_relationships"] = False
                    
            except Exception as e:
                progress_bar.progress(100)
                status_text.text("❌ Detection failed.")
                st.error(f"❌ Error in AI relationship detection: {str(e)}")
                print(f"AI relationship detection error: {str(e)}")
                st.session_state[f"ai_detecting_relationships"] = False
        
        st.markdown("---")
        
        source_table = st.selectbox("Source Table:", list(all_tables.keys()))
        target_table = st.selectbox("Target Table:", list(all_tables.keys()))
        
        if source_table and target_table and source_table != target_table:
            source_columns = list(all_tables[source_table].columns)
            target_columns = list(all_tables[target_table].columns)
            
            col1a, col1b = st.columns(2)
            with col1a:
                source_column = st.selectbox("Source Column:", source_columns, key="source_col")
            with col1b:
                target_column = st.selectbox("Target Column:", target_columns, key="target_col")
            
            relationship_type = st.selectbox("Relationship Type:", 
                                          ["One-to-One", "One-to-Many", "Many-to-One", "Many-to-Many"])
            
            if st.button("➕ Add Relationship", key="add_relationship_btn"):
                new_relationship = {
                    "source_table": source_table,
                    "target_table": target_table,
                    "source_column": source_column,
                    "target_column": target_column,
                    "type": relationship_type
                }
                st.session_state.table_relationships.append(new_relationship)
                st.success(f"✅ Relationship added: {source_table}.{source_column} → {target_table}.{target_column}")
    
    with col2:
        st.subheader("📊 Current Relationships")
        
        if st.session_state.table_relationships:
            for i, rel in enumerate(st.session_state.table_relationships):
                with st.expander(f"🔗 {rel['source_table']}.{rel['source_column']} → {rel['target_table']}.{rel['target_column']} ({rel['type']})"):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"**Type:** {rel['type']}")
                        st.write(f"**Source:** {rel['source_table']}.{rel['source_column']}")
                        st.write(f"**Target:** {rel['target_table']}.{rel['target_column']}")
                    with col2:
                        if st.button("🗑️", key=f"delete_{i}"):
                            st.session_state.table_relationships.pop(i)
                            st.rerun()
        else:
            st.info("No relationships defined yet.")
    
    # ER Diagram
    if st.session_state.table_relationships:
        st.subheader("📈 Entity Relationship Diagram")
        
        if st.button("🔄 Refresh Diagram", key="refresh_diagram_btn"):
            st.rerun()
        
        er_diagram = create_er_diagram(all_tables, st.session_state.table_relationships)
        if er_diagram:
            st.pyplot(er_diagram)

def ai_analysis_tab():
    """AI analysis tab with improved workflow and persistent SQL queries"""
    st.header("🤖 AI-Powered Analysis")
    
    if not st.session_state.loaded_tables and not st.session_state.data_processor.has_data():
        st.markdown('<div class="status-warning">⚠️ No data loaded. Please load data first.</div>', unsafe_allow_html=True)
        return
    
    # Check if model is initialized
    if not hasattr(st.session_state, 'llm_agent') or not st.session_state.llm_agent.is_initialized():
        st.warning("⚠️ Please initialize an AI model in the Data Load tab first.")
        return
    
    # AI Workflow: Combined Question Input and SQL Generation
    st.subheader("💬 Ask Your Question & Generate SQL")
    
    col1, col2 = st.columns([4, 1])
    
    with col1:
        user_query = st.text_area(
            "Enter your analysis question:",
            placeholder="e.g., What are the top 5 products by sales? Show me customer trends over time.",
            height=100,
            key="analysis_query"
        )
    
    with col2:
        if st.button("🔍 Generate SQL", type="primary", key="generate_sql_btn"):
            if user_query:
                with st.spinner("AI is analyzing your question and generating SQL..."):
                    try:
                        # Get all available tables
                        all_tables = {}
                        if st.session_state.loaded_tables:
                            all_tables.update(st.session_state.loaded_tables)
                        if st.session_state.data_processor.has_data():
                            all_tables['main'] = st.session_state.data_processor.get_data()
                        
                        # Generate SQL query with all table information
                        sql_query = st.session_state.llm_agent.generate_sql_query_with_tables(user_query, all_tables)
                        
                        # Store SQL query
                        st.session_state.current_sql_query = sql_query
                        st.session_state.current_user_query = user_query
                        
                        st.success("✅ SQL query generated!")
                        
                    except Exception as e:
                        st.error(f"❌ Error generating SQL: {str(e)}")
        
        with col2:
            if st.button("🎯 Quick SQL Examples", key="quick_sql_btn"):
                with st.spinner("Generating quick SQL examples..."):
                    try:
                        if st.session_state.loaded_tables:
                            current_table = st.session_state.current_table if hasattr(st.session_state, 'current_table') else list(st.session_state.loaded_tables.keys())[0]
                            current_data = st.session_state.loaded_tables[current_table]
                        else:
                            current_data = st.session_state.data_processor.get_data()
                        
                        quick_sql = st.session_state.llm_agent.get_quick_sql_insights(current_data)
                        st.session_state.quick_sql_examples = quick_sql
                        
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
    
    # Display generated SQL query (persistent and editable)
    if hasattr(st.session_state, 'current_sql_query') and st.session_state.current_sql_query:
        st.markdown("### 📝 SQL Query:")
        
        # Editable SQL query
        edited_sql = st.text_area(
            "Edit SQL Query:",
            value=st.session_state.current_sql_query,
            height=150,
            help="You can edit the generated SQL query or write your own custom query"
        )
        
        # Update the stored SQL query if edited
        if edited_sql != st.session_state.current_sql_query:
            st.session_state.current_sql_query = edited_sql
        
        # AI Workflow: Step 3 - Execute Query
        st.subheader("📊 Step 3: Execute Query")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if st.button("▶️ Execute Query", type="primary", key="execute_query_btn"):
                with st.spinner("Executing query..."):
                    try:
                        # Get all available tables
                        all_tables = {}
                        if st.session_state.loaded_tables:
                            all_tables.update(st.session_state.loaded_tables)
                        if st.session_state.data_processor.has_data():
                            all_tables['main'] = st.session_state.data_processor.get_data()
                        
                        # Execute SQL query
                        try:
                            # Register all tables in DuckDB
                            conn = duckdb.connect()
                            for table_name, df in all_tables.items():
                                conn.register(table_name, df)
                            
                            # Execute query
                            result_df = conn.execute(st.session_state.current_sql_query).fetchdf()
                            
                            if not result_df.empty:
                                st.success("✅ Query executed successfully!")
                                
                                # Store results
                                st.session_state.current_query_results = result_df
                                
                                # Display results summary
                                st.markdown("### 📋 Query Results Summary:")
                                st.markdown(f"**Rows returned:** {len(result_df)}")
                                st.markdown(f"**Columns:** {', '.join(result_df.columns)}")
                                
                                # Display results
                                st.markdown("### 📊 Data Preview:")
                                safe_dataframe_display(result_df, width='stretch')
                                
                            else:
                                st.warning("⚠️ Query returned no results.")
                                
                        except Exception as sql_error:
                            st.error(f"❌ SQL execution error: {str(sql_error)}")
                            st.info("💡 Try rephrasing your question or check the SQL query.")
                        
                    except Exception as e:
                        st.error(f"❌ Error executing query: {str(e)}")
        
        with col2:
            if st.button("🔄 Re-generate SQL", key="regenerate_sql_btn"):
                st.session_state.current_sql_query = None
                st.rerun()
    
    # Display persistent query results if available
    if hasattr(st.session_state, 'current_query_results') and st.session_state.current_query_results is not None:
        st.markdown("---")
        st.subheader("📊 Query Results")
        
        result_df = st.session_state.current_query_results
        
        # Display results summary
        st.markdown("### 📋 Results Summary:")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows Returned", len(result_df))
        with col2:
            st.metric("Columns", len(result_df.columns))
        with col3:
            st.metric("Memory Usage", f"{result_df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
        
        # Display results
        st.markdown("### 📊 Data Preview:")
        safe_dataframe_display(result_df, width='stretch')
    
    # Display Quick SQL Examples
    if hasattr(st.session_state, 'quick_sql_examples') and st.session_state.quick_sql_examples:
        st.markdown("### 🎯 Quick SQL Examples:")
        st.text_area("SQL Examples", value=st.session_state.quick_sql_examples, height=200, disabled=True)
    
    # Custom SQL execution section
    st.markdown("---")
    st.subheader("🔧 Custom SQL Execution")
    
    # Show available tables
    all_tables = {}
    if hasattr(st.session_state, 'loaded_tables'):
        all_tables.update(st.session_state.loaded_tables)
    if hasattr(st.session_state, 'data_processor') and st.session_state.data_processor.has_data():
        all_tables['main'] = st.session_state.data_processor.get_data()
    
    if all_tables:
        st.markdown("**Available Tables:**")
        for table_name, df in all_tables.items():
            st.markdown(f"- **{table_name}**: {df.shape[0]} rows × {df.shape[1]} columns")
            st.markdown(f"  - Columns: {', '.join(df.columns.tolist())}")
        
        # Custom SQL input
        custom_sql = st.text_area(
            "Enter Custom SQL Query:",
            placeholder="SELECT * FROM table_name LIMIT 10;",
            height=100,
            help="Write your own SQL query using the available tables"
        )
        
        if custom_sql and st.button("▶️ Execute Custom SQL", key="execute_custom_sql_btn"):
            with st.spinner("Executing custom SQL query..."):
                try:
                    # Register all tables in DuckDB
                    conn = duckdb.connect()
                    for table_name, df in all_tables.items():
                        conn.register(table_name, df)
                    
                    # Execute custom query
                    result_df = conn.execute(custom_sql).fetchdf()
                    
                    if not result_df.empty:
                        st.success("✅ Custom SQL query executed successfully!")
                        
                        # Store results
                        st.session_state.current_query_results = result_df
                        st.session_state.current_sql_query = custom_sql
                        st.session_state.current_user_query = "Custom SQL Query"
                        
                        # Display results summary
                        st.markdown("### 📋 Query Results Summary:")
                        st.markdown(f"**Rows returned:** {len(result_df)}")
                        st.markdown(f"**Columns:** {', '.join(result_df.columns)}")
                        
                        # Display results
                        st.markdown("### 📊 Data Preview:")
                        safe_dataframe_display(result_df, width='stretch')
                        
                    else:
                        st.warning("⚠️ Custom SQL query returned no results.")
                        
                except Exception as sql_error:
                    st.error(f"❌ SQL execution error: {str(sql_error)}")
                    st.info("💡 Check your SQL syntax and table names.")
    else:
        st.info("No tables available. Please load data first.")
    
    # AI Workflow: Step 4 - Visualization (Separate Button)
    if hasattr(st.session_state, 'current_query_results') and not st.session_state.current_query_results.empty:
        st.subheader("📈 Step 4: Create Visualization")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if st.button("📊 Generate Visualization", type="primary", key="generate_viz_btn"):
                with st.spinner("Creating visualization..."):
                    try:
                        # Create visualization
                        viz_fig = st.session_state.visualizer.create_auto_visualization(
                            st.session_state.current_query_results, 
                            st.session_state.current_user_query
                        )
                        if viz_fig:
                            st.session_state.current_visualization = viz_fig
                            st.success("✅ Visualization created!")
                        else:
                            st.warning("⚠️ Could not create visualization for this data.")
                    except Exception as e:
                        st.error(f"❌ Error creating visualization: {str(e)}")
        
        with col2:
            if st.button("🤖✨ Generate AI Insights", type="primary", key="generate_insights_btn"):
                with st.spinner("AI is analyzing the results..."):
                    try:
                        # Get source tables for enhanced analysis
                        source_tables = {}
                        if hasattr(st.session_state, 'loaded_tables'):
                            source_tables.update(st.session_state.loaded_tables)
                        if hasattr(st.session_state, 'data_processor') and st.session_state.data_processor.has_data():
                            source_tables['main'] = st.session_state.data_processor.get_data()
                        
                        # Generate AI analysis with source tables context
                        analysis = st.session_state.llm_agent.analyze_query_results(
                            st.session_state.current_user_query,
                            st.session_state.current_query_results,
                            source_tables
                        )
                        
                        # Store analysis
                        st.session_state.current_ai_analysis = analysis
                        st.session_state.analysis_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        
                        st.success("✅ AI analysis complete!")
                        
                    except Exception as e:
                        st.error(f"❌ Error generating analysis: {str(e)}")
    
    # Display Visualization
    if hasattr(st.session_state, 'current_visualization') and st.session_state.current_visualization:
        st.markdown("### 📈 Visualization:")
        st.plotly_chart(st.session_state.current_visualization, width='stretch')
    
    # Display AI Analysis
    if hasattr(st.session_state, 'current_ai_analysis') and st.session_state.current_ai_analysis:
        st.markdown("### 🧠 AI Analysis:")
        st.markdown(st.session_state.current_ai_analysis)
        
        # Store in insights history
        if 'insights_history' not in st.session_state:
            st.session_state.insights_history = []
        
        insight_entry = {
            'question': st.session_state.current_user_query,
            'sql_query': st.session_state.current_sql_query,
            'analysis': st.session_state.current_ai_analysis,
            'timestamp': st.session_state.analysis_timestamp,
            'data_shape': st.session_state.current_query_results.shape
        }
        
        # Add to history if not already there
        if not any(entry['timestamp'] == insight_entry['timestamp'] for entry in st.session_state.insights_history):
            st.session_state.insights_history.append(insight_entry)
    
    # Analysis history
    if hasattr(st.session_state, 'insights_history') and st.session_state.insights_history:
        st.markdown("---")
        st.subheader("📚 Recent Analysis")
        
        for i, entry in enumerate(st.session_state.insights_history[-3:]):  # Show last 3
            with st.expander(f"🔍 {entry['question'][:50]}... ({entry['timestamp']})"):
                st.markdown(f"**Question:** {entry['question']}")
                st.markdown(f"**SQL Query:**")
                st.code(entry['sql_query'], language="sql")
                st.markdown(f"**AI Analysis:**")
                st.markdown(entry['analysis'])
                st.markdown(f"**Data Shape:** {entry['data_shape']}")


def custom_sql_tab():
    """Custom SQL execution tab"""
    st.header("💻 Custom SQL Execution")
    
    if not st.session_state.loaded_tables and not st.session_state.data_processor.has_data():
        st.markdown('<div class="status-warning">⚠️ No data loaded. Please load data first.</div>', unsafe_allow_html=True)
        return
    
    # Get all available tables
    all_tables = {}
    if st.session_state.loaded_tables:
        all_tables.update(st.session_state.loaded_tables)
    if st.session_state.data_processor.has_data():
        all_tables["main"] = st.session_state.data_processor.get_data()
    
    if not all_tables:
        st.markdown('<div class="status-warning">⚠️ No tables available for SQL execution.</div>', unsafe_allow_html=True)
        return
    
    # Show available tables
    st.subheader("📊 Available Tables")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Tables", len(all_tables))
    
    with col2:
        total_rows = sum(df.shape[0] for df in all_tables.values())
        st.metric("Total Rows", f"{total_rows:,}")
    
    with col3:
        total_cols = sum(df.shape[1] for df in all_tables.values())
        st.metric("Total Columns", total_cols)
    
    # Table information
    with st.expander("📋 Table Details", expanded=False):
        for table_name, df in all_tables.items():
            st.markdown(f"**{table_name}** ({df.shape[0]} rows × {df.shape[1]} columns)")
            st.markdown(f"Columns: {', '.join(df.columns.tolist())}")
            st.markdown("---")
    
    # SQL Query Editor
    st.subheader("✏️ SQL Query Editor")
    
    # Pre-populate with a sample query
    sample_query = f"""-- Example queries for your data:
-- SELECT * FROM {list(all_tables.keys())[0]} LIMIT 10;
-- SELECT COUNT(*) as total_rows FROM {list(all_tables.keys())[0]};
-- SELECT * FROM {list(all_tables.keys())[0]} WHERE {list(all_tables[list(all_tables.keys())[0]].columns)[0]} IS NOT NULL;
"""
    
    custom_sql = st.text_area(
        "Write your SQL query:",
        value=sample_query,
        height=200,
        placeholder="SELECT * FROM table_name LIMIT 10;",
        help="Write your own SQL query using the available tables"
    )
    
    # Execute button
    col1, col2 = st.columns([1, 4])
    
    with col1:
        if st.button("▶️ Execute SQL", type="primary", key="execute_custom_sql_main_btn"):
            if custom_sql and custom_sql.strip():
                with st.spinner("Executing SQL query..."):
                    try:
                        # Register all tables in DuckDB
                        conn = duckdb.connect()
                        for table_name, df in all_tables.items():
                            conn.register(table_name, df)
                        
                        # Execute query
                        result_df = conn.execute(custom_sql).fetchdf()
                        
                        if not result_df.empty:
                            st.success("✅ Query executed successfully!")
                            
                            # Store results for visualization and insights
                            st.session_state.custom_sql_results = result_df
                            st.session_state.custom_sql_query = custom_sql
                            
                            # Display results
                            st.markdown("### 📊 Query Results")
                            safe_dataframe_display(result_df, width='stretch')
                            
                        else:
                            st.warning("⚠️ Query returned no results.")
                            
                    except Exception as sql_error:
                        st.error(f"❌ SQL execution error: {str(sql_error)}")
                        st.info("💡 Check your SQL syntax and table/column names.")
                    finally:
                        conn.close()
            else:
                st.warning("⚠️ Please enter a SQL query.")
    
    with col2:
        if st.button("🧹 Clear Query", key="clear_custom_sql_btn"):
            # Clear the query and results
            if 'custom_sql_query' in st.session_state:
                del st.session_state.custom_sql_query
            if 'custom_sql_results' in st.session_state:
                del st.session_state.custom_sql_results
            st.rerun()
    
    # Show results if available
    if hasattr(st.session_state, 'custom_sql_results') and st.session_state.custom_sql_results is not None:
        st.markdown("---")
        st.subheader("📊 Query Results")
        
        result_df = st.session_state.custom_sql_results
        
        # Results summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows Returned", len(result_df))
        with col2:
            st.metric("Columns", len(result_df.columns))
        with col3:
            st.metric("Memory Usage", f"{result_df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
        
        # Visualization and AI Analysis options
        st.markdown("### 🎨 Analysis Options")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Generate Visualization", key="custom_sql_viz_btn"):
                with st.spinner("Creating visualization..."):
                    try:
                        viz_fig = st.session_state.visualizer.create_auto_visualization(result_df, custom_sql)
                        if viz_fig:
                            st.plotly_chart(viz_fig, width='stretch')
                        else:
                            st.warning("⚠️ Could not create visualization for this data.")
                    except Exception as e:
                        st.error(f"❌ Error creating visualization: {str(e)}")
        
        with col2:
            if st.button("🤖 Generate AI Insights", key="custom_sql_insights_btn", type="primary"):
                if hasattr(st.session_state, 'llm_agent') and st.session_state.llm_agent.is_initialized():
                    with st.spinner("AI is analyzing the results..."):
                        try:
                            # Get source tables for enhanced analysis
                            source_tables = {}
                            if hasattr(st.session_state, 'loaded_tables'):
                                source_tables.update(st.session_state.loaded_tables)
                            if st.session_state.data_processor.has_data():
                                source_tables['main'] = st.session_state.data_processor.get_data()
                            
                            # Generate AI insights
                            analysis = st.session_state.llm_agent.analyze_query_results(
                                f"Custom SQL Query: {custom_sql}", 
                                result_df, 
                                source_tables
                            )
                            
                            # Store insights
                            insight_entry = {
                                'question': f"Custom SQL Analysis: {custom_sql[:50]}...",
                                'sql_query': custom_sql,
                                'data_shape': result_df.shape,
                                'insights': analysis,
                                'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                                'query_type': 'custom_sql'
                            }
                            
                            if 'insights_history' not in st.session_state:
                                st.session_state.insights_history = []
                            st.session_state.insights_history.append(insight_entry)
                            
                            st.success("✅ AI insights generated! Check the Insights tab.")
                            
                        except Exception as e:
                            st.error(f"❌ Error generating AI insights: {str(e)}")
                            print(f"AI insights error: {str(e)}")
                else:
                    st.warning("⚠️ Please initialize an AI model in the Data Load tab first.")

def insights_tab():
    """Insights tab with formatted document view and PDF download"""
    st.header("📄 AI Insights & Reports")
    
    # Initialize insights_history if it doesn't exist
    if 'insights_history' not in st.session_state:
        st.session_state.insights_history = []
    
    if not st.session_state.insights_history:
        st.markdown('<div class="status-warning">⚠️ No insights generated yet. Please use the AI Analysis tab to generate insights first.</div>', unsafe_allow_html=True)
        return
    
    # Display all insights
    st.subheader("📚 Generated Insights")
    
    for i, entry in enumerate(st.session_state.insights_history):
        with st.expander(f"📊 Analysis #{i+1}: {entry['question'][:60]}... ({entry['timestamp']})", expanded=(i == len(st.session_state.insights_history) - 1)):
            
            # Create a formatted document view
            st.markdown("---")
            st.markdown(f"### 📝 **Question**")
            st.markdown(f"*{entry['question']}*")
            
            st.markdown(f"### 🔍 **SQL Query**")
            st.code(entry['sql_query'], language="sql")
            
            st.markdown(f"### 📊 **Data Summary**")
            st.markdown(f"- **Rows:** {entry['data_shape'][0]}")
            st.markdown(f"- **Columns:** {entry['data_shape'][1]}")
            st.markdown(f"- **Generated:** {entry['timestamp']}")
            
            st.markdown(f"### 🧠 **AI Analysis**")
            st.markdown(entry['analysis'])
            
            # Export buttons for this specific insight
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button(f"📄 Export as PDF", key=f"pdf_{i}"):
                    try:
                        if REPORTLAB_AVAILABLE:
                            # Generate proper PDF using ReportLab
                            pdf_data = generate_pdf_report(entry, f"ai_insights_report_{entry['timestamp'].replace(':', '-').replace(' ', '_')}.pdf")
                            st.download_button(
                                label="Download PDF Report",
                                data=pdf_data,
                                file_name=f"ai_insights_report_{entry['timestamp'].replace(':', '-').replace(' ', '_')}.pdf",
                                mime="application/pdf"
                            )
                        else:
                            st.error("❌ ReportLab not available. Install with: pip install reportlab")
                    except Exception as e:
                        st.error(f"❌ Error creating PDF report: {str(e)}")
            
            with col2:
                if st.button(f"📊 Export as Excel", key=f"excel_{i}"):
                    try:
                        # Create Excel file
                        output = io.BytesIO()
                        with pd.ExcelWriter(output, engine='openpyxl') as writer:
                            # Analysis sheet
                            analysis_df = pd.DataFrame({
                                'Question': [entry['question']],
                                'SQL_Query': [entry['sql_query']],
                                'AI_Analysis': [entry['analysis']],
                                'Generated_At': [entry['timestamp']],
                                'Data_Rows': [entry['data_shape'][0]],
                                'Data_Columns': [entry['data_shape'][1]]
                            })
                            analysis_df.to_excel(writer, sheet_name='Analysis', index=False)
                        
                        st.download_button(
                            label="Download Excel Report",
                            data=output.getvalue(),
                            file_name=f"ai_insights_{entry['timestamp'].replace(':', '-').replace(' ', '_')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                    except Exception as e:
                        st.error(f"❌ Error creating Excel report: {str(e)}")
            
            with col3:
                if st.button(f"📝 Export as Word", key=f"word_{i}"):
                    try:
                        if DOCX_AVAILABLE:
                            # Generate Word document
                            word_data = generate_word_report(entry)
                            st.download_button(
                                label="Download Word Report",
                                data=word_data,
                                file_name=f"ai_insights_{entry['timestamp'].replace(':', '-').replace(' ', '_')}.docx",
                                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                            )
                        else:
                            st.error("❌ python-docx not available. Install with: pip install python-docx")
                    except Exception as e:
                        st.error(f"❌ Error creating Word report: {str(e)}")
    
    # Additional export options
    if len(st.session_state.insights_history) > 0:
        st.markdown("---")
        st.subheader("📤 Bulk Export Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📄 Export All as PDF", key="bulk_pdf"):
                try:
                    if REPORTLAB_AVAILABLE:
                        # Create a combined PDF with all insights
                        buffer = io.BytesIO()
                        doc = SimpleDocTemplate(buffer, pagesize=A4)
                        story = []
                        
                        for i, entry in enumerate(st.session_state.insights_history):
                            if i > 0:
                                story.append(PageBreak())
                            
                            # Add title for each insight
                            title_style = ParagraphStyle('Title', fontSize=18, spaceAfter=20)
                            story.append(Paragraph(f"Analysis #{i+1}: {entry['question'][:50]}...", title_style))
                            
                            # Add content
                            pdf_data = generate_pdf_report(entry, "")
                            # Note: This is a simplified version - in practice, you'd need to parse the PDF content
                            story.append(Paragraph(f"Question: {entry['question']}", getSampleStyleSheet()['Normal']))
                            story.append(Paragraph(f"SQL: {entry['sql_query']}", getSampleStyleSheet()['Code']))
                            story.append(Paragraph(f"Analysis: {entry['analysis']}", getSampleStyleSheet()['Normal']))
                        
                        doc.build(story)
                        buffer.seek(0)
                        
                        st.download_button(
                            label="Download All Insights as PDF",
                            data=buffer.getvalue(),
                            file_name=f"all_insights_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                            mime="application/pdf"
                        )
                    else:
                        st.error("❌ ReportLab not available. Install with: pip install reportlab")
                except Exception as e:
                    st.error(f"❌ Error creating bulk PDF: {str(e)}")
        
        with col2:
            if st.button("🗑️ Clear All Insights", type="secondary", key="clear_insights_btn_1"):
                st.session_state.insights_history = []
                st.rerun()
    
    # Clear all insights button
    if len(st.session_state.insights_history) > 0:
        st.markdown("---")
        if st.button("🗑️ Clear All Insights", type="secondary", key="clear_insights_btn_2"):
            st.session_state.insights_history = []
            st.rerun()

def settings_tab():
    """Settings and configuration tab"""
    st.header("⚙️ Settings & Configuration")
    
    # Create tabs within settings
    settings_tab1, settings_tab2, settings_tab3, settings_tab4 = st.tabs([
        "🤖 AI Models", 
        "🔗 Data Connectors", 
        "📤 Export Data", 
        "ℹ️ System Info"
    ])
    
    with settings_tab1:
        st.subheader("🤖 AI Model Management")
        
        if OLLAMA_AVAILABLE:
            available_models = get_available_models()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Available Models:** {len(available_models)}")
                if available_models:
                    for model in available_models:
                        st.write(f"• {model}")
                else:
                    st.warning("No models found")
            
            with col2:
                if st.button("🔄 Refresh Model List", key="refresh_models_btn"):
                    st.rerun()
                
                if st.button("📥 Open Model Setup", key="open_model_setup_btn"):
                    st.info("Run `python setup_models.py` in your terminal")
        else:
            st.error("Ollama configuration not available")
    
    with settings_tab2:
        st.subheader("🔗 Data Connector Configuration")
        
        # Google Analytics Connector
        with st.expander("📊 Google Analytics Connector"):
            st.markdown("Connect to Google Analytics to import web traffic data")
            
            col1, col2 = st.columns(2)
            
            with col1:
                ga_property_id = st.text_input("Property ID", placeholder="123456789", key="ga_property_id")
                ga_start_date = st.date_input("Start Date", key="ga_start_date")
                ga_end_date = st.date_input("End Date", key="ga_end_date")
            
            with col2:
                ga_metrics = st.multiselect(
                    "Metrics to Import",
                    ["sessions", "users", "pageviews", "bounceRate", "avgSessionDuration"],
                    default=["sessions", "users", "pageviews"],
                    key="ga_metrics"
                )
                ga_dimensions = st.multiselect(
                    "Dimensions to Import", 
                    ["date", "country", "deviceCategory", "trafficSource"],
                    default=["date", "country"],
                    key="ga_dimensions"
                )
            
            if st.button("🔗 Test GA Connection", key="test_ga_btn"):
                st.info("GA connector test - requires authentication setup")
                st.code("""
# To set up GA connector:
# 1. Create Google Cloud project
# 2. Enable Analytics Reporting API
# 3. Create service account credentials
# 4. Download JSON key file
# 5. Configure in connectors/ga_connector.py
                """)
        
        # Snowflake Connector
        with st.expander("❄️ Snowflake Connector"):
            st.markdown("Connect to Snowflake data warehouse")
            
            col1, col2 = st.columns(2)
            
            with col1:
                sf_account = st.text_input("Account", placeholder="your-account.snowflakecomputing.com", key="sf_account")
                sf_username = st.text_input("Username", key="sf_username")
                sf_password = st.text_input("Password", type="password", key="sf_password")
            
            with col2:
                sf_database = st.text_input("Database", placeholder="PROD_DB", key="sf_database")
                sf_schema = st.text_input("Schema", placeholder="PUBLIC", key="sf_schema")
                sf_warehouse = st.text_input("Warehouse", placeholder="COMPUTE_WH", key="sf_warehouse")
            
            if st.button("🔗 Test Snowflake Connection"):
                st.info("Snowflake connector test - requires valid credentials")
                st.code("""
# To set up Snowflake connector:
# 1. Get account URL from Snowflake console
# 2. Create user with appropriate permissions
# 3. Configure connection parameters
# 4. Test connection in connectors/snowflake_connector.py
                """)
        
        # Oncore Connector
        with st.expander("🏥 Oncore Connector"):
            st.markdown("Connect to Oncore clinical data system")
            
            col1, col2 = st.columns(2)
            
            with col1:
                oncore_base_url = st.text_input("Base URL", placeholder="https://your-oncore-instance.com", key="oncore_base_url")
                oncore_username = st.text_input("Username", key="oncore_username")
                oncore_password = st.text_input("Password", type="password", key="oncore_password")
            
            with col2:
                oncore_database = st.text_input("Database", placeholder="ONCORE_PROD", key="oncore_database")
                oncore_study_id = st.text_input("Study ID (optional)", key="oncore_study_id")
            
            if st.button("🔗 Test Oncore Connection"):
                st.info("Oncore connector test - requires valid credentials")
                st.code("""
# To set up Oncore connector:
# 1. Get Oncore instance URL
# 2. Create API user account
# 3. Configure database connection
# 4. Test connection in connectors/oncore_connector.py
                """)
        
        # Excel/CSV Connector (already working)
        with st.expander("📁 Excel/CSV Connector"):
            st.markdown("✅ Excel and CSV file upload is already configured and working")
            st.success("No additional configuration needed - use the Data Load tab to upload files")
    
    with settings_tab3:
        st.subheader("📤 Export Your Data")
        
        if st.session_state.loaded_tables or st.session_state.data_processor.has_data():
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("💾 Export as CSV"):
                    try:
                        # Get current data
                        if st.session_state.loaded_tables:
                            current_table = st.session_state.current_table if hasattr(st.session_state, 'current_table') else list(st.session_state.loaded_tables.keys())[0]
                            current_data = st.session_state.loaded_tables[current_table]
                        else:
                            current_data = st.session_state.data_processor.get_data()
                        
                        # Convert mixed data types to strings to avoid Arrow serialization issues
                        export_data = current_data.copy()
                        for col in export_data.columns:
                            if export_data[col].dtype == 'object':
                                export_data[col] = export_data[col].astype(str)
                            elif 'datetime' in str(export_data[col].dtype):
                                export_data[col] = export_data[col].dt.strftime('%Y-%m-%d %H:%M:%S')
                        
                        csv_data = export_data.to_csv(index=False)
                        st.download_button(
                            label="Download CSV",
                            data=csv_data,
                            file_name=f"business_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    except Exception as e:
                        st.error(f"Error exporting CSV: {str(e)}")
            
            with col2:
                if st.button("📊 Export All Tables"):
                    # Export all loaded tables
                    export_data = {}
                    if st.session_state.loaded_tables:
                        export_data.update(st.session_state.loaded_tables)
                    if st.session_state.data_processor.has_data():
                        export_data["main"] = st.session_state.data_processor.get_data()
                    
                    # Create Excel file with multiple sheets
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        for sheet_name, df in export_data.items():
                            df.to_excel(writer, sheet_name=sheet_name, index=False)
                    
                    st.download_button(
                        label="Download Excel",
                        data=output.getvalue(),
                        file_name=f"all_tables_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
        else:
            st.info("No data loaded to export")
    
    with settings_tab4:
        st.subheader("ℹ️ System Information")
        
        info_col1, info_col2 = st.columns(2)
        
        with info_col1:
            st.metric("Python Version", "3.10+")
            st.metric("Streamlit Version", "1.28.1")
            st.metric("DuckDB Version", "0.9.2")
        
        with info_col2:
            st.metric("Pandas Version", "2.1.3")
            st.metric("Ollama Integration", "Direct API")
            st.metric("Plotly Version", "5.17.0")
        
        # Additional system info
        st.markdown("---")
        st.subheader("🔧 Technical Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Available Connectors:**")
            st.write("• ✅ Excel/CSV (Active)")
            st.write("• 🔧 Google Analytics (Configurable)")
            st.write("• 🔧 Snowflake (Configurable)")
            st.write("• 🔧 Oncore (Configurable)")
        
        with col2:
            st.write("**System Status:**")
            st.write("• ✅ Offline Operation")
            st.write("• ✅ AI Analysis Ready")
            st.write("• ✅ Data Processing Active")
            st.write("• ✅ Visualization Ready")

if __name__ == "__main__":
    main()