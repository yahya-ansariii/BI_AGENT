"""
Configuration file for Business Insights Agent
"""

import os
from pathlib import Path

# Application settings
APP_NAME = "Business Insights Agent"
APP_VERSION = "1.0.0"
APP_DESCRIPTION = "Offline AI-powered business intelligence tool"

# File paths
BASE_DIR = Path(__file__).parent
SCHEMA_DIR = BASE_DIR / "schemas"
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
SCHEMA_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# Streamlit configuration
STREAMLIT_CONFIG = {
    "page_title": APP_NAME,
    "page_icon": "📊",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# DuckDB configuration
DUCKDB_CONFIG = {
    "memory_limit": "1GB",
    "threads": 4,
    "enable_progress_bar": True
}

# Ollama configuration
OLLAMA_CONFIG = {
    "base_url": "http://localhost:11434",
    "timeout": 60,
    "default_model": "llama2",
    "available_models": ["llama2", "codellama", "mistral", "phi"]
}

# Data processing configuration
DATA_CONFIG = {
    "max_file_size_mb": 100,
    "supported_formats": [".xlsx", ".xls", ".csv"],
    "sample_data_size": 1000,
    "chunk_size": 10000
}

# Visualization configuration
VIZ_CONFIG = {
    "default_theme": "plotly_white",
    "color_palette": "viridis",
    "max_categories": 20,
    "chart_height": 400
}

# LLM configuration
LLM_CONFIG = {
    "max_tokens": 2000,
    "temperature": 0.7,
    "top_p": 0.9,
    "max_retries": 3
}

# Schema configuration
SCHEMA_CONFIG = {
    "auto_detect_relationships": True,
    "max_relationship_depth": 3,
    "confidence_threshold": 0.7
}

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": LOGS_DIR / "app.log"
}

# Security configuration
SECURITY_CONFIG = {
    "enable_cors": True,
    "max_upload_size": 100 * 1024 * 1024,  # 100MB
    "allowed_file_types": ["xlsx", "xls", "csv"]
}

# Performance configuration
PERFORMANCE_CONFIG = {
    "enable_caching": True,
    "cache_ttl": 3600,  # 1 hour
    "max_concurrent_requests": 10,
    "enable_compression": True
}

# Feature flags
FEATURES = {
    "enable_ai_analysis": True,
    "enable_export": True,
    "enable_schema_management": True,
    "enable_real_time_updates": False,
    "enable_collaboration": False
}

# Default prompts for LLM
DEFAULT_PROMPTS = {
    "data_analysis": """
You are a business intelligence analyst. Analyze the provided data and provide insights.
Focus on:
1. Key trends and patterns
2. Business implications
3. Actionable recommendations
4. Data quality observations
""",
    "sales_analysis": """
Analyze the sales data and provide insights on:
1. Sales performance trends
2. Top performing products/categories
3. Customer behavior patterns
4. Revenue optimization opportunities
""",
    "customer_analysis": """
Analyze customer data and provide insights on:
1. Customer segmentation
2. Customer lifetime value
3. Retention patterns
4. Growth opportunities
"""
}

# Sample data configuration
SAMPLE_DATA_CONFIG = {
    "products": [
        "Laptop Pro", "Smartphone X", "Wireless Headphones", "Gaming Mouse",
        "Mechanical Keyboard", "Monitor 4K", "Webcam HD", "Tablet Air",
        "Smart Watch", "Bluetooth Speaker", "USB-C Hub", "Laptop Stand",
        "Desk Lamp", "Office Chair", "Standing Desk", "Cable Organizer",
        "Power Bank", "Car Charger", "Phone Case", "Screen Protector"
    ],
    "categories": [
        "Electronics", "Computers", "Accessories", "Mobile", "Gaming",
        "Office", "Home", "Travel", "Health", "Entertainment"
    ],
    "regions": ["North America", "Europe", "Asia Pacific", "Latin America", "Middle East"],
    "customer_segments": ["Premium", "Standard", "Basic", "Enterprise", "SMB"],
    "sales_reps": ["Alice Johnson", "Bob Smith", "Charlie Brown", "Diana Prince", "Eve Wilson"]
}

# Export configuration
EXPORT_CONFIG = {
    "csv_encoding": "utf-8",
    "excel_engine": "openpyxl",
    "json_indent": 2,
    "image_format": "png",
    "image_dpi": 300
}

# Error messages
ERROR_MESSAGES = {
    "file_too_large": "File size exceeds maximum allowed size of {max_size}MB",
    "unsupported_format": "Unsupported file format. Please use Excel (.xlsx, .xls) or CSV files",
    "ollama_not_running": "Ollama is not running. Please start Ollama service first",
    "model_not_found": "Model not found. Please pull the model first using 'ollama pull {model_name}'",
    "data_processing_error": "Error processing data: {error}",
    "visualization_error": "Error creating visualization: {error}",
    "llm_error": "Error generating AI insights: {error}"
}

# Success messages
SUCCESS_MESSAGES = {
    "data_loaded": "Data loaded successfully!",
    "model_initialized": "AI model initialized successfully!",
    "analysis_complete": "Analysis completed successfully!",
    "export_complete": "Export completed successfully!",
    "schema_updated": "Schema updated successfully!"
}

# Validation rules
VALIDATION_RULES = {
    "min_rows": 1,
    "max_rows": 1000000,
    "min_columns": 1,
    "max_columns": 1000,
    "required_columns": [],  # No specific required columns
    "date_formats": ["%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%Y-%m-%d %H:%M:%S"]
}

# API endpoints (for future use)
API_ENDPOINTS = {
    "health": "/api/health",
    "data": "/api/data",
    "analysis": "/api/analysis",
    "export": "/api/export",
    "schema": "/api/schema"
}

# Database configuration (for future use)
DATABASE_CONFIG = {
    "type": "sqlite",  # Default to SQLite for simplicity
    "path": DATA_DIR / "app.db",
    "backup_interval": 24,  # hours
    "max_connections": 10
}

# Monitoring configuration
MONITORING_CONFIG = {
    "enable_metrics": True,
    "metrics_interval": 60,  # seconds
    "log_performance": True,
    "alert_thresholds": {
        "memory_usage": 80,  # percentage
        "response_time": 5,  # seconds
        "error_rate": 5  # percentage
    }
}
