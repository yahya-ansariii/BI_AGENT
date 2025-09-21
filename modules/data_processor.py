"""
Data Processing Module
Handles Excel data loading, DuckDB operations, and data management
"""

import pandas as pd
import duckdb
import io
import numpy as np
from typing import Optional, List, Dict, Any
import psutil
import os

class DataProcessor:
    """Handles data processing operations using DuckDB and Pandas"""
    
    def __init__(self):
        """Initialize the data processor"""
        self.conn = duckdb.connect(':memory:')
        self.data = None
        self.data_name = None
        self.memory_usage = 0
        self.loaded_tables = {}
        
    def load_excel_data(self, file_upload) -> bool:
        """
        Load data from uploaded file (supports Excel, CSV, and other formats)
        
        Args:
            file_upload: Streamlit file uploader object
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            file_extension = file_upload.name.lower().split('.')[-1]
            
            # Handle different file formats
            if file_extension == 'csv':
                # Read CSV file
                self.data = pd.read_csv(file_upload, encoding='utf-8')
            elif file_extension in ['xlsx', 'xlsm', 'xlsb']:
                # Read Excel files with openpyxl engine
                self.data = pd.read_excel(file_upload, engine='openpyxl')
            elif file_extension in ['xls']:
                # Read older Excel files with xlrd engine
                self.data = pd.read_excel(file_upload, engine='xlrd')
            elif file_extension in ['ods', 'odf']:
                # Read OpenDocument files
                self.data = pd.read_excel(file_upload, engine='odf')
            else:
                # Try to read as CSV as fallback
                try:
                    self.data = pd.read_csv(file_upload, encoding='utf-8')
                except:
                    # If CSV fails, try Excel
                    self.data = pd.read_excel(file_upload)
            
            self.data_name = file_upload.name
            self._update_memory_usage()
            self._register_data_in_duckdb()
            return True
            
        except Exception as e:
            print(f"Error loading data file: {str(e)}")
            return False
    
    def load_sample_data(self) -> bool:
        """
        Load sample business data from demo_data folder Excel files
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            import os
            
            # Clear any existing loaded tables to ensure fresh data
            self.loaded_tables = {}
            
            # Path to demo_data folder
            demo_data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'demo_data')
            
            # Load sales.xlsx
            sales_file = os.path.join(demo_data_path, 'sales.xlsx')
            if os.path.exists(sales_file):
                sales_df = pd.read_excel(sales_file, sheet_name='Sheet1')
                self.loaded_tables['sales'] = sales_df
                print(f"Loaded sales data: {sales_df.shape[0]} rows × {sales_df.shape[1]} columns")
            
            # Load web_traffic.xlsx
            web_traffic_file = os.path.join(demo_data_path, 'web_traffic.xlsx')
            if os.path.exists(web_traffic_file):
                web_traffic_df = pd.read_excel(web_traffic_file, sheet_name='Sheet1')
                self.loaded_tables['web_traffic'] = web_traffic_df
                print(f"Loaded web traffic data: {web_traffic_df.shape[0]} rows × {web_traffic_df.shape[1]} columns")
            
            if not self.loaded_tables:
                print("No demo data files found in demo_data folder")
                return False
            
            # Set the main data to the first available table for backward compatibility
            first_table = list(self.loaded_tables.keys())[0]
            self.data = self.loaded_tables[first_table]
            self.data_name = f"Sample Data ({first_table})"
            self._update_memory_usage()
            self._register_data_in_duckdb()
            
            print(f"Successfully loaded {len(self.loaded_tables)} sample tables from demo_data folder")
            return True
            
        except Exception as e:
            print(f"Error loading sample data from demo_data folder: {str(e)}")
            return False
    
    def _update_memory_usage(self):
        """Update memory usage statistics"""
        if self.data is not None:
            self.memory_usage = self.data.memory_usage(deep=True).sum() / (1024 * 1024)  # MB
    
    def _register_data_in_duckdb(self):
        """Register the current data in DuckDB"""
        if self.data is not None:
            self.conn.register('business_data', self.data)
    
    def has_data(self) -> bool:
        """Check if data is loaded"""
        return self.data is not None and not self.data.empty
    
    def get_data(self) -> pd.DataFrame:
        """Get the current data"""
        return self.data if self.data is not None else pd.DataFrame()
    
    def get_loaded_tables(self) -> Dict[str, pd.DataFrame]:
        """Get all loaded tables"""
        return self.loaded_tables
    
    def get_data_preview(self, rows: int = 10, columns: List[str] = None) -> pd.DataFrame:
        """
        Get a preview of the data
        
        Args:
            rows: Number of rows to return
            columns: List of columns to include (None for all)
            
        Returns:
            pd.DataFrame: Data preview
        """
        if not self.has_data():
            return pd.DataFrame()
        
        data = self.data.head(rows)
        if columns:
            data = data[columns]
        
        return data
    
    def get_column_names(self) -> List[str]:
        """Get list of column names"""
        if not self.has_data():
            return []
        return list(self.data.columns)
    
    def get_column_count(self) -> int:
        """Get number of columns"""
        if not self.has_data():
            return 0
        return len(self.data.columns)
    
    def get_record_count(self) -> int:
        """Get number of records"""
        if not self.has_data():
            return 0
        return len(self.data)
    
    def get_memory_usage(self) -> float:
        """Get memory usage in MB"""
        return self.memory_usage
    
    def get_data_quality_score(self) -> float:
        """Calculate data quality score (0-100)"""
        if not self.has_data():
            return 0.0
        
        total_cells = self.data.size
        missing_cells = self.data.isnull().sum().sum()
        
        if total_cells == 0:
            return 100.0
        
        quality_score = ((total_cells - missing_cells) / total_cells) * 100
        return round(quality_score, 1)
    
    def get_data_summary(self) -> Dict[str, Any]:
        """Get comprehensive data summary"""
        if not self.has_data():
            return {}
        
        summary = {
            'shape': self.data.shape,
            'columns': list(self.data.columns),
            'dtypes': self.data.dtypes.astype(str),  # Convert numpy dtypes to strings
            'missing_values': self.data.isnull().sum(),
            'memory_usage': self.memory_usage,
            'quality_score': self.get_data_quality_score()
        }
        
        return summary
    
    def get_descriptive_stats(self) -> pd.DataFrame:
        """Get descriptive statistics for numeric columns"""
        if not self.has_data():
            return pd.DataFrame()
        
        numeric_data = self.data.select_dtypes(include=[np.number])
        if numeric_data.empty:
            return pd.DataFrame()
        
        return numeric_data.describe()
    
    def query_data(self, sql_query: str) -> pd.DataFrame:
        """
        Execute SQL query on the data
        
        Args:
            sql_query: SQL query string
            
        Returns:
            pd.DataFrame: Query results
        """
        try:
            result = self.conn.execute(sql_query).fetchdf()
            return result
        except Exception as e:
            print(f"SQL query error: {str(e)}")
            return pd.DataFrame()
    
    def export_to_csv(self) -> str:
        """Export data to CSV string with proper data type handling"""
        if not self.has_data():
            return ""
        
        # Create a copy and convert mixed data types to strings to avoid Arrow serialization issues
        export_data = self.data.copy()
        for col in export_data.columns:
            if export_data[col].dtype == 'object':
                export_data[col] = export_data[col].astype(str)
            elif 'datetime' in str(export_data[col].dtype):
                export_data[col] = export_data[col].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        return export_data.to_csv(index=False)
    
    def get_numeric_columns(self) -> List[str]:
        """Get list of numeric column names"""
        if not self.has_data():
            return []
        
        return list(self.data.select_dtypes(include=[np.number]).columns)
    
    def get_categorical_columns(self) -> List[str]:
        """Get list of categorical column names"""
        if not self.has_data():
            return []
        
        return list(self.data.select_dtypes(include=['object']).columns)
    
    def get_column_info(self, column_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific column
        
        Args:
            column_name: Name of the column
            
        Returns:
            Dict: Column information
        """
        if not self.has_data() or column_name not in self.data.columns:
            return {}
        
        column_data = self.data[column_name]
        
        info = {
            'name': column_name,
            'dtype': str(column_data.dtype),
            'count': len(column_data),
            'null_count': column_data.isnull().sum(),
            'unique_count': column_data.nunique()
        }
        
        if column_data.dtype in ['int64', 'float64']:
            info.update({
                'min': column_data.min(),
                'max': column_data.max(),
                'mean': column_data.mean(),
                'std': column_data.std()
            })
        else:
            info.update({
                'top_value': column_data.mode().iloc[0] if not column_data.mode().empty else None,
                'top_frequency': column_data.value_counts().iloc[0] if not column_data.empty else 0
            })
        
        return info
    
    def clear_data(self):
        """Clear all data and reset the processor"""
        self.data = None
        self.data_name = None
        self.memory_usage = 0
        self.loaded_tables = {}
        if self.conn:
            self.conn.close()
        self.conn = duckdb.connect(':memory:')
    
    def close(self):
        """Close DuckDB connection"""
        if self.conn:
            self.conn.close()