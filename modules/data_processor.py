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
        
    def load_excel_data(self, file_upload) -> bool:
        """
        Load Excel data from uploaded file
        
        Args:
            file_upload: Streamlit file uploader object
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Read Excel file
            if file_upload.name.endswith('.xlsx'):
                self.data = pd.read_excel(file_upload, engine='openpyxl')
            else:
                self.data = pd.read_excel(file_upload, engine='xlrd')
            
            self.data_name = file_upload.name
            self._update_memory_usage()
            self._register_data_in_duckdb()
            return True
            
        except Exception as e:
            print(f"Error loading Excel data: {str(e)}")
            return False
    
    def load_sample_data(self) -> bool:
        """
        Load sample business data for demonstration
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Generate sample sales data
            np.random.seed(42)
            n_records = 1000
            
            data = {
                'date': pd.date_range('2023-01-01', periods=n_records, freq='D'),
                'product_id': np.random.randint(1, 21, n_records),
                'product_name': [f'Product_{i}' for i in np.random.randint(1, 21, n_records)],
                'category': np.random.choice(['Electronics', 'Clothing', 'Books', 'Home', 'Sports'], n_records),
                'customer_id': np.random.randint(1000, 2000, n_records),
                'customer_segment': np.random.choice(['Premium', 'Standard', 'Basic'], n_records),
                'quantity': np.random.randint(1, 10, n_records),
                'unit_price': np.round(np.random.uniform(10, 500, n_records), 2),
                'total_sales': 0,  # Will be calculated
                'region': np.random.choice(['North', 'South', 'East', 'West'], n_records),
                'sales_rep': np.random.choice(['Alice', 'Bob', 'Charlie', 'Diana'], n_records)
            }
            
            self.data = pd.DataFrame(data)
            self.data['total_sales'] = self.data['quantity'] * self.data['unit_price']
            self.data_name = "sample_business_data"
            self._update_memory_usage()
            self._register_data_in_duckdb()
            return True
            
        except Exception as e:
            print(f"Error loading sample data: {str(e)}")
            return False
    
    def _register_data_in_duckdb(self):
        """Register the current data in DuckDB for SQL queries"""
        if self.data is not None:
            self.conn.register('business_data', self.data)
    
    def _update_memory_usage(self):
        """Update memory usage tracking"""
        if self.data is not None:
            self.memory_usage = self.data.memory_usage(deep=True).sum() / 1024 / 1024  # MB
    
    def has_data(self) -> bool:
        """Check if data is loaded"""
        return self.data is not None
    
    def get_data(self) -> Optional[pd.DataFrame]:
        """Get the current data"""
        return self.data
    
    def get_record_count(self) -> int:
        """Get number of records in the dataset"""
        return len(self.data) if self.data is not None else 0
    
    def get_column_count(self) -> int:
        """Get number of columns in the dataset"""
        return len(self.data.columns) if self.data is not None else 0
    
    def get_column_names(self) -> List[str]:
        """Get list of column names"""
        return list(self.data.columns) if self.data is not None else []
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        return self.memory_usage
    
    def get_data_quality_score(self) -> float:
        """Calculate data quality score based on completeness and consistency"""
        if self.data is None:
            return 0.0
        
        # Calculate completeness score
        total_cells = self.data.size
        missing_cells = self.data.isnull().sum().sum()
        completeness_score = ((total_cells - missing_cells) / total_cells) * 100
        
        # Calculate consistency score (basic check for duplicates)
        duplicate_rows = self.data.duplicated().sum()
        consistency_score = max(0, 100 - (duplicate_rows / len(self.data)) * 100)
        
        # Average of both scores
        return (completeness_score + consistency_score) / 2
    
    def get_data_summary(self) -> Dict[str, Any]:
        """Get comprehensive data summary"""
        if self.data is None:
            return {}
        
        summary = {
            'shape': self.data.shape,
            'dtypes': self.data.dtypes,
            'missing_values': self.data.isnull().sum(),
            'memory_usage': self.data.memory_usage(deep=True),
            'numeric_columns': self.data.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': self.data.select_dtypes(include=['object']).columns.tolist(),
            'datetime_columns': self.data.select_dtypes(include=['datetime64']).columns.tolist()
        }
        
        return summary
    
    def get_data_preview(self, n_rows: int = 10, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Get data preview with specified rows and columns"""
        if self.data is None:
            return pd.DataFrame()
        
        preview_data = self.data.copy()
        
        if columns:
            available_columns = [col for col in columns if col in preview_data.columns]
            preview_data = preview_data[available_columns]
        
        return preview_data.head(n_rows)
    
    def get_descriptive_stats(self) -> pd.DataFrame:
        """Get descriptive statistics for numeric columns"""
        if self.data is None:
            return pd.DataFrame()
        
        numeric_data = self.data.select_dtypes(include=[np.number])
        return numeric_data.describe()
    
    def execute_sql_query(self, query: str) -> pd.DataFrame:
        """
        Execute SQL query on the data using DuckDB
        
        Args:
            query: SQL query string
            
        Returns:
            pd.DataFrame: Query results
        """
        try:
            result = self.conn.execute(query).fetchdf()
            return result
        except Exception as e:
            print(f"Error executing SQL query: {str(e)}")
            return pd.DataFrame()
    
    def get_column_info(self, column_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific column"""
        if self.data is None or column_name not in self.data.columns:
            return {}
        
        column_data = self.data[column_name]
        
        info = {
            'name': column_name,
            'dtype': str(column_data.dtype),
            'count': len(column_data),
            'null_count': column_data.isnull().sum(),
            'unique_count': column_data.nunique(),
            'memory_usage': column_data.memory_usage(deep=True)
        }
        
        if column_data.dtype in ['int64', 'float64']:
            info.update({
                'min': column_data.min(),
                'max': column_data.max(),
                'mean': column_data.mean(),
                'std': column_data.std(),
                'median': column_data.median()
            })
        elif column_data.dtype == 'object':
            info.update({
                'most_common': column_data.value_counts().head(5).to_dict(),
                'sample_values': column_data.dropna().head(10).tolist()
            })
        
        return info
    
    def export_to_csv(self) -> str:
        """Export current data to CSV format"""
        if self.data is None:
            return ""
        
        csv_buffer = io.StringIO()
        self.data.to_csv(csv_buffer, index=False)
        return csv_buffer.getvalue()
    
    def get_data_relationships(self) -> Dict[str, Any]:
        """Analyze potential relationships between columns"""
        if self.data is None:
            return {}
        
        relationships = {
            'correlations': {},
            'potential_keys': [],
            'foreign_keys': []
        }
        
        # Calculate correlations for numeric columns
        numeric_data = self.data.select_dtypes(include=[np.number])
        if not numeric_data.empty:
            corr_matrix = numeric_data.corr()
            relationships['correlations'] = corr_matrix.to_dict()
        
        # Identify potential primary keys (columns with unique values)
        for col in self.data.columns:
            if self.data[col].nunique() == len(self.data):
                relationships['potential_keys'].append(col)
        
        # Identify potential foreign keys (columns that might reference other tables)
        for col in self.data.columns:
            if col.endswith('_id') and col != 'id':
                relationships['foreign_keys'].append(col)
        
        return relationships
    
    def close_connection(self):
        """Close DuckDB connection"""
        if self.conn:
            self.conn.close()
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        self.close_connection()
