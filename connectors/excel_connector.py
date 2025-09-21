"""
Excel Connector
Handles Excel file operations and data extraction
"""

import pandas as pd
import openpyxl
import duckdb
from typing import Optional, Dict, Any, List
from pathlib import Path
import io

class ExcelConnector:
    """Handles Excel file connections and data operations"""
    
    def __init__(self):
        """Initialize Excel connector"""
        self.supported_formats = ['.xlsx', '.xls']
        self.engine_map = {
            '.xlsx': 'openpyxl',
            '.xls': 'xlrd'
        }
        self.conn = duckdb.connect(':memory:')
    
    def connect(self, file_path: str) -> bool:
        """
        Connect to Excel file
        
        Args:
            file_path: Path to Excel file
            
        Returns:
            bool: True if connection successful
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            if file_path.suffix.lower() not in self.supported_formats:
                raise ValueError(f"Unsupported format: {file_path.suffix}")
            
            # Test file accessibility
            self._test_file_access(file_path)
            return True
            
        except Exception as e:
            print(f"Error connecting to Excel file: {str(e)}")
            return False
    
    def _test_file_access(self, file_path: Path):
        """Test if file can be accessed"""
        try:
            with open(file_path, 'rb') as f:
                f.read(1024)  # Read first 1KB
        except Exception as e:
            raise Exception(f"Cannot access file: {str(e)}")
    
    def get_sheet_names(self, file_path: str) -> List[str]:
        """
        Get list of sheet names from Excel file
        
        Args:
            file_path: Path to Excel file
            
        Returns:
            List of sheet names
        """
        try:
            file_path = Path(file_path)
            engine = self.engine_map.get(file_path.suffix.lower(), 'openpyxl')
            
            if engine == 'openpyxl':
                workbook = openpyxl.load_workbook(file_path, read_only=True)
                return workbook.sheetnames
            else:
                # For .xls files
                xl_file = pd.ExcelFile(file_path, engine=engine)
                return xl_file.sheet_names
                
        except Exception as e:
            print(f"Error getting sheet names: {str(e)}")
            return []
    
    def read_sheet(self, file_path: str, sheet_name: str = None, 
                   header_row: int = 0, nrows: int = None) -> pd.DataFrame:
        """
        Read data from Excel sheet
        
        Args:
            file_path: Path to Excel file
            sheet_name: Name of sheet to read (None for first sheet)
            header_row: Row number to use as header
            nrows: Number of rows to read (None for all)
            
        Returns:
            DataFrame with sheet data
        """
        try:
            file_path = Path(file_path)
            engine = self.engine_map.get(file_path.suffix.lower(), 'openpyxl')
            
            read_params = {
                'filepath_or_buffer': file_path,
                'engine': engine,
                'header': header_row,
                'nrows': nrows
            }
            
            if sheet_name:
                read_params['sheet_name'] = sheet_name
            
            df = pd.read_excel(**read_params)
            return df
            
        except Exception as e:
            print(f"Error reading Excel sheet: {str(e)}")
            return pd.DataFrame()
    
    def read_all_sheets(self, file_path: str) -> Dict[str, pd.DataFrame]:
        """
        Read all sheets from Excel file
        
        Args:
            file_path: Path to Excel file
            
        Returns:
            Dictionary mapping sheet names to DataFrames
        """
        try:
            file_path = Path(file_path)
            engine = self.engine_map.get(file_path.suffix.lower(), 'openpyxl')
            
            all_sheets = pd.read_excel(file_path, engine=engine, sheet_name=None)
            return all_sheets
            
        except Exception as e:
            print(f"Error reading all sheets: {str(e)}")
            return {}
    
    def get_sheet_info(self, file_path: str, sheet_name: str) -> Dict[str, Any]:
        """
        Get information about a specific sheet
        
        Args:
            file_path: Path to Excel file
            sheet_name: Name of sheet
            
        Returns:
            Dictionary with sheet information
        """
        try:
            df = self.read_sheet(file_path, sheet_name)
            
            info = {
                'sheet_name': sheet_name,
                'rows': len(df),
                'columns': len(df.columns),
                'column_names': list(df.columns),
                'dtypes': df.dtypes.to_dict(),
                'memory_usage': df.memory_usage(deep=True).sum(),
                'null_counts': df.isnull().sum().to_dict(),
                'sample_data': df.head(5).to_dict('records')
            }
            
            return info
            
        except Exception as e:
            print(f"Error getting sheet info: {str(e)}")
            return {}
    
    def validate_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate Excel data quality
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dictionary with validation results
        """
        validation = {
            'is_valid': True,
            'issues': [],
            'warnings': [],
            'quality_score': 100.0
        }
        
        # Check for empty DataFrame
        if df.empty:
            validation['is_valid'] = False
            validation['issues'].append("DataFrame is empty")
            validation['quality_score'] = 0.0
            return validation
        
        # Check for completely empty columns
        empty_columns = df.columns[df.isnull().all()].tolist()
        if empty_columns:
            validation['warnings'].append(f"Empty columns found: {empty_columns}")
            validation['quality_score'] -= len(empty_columns) * 5
        
        # Check for duplicate rows
        duplicate_rows = df.duplicated().sum()
        if duplicate_rows > 0:
            validation['warnings'].append(f"Duplicate rows found: {duplicate_rows}")
            validation['quality_score'] -= min(duplicate_rows * 2, 20)
        
        # Check for missing values
        missing_pct = (df.isnull().sum().sum() / df.size) * 100
        if missing_pct > 50:
            validation['issues'].append(f"High missing data percentage: {missing_pct:.1f}%")
            validation['quality_score'] -= missing_pct * 0.5
        elif missing_pct > 20:
            validation['warnings'].append(f"Moderate missing data: {missing_pct:.1f}%")
            validation['quality_score'] -= missing_pct * 0.2
        
        # Check for consistent data types
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check if column should be numeric
                try:
                    pd.to_numeric(df[col].dropna())
                    validation['warnings'].append(f"Column '{col}' might be numeric but stored as text")
                except:
                    pass
        
        validation['quality_score'] = max(0, validation['quality_score'])
        
        return validation
    
    def get_schema(self, file_input) -> Dict[str, Any]:
        """
        Get schema information from Excel file
        
        Args:
            file_input: Path to Excel file or Streamlit UploadedFile object
            
        Returns:
            Dictionary with table names and columns
        """
        try:
            # Handle Streamlit UploadedFile objects
            if hasattr(file_input, 'read'):
                # This is a Streamlit UploadedFile object
                file_input.seek(0)  # Reset file pointer
                file_data = file_input.read()
                
                # Create a temporary file-like object
                import tempfile
                import os
                
                # Create temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_file:
                    tmp_file.write(file_data)
                    tmp_file_path = tmp_file.name
                
                try:
                    # Get all sheet names
                    sheet_names = self.get_sheet_names(tmp_file_path)
                    schema = {}
                    
                    for sheet_name in sheet_names:
                        # Read sheet to get column information
                        df = self.read_sheet(tmp_file_path, sheet_name)
                        
                        # Get column information
                        columns = []
                        for col in df.columns:
                            columns.append({
                                'name': col,
                                'type': str(df[col].dtype),
                                'nullable': df[col].isnull().any()
                            })
                        
                        schema[sheet_name] = columns
                    
                    return schema
                    
                finally:
                    # Clean up temporary file
                    if os.path.exists(tmp_file_path):
                        os.unlink(tmp_file_path)
            
            else:
                # This is a regular file path
                file_path = Path(file_input)
                if not file_path.exists():
                    raise FileNotFoundError(f"File not found: {file_path}")
                
                # Get all sheet names
                sheet_names = self.get_sheet_names(str(file_path))
                schema = {}
                
                for sheet_name in sheet_names:
                    # Read sheet to get column information
                    df = self.read_sheet(str(file_path), sheet_name)
                    
                    # Get column information
                    columns = []
                    for col in df.columns:
                        columns.append({
                            'name': col,
                            'type': str(df[col].dtype),
                            'nullable': df[col].isnull().any()
                        })
                    
                    schema[sheet_name] = columns
                
                return schema
            
        except Exception as e:
            print(f"Error getting schema: {str(e)}")
            return {}
    
    def run_query(self, file_input, sql: str) -> pd.DataFrame:
        """
        Run SQL query on Excel file using DuckDB
        
        Args:
            file_input: Path to Excel file or Streamlit UploadedFile object
            sql: SQL query string
            
        Returns:
            DataFrame with query results
        """
        try:
            # Handle Streamlit UploadedFile objects
            if hasattr(file_input, 'read'):
                # This is a Streamlit UploadedFile object
                file_input.seek(0)  # Reset file pointer
                file_data = file_input.read()
                
                # Create a temporary file-like object
                import tempfile
                import os
                
                # Create temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_file:
                    tmp_file.write(file_data)
                    tmp_file_path = tmp_file.name
                
                try:
                    # Read all sheets and register with DuckDB
                    all_sheets = self.read_all_sheets(tmp_file_path)
                    
                    # Clear existing tables
                    self.conn.execute("DROP TABLE IF EXISTS sales")
                    self.conn.execute("DROP TABLE IF EXISTS web_traffic")
                    
                    # Register each sheet as a table
                    for sheet_name, df in all_sheets.items():
                        table_name = sheet_name.lower().replace(' ', '_')
                        self.conn.register(table_name, df)
                    
                    # Execute SQL query
                    result = self.conn.execute(sql).fetchdf()
                    
                    return result
                    
                finally:
                    # Clean up temporary file
                    if os.path.exists(tmp_file_path):
                        os.unlink(tmp_file_path)
            
            else:
                # This is a regular file path
                file_path = Path(file_input)
                if not file_path.exists():
                    raise FileNotFoundError(f"File not found: {file_path}")
                
                # Read all sheets and register with DuckDB
                all_sheets = self.read_all_sheets(str(file_path))
                
                # Clear existing tables
                self.conn.execute("DROP TABLE IF EXISTS sales")
                self.conn.execute("DROP TABLE IF EXISTS web_traffic")
                
                # Register each sheet as a table
                for sheet_name, df in all_sheets.items():
                    table_name = sheet_name.lower().replace(' ', '_')
                    self.conn.register(table_name, df)
                
                # Execute SQL query
                result = self.conn.execute(sql).fetchdf()
                
                return result
            
        except Exception as e:
            print(f"Error running query: {str(e)}")
            return pd.DataFrame()
    
    def export_to_excel(self, data: Dict[str, pd.DataFrame], 
                       output_path: str) -> bool:
        """
        Export data to Excel file
        
        Args:
            data: Dictionary mapping sheet names to DataFrames
            output_path: Path for output Excel file
            
        Returns:
            bool: True if successful
        """
        try:
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                for sheet_name, df in data.items():
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            return True
            
        except Exception as e:
            print(f"Error exporting to Excel: {str(e)}")
            return False
