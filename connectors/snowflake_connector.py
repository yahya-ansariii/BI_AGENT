"""
Snowflake Connector
Handles Snowflake database connections and operations
"""

import pandas as pd
from typing import Optional, Dict, Any, List
import json
from datetime import datetime

try:
    import snowflake.connector
    from snowflake.connector import DictCursor
    SNOWFLAKE_AVAILABLE = True
except ImportError:
    SNOWFLAKE_AVAILABLE = False
    print("Warning: Snowflake connector not available. Install with: pip install snowflake-connector-python")

class SnowflakeConnector:
    """Handles Snowflake database connections and operations"""
    
    def __init__(self):
        """Initialize Snowflake connector"""
        self.connection = None
        self.available = SNOWFLAKE_AVAILABLE
        
    def connect(self, account: str, user: str, password: str, 
                warehouse: str, database: str, schema: str = 'PUBLIC',
                role: str = None) -> bool:
        """
        Connect to Snowflake database
        
        Args:
            account: Snowflake account identifier
            user: Username
            password: Password
            warehouse: Warehouse name
            database: Database name
            schema: Schema name (default: PUBLIC)
            role: Role name (optional)
            
        Returns:
            bool: True if connection successful
        """
        if not self.available:
            print("Snowflake connector not available")
            return False
        
        try:
            connection_params = {
                'account': account,
                'user': user,
                'password': password,
                'warehouse': warehouse,
                'database': database,
                'schema': schema
            }
            
            if role:
                connection_params['role'] = role
            
            self.connection = snowflake.connector.connect(**connection_params)
            return True
            
        except Exception as e:
            print(f"Error connecting to Snowflake: {str(e)}")
            return False
    
    def disconnect(self):
        """Disconnect from Snowflake"""
        if self.connection:
            try:
                self.connection.close()
                self.connection = None
            except Exception as e:
                print(f"Error disconnecting from Snowflake: {str(e)}")
    
    def execute_query(self, query: str) -> pd.DataFrame:
        """
        Execute SQL query and return results as DataFrame
        
        Args:
            query: SQL query string
            
        Returns:
            DataFrame with query results
        """
        if not self.connection:
            print("Not connected to Snowflake")
            return pd.DataFrame()
        
        try:
            cursor = self.connection.cursor(DictCursor)
            cursor.execute(query)
            
            # Fetch results
            results = cursor.fetchall()
            
            # Convert to DataFrame
            if results:
                df = pd.DataFrame(results)
            else:
                df = pd.DataFrame()
            
            cursor.close()
            return df
            
        except Exception as e:
            print(f"Error executing query: {str(e)}")
            return pd.DataFrame()
    
    def get_tables(self, database: str = None, schema: str = None) -> List[Dict[str, str]]:
        """
        Get list of tables in database/schema
        
        Args:
            database: Database name (optional)
            schema: Schema name (optional)
            
        Returns:
            List of table information dictionaries
        """
        if not self.connection:
            return []
        
        try:
            query = """
            SELECT TABLE_CATALOG as database_name,
                   TABLE_SCHEMA as schema_name,
                   TABLE_NAME as table_name,
                   TABLE_TYPE as table_type
            FROM INFORMATION_SCHEMA.TABLES
            WHERE TABLE_TYPE = 'BASE TABLE'
            """
            
            if database:
                query += f" AND TABLE_CATALOG = '{database}'"
            if schema:
                query += f" AND TABLE_SCHEMA = '{schema}'"
            
            query += " ORDER BY TABLE_CATALOG, TABLE_SCHEMA, TABLE_NAME"
            
            df = self.execute_query(query)
            return df.to_dict('records')
            
        except Exception as e:
            print(f"Error getting tables: {str(e)}")
            return []
    
    def get_table_schema(self, table_name: str, database: str = None, 
                        schema: str = None) -> List[Dict[str, Any]]:
        """
        Get schema information for a table
        
        Args:
            table_name: Name of the table
            database: Database name (optional)
            schema: Schema name (optional)
            
        Returns:
            List of column information dictionaries
        """
        if not self.connection:
            return []
        
        try:
            query = """
            SELECT COLUMN_NAME as column_name,
                   DATA_TYPE as data_type,
                   IS_NULLABLE as is_nullable,
                   COLUMN_DEFAULT as column_default,
                   COMMENT as comment
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_NAME = %s
            """
            
            params = [table_name]
            
            if database:
                query += " AND TABLE_CATALOG = %s"
                params.append(database)
            if schema:
                query += " AND TABLE_SCHEMA = %s"
                params.append(schema)
            
            query += " ORDER BY ORDINAL_POSITION"
            
            cursor = self.connection.cursor()
            cursor.execute(query, params)
            results = cursor.fetchall()
            cursor.close()
            
            columns = []
            for row in results:
                columns.append({
                    'column_name': row[0],
                    'data_type': row[1],
                    'is_nullable': row[2] == 'YES',
                    'column_default': row[3],
                    'comment': row[4]
                })
            
            return columns
            
        except Exception as e:
            print(f"Error getting table schema: {str(e)}")
            return []
    
    def get_table_sample(self, table_name: str, limit: int = 100,
                        database: str = None, schema: str = None) -> pd.DataFrame:
        """
        Get sample data from a table
        
        Args:
            table_name: Name of the table
            limit: Number of rows to return
            database: Database name (optional)
            schema: Schema name (optional)
            
        Returns:
            DataFrame with sample data
        """
        if not self.connection:
            return pd.DataFrame()
        
        try:
            full_table_name = table_name
            if database and schema:
                full_table_name = f"{database}.{schema}.{table_name}"
            elif schema:
                full_table_name = f"{schema}.{table_name}"
            
            query = f"SELECT * FROM {full_table_name} LIMIT {limit}"
            return self.execute_query(query)
            
        except Exception as e:
            print(f"Error getting table sample: {str(e)}")
            return pd.DataFrame()
    
    def get_table_stats(self, table_name: str, database: str = None,
                       schema: str = None) -> Dict[str, Any]:
        """
        Get statistics for a table
        
        Args:
            table_name: Name of the table
            database: Database name (optional)
            schema: Schema name (optional)
            
        Returns:
            Dictionary with table statistics
        """
        if not self.connection:
            return {}
        
        try:
            full_table_name = table_name
            if database and schema:
                full_table_name = f"{database}.{schema}.{table_name}"
            elif schema:
                full_table_name = f"{schema}.{table_name}"
            
            # Get row count
            count_query = f"SELECT COUNT(*) as row_count FROM {full_table_name}"
            count_result = self.execute_query(count_query)
            row_count = count_result['ROW_COUNT'].iloc[0] if not count_result.empty else 0
            
            # Get table size (approximate)
            size_query = f"""
            SELECT BYTES, ROW_COUNT
            FROM INFORMATION_SCHEMA.TABLES
            WHERE TABLE_NAME = '{table_name}'
            """
            if database:
                size_query += f" AND TABLE_CATALOG = '{database}'"
            if schema:
                size_query += f" AND TABLE_SCHEMA = '{schema}'"
            
            size_result = self.execute_query(size_query)
            table_size = size_result['BYTES'].iloc[0] if not size_result.empty else 0
            
            stats = {
                'table_name': table_name,
                'row_count': row_count,
                'table_size_bytes': table_size,
                'table_size_mb': table_size / (1024 * 1024) if table_size else 0,
                'last_updated': datetime.now().isoformat()
            }
            
            return stats
            
        except Exception as e:
            print(f"Error getting table stats: {str(e)}")
            return {}
    
    def test_connection(self) -> bool:
        """
        Test the current connection
        
        Returns:
            bool: True if connection is working
        """
        if not self.connection:
            return False
        
        try:
            result = self.execute_query("SELECT 1 as test")
            return not result.empty and result['TEST'].iloc[0] == 1
        except:
            return False
    
    def __del__(self):
        """Cleanup connection on object destruction"""
        self.disconnect()
