"""
LLM Agent Module
Handles AI-powered data analysis using direct Ollama API calls
"""

import pandas as pd
import json
import requests
import os
from typing import Optional, Dict, Any, List
from datetime import datetime

class BusinessInsightsParser:
    """Custom parser for business insights output"""
    
    def parse(self, text: str) -> Dict[str, Any]:
        """Parse LLM output into structured format"""
        try:
            # Try to extract JSON from the response
            if "```json" in text:
                json_start = text.find("```json") + 7
                json_end = text.find("```", json_start)
                json_str = text[json_start:json_end].strip()
                return json.loads(json_str)
            elif "{" in text and "}" in text:
                # Try to find JSON-like structure
                start = text.find("{")
                end = text.rfind("}") + 1
                json_str = text[start:end]
                return json.loads(json_str)
            else:
                # Return as text analysis
                return {
                    "insights": text,
                    "confidence": 0.8,
                    "type": "text_analysis"
                }
        except Exception:
            return {
                "insights": text,
                "confidence": 0.6,
                "type": "text_analysis"
            }

class LLMAgent:
    """AI agent for business data analysis using direct Ollama API calls"""
    
    def __init__(self):
        """Initialize the LLM agent"""
        self.model_name = None
        self.insights_history = []
        self.ollama_base_url = "http://localhost:11434"
        self.parser = BusinessInsightsParser()
    
    def is_initialized(self) -> bool:
        """Check if the LLM agent is initialized"""
        return hasattr(self, 'model_name') and self.model_name is not None
        
    def initialize_model(self, model_name: str = "llama2") -> bool:
        """
        Initialize the Ollama model
        
        Args:
            model_name: Name of the Ollama model to use
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Check if Ollama is running
            if not self._check_ollama_connection():
                raise Exception("Ollama is not running. Please start Ollama first.")
            
            # Check if model is available
            if not self._check_model_availability(model_name):
                raise Exception(f"Model {model_name} is not available. Please pull it first.")
            
            self.model_name = model_name
            return True
            
        except Exception as e:
            print(f"Error initializing model: {str(e)}")
            return False
    
    def _check_ollama_connection(self) -> bool:
        """Check if Ollama is running"""
        try:
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def _check_model_availability(self, model_name: str) -> bool:
        """Check if model is available"""
        try:
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                return any(model['name'].startswith(model_name) for model in models)
            return False
        except:
            return False
    
    def _serialize_dataframe_for_json(self, df: pd.DataFrame) -> str:
        """Convert DataFrame to JSON-serializable string format"""
        try:
            # Create a copy to avoid modifying original
            df_copy = df.copy()
            
            # Convert Timestamp and other non-serializable objects to strings
            for col in df_copy.columns:
                if df_copy[col].dtype == 'datetime64[ns]':
                    df_copy[col] = df_copy[col].dt.strftime('%Y-%m-%d %H:%M:%S')
                elif df_copy[col].dtype == 'object':
                    # Convert any remaining non-serializable objects to strings
                    df_copy[col] = df_copy[col].astype(str)
            
            return str(df_copy.to_dict('records'))
        except Exception as e:
            return f"Error serializing data: {str(e)}"
    
    def list_models(self) -> List[str]:
        """List available models"""
        try:
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                return [model['name'] for model in models]
            return []
        except:
            return []
    
    def analyze_data(self, query: str, data: pd.DataFrame) -> str:
        """
        Analyze data based on user query
        
        Args:
            query: User's question about the data
            data: DataFrame to analyze
            
        Returns:
            str: Analysis results
        """
        try:
            # Prepare data summary for the LLM
            data_summary = self._prepare_data_summary(data)
            
            # Create prompt
            prompt = self._create_analysis_prompt(query, data_summary)
            
            # Call Ollama API
            response = self._call_ollama_api(prompt)
            
            # Store in history
            self.insights_history.append({
                "query": query,
                "response": response,
                "timestamp": datetime.now().isoformat(),
                "data_shape": data.shape
            })
            
            return response
            
        except Exception as e:
            error_msg = f"Error analyzing data: {str(e)}"
            print(error_msg)
            return error_msg
    
    def _prepare_data_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Prepare data summary for LLM analysis"""
        # Convert dtypes to strings for JSON serialization
        dtypes_dict = {col: str(dtype) for col, dtype in data.dtypes.items()}
        
        # Convert sample data to JSON-serializable format
        sample_data = data.head(5).copy()
        for col in sample_data.columns:
            if sample_data[col].dtype == 'datetime64[ns]':
                sample_data[col] = sample_data[col].dt.strftime('%Y-%m-%d %H:%M:%S')
            elif sample_data[col].dtype == 'object':
                sample_data[col] = sample_data[col].astype(str)
        
        summary = {
            "shape": data.shape,
            "columns": list(data.columns),
            "dtypes": dtypes_dict,
            "numeric_columns": data.select_dtypes(include=['number']).columns.tolist(),
            "categorical_columns": data.select_dtypes(include=['object']).columns.tolist(),
            "missing_values": data.isnull().sum().to_dict(),
            "sample_data": sample_data.to_dict('records')
        }
        
        # Add basic statistics for numeric columns
        numeric_data = data.select_dtypes(include=['number'])
        if not numeric_data.empty:
            summary["statistics"] = numeric_data.describe().to_dict()
        
        return summary
    
    def _create_analysis_prompt(self, query: str, data_summary: Dict[str, Any]) -> str:
        """Create analysis prompt for the LLM"""
        prompt = f"""
You are a business intelligence analyst. Analyze the following data based on the user's query.

Data Summary:
- Shape: {data_summary['shape']}
- Columns: {data_summary['columns']}
- Data Types: {data_summary['dtypes']}
- Missing Values: {data_summary['missing_values']}

Sample Data (first 5 rows):
{json.dumps(data_summary['sample_data'], indent=2)}

User Query: {query}

Please provide:
1. A clear analysis of the data related to the query
2. Key insights and patterns you observe
3. Specific numbers and statistics when relevant
4. Business recommendations based on the analysis
5. Any data quality issues you notice

Format your response as a structured analysis with clear sections.
"""
        return prompt
    
    def _call_ollama_api(self, prompt: str) -> str:
        """Call Ollama API directly with improved error handling"""
        try:
            response = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "max_tokens": 2000
                    }
                },
                timeout=120  # Increased timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', 'No response generated')
            else:
                return f"API Error: {response.status_code} - {response.text}"
                
        except requests.exceptions.Timeout:
            return "AI analysis timed out. Please try with a simpler query or check if Ollama is running properly."
        except requests.exceptions.ConnectionError:
            return "Cannot connect to Ollama. Please ensure Ollama is running (ollama serve)."
        except Exception as e:
            return f"API call failed: {str(e)}"
    
    def get_quick_insights(self, data: pd.DataFrame) -> str:
        """Get quick insights about the data"""
        if data.empty:
            return "No data available for analysis."
        
        insights = []
        
        # Basic data info
        insights.append(f"Dataset contains {data.shape[0]} rows and {data.shape[1]} columns.")
        
        # Missing values
        missing = data.isnull().sum().sum()
        if missing > 0:
            insights.append(f"Found {missing} missing values across the dataset.")
        else:
            insights.append("No missing values found - data quality looks good!")
        
        # Numeric columns analysis
        numeric_cols = data.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            insights.append(f"Found {len(numeric_cols)} numeric columns: {', '.join(numeric_cols)}")
            
            # Basic stats for first numeric column
            if len(numeric_cols) > 0:
                first_col = numeric_cols[0]
                col_data = data[first_col].dropna()
                if not col_data.empty:
                    insights.append(f"Sample stats for '{first_col}': min={col_data.min():.2f}, max={col_data.max():.2f}, mean={col_data.mean():.2f}")
        
        # Categorical columns
        cat_cols = data.select_dtypes(include=['object']).columns
        if len(cat_cols) > 0:
            insights.append(f"Found {len(cat_cols)} categorical columns: {', '.join(cat_cols)}")
        
        return "\n".join(insights)
    
    def sales_analysis(self, data: pd.DataFrame) -> str:
        """Perform sales analysis on the data"""
        prompt = """
Analyze this data for sales insights. Look for:
1. Sales trends and patterns
2. Top performing products/categories
3. Seasonal variations
4. Customer behavior patterns
5. Revenue analysis
6. Growth opportunities

Provide specific recommendations for improving sales performance.
"""
        return self.analyze_data(prompt, data)
    
    def customer_analysis(self, data: pd.DataFrame) -> str:
        """Perform customer analysis on the data"""
        prompt = """
Analyze this data for customer insights. Look for:
1. Customer segmentation opportunities
2. Customer lifetime value patterns
3. Churn risk indicators
4. Customer acquisition trends
5. Behavioral patterns
6. Retention strategies

Provide specific recommendations for customer engagement and retention.
"""
        return self.analyze_data(prompt, data)
    
    def get_insights_history(self) -> List[Dict[str, Any]]:
        """Get the history of insights generated"""
        return self.insights_history
    
    def clear_history(self):
        """Clear the insights history"""
        self.insights_history = []
    
    def generate_sql_query(self, question: str, data: pd.DataFrame) -> str:
        """Generate SQL query from natural language question"""
        try:
            # Get column information
            columns = list(data.columns)
            dtypes = {col: str(dtype) for col, dtype in data.dtypes.items()}
            
            # Create SQL generation prompt
            prompt = f"""
            You are a SQL expert. Generate a SQL query to answer this business question.
            
            Question: {question}
            
            Available data:
            - Table name: 'data'
            - Columns: {', '.join(columns)}
            - Column types: {dtypes}
            
            Sample data (first 3 rows):
            {self._serialize_dataframe_for_json(data.head(3))}
            
            Generate a SQL query that:
            1. Answers the question accurately and directly
            2. Uses proper SQL syntax for DuckDB
            3. Handles data types correctly
            4. ONLY includes WHERE clauses if explicitly mentioned in the question
            5. ONLY includes date filters if the question specifically asks for a date range
            6. ONLY includes payment method or other filters if explicitly mentioned
            7. Uses appropriate GROUP BY, ORDER BY, and LIMIT clauses as needed
            8. Focuses on the core question without adding unnecessary filtering
            
            IMPORTANT: Do not add filters that are not explicitly mentioned in the question.
            If the question asks for "total sales by region", do NOT add payment method or date filters.
            
            Return ONLY the SQL query, no explanations.
            """
            
            # Call Ollama API
            response = self._call_ollama_api(prompt)
            
            # Clean up the response to extract just the SQL
            sql_query = response.strip()
            if '```sql' in sql_query:
                sql_query = sql_query.split('```sql')[1].split('```')[0].strip()
            elif '```' in sql_query:
                sql_query = sql_query.split('```')[1].split('```')[0].strip()
            
            return sql_query
            
        except Exception as e:
            return f"SELECT * FROM data LIMIT 10; -- Error generating SQL: {str(e)}"

    def generate_sql_query_with_tables(self, question: str, tables: Dict[str, pd.DataFrame], relationships: List[Dict] = None) -> str:
        """Generate SQL query from natural language question using multiple tables and relationships"""
        try:
            # Prepare table information
            table_info = {}
            for table_name, df in tables.items():
                # Get column information and structure
                columns = list(df.columns)
                dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
                
                # Categorize columns by type
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                text_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
                datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
                
                table_info[table_name] = {
                    'table_name': table_name,
                    'columns': columns,
                    'dtypes': dtypes,
                    'row_count': len(df),
                    'numeric_columns': numeric_cols,
                    'text_columns': text_cols,
                    'datetime_columns': datetime_cols,
                    'column_info': {
                        col: {
                            'type': str(dtypes[col]),
                            'is_numeric': col in numeric_cols,
                            'is_text': col in text_cols,
                            'is_datetime': col in datetime_cols
                        } for col in columns
                    }
                }
            
            # Prepare relationships information
            relationships_info = ""
            if relationships:
                relationships_info = f"""
            
            Table Relationships:
            {json.dumps(relationships, indent=2)}
            
            Use these relationships to create proper JOINs when the question requires data from multiple tables.
            """
            
            # Create comprehensive SQL generation prompt
            prompt = f"""
            You are a SQL expert. Generate a SQL query to answer this business question using the available tables.
            
            Question: {question}
            
            Available Tables:
            {json.dumps(table_info, indent=2, default=str)}{relationships_info}
            
            Instructions:
            1. Use the ACTUAL table names provided (not generic names like 'data')
            2. Use the ACTUAL column names from the tables
            3. Generate a SQL query that answers the question accurately
            4. Use proper SQL syntax for DuckDB
            5. Handle data types correctly based on the column types shown
            6. Use JOINs if the question requires data from multiple tables
            7. Use the provided relationships to create proper JOINs between tables
            8. Use appropriate WHERE, GROUP BY, ORDER BY, and LIMIT clauses
            9. ONLY add filters that are explicitly mentioned in the question
            10. Focus on the core question without unnecessary filtering
            
            CRITICAL DATA TYPE RULES:
            - AVG(), SUM(), MIN(), MAX() only work on NUMERIC columns (int, float, decimal)
            - COUNT() works on any column type
            - For VARCHAR/TEXT columns, use COUNT() instead of AVG()
            - For date/time columns, use appropriate date functions
            - Check column data types before using aggregate functions
            - Look at the 'column_info' section to see which columns are numeric vs text
            - If a column is marked as 'is_text': true, do NOT use AVG() or SUM() on it
            - If a column is marked as 'is_numeric': true, you can use AVG(), SUM(), MIN(), MAX()
            
            Examples of good table usage:
            - If asking about "sales by product", use the actual table name like "sales" or "products"
            - If asking about "customer data", use the actual table name like "customers" or "customer_info"
            - Use actual column names like "product_name", "sales_amount", "customer_id", etc.
            - Use relationships to JOIN tables: table1.column = table2.column
            - For basic stats on VARCHAR columns: SELECT COUNT(*) as count, COUNT(DISTINCT column_name) as unique_count
            - For basic stats on NUMERIC columns: SELECT COUNT(*) as count, AVG(column_name) as avg_value, MIN(column_name) as min_value, MAX(column_name) as max_value
            
            Return ONLY the SQL query, no explanations.
            """
            
            # Call Ollama API
            response = self._call_ollama_api(prompt)
            
            # Clean up the response to extract just the SQL
            sql_query = response.strip()
            if '```sql' in sql_query:
                sql_query = sql_query.split('```sql')[1].split('```')[0].strip()
            elif '```' in sql_query:
                sql_query = sql_query.split('```')[1].split('```')[0].strip()
            
            return sql_query
            
        except Exception as e:
            return f"-- Error generating SQL query: {str(e)}"
    
    def get_quick_sql_insights(self, data: pd.DataFrame) -> str:
        """Generate quick SQL insights for the data"""
        try:
            columns = list(data.columns)
            dtypes = {col: str(dtype) for col, dtype in data.dtypes.items()}
            
            prompt = f"""
            Generate 3 useful SQL queries for this business data.
            
            Available data:
            - Columns: {', '.join(columns)}
            - Column types: {dtypes}
            
            Generate SQL queries for:
            1. Basic data overview
            2. Top performing items/categories
            3. Trends over time (if date columns exist)
            
            Return each query with a brief comment explaining what it does.
            """
            
            response = self._call_ollama_api(prompt)
            return response
            
        except Exception as e:
            return f"-- Error generating SQL insights: {str(e)}"
    
    def detect_relationships(self, tables: Dict[str, pd.DataFrame]) -> List[Dict]:
        """Detect potential relationships between tables using column name matching first, then AI with sample data"""
        try:
            print("Starting enhanced relationship detection...")
            
            # Step 1: Check for matching column names between tables
            direct_relationships = self._find_direct_column_matches(tables)
            print(f"Found {len(direct_relationships)} direct column matches")
            
            # Step 2: If no direct matches found, use AI with sample data
            ai_relationships = []
            if not direct_relationships:
                print("No direct matches found, using AI with sample data...")
                ai_relationships = self._detect_relationships_with_ai(tables)
                print(f"AI found {len(ai_relationships)} relationships")
            
            # Combine and validate all relationships
            all_relationships = direct_relationships + ai_relationships
            validated_relationships = self._validate_relationships(all_relationships, tables)
            
            print(f"Total validated relationships: {len(validated_relationships)}")
            return validated_relationships
            
        except Exception as e:
            print(f"Error detecting relationships: {str(e)}")
            return []
    
    def _find_direct_column_matches(self, tables: Dict[str, pd.DataFrame]) -> List[Dict]:
        """Find relationships based on matching column names between tables"""
        relationships = []
        table_names = list(tables.keys())
        
        for i, table1_name in enumerate(table_names):
            for j, table2_name in enumerate(table_names):
                if i >= j:  # Avoid duplicate comparisons
                    continue
                
                df1, df2 = tables[table1_name], tables[table2_name]
                
                # Get lowercase column names for comparison
                cols1_lower = {col.lower(): col for col in df1.columns}
                cols2_lower = {col.lower(): col for col in df2.columns}
                
                # Find matching column names (case-insensitive)
                common_cols = set(cols1_lower.keys()) & set(cols2_lower.keys())
                
                for common_col in common_cols:
                    actual_col1 = cols1_lower[common_col]
                    actual_col2 = cols2_lower[common_col]
                    
                    # Check if both columns have compatible data types
                    if self._are_columns_compatible(df1[actual_col1], df2[actual_col2]):
                        # Determine relationship type based on uniqueness
                        rel_type = self._determine_relationship_type(df1[actual_col1], df2[actual_col2])
                        
                        relationship = {
                            'source_table': table1_name,
                            'target_table': table2_name,
                            'source_column': actual_col1,
                            'target_column': actual_col2,
                            'relationship_type': rel_type,
                            'confidence': 'High',
                            'reasoning': f'Direct column name match: {actual_col1} = {actual_col2}',
                            'detection_method': 'direct_match'
                        }
                        relationships.append(relationship)
        
        return relationships
    
    def _are_columns_compatible(self, col1: pd.Series, col2: pd.Series) -> bool:
        """Check if two columns are compatible for relationship"""
        # Check data type compatibility
        if col1.dtype != col2.dtype:
            # Allow some type conversions
            if not self._are_types_compatible(col1.dtype, col2.dtype):
                return False
        
        # Check if both columns have reasonable data
        if col1.isna().all() or col2.isna().all():
            return False
            
        return True
    
    def _are_types_compatible(self, dtype1, dtype2) -> bool:
        """Check if two data types are compatible for relationships"""
        # Numeric types are compatible
        if pd.api.types.is_numeric_dtype(dtype1) and pd.api.types.is_numeric_dtype(dtype2):
            return True
        
        # String types are compatible
        if pd.api.types.is_string_dtype(dtype1) and pd.api.types.is_string_dtype(dtype2):
            return True
        
        # Same exact type
        if dtype1 == dtype2:
            return True
            
        return False
    
    def _determine_relationship_type(self, col1: pd.Series, col2: pd.Series) -> str:
        """Determine relationship type based on column uniqueness"""
        # Check uniqueness (approximate)
        unique1 = col1.nunique()
        unique2 = col2.nunique()
        total1 = len(col1)
        total2 = len(col2)
        
        # If one column has mostly unique values, it's likely the primary key
        if unique1 / total1 > 0.8 and unique2 / total2 < 0.8:
            return "One-to-Many"  # col1 -> col2
        elif unique2 / total2 > 0.8 and unique1 / total1 < 0.8:
            return "Many-to-One"  # col1 -> col2
        elif unique1 / total1 > 0.8 and unique2 / total2 > 0.8:
            return "One-to-One"
        else:
            return "Many-to-Many"
    
    def _detect_relationships_with_ai(self, tables: Dict[str, pd.DataFrame]) -> List[Dict]:
        """Use AI to detect relationships with sample data"""
        # Prepare table information with sample data
        table_info = {}
        for table_name, df in tables.items():
            # Get column information
            columns = list(df.columns)
            dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
            
            # Categorize columns by type
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            text_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
            datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
            
            # Get sample data (5-10 records)
            sample_size = min(10, len(df))
            sample_data = df.head(sample_size).to_dict('records')
            
            # Identify potential key columns
            potential_keys = []
            for col in columns:
                col_lower = col.lower()
                # Look for common key patterns
                if any(pattern in col_lower for pattern in ['id', 'key', 'pk', 'primary']):
                    potential_keys.append(col)
                elif col_lower in ['id']:  # Simple 'id' column
                    potential_keys.append(col)
            
            table_info[table_name] = {
                'table_name': table_name,
                'columns': columns,
                'dtypes': dtypes,
                'row_count': len(df),
                'numeric_columns': numeric_cols,
                'text_columns': text_cols,
                'datetime_columns': datetime_cols,
                'potential_keys': potential_keys,
                'sample_data': sample_data  # Include sample data for AI analysis
            }
            
        # Create relationship detection prompt with sample data
        prompt = f"""
        Analyze these database tables and suggest potential relationships based on their structure and sample data.
        
        Table Structure and Sample Data:
        {json.dumps(table_info, indent=2, default=str)}
        
        CRITICAL: You MUST only use column names that actually exist in the tables shown above. Do not invent or suggest column names that are not present in the table structure.
        
        Based on table names, column names, data types, potential key columns, and sample data, suggest relationships in this format:
        [
            {{
                "source_table": "table1",
                "target_table": "table2", 
                "source_column": "id",
                "target_column": "table1_id",
                "relationship_type": "One-to-Many",
                "confidence": "High",
                "reasoning": "Column naming pattern suggests foreign key relationship"
            }}
        ]
        
        VALID RELATIONSHIP TYPES (use only these):
        - "One-to-One": Each record in one table matches exactly one record in another
        - "One-to-Many": One record can match many records in another table
        - "Many-to-One": Many records can match one record in another table  
        - "Many-to-Many": Records can have multiple matches in both directions
        
        Focus on:
        - Primary key to foreign key relationships (id, _id, _key columns)
        - Common naming patterns (customer_id, product_id, user_id, etc.)
        - Data type compatibility (matching types between keys)
        - Logical business relationships based on table/column names
        - Potential keys identified in each table
        - Sample data patterns that suggest relationships
        
        VALIDATION RULES:
        1. source_table and target_table must exist in the table list above
        2. source_column must exist in the source_table's columns
        3. target_column must exist in the target_table's columns
        4. Only suggest relationships between tables that have logical connections
        5. Use exact column names as they appear in the table structure
        6. relationship_type MUST be one of: "One-to-One", "One-to-Many", "Many-to-One", "Many-to-Many"
        7. DO NOT use invalid relationship types like "Many-to-None", "None-to-Many", etc.
        
        Examples of good relationships (only if these exact columns exist):
        - users.id → orders.user_id (One-to-Many)
        - products.id → order_items.product_id (One-to-Many)
        - categories.id → products.category_id (One-to-Many)
        - customers.id → orders.customer_id (One-to-Many)
        
        IMPORTANT: Return ONLY the JSON array. Do not include any explanatory text, comments, or other content. Start your response with [ and end with ].
        """
        
        response = self._call_ollama_api(prompt)
        
        # Parse JSON response - handle cases where AI adds extra text
        try:
            # First try direct JSON parsing
            relationships = json.loads(response)
            if not isinstance(relationships, list):
                return []
        except json.JSONDecodeError:
            # Try to extract JSON from response if AI added extra text
            try:
                # Look for JSON array pattern in the response
                import re
                json_match = re.search(r'\[.*\]', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    relationships = json.loads(json_str)
                else:
                    print(f"Could not find JSON array in response: {response[:200]}...")
                    return []
            except (json.JSONDecodeError, AttributeError) as parse_err:
                print(f"JSON parsing error in relationship detection: {str(parse_err)}")
                print(f"Response was: {response[:200]}...")
                return []
        
        return relationships if isinstance(relationships, list) else []
    
    def _validate_relationships(self, relationships: List[Dict], tables: Dict[str, pd.DataFrame]) -> List[Dict]:
        """Validate relationships against actual table columns"""
        validated_relationships = []
        for rel in relationships:
            if not isinstance(rel, dict):
                continue
                
            # Check if all required fields exist
            required_fields = ['source_table', 'target_table', 'source_column', 'target_column']
            if not all(field in rel for field in required_fields):
                continue
            
            source_table = rel['source_table']
            target_table = rel['target_table']
            source_column = rel['source_column']
            target_column = rel['target_column']
            
            # Validate that tables exist
            if source_table not in tables or target_table not in tables:
                continue
            
            # Validate that columns exist in their respective tables
            source_columns = list(tables[source_table].columns)
            target_columns = list(tables[target_table].columns)
            
            if source_column not in source_columns or target_column not in target_columns:
                print(f"Skipping invalid relationship: {source_table}.{source_column} -> {target_table}.{target_column}")
                print(f"Available columns in {source_table}: {source_columns}")
                print(f"Available columns in {target_table}: {target_columns}")
                continue
            
            # Validate relationship type
            valid_relationship_types = ['One-to-One', 'One-to-Many', 'Many-to-One', 'Many-to-Many']
            relationship_type = rel.get('relationship_type', 'One-to-Many')
            
            if relationship_type not in valid_relationship_types:
                print(f"Skipping invalid relationship type: {relationship_type}")
                print(f"Valid types are: {valid_relationship_types}")
                continue
            
            # Add validation info to the relationship
            rel['validated'] = True
            rel['validation_note'] = "Column existence and relationship type verified"
            validated_relationships.append(rel)
        
        print(f"AI suggested {len(relationships)} relationships, validated {len(validated_relationships)} as valid")
        return validated_relationships

    def analyze_query_results(self, question: str, results: pd.DataFrame, source_tables: Dict = None) -> str:
        """Analyze SQL query results and provide enhanced insights using source tables"""
        try:
            # Prepare results summary - structure only
            numeric_cols = results.select_dtypes(include=['number']).columns.tolist()
            text_cols = results.select_dtypes(include=['object', 'string']).columns.tolist()
            datetime_cols = results.select_dtypes(include=['datetime64']).columns.tolist()
            
            results_summary = {
                "shape": results.shape,
                "columns": list(results.columns),
                "column_types": {
                    "numeric": numeric_cols,
                    "text": text_cols,
                    "datetime": datetime_cols
                },
                "numeric_summary": results.describe().to_dict() if not results.select_dtypes(include=['number']).empty else {}
            }
            
            # Prepare source tables context - structure only
            source_context = ""
            if source_tables:
                source_context = "\n\nSource Tables Context:\n"
                for table_name, table_df in source_tables.items():
                    numeric_cols = table_df.select_dtypes(include=['number']).columns.tolist()
                    text_cols = table_df.select_dtypes(include=['object', 'string']).columns.tolist()
                    datetime_cols = table_df.select_dtypes(include=['datetime64']).columns.tolist()
                    
                    source_context += f"\n{table_name} ({table_df.shape[0]} rows):\n"
                    source_context += f"- Columns: {', '.join(table_df.columns)}\n"
                    source_context += f"- Numeric columns: {', '.join(numeric_cols) if numeric_cols else 'None'}\n"
                    source_context += f"- Text columns: {', '.join(text_cols) if text_cols else 'None'}\n"
                    source_context += f"- Datetime columns: {', '.join(datetime_cols) if datetime_cols else 'None'}\n"
            
            # Create enhanced analysis prompt
            prompt = f"""
            You are a senior business analyst. Analyze these SQL query results and provide concise, professional business insights.
            
            Original Question: {question}
            
            Query Results:
            - Shape: {results_summary['shape']}
            - Columns: {', '.join(results_summary['columns'])}
            - Column Types: {results_summary['column_types']}
            - Numeric Summary: {results_summary['numeric_summary']}
            {source_context}
            
            Provide a concise analysis with:
            1. **Key Findings** (2-3 bullet points)
            2. **Business Impact** (what this means for the business)
            3. **Recommendations** (specific actionable next steps)
            
            Keep insights professional, data-driven, and actionable. Focus on business value.
            """
            
            # Call Ollama API
            response = self._call_ollama_api(prompt)
            return response
            
        except Exception as e:
            return f"Error analyzing query results: {str(e)}"