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
            {data.head(3).to_dict('records')}
            
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

    def generate_sql_query_with_tables(self, question: str, tables: Dict[str, pd.DataFrame]) -> str:
        """Generate SQL query from natural language question using multiple tables"""
        try:
            # Prepare table information
            table_info = {}
            for table_name, df in tables.items():
                table_info[table_name] = {
                    'columns': list(df.columns),
                    'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
                    'sample_data': df.head(2).to_dict('records'),
                    'row_count': len(df)
                }
            
            # Create comprehensive SQL generation prompt
            prompt = f"""
            You are a SQL expert. Generate a SQL query to answer this business question using the available tables.
            
            Question: {question}
            
            Available Tables:
            {json.dumps(table_info, indent=2, default=str)}
            
            Instructions:
            1. Use the ACTUAL table names provided (not generic names like 'data')
            2. Use the ACTUAL column names from the tables
            3. Generate a SQL query that answers the question accurately
            4. Use proper SQL syntax for DuckDB
            5. Handle data types correctly based on the column types shown
            6. Use JOINs if the question requires data from multiple tables
            7. Use appropriate WHERE, GROUP BY, ORDER BY, and LIMIT clauses
            8. ONLY add filters that are explicitly mentioned in the question
            9. Focus on the core question without unnecessary filtering
            
            Examples of good table usage:
            - If asking about "sales by product", use the actual table name like "sales" or "products"
            - If asking about "customer data", use the actual table name like "customers" or "customer_info"
            - Use actual column names like "product_name", "sales_amount", "customer_id", etc.
            
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
        """Detect potential relationships between tables using AI"""
        try:
            # Prepare table information
            table_info = {}
            for table_name, df in tables.items():
                table_info[table_name] = {
                    'columns': list(df.columns),
                    'sample_data': df.head(2).to_dict('records'),
                    'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()}
                }
            
            # Create relationship detection prompt
            prompt = f"""
            Analyze these database tables and suggest potential relationships between them.
            
            Tables:
            {json.dumps(table_info, indent=2)}
            
            Based on column names, data types, and sample data, suggest relationships in this format:
            [
                {{
                    "source_table": "table1",
                    "target_table": "table2", 
                    "source_column": "id",
                    "target_column": "table1_id",
                    "relationship_type": "One-to-Many",
                    "confidence": "High"
                }}
            ]
            
            Look for:
            - Primary key to foreign key relationships (id, _id, _key columns)
            - Common naming patterns (customer_id, product_id, etc.)
            - Data type compatibility
            - Logical business relationships
            
            Return only valid JSON array, no other text.
            """
            
            response = self._call_ollama_api(prompt)
            
            # Parse JSON response
            try:
                relationships = json.loads(response)
                return relationships if isinstance(relationships, list) else []
            except json.JSONDecodeError:
                return []
                
        except Exception as e:
            print(f"Error detecting relationships: {str(e)}")
            return []

    def analyze_query_results(self, question: str, results: pd.DataFrame, source_tables: Dict = None) -> str:
        """Analyze SQL query results and provide enhanced insights using source tables"""
        try:
            # Prepare results summary
            results_summary = {
                "shape": results.shape,
                "columns": list(results.columns),
                "sample_data": results.head(5).to_dict('records'),
                "numeric_summary": results.describe().to_dict() if not results.select_dtypes(include=['number']).empty else {}
            }
            
            # Prepare source tables context
            source_context = ""
            if source_tables:
                source_context = "\n\nSource Tables Context:\n"
                for table_name, table_df in source_tables.items():
                    source_context += f"\n{table_name} ({table_df.shape[0]} rows):\n"
                    source_context += f"- Columns: {', '.join(table_df.columns)}\n"
                    source_context += f"- Sample data: {table_df.head(2).to_dict('records')}\n"
            
            # Create enhanced analysis prompt
            prompt = f"""
            You are a senior business analyst. Analyze these SQL query results and provide concise, professional business insights.
            
            Original Question: {question}
            
            Query Results:
            - Shape: {results_summary['shape']}
            - Columns: {', '.join(results_summary['columns'])}
            - Sample Results: {results_summary['sample_data']}
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