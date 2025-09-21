"""
LLM Agent Module
Handles AI-powered data analysis using LangChain and Ollama
"""

import pandas as pd
import json
from typing import Optional, Dict, Any, List
import requests
import os
from datetime import datetime

try:
    from langchain.llms import Ollama
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
    from langchain.schema import BaseOutputParser
    from langchain.callbacks.manager import CallbackManager
    from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
except ImportError:
    print("Warning: LangChain not available. LLM features will be limited.")
    Ollama = None
    PromptTemplate = None
    LLMChain = None
    BaseOutputParser = None
    CallbackManager = None
    StreamingStdOutCallbackHandler = None

class BusinessInsightsParser(BaseOutputParser):
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
                json_start = text.find("{")
                json_end = text.rfind("}") + 1
                json_str = text[json_start:json_end]
                return json.loads(json_str)
            else:
                return {
                    "insights": text,
                    "timestamp": datetime.now().isoformat(),
                    "type": "text_analysis"
                }
        except Exception as e:
            return {
                "insights": text,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "type": "text_analysis"
            }

class LLMAgent:
    """AI agent for business data analysis using LangChain and Ollama"""
    
    def __init__(self):
        """Initialize the LLM agent"""
        self.llm = None
        self.model_name = None
        self.insights_history = []
        self.ollama_base_url = "http://localhost:11434"
        
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
            
            if Ollama is not None:
                # Initialize LangChain Ollama
                callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
                self.llm = Ollama(
                    model=model_name,
                    base_url=self.ollama_base_url,
                    callback_manager=callback_manager
                )
            else:
                # Fallback to direct API calls
                self.llm = "api_mode"
            
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
        """Check if the specified model is available"""
        try:
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                return any(model['name'].startswith(model_name) for model in models)
            return False
        except:
            return False
    
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
            
            if self.llm == "api_mode":
                # Use direct API call
                response = self._call_ollama_api(prompt)
            else:
                # Use LangChain
                response = self._call_langchain_llm(prompt)
            
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
        summary = {
            "shape": data.shape,
            "columns": list(data.columns),
            "dtypes": data.dtypes.to_dict(),
            "numeric_columns": data.select_dtypes(include=['number']).columns.tolist(),
            "categorical_columns": data.select_dtypes(include=['object']).columns.tolist(),
            "missing_values": data.isnull().sum().to_dict(),
            "sample_data": data.head(5).to_dict('records')
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
        """Call Ollama API directly"""
        try:
            response = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=60
            )
            
            if response.status_code == 200:
                return response.json().get('response', 'No response generated')
            else:
                return f"API Error: {response.status_code}"
                
        except Exception as e:
            return f"API call failed: {str(e)}"
    
    def _call_langchain_llm(self, prompt: str) -> str:
        """Call LLM using LangChain"""
        try:
            if self.llm is None:
                return "LLM not initialized"
            
            response = self.llm(prompt)
            return response
            
        except Exception as e:
            return f"LangChain call failed: {str(e)}"
    
    def get_quick_insights(self, data: pd.DataFrame) -> str:
        """Get quick insights about the data"""
        try:
            insights = []
            
            # Basic data info
            insights.append(f"📊 Dataset Overview:")
            insights.append(f"   • {data.shape[0]:,} records, {data.shape[1]} columns")
            insights.append(f"   • Memory usage: {data.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
            
            # Data quality
            missing_pct = (data.isnull().sum().sum() / data.size) * 100
            insights.append(f"   • Data completeness: {100 - missing_pct:.1f}%")
            
            # Numeric columns analysis
            numeric_cols = data.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                insights.append(f"\n📈 Numeric Analysis:")
                for col in numeric_cols[:3]:  # Show top 3 numeric columns
                    col_data = data[col].dropna()
                    if len(col_data) > 0:
                        insights.append(f"   • {col}: {col_data.mean():.2f} avg, {col_data.min():.2f}-{col_data.max():.2f} range")
            
            # Categorical columns analysis
            categorical_cols = data.select_dtypes(include=['object']).columns
            if len(categorical_cols) > 0:
                insights.append(f"\n📋 Categorical Analysis:")
                for col in categorical_cols[:3]:  # Show top 3 categorical columns
                    unique_count = data[col].nunique()
                    most_common = data[col].value_counts().head(1).index[0] if unique_count > 0 else "N/A"
                    insights.append(f"   • {col}: {unique_count} unique values, most common: {most_common}")
            
            return "\n".join(insights)
            
        except Exception as e:
            return f"Error generating quick insights: {str(e)}"
    
    def sales_analysis(self, data: pd.DataFrame) -> str:
        """Perform sales analysis"""
        try:
            if 'total_sales' not in data.columns:
                return "❌ No sales data found. Please ensure your data has a 'total_sales' column."
            
            insights = []
            insights.append("💰 Sales Analysis Report")
            insights.append("=" * 50)
            
            # Total sales
            total_sales = data['total_sales'].sum()
            insights.append(f"Total Sales: ${total_sales:,.2f}")
            
            # Average order value
            avg_order_value = data['total_sales'].mean()
            insights.append(f"Average Order Value: ${avg_order_value:.2f}")
            
            # Top performing products
            if 'product_name' in data.columns:
                top_products = data.groupby('product_name')['total_sales'].sum().nlargest(5)
                insights.append(f"\n🏆 Top 5 Products:")
                for product, sales in top_products.items():
                    insights.append(f"   • {product}: ${sales:,.2f}")
            
            # Sales by category
            if 'category' in data.columns:
                category_sales = data.groupby('category')['total_sales'].sum().sort_values(ascending=False)
                insights.append(f"\n📊 Sales by Category:")
                for category, sales in category_sales.items():
                    pct = (sales / total_sales) * 100
                    insights.append(f"   • {category}: ${sales:,.2f} ({pct:.1f}%)")
            
            # Sales trends
            if 'date' in data.columns:
                data['date'] = pd.to_datetime(data['date'])
                monthly_sales = data.groupby(data['date'].dt.to_period('M'))['total_sales'].sum()
                best_month = monthly_sales.idxmax()
                worst_month = monthly_sales.idxmin()
                insights.append(f"\n📈 Sales Trends:")
                insights.append(f"   • Best Month: {best_month} (${monthly_sales[best_month]:,.2f})")
                insights.append(f"   • Worst Month: {worst_month} (${monthly_sales[worst_month]:,.2f})")
            
            return "\n".join(insights)
            
        except Exception as e:
            return f"Error in sales analysis: {str(e)}"
    
    def customer_analysis(self, data: pd.DataFrame) -> str:
        """Perform customer analysis"""
        try:
            insights = []
            insights.append("👥 Customer Analysis Report")
            insights.append("=" * 50)
            
            # Customer count
            if 'customer_id' in data.columns:
                unique_customers = data['customer_id'].nunique()
                total_orders = len(data)
                avg_orders_per_customer = total_orders / unique_customers
                insights.append(f"Total Customers: {unique_customers:,}")
                insights.append(f"Total Orders: {total_orders:,}")
                insights.append(f"Average Orders per Customer: {avg_orders_per_customer:.1f}")
            
            # Customer segments
            if 'customer_segment' in data.columns:
                segment_analysis = data.groupby('customer_segment').agg({
                    'customer_id': 'nunique',
                    'total_sales': ['sum', 'mean']
                }).round(2)
                
                insights.append(f"\n🎯 Customer Segments:")
                for segment in segment_analysis.index:
                    customers = segment_analysis.loc[segment, ('customer_id', 'nunique')]
                    total_sales = segment_analysis.loc[segment, ('total_sales', 'sum')]
                    avg_sales = segment_analysis.loc[segment, ('total_sales', 'mean')]
                    insights.append(f"   • {segment}: {customers} customers, ${total_sales:,.2f} total, ${avg_sales:.2f} avg")
            
            # Geographic analysis
            if 'region' in data.columns:
                region_analysis = data.groupby('region').agg({
                    'customer_id': 'nunique',
                    'total_sales': 'sum'
                }).sort_values('total_sales', ascending=False)
                
                insights.append(f"\n🌍 Geographic Distribution:")
                for region in region_analysis.index:
                    customers = region_analysis.loc[region, 'customer_id']
                    sales = region_analysis.loc[region, 'total_sales']
                    insights.append(f"   • {region}: {customers} customers, ${sales:,.2f} sales")
            
            # Customer value analysis
            if 'customer_id' in data.columns and 'total_sales' in data.columns:
                customer_value = data.groupby('customer_id')['total_sales'].sum().sort_values(ascending=False)
                top_customers = customer_value.head(5)
                insights.append(f"\n💎 Top 5 Customers by Value:")
                for i, (customer_id, value) in enumerate(top_customers.items(), 1):
                    insights.append(f"   {i}. Customer {customer_id}: ${value:,.2f}")
            
            return "\n".join(insights)
            
        except Exception as e:
            return f"Error in customer analysis: {str(e)}"
    
    def get_insights_history(self) -> List[Dict[str, Any]]:
        """Get history of all insights generated"""
        return self.insights_history
    
    def clear_history(self):
        """Clear insights history"""
        self.insights_history = []
    
    def export_insights(self, filepath: str) -> bool:
        """Export insights history to JSON file"""
        try:
            with open(filepath, 'w') as f:
                json.dump(self.insights_history, f, indent=2)
            return True
        except Exception as e:
            print(f"Error exporting insights: {str(e)}")
            return False
