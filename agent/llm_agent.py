"""
LLM Agent Module
Handles AI-powered analysis using Ollama via subprocess
"""

import subprocess
import json
from typing import Optional, Dict, Any, List
from datetime import datetime

class LLMAgent:
    """AI agent for business data analysis using Ollama"""
    
    def __init__(self):
        """Initialize the LLM agent"""
        self.model_name = "llama3:8b-instruct"
        self.insights_history = []
        
    def query_llm(self, prompt: str, model: str = "llama3:8b-instruct") -> str:
        """
        Query LLM using Ollama via subprocess
        
        Args:
            prompt: The prompt to send to the LLM
            model: The Ollama model to use
            
        Returns:
            str: LLM response
        """
        try:
            # Prepare the command
            cmd = ["ollama", "run", model]
            
            # Run the command with the prompt
            result = subprocess.run(
                cmd,
                input=prompt,
                text=True,
                capture_output=True,
                timeout=60
            )
            
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                error_msg = f"Ollama error: {result.stderr}"
                print(error_msg)
                return error_msg
                
        except subprocess.TimeoutExpired:
            error_msg = "Ollama request timed out"
            print(error_msg)
            return error_msg
        except FileNotFoundError:
            error_msg = "Ollama not found. Please install Ollama and ensure it's in your PATH"
            print(error_msg)
            return error_msg
        except Exception as e:
            error_msg = f"Error querying LLM: {str(e)}"
            print(error_msg)
            return error_msg
    
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
            
            # Query the LLM
            response = self.query_llm(prompt, self.model_name)
            
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
    
    def _prepare_data_summary(self, data) -> Dict[str, Any]:
        """Prepare data summary for LLM analysis"""
        import pandas as pd
        
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
    
    def get_quick_insights(self, data) -> str:
        """Get quick insights about the data"""
        try:
            import pandas as pd
            
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