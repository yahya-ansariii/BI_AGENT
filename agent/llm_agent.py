"""
LLM Agent Module
Handles AI-powered analysis using Ollama via subprocess
"""

import subprocess
import json
import os
import sys
from typing import Optional, Dict, Any, List
from datetime import datetime
from pathlib import Path

# Add parent directory to path for config import
sys.path.append(str(Path(__file__).parent.parent))

try:
    from config.ollama_config import ollama_config
except ImportError:
    # Fallback if config not available
    ollama_config = None

class LLMAgent:
    """AI agent for business data analysis using Ollama"""
    
    def __init__(self, model_path: str = None):
        """
        Initialize the LLM agent
        
        Args:
            model_path: Custom path for Ollama models (optional)
        """
        self.model_name = "llama3:8b-instruct"
        self.insights_history = []
        
        # Set custom model path if provided
        if model_path and ollama_config:
            ollama_config.set_custom_model_path(model_path)
            print(f"Using custom model path: {ollama_config.get_model_path()}")
        elif ollama_config:
            print(f"Using model path: {ollama_config.get_model_path()}")
        
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
            # Prepare the command with custom model path
            if ollama_config:
                cmd = ollama_config.get_ollama_command(model)
                env = os.environ.copy()
                env['OLLAMA_MODELS'] = ollama_config.get_model_path()
            else:
                cmd = ["ollama", "run", model]
                env = None
            
            # Run the command with the prompt
            result = subprocess.run(
                cmd,
                input=prompt,
                text=True,
                capture_output=True,
                timeout=60,
                env=env
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
    
    def set_model_path(self, path: str) -> bool:
        """
        Set custom model path for Ollama
        
        Args:
            path: Path to store Ollama models
            
        Returns:
            bool: True if path is valid and accessible
        """
        if ollama_config:
            return ollama_config.set_custom_model_path(path)
        return False
    
    def get_model_path(self) -> str:
        """Get current model path"""
        if ollama_config:
            return ollama_config.get_model_path()
        return "~/.ollama/models"
    
    def pull_model(self, model: str) -> bool:
        """
        Pull a model to the configured path
        
        Args:
            model: Model name to pull
            
        Returns:
            bool: True if successful
        """
        if ollama_config:
            return ollama_config.pull_model(model)
        return False
    
    def list_models(self) -> list:
        """Get list of available models"""
        if ollama_config:
            return ollama_config.get_available_models()
        return []
    
    def check_model_exists(self, model: str) -> bool:
        """Check if a model exists"""
        if ollama_config:
            return ollama_config.check_model_exists(model)
        return False
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific model"""
        try:
            # Check if model exists
            exists = self.check_model_exists(model_name)
            if not exists:
                return {
                    "name": model_name,
                    "exists": False,
                    "size": "Unknown",
                    "status": "Not found"
                }
            
            # Get model size and other info
            model_path = self.get_model_path()
            model_file = os.path.join(model_path, model_name)
            
            size = "Unknown"
            if os.path.exists(model_file):
                if os.path.isfile(model_file):
                    size = f"{os.path.getsize(model_file) / (1024*1024*1024):.2f} GB"
                elif os.path.isdir(model_file):
                    total_size = 0
                    for dirpath, dirnames, filenames in os.walk(model_file):
                        for filename in filenames:
                            filepath = os.path.join(dirpath, filename)
                            total_size += os.path.getsize(filepath)
                    size = f"{total_size / (1024*1024*1024):.2f} GB"
            
            return {
                "name": model_name,
                "exists": True,
                "size": size,
                "status": "Available",
                "path": model_file
            }
        except Exception as e:
            return {
                "name": model_name,
                "exists": False,
                "size": "Unknown",
                "status": f"Error: {str(e)}"
            }