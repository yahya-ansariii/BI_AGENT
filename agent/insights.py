"""
Insights Module
Generates business insights from query results using LLM
"""

import pandas as pd
from typing import Dict, Any, List
from .llm_agent import LLMAgent

class InsightsGenerator:
    """Generates business insights from query results"""
    
    def __init__(self, primary_model="llama3:8b-instruct", secondary_model="llama2:7b"):
        """Initialize insights generator with two models"""
        self.primary_llm_agent = LLMAgent()
        self.secondary_llm_agent = LLMAgent()
        self.primary_model = primary_model
        self.secondary_model = secondary_model
    
    def generate_insights(self, question: str, sql: str, result_df: pd.DataFrame) -> Dict[str, str]:
        """
        Generate insights from query results using two different models
        
        Args:
            question: Original natural language question
            sql: SQL query that was executed
            result_df: DataFrame with query results
            
        Returns:
            Dict[str, str]: Generated insights from both models
        """
        try:
            # Prepare data summary
            data_summary = self._prepare_result_summary(result_df)
            
            # Create prompt
            prompt = self._create_insights_prompt(question, sql, data_summary)
            
            # Query both LLMs for insights
            primary_response = self.primary_llm_agent.query_llm(prompt, self.primary_model)
            secondary_response = self.secondary_llm_agent.query_llm(prompt, self.secondary_model)
            
            return {
                "primary_model": {
                    "model_name": self.primary_model,
                    "insights": primary_response
                },
                "secondary_model": {
                    "model_name": self.secondary_model,
                    "insights": secondary_response
                }
            }
            
        except Exception as e:
            error_msg = f"Error generating insights: {str(e)}"
            return {
                "primary_model": {
                    "model_name": self.primary_model,
                    "insights": error_msg
                },
                "secondary_model": {
                    "model_name": self.secondary_model,
                    "insights": error_msg
                }
            }
    
    def _prepare_result_summary(self, result_df: pd.DataFrame) -> Dict[str, Any]:
        """Prepare summary of query results"""
        summary = {
            "row_count": len(result_df),
            "column_count": len(result_df.columns),
            "columns": list(result_df.columns),
            "dtypes": result_df.dtypes.to_dict(),
            "sample_data": result_df.head(10).to_dict('records'),
            "numeric_summary": {},
            "categorical_summary": {}
        }
        
        # Add numeric column summaries
        numeric_cols = result_df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            summary["numeric_summary"][col] = {
                "min": float(result_df[col].min()),
                "max": float(result_df[col].max()),
                "mean": float(result_df[col].mean()),
                "sum": float(result_df[col].sum()),
                "count": int(result_df[col].count())
            }
        
        # Add categorical column summaries
        categorical_cols = result_df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            value_counts = result_df[col].value_counts().head(5)
            summary["categorical_summary"][col] = {
                "unique_count": int(result_df[col].nunique()),
                "top_values": value_counts.to_dict()
            }
        
        return summary
    
    def _create_insights_prompt(self, question: str, sql: str, data_summary: Dict[str, Any]) -> str:
        """Create prompt for insights generation"""
        
        prompt = f"""You are a business intelligence analyst. Analyze the following query results and provide insights.

Original Question: {question}

SQL Query Executed:
{sql}

Query Results Summary:
- Number of rows: {data_summary['row_count']}
- Number of columns: {data_summary['column_count']}
- Columns: {data_summary['columns']}

Sample Data (top 10 rows):
{data_summary['sample_data']}

Numeric Column Statistics:
{data_summary['numeric_summary']}

Categorical Column Statistics:
{data_summary['categorical_summary']}

Please provide:
1. Key findings from the data
2. Trends and patterns observed
3. Business implications
4. Actionable recommendations
5. Any data quality concerns

Keep the response concise and focused on business value.

Insights:"""
        
        return prompt
    
    def generate_chart_recommendations(self, result_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Generate chart recommendations based on query results
        
        Args:
            result_df: DataFrame with query results
            
        Returns:
            List of chart recommendation dictionaries
        """
        recommendations = []
        
        try:
            # Check for date columns
            date_cols = result_df.select_dtypes(include=['datetime64']).columns
            if len(date_cols) > 0:
                recommendations.append({
                    "type": "line_chart",
                    "title": "Trend Over Time",
                    "x_axis": date_cols[0],
                    "y_axis": [col for col in result_df.columns if col != date_cols[0]],
                    "description": "Shows trends over time"
                })
            
            # Check for categorical columns
            categorical_cols = result_df.select_dtypes(include=['object']).columns
            numeric_cols = result_df.select_dtypes(include=['number']).columns
            
            if len(categorical_cols) > 0 and len(numeric_cols) > 0:
                recommendations.append({
                    "type": "bar_chart",
                    "title": "Comparison by Category",
                    "x_axis": categorical_cols[0],
                    "y_axis": numeric_cols[0],
                    "description": "Compares values across categories"
                })
            
            # Check for multiple numeric columns
            if len(numeric_cols) > 1:
                recommendations.append({
                    "type": "scatter_plot",
                    "title": "Correlation Analysis",
                    "x_axis": numeric_cols[0],
                    "y_axis": numeric_cols[1],
                    "description": "Shows relationship between two numeric variables"
                })
            
            # Check for single numeric column
            if len(numeric_cols) == 1:
                recommendations.append({
                    "type": "histogram",
                    "title": "Distribution",
                    "x_axis": numeric_cols[0],
                    "y_axis": None,
                    "description": "Shows distribution of values"
                })
            
        except Exception as e:
            print(f"Error generating chart recommendations: {str(e)}")
        
        return recommendations
    
    def get_data_quality_assessment(self, result_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Assess data quality of query results
        
        Args:
            result_df: DataFrame with query results
            
        Returns:
            Dictionary with data quality assessment
        """
        assessment = {
            "overall_score": 100,
            "issues": [],
            "warnings": [],
            "recommendations": []
        }
        
        try:
            # Check for empty results
            if result_df.empty:
                assessment["issues"].append("Query returned no results")
                assessment["overall_score"] = 0
                return assessment
            
            # Check for missing values
            missing_counts = result_df.isnull().sum()
            total_cells = result_df.size
            
            if missing_counts.sum() > 0:
                missing_pct = (missing_counts.sum() / total_cells) * 100
                if missing_pct > 50:
                    assessment["issues"].append(f"High missing data: {missing_pct:.1f}%")
                    assessment["overall_score"] -= 30
                elif missing_pct > 20:
                    assessment["warnings"].append(f"Moderate missing data: {missing_pct:.1f}%")
                    assessment["overall_score"] -= 15
                else:
                    assessment["warnings"].append(f"Low missing data: {missing_pct:.1f}%")
                    assessment["overall_score"] -= 5
            
            # Check for duplicate rows
            duplicate_count = result_df.duplicated().sum()
            if duplicate_count > 0:
                duplicate_pct = (duplicate_count / len(result_df)) * 100
                if duplicate_pct > 50:
                    assessment["issues"].append(f"High duplicate data: {duplicate_pct:.1f}%")
                    assessment["overall_score"] -= 25
                else:
                    assessment["warnings"].append(f"Some duplicate data: {duplicate_pct:.1f}%")
                    assessment["overall_score"] -= 10
            
            # Check for data consistency
            for col in result_df.columns:
                if result_df[col].dtype == 'object':
                    # Check for inconsistent formatting
                    unique_values = result_df[col].dropna().unique()
                    if len(unique_values) > 0:
                        # Check for mixed case
                        case_variations = set([str(val).lower() for val in unique_values])
                        if len(case_variations) < len(unique_values):
                            assessment["warnings"].append(f"Column '{col}' has inconsistent casing")
                            assessment["overall_score"] -= 5
            
            # Ensure score doesn't go below 0
            assessment["overall_score"] = max(0, assessment["overall_score"])
            
            # Generate recommendations
            if assessment["overall_score"] < 70:
                assessment["recommendations"].append("Consider data cleaning before analysis")
            if missing_counts.sum() > 0:
                assessment["recommendations"].append("Address missing data issues")
            if duplicate_count > 0:
                assessment["recommendations"].append("Remove or investigate duplicate records")
            
        except Exception as e:
            assessment["issues"].append(f"Error assessing data quality: {str(e)}")
            assessment["overall_score"] = 0
        
        return assessment
    
    def set_models(self, primary_model: str, secondary_model: str):
        """
        Set the models to use for insights generation
        
        Args:
            primary_model: Primary model name
            secondary_model: Secondary model name
        """
        self.primary_model = primary_model
        self.secondary_model = secondary_model
    
    def get_available_models(self) -> List[str]:
        """Get list of available models from both agents"""
        primary_models = self.primary_llm_agent.list_models()
        secondary_models = self.secondary_llm_agent.list_models()
        
        # Combine and deduplicate
        all_models = list(set(primary_models + secondary_models))
        return all_models
