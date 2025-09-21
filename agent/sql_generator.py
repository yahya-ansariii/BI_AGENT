"""
SQL Generator Module
Generates SQL queries from natural language questions using LLM
"""

import json
from typing import Dict, Any, List
from .llm_agent import LLMAgent

class SQLGenerator:
    """Generates SQL queries from natural language questions"""
    
    def __init__(self):
        """Initialize SQL generator"""
        self.llm_agent = LLMAgent()
    
    def generate_sql(self, question: str, schema: Dict[str, Any], 
                    relationships: List[Dict[str, str]] = None) -> str:
        """
        Generate SQL query from natural language question
        
        Args:
            question: Natural language question
            schema: Dictionary with table schemas
            relationships: List of relationship dictionaries
            
        Returns:
            str: Generated SQL query
        """
        try:
            # Create prompt with schema and relationships
            prompt = self._create_sql_prompt(question, schema, relationships)
            
            # Query LLM for SQL generation
            response = self.llm_agent.query_llm(prompt)
            
            # Extract SQL from response (remove any explanations)
            sql = self._extract_sql(response)
            
            return sql
            
        except Exception as e:
            print(f"Error generating SQL: {str(e)}")
            return ""
    
    def _create_sql_prompt(self, question: str, schema: Dict[str, Any], 
                          relationships: List[Dict[str, str]] = None) -> str:
        """Create prompt for SQL generation"""
        
        # Format schema information
        schema_text = "Database Schema:\n"
        for table_name, columns in schema.items():
            schema_text += f"\nTable: {table_name}\n"
            for col in columns:
                schema_text += f"  - {col['name']} ({col['type']})\n"
        
        # Add relationships if provided
        relationships_text = ""
        if relationships:
            relationships_text = "\n\nTable Relationships:\n"
            for rel in relationships:
                relationships_text += f"  - {rel['from_table']}.{rel['from_column']} = {rel['to_table']}.{rel['to_column']}\n"
        
        prompt = f"""You are a SQL expert. Generate a SQL query based on the natural language question.

{schema_text}{relationships_text}

Question: {question}

Instructions:
- Only output the SQL query, no explanations
- Use proper SQL syntax
- Include appropriate JOINs if multiple tables are needed
- Use meaningful column aliases
- Order results logically

SQL Query:"""
        
        return prompt
    
    def _extract_sql(self, response: str) -> str:
        """Extract SQL query from LLM response"""
        # Remove any markdown formatting
        sql = response.strip()
        
        # Remove code blocks if present
        if sql.startswith("```sql"):
            sql = sql[6:]
        elif sql.startswith("```"):
            sql = sql[3:]
        
        if sql.endswith("```"):
            sql = sql[:-3]
        
        # Clean up the SQL
        sql = sql.strip()
        
        # Ensure it starts with SELECT
        if not sql.upper().startswith("SELECT"):
            # Try to find SELECT in the response
            lines = sql.split('\n')
            for i, line in enumerate(lines):
                if line.strip().upper().startswith("SELECT"):
                    sql = '\n'.join(lines[i:])
                    break
        
        return sql
    
    def validate_sql(self, sql: str) -> Dict[str, Any]:
        """
        Validate SQL query syntax
        
        Args:
            sql: SQL query to validate
            
        Returns:
            Dictionary with validation results
        """
        validation = {
            "is_valid": True,
            "errors": [],
            "warnings": []
        }
        
        if not sql.strip():
            validation["is_valid"] = False
            validation["errors"].append("Empty SQL query")
            return validation
        
        # Basic SQL validation
        sql_upper = sql.upper().strip()
        
        if not sql_upper.startswith("SELECT"):
            validation["errors"].append("Query must start with SELECT")
            validation["is_valid"] = False
        
        # Check for common issues
        if "DROP" in sql_upper or "DELETE" in sql_upper or "UPDATE" in sql_upper or "INSERT" in sql_upper:
            validation["warnings"].append("Query contains data modification statements")
        
        if ";" in sql:
            validation["warnings"].append("Query contains semicolon (may cause issues)")
        
        return validation
    
    def get_query_explanation(self, sql: str) -> str:
        """
        Get explanation of what the SQL query does
        
        Args:
            sql: SQL query to explain
            
        Returns:
            str: Explanation of the query
        """
        try:
            prompt = f"""Explain what this SQL query does in simple terms:

{sql}

Provide a brief explanation of:
1. What data is being selected
2. Which tables are involved
3. What filtering/grouping is applied
4. What the results will show

Explanation:"""
            
            response = self.llm_agent.query_llm(prompt)
            return response
            
        except Exception as e:
            return f"Error generating explanation: {str(e)}"
    
    def suggest_optimizations(self, sql: str) -> List[str]:
        """
        Suggest optimizations for the SQL query
        
        Args:
            sql: SQL query to optimize
            
        Returns:
            List of optimization suggestions
        """
        try:
            prompt = f"""Analyze this SQL query and suggest optimizations:

{sql}

Provide specific suggestions for:
1. Performance improvements
2. Query structure improvements
3. Index recommendations
4. Best practices

Suggestions:"""
            
            response = self.llm_agent.query_llm(prompt)
            
            # Parse response into list of suggestions
            suggestions = []
            lines = response.split('\n')
            for line in lines:
                line = line.strip()
                if line and (line.startswith('-') or line.startswith('•') or line.startswith('1.') or line.startswith('2.') or line.startswith('3.') or line.startswith('4.')):
                    suggestions.append(line)
            
            return suggestions if suggestions else [response]
            
        except Exception as e:
            return [f"Error generating suggestions: {str(e)}"]
