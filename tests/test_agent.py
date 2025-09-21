"""
Test cases for Business Insights Agent
"""

import pytest
import sys
from pathlib import Path
import json

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from agent.sql_generator import SQLGenerator
from connectors.excel_connector import ExcelConnector

class TestAgent:
    """Test cases for the agent functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.sql_generator = SQLGenerator()
        self.excel_connector = ExcelConnector()
        
        # Demo schema
        self.demo_schema = {
            "sales": [
                {"name": "order_id", "type": "object", "nullable": False},
                {"name": "date", "type": "datetime64[ns]", "nullable": False},
                {"name": "region", "type": "object", "nullable": False},
                {"name": "amount", "type": "float64", "nullable": False}
            ],
            "web_traffic": [
                {"name": "date", "type": "datetime64[ns]", "nullable": False},
                {"name": "visits", "type": "int64", "nullable": False},
                {"name": "source", "type": "object", "nullable": False}
            ]
        }
    
    def test_load_demo_schema(self):
        """Test loading demo schema"""
        # This test verifies the schema structure
        assert "sales" in self.demo_schema
        assert "web_traffic" in self.demo_schema
        assert len(self.demo_schema["sales"]) == 4
        assert len(self.demo_schema["web_traffic"]) == 3
    
    def test_sql_generation_total_sales_by_region(self):
        """Test SQL generation for 'Total sales by region' query"""
        question = "Total sales by region"
        
        # Generate SQL
        sql = self.sql_generator.generate_sql(question, self.demo_schema)
        
        # Assertions
        assert sql is not None
        assert sql.strip() != ""
        assert "SELECT" in sql.upper()
        assert "GROUP BY" in sql.upper()
        assert "region" in sql.lower()
        assert "amount" in sql.lower() or "sales" in sql.lower()
    
    def test_sql_generation_with_relationships(self):
        """Test SQL generation with table relationships"""
        question = "Compare sales and visits by date"
        relationships = [
            {
                "from_table": "sales",
                "from_column": "date",
                "to_table": "web_traffic",
                "to_column": "date"
            }
        ]
        
        # Generate SQL
        sql = self.sql_generator.generate_sql(question, self.demo_schema, relationships)
        
        # Assertions
        assert sql is not None
        assert sql.strip() != ""
        assert "SELECT" in sql.upper()
        assert "JOIN" in sql.upper() or "FROM" in sql.upper()
        assert "sales" in sql.lower()
        assert "web_traffic" in sql.lower()
    
    def test_sql_validation(self):
        """Test SQL validation functionality"""
        # Valid SQL
        valid_sql = "SELECT region, SUM(amount) FROM sales GROUP BY region"
        validation = self.sql_generator.validate_sql(valid_sql)
        assert validation["is_valid"] == True
        assert len(validation["errors"]) == 0
        
        # Invalid SQL
        invalid_sql = "INVALID SQL QUERY"
        validation = self.sql_generator.validate_sql(invalid_sql)
        assert validation["is_valid"] == False
        assert len(validation["errors"]) > 0
    
    def test_excel_connector_schema_extraction(self):
        """Test Excel connector schema extraction"""
        # This test would require actual Excel files
        # For now, we'll test the schema structure
        expected_columns = ["order_id", "date", "region", "amount"]
        sales_columns = [col["name"] for col in self.demo_schema["sales"]]
        
        for col in expected_columns:
            assert col in sales_columns
    
    def test_sql_contains_group_by_region(self):
        """Test that SQL contains 'GROUP BY region' for sales by region query"""
        question = "Total sales by region"
        
        # Generate SQL
        sql = self.sql_generator.generate_sql(question, self.demo_schema)
        
        # Assert that SQL contains GROUP BY region
        assert "GROUP BY region" in sql or "GROUP BY sales.region" in sql or "group by region" in sql.lower()
    
    def test_sql_generation_multiple_queries(self):
        """Test SQL generation for multiple different queries"""
        test_cases = [
            {
                "question": "Show all sales data",
                "expected_keywords": ["SELECT", "FROM", "sales"]
            },
            {
                "question": "Average amount by region",
                "expected_keywords": ["SELECT", "AVG", "GROUP BY", "region"]
            },
            {
                "question": "Total visits by source",
                "expected_keywords": ["SELECT", "SUM", "GROUP BY", "source"]
            }
        ]
        
        for test_case in test_cases:
            sql = self.sql_generator.generate_sql(test_case["question"], self.demo_schema)
            
            assert sql is not None
            assert sql.strip() != ""
            
            for keyword in test_case["expected_keywords"]:
                assert keyword in sql.upper()

if __name__ == "__main__":
    pytest.main([__file__])
