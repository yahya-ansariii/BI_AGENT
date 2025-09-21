"""
Schema Manager Module
Handles data schema and relationship storage using JSON files
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime
import os
from pathlib import Path

class SchemaManager:
    """Manages data schemas and relationships using JSON storage"""
    
    def __init__(self, schema_dir: str = "schemas"):
        """
        Initialize the schema manager
        
        Args:
            schema_dir: Directory to store schema files
        """
        self.schema_dir = Path(schema_dir)
        self.schema_dir.mkdir(exist_ok=True)
        
        self.schema_file = self.schema_dir / "data_schema.json"
        self.relationships_file = self.schema_dir / "data_relationships.json"
        self.metadata_file = self.schema_dir / "metadata.json"
        
        # Initialize schema files if they don't exist
        self._initialize_schema_files()
    
    def _initialize_schema_files(self):
        """Initialize schema files with default structure"""
        if not self.schema_file.exists():
            self._save_schema({
                "version": "1.0",
                "created_at": datetime.now().isoformat(),
                "tables": {},
                "relationships": [],
                "metadata": {}
            })
        
        if not self.relationships_file.exists():
            self._save_relationships({
                "version": "1.0",
                "created_at": datetime.now().isoformat(),
                "relationships": []
            })
        
        if not self.metadata_file.exists():
            self._save_metadata({
                "version": "1.0",
                "created_at": datetime.now().isoformat(),
                "data_sources": [],
                "last_updated": None
            })
    
    def _save_schema(self, schema: Dict[str, Any]):
        """Save schema to JSON file"""
        with open(self.schema_file, 'w') as f:
            json.dump(schema, f, indent=2)
    
    def _save_relationships(self, relationships: Dict[str, Any]):
        """Save relationships to JSON file"""
        with open(self.relationships_file, 'w') as f:
            json.dump(relationships, f, indent=2)
    
    def _save_metadata(self, metadata: Dict[str, Any]):
        """Save metadata to JSON file"""
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _load_schema(self) -> Dict[str, Any]:
        """Load schema from JSON file"""
        try:
            with open(self.schema_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {"tables": {}, "relationships": [], "metadata": {}}
    
    def _load_relationships(self) -> Dict[str, Any]:
        """Load relationships from JSON file"""
        try:
            with open(self.relationships_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {"relationships": []}
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load metadata from JSON file"""
        try:
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {"data_sources": [], "last_updated": None}
    
    def analyze_data_schema(self, data: pd.DataFrame, table_name: str = "main_table") -> Dict[str, Any]:
        """
        Analyze data and create schema information
        
        Args:
            data: DataFrame to analyze
            table_name: Name for the table in schema
            
        Returns:
            Dict containing schema information
        """
        schema_info = {
            "table_name": table_name,
            "columns": {},
            "primary_keys": [],
            "foreign_keys": [],
            "indexes": [],
            "constraints": [],
            "data_types": {},
            "statistics": {},
            "created_at": datetime.now().isoformat()
        }
        
        # Analyze each column
        for column in data.columns:
            column_info = self._analyze_column(data[column], column)
            schema_info["columns"][column] = column_info
            schema_info["data_types"][column] = str(data[column].dtype)
        
        # Identify potential primary keys
        schema_info["primary_keys"] = self._identify_primary_keys(data)
        
        # Identify potential foreign keys
        schema_info["foreign_keys"] = self._identify_foreign_keys(data)
        
        # Generate statistics
        schema_info["statistics"] = self._generate_column_statistics(data)
        
        return schema_info
    
    def _analyze_column(self, series: pd.Series, column_name: str) -> Dict[str, Any]:
        """Analyze individual column properties"""
        column_info = {
            "name": column_name,
            "dtype": str(series.dtype),
            "nullable": series.isnull().any(),
            "unique_count": series.nunique(),
            "total_count": len(series),
            "null_count": series.isnull().sum(),
            "null_percentage": (series.isnull().sum() / len(series)) * 100,
            "unique_percentage": (series.nunique() / len(series)) * 100
        }
        
        # Add type-specific information
        if series.dtype in ['int64', 'float64']:
            column_info.update({
                "min_value": float(series.min()) if not series.empty else None,
                "max_value": float(series.max()) if not series.empty else None,
                "mean_value": float(series.mean()) if not series.empty else None,
                "std_value": float(series.std()) if not series.empty else None,
                "is_numeric": True
            })
        elif series.dtype == 'object':
            column_info.update({
                "is_categorical": True,
                "most_common_value": series.value_counts().index[0] if not series.empty else None,
                "most_common_count": int(series.value_counts().iloc[0]) if not series.empty else 0,
                "sample_values": series.dropna().head(5).tolist()
            })
        elif 'datetime' in str(series.dtype):
            column_info.update({
                "is_datetime": True,
                "min_date": str(series.min()) if not series.empty else None,
                "max_date": str(series.max()) if not series.empty else None
            })
        
        return column_info
    
    def _identify_primary_keys(self, data: pd.DataFrame) -> List[str]:
        """Identify potential primary key columns"""
        primary_keys = []
        
        for column in data.columns:
            # Check if column has unique values
            if data[column].nunique() == len(data):
                primary_keys.append(column)
            # Check if column name suggests it's an ID
            elif column.lower() in ['id', 'key', 'pk', 'primary_key']:
                primary_keys.append(column)
        
        return primary_keys
    
    def _identify_foreign_keys(self, data: pd.DataFrame) -> List[Dict[str, str]]:
        """Identify potential foreign key columns"""
        foreign_keys = []
        
        for column in data.columns:
            # Check if column name suggests it's a foreign key
            if column.lower().endswith('_id') and column.lower() != 'id':
                # Try to identify referenced table
                referenced_table = column.lower().replace('_id', '')
                foreign_keys.append({
                    "column": column,
                    "referenced_table": referenced_table,
                    "referenced_column": "id"
                })
        
        return foreign_keys
    
    def _generate_column_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate overall statistics for the dataset"""
        stats = {
            "total_rows": len(data),
            "total_columns": len(data.columns),
            "memory_usage_mb": data.memory_usage(deep=True).sum() / 1024 / 1024,
            "duplicate_rows": data.duplicated().sum(),
            "completeness_score": ((data.size - data.isnull().sum().sum()) / data.size) * 100,
            "numeric_columns": len(data.select_dtypes(include=[np.number]).columns),
            "categorical_columns": len(data.select_dtypes(include=['object']).columns),
            "datetime_columns": len(data.select_dtypes(include=['datetime64']).columns)
        }
        
        return stats
    
    def update_schema_from_data(self, data: pd.DataFrame, table_name: str = "main_table") -> bool:
        """
        Update schema with new data analysis
        
        Args:
            data: DataFrame to analyze
            table_name: Name for the table
            
        Returns:
            bool: True if successful
        """
        try:
            # Analyze the data
            schema_info = self.analyze_data_schema(data, table_name)
            
            # Load existing schema
            schema = self._load_schema()
            
            # Update schema
            schema["tables"][table_name] = schema_info
            schema["last_updated"] = datetime.now().isoformat()
            
            # Save updated schema
            self._save_schema(schema)
            
            # Update metadata
            metadata = self._load_metadata()
            metadata["last_updated"] = datetime.now().isoformat()
            if table_name not in metadata["data_sources"]:
                metadata["data_sources"].append(table_name)
            self._save_metadata(metadata)
            
            return True
            
        except Exception as e:
            print(f"Error updating schema: {str(e)}")
            return False
    
    def get_schema(self, table_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get schema information
        
        Args:
            table_name: Specific table name, or None for all tables
            
        Returns:
            Dict containing schema information
        """
        schema = self._load_schema()
        
        if table_name:
            return schema.get("tables", {}).get(table_name, {})
        else:
            return schema
    
    def add_relationship(self, from_table: str, from_column: str, 
                        to_table: str, to_column: str, 
                        relationship_type: str = "one_to_many") -> bool:
        """
        Add a relationship between tables
        
        Args:
            from_table: Source table name
            from_column: Source column name
            to_table: Target table name
            to_column: Target column name
            relationship_type: Type of relationship
            
        Returns:
            bool: True if successful
        """
        try:
            relationships = self._load_relationships()
            
            new_relationship = {
                "id": f"{from_table}.{from_column} -> {to_table}.{to_column}",
                "from_table": from_table,
                "from_column": from_column,
                "to_table": to_table,
                "to_column": to_column,
                "relationship_type": relationship_type,
                "created_at": datetime.now().isoformat()
            }
            
            # Check if relationship already exists
            existing_ids = [rel["id"] for rel in relationships["relationships"]]
            if new_relationship["id"] not in existing_ids:
                relationships["relationships"].append(new_relationship)
                self._save_relationships(relationships)
            
            return True
            
        except Exception as e:
            print(f"Error adding relationship: {str(e)}")
            return False
    
    def get_relationships(self) -> List[Dict[str, Any]]:
        """Get all relationships"""
        relationships = self._load_relationships()
        return relationships.get("relationships", [])
    
    def detect_relationships(self, data: pd.DataFrame, table_name: str = "main_table") -> List[Dict[str, Any]]:
        """
        Automatically detect potential relationships in the data
        
        Args:
            data: DataFrame to analyze
            table_name: Name of the table
            
        Returns:
            List of detected relationships
        """
        detected_relationships = []
        
        # Look for foreign key patterns
        for column in data.columns:
            if column.lower().endswith('_id') and column.lower() != 'id':
                referenced_table = column.lower().replace('_id', '')
                relationship = {
                    "from_table": table_name,
                    "from_column": column,
                    "to_table": referenced_table,
                    "to_column": "id",
                    "relationship_type": "many_to_one",
                    "confidence": "high",
                    "detected_at": datetime.now().isoformat()
                }
                detected_relationships.append(relationship)
        
        # Look for potential categorical relationships
        categorical_columns = data.select_dtypes(include=['object']).columns
        for column in categorical_columns:
            unique_values = data[column].nunique()
            total_values = len(data)
            
            # If column has few unique values relative to total, might be categorical
            if unique_values < total_values * 0.1 and unique_values > 1:
                relationship = {
                    "from_table": table_name,
                    "from_column": column,
                    "to_table": f"{column}_lookup",
                    "to_column": "value",
                    "relationship_type": "many_to_one",
                    "confidence": "medium",
                    "detected_at": datetime.now().isoformat()
                }
                detected_relationships.append(relationship)
        
        return detected_relationships
    
    def export_schema(self, filepath: str) -> bool:
        """
        Export schema to a JSON file
        
        Args:
            filepath: Path to export file
            
        Returns:
            bool: True if successful
        """
        try:
            schema = self._load_schema()
            relationships = self._load_relationships()
            metadata = self._load_metadata()
            
            export_data = {
                "schema": schema,
                "relationships": relationships,
                "metadata": metadata,
                "exported_at": datetime.now().isoformat()
            }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            return True
            
        except Exception as e:
            print(f"Error exporting schema: {str(e)}")
            return False
    
    def import_schema(self, filepath: str) -> bool:
        """
        Import schema from a JSON file
        
        Args:
            filepath: Path to import file
            
        Returns:
            bool: True if successful
        """
        try:
            with open(filepath, 'r') as f:
                import_data = json.load(f)
            
            if "schema" in import_data:
                self._save_schema(import_data["schema"])
            
            if "relationships" in import_data:
                self._save_relationships(import_data["relationships"])
            
            if "metadata" in import_data:
                self._save_metadata(import_data["metadata"])
            
            return True
            
        except Exception as e:
            print(f"Error importing schema: {str(e)}")
            return False
    
    def get_schema_summary(self) -> Dict[str, Any]:
        """Get a summary of the current schema"""
        schema = self._load_schema()
        relationships = self._load_relationships()
        metadata = self._load_metadata()
        
        summary = {
            "total_tables": len(schema.get("tables", {})),
            "total_relationships": len(relationships.get("relationships", [])),
            "data_sources": metadata.get("data_sources", []),
            "last_updated": metadata.get("last_updated"),
            "schema_version": schema.get("version", "1.0")
        }
        
        # Add table details
        table_details = {}
        for table_name, table_info in schema.get("tables", {}).items():
            table_details[table_name] = {
                "columns": len(table_info.get("columns", {})),
                "primary_keys": len(table_info.get("primary_keys", [])),
                "foreign_keys": len(table_info.get("foreign_keys", [])),
                "rows": table_info.get("statistics", {}).get("total_rows", 0)
            }
        
        summary["tables"] = table_details
        
        return summary
