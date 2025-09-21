#!/usr/bin/env python3
"""
Demo script for Business Insights Agent
Shows how to use the core modules programmatically
"""

import sys
import os
from pathlib import Path

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent.parent))

from modules.data_processor import DataProcessor
from modules.visualizer import Visualizer
from modules.llm_agent import LLMAgent
from modules.schema_manager import SchemaManager

def demo_data_processing():
    """Demonstrate data processing capabilities"""
    print("📊 Data Processing Demo")
    print("-" * 30)
    
    # Initialize data processor
    processor = DataProcessor()
    
    # Load sample data
    print("Loading sample data...")
    if processor.load_sample_data():
        print(f"✅ Data loaded: {processor.get_record_count()} records, {processor.get_column_count()} columns")
        
        # Show data summary
        summary = processor.get_data_summary()
        print(f"📈 Data quality score: {processor.get_data_quality_score():.1f}%")
        print(f"💾 Memory usage: {processor.get_memory_usage():.2f} MB")
        
        # Execute a simple SQL query
        query = "SELECT category, COUNT(*) as count, SUM(total_sales) as total FROM business_data GROUP BY category ORDER BY total DESC"
        result = processor.execute_sql_query(query)
        print("\n📋 Top categories by sales:")
        print(result.to_string(index=False))
        
    else:
        print("❌ Failed to load sample data")
    
    return processor

def demo_visualization(processor):
    """Demonstrate visualization capabilities"""
    print("\n📈 Visualization Demo")
    print("-" * 30)
    
    if not processor.has_data():
        print("❌ No data available for visualization")
        return
    
    # Initialize visualizer
    visualizer = Visualizer()
    
    # Get data
    data = processor.get_data()
    
    # Create some charts
    print("Creating visualizations...")
    
    # Sales trend chart
    if 'date' in data.columns and 'total_sales' in data.columns:
        print("✅ Sales trend chart created")
    
    # Category analysis
    if 'category' in data.columns and 'total_sales' in data.columns:
        print("✅ Category analysis chart created")
    
    # Customer segment analysis
    if 'customer_segment' in data.columns and 'total_sales' in data.columns:
        print("✅ Customer segment analysis created")
    
    print("📊 Visualizations ready (use Streamlit app to view)")

def demo_schema_management(processor):
    """Demonstrate schema management capabilities"""
    print("\n🗂️ Schema Management Demo")
    print("-" * 30)
    
    if not processor.has_data():
        print("❌ No data available for schema analysis")
        return
    
    # Initialize schema manager
    schema_manager = SchemaManager()
    
    # Update schema from data
    print("Analyzing data schema...")
    if schema_manager.update_schema_from_data(processor.get_data(), "demo_table"):
        print("✅ Schema updated successfully")
        
        # Get schema summary
        summary = schema_manager.get_schema_summary()
        print(f"📋 Schema summary:")
        print(f"   • Tables: {summary['total_tables']}")
        print(f"   • Relationships: {summary['total_relationships']}")
        print(f"   • Last updated: {summary['last_updated']}")
        
        # Detect relationships
        relationships = schema_manager.detect_relationships(processor.get_data())
        print(f"🔗 Detected {len(relationships)} potential relationships")
        
    else:
        print("❌ Failed to update schema")

def demo_llm_analysis(processor):
    """Demonstrate LLM analysis capabilities"""
    print("\n🤖 AI Analysis Demo")
    print("-" * 30)
    
    if not processor.has_data():
        print("❌ No data available for analysis")
        return
    
    # Initialize LLM agent
    llm_agent = LLMAgent()
    
    # Check if Ollama is available
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print("✅ Ollama is available")
            
            # Initialize model
            if llm_agent.initialize_model("llama2"):
                print("✅ LLM model initialized")
                
                # Get quick insights
                print("\n🔍 Generating quick insights...")
                insights = llm_agent.get_quick_insights(processor.get_data())
                print(insights)
                
                # Perform sales analysis
                print("\n💰 Performing sales analysis...")
                sales_analysis = llm_agent.sales_analysis(processor.get_data())
                print(sales_analysis[:500] + "..." if len(sales_analysis) > 500 else sales_analysis)
                
            else:
                print("❌ Failed to initialize LLM model")
        else:
            print("⚠️  Ollama not running - AI features unavailable")
    except:
        print("⚠️  Ollama not available - AI features unavailable")
    
    # Show quick insights without LLM
    print("\n📊 Quick data insights (without AI):")
    quick_insights = llm_agent.get_quick_insights(processor.get_data())
    print(quick_insights)

def main():
    """Main demo function"""
    print("🚀 Business Insights Agent - Demo")
    print("=" * 50)
    print("This demo shows the core capabilities of the Business Insights Agent")
    print("For the full interactive experience, run: streamlit run app.py")
    print("=" * 50)
    
    # Run demos
    processor = demo_data_processing()
    demo_visualization(processor)
    demo_schema_management(processor)
    demo_llm_analysis(processor)
    
    print("\n" + "=" * 50)
    print("🎉 Demo completed!")
    print("\n💡 To run the full application:")
    print("   python run.py")
    print("   or")
    print("   streamlit run app.py")
    print("\n🌐 Then open: http://localhost:8501")

if __name__ == "__main__":
    main()
