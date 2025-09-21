#!/usr/bin/env python3
"""
Create demo Excel files for the Business Insights Agent
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def create_demo_data():
    """Create demo Excel files"""
    
    # Ensure demo_data directory exists
    os.makedirs('demo_data', exist_ok=True)
    
    # Create sales data
    sales_data = {
        'order_id': ['ORD-0001', 'ORD-0002', 'ORD-0003', 'ORD-0004', 'ORD-0005'],
        'date': pd.date_range('2024-01-01', periods=5, freq='D'),
        'region': ['North', 'South', 'East', 'West', 'North'],
        'amount': [150.50, 275.75, 89.99, 320.00, 195.25]
    }
    sales_df = pd.DataFrame(sales_data)
    sales_df.to_excel('demo_data/sales.xlsx', index=False)
    print("✅ Created demo_data/sales.xlsx")
    
    # Create web traffic data
    traffic_data = {
        'date': pd.date_range('2024-01-01', periods=5, freq='D'),
        'visits': [1200, 1350, 980, 1450, 1100],
        'source': ['Google', 'Facebook', 'Direct', 'Google', 'Twitter']
    }
    traffic_df = pd.DataFrame(traffic_data)
    traffic_df.to_excel('demo_data/web_traffic.xlsx', index=False)
    print("✅ Created demo_data/web_traffic.xlsx")
    
    print("\n🎉 Demo Excel files created successfully!")

if __name__ == "__main__":
    create_demo_data()
