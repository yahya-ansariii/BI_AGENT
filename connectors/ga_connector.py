"""
Google Analytics Connector
Handles Google Analytics data extraction and processing
"""

import pandas as pd
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import json

try:
    from google.analytics.data_v1beta import BetaAnalyticsDataClient
    from google.analytics.data_v1beta.types import (
        DateRange,
        Dimension,
        Metric,
        RunReportRequest,
        OrderBy
    )
    GA_AVAILABLE = True
except ImportError:
    GA_AVAILABLE = False
    print("Warning: Google Analytics connector not available. Install with: pip install google-analytics-data")

class GoogleAnalyticsConnector:
    """Handles Google Analytics data connections and operations"""
    
    def __init__(self):
        """Initialize Google Analytics connector"""
        self.client = None
        self.property_id = None
        self.available = GA_AVAILABLE
        
    def connect(self, credentials_path: str, property_id: str) -> bool:
        """
        Connect to Google Analytics
        
        Args:
            credentials_path: Path to Google Cloud credentials JSON file
            property_id: GA4 Property ID
            
        Returns:
            bool: True if connection successful
        """
        if not self.available:
            print("Google Analytics connector not available")
            return False
        
        try:
            # Set up credentials
            import os
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
            
            # Initialize client
            self.client = BetaAnalyticsDataClient()
            self.property_id = property_id
            
            # Test connection
            return self.test_connection()
            
        except Exception as e:
            print(f"Error connecting to Google Analytics: {str(e)}")
            return False
    
    def test_connection(self) -> bool:
        """
        Test the current connection
        
        Returns:
            bool: True if connection is working
        """
        if not self.client or not self.property_id:
            return False
        
        try:
            # Simple test query
            request = RunReportRequest(
                property=f"properties/{self.property_id}",
                dimensions=[Dimension(name="date")],
                metrics=[Metric(name="sessions")],
                date_ranges=[DateRange(start_date="2023-01-01", end_date="2023-01-02")]
            )
            
            response = self.client.run_report(request)
            return True
            
        except Exception as e:
            print(f"Error testing GA connection: {str(e)}")
            return False
    
    def get_website_traffic(self, start_date: str, end_date: str, 
                           limit: int = 1000) -> pd.DataFrame:
        """
        Get website traffic data
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            limit: Maximum number of rows to return
            
        Returns:
            DataFrame with traffic data
        """
        if not self.client or not self.property_id:
            return pd.DataFrame()
        
        try:
            request = RunReportRequest(
                property=f"properties/{self.property_id}",
                dimensions=[
                    Dimension(name="date"),
                    Dimension(name="country"),
                    Dimension(name="city"),
                    Dimension(name="deviceCategory"),
                    Dimension(name="operatingSystem"),
                    Dimension(name="browser"),
                    Dimension(name="source"),
                    Dimension(name="medium"),
                    Dimension(name="campaign")
                ],
                metrics=[
                    Metric(name="sessions"),
                    Metric(name="users"),
                    Metric(name="newUsers"),
                    Metric(name="sessionsPerUser"),
                    Metric(name="averageSessionDuration"),
                    Metric(name="bounceRate"),
                    Metric(name="pageviews"),
                    Metric(name="screenPageViews")
                ],
                date_ranges=[DateRange(start_date=start_date, end_date=end_date)],
                order_bys=[OrderBy(dimension=OrderBy.DimensionOrderBy(dimension_name="date"))],
                limit=limit
            )
            
            response = self.client.run_report(request)
            
            # Convert to DataFrame
            data = []
            for row in response.rows:
                row_data = {}
                
                # Add dimensions
                for i, dimension in enumerate(request.dimensions):
                    row_data[dimension.name] = row.dimension_values[i].value
                
                # Add metrics
                for i, metric in enumerate(request.metrics):
                    row_data[metric.name] = row.metric_values[i].value
                
                data.append(row_data)
            
            df = pd.DataFrame(data)
            
            # Convert numeric columns
            numeric_columns = [
                'sessions', 'users', 'newUsers', 'sessionsPerUser',
                'averageSessionDuration', 'bounceRate', 'pageviews', 'screenPageViews'
            ]
            
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Convert date column
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            
            return df
            
        except Exception as e:
            print(f"Error getting website traffic: {str(e)}")
            return pd.DataFrame()
    
    def get_page_performance(self, start_date: str, end_date: str,
                           limit: int = 1000) -> pd.DataFrame:
        """
        Get page performance data
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            limit: Maximum number of rows to return
            
        Returns:
            DataFrame with page performance data
        """
        if not self.client or not self.property_id:
            return pd.DataFrame()
        
        try:
            request = RunReportRequest(
                property=f"properties/{self.property_id}",
                dimensions=[
                    Dimension(name="date"),
                    Dimension(name="pagePath"),
                    Dimension(name="pageTitle"),
                    Dimension(name="landingPage"),
                    Dimension(name="exitPage")
                ],
                metrics=[
                    Metric(name="screenPageViews"),
                    Metric(name="uniquePageviews"),
                    Metric(name="averageSessionDuration"),
                    Metric(name="bounceRate"),
                    Metric(name="exitRate"),
                    Metric(name="pageValue")
                ],
                date_ranges=[DateRange(start_date=start_date, end_date=end_date)],
                order_bys=[OrderBy(metric=OrderBy.MetricOrderBy(metric_name="screenPageViews"))],
                limit=limit
            )
            
            response = self.client.run_report(request)
            
            # Convert to DataFrame
            data = []
            for row in response.rows:
                row_data = {}
                
                # Add dimensions
                for i, dimension in enumerate(request.dimensions):
                    row_data[dimension.name] = row.dimension_values[i].value
                
                # Add metrics
                for i, metric in enumerate(request.metrics):
                    row_data[metric.name] = row.metric_values[i].value
                
                data.append(row_data)
            
            df = pd.DataFrame(data)
            
            # Convert numeric columns
            numeric_columns = [
                'screenPageViews', 'uniquePageviews', 'averageSessionDuration',
                'bounceRate', 'exitRate', 'pageValue'
            ]
            
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Convert date column
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            
            return df
            
        except Exception as e:
            print(f"Error getting page performance: {str(e)}")
            return pd.DataFrame()
    
    def get_ecommerce_data(self, start_date: str, end_date: str,
                          limit: int = 1000) -> pd.DataFrame:
        """
        Get ecommerce data
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            limit: Maximum number of rows to return
            
        Returns:
            DataFrame with ecommerce data
        """
        if not self.client or not self.property_id:
            return pd.DataFrame()
        
        try:
            request = RunReportRequest(
                property=f"properties/{self.property_id}",
                dimensions=[
                    Dimension(name="date"),
                    Dimension(name="itemName"),
                    Dimension(name="itemCategory"),
                    Dimension(name="itemBrand"),
                    Dimension(name="transactionId"),
                    Dimension(name="source"),
                    Dimension(name="medium"),
                    Dimension(name="campaign")
                ],
                metrics=[
                    Metric(name="itemRevenue"),
                    Metric(name="itemQuantity"),
                    Metric(name="purchaseRevenue"),
                    Metric(name="purchases"),
                    Metric(name="transactions"),
                    Metric(name="totalRevenue")
                ],
                date_ranges=[DateRange(start_date=start_date, end_date=end_date)],
                order_bys=[OrderBy(metric=OrderBy.MetricOrderBy(metric_name="itemRevenue"))],
                limit=limit
            )
            
            response = self.client.run_report(request)
            
            # Convert to DataFrame
            data = []
            for row in response.rows:
                row_data = {}
                
                # Add dimensions
                for i, dimension in enumerate(request.dimensions):
                    row_data[dimension.name] = row.dimension_values[i].value
                
                # Add metrics
                for i, metric in enumerate(request.metrics):
                    row_data[metric.name] = row.metric_values[i].value
                
                data.append(row_data)
            
            df = pd.DataFrame(data)
            
            # Convert numeric columns
            numeric_columns = [
                'itemRevenue', 'itemQuantity', 'purchaseRevenue',
                'purchases', 'transactions', 'totalRevenue'
            ]
            
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Convert date column
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            
            return df
            
        except Exception as e:
            print(f"Error getting ecommerce data: {str(e)}")
            return pd.DataFrame()
    
    def get_audience_data(self, start_date: str, end_date: str,
                         limit: int = 1000) -> pd.DataFrame:
        """
        Get audience demographics data
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            limit: Maximum number of rows to return
            
        Returns:
            DataFrame with audience data
        """
        if not self.client or not self.property_id:
            return pd.DataFrame()
        
        try:
            request = RunReportRequest(
                property=f"properties/{self.property_id}",
                dimensions=[
                    Dimension(name="date"),
                    Dimension(name="country"),
                    Dimension(name="region"),
                    Dimension(name="city"),
                    Dimension(name="age"),
                    Dimension(name="gender"),
                    Dimension(name="userType"),
                    Dimension(name="sessionSource"),
                    Dimension(name="sessionMedium")
                ],
                metrics=[
                    Metric(name="sessions"),
                    Metric(name="users"),
                    Metric(name="newUsers"),
                    Metric(name="sessionsPerUser"),
                    Metric(name="averageSessionDuration"),
                    Metric(name="bounceRate")
                ],
                date_ranges=[DateRange(start_date=start_date, end_date=end_date)],
                order_bys=[OrderBy(metric=OrderBy.MetricOrderBy(metric_name="sessions"))],
                limit=limit
            )
            
            response = self.client.run_report(request)
            
            # Convert to DataFrame
            data = []
            for row in response.rows:
                row_data = {}
                
                # Add dimensions
                for i, dimension in enumerate(request.dimensions):
                    row_data[dimension.name] = row.dimension_values[i].value
                
                # Add metrics
                for i, metric in enumerate(request.metrics):
                    row_data[metric.name] = row.metric_values[i].value
                
                data.append(row_data)
            
            df = pd.DataFrame(data)
            
            # Convert numeric columns
            numeric_columns = [
                'sessions', 'users', 'newUsers', 'sessionsPerUser',
                'averageSessionDuration', 'bounceRate'
            ]
            
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Convert date column
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            
            return df
            
        except Exception as e:
            print(f"Error getting audience data: {str(e)}")
            return pd.DataFrame()
    
    def get_custom_report(self, dimensions: List[str], metrics: List[str],
                         start_date: str, end_date: str, limit: int = 1000) -> pd.DataFrame:
        """
        Get custom report with specified dimensions and metrics
        
        Args:
            dimensions: List of dimension names
            metrics: List of metric names
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            limit: Maximum number of rows to return
            
        Returns:
            DataFrame with custom report data
        """
        if not self.client or not self.property_id:
            return pd.DataFrame()
        
        try:
            # Convert to Dimension and Metric objects
            dimension_objects = [Dimension(name=dim) for dim in dimensions]
            metric_objects = [Metric(name=metric) for metric in metrics]
            
            request = RunReportRequest(
                property=f"properties/{self.property_id}",
                dimensions=dimension_objects,
                metrics=metric_objects,
                date_ranges=[DateRange(start_date=start_date, end_date=end_date)],
                limit=limit
            )
            
            response = self.client.run_report(request)
            
            # Convert to DataFrame
            data = []
            for row in response.rows:
                row_data = {}
                
                # Add dimensions
                for i, dimension in enumerate(request.dimensions):
                    row_data[dimension.name] = row.dimension_values[i].value
                
                # Add metrics
                for i, metric in enumerate(request.metrics):
                    row_data[metric.name] = row.metric_values[i].value
                
                data.append(row_data)
            
            df = pd.DataFrame(data)
            
            # Convert numeric columns
            for col in metrics:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Convert date column if present
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            
            return df
            
        except Exception as e:
            print(f"Error getting custom report: {str(e)}")
            return pd.DataFrame()
