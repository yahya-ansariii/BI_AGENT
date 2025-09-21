"""
Oncore Connector
Handles Oncore system data extraction and processing
"""

import pandas as pd
from typing import Optional, Dict, Any, List
from datetime import datetime
import json
import requests
from urllib.parse import urljoin

class OncoreConnector:
    """Handles Oncore system data connections and operations"""
    
    def __init__(self):
        """Initialize Oncore connector"""
        self.base_url = None
        self.api_key = None
        self.session = None
        self.connected = False
        
    def connect(self, base_url: str, api_key: str) -> bool:
        """
        Connect to Oncore system
        
        Args:
            base_url: Oncore API base URL
            api_key: API key for authentication
            
        Returns:
            bool: True if connection successful
        """
        try:
            self.base_url = base_url.rstrip('/')
            self.api_key = api_key
            
            # Initialize session
            self.session = requests.Session()
            self.session.headers.update({
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            })
            
            # Test connection
            return self.test_connection()
            
        except Exception as e:
            print(f"Error connecting to Oncore: {str(e)}")
            return False
    
    def test_connection(self) -> bool:
        """
        Test the current connection
        
        Returns:
            bool: True if connection is working
        """
        if not self.session or not self.base_url:
            return False
        
        try:
            # Test endpoint (adjust based on actual Oncore API)
            test_url = urljoin(self.base_url, '/api/health')
            response = self.session.get(test_url, timeout=10)
            
            if response.status_code == 200:
                self.connected = True
                return True
            else:
                print(f"Connection test failed with status: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"Error testing Oncore connection: {str(e)}")
            return False
    
    def get_protocols(self, limit: int = 100) -> pd.DataFrame:
        """
        Get protocols data from Oncore
        
        Args:
            limit: Maximum number of records to return
            
        Returns:
            DataFrame with protocols data
        """
        if not self.connected:
            return pd.DataFrame()
        
        try:
            url = urljoin(self.base_url, '/api/protocols')
            params = {'limit': limit}
            
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            # Convert to DataFrame
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict) and 'data' in data:
                df = pd.DataFrame(data['data'])
            else:
                df = pd.DataFrame([data])
            
            return df
            
        except Exception as e:
            print(f"Error getting protocols: {str(e)}")
            return pd.DataFrame()
    
    def get_subjects(self, protocol_id: str = None, limit: int = 100) -> pd.DataFrame:
        """
        Get subjects data from Oncore
        
        Args:
            protocol_id: Specific protocol ID (optional)
            limit: Maximum number of records to return
            
        Returns:
            DataFrame with subjects data
        """
        if not self.connected:
            return pd.DataFrame()
        
        try:
            url = urljoin(self.base_url, '/api/subjects')
            params = {'limit': limit}
            
            if protocol_id:
                params['protocol_id'] = protocol_id
            
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            # Convert to DataFrame
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict) and 'data' in data:
                df = pd.DataFrame(data['data'])
            else:
                df = pd.DataFrame([data])
            
            return df
            
        except Exception as e:
            print(f"Error getting subjects: {str(e)}")
            return pd.DataFrame()
    
    def get_events(self, subject_id: str = None, protocol_id: str = None,
                  start_date: str = None, end_date: str = None,
                  limit: int = 100) -> pd.DataFrame:
        """
        Get events data from Oncore
        
        Args:
            subject_id: Specific subject ID (optional)
            protocol_id: Specific protocol ID (optional)
            start_date: Start date filter (optional)
            end_date: End date filter (optional)
            limit: Maximum number of records to return
            
        Returns:
            DataFrame with events data
        """
        if not self.connected:
            return pd.DataFrame()
        
        try:
            url = urljoin(self.base_url, '/api/events')
            params = {'limit': limit}
            
            if subject_id:
                params['subject_id'] = subject_id
            if protocol_id:
                params['protocol_id'] = protocol_id
            if start_date:
                params['start_date'] = start_date
            if end_date:
                params['end_date'] = end_date
            
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            # Convert to DataFrame
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict) and 'data' in data:
                df = pd.DataFrame(data['data'])
            else:
                df = pd.DataFrame([data])
            
            # Convert date columns if present
            date_columns = ['event_date', 'created_date', 'updated_date', 'date']
            for col in date_columns:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
            
            return df
            
        except Exception as e:
            print(f"Error getting events: {str(e)}")
            return pd.DataFrame()
    
    def get_adverse_events(self, protocol_id: str = None, 
                          severity: str = None, limit: int = 100) -> pd.DataFrame:
        """
        Get adverse events data from Oncore
        
        Args:
            protocol_id: Specific protocol ID (optional)
            severity: Filter by severity level (optional)
            limit: Maximum number of records to return
            
        Returns:
            DataFrame with adverse events data
        """
        if not self.connected:
            return pd.DataFrame()
        
        try:
            url = urljoin(self.base_url, '/api/adverse-events')
            params = {'limit': limit}
            
            if protocol_id:
                params['protocol_id'] = protocol_id
            if severity:
                params['severity'] = severity
            
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            # Convert to DataFrame
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict) and 'data' in data:
                df = pd.DataFrame(data['data'])
            else:
                df = pd.DataFrame([data])
            
            # Convert date columns if present
            date_columns = ['event_date', 'reported_date', 'created_date']
            for col in date_columns:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
            
            return df
            
        except Exception as e:
            print(f"Error getting adverse events: {str(e)}")
            return pd.DataFrame()
    
    def get_protocol_summary(self, protocol_id: str) -> Dict[str, Any]:
        """
        Get summary statistics for a protocol
        
        Args:
            protocol_id: Protocol ID
            
        Returns:
            Dictionary with protocol summary
        """
        if not self.connected:
            return {}
        
        try:
            # Get protocol details
            protocol_url = urljoin(self.base_url, f'/api/protocols/{protocol_id}')
            protocol_response = self.session.get(protocol_url)
            protocol_response.raise_for_status()
            protocol_data = protocol_response.json()
            
            # Get subjects count
            subjects_df = self.get_subjects(protocol_id=protocol_id)
            subjects_count = len(subjects_df)
            
            # Get events count
            events_df = self.get_events(protocol_id=protocol_id)
            events_count = len(events_df)
            
            # Get adverse events count
            ae_df = self.get_adverse_events(protocol_id=protocol_id)
            ae_count = len(ae_df)
            
            summary = {
                'protocol_id': protocol_id,
                'protocol_name': protocol_data.get('name', 'Unknown'),
                'subjects_count': subjects_count,
                'events_count': events_count,
                'adverse_events_count': ae_count,
                'last_updated': datetime.now().isoformat()
            }
            
            return summary
            
        except Exception as e:
            print(f"Error getting protocol summary: {str(e)}")
            return {}
    
    def get_data_quality_report(self) -> Dict[str, Any]:
        """
        Get data quality report for Oncore data
        
        Returns:
            Dictionary with data quality metrics
        """
        if not self.connected:
            return {}
        
        try:
            # Get all protocols
            protocols_df = self.get_protocols(limit=1000)
            
            quality_report = {
                'total_protocols': len(protocols_df),
                'protocols_with_subjects': 0,
                'protocols_with_events': 0,
                'protocols_with_ae': 0,
                'data_completeness': 0,
                'last_updated': datetime.now().isoformat()
            }
            
            if not protocols_df.empty:
                # Check each protocol
                for _, protocol in protocols_df.iterrows():
                    protocol_id = protocol.get('id') or protocol.get('protocol_id')
                    
                    if protocol_id:
                        # Check if protocol has subjects
                        subjects_df = self.get_subjects(protocol_id=protocol_id, limit=1)
                        if not subjects_df.empty:
                            quality_report['protocols_with_subjects'] += 1
                        
                        # Check if protocol has events
                        events_df = self.get_events(protocol_id=protocol_id, limit=1)
                        if not events_df.empty:
                            quality_report['protocols_with_events'] += 1
                        
                        # Check if protocol has adverse events
                        ae_df = self.get_adverse_events(protocol_id=protocol_id, limit=1)
                        if not ae_df.empty:
                            quality_report['protocols_with_ae'] += 1
                
                # Calculate data completeness
                total_protocols = quality_report['total_protocols']
                if total_protocols > 0:
                    completeness = (
                        quality_report['protocols_with_subjects'] +
                        quality_report['protocols_with_events'] +
                        quality_report['protocols_with_ae']
                    ) / (total_protocols * 3) * 100
                    quality_report['data_completeness'] = round(completeness, 2)
            
            return quality_report
            
        except Exception as e:
            print(f"Error getting data quality report: {str(e)}")
            return {}
    
    def export_data(self, data_type: str, **kwargs) -> pd.DataFrame:
        """
        Generic method to export data from Oncore
        
        Args:
            data_type: Type of data to export ('protocols', 'subjects', 'events', 'adverse_events')
            **kwargs: Additional parameters for the specific data type
            
        Returns:
            DataFrame with exported data
        """
        if not self.connected:
            return pd.DataFrame()
        
        try:
            if data_type == 'protocols':
                return self.get_protocols(**kwargs)
            elif data_type == 'subjects':
                return self.get_subjects(**kwargs)
            elif data_type == 'events':
                return self.get_events(**kwargs)
            elif data_type == 'adverse_events':
                return self.get_adverse_events(**kwargs)
            else:
                print(f"Unknown data type: {data_type}")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"Error exporting {data_type}: {str(e)}")
            return pd.DataFrame()
    
    def disconnect(self):
        """Disconnect from Oncore"""
        if self.session:
            self.session.close()
            self.session = None
        self.connected = False
    
    def __del__(self):
        """Cleanup connection on object destruction"""
        self.disconnect()
