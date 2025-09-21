"""
Visualization Module
Handles data visualization using Matplotlib, Plotly, and Seaborn
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, List, Dict, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

class Visualizer:
    """Handles data visualization operations"""
    
    def __init__(self):
        """Initialize the visualizer with default settings"""
        self.set_style()
        
    def set_style(self):
        """Set default plotting style"""
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def create_sales_trend_chart(self, data: pd.DataFrame, date_col: str = 'date', 
                                value_col: str = 'total_sales') -> go.Figure:
        """Create sales trend chart over time"""
        try:
            # Ensure date column is datetime
            data[date_col] = pd.to_datetime(data[date_col])
            
            # Group by date and sum sales
            daily_sales = data.groupby(date_col)[value_col].sum().reset_index()
            
            fig = px.line(
                daily_sales, 
                x=date_col, 
                y=value_col,
                title="Sales Trend Over Time",
                labels={value_col: "Total Sales", date_col: "Date"}
            )
            
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Total Sales ($)",
                hovermode='x unified'
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating sales trend chart: {str(e)}")
            return go.Figure()
    
    def create_category_analysis(self, data: pd.DataFrame, category_col: str = 'category',
                               value_col: str = 'total_sales') -> go.Figure:
        """Create category analysis chart"""
        try:
            # Group by category
            category_data = data.groupby(category_col)[value_col].sum().reset_index()
            category_data = category_data.sort_values(value_col, ascending=False)
            
            fig = px.bar(
                category_data,
                x=category_col,
                y=value_col,
                title="Sales by Category",
                labels={value_col: "Total Sales", category_col: "Category"},
                color=value_col,
                color_continuous_scale="viridis"
            )
            
            fig.update_layout(
                xaxis_title="Category",
                yaxis_title="Total Sales ($)",
                xaxis_tickangle=-45
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating category analysis: {str(e)}")
            return go.Figure()
    
    def create_customer_segment_analysis(self, data: pd.DataFrame, 
                                       segment_col: str = 'customer_segment',
                                       value_col: str = 'total_sales') -> go.Figure:
        """Create customer segment analysis pie chart"""
        try:
            segment_data = data.groupby(segment_col)[value_col].sum().reset_index()
            
            fig = px.pie(
                segment_data,
                values=value_col,
                names=segment_col,
                title="Sales Distribution by Customer Segment",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            
            fig.update_traces(textposition='inside', textinfo='percent+label')
            
            return fig
            
        except Exception as e:
            print(f"Error creating customer segment analysis: {str(e)}")
            return go.Figure()
    
    def create_geographic_analysis(self, data: pd.DataFrame, region_col: str = 'region',
                                 value_col: str = 'total_sales') -> go.Figure:
        """Create geographic analysis chart"""
        try:
            region_data = data.groupby(region_col)[value_col].sum().reset_index()
            
            fig = px.bar(
                region_data,
                x=region_col,
                y=value_col,
                title="Sales by Region",
                labels={value_col: "Total Sales", region_col: "Region"},
                color=value_col,
                color_continuous_scale="blues"
            )
            
            fig.update_layout(
                xaxis_title="Region",
                yaxis_title="Total Sales ($)"
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating geographic analysis: {str(e)}")
            return go.Figure()
    
    def create_correlation_heatmap(self, data: pd.DataFrame) -> go.Figure:
        """Create correlation heatmap for numeric columns"""
        try:
            numeric_data = data.select_dtypes(include=[np.number])
            
            if numeric_data.empty:
                return go.Figure()
            
            corr_matrix = numeric_data.corr()
            
            fig = px.imshow(
                corr_matrix,
                title="Correlation Heatmap",
                color_continuous_scale="RdBu",
                aspect="auto"
            )
            
            fig.update_layout(
                xaxis_title="Variables",
                yaxis_title="Variables"
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating correlation heatmap: {str(e)}")
            return go.Figure()
    
    def create_top_products_chart(self, data: pd.DataFrame, product_col: str = 'product_name',
                                value_col: str = 'total_sales', top_n: int = 10) -> go.Figure:
        """Create top products chart"""
        try:
            product_data = data.groupby(product_col)[value_col].sum().reset_index()
            top_products = product_data.nlargest(top_n, value_col)
            
            fig = px.bar(
                top_products,
                x=value_col,
                y=product_col,
                orientation='h',
                title=f"Top {top_n} Products by Sales",
                labels={value_col: "Total Sales", product_col: "Product"},
                color=value_col,
                color_continuous_scale="viridis"
            )
            
            fig.update_layout(
                xaxis_title="Total Sales ($)",
                yaxis_title="Product",
                yaxis={'categoryorder': 'total ascending'}
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating top products chart: {str(e)}")
            return go.Figure()
    
    def create_sales_rep_performance(self, data: pd.DataFrame, rep_col: str = 'sales_rep',
                                   value_col: str = 'total_sales') -> go.Figure:
        """Create sales representative performance chart"""
        try:
            rep_data = data.groupby(rep_col)[value_col].sum().reset_index()
            rep_data = rep_data.sort_values(value_col, ascending=False)
            
            fig = px.bar(
                rep_data,
                x=rep_col,
                y=value_col,
                title="Sales Representative Performance",
                labels={value_col: "Total Sales", rep_col: "Sales Representative"},
                color=value_col,
                color_continuous_scale="greens"
            )
            
            fig.update_layout(
                xaxis_title="Sales Representative",
                yaxis_title="Total Sales ($)"
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating sales rep performance chart: {str(e)}")
            return go.Figure()
    
    def create_monthly_trend(self, data: pd.DataFrame, date_col: str = 'date',
                           value_col: str = 'total_sales') -> go.Figure:
        """Create monthly trend analysis"""
        try:
            data[date_col] = pd.to_datetime(data[date_col])
            data['month'] = data[date_col].dt.to_period('M')
            
            monthly_data = data.groupby('month')[value_col].sum().reset_index()
            monthly_data['month_str'] = monthly_data['month'].astype(str)
            
            fig = px.line(
                monthly_data,
                x='month_str',
                y=value_col,
                title="Monthly Sales Trend",
                labels={value_col: "Total Sales", 'month_str': "Month"}
            )
            
            fig.update_layout(
                xaxis_title="Month",
                yaxis_title="Total Sales ($)",
                xaxis_tickangle=-45
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating monthly trend: {str(e)}")
            return go.Figure()
    
    def create_dashboard_summary(self, data: pd.DataFrame) -> List[go.Figure]:
        """Create a comprehensive dashboard with multiple charts"""
        charts = []
        
        try:
            # Sales trend
            if 'date' in data.columns and 'total_sales' in data.columns:
                charts.append(self.create_sales_trend_chart(data))
            
            # Category analysis
            if 'category' in data.columns and 'total_sales' in data.columns:
                charts.append(self.create_category_analysis(data))
            
            # Customer segment analysis
            if 'customer_segment' in data.columns and 'total_sales' in data.columns:
                charts.append(self.create_customer_segment_analysis(data))
            
            # Geographic analysis
            if 'region' in data.columns and 'total_sales' in data.columns:
                charts.append(self.create_geographic_analysis(data))
            
            # Top products
            if 'product_name' in data.columns and 'total_sales' in data.columns:
                charts.append(self.create_top_products_chart(data))
            
            # Sales rep performance
            if 'sales_rep' in data.columns and 'total_sales' in data.columns:
                charts.append(self.create_sales_rep_performance(data))
            
            # Correlation heatmap
            charts.append(self.create_correlation_heatmap(data))
            
        except Exception as e:
            print(f"Error creating dashboard summary: {str(e)}")
        
        return charts
    
    def create_custom_chart(self, data: pd.DataFrame, chart_type: str, 
                          x_col: str, y_col: str, **kwargs) -> go.Figure:
        """Create custom chart based on parameters"""
        try:
            if chart_type == 'bar':
                fig = px.bar(data, x=x_col, y=y_col, **kwargs)
            elif chart_type == 'line':
                fig = px.line(data, x=x_col, y=y_col, **kwargs)
            elif chart_type == 'scatter':
                fig = px.scatter(data, x=x_col, y=y_col, **kwargs)
            elif chart_type == 'pie':
                fig = px.pie(data, values=y_col, names=x_col, **kwargs)
            elif chart_type == 'histogram':
                fig = px.histogram(data, x=x_col, **kwargs)
            else:
                raise ValueError(f"Unsupported chart type: {chart_type}")
            
            return fig
            
        except Exception as e:
            print(f"Error creating custom chart: {str(e)}")
            return go.Figure()
    
    def create_matplotlib_plot(self, data: pd.DataFrame, plot_type: str = 'line',
                             x_col: str = None, y_col: str = None) -> plt.Figure:
        """Create matplotlib plot for advanced customization"""
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            if plot_type == 'line' and x_col and y_col:
                ax.plot(data[x_col], data[y_col])
                ax.set_xlabel(x_col)
                ax.set_ylabel(y_col)
                ax.set_title(f"{y_col} vs {x_col}")
            
            elif plot_type == 'bar' and x_col and y_col:
                ax.bar(data[x_col], data[y_col])
                ax.set_xlabel(x_col)
                ax.set_ylabel(y_col)
                ax.set_title(f"{y_col} by {x_col}")
                plt.xticks(rotation=45)
            
            elif plot_type == 'histogram' and x_col:
                ax.hist(data[x_col].dropna(), bins=30)
                ax.set_xlabel(x_col)
                ax.set_ylabel('Frequency')
                ax.set_title(f"Distribution of {x_col}")
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            print(f"Error creating matplotlib plot: {str(e)}")
            return plt.figure()
    
    def get_available_visualizations(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Get list of available visualizations based on data columns"""
        visualizations = []
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
        datetime_cols = data.select_dtypes(include=['datetime64']).columns.tolist()
        
        # Sales trend (if date and sales columns exist)
        if any('date' in col.lower() for col in data.columns) and 'total_sales' in data.columns:
            visualizations.append({
                'name': 'Sales Trend',
                'type': 'line',
                'description': 'Show sales trend over time',
                'required_columns': ['date', 'total_sales']
            })
        
        # Category analysis
        if any('categor' in col.lower() for col in data.columns) and 'total_sales' in data.columns:
            visualizations.append({
                'name': 'Category Analysis',
                'type': 'bar',
                'description': 'Analyze sales by category',
                'required_columns': ['category', 'total_sales']
            })
        
        # Correlation heatmap
        if len(numeric_cols) > 1:
            visualizations.append({
                'name': 'Correlation Heatmap',
                'type': 'heatmap',
                'description': 'Show correlations between numeric variables',
                'required_columns': numeric_cols
            })
        
        # Distribution plots
        for col in numeric_cols:
            visualizations.append({
                'name': f'{col} Distribution',
                'type': 'histogram',
                'description': f'Show distribution of {col}',
                'required_columns': [col]
            })
        
        return visualizations
    
    def create_auto_visualization(self, data: pd.DataFrame, query: str = "") -> Optional[go.Figure]:
        """Automatically create the best visualization for the data and query"""
        try:
            if data.empty:
                return None
            
            # Determine the best chart type based on data and query
            numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
            categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
            
            # If only one numeric column and multiple rows, create a bar chart
            if len(numeric_cols) == 1 and len(categorical_cols) >= 1 and len(data) <= 20:
                cat_col = categorical_cols[0]
                num_col = numeric_cols[0]
                
                fig = px.bar(
                    data, 
                    x=cat_col, 
                    y=num_col,
                    title=f"{num_col} by {cat_col}",
                    color=num_col,
                    color_continuous_scale='viridis'
                )
                fig.update_layout(
                    xaxis_tickangle=-45,
                    showlegend=False
                )
                return fig
            
            # If two numeric columns, create scatter plot
            elif len(numeric_cols) >= 2 and len(data) <= 1000:
                x_col = numeric_cols[0]
                y_col = numeric_cols[1]
                
                fig = px.scatter(
                    data, 
                    x=x_col, 
                    y=y_col,
                    title=f"{y_col} vs {x_col}",
                    color=y_col,
                    color_continuous_scale='viridis'
                )
                return fig
            
            # If one numeric column and many rows, create histogram
            elif len(numeric_cols) == 1 and len(data) > 20:
                num_col = numeric_cols[0]
                
                fig = px.histogram(
                    data, 
                    x=num_col,
                    title=f"Distribution of {num_col}",
                    nbins=min(30, len(data)//10 + 1)
                )
                return fig
            
            # If categorical data only, create pie chart
            elif len(categorical_cols) >= 1 and len(numeric_cols) == 0 and len(data) <= 10:
                cat_col = categorical_cols[0]
                value_counts = data[cat_col].value_counts()
                
                fig = px.pie(
                    values=value_counts.values,
                    names=value_counts.index,
                    title=f"Distribution of {cat_col}"
                )
                return fig
            
            # Default: show first few rows as table
            else:
                # Create a simple table visualization
                fig = go.Figure(data=[go.Table(
                    header=dict(values=list(data.columns),
                              fill_color='paleturquoise',
                              align='left'),
                    cells=dict(values=[data[col] for col in data.columns],
                             fill_color='lavender',
                             align='left'))
                ])
                fig.update_layout(title="Data Table")
                return fig
                
        except Exception as e:
            print(f"Error creating auto visualization: {str(e)}")
            return None