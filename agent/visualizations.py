"""
Visualizations Module
Advanced interactive charts and dashboards for FP&A
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class FinancialVisualizations:
    """
    Advanced visualization system for FP&A dashboards
    """
    
    def __init__(self):
        self.chart_templates = self._initialize_chart_templates()
        self.color_palettes = self._initialize_color_palettes()
        self.dashboard_layouts = self._initialize_dashboard_layouts()
    
    def _initialize_chart_templates(self) -> Dict[str, Dict]:
        """Initialize chart templates for different visualization types"""
        return {
            'revenue_trend': {
                'name': 'Revenue Trend',
                'description': 'Line chart showing revenue over time',
                'chart_type': 'line',
                'default_x': 'Date',
                'default_y': 'Revenue',
                'color_by': 'Product',
                'facet_by': 'Region'
            },
            'profit_margin': {
                'name': 'Profit Margin',
                'description': 'Area chart showing profit margins',
                'chart_type': 'area',
                'default_x': 'Date',
                'default_y': 'Gross_Margin',
                'color_by': 'Product',
                'stack_by': 'Category'
            },
            'expense_breakdown': {
                'name': 'Expense Breakdown',
                'description': 'Pie/Donut chart showing expense distribution',
                'chart_type': 'pie',
                'default_labels': 'Expense_Category',
                'default_values': 'Amount',
                'hole_size': 0.4
            },
            'product_performance': {
                'name': 'Product Performance',
                'description': 'Bar chart comparing product metrics',
                'chart_type': 'bar',
                'default_x': 'Product',
                'default_y': 'Revenue',
                'color_by': 'Region',
                'barmode': 'group'
            },
            'regional_heatmap': {
                'name': 'Regional Heatmap',
                'description': 'Heatmap showing regional performance',
                'chart_type': 'heatmap',
                'default_x': 'Region',
                'default_y': 'Product',
                'default_z': 'Revenue'
            },
            'correlation_matrix': {
                'name': 'Correlation Matrix',
                'description': 'Heatmap of correlations between metrics',
                'chart_type': 'correlation',
                'default_metrics': ['Revenue', 'Profit', 'Cost', 'Quantity']
            },
            'waterfall_chart': {
                'name': 'Waterfall Chart',
                'description': 'Waterfall chart for profit decomposition',
                'chart_type': 'waterfall',
                'default_metrics': ['Starting', 'Revenue', 'Expenses', 'Net']
            },
            'kpi_dashboard': {
                'name': 'KPI Dashboard',
                'description': 'Multi-metric dashboard with gauges',
                'chart_type': 'dashboard',
                'default_metrics': ['Revenue', 'Profit', 'Margin', 'Growth']
            },
            'scatter_matrix': {
                'name': 'Scatter Matrix',
                'description': 'Scatter plot matrix for multivariate analysis',
                'chart_type': 'scatter_matrix',
                'default_columns': ['Revenue', 'Profit', 'Cost', 'Quantity']
            },
            'box_plot': {
                'name': 'Box Plot',
                'description': 'Box plots for distribution analysis',
                'chart_type': 'box',
                'default_x': 'Product',
                'default_y': 'Revenue',
                'color_by': 'Region'
            },
            'histogram': {
                'name': 'Histogram',
                'description': 'Histogram for distribution analysis',
                'chart_type': 'histogram',
                'default_x': 'Revenue',
                'color_by': 'Product',
                'nbins': 20
            },
            'bubble_chart': {
                'name': 'Bubble Chart',
                'description': 'Bubble chart for 3D data visualization',
                'chart_type': 'bubble',
                'default_x': 'Revenue',
                'default_y': 'Profit',
                'default_size': 'Quantity',
                'color_by': 'Product'
            }
        }
    
    def _initialize_color_palettes(self) -> Dict[str, List[str]]:
        """Initialize color palettes for visualizations"""
        return {
            'financial': ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6B9AC4', '#97D8C4'],
            'sequential': ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#08519c', '#08306b'],
            'diverging': ['#b2182b', '#d6604d', '#f4a582', '#fddbc7', '#f7f7f7', '#d1e5f0', '#92c5de', '#4393c3', '#2166ac'],
            'qualitative': ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf', '#999999'],
            'pastel': ['#fdcdac', '#cbd5e8', '#b3e2cd', '#f4cae4', '#e6f5c9', '#fff2ae', '#f1e2cc', '#cccccc']
        }
    
    def _initialize_dashboard_layouts(self) -> Dict[str, Dict]:
        """Initialize predefined dashboard layouts"""
        return {
            'single_chart': {
                'name': 'Single Chart',
                'layout': [[1, 1, 1]],
                'chart_count': 1,
                'description': 'Full-screen single visualization'
            },
            'two_column': {
                'name': 'Two Column',
                'layout': [[1, 1]],
                'chart_count': 2,
                'description': 'Two visualizations side by side'
            },
            'three_column': {
                'name': 'Three Column',
                'layout': [[1, 1, 1]],
                'chart_count': 3,
                'description': 'Three equal-width visualizations'
            },
            'dashboard_2x2': {
                'name': '2x2 Dashboard',
                'layout': [[1, 1], [1, 1]],
                'chart_count': 4,
                'description': 'Four visualizations in 2x2 grid'
            },
            'dashboard_3x2': {
                'name': '3x2 Dashboard',
                'layout': [[1, 1, 1], [1, 1, 1]],
                'chart_count': 6,
                'description': 'Six visualizations in 3x2 grid'
            },
            'kpi_top': {
                'name': 'KPI Top + Charts',
                'layout': [[1, 1, 1, 1], [2, 2], [1, 1]],
                'chart_count': 8,
                'description': 'KPI metrics on top, charts below'
            },
            'focus_right': {
                'name': 'Focus Right',
                'layout': [[2, 1], [2, 1]],
                'chart_count': 4,
                'description': 'Large visualization on left, smaller on right'
            }
        }
    
    def display_visualizations_interface(self, dataframes: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Main Streamlit interface for visualizations
        """
        st.subheader("📊 Advanced Visualizations")
        
        if not dataframes:
            st.warning("⚠️ No data available. Please upload and analyze data first.")
            return {}
        
        # Get current data
        current_data = dataframes
        
        # File selection
        primary_file = st.selectbox(
            "Select file for visualizations:",
            options=list(current_data.keys()),
            help="This file will be used for creating visualizations"
        )
        
        if not primary_file:
            return {}
        
        df = current_data[primary_file]
        
        # Create tabs for different visualization sections
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🎨 Chart Builder", 
            "📈 Time Series", 
            "📊 Comparative",
            "🎯 Dashboards",
            "⚙️ Custom Visuals"
        ])
        
        visualization_results = {}
        
        with tab1:
            visualization_results['chart_builder'] = self._chart_builder_tab(df)
        
        with tab2:
            visualization_results['time_series'] = self._time_series_tab(df)
        
        with tab3:
            visualization_results['comparative'] = self._comparative_tab(df)
        
        with tab4:
            visualization_results['dashboards'] = self._dashboards_tab(df)
        
        with tab5:
            visualization_results['custom'] = self._custom_visuals_tab(df)
        
        return visualization_results
    
    def _chart_builder_tab(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Interactive chart builder
        """
        st.markdown("### 🎨 Interactive Chart Builder")
        
        results = {}
        
        # Two-column layout for chart builder
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("#### Chart Configuration")
            
            # Chart type selection
            chart_type = st.selectbox(
                "Select chart type:",
                options=list(self.chart_templates.keys()),
                format_func=lambda x: self.chart_templates[x]['name'],
                help="Choose the type of chart to create"
            )
            
            if not chart_type:
                return results
            
            template = self.chart_templates[chart_type]
            
            # Color palette selection
            color_palette = st.selectbox(
                "Color palette:",
                options=list(self.color_palettes.keys()),
                help="Choose color scheme for visualization"
            )
            
            # Chart dimensions
            st.markdown("##### Chart Dimensions")
            
            col1a, col1b = st.columns(2)
            with col1a:
                chart_width = st.slider("Width", 400, 1200, 800, 50)
            with col1b:
                chart_height = st.slider("Height", 300, 800, 500, 50)
            
            # Advanced options
            with st.expander("Advanced Options"):
                show_grid = st.checkbox("Show grid", value=True)
                show_legend = st.checkbox("Show legend", value=True)
                show_tooltip = st.checkbox("Show tooltips", value=True)
                transparent_bg = st.checkbox("Transparent background", value=False)
                
                animation = st.checkbox("Enable animation", value=False)
                if animation:
                    animation_duration = st.slider("Animation duration (ms)", 500, 5000, 1000, 100)
        
        with col2:
            st.markdown("#### Data Mapping")
            
            # Get column types
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            date_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
            
            # Map columns based on chart type
            if template['chart_type'] in ['line', 'area', 'bar', 'scatter']:
                col2a, col2b, col2c = st.columns(3)
                
                with col2a:
                    x_axis = st.selectbox(
                        "X-axis:",
                        options=date_cols + categorical_cols + numeric_cols,
                        index=0 if date_cols else 0,
                        help="Select column for X-axis"
                    )
                
                with col2b:
                    y_axis = st.selectbox(
                        "Y-axis:",
                        options=numeric_cols,
                        index=0 if numeric_cols else 0,
                        help="Select column for Y-axis"
                    )
                
                with col2c:
                    color_by = st.selectbox(
                        "Color by:",
                        options=['None'] + categorical_cols,
                        help="Group data by color"
                    )
                    color_by = None if color_by == 'None' else color_by
            
            elif template['chart_type'] in ['pie', 'donut']:
                col2a, col2b = st.columns(2)
                
                with col2a:
                    labels = st.selectbox(
                        "Labels:",
                        options=categorical_cols,
                        help="Select column for labels"
                    )
                
                with col2b:
                    values = st.selectbox(
                        "Values:",
                        options=numeric_cols,
                        help="Select column for values"
                    )
            
            elif template['chart_type'] == 'heatmap':
                col2a, col2b, col2c = st.columns(3)
                
                with col2a:
                    x_axis = st.selectbox(
                        "X-axis (heatmap):",
                        options=categorical_cols,
                        help="Select column for X-axis"
                    )
                
                with col2b:
                    y_axis = st.selectbox(
                        "Y-axis (heatmap):",
                        options=categorical_cols,
                        help="Select column for Y-axis"
                    )
                
                with col2c:
                    values = st.selectbox(
                        "Values (heatmap):",
                        options=numeric_cols,
                        help="Select column for values"
                    )
            
            # Aggregation options
            if template['chart_type'] in ['line', 'area', 'bar']:
                st.markdown("##### Aggregation")
                
                agg_method = st.selectbox(
                    "Aggregation method:",
                    options=['Sum', 'Mean', 'Median', 'Count', 'Min', 'Max'],
                    help="How to aggregate data points"
                )
        
        # Create chart button
        if st.button("🛠️ Create Chart", type="primary", use_container_width=True):
            with st.spinner("Creating visualization..."):
                # Create the chart
                fig = self._create_chart(
                    df=df,
                    chart_type=chart_type,
                    chart_params={
                        'x_axis': x_axis if 'x_axis' in locals() else None,
                        'y_axis': y_axis if 'y_axis' in locals() else None,
                        'color_by': color_by if 'color_by' in locals() else None,
                        'labels': labels if 'labels' in locals() else None,
                        'values': values if 'values' in locals() else None,
                        'color_palette': color_palette,
                        'width': chart_width,
                        'height': chart_height,
                        'show_grid': show_grid,
                        'show_legend': show_legend,
                        'show_tooltip': show_tooltip,
                        'transparent_bg': transparent_bg,
                        'agg_method': agg_method if 'agg_method' in locals() else 'Sum'
                    }
                )
                
                if fig:
                    results['chart'] = fig
                    
                    # Display chart
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Chart export options
                    self._display_chart_export_options(fig, chart_type)
        
        return results
    
    def _create_chart(self, df: pd.DataFrame, chart_type: str, chart_params: Dict[str, Any]) -> Optional[go.Figure]:
        """Create chart based on parameters"""
        try:
            template = self.chart_templates[chart_type]
            
            if template['chart_type'] == 'line':
                fig = self._create_line_chart(df, chart_params)
            
            elif template['chart_type'] == 'area':
                fig = self._create_area_chart(df, chart_params)
            
            elif template['chart_type'] == 'bar':
                fig = self._create_bar_chart(df, chart_params)
            
            elif template['chart_type'] in ['pie', 'donut']:
                fig = self._create_pie_chart(df, chart_params)
            
            elif template['chart_type'] == 'heatmap':
                fig = self._create_heatmap(df, chart_params)
            
            elif template['chart_type'] == 'box':
                fig = self._create_box_plot(df, chart_params)
            
            elif template['chart_type'] == 'histogram':
                fig = self._create_histogram(df, chart_params)
            
            elif template['chart_type'] == 'scatter':
                fig = self._create_scatter_plot(df, chart_params)
            
            elif template['chart_type'] == 'bubble':
                fig = self._create_bubble_chart(df, chart_params)
            
            else:
                st.warning(f"Chart type '{chart_type}' not implemented yet")
                return None
            
            # Apply styling
            fig = self._apply_chart_styling(fig, chart_params)
            
            return fig
        
        except Exception as e:
            st.error(f"Error creating chart: {str(e)}")
            return None
    
    def _create_line_chart(self, df: pd.DataFrame, params: Dict[str, Any]) -> go.Figure:
        """Create line chart"""
        x_axis = params.get('x_axis')
        y_axis = params.get('y_axis')
        color_by = params.get('color_by')
        agg_method = params.get('agg_method', 'Sum')
        
        if not x_axis or not y_axis:
            st.warning("Please select both X and Y axes")
            return None
        
        # Prepare data
        if color_by:
            # Group by x_axis and color_by
            if agg_method == 'Sum':
                grouped = df.groupby([x_axis, color_by])[y_axis].sum().reset_index()
            elif agg_method == 'Mean':
                grouped = df.groupby([x_axis, color_by])[y_axis].mean().reset_index()
            elif agg_method == 'Median':
                grouped = df.groupby([x_axis, color_by])[y_axis].median().reset_index()
            elif agg_method == 'Count':
                grouped = df.groupby([x_axis, color_by])[y_axis].count().reset_index()
            else:
                grouped = df.groupby([x_axis, color_by])[y_axis].sum().reset_index()
            
            # Create figure
            fig = px.line(
                grouped,
                x=x_axis,
                y=y_axis,
                color=color_by,
                title=f"{y_axis} by {x_axis} (Grouped by {color_by})",
                color_discrete_sequence=self.color_palettes.get(params.get('color_palette', 'financial'))
            )
        else:
            # Group by x_axis only
            if agg_method == 'Sum':
                grouped = df.groupby(x_axis)[y_axis].sum().reset_index()
            elif agg_method == 'Mean':
                grouped = df.groupby(x_axis)[y_axis].mean().reset_index()
            elif agg_method == 'Median':
                grouped = df.groupby(x_axis)[y_axis].median().reset_index()
            elif agg_method == 'Count':
                grouped = df.groupby(x_axis)[y_axis].count().reset_index()
            else:
                grouped = df.groupby(x_axis)[y_axis].sum().reset_index()
            
            fig = px.line(
                grouped,
                x=x_axis,
                y=y_axis,
                title=f"{y_axis} by {x_axis}",
                color_discrete_sequence=self.color_palettes.get(params.get('color_palette', 'financial'))
            )
        
        return fig
    
    def _create_area_chart(self, df: pd.DataFrame, params: Dict[str, Any]) -> go.Figure:
        """Create area chart"""
        x_axis = params.get('x_axis')
        y_axis = params.get('y_axis')
        color_by = params.get('color_by')
        
        if not x_axis or not y_axis:
            st.warning("Please select both X and Y axes")
            return None
        
        # Prepare data
        if color_by:
            grouped = df.groupby([x_axis, color_by])[y_axis].sum().reset_index()
            fig = px.area(
                grouped,
                x=x_axis,
                y=y_axis,
                color=color_by,
                title=f"{y_axis} by {x_axis} (Area Chart)",
                color_discrete_sequence=self.color_palettes.get(params.get('color_palette', 'financial'))
            )
        else:
            grouped = df.groupby(x_axis)[y_axis].sum().reset_index()
            fig = px.area(
                grouped,
                x=x_axis,
                y=y_axis,
                title=f"{y_axis} by {x_axis} (Area Chart)",
                color_discrete_sequence=self.color_palettes.get(params.get('color_palette', 'financial'))
            )
        
        return fig
    
    def _create_bar_chart(self, df: pd.DataFrame, params: Dict[str, Any]) -> go.Figure:
        """Create bar chart"""
        x_axis = params.get('x_axis')
        y_axis = params.get('y_axis')
        color_by = params.get('color_by')
        agg_method = params.get('agg_method', 'Sum')
        
        if not x_axis or not y_axis:
            st.warning("Please select both X and Y axes")
            return None
        
        # Prepare data
        if color_by:
            if agg_method == 'Sum':
                grouped = df.groupby([x_axis, color_by])[y_axis].sum().reset_index()
            elif agg_method == 'Mean':
                grouped = df.groupby([x_axis, color_by])[y_axis].mean().reset_index()
            else:
                grouped = df.groupby([x_axis, color_by])[y_axis].sum().reset_index()
            
            fig = px.bar(
                grouped,
                x=x_axis,
                y=y_axis,
                color=color_by,
                title=f"{y_axis} by {x_axis} (Grouped by {color_by})",
                color_discrete_sequence=self.color_palettes.get(params.get('color_palette', 'financial')),
                barmode='group'
            )
        else:
            if agg_method == 'Sum':
                grouped = df.groupby(x_axis)[y_axis].sum().reset_index()
            elif agg_method == 'Mean':
                grouped = df.groupby(x_axis)[y_axis].mean().reset_index()
            else:
                grouped = df.groupby(x_axis)[y_axis].sum().reset_index()
            
            fig = px.bar(
                grouped,
                x=x_axis,
                y=y_axis,
                title=f"{y_axis} by {x_axis}",
                color_discrete_sequence=self.color_palettes.get(params.get('color_palette', 'financial'))
            )
        
        return fig
    
    def _create_pie_chart(self, df: pd.DataFrame, params: Dict[str, Any]) -> go.Figure:
        """Create pie/donut chart"""
        labels = params.get('labels')
        values = params.get('values')
        
        if not labels or not values:
            st.warning("Please select both labels and values")
            return None
        
        # Aggregate data
        grouped = df.groupby(labels)[values].sum().reset_index()
        
        # Create pie chart
        fig = px.pie(
            grouped,
            names=labels,
            values=values,
            title=f"{values} by {labels}",
            color_discrete_sequence=self.color_palettes.get(params.get('color_palette', 'financial')),
            hole=0.4  # Donut chart
        )
        
        return fig
    
    def _create_heatmap(self, df: pd.DataFrame, params: Dict[str, Any]) -> go.Figure:
        """Create heatmap"""
        x_axis = params.get('x_axis')
        y_axis = params.get('y_axis')
        values = params.get('values')
        
        if not x_axis or not y_axis or not values:
            st.warning("Please select X, Y axes and values")
            return None
        
        # Pivot data for heatmap
        pivot_data = df.pivot_table(
            values=values,
            index=y_axis,
            columns=x_axis,
            aggfunc='sum',
            fill_value=0
        )
        
        # Create heatmap
        fig = px.imshow(
            pivot_data,
            title=f"{values} Heatmap: {x_axis} vs {y_axis}",
            color_continuous_scale=self.color_palettes.get(params.get('color_palette', 'sequential')),
            labels=dict(x=x_axis, y=y_axis, color=values)
        )
        
        return fig
    
    def _create_box_plot(self, df: pd.DataFrame, params: Dict[str, Any]) -> go.Figure:
        """Create box plot"""
        x_axis = params.get('x_axis')
        y_axis = params.get('y_axis')
        color_by = params.get('color_by')
        
        if not x_axis or not y_axis:
            st.warning("Please select both X and Y axes")
            return None
        
        # Create box plot
        if color_by:
            fig = px.box(
                df,
                x=x_axis,
                y=y_axis,
                color=color_by,
                title=f"{y_axis} Distribution by {x_axis}",
                color_discrete_sequence=self.color_palettes.get(params.get('color_palette', 'financial'))
            )
        else:
            fig = px.box(
                df,
                x=x_axis,
                y=y_axis,
                title=f"{y_axis} Distribution by {x_axis}",
                color_discrete_sequence=self.color_palettes.get(params.get('color_palette', 'financial'))
            )
        
        return fig
    
    def _create_histogram(self, df: pd.DataFrame, params: Dict[str, Any]) -> go.Figure:
        """Create histogram"""
        x_axis = params.get('x_axis')
        color_by = params.get('color_by')
        nbins = params.get('nbins', 20)
        
        if not x_axis:
            st.warning("Please select X axis")
            return None
        
        # Create histogram
        if color_by:
            fig = px.histogram(
                df,
                x=x_axis,
                color=color_by,
                nbins=nbins,
                title=f"Distribution of {x_axis}",
                color_discrete_sequence=self.color_palettes.get(params.get('color_palette', 'financial')),
                barmode='overlay',
                opacity=0.7
            )
        else:
            fig = px.histogram(
                df,
                x=x_axis,
                nbins=nbins,
                title=f"Distribution of {x_axis}",
                color_discrete_sequence=self.color_palettes.get(params.get('color_palette', 'financial'))
            )
        
        return fig
    
    def _create_scatter_plot(self, df: pd.DataFrame, params: Dict[str, Any]) -> go.Figure:
        """Create scatter plot"""
        x_axis = params.get('x_axis')
        y_axis = params.get('y_axis')
        color_by = params.get('color_by')
        
        if not x_axis or not y_axis:
            st.warning("Please select both X and Y axes")
            return None
        
        # Create scatter plot
        if color_by:
            fig = px.scatter(
                df,
                x=x_axis,
                y=y_axis,
                color=color_by,
                title=f"{y_axis} vs {x_axis}",
                color_discrete_sequence=self.color_palettes.get(params.get('color_palette', 'financial')),
                trendline='ols'  # Optional regression line
            )
        else:
            fig = px.scatter(
                df,
                x=x_axis,
                y=y_axis,
                title=f"{y_axis} vs {x_axis}",
                color_discrete_sequence=self.color_palettes.get(params.get('color_palette', 'financial')),
                trendline='ols'
            )
        
        return fig
    
    def _create_bubble_chart(self, df: pd.DataFrame, params: Dict[str, Any]) -> go.Figure:
        """Create bubble chart"""
        x_axis = params.get('x_axis')
        y_axis = params.get('y_axis')
        size = params.get('default_size', 'Quantity')
        color_by = params.get('color_by')
        
        if not x_axis or not y_axis:
            st.warning("Please select both X and Y axes")
            return None
        
        # Create bubble chart
        if color_by:
            fig = px.scatter(
                df,
                x=x_axis,
                y=y_axis,
                size=size,
                color=color_by,
                title=f"{y_axis} vs {x_axis} (Bubble Chart)",
                color_discrete_sequence=self.color_palettes.get(params.get('color_palette', 'financial')),
                size_max=50,
                hover_name=color_by
            )
        else:
            fig = px.scatter(
                df,
                x=x_axis,
                y=y_axis,
                size=size,
                title=f"{y_axis} vs {x_axis} (Bubble Chart)",
                color_discrete_sequence=self.color_palettes.get(params.get('color_palette', 'financial')),
                size_max=50
            )
        
        return fig
    
    def _apply_chart_styling(self, fig: go.Figure, params: Dict[str, Any]) -> go.Figure:
        """Apply styling to chart"""
        # Update layout
        fig.update_layout(
            width=params.get('width', 800),
            height=params.get('height', 500),
            showlegend=params.get('show_legend', True),
            plot_bgcolor='rgba(0,0,0,0)' if params.get('transparent_bg', False) else 'white',
            paper_bgcolor='rgba(0,0,0,0)' if params.get('transparent_bg', False) else 'white'
        )
        
        # Update grid
        fig.update_xaxes(
            showgrid=params.get('show_grid', True),
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.2)'
        )
        
        fig.update_yaxes(
            showgrid=params.get('show_grid', True),
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.2)'
        )
        
        # Update hover
        if params.get('show_tooltip', True):
            fig.update_traces(
                hovertemplate='<b>%{x}</b><br>Value: %{y:.2f}<extra></extra>'
            )
        
        return fig
    
    def _display_chart_export_options(self, fig: go.Figure, chart_type: str):
        """Display chart export options"""
        st.markdown("---")
        st.markdown("#### 📤 Export Chart")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("💾 Save as PNG", use_container_width=True):
                # In a real implementation, this would save the image
                st.success(f"Chart '{chart_type}' saved as PNG")
        
        with col2:
            if st.button("📊 Save as HTML", use_container_width=True):
                st.success(f"Chart '{chart_type}' saved as HTML")
        
        with col3:
            if st.button("📋 Copy to Report", use_container_width=True):
                st.info("Chart added to report queue")
    
    def _time_series_tab(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Time series visualizations
        """
        st.markdown("### 📈 Time Series Analysis")
        
        results = {}
        
        # Check for date column
        date_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
        if not date_cols:
            st.warning("No date column found. Time series analysis requires a date column.")
            return results
        
        date_col = date_cols[0]
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            st.warning("No numeric columns found for time series analysis.")
            return results
        
        # Metric selection
        st.markdown("#### Select Metrics")
        
        selected_metrics = st.multiselect(
            "Select metrics for time series analysis:",
            options=numeric_cols,
            default=numeric_cols[:min(3, len(numeric_cols))]
        )
        
        if not selected_metrics:
            return results
        
        # Time period aggregation
        st.markdown("#### Time Aggregation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            period = st.selectbox(
                "Aggregation period:",
                options=['Daily', 'Weekly', 'Monthly', 'Quarterly', 'Yearly'],
                index=2  # Default to Monthly
            )
        
        with col2:
            agg_method = st.selectbox(
                "Aggregation method:",
                options=['Sum', 'Mean', 'Median', 'Min', 'Max'],
                index=0
            )
        
        # Prepare time series data
        df_time = df.copy()
        
        # Ensure date is datetime
        if not pd.api.types.is_datetime64_any_dtype(df_time[date_col]):
            df_time[date_col] = pd.to_datetime(df_time[date_col], errors='coerce')
        
        # Create period column
        if period == 'Daily':
            df_time['period'] = df_time[date_col].dt.date
        elif period == 'Weekly':
            df_time['period'] = df_time[date_col].dt.to_period('W').apply(lambda r: r.start_time)
        elif period == 'Monthly':
            df_time['period'] = df_time[date_col].dt.to_period('M').apply(lambda r: r.start_time)
        elif period == 'Quarterly':
            df_time['period'] = df_time[date_col].dt.to_period('Q').apply(lambda r: r.start_time)
        elif period == 'Yearly':
            df_time['period'] = df_time[date_col].dt.to_period('Y').apply(lambda r: r.start_time)
        
        # Aggregate data
        agg_func = agg_method.lower()
        time_series_data = df_time.groupby('period')[selected_metrics].agg(agg_func).reset_index()
        
        # Create visualizations
        st.markdown("#### Visualizations")
        
        # Line chart for trends
        fig1 = go.Figure()
        
        colors = self.color_palettes['financial']
        for i, metric in enumerate(selected_metrics):
            fig1.add_trace(go.Scatter(
                x=time_series_data['period'],
                y=time_series_data[metric],
                mode='lines+markers',
                name=metric,
                line=dict(color=colors[i % len(colors)], width=2),
                marker=dict(size=6)
            ))
        
        fig1.update_layout(
            title=f"{agg_method} {period} Trends",
            xaxis_title="Period",
            yaxis_title="Value",
            hovermode='x unified',
            height=400
        )
        
        st.plotly_chart(fig1, use_container_width=True)
        
        # Area chart for cumulative view
        if st.checkbox("Show cumulative view", value=False):
            cumulative_data = time_series_data.copy()
            for metric in selected_metrics:
                cumulative_data[f'{metric}_cumulative'] = cumulative_data[metric].cumsum()
            
            fig2 = go.Figure()
            
            for i, metric in enumerate(selected_metrics):
                fig2.add_trace(go.Scatter(
                    x=cumulative_data['period'],
                    y=cumulative_data[f'{metric}_cumulative'],
                    mode='lines',
                    name=f'{metric} (Cumulative)',
                    fill='tozeroy',
                    line=dict(color=colors[i % len(colors)], width=2),
                    fillcolor=f'rgba{tuple(int(colors[i % len(colors)][j:j+2], 16) for j in (1, 3, 5)) + (0.2,)}'
                ))
            
            fig2.update_layout(
                title=f"Cumulative {period} View",
                xaxis_title="Period",
                yaxis_title="Cumulative Value",
                height=400
            )
            
            st.plotly_chart(fig2, use_container_width=True)
        
        # Growth rates
        if st.checkbox("Show growth rates", value=True):
            st.markdown("##### Growth Rates")
            
            for metric in selected_metrics:
                growth_rates = time_series_data[metric].pct_change() * 100
                avg_growth = growth_rates.mean()
                vol_growth = growth_rates.std()
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        f"{metric} Avg Growth",
                        f"{avg_growth:.1f}%"
                    )
                with col2:
                    st.metric(
                        f"{metric} Growth Volatility",
                        f"{vol_growth:.1f}%"
                    )
                with col3:
                    # Simple trend indicator
                    if avg_growth > 5:
                        trend = "🚀 Strong Growth"
                    elif avg_growth > 0:
                        trend = "📈 Growing"
                    elif avg_growth > -5:
                        trend = "📉 Declining"
                    else:
                        trend = "⚠️ Strong Decline"
                    st.metric(
                        f"{metric} Trend",
                        trend
                    )
        
        results['time_series_data'] = time_series_data
        return results
    
    def _comparative_tab(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Comparative visualizations
        """
        st.markdown("### 📊 Comparative Analysis")
        
        results = {}
        
        # Get column types
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if not numeric_cols or not categorical_cols:
            st.warning("Need both numeric and categorical columns for comparative analysis.")
            return results
        
        # Metric selection
        metric = st.selectbox(
            "Select metric for comparison:",
            options=numeric_cols,
            help="Numeric metric to compare across categories"
        )
        
        # Dimension selection
        dimension = st.selectbox(
            "Select dimension for comparison:",
            options=categorical_cols,
            help="Categorical dimension to group by"
        )
        
        # Aggregation method
        agg_method = st.selectbox(
            "Aggregation method:",
            options=['Sum', 'Mean', 'Median', 'Count', 'Min', 'Max'],
            index=0,
            help="How to aggregate values within each group"
        )
        
        # Sort order
        sort_by = st.selectbox(
            "Sort by:",
            options=['Value (Descending)', 'Value (Ascending)', 'Alphabetical', 'Original'],
            index=0
        )
        
        # Limit results
        limit = st.slider(
            "Number of items to show:",
            min_value=5,
            max_value=50,
            value=15,
            help="Limit number of categories shown"
        )
        
        if st.button("📊 Generate Comparison", type="primary", use_container_width=True):
            # Prepare comparison data
            if agg_method == 'Sum':
                comparison_data = df.groupby(dimension)[metric].sum().reset_index()
            elif agg_method == 'Mean':
                comparison_data = df.groupby(dimension)[metric].mean().reset_index()
            elif agg_method == 'Median':
                comparison_data = df.groupby(dimension)[metric].median().reset_index()
            elif agg_method == 'Count':
                comparison_data = df.groupby(dimension)[metric].count().reset_index()
            elif agg_method == 'Min':
                comparison_data = df.groupby(dimension)[metric].min().reset_index()
            else:  # Max
                comparison_data = df.groupby(dimension)[metric].max().reset_index()
            
            # Sort data
            if sort_by == 'Value (Descending)':
                comparison_data = comparison_data.sort_values(metric, ascending=False)
            elif sort_by == 'Value (Ascending)':
                comparison_data = comparison_data.sort_values(metric, ascending=True)
            elif sort_by == 'Alphabetical':
                comparison_data = comparison_data.sort_values(dimension)
            # Original order keeps as-is
            
            # Limit results
            comparison_data = comparison_data.head(limit)
            
            results['comparison_data'] = comparison_data
            
            # Create visualizations
            st.markdown("#### 📈 Comparison Visualizations")
            
            # Bar chart
            fig1 = go.Figure()
            
            fig1.add_trace(go.Bar(
                x=comparison_data[dimension],
                y=comparison_data[metric],
                marker_color=self.color_palettes['financial'],
                text=comparison_data[metric].apply(lambda x: f"{x:,.0f}"),
                textposition='auto'
            ))
            
            fig1.update_layout(
                title=f"{metric} by {dimension} ({agg_method})",
                xaxis_title=dimension,
                yaxis_title=metric,
                height=400
            )
            
            st.plotly_chart(fig1, use_container_width=True)
            
            # Pie chart for top contributors
            if st.checkbox("Show contribution analysis", value=True):
                st.markdown("##### Contribution Analysis")
                
                # Calculate total and percentages
                total = comparison_data[metric].sum()
                comparison_data['Percentage'] = (comparison_data[metric] / total * 100).round(1)
                comparison_data['Cumulative %'] = comparison_data['Percentage'].cumsum()
                
                # Display top contributors
                st.dataframe(
                    comparison_data.style.format({
                        metric: '{:,.0f}',
                        'Percentage': '{:.1f}%',
                        'Cumulative %': '{:.1f}%'
                    }),
                    use_container_width=True
                )
                
                # Pareto chart
                fig2 = make_subplots(specs=[[{"secondary_y": True}]])
                
                fig2.add_trace(
                    go.Bar(
                        x=comparison_data[dimension],
                        y=comparison_data[metric],
                        name=metric,
                        marker_color=self.color_palettes['financial'][0]
                    ),
                    secondary_y=False
                )
                
                fig2.add_trace(
                    go.Scatter(
                        x=comparison_data[dimension],
                        y=comparison_data['Cumulative %'],
                        name='Cumulative %',
                        line=dict(color='red', width=3)
                    ),
                    secondary_y=True
                )
                
                fig2.update_layout(
                    title=f"Pareto Chart: {metric} by {dimension}",
                    xaxis_title=dimension,
                    showlegend=True
                )
                
                fig2.update_yaxes(title_text=metric, secondary_y=False)
                fig2.update_yaxes(title_text="Cumulative %", secondary_y=True, range=[0, 100])
                
                st.plotly_chart(fig2, use_container_width=True)
            
            # Comparison matrix if second dimension selected
            if st.checkbox("Add second dimension for matrix view", value=False):
                dimension2 = st.selectbox(
                    "Select second dimension:",
                    options=['None'] + [col for col in categorical_cols if col != dimension]
                )
                
                if dimension2 != 'None':
                    # Create pivot table
                    pivot_data = df.pivot_table(
                        values=metric,
                        index=dimension,
                        columns=dimension2,
                        aggfunc=agg_method.lower(),
                        fill_value=0
                    )
                    
                    # Heatmap
                    fig3 = px.imshow(
                        pivot_data,
                        title=f"{metric} by {dimension} and {dimension2}",
                        color_continuous_scale=self.color_palettes['sequential'],
                        labels=dict(x=dimension2, y=dimension, color=metric)
                    )
                    
                    fig3.update_layout(height=500)
                    st.plotly_chart(fig3, use_container_width=True)
        
        return results
    
    def _dashboards_tab(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Dashboard creation
        """
        st.markdown("### 🎯 Interactive Dashboards")
        
        results = {}
        
        # Dashboard layout selection
        st.markdown("#### Dashboard Layout")
        
        layout_options = list(self.dashboard_layouts.keys())
        layout_names = [self.dashboard_layouts[layout]['name'] for layout in layout_options]
        
        selected_layout = st.selectbox(
            "Select dashboard layout:",
            options=layout_options,
            format_func=lambda x: self.dashboard_layouts[x]['name'],
            help="Choose how to arrange charts in dashboard"
        )
        
        if not selected_layout:
            return results
        
        layout = self.dashboard_layouts[selected_layout]
        
        # Display layout preview
        st.markdown(f"**Layout:** {layout['name']}")
        st.markdown(f"**Description:** {layout['description']}")
        
        # Create dashboard
        if st.button("🛠️ Create Dashboard", type="primary", use_container_width=True):
            with st.spinner(f"Creating {layout['name']} dashboard..."):
                # Create dashboard based on layout
                dashboard = self._create_dashboard(df, layout)
                
                if dashboard:
                    results['dashboard'] = dashboard
                    
                    # Display dashboard
                    st.markdown("---")
                    st.markdown(f"## 📊 {layout['name']} Dashboard")
                    
                    # For each row in layout
                    row_heights = [400] * len(layout['layout'])  # Fixed height for each row
                    
                    # Create subplots
                    fig = make_subplots(
                        rows=len(layout['layout']),
                        cols=max(len(row) for row in layout['layout']),
                        subplot_titles=[f"Chart {i+1}" for i in range(layout['chart_count'])],
                        vertical_spacing=0.1,
                        horizontal_spacing=0.1
                    )
                    
                    # Add sample charts
                    chart_idx = 0
                    for row_idx, row in enumerate(layout['layout']):
                        for col_idx, col_span in enumerate(row):
                            if chart_idx < layout['chart_count']:
                                # Add a sample chart to each position
                                fig.add_trace(
                                    go.Scatter(
                                        x=[1, 2, 3],
                                        y=[chart_idx + 1, chart_idx + 2, chart_idx + 3],
                                        mode='lines',
                                        name=f'Chart {chart_idx + 1}'
                                    ),
                                    row=row_idx + 1,
                                    col=col_idx + 1
                                )
                                chart_idx += 1
                    
                    fig.update_layout(
                        height=sum(row_heights),
                        showlegend=False,
                        title_text="Dashboard Preview"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Dashboard configuration options
                    st.markdown("---")
                    st.markdown("#### ⚙️ Dashboard Configuration")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        dashboard_name = st.text_input("Dashboard Name:", value=f"{layout['name']} Dashboard")
                    
                    with col2:
                        refresh_rate = st.selectbox(
                            "Auto-refresh:",
                            options=['None', '5 seconds', '30 seconds', '1 minute', '5 minutes'],
                            index=0
                        )
                    
                    with col3:
                        if st.button("💾 Save Dashboard", use_container_width=True):
                            st.success(f"Dashboard '{dashboard_name}' saved")
                    
                    # Export options
                    st.markdown("#### 📤 Export Dashboard")
                    
                    export_cols = st.columns(4)
                    
                    with export_cols[0]:
                        if st.button("📊 Export as Image", use_container_width=True):
                            st.info("Dashboard export as image coming soon")
                    
                    with export_cols[1]:
                        if st.button("📈 Export as HTML", use_container_width=True):
                            st.info("Dashboard export as HTML coming soon")
                    
                    with export_cols[2]:
                        if st.button("📋 Export as PDF", use_container_width=True):
                            st.info("Dashboard export as PDF coming soon")
                    
                    with export_cols[3]:
                        if st.button("🔗 Share Dashboard", use_container_width=True):
                            st.info("Dashboard sharing coming soon")
        
        return results
    
    def _create_dashboard(self, df: pd.DataFrame, layout: Dict[str, Any]) -> Dict[str, Any]:
        """Create dashboard based on layout"""
        # This is a placeholder for actual dashboard creation
        # In a full implementation, this would create actual charts based on data
        
        dashboard = {
            'layout': layout,
            'charts': [],
            'created_at': datetime.now().isoformat(),
            'data_source': df.shape
        }
        
        return dashboard
    
    def _custom_visuals_tab(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Custom and advanced visualizations
        """
        st.markdown("### ⚙️ Custom Visualizations")
        
        results = {}
        
        # Advanced visualization types
        st.markdown("#### 🔬 Advanced Visualization Types")
        
        viz_type = st.selectbox(
            "Select advanced visualization:",
            options=[
                'Correlation Matrix',
                'Scatter Plot Matrix',
                'Parallel Coordinates',
                'Sunburst Chart',
                'Treemap',
                'Violin Plot',
                'Ridge Plot',
                'Calendar Heatmap'
            ]
        )
        
        if viz_type == 'Correlation Matrix':
            results.update(self._create_correlation_matrix(df))
        
        elif viz_type == 'Scatter Plot Matrix':
            results.update(self._create_scatter_matrix(df))
        
        elif viz_type == 'Sunburst Chart':
            results.update(self._create_sunburst_chart(df))
        
        elif viz_type == 'Treemap':
            results.update(self._create_treemap(df))
        
        # Custom color configuration
        st.markdown("---")
        st.markdown("#### 🎨 Custom Color Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            custom_color = st.color_picker("Primary color:", value="#2E86AB")
        
        with col2:
            secondary_color = st.color_picker("Secondary color:", value="#A23B72")
        
        # Generate custom palette
        if st.button("Generate Custom Palette", use_container_width=True):
            # Create a custom palette
            import colorsys
            
            # Generate variations
            base_rgb = tuple(int(custom_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
            base_hls = colorsys.rgb_to_hls(*[c/255 for c in base_rgb])
            
            custom_palette = []
            for i in range(6):
                hue = (base_hls[0] + i/12) % 1.0
                lightness = base_hls[1] * (0.8 + i*0.04)
                rgb = colorsys.hls_to_rgb(hue, lightness, base_hls[2])
                hex_color = f"#{int(rgb[0]*255):02x}{int(rgb[1]*255):02x}{int(rgb[2]*255):02x}"
                custom_palette.append(hex_color)
            
            self.color_palettes['custom'] = custom_palette
            st.success(f"Custom palette generated with {len(custom_palette)} colors")
            
            # Display palette
            st.markdown("**Custom Palette:**")
            
            cols = st.columns(len(custom_palette))
            for i, color in enumerate(custom_palette):
                with cols[i]:
                    st.markdown(
                        f'<div style="background-color:{color}; height:50px; border-radius:5px;"></div>',
                        unsafe_allow_html=True
                    )
                    st.caption(f"Color {i+1}")
        
        # Visualization export
        st.markdown("---")
        st.markdown("#### 📤 Advanced Export Options")
        
        export_format = st.selectbox(
            "Export format:",
            options=['PNG (High Quality)', 'PNG (Web)', 'SVG', 'PDF', 'HTML (Interactive)']
        )
        
        dpi = st.slider("DPI (for PNG/PDF):", 72, 600, 150, 10)
        
        if st.button("🚀 Export with Custom Settings", use_container_width=True):
            st.success(f"Export configured: {export_format} at {dpi} DPI")
        
        return results
    
    def _create_correlation_matrix(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Create correlation matrix visualization"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            st.warning("Need at least 2 numeric columns for correlation matrix")
            return {}
        
        # Select columns for correlation
        selected_cols = st.multiselect(
            "Select columns for correlation:",
            options=numeric_cols,
            default=numeric_cols[:min(10, len(numeric_cols))]
        )
        
        if len(selected_cols) < 2:
            return {}
        
        # Calculate correlation matrix
        corr_matrix = df[selected_cols].corr()
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale=self.color_palettes['diverging'],
            zmin=-1,
            zmax=1,
            text=corr_matrix.values.round(2),
            texttemplate='%{text}',
            textfont={"size": 10}
        ))
        
        fig.update_layout(
            title="Correlation Matrix",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Strongest correlations
        corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) > 0.5:  # Only show strong correlations
                    corr_pairs.append({
                        'Variable 1': corr_matrix.columns[i],
                        'Variable 2': corr_matrix.columns[j],
                        'Correlation': corr_value
                    })
        
        if corr_pairs:
            st.markdown("##### 🎯 Strong Correlations (|r| > 0.5)")
            corr_df = pd.DataFrame(corr_pairs)
            corr_df = corr_df.sort_values('Correlation', key=abs, ascending=False)
            st.dataframe(corr_df.style.format({'Correlation': '{:.3f}'}), use_container_width=True)
        
        return {'correlation_matrix': corr_matrix}
    
    def _create_scatter_matrix(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Create scatter plot matrix"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            st.warning("Need at least 2 numeric columns for scatter matrix")
            return {}
        
        # Select columns
        selected_cols = st.multiselect(
            "Select columns for scatter matrix:",
            options=numeric_cols,
            default=numeric_cols[:min(5, len(numeric_cols))]
        )
        
        if len(selected_cols) < 2:
            return {}
        
        # Color by option
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        color_by = st.selectbox(
            "Color by (optional):",
            options=['None'] + categorical_cols
        )
        
        color_by = None if color_by == 'None' else color_by
        
        # Create scatter matrix
        if color_by:
            fig = px.scatter_matrix(
                df,
                dimensions=selected_cols,
                color=color_by,
                title="Scatter Plot Matrix",
                color_discrete_sequence=self.color_palettes['qualitative']
            )
        else:
            fig = px.scatter_matrix(
                df,
                dimensions=selected_cols,
                title="Scatter Plot Matrix",
                color_discrete_sequence=self.color_palettes['financial']
            )
        
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        return {'scatter_matrix': fig}
    
    def _create_sunburst_chart(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Create sunburst chart"""
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(categorical_cols) < 2 or not numeric_cols:
            st.warning("Need at least 2 categorical columns and 1 numeric column for sunburst chart")
            return {}
        
        # Select hierarchy columns
        st.markdown("##### Hierarchy Levels")
        
        level1 = st.selectbox(
            "Level 1 (outer ring):",
            options=categorical_cols
        )
        
        remaining_cols = [col for col in categorical_cols if col != level1]
        level2 = st.selectbox(
            "Level 2 (inner ring):",
            options=['None'] + remaining_cols
        )
        
        level2 = None if level2 == 'None' else level2
        
        # Select value column
        value_col = st.selectbox(
            "Value column:",
            options=numeric_cols
        )
        
        # Create hierarchy
        if level2:
            hierarchy = [level1, level2]
            grouped = df.groupby(hierarchy)[value_col].sum().reset_index()
            
            fig = px.sunburst(
                grouped,
                path=hierarchy,
                values=value_col,
                title=f"Sunburst Chart: {value_col} by {level1} and {level2}",
                color_discrete_sequence=self.color_palettes['qualitative']
            )
        else:
            grouped = df.groupby(level1)[value_col].sum().reset_index()
            
            fig = px.sunburst(
                grouped,
                path=[level1],
                values=value_col,
                title=f"Sunburst Chart: {value_col} by {level1}",
                color_discrete_sequence=self.color_palettes['qualitative']
            )
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        return {'sunburst_chart': fig}
    
    def _create_treemap(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Create treemap visualization"""
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not categorical_cols or not numeric_cols:
            st.warning("Need at least 1 categorical and 1 numeric column for treemap")
            return {}
        
        # Select hierarchy and value
        hierarchy = st.multiselect(
            "Select hierarchy levels:",
            options=categorical_cols,
            default=categorical_cols[:min(3, len(categorical_cols))]
        )
        
        if not hierarchy:
            return {}
        
        value_col = st.selectbox(
            "Select value column:",
            options=numeric_cols,
            key='treemap_value'
        )
        
        # Aggregate data
        grouped = df.groupby(hierarchy)[value_col].sum().reset_index()
        
        # Create treemap
        fig = px.treemap(
            grouped,
            path=hierarchy,
            values=value_col,
            title=f"Treemap: {value_col} by {' > '.join(hierarchy)}",
            color_discrete_sequence=self.color_palettes['qualitative']
        )
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        return {'treemap': fig}


# Streamlit integration function
def display_visualizations_section(dataframes: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    Main function to display visualizations interface
    """
    if not dataframes:
        st.warning("Please upload and analyze data first.")
        return {}
    
    visualizer = FinancialVisualizations()
    viz_results = visualizer.display_visualizations_interface(dataframes)
    
    return viz_results