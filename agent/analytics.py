"""
Analytics Module
Core FP&A analytics: KPIs, ratios, variance analysis, cash flow analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class FinancialAnalytics:
    """
    Professional-grade financial analytics for FP&A
    """
    
    def __init__(self):
        self.kpi_definitions = self._initialize_kpi_definitions()
        self.analysis_results = {}
        self.benchmarks = {}
        self.trend_periods = ['Daily', 'Weekly', 'Monthly', 'Quarterly', 'Yearly']
    
    def _initialize_kpi_definitions(self) -> Dict[str, Dict]:
        """Initialize standard FP&A KPI definitions"""
        return {
            'revenue': {
                'name': 'Revenue',
                'description': 'Total income from sales',
                'formula': 'SUM(Revenue)',
                'category': 'Income',
                'direction': 'higher',
                'format': 'currency'
            },
            'gross_profit': {
                'name': 'Gross Profit',
                'description': 'Revenue minus Cost of Goods Sold',
                'formula': 'Revenue - COGS',
                'category': 'Profitability',
                'direction': 'higher',
                'format': 'currency'
            },
            'gross_margin': {
                'name': 'Gross Margin %',
                'description': 'Gross Profit as percentage of Revenue',
                'formula': '(Gross Profit / Revenue) * 100',
                'category': 'Profitability',
                'direction': 'higher',
                'format': 'percentage',
                'benchmark': {'good': 40, 'warning': 20, 'poor': 10}
            },
            'net_income': {
                'name': 'Net Income',
                'description': 'Total profit after all expenses',
                'formula': 'Revenue - All Expenses',
                'category': 'Profitability',
                'direction': 'higher',
                'format': 'currency'
            },
            'ebitda': {
                'name': 'EBITDA',
                'description': 'Earnings Before Interest, Taxes, Depreciation, Amortization',
                'formula': 'Net Income + Interest + Taxes + Depreciation + Amortization',
                'category': 'Profitability',
                'direction': 'higher',
                'format': 'currency'
            },
            'ebitda_margin': {
                'name': 'EBITDA Margin %',
                'description': 'EBITDA as percentage of Revenue',
                'formula': '(EBITDA / Revenue) * 100',
                'category': 'Profitability',
                'direction': 'higher',
                'format': 'percentage',
                'benchmark': {'good': 20, 'warning': 10, 'poor': 5}
            },
            'operating_expense_ratio': {
                'name': 'Operating Expense Ratio',
                'description': 'Operating Expenses as percentage of Revenue',
                'formula': '(Operating Expenses / Revenue) * 100',
                'category': 'Efficiency',
                'direction': 'lower',
                'format': 'percentage',
                'benchmark': {'good': 30, 'warning': 50, 'poor': 70}
            },
            'current_ratio': {
                'name': 'Current Ratio',
                'description': 'Current Assets / Current Liabilities',
                'formula': 'Current Assets / Current Liabilities',
                'category': 'Liquidity',
                'direction': 'higher',
                'format': 'decimal',
                'benchmark': {'good': 2.0, 'warning': 1.5, 'poor': 1.0}
            },
            'quick_ratio': {
                'name': 'Quick Ratio',
                'description': '(Current Assets - Inventory) / Current Liabilities',
                'formula': '(Current Assets - Inventory) / Current Liabilities',
                'category': 'Liquidity',
                'direction': 'higher',
                'format': 'decimal',
                'benchmark': {'good': 1.0, 'warning': 0.5, 'poor': 0.2}
            },
            'roi': {
                'name': 'Return on Investment',
                'description': 'Net Profit / Total Investment',
                'formula': '(Net Profit / Total Investment) * 100',
                'category': 'Returns',
                'direction': 'higher',
                'format': 'percentage',
                'benchmark': {'good': 15, 'warning': 5, 'poor': 0}
            },
            'cac': {
                'name': 'Customer Acquisition Cost',
                'description': 'Total Marketing Cost / New Customers',
                'formula': 'Marketing Cost / New Customers',
                'category': 'Customer',
                'direction': 'lower',
                'format': 'currency'
            },
            'ltv': {
                'name': 'Customer Lifetime Value',
                'description': 'Average Revenue per Customer * Customer Lifespan',
                'formula': 'Avg Revenue per Customer * Avg Customer Lifespan',
                'category': 'Customer',
                'direction': 'higher',
                'format': 'currency'
            },
            'ltv_cac_ratio': {
                'name': 'LTV:CAC Ratio',
                'description': 'Lifetime Value to Customer Acquisition Cost Ratio',
                'formula': 'LTV / CAC',
                'category': 'Customer',
                'direction': 'higher',
                'format': 'decimal',
                'benchmark': {'good': 3.0, 'warning': 1.5, 'poor': 1.0}
            },
            'inventory_turnover': {
                'name': 'Inventory Turnover',
                'description': 'Cost of Goods Sold / Average Inventory',
                'formula': 'COGS / Average Inventory',
                'category': 'Efficiency',
                'direction': 'higher',
                'format': 'decimal',
                'benchmark': {'good': 8, 'warning': 4, 'poor': 2}
            },
            'days_sales_outstanding': {
                'name': 'Days Sales Outstanding',
                'description': 'Average time to collect payment',
                'formula': '(Accounts Receivable / Revenue) * 365',
                'category': 'Efficiency',
                'direction': 'lower',
                'format': 'days',
                'benchmark': {'good': 30, 'warning': 45, 'poor': 60}
            }
        }
    
    def display_analytics_dashboard(self, dataframes: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Main Streamlit interface for financial analytics
        """
        st.subheader("📈 Financial Analytics Dashboard")
        
        if not dataframes:
            st.warning("⚠️ No data available. Please upload and filter data first.")
            return {}
        
        # Get current data (filtered if available, otherwise original)
        current_data = dataframes
        
        # File selection for primary analysis
        primary_file = st.selectbox(
            "Select primary file for analysis:",
            options=list(current_data.keys()),
            help="This file will be used for detailed analysis and visualizations"
        )
        
        if not primary_file:
            return {}
        
        df = current_data[primary_file]
        
        # Detect financial columns
        financial_cols = self._detect_financial_columns(df)
        
        if not financial_cols.get('revenue_columns') and not financial_cols.get('expense_columns'):
            st.warning("⚠️ No financial columns detected. Please ensure data contains revenue, cost, or expense columns.")
            st.info("**Tip:** Column names should contain keywords like 'revenue', 'sales', 'cost', 'expense'")
            return {}
        
        # Create tabs for different analytics sections
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 KPI Dashboard", 
            "📈 Trend Analysis", 
            "📊 Variance Analysis",
            "💰 Profitability",
            "💧 Cash Flow",
            "📋 Deep Dive"
        ])
        
        analysis_results = {}
        
        with tab1:
            analysis_results['kpi_dashboard'] = self._display_kpi_dashboard(df, financial_cols)
        
        with tab2:
            analysis_results['trend_analysis'] = self._display_trend_analysis(df, financial_cols)
        
        with tab3:
            analysis_results['variance_analysis'] = self._display_variance_analysis(df, financial_cols)
        
        with tab4:
            analysis_results['profitability_analysis'] = self._display_profitability_analysis(df, financial_cols)
        
        with tab5:
            analysis_results['cash_flow_analysis'] = self._display_cash_flow_analysis(df, financial_cols)
        
        with tab6:
            analysis_results['deep_dive'] = self._display_deep_dive_analysis(df, financial_cols)
        
        # Store results
        self.analysis_results = analysis_results
        
        return analysis_results
    
    def _detect_financial_columns(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Detect financial columns in the dataframe
        """
        financial_keywords = {
            'revenue_columns': ['revenue', 'sales', 'income', 'turnover', 'gross_sales'],
            'expense_columns': ['expense', 'cost', 'cogs', 'operating_expense', 'overhead', 'spend'],
            'profit_columns': ['profit', 'margin', 'net_income', 'ebitda', 'gross_profit'],
            'quantity_columns': ['quantity', 'volume', 'units', 'qty'],
            'price_columns': ['price', 'unit_price', 'rate', 'fee'],
            'asset_columns': ['asset', 'inventory', 'receivable', 'cash', 'property'],
            'liability_columns': ['liability', 'debt', 'payable', 'loan'],
            'date_columns': ['date', 'time', 'month', 'year', 'quarter', 'period']
        }
        
        detected_cols = {key: [] for key in financial_keywords.keys()}
        
        for col in df.columns:
            col_lower = str(col).lower()
            
            for category, keywords in financial_keywords.items():
                if any(keyword in col_lower for keyword in keywords):
                    detected_cols[category].append(col)
        
        # Also detect numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        detected_cols['numeric_columns'] = numeric_cols
        
        # Detect categorical columns for segmentation
        categorical_cols = []
        for col in df.columns:
            if df[col].dtype == 'object' or df[col].dtype.name == 'category':
                unique_vals = df[col].dropna().unique()
                if len(unique_vals) <= 50:  # Reasonable for segmentation
                    categorical_cols.append(col)
        
        detected_cols['categorical_columns'] = categorical_cols
        
        return detected_cols
    
    def _display_kpi_dashboard(self, df: pd.DataFrame, financial_cols: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        Display KPI dashboard with metrics
        """
        st.markdown("### 📊 Key Performance Indicators")
        
        # Calculate KPIs
        kpi_values = self._calculate_kpis(df, financial_cols)
        
        # Display KPIs in a grid
        cols_per_row = 4
        kpi_items = list(kpi_values.items())
        
        for i in range(0, len(kpi_items), cols_per_row):
            row_kpis = kpi_items[i:i + cols_per_row]
            cols = st.columns(cols_per_row)
            
            for j, (kpi_name, kpi_data) in enumerate(row_kpis):
                with cols[j]:
                    self._display_kpi_card(kpi_name, kpi_data)
        
        # KPI Details
        with st.expander("📋 KPI Details & Definitions", expanded=False):
            for kpi_name, kpi_data in kpi_values.items():
                st.markdown(f"**{kpi_data['name']}**")
                st.write(f"*Value:* {kpi_data['formatted_value']}")
                st.write(f"*Description:* {kpi_data.get('description', 'N/A')}")
                st.write(f"*Formula:* {kpi_data.get('formula', 'N/A')}")
                if 'benchmark_status' in kpi_data:
                    st.write(f"*Benchmark:* {kpi_data['benchmark_status']}")
                st.markdown("---")
        
        return kpi_values
    
    def _calculate_kpis(self, df: pd.DataFrame, financial_cols: Dict[str, List[str]]) -> Dict[str, Dict]:
        """
        Calculate all KPIs from the data
        """
        kpi_values = {}
        
        # Get primary columns
        revenue_cols = financial_cols.get('revenue_columns', [])
        expense_cols = financial_cols.get('expense_columns', [])
        profit_cols = financial_cols.get('profit_columns', [])
        numeric_cols = financial_cols.get('numeric_columns', [])
        
        # Use first revenue column if available
        primary_revenue_col = revenue_cols[0] if revenue_cols else None
        primary_expense_col = expense_cols[0] if expense_cols else None
        
        # Calculate basic metrics
        if primary_revenue_col and primary_revenue_col in df.columns:
            total_revenue = df[primary_revenue_col].sum()
            avg_revenue = df[primary_revenue_col].mean()
            
            kpi_values['total_revenue'] = {
                'name': 'Total Revenue',
                'value': total_revenue,
                'formatted_value': f"${total_revenue:,.0f}",
                'description': 'Sum of all revenue',
                'formula': f'SUM({primary_revenue_col})',
                'category': 'Income'
            }
            
            kpi_values['avg_revenue'] = {
                'name': 'Average Revenue',
                'value': avg_revenue,
                'formatted_value': f"${avg_revenue:,.2f}",
                'description': 'Average revenue per record',
                'formula': f'AVG({primary_revenue_col})',
                'category': 'Income'
            }
        
        if primary_expense_col and primary_expense_col in df.columns:
            total_expense = df[primary_expense_col].sum()
            
            kpi_values['total_expense'] = {
                'name': 'Total Expense',
                'value': total_expense,
                'formatted_value': f"${total_expense:,.0f}",
                'description': 'Sum of all expenses',
                'formula': f'SUM({primary_expense_col})',
                'category': 'Expense'
            }
        
        # Calculate profit if we have both revenue and expense
        if primary_revenue_col and primary_expense_col:
            if primary_revenue_col in df.columns and primary_expense_col in df.columns:
                total_profit = df[primary_revenue_col].sum() - df[primary_expense_col].sum()
                gross_margin = (total_profit / df[primary_revenue_col].sum() * 100) if df[primary_revenue_col].sum() > 0 else 0
                
                kpi_values['total_profit'] = {
                    'name': 'Total Profit',
                    'value': total_profit,
                    'formatted_value': f"${total_profit:,.0f}",
                    'description': 'Revenue minus Expenses',
                    'formula': f'SUM({primary_revenue_col}) - SUM({primary_expense_col})',
                    'category': 'Profitability'
                }
                
                kpi_values['gross_margin'] = {
                    'name': 'Gross Margin %',
                    'value': gross_margin,
                    'formatted_value': f"{gross_margin:.2f}%",
                    'description': 'Profit as percentage of Revenue',
                    'formula': f'(SUM({primary_revenue_col}) - SUM({primary_expense_col})) / SUM({primary_revenue_col}) * 100',
                    'category': 'Profitability',
                    'benchmark': self.kpi_definitions['gross_margin']['benchmark'],
                    'benchmark_status': self._get_benchmark_status(gross_margin, self.kpi_definitions['gross_margin']['benchmark'])
                }
        
        # Calculate additional metrics
        date_cols = financial_cols.get('date_columns', [])
        if date_cols and primary_revenue_col:
            date_col = date_cols[0]
            if date_col in df.columns:
                # Ensure date column is datetime
                df_date = df.copy()
                if not pd.api.types.is_datetime64_any_dtype(df_date[date_col]):
                    df_date[date_col] = pd.to_datetime(df_date[date_col], errors='coerce')
                
                # Revenue growth
                df_date_sorted = df_date.sort_values(date_col)
                if len(df_date_sorted) >= 2:
                    recent_periods = self._get_recent_periods(df_date_sorted, date_col, primary_revenue_col)
                    
                    for period_name, period_data in recent_periods.items():
                        if len(period_data) >= 2:
                            current_rev = period_data.iloc[-1][primary_revenue_col]
                            previous_rev = period_data.iloc[-2][primary_revenue_col]
                            growth = ((current_rev - previous_rev) / previous_rev * 100) if previous_rev != 0 else 0
                            
                            kpi_values[f'{period_name.lower()}_growth'] = {
                                'name': f'{period_name} Revenue Growth',
                                'value': growth,
                                'formatted_value': f"{growth:+.2f}%",
                                'description': f'{period_name} over {period_name} revenue growth',
                                'formula': f'((Current {period_name} - Previous {period_name}) / Previous {period_name}) * 100',
                                'category': 'Growth'
                            }
        
        # Calculate ratios if we have required data
        if primary_revenue_col and primary_expense_col:
            # Operating expense ratio
            op_ex_ratio = (df[primary_expense_col].sum() / df[primary_revenue_col].sum() * 100) if df[primary_revenue_col].sum() > 0 else 0
            
            kpi_values['operating_expense_ratio'] = {
                'name': 'Operating Expense Ratio',
                'value': op_ex_ratio,
                'formatted_value': f"{op_ex_ratio:.2f}%",
                'description': 'Expenses as percentage of Revenue',
                'formula': f'SUM({primary_expense_col}) / SUM({primary_revenue_col}) * 100',
                'category': 'Efficiency',
                'benchmark': self.kpi_definitions['operating_expense_ratio']['benchmark'],
                'benchmark_status': self._get_benchmark_status(op_ex_ratio, self.kpi_definitions['operating_expense_ratio']['benchmark'], inverse=True)
            }
        
        # Add count-based KPIs
        kpi_values['total_records'] = {
            'name': 'Total Records',
            'value': len(df),
            'formatted_value': f"{len(df):,}",
            'description': 'Number of data records',
            'category': 'Volume'
        }
        
        if financial_cols.get('categorical_columns'):
            for cat_col in financial_cols['categorical_columns'][:3]:  # First 3 categorical columns
                unique_count = df[cat_col].nunique()
                kpi_values[f'unique_{cat_col}'] = {
                    'name': f'Unique {cat_col}',
                    'value': unique_count,
                    'formatted_value': f"{unique_count}",
                    'description': f'Number of unique {cat_col} values',
                    'category': 'Diversity'
                }
        
        return kpi_values
    
    def _get_recent_periods(self, df: pd.DataFrame, date_col: str, value_col: str) -> Dict[str, pd.DataFrame]:
        """Get data grouped by recent periods"""
        periods = {}
        
        if pd.api.types.is_datetime64_any_dtype(df[date_col]):
            # Monthly
            df['year_month'] = df[date_col].dt.to_period('M')
            monthly = df.groupby('year_month')[value_col].sum().reset_index()
            if len(monthly) >= 2:
                periods['Monthly'] = monthly.tail(6)  # Last 6 months
            
            # Quarterly
            df['year_quarter'] = df[date_col].dt.to_period('Q')
            quarterly = df.groupby('year_quarter')[value_col].sum().reset_index()
            if len(quarterly) >= 2:
                periods['Quarterly'] = quarterly.tail(4)  # Last 4 quarters
        
        return periods
    
    def _get_benchmark_status(self, value: float, benchmark: Dict[str, float], inverse: bool = False) -> str:
        """Get benchmark status for a KPI"""
        if not benchmark:
            return "No benchmark available"
        
        if inverse:  # For metrics where lower is better
            if value <= benchmark.get('good', 0):
                return "✅ Good"
            elif value <= benchmark.get('warning', 0):
                return "⚠️ Warning"
            else:
                return "❌ Poor"
        else:  # For metrics where higher is better
            if value >= benchmark.get('good', 0):
                return "✅ Good"
            elif value >= benchmark.get('warning', 0):
                return "⚠️ Warning"
            else:
                return "❌ Poor"
    
    def _display_kpi_card(self, kpi_name: str, kpi_data: Dict[str, Any]):
        """Display a single KPI card"""
        # Create card with appropriate color based on benchmark
        color = "#10B981"  # Default green
        
        if 'benchmark_status' in kpi_data:
            if "Warning" in kpi_data['benchmark_status']:
                color = "#F59E0B"  # Amber
            elif "Poor" in kpi_data['benchmark_status']:
                color = "#EF4444"  # Red
        
        # Create HTML card
        card_html = f"""
        <div style="
            background: white;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid {color};
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            margin-bottom: 1rem;
        ">
            <div style="font-size: 0.875rem; color: #6B7280; margin-bottom: 0.25rem;">
                {kpi_data['name']}
            </div>
            <div style="font-size: 1.5rem; font-weight: bold; color: #111827;">
                {kpi_data['formatted_value']}
            </div>
        """
        
        if 'benchmark_status' in kpi_data:
            card_html += f"""
            <div style="font-size: 0.75rem; margin-top: 0.5rem; color: {color};">
                {kpi_data['benchmark_status']}
            </div>
            """
        
        card_html += "</div>"
        
        st.markdown(card_html, unsafe_allow_html=True)
    
    def _display_trend_analysis(self, df: pd.DataFrame, financial_cols: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        Display trend analysis with time series visualizations
        """
        st.markdown("### 📈 Trend Analysis")
        
        results = {}
        
        # Check for date column
        date_cols = financial_cols.get('date_columns', [])
        if not date_cols:
            st.warning("No date column found for trend analysis.")
            return results
        
        date_col = date_cols[0]
        
        # Ensure date column is datetime
        df_trend = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df_trend[date_col]):
            df_trend[date_col] = pd.to_datetime(df_trend[date_col], errors='coerce')
        
        # Select metrics to analyze
        numeric_cols = financial_cols.get('numeric_columns', [])
        if not numeric_cols:
            st.warning("No numeric columns found for trend analysis.")
            return results
        
        # Metric selection
        selected_metrics = st.multiselect(
            "Select metrics to analyze:",
            options=numeric_cols,
            default=numeric_cols[:min(3, len(numeric_cols))]
        )
        
        if not selected_metrics:
            return results
        
        # Time period selection
        period = st.selectbox(
            "Select time period:",
            options=self.trend_periods,
            index=2  # Default to Monthly
        )
        
        # Group by period
        period_data = {}
        
        if period == 'Daily':
            df_trend['period'] = df_trend[date_col].dt.date
        elif period == 'Weekly':
            df_trend['period'] = df_trend[date_col].dt.to_period('W').apply(lambda r: r.start_time)
        elif period == 'Monthly':
            df_trend['period'] = df_trend[date_col].dt.to_period('M').apply(lambda r: r.start_time)
        elif period == 'Quarterly':
            df_trend['period'] = df_trend[date_col].dt.to_period('Q').apply(lambda r: r.start_time)
        elif period == 'Yearly':
            df_trend['period'] = df_trend[date_col].dt.to_period('Y').apply(lambda r: r.start_time)
        
        # Create visualizations
        fig = make_subplots(
            rows=len(selected_metrics), 
            cols=1,
            subplot_titles=[f"{metric} Trend" for metric in selected_metrics],
            vertical_spacing=0.1
        )
        
        for i, metric in enumerate(selected_metrics):
            # Group by period
            metric_trend = df_trend.groupby('period')[metric].sum().reset_index()
            
            # Calculate growth rates
            metric_trend['growth'] = metric_trend[metric].pct_change() * 100
            metric_trend['growth_smoothed'] = metric_trend['growth'].rolling(window=3, center=True).mean()
            
            # Add to results
            results[metric] = {
                'trend_data': metric_trend,
                'latest_value': metric_trend[metric].iloc[-1] if len(metric_trend) > 0 else 0,
                'growth_rate': metric_trend['growth'].iloc[-1] if len(metric_trend) > 1 else 0
            }
            
            # Add line chart
            fig.add_trace(
                go.Scatter(
                    x=metric_trend['period'],
                    y=metric_trend[metric],
                    mode='lines+markers',
                    name=metric,
                    line=dict(width=2),
                    marker=dict(size=6)
                ),
                row=i+1, col=1
            )
            
            # Add growth as bar chart on secondary y-axis
            if i == 0:  # Only show growth for first metric for clarity
                fig.add_trace(
                    go.Bar(
                        x=metric_trend['period'],
                        y=metric_trend['growth'],
                        name='Growth %',
                        marker_color='rgba(255, 165, 0, 0.3)',
                        yaxis='y2'
                    ),
                    row=i+1, col=1
                )
                
                # Add secondary y-axis
                fig.update_layout(
                    yaxis2=dict(
                        title="Growth %",
                        overlaying="y",
                        side="right"
                    )
                )
        
        fig.update_layout(
            height=300 * len(selected_metrics),
            showlegend=True,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Trend statistics
        st.markdown("#### 📊 Trend Statistics")
        
        for metric in selected_metrics:
            trend_data = results[metric]['trend_data']
            
            if len(trend_data) > 1:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    label = f"Latest {metric}"
                    value = f"${trend_data[metric].iloc[-1]:,.0f}" if any(word in metric.lower() for word in ['revenue', 'cost', 'profit', 'price']) else f"{trend_data[metric].iloc[-1]:,.0f}"
                    delta = f"{trend_data['growth'].iloc[-1]:+.1f}%" if len(trend_data) > 1 else ""
                    st.metric(label=label, value=value, delta=delta)
                
                with col2:
                    avg_growth = trend_data['growth'].dropna().mean()
                    st.metric(
                        label=f"Avg {period} Growth",
                        value=f"{avg_growth:.1f}%"
                    )
                
                with col3:
                    volatility = trend_data['growth'].dropna().std()
                    st.metric(
                        label="Volatility",
                        value=f"{volatility:.1f}%"
                    )
                
                with col4:
                    correlation = trend_data[metric].corr(pd.Series(range(len(trend_data)))) if len(trend_data) > 2 else 0
                    st.metric(
                        label="Trend Strength",
                        value=f"{correlation:.2f}" if not pd.isna(correlation) else "N/A"
                    )
                
                st.markdown("---")
        
        return results
    
    def _display_variance_analysis(self, df: pd.DataFrame, financial_cols: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        Display variance analysis (Actual vs Budget vs Forecast)
        """
        st.markdown("### 📊 Variance Analysis")
        
        results = {}
        
        # Check for scenario column
        scenario_keywords = ['scenario', 'version', 'type', 'budget', 'actual', 'forecast']
        scenario_col = None
        
        for col in df.columns:
            if any(keyword in str(col).lower() for keyword in scenario_keywords):
                scenario_col = col
                break
        
        if not scenario_col:
            st.info("No scenario column found. Variance analysis requires scenario labels (Actual, Budget, Forecast).")
            st.info("**Tip:** Add a column named 'Scenario' with values like 'Actual', 'Budget', 'Forecast'")
            return results
        
        # Get unique scenarios
        scenarios = df[scenario_col].dropna().unique()
        
        if len(scenarios) < 2:
            st.warning(f"Only one scenario found ({scenarios[0]}). Need at least 2 scenarios for variance analysis.")
            return results
        
        # Select scenarios to compare
        selected_scenarios = st.multiselect(
            "Select scenarios to compare:",
            options=list(scenarios),
            default=list(scenarios)[:min(2, len(scenarios))]
        )
        
        if len(selected_scenarios) < 2:
            return results
        
        # Select metric for comparison
        numeric_cols = financial_cols.get('numeric_columns', [])
        if not numeric_cols:
            return results
        
        metric = st.selectbox(
            "Select metric for variance analysis:",
            options=numeric_cols
        )
        
        # Group by scenario and any additional dimensions
        dimension_options = ['None'] + financial_cols.get('categorical_columns', [])
        dimension = st.selectbox(
            "Group by dimension (optional):",
            options=dimension_options
        )
        
        # Perform analysis
        if dimension == 'None':
            # Simple scenario comparison
            scenario_data = df[df[scenario_col].isin(selected_scenarios)]
            variance_data = scenario_data.groupby(scenario_col)[metric].agg(['sum', 'mean', 'count']).reset_index()
            
            # Calculate variance if we have exactly 2 scenarios
            if len(selected_scenarios) == 2:
                scenario1_val = variance_data.loc[variance_data[scenario_col] == selected_scenarios[0], 'sum'].iloc[0]
                scenario2_val = variance_data.loc[variance_data[scenario_col] == selected_scenarios[1], 'sum'].iloc[0]
                
                variance_amount = scenario2_val - scenario1_val
                variance_pct = (variance_amount / scenario1_val * 100) if scenario1_val != 0 else 0
                
                results['variance'] = {
                    'scenario1': selected_scenarios[0],
                    'scenario2': selected_scenarios[1],
                    'value1': scenario1_val,
                    'value2': scenario2_val,
                    'variance_amount': variance_amount,
                    'variance_pct': variance_pct
                }
                
                # Display variance - FIXED: Convert scenario names to strings
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        label=str(selected_scenarios[0]),  # Convert to string
                        value=f"${scenario1_val:,.0f}" if any(word in metric.lower() for word in ['revenue', 'cost', 'profit', 'price']) else f"{scenario1_val:,.0f}"
                    )
                with col2:
                    st.metric(
                        label=str(selected_scenarios[1]),  # Convert to string
                        value=f"${scenario2_val:,.0f}" if any(word in metric.lower() for word in ['revenue', 'cost', 'profit', 'price']) else f"{scenario2_val:,.0f}"
                    )
                with col3:
                    st.metric(
                        label="Variance",  # Already a string
                        value=f"${variance_amount:,.0f}" if any(word in metric.lower() for word in ['revenue', 'cost', 'profit', 'price']) else f"{variance_amount:,.0f}",
                        delta=f"{variance_pct:+.1f}%"
                    )
                
                # Visualization
                fig = go.Figure(data=[
                    go.Bar(
                        name=str(selected_scenarios[0]),  # Convert to string
                        x=[str(selected_scenarios[0])],  # Convert to string
                        y=[scenario1_val],
                        marker_color='#3B82F6'
                    ),
                    go.Bar(
                        name=str(selected_scenarios[1]),  # Convert to string
                        x=[str(selected_scenarios[1])],  # Convert to string
                        y=[scenario2_val],
                        marker_color='#EF4444'
                    )
                ])
                
                fig.update_layout(
                    title=f"{metric} Comparison",
                    yaxis_title=metric,
                    showlegend=True,
                    plot_bgcolor='rgba(240, 242, 246, 0.5)'
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        else:
            # Group by dimension
            scenario_dim_data = df[df[scenario_col].isin(selected_scenarios)]
            
            # Pivot table
            pivot_data = pd.pivot_table(
                scenario_dim_data,
                values=metric,
                index=dimension,
                columns=scenario_col,
                aggfunc='sum'
            ).reset_index()
            
            # Fill missing with 0
            pivot_data = pivot_data.fillna(0)
            
            # Calculate variance for each dimension
            if len(selected_scenarios) == 2:
                pivot_data['Variance'] = pivot_data[selected_scenarios[1]] - pivot_data[selected_scenarios[0]]
                pivot_data['Variance %'] = (pivot_data['Variance'] / pivot_data[selected_scenarios[0]] * 100).replace([np.inf, -np.inf], 0)
                
                # Sort by largest variance
                pivot_data = pivot_data.sort_values('Variance', key=abs, ascending=False)
                
                results['variance_by_dimension'] = pivot_data
                
                # Display top variances
                st.markdown("#### 📋 Top Variances by Dimension")
                
                # Show as table
                display_cols = [dimension, selected_scenarios[0], selected_scenarios[1], 'Variance', 'Variance %']
                formatted_df = pivot_data[display_cols].head(10).copy()
                
                # Format numbers
                for col in [selected_scenarios[0], selected_scenarios[1], 'Variance']:
                    if any(word in metric.lower() for word in ['revenue', 'cost', 'profit', 'price']):
                        formatted_df[col] = formatted_df[col].apply(lambda x: f"${x:,.0f}")
                    else:
                        formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:,.0f}")
                
                formatted_df['Variance %'] = formatted_df['Variance %'].apply(lambda x: f"{x:+.1f}%")
                
                st.dataframe(formatted_df, use_container_width=True, height=400)
                
                # Visualization
                top_n = min(10, len(pivot_data))
                fig = go.Figure()
                
                # Convert scenario names to strings
                scenario1_str = str(selected_scenarios[0])
                scenario2_str = str(selected_scenarios[1])
                
                fig.add_trace(go.Bar(
                    name=scenario1_str,
                    x=pivot_data.head(top_n)[dimension],
                    y=pivot_data.head(top_n)[selected_scenarios[0]],
                    text=pivot_data.head(top_n)[selected_scenarios[0]].apply(
                        lambda x: f"${x:,.0f}" if any(word in metric.lower() for word in ['revenue', 'cost', 'profit', 'price']) else f"{x:,.0f}"
                    ),
                    marker_color='#3B82F6'
                ))
                
                fig.add_trace(go.Bar(
                    name=scenario2_str,
                    x=pivot_data.head(top_n)[dimension],
                    y=pivot_data.head(top_n)[selected_scenarios[1]],
                    text=pivot_data.head(top_n)[selected_scenarios[1]].apply(
                        lambda x: f"${x:,.0f}" if any(word in metric.lower() for word in ['revenue', 'cost', 'profit', 'price']) else f"{x:,.0f}"
                    ),
                    marker_color='#EF4444'
                ))
                
                fig.update_layout(
                    title=f"{metric} by {dimension}",
                    xaxis_title=dimension,
                    yaxis_title=metric,
                    barmode='group',
                    plot_bgcolor='rgba(240, 242, 246, 0.5)',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        return results
    
    def _display_profitability_analysis(self, df: pd.DataFrame, financial_cols: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        Display profitability analysis
        """
        st.markdown("### 💰 Profitability Analysis")
        
        results = {}
        
        # Check for required columns
        revenue_cols = financial_cols.get('revenue_columns', [])
        expense_cols = financial_cols.get('expense_columns', [])
        
        if not revenue_cols or not expense_cols:
            st.warning("Need both revenue and expense columns for profitability analysis.")
            return results
        
        revenue_col = revenue_cols[0]
        expense_col = expense_cols[0]
        
        # Calculate profitability metrics
        total_revenue = df[revenue_col].sum()
        total_expense = df[expense_col].sum()
        total_profit = total_revenue - total_expense
        
        gross_margin = (total_profit / total_revenue * 100) if total_revenue > 0 else 0
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Total Revenue",
                value=f"${total_revenue:,.0f}"
            )
        
        with col2:
            st.metric(
                label="Total Expenses",
                value=f"${total_expense:,.0f}"
            )
        
        with col3:
            st.metric(
                label="Total Profit",
                value=f"${total_profit:,.0f}",
                delta=f"{gross_margin:.1f}%"
            )
        
        with col4:
            profit_margin_benchmark = self._get_benchmark_status(
                gross_margin, 
                self.kpi_definitions['gross_margin']['benchmark']
            )
            st.metric(
                label="Gross Margin",
                value=f"{gross_margin:.1f}%",
                delta=profit_margin_benchmark.split()[1] if ' ' in profit_margin_benchmark else profit_margin_benchmark
            )
        
        # Profitability by segment
        st.markdown("#### 📊 Profitability by Segment")
        
        segment_options = ['None'] + financial_cols.get('categorical_columns', [])
        segment = st.selectbox(
            "Analyze profitability by:",
            options=segment_options,
            key='profitability_segment'
        )
        
        if segment != 'None':
            segment_profit = df.groupby(segment).agg({
                revenue_col: 'sum',
                expense_col: 'sum'
            }).reset_index()
            
            segment_profit['Profit'] = segment_profit[revenue_col] - segment_profit[expense_col]
            segment_profit['Margin %'] = (segment_profit['Profit'] / segment_profit[revenue_col] * 100).replace([np.inf, -np.inf], 0)
            segment_profit = segment_profit.sort_values('Profit', ascending=False)
            
            # Display as table with custom styling
            st.dataframe(
                segment_profit.style.format({
                    revenue_col: '${:,.0f}',
                    expense_col: '${:,.0f}',
                    'Profit': '${:,.0f}',
                    'Margin %': '{:.1f}%'
                }).apply(
                    lambda x: ['background-color: #F0F9FF' if i % 2 == 0 else '' for i in range(len(x))],
                    axis=0
                ),
                use_container_width=True,
                height=400
            )
            
            results['segment_profitability'] = segment_profit
            
            # Visualization
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=[f"Profit by {segment}", f"Margin % by {segment}"],
                vertical_spacing=0.15
            )
            
            # Profit bars with conditional coloring
            colors = ['#10B981' if x >= 0 else '#EF4444' for x in segment_profit['Profit']]
            
            fig.add_trace(
                go.Bar(
                    x=segment_profit[segment],
                    y=segment_profit['Profit'],
                    name='Profit',
                    marker_color=colors,
                    text=segment_profit['Profit'].apply(lambda x: f"${x:,.0f}"),
                    textposition='auto'
                ),
                row=1, col=1
            )
            
            # Margin line
            fig.add_trace(
                go.Scatter(
                    x=segment_profit[segment],
                    y=segment_profit['Margin %'],
                    mode='lines+markers',
                    name='Margin %',
                    line=dict(color='#3B82F6', width=2),
                    marker=dict(size=8, color='#1D4ED8')
                ),
                row=2, col=1
            )
            
            fig.update_layout(
                height=600, 
                showlegend=True,
                plot_bgcolor='rgba(240, 242, 246, 0.5)'
            )
            fig.update_yaxes(title_text="Profit ($)", row=1, col=1)
            fig.update_yaxes(title_text="Margin %", row=2, col=1)
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Profitability over time
        date_cols = financial_cols.get('date_columns', [])
        if date_cols:
            st.markdown("#### 📈 Profitability Trend")
            
            date_col = date_cols[0]
            df_time = df.copy()
            
            if not pd.api.types.is_datetime64_any_dtype(df_time[date_col]):
                df_time[date_col] = pd.to_datetime(df_time[date_col], errors='coerce')
            
            period = st.selectbox(
                "Time period for trend:",
                options=self.trend_periods,
                index=2,
                key='profit_period'
            )
            
            if period == 'Monthly':
                df_time['period'] = df_time[date_col].dt.to_period('M').apply(lambda r: r.start_time)
            elif period == 'Quarterly':
                df_time['period'] = df_time[date_col].dt.to_period('Q').apply(lambda r: r.start_time)
            elif period == 'Yearly':
                df_time['period'] = df_time[date_col].dt.to_period('Y').apply(lambda r: r.start_time)
            
            time_profit = df_time.groupby('period').agg({
                revenue_col: 'sum',
                expense_col: 'sum'
            }).reset_index()
            
            time_profit['Profit'] = time_profit[revenue_col] - time_profit[expense_col]
            time_profit['Margin %'] = (time_profit['Profit'] / time_profit[revenue_col] * 100).replace([np.inf, -np.inf], 0)
            
            # Dual-axis chart
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            fig.add_trace(
                go.Bar(
                    x=time_profit['period'].astype(str),
                    y=time_profit['Profit'],
                    name='Profit',
                    marker_color='#10B981',
                    opacity=0.7,
                    text=time_profit['Profit'].apply(lambda x: f"${x:,.0f}"),
                    textposition='auto'
                ),
                secondary_y=False
            )
            
            fig.add_trace(
                go.Scatter(
                    x=time_profit['period'].astype(str),
                    y=time_profit['Margin %'],
                    mode='lines+markers',
                    name='Margin %',
                    line=dict(color='#3B82F6', width=3),
                    marker=dict(size=8, color='#1D4ED8')
                ),
                secondary_y=True
            )
            
            fig.update_layout(
                title=f"Profitability Trend ({period})",
                xaxis_title="Period",
                showlegend=True,
                plot_bgcolor='rgba(240, 242, 246, 0.5)',
                height=500
            )
            
            fig.update_yaxes(title_text="Profit ($)", secondary_y=False)
            fig.update_yaxes(title_text="Margin %", secondary_y=True)
            
            st.plotly_chart(fig, use_container_width=True)
            
            results['time_profitability'] = time_profit
        
        return results
    
    def _display_cash_flow_analysis(self, df: pd.DataFrame, financial_cols: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        Display cash flow analysis
        """
        st.markdown("### 💧 Cash Flow Analysis")
        
        results = {}
        
        # Cash flow columns detection
        cash_keywords = {
            'inflow': ['receivable', 'collection', 'payment_received', 'cash_in', 'revenue'],
            'outflow': ['payable', 'payment_made', 'expense', 'cost', 'cash_out'],
            'balance': ['balance', 'cash_balance', 'bank_balance']
        }
        
        cash_cols = {key: [] for key in cash_keywords.keys()}
        
        for col in df.columns:
            col_lower = str(col).lower()
            for category, keywords in cash_keywords.items():
                if any(keyword in col_lower for keyword in keywords):
                    cash_cols[category].append(col)
        
        # If no specific cash columns, use revenue and expense as proxy
        if not cash_cols['inflow'] and financial_cols.get('revenue_columns'):
            cash_cols['inflow'] = financial_cols['revenue_columns'][:1]
        if not cash_cols['outflow'] and financial_cols.get('expense_columns'):
            cash_cols['outflow'] = financial_cols['expense_columns'][:1]
        
        if not cash_cols['inflow'] or not cash_cols['outflow']:
            st.warning("Need both cash inflow and outflow columns for cash flow analysis.")
            st.info("**Tip:** Use columns with keywords like 'revenue' (inflow) and 'expense' (outflow)")
            return results
        
        inflow_col = cash_cols['inflow'][0]
        outflow_col = cash_cols['outflow'][0]
        
        # Calculate cash flow metrics
        total_inflow = df[inflow_col].sum()
        total_outflow = df[outflow_col].sum()
        net_cash_flow = total_inflow - total_outflow
        cash_flow_margin = (net_cash_flow / total_inflow * 100) if total_inflow > 0 else 0
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Total Inflow",
                value=f"${total_inflow:,.0f}"
            )
        
        with col2:
            st.metric(
                label="Total Outflow",
                value=f"${total_outflow:,.0f}"
            )
        
        with col3:
            st.metric(
                label="Net Cash Flow",
                value=f"${net_cash_flow:,.0f}",
                delta=f"{cash_flow_margin:.1f}%"
            )
        
        with col4:
            burn_rate = total_outflow / 30 if total_outflow > 0 else 0  # Monthly burn rate
            st.metric(
                label="Monthly Burn Rate",
                value=f"${burn_rate:,.0f}/month"
            )
        
        # Cash flow by time period
        date_cols = financial_cols.get('date_columns', [])
        if date_cols:
            st.markdown("#### 📈 Cash Flow Trend")
            
            date_col = date_cols[0]
            df_cash = df.copy()
            
            if not pd.api.types.is_datetime64_any_dtype(df_cash[date_col]):
                df_cash[date_col] = pd.to_datetime(df_cash[date_col], errors='coerce')
            
            period = st.selectbox(
                "Time period:",
                options=self.trend_periods,
                index=2,
                key='cash_period'
            )
            
            if period == 'Monthly':
                df_cash['period'] = df_cash[date_col].dt.to_period('M').apply(lambda r: r.start_time)
            elif period == 'Quarterly':
                df_cash['period'] = df_cash[date_col].dt.to_period('Q').apply(lambda r: r.start_time)
            
            cash_trend = df_cash.groupby('period').agg({
                inflow_col: 'sum',
                outflow_col: 'sum'
            }).reset_index()
            
            cash_trend['Net Cash Flow'] = cash_trend[inflow_col] - cash_trend[outflow_col]
            cash_trend['Cumulative Cash'] = cash_trend['Net Cash Flow'].cumsum()
            
            # Waterfall chart for cash flow
            fig = go.Figure(go.Waterfall(
                name="Cash Flow",
                orientation="v",
                measure=["relative"] * (len(cash_trend) - 1) + ["total"],
                x=cash_trend['period'].astype(str),
                y=cash_trend['Net Cash Flow'],
                text=cash_trend['Net Cash Flow'].apply(lambda x: f"${x:,.0f}"),
                connector={"line": {"color": "rgb(63, 63, 63)"}},
                increasing={"marker": {"color": "#10B981"}},
                decreasing={"marker": {"color": "#EF4444"}},
                totals={"marker": {"color": "#3B82F6"}}
            ))
            
            fig.update_layout(
                title=f"Cash Flow Waterfall ({period})",
                showlegend=True,
                waterfallgap=0.3,
                plot_bgcolor='rgba(240, 242, 246, 0.5)',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Cumulative cash line chart
            fig2 = go.Figure()
            
            fig2.add_trace(go.Scatter(
                x=cash_trend['period'].astype(str),
                y=cash_trend['Cumulative Cash'],
                mode='lines+markers',
                name='Cumulative Cash',
                line=dict(color='#10B981', width=3),
                fill='tozeroy',
                fillcolor='rgba(16, 185, 129, 0.1)',
                marker=dict(size=8, color='#047857')
            ))
            
            fig2.update_layout(
                title=f"Cumulative Cash Position ({period})",
                yaxis_title="Cumulative Cash ($)",
                showlegend=True,
                plot_bgcolor='rgba(240, 242, 246, 0.5)',
                height=400
            )
            
            st.plotly_chart(fig2, use_container_width=True)
            
            results['cash_trend'] = cash_trend
        
        # Cash flow ratios
        st.markdown("#### 📊 Cash Flow Ratios")
        
        if total_outflow > 0:
            cash_flow_ratios = {
                'Cash Flow Margin': cash_flow_margin,
                'Inflow to Outflow Ratio': total_inflow / total_outflow,
                'Operating Cash Flow Ratio': net_cash_flow / total_outflow if total_outflow > 0 else 0
            }
            
            ratio_df = pd.DataFrame(list(cash_flow_ratios.items()), columns=['Ratio', 'Value'])
            
            # Format based on ratio type
            def format_ratio(row):
                if 'Ratio' in row['Ratio']:
                    return f"{row['Value']:.2f}x"
                elif 'Margin' in row['Ratio']:
                    return f"{row['Value']:.1f}%"
                else:
                    return f"{row['Value']:.2f}"
            
            ratio_df['Formatted'] = ratio_df.apply(format_ratio, axis=1)
            
            # Display with custom styling
            st.dataframe(
                ratio_df[['Ratio', 'Formatted']].style.apply(
                    lambda x: ['background-color: #F0F9FF' if i % 2 == 0 else '' for i in range(len(x))],
                    axis=0
                ),
                use_container_width=True,
                height=150
            )
            
            results['cash_ratios'] = cash_flow_ratios
        
        return results
    
    def _display_deep_dive_analysis(self, df: pd.DataFrame, financial_cols: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        Display deep dive analysis with advanced insights
        """
        st.markdown("### 📋 Deep Dive Analysis")
        
        results = {}
        
        # Top/Bottom analysis
        st.markdown("#### 🏆 Top/Bottom Performers")
        
        numeric_cols = financial_cols.get('numeric_columns', [])
        categorical_cols = financial_cols.get('categorical_columns', [])
        
        if numeric_cols and categorical_cols:
            metric = st.selectbox(
                "Select metric for ranking:",
                options=numeric_cols,
                key='deep_dive_metric'
            )
            
            dimension = st.selectbox(
                "Rank by dimension:",
                options=categorical_cols,
                key='deep_dive_dimension'
            )
            
            n_items = st.slider(
                "Number of items to show:",
                min_value=3,
                max_value=20,
                value=10,
                key='n_items_slider'
            )
            
            # Calculate top performers
            top_performers = df.groupby(dimension)[metric].sum().nlargest(n_items).reset_index()
            bottom_performers = df.groupby(dimension)[metric].sum().nsmallest(n_items).reset_index()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**Top {n_items} {dimension} by {metric}**")
                st.dataframe(
                    top_performers.style.format({
                        metric: '${:,.0f}' if any(word in metric.lower() for word in ['revenue', 'cost', 'profit', 'price']) else '{:,.0f}'
                    }).apply(
                        lambda x: ['background-color: #F0F9FF' if i % 2 == 0 else '' for i in range(len(x))],
                        axis=0
                    ),
                    use_container_width=True,
                    height=400
                )
            
            with col2:
                st.markdown(f"**Bottom {n_items} {dimension} by {metric}**")
                st.dataframe(
                    bottom_performers.style.format({
                        metric: '${:,.0f}' if any(word in metric.lower() for word in ['revenue', 'cost', 'profit', 'price']) else '{:,.0f}'
                    }).apply(
                        lambda x: ['background-color: #F0F9FF' if i % 2 == 0 else '' for i in range(len(x))],
                        axis=0
                    ),
                    use_container_width=True,
                    height=400
                )
            
            results['top_performers'] = top_performers
            results['bottom_performers'] = bottom_performers
            
            # Pareto analysis (80/20 rule)
            st.markdown("#### 📊 Pareto Analysis (80/20 Rule)")
            
            pareto_data = df.groupby(dimension)[metric].sum().sort_values(ascending=False).reset_index()
            pareto_data['Cumulative'] = pareto_data[metric].cumsum()
            pareto_data['Cumulative %'] = (pareto_data['Cumulative'] / pareto_data[metric].sum() * 100)
            
            # Find 80% point
            eighty_percent_point = pareto_data[pareto_data['Cumulative %'] >= 80].iloc[0] if len(pareto_data[pareto_data['Cumulative %'] >= 80]) > 0 else None
            
            if eighty_percent_point is not None:
                num_items_80 = pareto_data[pareto_data['Cumulative %'] <= 80].shape[0]
                total_items = len(pareto_data)
                
                st.info(f"**📌 Pareto Insight:** {num_items_80} out of {total_items} {dimension}s ({(num_items_80/total_items*100):.1f}%) contribute to 80% of total {metric}")
                
                # Pareto chart
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                
                fig.add_trace(
                    go.Bar(
                        x=pareto_data[dimension],
                        y=pareto_data[metric],
                        name=metric,
                        marker_color='#3B82F6',
                        opacity=0.7
                    ),
                    secondary_y=False
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=pareto_data[dimension],
                        y=pareto_data['Cumulative %'],
                        name='Cumulative %',
                        line=dict(color='#EF4444', width=3),
                        marker=dict(size=6, color='#DC2626')
                    ),
                    secondary_y=True
                )
                
                # Add 80% line
                fig.add_hline(
                    y=80,
                    line_dash="dash",
                    line_color="#10B981",
                    opacity=0.7,
                    annotation_text="80% Threshold",
                    annotation_position="top right",
                    secondary_y=True
                )
                
                fig.update_layout(
                    title=f"Pareto Chart: {metric} by {dimension}",
                    xaxis_title=dimension,
                    showlegend=True,
                    plot_bgcolor='rgba(240, 242, 246, 0.5)',
                    height=500
                )
                
                fig.update_yaxes(title_text=metric, secondary_y=False)
                fig.update_yaxes(title_text="Cumulative %", secondary_y=True, range=[0, 100])
                
                st.plotly_chart(fig, use_container_width=True)
            
            results['pareto_analysis'] = pareto_data
        
        # Correlation analysis
        st.markdown("#### 🔗 Correlation Analysis")
        
        if len(numeric_cols) >= 2:
            selected_numeric = st.multiselect(
                "Select numeric columns for correlation:",
                options=numeric_cols,
                default=numeric_cols[:min(5, len(numeric_cols))],
                key='correlation_cols'
            )
            
            if len(selected_numeric) >= 2:
                corr_matrix = df[selected_numeric].corr()
                
                # Heatmap
                fig = go.Figure(data=go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.index,
                    colorscale='RdBu',
                    zmin=-1,
                    zmax=1,
                    text=corr_matrix.values.round(2),
                    texttemplate='%{text}',
                    textfont={"size": 12, "color": "white"},
                    hovertemplate='Correlation between %{x} and %{y}: %{text}<extra></extra>'
                ))
                
                fig.update_layout(
                    title="Correlation Heatmap",
                    height=500,
                    xaxis_showgrid=False,
                    yaxis_showgrid=False,
                    plot_bgcolor='rgba(240, 242, 246, 0.5)'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Strongest correlations
                corr_pairs = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_pairs.append({
                            'Variable 1': corr_matrix.columns[i],
                            'Variable 2': corr_matrix.columns[j],
                            'Correlation': corr_matrix.iloc[i, j]
                        })
                
                corr_df = pd.DataFrame(corr_pairs)
                corr_df['Abs Correlation'] = corr_df['Correlation'].abs()
                corr_df = corr_df.sort_values('Abs Correlation', ascending=False)
                
                st.markdown("**🔍 Strongest Correlations:**")
                st.dataframe(
                    corr_df.head(10).style.format({'Correlation': '{:.3f}', 'Abs Correlation': '{:.3f}'}).apply(
                        lambda x: ['background-color: #F0F9FF' if i % 2 == 0 else '' for i in range(len(x))],
                        axis=0
                    ),
                    use_container_width=True,
                    height=400
                )
                
                results['correlation_analysis'] = corr_matrix
        
        return results


# Streamlit integration function
def display_analytics_section(dataframes: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    Main function to display analytics interface
    """
    if not dataframes:
        st.warning("Please upload and filter data first.")
        return {}
    
    analytics = FinancialAnalytics()
    analysis_results = analytics.display_analytics_dashboard(dataframes)
    
    return analysis_results