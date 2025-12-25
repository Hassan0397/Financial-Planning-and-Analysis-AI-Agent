"""
Scenario Analysis Module
What-if scenario modeling for FP&A
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


class ScenarioAnalyzer:
    """
    What-if scenario analysis for financial planning
    """
    
    def __init__(self):
        self.scenarios = {}
        self.base_scenario = {}
        self.scenario_results = {}
        self.assumption_templates = self._initialize_assumption_templates()
    
    def _initialize_assumption_templates(self) -> Dict[str, Dict]:
        """Initialize common scenario assumption templates"""
        return {
            'revenue_growth': {
                'name': 'Revenue Growth Rate',
                'description': 'Annual revenue growth rate',
                'default': 0.05,
                'min': -0.5,
                'max': 2.0,
                'step': 0.01,
                'format': 'percentage',
                'impact_metrics': ['Revenue', 'Profit', 'EBITDA', 'Cash Flow']
            },
            'price_change': {
                'name': 'Price Change',
                'description': 'Change in product/service prices',
                'default': 0.0,
                'min': -0.3,
                'max': 0.5,
                'step': 0.01,
                'format': 'percentage',
                'impact_metrics': ['Revenue', 'Profit Margin', 'Unit Economics']
            },
            'volume_change': {
                'name': 'Volume Change',
                'description': 'Change in sales volume/quantity',
                'default': 0.0,
                'min': -0.5,
                'max': 1.0,
                'step': 0.01,
                'format': 'percentage',
                'impact_metrics': ['Revenue', 'COGS', 'Variable Costs']
            },
            'cogs_change': {
                'name': 'COGS Change',
                'description': 'Change in Cost of Goods Sold',
                'default': 0.0,
                'min': -0.2,
                'max': 0.3,
                'step': 0.01,
                'format': 'percentage',
                'impact_metrics': ['Gross Profit', 'Profit Margin', 'COGS']
            },
            'operating_expense_change': {
                'name': 'Operating Expense Change',
                'description': 'Change in operating expenses',
                'default': 0.0,
                'min': -0.3,
                'max': 0.4,
                'step': 0.01,
                'format': 'percentage',
                'impact_metrics': ['Operating Profit', 'EBITDA', 'Net Income']
            },
            'employee_count': {
                'name': 'Employee Count Change',
                'description': 'Change in number of employees',
                'default': 0.0,
                'min': -0.5,
                'max': 1.0,
                'step': 0.01,
                'format': 'percentage',
                'impact_metrics': ['Personnel Costs', 'Operating Expenses', 'Productivity']
            },
            'salary_change': {
                'name': 'Salary Change',
                'description': 'Change in average salary',
                'default': 0.03,
                'min': -0.1,
                'max': 0.2,
                'step': 0.01,
                'format': 'percentage',
                'impact_metrics': ['Personnel Costs', 'Operating Expenses']
            },
            'interest_rate': {
                'name': 'Interest Rate Change',
                'description': 'Change in interest rates',
                'default': 0.0,
                'min': -0.02,
                'max': 0.05,
                'step': 0.001,
                'format': 'percentage',
                'impact_metrics': ['Interest Expense', 'Net Income', 'Debt Service']
            },
            'tax_rate': {
                'name': 'Tax Rate Change',
                'description': 'Change in effective tax rate',
                'default': 0.25,
                'min': 0.0,
                'max': 0.5,
                'step': 0.01,
                'format': 'percentage',
                'impact_metrics': ['Net Income', 'Tax Expense', 'After-tax Profit']
            },
            'payment_terms': {
                'name': 'Payment Terms Change',
                'description': 'Change in accounts receivable days',
                'default': 0,
                'min': -30,
                'max': 30,
                'step': 1,
                'format': 'days',
                'impact_metrics': ['Cash Flow', 'Working Capital', 'DSO']
            },
            'inventory_turnover': {
                'name': 'Inventory Turnover Change',
                'description': 'Change in inventory turnover rate',
                'default': 0.0,
                'min': -0.5,
                'max': 1.0,
                'step': 0.1,
                'format': 'decimal',
                'impact_metrics': ['Inventory Costs', 'Working Capital', 'COGS']
            },
            'market_share': {
                'name': 'Market Share Change',
                'description': 'Change in market share',
                'default': 0.0,
                'min': -0.2,
                'max': 0.3,
                'step': 0.01,
                'format': 'percentage',
                'impact_metrics': ['Revenue', 'Volume', 'Market Position']
            },
            'exchange_rate': {
                'name': 'Exchange Rate Change',
                'description': 'Change in foreign exchange rates',
                'default': 0.0,
                'min': -0.3,
                'max': 0.3,
                'step': 0.01,
                'format': 'percentage',
                'impact_metrics': ['Foreign Revenue', 'COGS', 'Currency Impact']
            },
            'discount_rate': {
                'name': 'Discount Rate Change',
                'description': 'Change in customer discount rate',
                'default': 0.0,
                'min': -0.1,
                'max': 0.2,
                'step': 0.01,
                'format': 'percentage',
                'impact_metrics': ['Revenue', 'Gross Margin', 'Net Price']
            }
        }
    
    def display_scenario_interface(self, dataframes: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Main Streamlit interface for scenario analysis
        """
        st.subheader("🎯 Scenario & What-If Analysis")
        
        if not dataframes:
            st.warning("⚠️ No data available. Please upload and analyze data first.")
            return {}
        
        # Get current data
        current_data = dataframes
        
        # File selection
        primary_file = st.selectbox(
            "Select file for scenario analysis:",
            options=list(current_data.keys()),
            help="This file should contain financial data for scenario modeling"
        )
        
        if not primary_file:
            return {}
        
        df = current_data[primary_file]
        
        # Detect financial metrics
        financial_metrics = self._detect_financial_metrics(df)
        
        if not financial_metrics:
            st.error("❌ No financial metrics found for scenario analysis.")
            st.info("**Tip:** Ensure your data contains columns like Revenue, Expenses, Profit, etc.")
            return {}
        
        # Create tabs for different scenario analysis sections
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🔧 Create Scenarios", 
            "📊 Compare Scenarios", 
            "📈 Impact Analysis",
            "🎯 Quick Scenarios",
            "📋 Scenario Library"
        ])
        
        scenario_results = {}
        
        with tab1:
            scenario_results['created_scenarios'] = self._create_scenarios_tab(df, financial_metrics)
        
        with tab2:
            scenario_results['comparison'] = self._compare_scenarios_tab(df, financial_metrics)
        
        with tab3:
            scenario_results['impact_analysis'] = self._impact_analysis_tab(df, financial_metrics)
        
        with tab4:
            scenario_results['quick_scenarios'] = self._quick_scenarios_tab(df, financial_metrics)
        
        with tab5:
            scenario_results['scenario_library'] = self._scenario_library_tab()
        
        # Store results
        self.scenario_results = scenario_results
        
        return scenario_results
    
    def _detect_financial_metrics(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Detect financial metrics in the dataframe"""
        financial_keywords = {
            'revenue': ['revenue', 'sales', 'income', 'turnover'],
            'expense': ['expense', 'cost', 'cogs', 'operating_expense', 'overhead'],
            'profit': ['profit', 'margin', 'net_income', 'ebitda', 'gross_profit'],
            'cash_flow': ['cash', 'flow', 'receivable', 'payable', 'working_capital'],
            'volume': ['quantity', 'volume', 'units', 'qty', 'customers'],
            'price': ['price', 'unit_price', 'rate', 'fee'],
            'asset': ['asset', 'inventory', 'receivable', 'property'],
            'liability': ['liability', 'debt', 'payable', 'loan']
        }
        
        detected_metrics = {key: [] for key in financial_keywords.keys()}
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in numeric_cols:
            col_lower = str(col).lower()
            
            for category, keywords in financial_keywords.items():
                if any(keyword in col_lower for keyword in keywords):
                    detected_metrics[category].append(col)
        
        # Also add all numeric columns as potential metrics
        detected_metrics['all_numeric'] = numeric_cols
        
        return detected_metrics
    
    def _create_scenarios_tab(self, df: pd.DataFrame, financial_metrics: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        Create custom what-if scenarios
        """
        st.markdown("### 🔧 Create Custom Scenarios")
        
        results = {}
        
        # Scenario name
        scenario_name = st.text_input(
            "Scenario Name:",
            placeholder="e.g., 'Optimistic Growth', 'Cost Reduction', 'Market Expansion'",
            help="Give your scenario a descriptive name"
        )
        
        if not scenario_name:
            st.info("Enter a scenario name to begin")
            return results
        
        # Description
        scenario_description = st.text_area(
            "Scenario Description:",
            placeholder="Describe the assumptions and rationale for this scenario...",
            help="Explain what this scenario represents and why"
        )
        
        # Two-column layout for assumptions
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Financial Assumptions")
            
            selected_assumptions = {}
            
            # Revenue assumptions
            if financial_metrics.get('revenue'):
                st.markdown("##### Revenue Assumptions")
                
                revenue_col = st.selectbox(
                    "Select revenue metric:",
                    options=financial_metrics['revenue'],
                    key='scenario_revenue'
                )
                
                revenue_change = st.slider(
                    "Revenue Change %",
                    min_value=-50.0,
                    max_value=200.0,
                    value=0.0,
                    step=0.5,
                    help="Percentage change in revenue"
                )
                
                selected_assumptions[f'{revenue_col}_change'] = revenue_change / 100
            
            # Expense assumptions
            if financial_metrics.get('expense'):
                st.markdown("##### Expense Assumptions")
                
                expense_cols = st.multiselect(
                    "Select expense metrics to adjust:",
                    options=financial_metrics['expense'],
                    default=financial_metrics['expense'][:min(3, len(financial_metrics['expense']))]
                )
                
                for expense_col in expense_cols:
                    expense_change = st.slider(
                        f"{expense_col} Change %",
                        min_value=-30.0,
                        max_value=50.0,
                        value=0.0,
                        step=0.5,
                        key=f'expense_{expense_col}'
                    )
                    
                    selected_assumptions[f'{expense_col}_change'] = expense_change / 100
        
        with col2:
            st.markdown("#### 🎯 Business Drivers")
            
            # Volume assumptions
            if financial_metrics.get('volume'):
                st.markdown("##### Volume Assumptions")
                
                volume_col = st.selectbox(
                    "Select volume metric:",
                    options=financial_metrics['volume'],
                    key='scenario_volume'
                )
                
                volume_change = st.slider(
                    "Volume Change %",
                    min_value=-50.0,
                    max_value=150.0,
                    value=0.0,
                    step=1.0,
                    help="Percentage change in sales volume"
                )
                
                selected_assumptions[f'{volume_col}_change'] = volume_change / 100
            
            # Price assumptions
            if financial_metrics.get('price'):
                st.markdown("##### Price Assumptions")
                
                price_col = st.selectbox(
                    "Select price metric:",
                    options=financial_metrics['price'],
                    key='scenario_price'
                )
                
                price_change = st.slider(
                    "Price Change %",
                    min_value=-20.0,
                    max_value=50.0,
                    value=0.0,
                    step=0.5,
                    help="Percentage change in prices"
                )
                
                selected_assumptions[f'{price_col}_change'] = price_change / 100
            
            # Additional business drivers
            st.markdown("##### Other Assumptions")
            
            market_growth = st.slider(
                "Market Growth Rate %",
                min_value=-10.0,
                max_value=30.0,
                value=5.0,
                step=0.5
            )
            selected_assumptions['market_growth'] = market_growth / 100
            
            inflation_rate = st.slider(
                "Inflation Rate %",
                min_value=0.0,
                max_value=15.0,
                value=2.5,
                step=0.1
            )
            selected_assumptions['inflation_rate'] = inflation_rate / 100
        
        # Time horizon
        st.markdown("#### 📅 Time Horizon")
        
        col1, col2 = st.columns(2)
        
        with col1:
            time_periods = st.number_input(
                "Number of periods:",
                min_value=1,
                max_value=60,
                value=12,
                help="Number of future periods for scenario projection"
            )
        
        with col2:
            period_type = st.selectbox(
                "Period type:",
                options=['Months', 'Quarters', 'Years'],
                help="Time period for projections"
            )
        
        # Apply scenario button
        if st.button("🚀 Apply Scenario", type="primary", use_container_width=True):
            with st.spinner("Calculating scenario impacts..."):
                # Calculate scenario impacts
                scenario_data = self._calculate_scenario_impacts(
                    df=df,
                    scenario_name=scenario_name,
                    assumptions=selected_assumptions,
                    time_periods=time_periods,
                    period_type=period_type
                )
                
                if scenario_data:
                    results = scenario_data
                    
                    # Store scenario
                    scenario_key = f"{scenario_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    self.scenarios[scenario_key] = {
                        'name': scenario_name,
                        'description': scenario_description,
                        'assumptions': selected_assumptions,
                        'data': scenario_data,
                        'created_at': datetime.now().isoformat()
                    }
                    
                    # Store in session state
                    if 'scenarios' not in st.session_state:
                        st.session_state.scenarios = {}
                    st.session_state.scenarios[scenario_key] = self.scenarios[scenario_key]
                    
                    # Display scenario results
                    self._display_scenario_results(scenario_data, scenario_name)
        
        return results
    
    def _calculate_scenario_impacts(self, df: pd.DataFrame, scenario_name: str, 
                                  assumptions: Dict[str, float], time_periods: int, 
                                  period_type: str) -> Dict[str, Any]:
        """Calculate the impacts of scenario assumptions"""
        results = {
            'scenario_name': scenario_name,
            'assumptions': assumptions,
            'time_periods': time_periods,
            'period_type': period_type,
            'base_values': {},
            'scenario_values': {},
            'impacts': {},
            'projections': {}
        }
        
        try:
            # Calculate base values (current state)
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                # Use appropriate aggregation based on column type
                if any(keyword in str(col).lower() for keyword in ['revenue', 'sales', 'profit', 'income']):
                    base_value = df[col].sum()
                else:
                    base_value = df[col].mean()
                
                results['base_values'][col] = base_value
            
            # Apply assumptions to calculate scenario values
            for col, base_value in results['base_values'].items():
                scenario_value = base_value
                
                # Apply relevant assumptions
                for assumption_key, assumption_value in assumptions.items():
                    if col in assumption_key or assumption_key in ['market_growth', 'inflation_rate']:
                        # Simple proportional adjustment for now
                        scenario_value = scenario_value * (1 + assumption_value)
                
                results['scenario_values'][col] = scenario_value
                
                # Calculate impact
                impact_amount = scenario_value - base_value
                impact_pct = (impact_amount / base_value * 100) if base_value != 0 else 0
                
                results['impacts'][col] = {
                    'base': base_value,
                    'scenario': scenario_value,
                    'change_amount': impact_amount,
                    'change_pct': impact_pct
                }
            
            # Create projections over time
            projection_data = []
            
            for period in range(time_periods + 1):  # Include period 0 (current)
                period_data = {'Period': period}
                
                # Apply growth assumptions over time
                for col, base_value in results['base_values'].items():
                    projected_value = base_value
                    
                    # Apply growth over periods
                    for assumption_key, assumption_value in assumptions.items():
                        if col in assumption_key or assumption_key in ['market_growth', 'inflation_rate']:
                            # Compound growth
                            projected_value = projected_value * ((1 + assumption_value) ** period)
                    
                    period_data[col] = projected_value
                
                projection_data.append(period_data)
            
            results['projections_df'] = pd.DataFrame(projection_data)
            
            # Calculate summary metrics
            results['summary'] = self._calculate_scenario_summary(results)
            
            return results
        
        except Exception as e:
            st.error(f"Error calculating scenario: {str(e)}")
            return {}
    
    def _calculate_scenario_summary(self, scenario_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate summary metrics for scenario"""
        impacts = scenario_data['impacts']
        
        if not impacts:
            return {}
        
        # Find top impacts
        impact_items = []
        for metric, impact in impacts.items():
            impact_items.append({
                'metric': metric,
                'change_pct': impact['change_pct'],
                'change_amount': impact['change_amount']
            })
        
        # Sort by absolute impact
        impact_items.sort(key=lambda x: abs(x['change_amount']), reverse=True)
        
        # Calculate totals
        total_base = sum(impact['base'] for impact in impacts.values())
        total_scenario = sum(impact['scenario'] for impact in impacts.values())
        total_change = total_scenario - total_base
        total_change_pct = (total_change / total_base * 100) if total_base != 0 else 0
        
        summary = {
            'total_base': total_base,
            'total_scenario': total_scenario,
            'total_change': total_change,
            'total_change_pct': total_change_pct,
            'top_impacts': impact_items[:5],  # Top 5 impacts
            'positive_impacts': [item for item in impact_items if item['change_amount'] > 0],
            'negative_impacts': [item for item in impact_items if item['change_amount'] < 0]
        }
        
        return summary
    
    def _display_scenario_results(self, scenario_data: Dict[str, Any], scenario_name: str):
        """Display scenario results"""
        st.markdown(f"### 📊 Results: {scenario_name}")
        
        # Summary metrics
        summary = scenario_data.get('summary', {})
        
        if summary:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Base Total",
                    f"${summary['total_base']:,.0f}" if summary['total_base'] != 0 else f"{summary['total_base']:,.0f}"
                )
            
            with col2:
                st.metric(
                    "Scenario Total",
                    f"${summary['total_scenario']:,.0f}" if summary['total_scenario'] != 0 else f"{summary['total_scenario']:,.0f}"
                )
            
            with col3:
                st.metric(
                    "Total Change",
                    f"${summary['total_change']:,.0f}" if summary['total_change'] != 0 else f"{summary['total_change']:,.0f}"
                )
            
            with col4:
                st.metric(
                    "Change %",
                    f"{summary['total_change_pct']:+.1f}%",
                    delta_color="normal" if summary['total_change_pct'] >= 0 else "inverse"
                )
        
        # Top impacts
        st.markdown("#### 🎯 Top Impacts")
        
        top_impacts = summary.get('top_impacts', [])
        if top_impacts:
            impact_data = []
            for impact in top_impacts:
                impact_data.append({
                    'Metric': impact['metric'],
                    'Base Value': impact.get('base', 0),
                    'Scenario Value': impact.get('scenario', 0),
                    'Change Amount': impact['change_amount'],
                    'Change %': f"{impact['change_pct']:+.1f}%"
                })
            
            impact_df = pd.DataFrame(impact_data)
            
            # Format numbers
            for col in ['Base Value', 'Scenario Value', 'Change Amount']:
                impact_df[col] = impact_df[col].apply(lambda x: f"${x:,.0f}" if x != 0 else f"{x:,.0f}")
            
            st.dataframe(impact_df, use_container_width=True)
        
        # Projections over time
        st.markdown("#### 📈 Projections Over Time")
        
        projections_df = scenario_data.get('projections_df')
        if projections_df is not None and not projections_df.empty:
            # Select metrics to display
            numeric_cols = [col for col in projections_df.columns if col != 'Period']
            
            if len(numeric_cols) > 0:
                selected_metrics = st.multiselect(
                    "Select metrics to display:",
                    options=numeric_cols,
                    default=numeric_cols[:min(3, len(numeric_cols))]
                )
                
                if selected_metrics:
                    # Create line chart
                    fig = go.Figure()
                    
                    for metric in selected_metrics:
                        fig.add_trace(go.Scatter(
                            x=projections_df['Period'],
                            y=projections_df[metric],
                            mode='lines+markers',
                            name=metric,
                            line=dict(width=2)
                        ))
                    
                    fig.update_layout(
                        title=f"{scenario_name} Projections",
                        xaxis_title=f"Period ({scenario_data['period_type']})",
                        yaxis_title="Value",
                        hovermode='x unified',
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display projection table
                    with st.expander("📋 View Projection Table"):
                        display_df = projections_df.copy()
                        
                        # Format numbers
                        for col in selected_metrics:
                            display_df[col] = display_df[col].apply(lambda x: f"${x:,.0f}" if x != 0 else f"{x:,.0f}")
                        
                        st.dataframe(display_df[['Period'] + selected_metrics], use_container_width=True)
        
        # Assumptions used
        st.markdown("#### 🔧 Assumptions Used")
        
        assumptions = scenario_data.get('assumptions', {})
        if assumptions:
            assumption_data = []
            for key, value in assumptions.items():
                assumption_data.append({
                    'Assumption': key,
                    'Value': f"{value*100:.1f}%" if 'rate' in key or 'change' in key else f"{value:.3f}"
                })
            
            assumption_df = pd.DataFrame(assumption_data)
            st.dataframe(assumption_df, use_container_width=True)
    
    def _compare_scenarios_tab(self, df: pd.DataFrame, financial_metrics: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        Compare multiple scenarios
        """
        st.markdown("### 📊 Compare Scenarios")
        
        results = {}
        
        # Check for saved scenarios
        saved_scenarios = st.session_state.get('scenarios', {})
        
        if not saved_scenarios:
            st.info("No scenarios saved yet. Create scenarios in the 'Create Scenarios' tab first.")
            return results
        
        # Scenario selection
        scenario_keys = list(saved_scenarios.keys())
        scenario_names = [saved_scenarios[key]['name'] for key in scenario_keys]
        
        selected_scenarios = st.multiselect(
            "Select scenarios to compare:",
            options=scenario_names,
            default=scenario_names[:min(3, len(scenario_names))]
        )
        
        if len(selected_scenarios) < 2:
            st.info("Select at least 2 scenarios to compare")
            return results
        
        # Metric selection for comparison
        all_metrics = []
        for scenario_name in selected_scenarios:
            # Find the scenario key
            scenario_key = None
            for key, scenario in saved_scenarios.items():
                if scenario['name'] == scenario_name:
                    scenario_key = key
                    break
            
            if scenario_key and 'impacts' in saved_scenarios[scenario_key]['data']:
                impacts = saved_scenarios[scenario_key]['data']['impacts']
                all_metrics.extend(list(impacts.keys()))
        
        unique_metrics = list(set(all_metrics))
        
        if not unique_metrics:
            st.warning("No metrics available for comparison")
            return results
        
        comparison_metric = st.selectbox(
            "Select metric for comparison:",
            options=unique_metrics
        )
        
        # Comparison type
        comparison_type = st.radio(
            "Comparison type:",
            ["Absolute Values", "Percentage Change", "Change from Base"],
            horizontal=True
        )
        
        if st.button("🔍 Compare Scenarios", type="primary", use_container_width=True):
            # Gather comparison data
            comparison_data = []
            
            for scenario_name in selected_scenarios:
                # Find the scenario
                scenario_data = None
                for key, scenario in saved_scenarios.items():
                    if scenario['name'] == scenario_name:
                        scenario_data = scenario['data']
                        break
                
                if scenario_data and comparison_metric in scenario_data['impacts']:
                    impact = scenario_data['impacts'][comparison_metric]
                    
                    if comparison_type == "Absolute Values":
                        value = impact['scenario']
                    elif comparison_type == "Percentage Change":
                        value = impact['change_pct']
                    else:  # Change from Base
                        value = impact['change_amount']
                    
                    comparison_data.append({
                        'Scenario': scenario_name,
                        'Value': value,
                        'Base': impact['base'],
                        'Scenario_Value': impact['scenario'],
                        'Change_Pct': impact['change_pct']
                    })
            
            if comparison_data:
                results['comparison_data'] = comparison_data
                
                # Create comparison chart
                self._display_scenario_comparison(comparison_data, comparison_metric, comparison_type)
                
                # Create comparison table
                self._display_comparison_table(comparison_data, comparison_metric, comparison_type)
        
        return results
    
    def _display_scenario_comparison(self, comparison_data: List[Dict], metric: str, comparison_type: str):
        """Display scenario comparison chart"""
        # Create dataframe
        df_comparison = pd.DataFrame(comparison_data)
        
        # Sort by value
        df_comparison = df_comparison.sort_values('Value', ascending=False)
        
        # Create bar chart
        fig = go.Figure()
        
        # Color based on value (green for positive, red for negative)
        colors = []
        for val in df_comparison['Value']:
            if comparison_type == "Percentage Change":
                colors.append('green' if val >= 0 else 'red')
            elif comparison_type == "Change from Base":
                colors.append('green' if val >= 0 else 'red')
            else:
                colors.append('blue')  # Absolute values
        
        fig.add_trace(go.Bar(
            x=df_comparison['Scenario'],
            y=df_comparison['Value'],
            marker_color=colors,
            text=df_comparison['Value'].apply(lambda x: f"{x:,.0f}" if comparison_type != "Percentage Change" else f"{x:+.1f}%"),
            textposition='auto'
        ))
        
        fig.update_layout(
            title=f"{metric} - {comparison_type} Comparison",
            xaxis_title="Scenario",
            yaxis_title=comparison_type,
            showlegend=False,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _display_comparison_table(self, comparison_data: List[Dict], metric: str, comparison_type: str):
        """Display scenario comparison table"""
        # Create formatted table
        table_data = []
        
        for item in comparison_data:
            if comparison_type == "Absolute Values":
                value_display = f"${item['Value']:,.0f}" if item['Value'] != 0 else f"{item['Value']:,.0f}"
                base_display = f"${item['Base']:,.0f}" if item['Base'] != 0 else f"{item['Base']:,.0f}"
                scenario_display = f"${item['Scenario_Value']:,.0f}" if item['Scenario_Value'] != 0 else f"{item['Scenario_Value']:,.0f}"
                
                table_data.append({
                    'Scenario': item['Scenario'],
                    'Base Value': base_display,
                    'Scenario Value': scenario_display,
                    'Change Amount': f"${item['Scenario_Value'] - item['Base']:,.0f}",
                    'Change %': f"{item['Change_Pct']:+.1f}%"
                })
            
            elif comparison_type == "Percentage Change":
                table_data.append({
                    'Scenario': item['Scenario'],
                    'Change %': f"{item['Value']:+.1f}%",
                    'Base Value': f"${item['Base']:,.0f}" if item['Base'] != 0 else f"{item['Base']:,.0f}",
                    'Scenario Value': f"${item['Scenario_Value']:,.0f}" if item['Scenario_Value'] != 0 else f"{item['Scenario_Value']:,.0f}",
                    'Change Amount': f"${item['Scenario_Value'] - item['Base']:,.0f}"
                })
            
            else:  # Change from Base
                table_data.append({
                    'Scenario': item['Scenario'],
                    'Change Amount': f"${item['Value']:,.0f}" if item['Value'] != 0 else f"{item['Value']:,.0f}",
                    'Change %': f"{item['Change_Pct']:+.1f}%",
                    'Base Value': f"${item['Base']:,.0f}" if item['Base'] != 0 else f"{item['Base']:,.0f}",
                    'Scenario Value': f"${item['Scenario_Value']:,.0f}" if item['Scenario_Value'] != 0 else f"{item['Scenario_Value']:,.0f}"
                })
        
        comparison_df = pd.DataFrame(table_data)
        
        # Highlight best/worst based on comparison type
        def highlight_rows(row):
            if comparison_type == "Percentage Change":
                # Higher is better for percentage change
                best_value = max([item['Change_Pct'] for item in comparison_data])
                worst_value = min([item['Change_Pct'] for item in comparison_data])
                
                if row['Change %'] == f"{best_value:+.1f}%":
                    return ['background-color: lightgreen'] * len(row)
                elif row['Change %'] == f"{worst_value:+.1f}%":
                    return ['background-color: lightcoral'] * len(row)
            
            elif comparison_type == "Change from Base":
                # Higher is better for change amount
                best_value = max([item['Value'] for item in comparison_data])
                worst_value = min([item['Value'] for item in comparison_data])
                
                if row['Change Amount'] == f"${best_value:,.0f}":
                    return ['background-color: lightgreen'] * len(row)
                elif row['Change Amount'] == f"${worst_value:,.0f}":
                    return ['background-color: lightcoral'] * len(row)
            
            return [''] * len(row)
        
        styled_df = comparison_df.style.apply(highlight_rows, axis=1)
        st.dataframe(styled_df, use_container_width=True)
    
    def _impact_analysis_tab(self, df: pd.DataFrame, financial_metrics: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        Analyze impact of individual assumptions
        """
        st.markdown("### 📈 Sensitivity Analysis")
        
        results = {}
        
        st.markdown("""
        **Sensitivity Analysis** shows how changes in individual assumptions affect key metrics.
        This helps identify which assumptions have the greatest impact on results.
        """)
        
        # Select base metric
        all_metrics = financial_metrics.get('all_numeric', [])
        
        if not all_metrics:
            st.warning("No numeric metrics found for sensitivity analysis")
            return results
        
        target_metric = st.selectbox(
            "Select target metric to analyze:",
            options=all_metrics,
            help="This metric will be analyzed for sensitivity to changes"
        )
        
        # Select assumptions to test
        st.markdown("#### 🎯 Test Assumptions")
        
        available_assumptions = list(self.assumption_templates.keys())
        selected_assumptions = st.multiselect(
            "Select assumptions to test:",
            options=available_assumptions,
            default=available_assumptions[:min(5, len(available_assumptions))]
        )
        
        if not selected_assumptions:
            return results
        
        # Test ranges for each assumption
        sensitivity_ranges = {}
        
        for assumption_key in selected_assumptions:
            assumption = self.assumption_templates[assumption_key]
            
            st.markdown(f"##### {assumption['name']}")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                min_val = st.number_input(
                    "Min %",
                    value=float(assumption['min'] * 100),
                    step=0.5,
                    key=f"{assumption_key}_min"
                )
            
            with col2:
                max_val = st.number_input(
                    "Max %",
                    value=float(assumption['max'] * 100),
                    step=0.5,
                    key=f"{assumption_key}_max"
                )
            
            with col3:
                step_val = st.number_input(
                    "Step %",
                    value=float(assumption['step'] * 100),
                    step=0.1,
                    key=f"{assumption_key}_step"
                )
            
            sensitivity_ranges[assumption_key] = {
                'min': min_val / 100,
                'max': max_val / 100,
                'step': step_val / 100
            }
        
        # Run sensitivity analysis
        if st.button("📊 Run Sensitivity Analysis", type="primary", use_container_width=True):
            with st.spinner("Running sensitivity analysis..."):
                sensitivity_results = self._run_sensitivity_analysis(
                    df=df,
                    target_metric=target_metric,
                    assumptions=selected_assumptions,
                    ranges=sensitivity_ranges
                )
                
                if sensitivity_results:
                    results = sensitivity_results
                    
                    # Display sensitivity results
                    self._display_sensitivity_results(sensitivity_results, target_metric)
        
        return results
    
    def _run_sensitivity_analysis(self, df: pd.DataFrame, target_metric: str, 
                                assumptions: List[str], ranges: Dict[str, Dict]) -> Dict[str, Any]:
        """Run sensitivity analysis for given assumptions"""
        results = {
            'target_metric': target_metric,
            'assumptions': assumptions,
            'sensitivity_data': {},
            'tornado_data': {}
        }
        
        try:
            # Calculate base value
            if any(keyword in str(target_metric).lower() for keyword in ['revenue', 'sales', 'profit', 'income']):
                base_value = df[target_metric].sum()
            else:
                base_value = df[target_metric].mean()
            
            # Test each assumption
            for assumption_key in assumptions:
                assumption_range = ranges[assumption_key]
                assumption = self.assumption_templates[assumption_key]
                
                # Generate test values
                test_values = np.arange(
                    assumption_range['min'],
                    assumption_range['max'] + assumption_range['step'],
                    assumption_range['step']
                )
                
                # Calculate impacts
                impacts = []
                for test_value in test_values:
                    # Simple proportional impact calculation
                    # In a more advanced version, this would use proper financial modeling
                    impact_value = base_value * (1 + test_value)
                    impact_pct = test_value * 100  # Since it's proportional
                    
                    impacts.append({
                        'assumption_value': test_value,
                        'impact_value': impact_value,
                        'impact_pct': impact_pct,
                        'change_from_base': impact_value - base_value
                    })
                
                results['sensitivity_data'][assumption_key] = {
                    'name': assumption['name'],
                    'base_value': base_value,
                    'impacts': impacts,
                    'elasticity': self._calculate_elasticity(impacts, base_value)
                }
            
            # Prepare tornado chart data
            tornado_data = []
            for assumption_key, data in results['sensitivity_data'].items():
                # Get min and max impacts
                min_impact = min(data['impacts'], key=lambda x: x['impact_value'])
                max_impact = max(data['impacts'], key=lambda x: x['impact_value'])
                
                tornado_data.append({
                    'Assumption': data['name'],
                    'Min Impact': min_impact['change_from_base'],
                    'Max Impact': max_impact['change_from_base'],
                    'Range': max_impact['change_from_base'] - min_impact['change_from_base']
                })
            
            # Sort by range (largest impact first)
            tornado_data.sort(key=lambda x: abs(x['Range']), reverse=True)
            results['tornado_data'] = tornado_data
            
            return results
        
        except Exception as e:
            st.error(f"Sensitivity analysis failed: {str(e)}")
            return {}
    
    def _calculate_elasticity(self, impacts: List[Dict], base_value: float) -> float:
        """Calculate elasticity (sensitivity) of metric to assumption changes"""
        if len(impacts) < 2 or base_value == 0:
            return 0.0
        
        # Simple elasticity calculation: % change in metric / % change in assumption
        first_impact = impacts[0]
        last_impact = impacts[-1]
        
        metric_change_pct = ((last_impact['impact_value'] - first_impact['impact_value']) / base_value) * 100
        assumption_change_pct = (last_impact['assumption_value'] - first_impact['assumption_value']) * 100
        
        if assumption_change_pct == 0:
            return 0.0
        
        elasticity = metric_change_pct / assumption_change_pct
        return elasticity
    
    def _display_sensitivity_results(self, sensitivity_results: Dict[str, Any], target_metric: str):
        """Display sensitivity analysis results"""
        # Tornado chart
        st.markdown("#### 🌪️ Tornado Chart (Sensitivity Analysis)")
        
        tornado_data = sensitivity_results.get('tornado_data', [])
        
        if tornado_data:
            # Create tornado chart
            fig = go.Figure()
            
            # Add bars for negative impacts
            fig.add_trace(go.Bar(
                y=[d['Assumption'] for d in tornado_data],
                x=[min(0, d['Min Impact']) for d in tornado_data],
                name='Negative Impact',
                orientation='h',
                marker_color='red',
                text=[f"${d['Min Impact']:,.0f}" for d in tornado_data],
                textposition='auto'
            ))
            
            # Add bars for positive impacts
            fig.add_trace(go.Bar(
                y=[d['Assumption'] for d in tornado_data],
                x=[max(0, d['Max Impact']) for d in tornado_data],
                name='Positive Impact',
                orientation='h',
                marker_color='green',
                text=[f"${d['Max Impact']:,.0f}" for d in tornado_data],
                textposition='auto'
            ))
            
            fig.update_layout(
                title=f"Sensitivity of {target_metric} to Assumption Changes",
                xaxis_title=f"Change in {target_metric} ($)",
                yaxis_title="Assumption",
                barmode='overlay',
                height=max(300, len(tornado_data) * 40),
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display elasticity/sensitivity metrics
            st.markdown("#### 📊 Sensitivity Metrics")
            
            sensitivity_data = sensitivity_results.get('sensitivity_data', {})
            if sensitivity_data:
                elasticity_data = []
                for assumption_key, data in sensitivity_data.items():
                    elasticity_data.append({
                        'Assumption': data['name'],
                        'Elasticity': f"{data['elasticity']:.2f}",
                        'Impact Range': f"${data['impacts'][-1]['change_from_base'] - data['impacts'][0]['change_from_base']:,.0f}",
                        'Description': 'More Sensitive' if abs(data['elasticity']) > 1 else 'Less Sensitive'
                    })
                
                elasticity_df = pd.DataFrame(elasticity_data)
                st.dataframe(elasticity_df, use_container_width=True)
        
        # Detailed sensitivity curves
        st.markdown("#### 📈 Sensitivity Curves")
        
        sensitivity_data = sensitivity_results.get('sensitivity_data', {})
        
        if sensitivity_data and len(sensitivity_data) > 0:
            # Select assumption to view detailed curve
            selected_assumption = st.selectbox(
                "View detailed sensitivity curve for:",
                options=list(sensitivity_data.keys())
            )
            
            if selected_assumption:
                data = sensitivity_data[selected_assumption]
                
                # Create sensitivity curve
                fig = go.Figure()
                
                x_values = [impact['assumption_value'] * 100 for impact in data['impacts']]  # Convert to %
                y_values = [impact['impact_value'] for impact in data['impacts']]
                
                fig.add_trace(go.Scatter(
                    x=x_values,
                    y=y_values,
                    mode='lines+markers',
                    name='Sensitivity Curve',
                    line=dict(color='blue', width=2),
                    marker=dict(size=6)
                ))
                
                # Add base value line
                fig.add_hline(
                    y=data['base_value'],
                    line_dash="dash",
                    line_color="red",
                    annotation_text="Base Value"
                )
                
                fig.update_layout(
                    title=f"Sensitivity of {target_metric} to {data['name']}",
                    xaxis_title=f"{data['name']} Change (%)",
                    yaxis_title=target_metric,
                    hovermode='x unified',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    def _quick_scenarios_tab(self, df: pd.DataFrame, financial_metrics: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        Pre-defined quick scenarios
        """
        st.markdown("### 🎯 Quick Scenarios")
        
        results = {}
        
        st.markdown("""
        **Quick Scenarios** provide instant what-if analysis with common business scenarios.
        """)
        
        # Quick scenario templates
        quick_scenarios = {
            'growth_optimistic': {
                'name': 'Optimistic Growth',
                'description': 'High growth scenario with favorable market conditions',
                'assumptions': {
                    'revenue_growth': 0.20,  # 20% revenue growth
                    'volume_change': 0.15,    # 15% volume increase
                    'price_change': 0.05,     # 5% price increase
                    'cogs_change': -0.05,     # 5% COGS reduction
                    'operating_expense_change': 0.10  # 10% OpEx increase
                }
            },
            'growth_conservative': {
                'name': 'Conservative Growth',
                'description': 'Moderate growth with controlled expenses',
                'assumptions': {
                    'revenue_growth': 0.08,   # 8% revenue growth
                    'volume_change': 0.05,    # 5% volume increase
                    'price_change': 0.02,     # 2% price increase
                    'cogs_change': 0.00,      # No COGS change
                    'operating_expense_change': 0.03  # 3% OpEx increase
                }
            },
            'cost_reduction': {
                'name': 'Cost Reduction',
                'description': 'Focus on cost optimization and efficiency',
                'assumptions': {
                    'revenue_growth': 0.05,   # 5% revenue growth
                    'cogs_change': -0.15,     # 15% COGS reduction
                    'operating_expense_change': -0.10,  # 10% OpEx reduction
                    'employee_count': -0.05,  # 5% headcount reduction
                    'inventory_turnover': 0.20  # 20% improvement in inventory turnover
                }
            },
            'market_expansion': {
                'name': 'Market Expansion',
                'description': 'Aggressive expansion into new markets',
                'assumptions': {
                    'revenue_growth': 0.30,   # 30% revenue growth
                    'volume_change': 0.25,    # 25% volume increase
                    'market_share': 0.10,     # 10% market share increase
                    'operating_expense_change': 0.20,  # 20% OpEx increase (investment)
                    'discount_rate': 0.05     # 5% higher discounts for penetration
                }
            },
            'recession_scenario': {
                'name': 'Recession Scenario',
                'description': 'Economic downturn with reduced demand',
                'assumptions': {
                    'revenue_growth': -0.15,  # 15% revenue decline
                    'volume_change': -0.20,   # 20% volume decrease
                    'price_change': -0.05,    # 5% price reduction
                    'cogs_change': -0.10,     # 10% COGS reduction (cost cutting)
                    'operating_expense_change': -0.15  # 15% OpEx reduction
                }
            }
        }
        
        # Display scenario cards
        cols = st.columns(3)
        
        for i, (scenario_key, scenario) in enumerate(quick_scenarios.items()):
            with cols[i % 3]:
                st.markdown(f"""
                <div style="
                    background: white;
                    padding: 1rem;
                    border-radius: 0.5rem;
                    border: 1px solid #e5e7eb;
                    margin-bottom: 1rem;
                    height: 200px;
                ">
                    <h4>{scenario['name']}</h4>
                    <p style="font-size: 0.875rem; color: #6b7280;">
                        {scenario['description']}
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button(f"Apply {scenario['name']}", key=f"quick_{scenario_key}", use_container_width=True):
                    with st.spinner(f"Applying {scenario['name']} scenario..."):
                        # Calculate scenario impacts
                        scenario_data = self._calculate_scenario_impacts(
                            df=df,
                            scenario_name=scenario['name'],
                            assumptions=scenario['assumptions'],
                            time_periods=12,
                            period_type='Months'
                        )
                        
                        if scenario_data:
                            results = scenario_data
                            
                            # Store scenario
                            scenario_key = f"{scenario['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                            self.scenarios[scenario_key] = {
                                'name': scenario['name'],
                                'description': scenario['description'],
                                'assumptions': scenario['assumptions'],
                                'data': scenario_data,
                                'created_at': datetime.now().isoformat(),
                                'is_quick_scenario': True
                            }
                            
                            # Store in session state
                            if 'scenarios' not in st.session_state:
                                st.session_state.scenarios = {}
                            st.session_state.scenarios[scenario_key] = self.scenarios[scenario_key]
                            
                            # Display results
                            self._display_scenario_results(scenario_data, scenario['name'])
        
        return results
    
    def _scenario_library_tab(self) -> Dict[str, Any]:
        """
        Manage saved scenarios
        """
        st.markdown("### 📋 Scenario Library")
        
        results = {}
        
        # Check for saved scenarios
        saved_scenarios = st.session_state.get('scenarios', {})
        
        if not saved_scenarios:
            st.info("No scenarios saved yet. Create scenarios to build your library.")
            return results
        
        # Display scenario library
        for scenario_key, scenario in saved_scenarios.items():
            with st.expander(f"📁 {scenario['name']}", expanded=False):
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    st.write(f"**Description:** {scenario.get('description', 'No description')}")
                    st.write(f"**Created:** {scenario.get('created_at', 'Unknown')}")
                    
                    # Show key assumptions
                    if 'assumptions' in scenario:
                        st.write("**Key Assumptions:**")
                        for key, value in scenario['assumptions'].items():
                            st.write(f"- {key}: {value*100:.1f}%" if isinstance(value, float) else f"- {key}: {value}")
                
                with col2:
                    if st.button("📊 View", key=f"view_{scenario_key}", use_container_width=True):
                        # This would load and display the scenario
                        st.info(f"Viewing {scenario['name']}")
                
                with col3:
                    if st.button("🗑️ Delete", key=f"delete_{scenario_key}", use_container_width=True):
                        # Remove from session state
                        if 'scenarios' in st.session_state:
                            del st.session_state.scenarios[scenario_key]
                        st.rerun()
        
        # Export scenarios
        st.markdown("---")
        st.markdown("#### 📤 Export Scenarios")
        
        if st.button("Export All Scenarios to Excel", use_container_width=True):
            # Create export data
            export_data = []
            for scenario_key, scenario in saved_scenarios.items():
                if 'data' in scenario and 'summary' in scenario['data']:
                    summary = scenario['data']['summary']
                    export_data.append({
                        'Scenario Name': scenario['name'],
                        'Description': scenario.get('description', ''),
                        'Total Base': summary.get('total_base', 0),
                        'Total Scenario': summary.get('total_scenario', 0),
                        'Total Change': summary.get('total_change', 0),
                        'Total Change %': summary.get('total_change_pct', 0),
                        'Created': scenario.get('created_at', '')
                    })
            
            if export_data:
                export_df = pd.DataFrame(export_data)
                
                # Create download button
                csv = export_df.to_csv(index=False)
                st.download_button(
                    label="Download Scenarios (CSV)",
                    data=csv,
                    file_name=f"scenarios_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        return results


# Streamlit integration function
def display_scenario_section(dataframes: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    Main function to display scenario analysis interface
    """
    if not dataframes:
        st.warning("Please upload and analyze data first.")
        return {}
    
    analyzer = ScenarioAnalyzer()
    scenario_results = analyzer.display_scenario_interface(dataframes)
    
    return scenario_results