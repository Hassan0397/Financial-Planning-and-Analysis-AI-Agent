"""
Filters Module
Interactive filtering system with global filter application
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, date
import plotly.express as px
import plotly.graph_objects as go


class FilterManager:
    """
    Manages interactive filters for FP&A data with global application
    """
    
    def __init__(self):
        self.active_filters = {}
        self.filter_history = []
        self.filter_presets = {}
        self.global_filter_state = {}
    
    def display_filter_interface(self, dataframes: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Main Streamlit interface for filter management
        """
        st.subheader("⚙️ Interactive Data Filters")
        
        if not dataframes:
            st.warning("⚠️ No data uploaded. Please upload data first.")
            return {}
        
        # Initialize session state for filters
        if 'filter_state' not in st.session_state:
            st.session_state.filter_state = {}
        
        # Initialize filter_history in session state if not exists
        if 'filter_history' not in st.session_state:
            st.session_state.filter_history = []
        
        # Initialize filtered_data in session state if not exists
        if 'filtered_data' not in st.session_state:
            st.session_state.filtered_data = dataframes.copy()
        
        # File selection for filter setup
        selected_file = st.selectbox(
            "Select primary file for filters:",
            options=list(dataframes.keys()),
            help="Filters will be detected from this file and can be applied globally"
        )
        
        if selected_file:
            df = dataframes[selected_file]
            
            # Two-column layout
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Detect available filters from data
                available_filters = self._detect_available_filters(df, selected_file)
                
                # Display filter creation interface
                created_filters = self._create_filters_interface(available_filters, df)
                
                # Apply filters button
                if created_filters:
                    if st.button("🚀 Apply All Filters", type="primary", use_container_width=True):
                        filtered_data = self._apply_filters_to_all(dataframes, created_filters)
                        st.session_state.filter_state = created_filters
                        st.session_state.filtered_data = filtered_data
                        st.success(f"✅ Applied {len(created_filters)} filters to all data")
                        return filtered_data
            
            with col2:
                # Current filter status
                self._display_filter_status(created_filters if 'created_filters' in locals() else {})
                
                # Quick actions
                st.markdown("### ⚡ Quick Actions")
                
                if st.button("🧹 Clear All Filters", use_container_width=True):
                    st.session_state.filter_state = {}
                    st.session_state.filtered_data = dataframes.copy()
                    st.rerun()
                
                if st.button("💾 Save Filter Preset", use_container_width=True):
                    self._save_filter_preset(created_filters)
                
                # Load saved presets
                if self.filter_presets:
                    preset_name = st.selectbox(
                        "Load saved preset:",
                        options=list(self.filter_presets.keys())
                    )
                    if st.button("📂 Load Preset", use_container_width=True):
                        st.session_state.filter_state = self.filter_presets[preset_name]
                        st.rerun()
        
        # If filters are already applied, show filtered data
        if 'filtered_data' in st.session_state:
            return st.session_state.filtered_data
        
        return dataframes
    
    def _detect_available_filters(self, df: pd.DataFrame, filename: str) -> Dict[str, Any]:
        """
        Detect filterable columns and their properties
        """
        available_filters = {
            'categorical': {},
            'numerical': {},
            'date': {},
            'boolean': {},
            'special': {}
        }
        
        # Categorical filters
        cat_cols = []
        for col in df.columns:
            if df[col].dtype == 'object' or df[col].dtype.name == 'category':
                unique_vals = df[col].dropna().unique()
                if 1 < len(unique_vals) <= 50:  # Reasonable number for dropdown
                    cat_cols.append(col)
                    available_filters['categorical'][col] = {
                        'type': 'categorical',
                        'unique_values': list(unique_vals),
                        'unique_count': len(unique_vals)
                    }
        
        # Numerical filters
        num_cols = df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            if df[col].notna().any():
                available_filters['numerical'][col] = {
                    'type': 'numerical',
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std())
                }
        
        # Date filters
        date_cols = []
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                date_cols.append(col)
                available_filters['date'][col] = {
                    'type': 'date',
                    'min': df[col].min(),
                    'max': df[col].max()
                }
            else:
                # Try to detect date columns by name
                col_lower = col.lower()
                if any(keyword in col_lower for keyword in ['date', 'time', 'month', 'year', 'period']):
                    try:
                        date_series = pd.to_datetime(df[col], errors='coerce')
                        if date_series.notna().any():
                            date_cols.append(col)
                            available_filters['date'][col] = {
                                'type': 'date',
                                'min': date_series.min(),
                                'max': date_series.max()
                            }
                    except:
                        pass
        
        # Boolean filters
        for col in df.columns:
            unique_vals = df[col].dropna().unique()
            if len(unique_vals) == 2 and set(unique_vals) == {0, 1}:
                available_filters['boolean'][col] = {
                    'type': 'boolean',
                    'values': ['0 (False)', '1 (True)', 'Both']
                }
        
        # Special filters for financial data
        financial_keywords = {
            'product': ['product', 'item', 'sku', 'service'],
            'region': ['region', 'country', 'state', 'territory', 'area'],
            'department': ['department', 'division', 'team', 'segment'],
            'scenario': ['scenario', 'version', 'budget', 'actual', 'forecast']
        }
        
        for col in df.columns:
            col_lower = col.lower()
            for category, keywords in financial_keywords.items():
                if any(keyword in col_lower for keyword in keywords):
                    if col not in available_filters['categorical'] and col not in available_filters['special']:
                        unique_vals = df[col].dropna().unique()
                        if len(unique_vals) > 0:
                            available_filters['special'][col] = {
                                'type': 'special',
                                'category': category,
                                'unique_values': list(unique_vals)[:100],  # Limit for display
                                'unique_count': len(unique_vals)
                            }
        
        return available_filters
    
    def _create_filters_interface(self, available_filters: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
        """
        Create interactive filter widgets
        """
        created_filters = {}
        
        st.markdown("### 🎯 Create Filters")
        
        # Filter type selection
        filter_type = st.selectbox(
            "Add filter type:",
            ["Categorical (Dropdown)", "Numerical (Range)", "Date Range", "Text Search", "Boolean", "Special (FP&A)"],
            help="Select the type of filter to create"
        )
        
        # Create filter based on type
        if filter_type == "Categorical (Dropdown)":
            if available_filters['categorical']:
                col = st.selectbox(
                    "Select column:",
                    options=list(available_filters['categorical'].keys())
                )
                
                if col:
                    filter_info = available_filters['categorical'][col]
                    selected_values = st.multiselect(
                        f"Select values for '{col}':",
                        options=filter_info['unique_values'],
                        default=filter_info['unique_values'][:min(3, len(filter_info['unique_values']))],
                        help=f"Showing {filter_info['unique_count']} unique values"
                    )
                    
                    if selected_values:
                        created_filters[col] = {
                            'type': 'categorical',
                            'values': selected_values,
                            'operation': 'include'
                        }
            else:
                st.info("No categorical columns detected for filtering.")
        
        elif filter_type == "Numerical (Range)":
            if available_filters['numerical']:
                col = st.selectbox(
                    "Select numeric column:",
                    options=list(available_filters['numerical'].keys())
                )
                
                if col:
                    filter_info = available_filters['numerical'][col]
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        min_val = st.number_input(
                            f"Minimum {col}:",
                            value=float(filter_info['min']),
                            min_value=float(filter_info['min']),
                            max_value=float(filter_info['max'])
                        )
                    with col2:
                        max_val = st.number_input(
                            f"Maximum {col}:",
                            value=float(filter_info['max']),
                            min_value=float(filter_info['min']),
                            max_value=float(filter_info['max'])
                        )
                    
                    # Add outlier handling option
                    outlier_option = st.checkbox(f"Exclude outliers from {col}")
                    if outlier_option:
                        Q1 = df[col].quantile(0.25)
                        Q3 = df[col].quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        st.info(f"Outlier bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
                    
                    created_filters[col] = {
                        'type': 'numerical',
                        'min': min_val,
                        'max': max_val,
                        'exclude_outliers': outlier_option
                    }
            else:
                st.info("No numerical columns detected for filtering.")
        
        elif filter_type == "Date Range":
            if available_filters['date']:
                col = st.selectbox(
                    "Select date column:",
                    options=list(available_filters['date'].keys())
                )
                
                if col:
                    filter_info = available_filters['date'][col]
                    
                    # Convert to date objects for date_input
                    if isinstance(filter_info['min'], pd.Timestamp):
                        min_date = filter_info['min'].date()
                        max_date = filter_info['max'].date()
                    else:
                        min_date = filter_info['min']
                        max_date = filter_info['max']
                    
                    date_range = st.date_input(
                        f"Select date range for '{col}':",
                        value=[min_date, max_date],
                        min_value=min_date,
                        max_value=max_date
                    )
                    
                    if len(date_range) == 2:
                        created_filters[col] = {
                            'type': 'date',
                            'start_date': date_range[0],
                            'end_date': date_range[1]
                        }
                        
                        # Add time period quick selects
                        st.markdown("**Quick periods:**")
                        quick_cols = st.columns(5)
                        with quick_cols[0]:
                            if st.button("Last 30 days", use_container_width=True):
                                created_filters[col]['start_date'] = max_date - pd.Timedelta(days=30)
                                created_filters[col]['end_date'] = max_date
                                st.rerun()
                        with quick_cols[1]:
                            if st.button("Last Quarter", use_container_width=True):
                                created_filters[col]['start_date'] = max_date - pd.Timedelta(days=90)
                                created_filters[col]['end_date'] = max_date
                                st.rerun()
            else:
                st.info("No date columns detected for filtering.")
        
        elif filter_type == "Text Search":
            # Simple text search filter
            text_cols = [col for col in df.columns if df[col].dtype == 'object']
            if text_cols:
                col = st.selectbox(
                    "Select text column:",
                    options=text_cols
                )
                
                search_term = st.text_input(
                    f"Search in '{col}':",
                    help="Enter text to search for (supports partial matches)"
                )
                
                if search_term:
                    created_filters[col] = {
                        'type': 'text_search',
                        'search_term': search_term,
                        'case_sensitive': st.checkbox("Case sensitive", value=False)
                    }
            else:
                st.info("No text columns detected for filtering.")
        
        elif filter_type == "Boolean":
            if available_filters['boolean']:
                col = st.selectbox(
                    "Select boolean column:",
                    options=list(available_filters['boolean'].keys())
                )
                
                if col:
                    selected_value = st.radio(
                        f"Filter '{col}' by:",
                        options=available_filters['boolean'][col]['values']
                    )
                    
                    created_filters[col] = {
                        'type': 'boolean',
                        'value': selected_value
                    }
            else:
                st.info("No boolean columns detected for filtering.")
        
        elif filter_type == "Special (FP&A)":
            if available_filters['special']:
                # Group by category
                special_categories = {}
                for col, info in available_filters['special'].items():
                    category = info['category']
                    if category not in special_categories:
                        special_categories[category] = []
                    special_categories[category].append((col, info))
                
                # Display by category
                selected_category = st.selectbox(
                    "Select FP&A category:",
                    options=list(special_categories.keys())
                )
                
                if selected_category:
                    cols_in_category = [col for col, _ in special_categories[selected_category]]
                    col = st.selectbox(
                        f"Select {selected_category} column:",
                        options=cols_in_category
                    )
                    
                    if col:
                        filter_info = special_categories[selected_category][cols_in_category.index(col)][1]
                        
                        # Special handling based on category
                        if selected_category == 'scenario':
                            scenario_options = filter_info['unique_values']
                            selected_scenarios = st.multiselect(
                                f"Select {selected_category}(s):",
                                options=scenario_options,
                                default=['Actual'] if 'Actual' in scenario_options else scenario_options[:1]
                            )
                            
                            if selected_scenarios:
                                created_filters[col] = {
                                    'type': 'special',
                                    'category': selected_category,
                                    'values': selected_scenarios
                                }
                        
                        else:  # product, region, department
                            selected_values = st.multiselect(
                                f"Select {selected_category}(s):",
                                options=filter_info['unique_values'],
                                default=filter_info['unique_values'][:min(5, len(filter_info['unique_values']))]
                            )
                            
                            if selected_values:
                                created_filters[col] = {
                                    'type': 'special',
                                    'category': selected_category,
                                    'values': selected_values
                                }
                                
                                # Add "Select All" / "Select None" buttons
                                col1, col2 = st.columns(2)
                                with col1:
                                    if st.button(f"Select All {selected_category}s", use_container_width=True):
                                        created_filters[col]['values'] = filter_info['unique_values']
                                        st.rerun()
            else:
                st.info("No special FP&A columns detected for filtering.")
        
        # Display current filters being created
        if created_filters:
            st.markdown("---")
            st.markdown("### 🔧 Current Filter Set")
            
            for i, (col, filter_config) in enumerate(created_filters.items()):
                with st.expander(f"Filter {i+1}: {col} ({filter_config['type']})", expanded=True):
                    self._display_filter_config(col, filter_config, df)
                    
                    # Option to remove this filter
                    if st.button(f"❌ Remove this filter", key=f"remove_{col}"):
                        del created_filters[col]
                        st.rerun()
        
        return created_filters
    
    def _display_filter_config(self, col: str, filter_config: Dict[str, Any], df: pd.DataFrame):
        """Display detailed filter configuration"""
        filter_type = filter_config['type']
        
        if filter_type == 'categorical':
            st.write(f"**Type:** Categorical (Dropdown)")
            st.write(f"**Operation:** Include selected values")
            st.write(f"**Selected values:** {', '.join(map(str, filter_config['values']))}")
            st.write(f"**Coverage:** {len(filter_config['values'])} / {df[col].nunique()} unique values")
        
        elif filter_type == 'numerical':
            st.write(f"**Type:** Numerical Range")
            st.write(f"**Range:** [{filter_config['min']}, {filter_config['max']}]")
            if filter_config.get('exclude_outliers', False):
                st.write("**Outlier exclusion:** Enabled")
        
        elif filter_type == 'date':
            st.write(f"**Type:** Date Range")
            st.write(f"**From:** {filter_config['start_date']}")
            st.write(f"**To:** {filter_config['end_date']}")
        
        elif filter_type == 'text_search':
            st.write(f"**Type:** Text Search")
            st.write(f"**Search term:** '{filter_config['search_term']}'")
            st.write(f"**Case sensitive:** {filter_config['case_sensitive']}")
        
        elif filter_type == 'boolean':
            st.write(f"**Type:** Boolean")
            st.write(f"**Value:** {filter_config['value']}")
        
        elif filter_type == 'special':
            st.write(f"**Type:** Special (FP&A)")
            st.write(f"**Category:** {filter_config['category']}")
            st.write(f"**Selected {filter_config['category']}s:** {', '.join(map(str, filter_config['values']))}")
    
    def _apply_filters_to_all(self, dataframes: Dict[str, pd.DataFrame], 
                             filters: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
        """
        Apply filters to all uploaded dataframes
        """
        filtered_dataframes = {}
        
        # Ensure filter_history exists in session state
        if 'filter_history' not in st.session_state:
            st.session_state.filter_history = []
        
        for filename, df in dataframes.items():
            filtered_df = df.copy()
            
            # Apply each filter
            for col, filter_config in filters.items():
                if col in filtered_df.columns:
                    filtered_df = self._apply_single_filter(filtered_df, col, filter_config)
            
            filtered_dataframes[filename] = filtered_df
            
            # Log the filtering effect
            original_rows = len(df)
            filtered_rows = len(filtered_df)
            reduction = original_rows - filtered_rows
            reduction_pct = (reduction / original_rows * 100) if original_rows > 0 else 0
            
            filter_record = {
                'timestamp': datetime.now().isoformat(),
                'filename': filename,
                'original_rows': original_rows,
                'filtered_rows': filtered_rows,
                'reduction': reduction,
                'reduction_pct': reduction_pct,
                'filters_applied': list(filters.keys())
            }
            
            # Append to session state filter_history
            st.session_state.filter_history.append(filter_record)
        
        return filtered_dataframes
    
    def _apply_single_filter(self, df: pd.DataFrame, col: str, 
                           filter_config: Dict[str, Any]) -> pd.DataFrame:
        """Apply a single filter to a dataframe"""
        filter_type = filter_config['type']
        
        try:
            if filter_type == 'categorical':
                selected_values = filter_config['values']
                return df[df[col].isin(selected_values)]
            
            elif filter_type == 'numerical':
                min_val = filter_config['min']
                max_val = filter_config['max']
                
                filtered = df[(df[col] >= min_val) & (df[col] <= max_val)]
                
                # Exclude outliers if requested
                if filter_config.get('exclude_outliers', False):
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    filtered = filtered[(filtered[col] >= lower_bound) & (filtered[col] <= upper_bound)]
                
                return filtered
            
            elif filter_type == 'date':
                start_date = pd.to_datetime(filter_config['start_date'])
                end_date = pd.to_datetime(filter_config['end_date'])
                
                # Ensure column is datetime
                if not pd.api.types.is_datetime64_any_dtype(df[col]):
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                
                return df[(df[col] >= start_date) & (df[col] <= end_date)]
            
            elif filter_type == 'text_search':
                search_term = filter_config['search_term']
                case_sensitive = filter_config['case_sensitive']
                
                if case_sensitive:
                    return df[df[col].astype(str).str.contains(search_term, na=False)]
                else:
                    return df[df[col].astype(str).str.contains(search_term, case=False, na=False)]
            
            elif filter_type == 'boolean':
                value_str = filter_config['value']
                if value_str == '0 (False)':
                    return df[df[col] == 0]
                elif value_str == '1 (True)':
                    return df[df[col] == 1]
                else:  # 'Both'
                    return df
            
            elif filter_type == 'special':
                selected_values = filter_config['values']
                return df[df[col].isin(selected_values)]
        
        except Exception as e:
            st.error(f"Error applying filter on '{col}': {str(e)}")
            return df
        
        return df
    
    def _display_filter_status(self, filters: Dict[str, Any]):
        """Display current filter status"""
        st.markdown("### 📊 Filter Status")
        
        if not filters:
            st.info("No active filters")
            return
        
        st.success(f"**{len(filters)}** active filter(s)")
        
        # Show quick stats
        filter_types = {}
        for filter_config in filters.values():
            filter_type = filter_config['type']
            filter_types[filter_type] = filter_types.get(filter_type, 0) + 1
        
        for filter_type, count in filter_types.items():
            st.write(f"• {count} {filter_type} filter(s)")
        
        # Show impact if filtered data exists
        if 'filtered_data' in st.session_state and 'uploaded_data' in st.session_state:
            total_original = sum(len(df) for df in st.session_state.uploaded_data.values())
            total_filtered = sum(len(df) for df in st.session_state.filtered_data.values())
            
            if total_original > 0:
                reduction = total_original - total_filtered
                reduction_pct = (reduction / total_original * 100)
                
                st.metric(
                    "Data Reduction",
                    f"{reduction:,} rows",
                    f"{reduction_pct:.1f}%"
                )
        
        # Visual filter representation
        if filters:
            st.markdown("---")
            st.markdown("### 🎨 Filter Visualization")
            
            # Create a simple visualization of filter combinations
            fig = go.Figure()
            
            # Add a representation of filter coverage
            categories = list(filters.keys())
            values = [1] * len(categories)  # Placeholder values
            
            fig.add_trace(go.Bar(
                x=categories,
                y=values,
                marker_color='lightblue',
                text=[filters[cat]['type'] for cat in categories],
                textposition='auto',
            ))
            
            fig.update_layout(
                title="Active Filters Overview",
                xaxis_title="Filter Columns",
                yaxis_title="Filter Count",
                showlegend=False,
                height=200,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def _save_filter_preset(self, filters: Dict[str, Any]):
        """Save current filter set as a preset"""
        preset_name = st.text_input("Preset name:", key="preset_name")
        
        if preset_name and filters:
            self.filter_presets[preset_name] = filters.copy()
            st.success(f"✅ Preset '{preset_name}' saved with {len(filters)} filters")
            
            # Store in session state
            if 'filter_presets' not in st.session_state:
                st.session_state.filter_presets = {}
            st.session_state.filter_presets[preset_name] = filters.copy()
    
    def get_filtered_data_summary(self, original_data: Dict[str, pd.DataFrame], 
                                 filtered_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Generate summary of filtering impact"""
        summary = {
            'total_files': len(original_data),
            'original_total_rows': 0,
            'filtered_total_rows': 0,
            'file_summaries': {},
            'overall_reduction_pct': 0
        }
        
        for filename in original_data.keys():
            orig_rows = len(original_data[filename])
            filt_rows = len(filtered_data.get(filename, pd.DataFrame()))
            
            summary['original_total_rows'] += orig_rows
            summary['filtered_total_rows'] += filt_rows
            
            reduction = orig_rows - filt_rows
            reduction_pct = (reduction / orig_rows * 100) if orig_rows > 0 else 0
            
            summary['file_summaries'][filename] = {
                'original_rows': orig_rows,
                'filtered_rows': filt_rows,
                'reduction': reduction,
                'reduction_pct': reduction_pct
            }
        
        if summary['original_total_rows'] > 0:
            total_reduction = summary['original_total_rows'] - summary['filtered_total_rows']
            summary['overall_reduction_pct'] = (total_reduction / summary['original_total_rows'] * 100)
        
        return summary
    
    def create_filter_export(self) -> Dict[str, Any]:
        """Create exportable filter configuration"""
        # Use session state filter_history if available, otherwise use class instance
        filter_history = st.session_state.get('filter_history', self.filter_history)
        
        export_config = {
            'timestamp': datetime.now().isoformat(),
            'active_filters': self.active_filters.copy(),
            'filter_presets': self.filter_presets.copy(),
            'filter_history': filter_history[-10:] if filter_history else []  # Last 10 entries
        }
        return export_config


# Streamlit integration function
def display_filter_section(dataframes: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Main function to display filter interface
    """
    if not dataframes:
        st.warning("Please upload data first in the Data Upload section.")
        return {}
    
    filter_manager = FilterManager()
    filtered_dataframes = filter_manager.display_filter_interface(dataframes)
    
    # Display impact summary if filters were applied
    if 'filtered_data' in st.session_state and 'uploaded_data' in st.session_state:
        st.markdown("---")
        st.markdown("### 📈 Filter Impact Analysis")
        
        summary = filter_manager.get_filtered_data_summary(
            st.session_state.uploaded_data,
            st.session_state.filtered_data
        )
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Original Data",
                f"{summary['original_total_rows']:,} rows",
                delta=None
            )
        with col2:
            st.metric(
                "Filtered Data",
                f"{summary['filtered_total_rows']:,} rows",
                delta=f"-{summary['original_total_rows'] - summary['filtered_total_rows']:,}"
            )
        with col3:
            st.metric(
                "Reduction",
                f"{summary['overall_reduction_pct']:.1f}%",
                delta_color="inverse"
            )
        
        # Show file-by-file impact
        with st.expander("📋 Detailed Impact by File"):
            for filename, file_summary in summary['file_summaries'].items():
                col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                with col1:
                    st.write(f"**{filename}**")
                with col2:
                    st.write(f"Original: {file_summary['original_rows']:,}")
                with col3:
                    st.write(f"Filtered: {file_summary['filtered_rows']:,}")
                with col4:
                    st.write(f"Reduction: {file_summary['reduction_pct']:.1f}%")
        
        # Export filter configuration
        if st.button("📤 Export Filter Configuration", use_container_width=True):
            export_config = filter_manager.create_filter_export()
            st.download_button(
                label="Download Filter Config (JSON)",
                data=pd.DataFrame([export_config]).to_json(orient='records', indent=2),
                file_name=f"filter_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
    
    return filtered_dataframes