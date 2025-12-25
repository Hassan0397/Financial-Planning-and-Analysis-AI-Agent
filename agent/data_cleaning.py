"""
Data Cleaning Module
Handles data preprocessing, cleaning, and transformation for FP&A analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import re
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class DataCleaner:
    """
    Comprehensive data cleaning and preprocessing for financial data
    """
    
    def __init__(self):
        self.cleaning_log = []
        self.transformations_applied = {}
        self.column_mapping = {}
        self.data_quality_report = {}
    
    def display_cleaning_interface(self, dataframes: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Main Streamlit interface for data cleaning
        """
        st.subheader("🧹 Data Cleaning & Preprocessing")
        
        # Display comprehensive instructions first
        self._display_instructions()
        
        if not dataframes:
            st.warning("⚠️ No data uploaded. Please upload data first.")
            return {}
        
        # File selection
        selected_file = st.selectbox(
            "Select file to clean:",
            options=list(dataframes.keys()),
            help="Choose which file to clean and preprocess"
        )
        
        if selected_file:
            df = dataframes[selected_file].copy()
            
            # Display current state
            with st.expander("🔍 Current Data Overview", expanded=True):
                self._display_data_overview(df, selected_file)
            
            # Initialize session state for tracking changes
            if 'original_data' not in st.session_state:
                st.session_state.original_data = df.copy()
            
            # Cleaning options in tabs
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "🔧 Basic Cleaning", 
                "📊 Column Management", 
                "📅 Date Handling", 
                "💰 Financial Specific", 
                "⚙️ Advanced"
            ])
            
            cleaned_df = df.copy()
            
            with tab1:
                cleaned_df = self._basic_cleaning_tab(cleaned_df, selected_file)
            
            with tab2:
                cleaned_df = self._column_management_tab(cleaned_df, selected_file)
            
            with tab3:
                cleaned_df = self._date_handling_tab(cleaned_df, selected_file)
            
            with tab4:
                cleaned_df = self._financial_specific_tab(cleaned_df, selected_file)
            
            with tab5:
                cleaned_df = self._advanced_cleaning_tab(cleaned_df, selected_file)
            
            # Compare and apply changes
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("✅ Apply Changes", type="primary", use_container_width=True, 
                           help="Apply all cleaning operations to the selected file"):
                    dataframes[selected_file] = cleaned_df
                    self._log_cleaning_operations(selected_file)
                    st.success(f"✅ Cleaning applied to {selected_file}")
                    st.rerun()
            
            with col2:
                if st.button("🔄 Reset Changes", use_container_width=True, 
                           help="Reset all changes made in this session"):
                    if 'original_data' in st.session_state:
                        dataframes[selected_file] = st.session_state.original_data.copy()
                    st.rerun()
            
            with col3:
                if st.button("📋 Apply to All Files", use_container_width=True,
                           help="Apply similar cleaning patterns to all uploaded files"):
                    cleaned_dataframes = self._apply_to_all_files(dataframes, selected_file)
                    for filename, df_clean in cleaned_dataframes.items():
                        dataframes[filename] = df_clean
                    st.success(f"✅ Cleaning patterns applied to all {len(cleaned_dataframes)} files")
            
            # Show comparison
            if not df.equals(cleaned_df):
                with st.expander("📋 Changes Summary", expanded=True):
                    self._show_changes_summary(df, cleaned_df)
            
            # Preview cleaned data
            with st.expander("👁️ Preview Cleaned Data"):
                st.dataframe(cleaned_df.head(50), use_container_width=True)
                st.caption(f"Showing first 50 rows of {len(cleaned_df):,} total rows")
            
            return dataframes
        
        return dataframes
    
    def _display_instructions(self):
        """Display comprehensive cleaning instructions"""
        with st.expander("📚 How to Clean Your Data - Step by Step Guide", expanded=True):
            st.markdown("""
            ### **Beginner-Friendly Data Cleaning Guide**
            
            **🎯 Goal:** Transform raw data into clean, analysis-ready format
            
            **🔧 Step 1: Basic Cleaning (Start Here)**
            - **Missing Values**: Choose how to handle empty cells
              - *Remove rows*: Delete entire rows with missing data
              - *Fill with mean/median*: Replace missing numbers with average values
              - *Fill with mode*: Replace missing text with most common value
              - *Forward/Backward fill*: Copy values from previous/next rows
            
            - **Duplicates**: Remove identical rows to avoid double-counting
            - **Text Cleaning**: Trim extra spaces and fix inconsistent capitalization
            
            **📊 Step 2: Column Management**
            - **Rename Columns**: Make column names clear and consistent
              - Example: Change 'cust_id' to 'customer_id'
            
            - **Select Columns**: Keep only columns you need for analysis
            - **Convert Data Types**: Ensure correct formats
              - *Numeric*: For numbers and calculations
              - *Text*: For names, descriptions
              - *Date*: For time-based analysis
              - *Category*: For limited options (like product types)
            
            **📅 Step 3: Date Handling**
            - **Convert to Dates**: Transform text dates (like '2024-01-15') to proper date format
            - **Extract Parts**: Create new columns for year, month, quarter from dates
            - **Filter Dates**: Select specific date ranges for analysis
            
            **💰 Step 4: Financial Data Specific**
            - **Currency Columns**: Automatically detects money-related columns
            - **Clean Currency**: Remove $, € symbols and convert to numbers
            - **Handle Negative Values**: Decide how to treat negative numbers
            - **Zero Values**: Replace or adjust zero financial values
            
            **⚙️ Step 5: Advanced Cleaning**
            - **Outlier Detection**: Find unusual values that might be errors
            - **Data Normalization**: Scale numbers for better comparison
            - **Advanced Transformations**: Complex cleaning operations
            
            **💡 Pro Tips:**
            1. **Start with Basic Cleaning** - Fix missing values and duplicates first
            2. **Check Data Types** - Ensure numbers are numeric, dates are date format
            3. **Preview Changes** - Always check the Changes Summary before applying
            4. **Save Original** - Use 'Reset Changes' if you make a mistake
            5. **Apply to All Files** - Use this button when working with multiple similar files
            
            **⚠️ Important Notes:**
            - Changes are NOT saved until you click **"✅ Apply Changes"**
            - Click **"🔄 Reset Changes"** to undo everything in current session
            - **"📋 Apply to All Files"** copies cleaning patterns to other files
            """)
    
    def _display_data_overview(self, df: pd.DataFrame, filename: str):
        """
        Display comprehensive data overview
        """
        st.markdown(f"**File:** `{filename}`")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Rows", f"{df.shape[0]:,}", 
                     help="Number of data records")
        with col2:
            st.metric("Total Columns", df.shape[1],
                     help="Number of data fields")
        with col3:
            null_count = df.isnull().sum().sum()
            null_percent = (null_count / (df.shape[0] * df.shape[1]) * 100).round(2)
            st.metric("Null Values", f"{null_count:,}", f"{null_percent}%",
                     help="Empty or missing cells")
        with col4:
            duplicate_count = df.duplicated().sum()
            st.metric("Duplicates", duplicate_count,
                     help="Identical rows")
        
        # Data quality score
        quality_score = max(0, 100 - null_percent - (duplicate_count/len(df)*100))
        st.progress(quality_score/100, text=f"Data Quality Score: {quality_score:.1f}/100")
        
        # Data types summary
        st.markdown("**📊 Data Types Summary:**")
        dtype_counts = df.dtypes.value_counts()
        for dtype, count in dtype_counts.items():
            st.caption(f"• {dtype}: {count} columns")
        
        # Quick stats
        with st.expander("📈 Quick Statistics", expanded=False):
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                stats_df = df[numeric_cols].describe().T[['mean', 'min', 'max', 'std']]
                stats_df.columns = ['Average', 'Minimum', 'Maximum', 'Std Dev']
                st.dataframe(stats_df.style.format("{:,.2f}"), use_container_width=True)
            else:
                st.info("No numeric columns found for statistics")
    
    def _basic_cleaning_tab(self, df: pd.DataFrame, filename: str) -> pd.DataFrame:
        """
        Basic cleaning operations
        """
        st.markdown("### 🔧 Basic Data Cleaning")
        st.info("Start here! Fix missing values, duplicates, and text formatting.")
        
        cleaned_df = df.copy()
        
        # 1. Handle missing values
        st.markdown("#### 1. Handle Missing Values")
        st.caption("Empty cells can cause errors in analysis. Choose how to handle them.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            missing_action = st.selectbox(
                "Action for missing values:",
                ["Select action", "Remove rows", "Fill with mean", "Fill with median", 
                 "Fill with mode", "Fill with zero", "Forward fill", "Backward fill"],
                key="missing_action",
                help="Choose how to handle empty cells"
            )
        
        with col2:
            if missing_action != "Select action":
                st.write("**Affected Columns:**")
                null_series = cleaned_df.isnull().sum()
                null_cols = null_series[null_series > 0]
                if len(null_cols) > 0:
                    for col, count in null_cols.items():
                        st.caption(f"• {col}: {count} missing values")
                else:
                    st.success("✓ No missing values found")
        
        if missing_action != "Select action":
            numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
            categorical_cols = cleaned_df.select_dtypes(exclude=[np.number]).columns
            
            if missing_action == "Remove rows":
                initial_rows = len(cleaned_df)
                cleaned_df = cleaned_df.dropna()
                removed = initial_rows - len(cleaned_df)
                if removed > 0:
                    st.success(f"✅ Removed {removed} rows with missing values")
                    self.transformations_applied['missing_values_removed'] = removed
                else:
                    st.info("No rows were removed")
            
            elif missing_action == "Fill with mean" and len(numeric_cols) > 0:
                for col in numeric_cols:
                    if cleaned_df[col].isnull().any():
                        mean_val = cleaned_df[col].mean()
                        filled_count = cleaned_df[col].isnull().sum()
                        cleaned_df[col] = cleaned_df[col].fillna(mean_val)
                        st.success(f"✅ Filled {filled_count} missing values in '{col}' with mean: {mean_val:.2f}")
            
            elif missing_action == "Fill with median" and len(numeric_cols) > 0:
                for col in numeric_cols:
                    if cleaned_df[col].isnull().any():
                        median_val = cleaned_df[col].median()
                        filled_count = cleaned_df[col].isnull().sum()
                        cleaned_df[col] = cleaned_df[col].fillna(median_val)
                        st.success(f"✅ Filled {filled_count} missing values in '{col}' with median: {median_val:.2f}")
            
            elif missing_action == "Fill with mode":
                for col in cleaned_df.columns:
                    if cleaned_df[col].isnull().any():
                        mode_vals = cleaned_df[col].mode()
                        if len(mode_vals) > 0:
                            mode_val = mode_vals[0]
                            filled_count = cleaned_df[col].isnull().sum()
                            cleaned_df[col] = cleaned_df[col].fillna(mode_val)
                            st.success(f"✅ Filled {filled_count} missing values in '{col}' with mode: {mode_val}")
            
            elif missing_action == "Fill with zero":
                filled_count = cleaned_df.isnull().sum().sum()
                cleaned_df = cleaned_df.fillna(0)
                st.success(f"✅ Filled {filled_count} missing values with 0")
            
            elif missing_action == "Forward fill":
                cleaned_df = cleaned_df.ffill()
                st.success("✅ Forward fill applied - missing values filled from previous row")
            
            elif missing_action == "Backward fill":
                cleaned_df = cleaned_df.bfill()
                st.success("✅ Backward fill applied - missing values filled from next row")
        
        # 2. Remove duplicates
        st.markdown("#### 2. Remove Duplicates")
        st.caption("Identical rows can skew your analysis results.")
        
        if st.checkbox("Remove duplicate rows", key="remove_duplicates",
                      help="Check to remove completely identical rows"):
            initial_rows = len(cleaned_df)
            cleaned_df = cleaned_df.drop_duplicates()
            removed = initial_rows - len(cleaned_df)
            if removed > 0:
                st.success(f"✅ Removed {removed} duplicate rows")
                self.transformations_applied['duplicates_removed'] = removed
            else:
                st.info("No duplicates found")
        
        # 3. Text cleaning
        st.markdown("#### 3. Text Cleaning")
        st.caption("Fix common text formatting issues.")
        
        text_cols = cleaned_df.select_dtypes(include=['object']).columns
        
        if len(text_cols) > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                if st.checkbox("Trim whitespace", 
                             help="Remove extra spaces from beginning and end of text"):
                    for col in text_cols:
                        cleaned_df[col] = cleaned_df[col].astype(str).str.strip()
                    st.success("✅ Trimmed whitespace from text columns")
            
            with col2:
                if st.checkbox("Fix inconsistent casing",
                             help="Standardize text capitalization"):
                    for col in text_cols:
                        # Sample to decide casing
                        sample_values = cleaned_df[col].dropna().head(20).astype(str)
                        if len(sample_values) > 0:
                            sample_text = ' '.join(sample_values.str.lower())
                            # Check for business names
                            business_indicators = ['inc', 'corp', 'ltd', 'llc', 'co', 'group']
                            if any(indicator in sample_text for indicator in business_indicators):
                                cleaned_df[col] = cleaned_df[col].astype(str).str.title()
                            else:
                                # Default to proper case
                                cleaned_df[col] = cleaned_df[col].astype(str).str.title()
                    st.success("✅ Standardized text casing")
        else:
            st.info("No text columns found for cleaning")
        
        return cleaned_df
    
    def _column_management_tab(self, df: pd.DataFrame, filename: str) -> pd.DataFrame:
        """
        Column renaming, dropping, and type conversion
        """
        st.markdown("### 📊 Column Management")
        st.info("Organize your columns: rename, select, and convert data types.")
        
        cleaned_df = df.copy()
        
        # 1. Column renaming
        st.markdown("#### 1. Rename Columns")
        st.caption("Make column names clear and descriptive. Example: 'cust_id' → 'customer_id'")
        
        if st.checkbox("Rename columns", key="rename_cols"):
            columns = cleaned_df.columns.tolist()
            rename_mapping = {}
            
            st.markdown("**Current Name → New Name**")
            for col in columns:
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.code(col)
                with col2:
                    new_name = st.text_input(
                        f"New name for: {col}",
                        value=col,
                        key=f"rename_{col}",
                        label_visibility="collapsed"
                    )
                    if new_name != col and new_name.strip() != "":
                        rename_mapping[col] = new_name.strip()
            
            if rename_mapping:
                if st.button("Apply Renaming", key="apply_renaming"):
                    cleaned_df = cleaned_df.rename(columns=rename_mapping)
                    self.column_mapping.update(rename_mapping)
                    st.success(f"✅ Renamed {len(rename_mapping)} columns")
        
        # 2. Column selection
        st.markdown("#### 2. Select Columns to Keep")
        st.caption("Focus your analysis by keeping only relevant columns.")
        
        columns_to_keep = st.multiselect(
            "Select columns to include:",
            options=cleaned_df.columns.tolist(),
            default=cleaned_df.columns.tolist(),
            help="Deselect columns you don't need for analysis"
        )
        
        if len(columns_to_keep) < len(cleaned_df.columns):
            columns_to_drop = [col for col in cleaned_df.columns if col not in columns_to_keep]
            cleaned_df = cleaned_df[columns_to_keep]
            st.success(f"✅ Kept {len(columns_to_keep)} columns, dropped {len(columns_to_drop)}")
            st.caption(f"Dropped: {', '.join(columns_to_drop[:5])}{'...' if len(columns_to_drop) > 5 else ''}")
        
        # 3. Data type conversion
        st.markdown("#### 3. Convert Data Types")
        st.caption("Ensure each column has the correct data type for accurate analysis.")
        
        if st.checkbox("Convert data types", key="type_conversion"):
            st.markdown("**Column → Current Type → New Type**")
            
            for col in cleaned_df.columns:
                col1, col2, col3 = st.columns([2, 2, 1])
                with col1:
                    st.write(f"**{col}**")
                with col2:
                    current_dtype = str(cleaned_df[col].dtype)
                    st.caption(f"Current: {current_dtype}")
                    
                    new_dtype = st.selectbox(
                        "Convert to:",
                        ["Keep as is", "Numeric", "Text", "Date", "Category", "Boolean"],
                        index=0,
                        key=f"dtype_{col}",
                        label_visibility="collapsed"
                    )
                with col3:
                    if new_dtype != "Keep as is":
                        try:
                            if new_dtype == "Numeric":
                                # Try to clean numeric data first
                                temp_series = cleaned_df[col].replace(['$', ',', '%', '€', '£'], '', regex=True)
                                cleaned_df[col] = pd.to_numeric(temp_series, errors='coerce')
                                converted = cleaned_df[col].notna().sum()
                                st.success(f"✓ {converted}")
                                
                            elif new_dtype == "Text":
                                cleaned_df[col] = cleaned_df[col].astype(str)
                                st.success("✓")
                                
                            elif new_dtype == "Date":
                                cleaned_df[col] = pd.to_datetime(cleaned_df[col], errors='coerce')
                                converted = cleaned_df[col].notna().sum()
                                st.success(f"✓ {converted}")
                                
                            elif new_dtype == "Category":
                                cleaned_df[col] = cleaned_df[col].astype('category')
                                st.success("✓")
                                
                            elif new_dtype == "Boolean":
                                cleaned_df[col] = cleaned_df[col].astype(bool)
                                st.success("✓")
                                
                        except Exception as e:
                            st.error(f"✗ Error: {str(e)[:50]}")
        
        return cleaned_df
    
    def _date_handling_tab(self, df: pd.DataFrame, filename: str) -> pd.DataFrame:
        """
        Specialized date handling for financial data
        """
        st.markdown("### 📅 Date & Time Handling")
        st.info("Proper date handling is crucial for time-based financial analysis.")
        
        cleaned_df = df.copy()
        
        # Detect date columns automatically
        date_candidates = []
        for col in cleaned_df.columns:
            col_lower = col.lower()
            date_keywords = ['date', 'time', 'month', 'year', 'quarter', 'period', 
                            'day', 'week', 'created', 'modified', 'timestamp']
            if any(keyword in col_lower for keyword in date_keywords):
                date_candidates.append(col)
        
        # Also check data types
        for col in cleaned_df.columns:
            if pd.api.types.is_datetime64_any_dtype(cleaned_df[col]):
                if col not in date_candidates:
                    date_candidates.append(col)
        
        if date_candidates:
            st.markdown(f"#### 📅 Detected Date Columns ({len(date_candidates)})")
            st.caption(f"Found: {', '.join(date_candidates[:5])}{'...' if len(date_candidates) > 5 else ''}")
            
            for date_col in date_candidates:
                with st.expander(f"Process: {date_col}", expanded=False):
                    # Show sample values
                    sample_values = cleaned_df[date_col].dropna().head(3)
                    if len(sample_values) > 0:
                        st.caption(f"Sample: {', '.join(map(str, sample_values.tolist()))}")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Date conversion
                        if st.button(f"Convert to Date", key=f"convert_{date_col}"):
                            try:
                                original_non_null = cleaned_df[date_col].notna().sum()
                                cleaned_df[date_col] = pd.to_datetime(cleaned_df[date_col], errors='coerce')
                                converted_count = cleaned_df[date_col].notna().sum()
                                
                                if converted_count > 0:
                                    st.success(f"✅ Converted {converted_count}/{original_non_null} values")
                                    st.caption(f"Date range: {cleaned_df[date_col].min().date()} to {cleaned_df[date_col].max().date()}")
                                else:
                                    st.error("❌ Conversion failed for all values")
                            except Exception as e:
                                st.error(f"❌ Error: {str(e)}")
                    
                    with col2:
                        # Extract date parts if already datetime
                        if pd.api.types.is_datetime64_any_dtype(cleaned_df[date_col]):
                            if st.button(f"Extract Date Parts", key=f"extract_{date_col}"):
                                try:
                                    cleaned_df[f"{date_col}_year"] = cleaned_df[date_col].dt.year
                                    cleaned_df[f"{date_col}_quarter"] = cleaned_df[date_col].dt.quarter
                                    cleaned_df[f"{date_col}_month"] = cleaned_df[date_col].dt.month
                                    cleaned_df[f"{date_col}_month_name"] = cleaned_df[date_col].dt.month_name()
                                    cleaned_df[f"{date_col}_day"] = cleaned_df[date_col].dt.day
                                    cleaned_df[f"{date_col}_weekday"] = cleaned_df[date_col].dt.day_name()
                                    cleaned_df[f"{date_col}_week"] = cleaned_df[date_col].dt.isocalendar().week
                                    
                                    st.success("✅ Added date part columns:")
                                    st.caption("Year, Quarter, Month, Month Name, Day, Weekday, Week")
                                except Exception as e:
                                    st.error(f"❌ Error: {str(e)}")
                    
                    # Date filtering
                    if pd.api.types.is_datetime64_any_dtype(cleaned_df[date_col]):
                        st.markdown("**Filter by Date Range:**")
                        min_date = cleaned_df[date_col].min().date()
                        max_date = cleaned_df[date_col].max().date()
                        
                        date_range = st.date_input(
                            f"Select range for {date_col}",
                            value=[min_date, max_date],
                            min_value=min_date,
                            max_value=max_date,
                            key=f"range_{date_col}"
                        )
                        
                        if len(date_range) == 2:
                            mask = (cleaned_df[date_col].dt.date >= date_range[0]) & \
                                   (cleaned_df[date_col].dt.date <= date_range[1])
                            filtered_count = mask.sum()
                            total_count = len(cleaned_df)
                            
                            if filtered_count < total_count:
                                if st.button(f"Apply Filter", key=f"filter_{date_col}"):
                                    cleaned_df = cleaned_df[mask]
                                    st.success(f"✅ Filtered to {filtered_count:,} rows (from {total_count:,})")
        
        else:
            st.info("ℹ️ No date-like columns detected automatically. You can manually convert text columns to dates in the Column Management tab.")
        
        # Set primary date column
        date_cols = [col for col in cleaned_df.columns if pd.api.types.is_datetime64_any_dtype(cleaned_df[col])]
        if len(date_cols) > 0:
            st.markdown("#### 🎯 Primary Date Column")
            st.caption("Select the main date column for time-series analysis")
            
            primary_date = st.selectbox(
                "Primary date column:",
                options=date_cols,
                index=0,
                help="This column will be used for sorting and time-based analysis"
            )
            
            if primary_date:
                cleaned_df = cleaned_df.sort_values(primary_date)
                st.session_state['primary_date_column'] = primary_date
                st.success(f"✅ Set '{primary_date}' as primary date column and sorted data")
        
        return cleaned_df
    
    def _financial_specific_tab(self, df: pd.DataFrame, filename: str) -> pd.DataFrame:
        """
        Financial data specific cleaning operations
        """
        st.markdown("### 💰 Financial Data Cleaning")
        st.info("Specialized cleaning for financial and monetary data.")
        
        cleaned_df = df.copy()
        
        # Detect financial columns
        financial_keywords = {
            'revenue': ['revenue', 'sales', 'income', 'turnover', 'rev_'],
            'expense': ['expense', 'cost', 'spend', 'outflow', 'payment', 'fee'],
            'profit': ['profit', 'margin', 'gross', 'net', 'ebitda', 'earning'],
            'quantity': ['quantity', 'volume', 'units', 'qty', 'amount', 'count'],
            'price': ['price', 'rate', 'fee', 'charge', 'amount', 'value'],
            'balance': ['balance', 'asset', 'liability', 'equity', 'capital']
        }
        
        detected_financial_cols = {}
        for col in cleaned_df.columns:
            col_lower = col.lower().replace('_', ' ').replace('-', ' ')
            for category, keywords in financial_keywords.items():
                if any(keyword in col_lower for keyword in keywords):
                    if category not in detected_financial_cols:
                        detected_financial_cols[category] = []
                    detected_financial_cols[category].append(col)
        
        if detected_financial_cols:
            st.markdown("#### 📊 Detected Financial Columns")
            for category, cols in detected_financial_cols.items():
                st.caption(f"**{category.title()}:** {', '.join(cols[:3])}{'...' if len(cols) > 3 else ''}")
            
            # Currency handling
            st.markdown("#### 💵 Currency Cleaning")
            
            currency_cols = []
            for category in ['revenue', 'expense', 'profit', 'price', 'balance']:
                if category in detected_financial_cols:
                    currency_cols.extend(detected_financial_cols[category])
            
            if currency_cols:
                selected_currency_cols = st.multiselect(
                    "Select monetary columns to clean:",
                    options=currency_cols,
                    default=currency_cols[:3] if len(currency_cols) > 3 else currency_cols,
                    help="Columns that contain currency values"
                )
                
                if selected_currency_cols:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        remove_symbols = st.checkbox("Remove currency symbols", value=True,
                                                   help="Remove $, €, £, ₹, ¥ symbols")
                    
                    with col2:
                        remove_commas = st.checkbox("Remove thousand separators", value=True,
                                                  help="Remove commas in numbers like 1,000 → 1000")
                    
                    if st.button("Clean Currency Values", key="clean_currency"):
                        for col in selected_currency_cols:
                            original_sample = str(cleaned_df[col].iloc[0]) if len(cleaned_df) > 0 else ""
                            
                            if remove_symbols:
                                cleaned_df[col] = cleaned_df[col].astype(str).str.replace(r'[\$,€,£,₹,¥]', '', regex=True)
                            
                            if remove_commas:
                                cleaned_df[col] = cleaned_df[col].astype(str).str.replace(',', '')
                            
                            # Convert to numeric
                            cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
                            
                            cleaned_count = cleaned_df[col].notna().sum()
                            st.success(f"✅ Cleaned '{col}': {cleaned_count:,} valid numeric values")
                            
                            if len(cleaned_df) > 0:
                                new_sample = cleaned_df[col].iloc[0]
                                st.caption(f"Before: {original_sample} → After: {new_sample}")
            
            # Negative value handling
            st.markdown("#### ⚖️ Negative Values")
            st.caption("Financial data often has negative values (credits, refunds, losses)")
            
            numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns.tolist()
            financial_numeric_cols = [col for col in numeric_cols if col in currency_cols]
            
            if financial_numeric_cols:
                selected_col = st.selectbox(
                    "Select column to check for negatives:",
                    options=financial_numeric_cols,
                    help="Check negative values in financial columns"
                )
                
                if selected_col:
                    negative_count = (cleaned_df[selected_col] < 0).sum()
                    total_count = cleaned_df[selected_col].notna().sum()
                    
                    if negative_count > 0:
                        st.warning(f"⚠️ Found {negative_count:,} negative values in '{selected_col}' ({negative_count/total_count*100:.1f}%)")
                        
                        action = st.radio(
                            f"Action for negative values in '{selected_col}':",
                            ["Keep as is (normal for financial data)",
                             "Convert to positive (absolute value)",
                             "Set negative values to zero",
                             "Create credit indicator column"],
                            key=f"neg_action_{selected_col}"
                        )
                        
                        if action != "Keep as is (normal for financial data)":
                            if st.button(f"Apply to '{selected_col}'", key=f"apply_neg_{selected_col}"):
                                if action == "Convert to positive (absolute value)":
                                    cleaned_df[selected_col] = cleaned_df[selected_col].abs()
                                    st.success(f"✅ Converted {negative_count:,} negative values to positive")
                                
                                elif action == "Set negative values to zero":
                                    cleaned_df[selected_col] = cleaned_df[selected_col].clip(lower=0)
                                    st.success(f"✅ Set {negative_count:,} negative values to zero")
                                
                                elif action == "Create credit indicator column":
                                    cleaned_df[f"{selected_col}_is_credit"] = cleaned_df[selected_col] < 0
                                    cleaned_df[selected_col] = cleaned_df[selected_col].abs()
                                    st.success(f"✅ Created credit indicator and converted to positive")
                    else:
                        st.success(f"✅ No negative values found in '{selected_col}'")
        
        else:
            st.info("ℹ️ No financial columns detected automatically. You can still clean numeric columns in other tabs.")
        
        return cleaned_df
    
    def _advanced_cleaning_tab(self, df: pd.DataFrame, filename: str) -> pd.DataFrame:
        """
        Advanced cleaning operations
        """
        st.markdown("### ⚙️ Advanced Cleaning")
        st.info("Advanced operations for data scientists and experienced users.")
        
        cleaned_df = df.copy()
        
        # Outlier detection
        st.markdown("#### 📊 Outlier Detection")
        st.caption("Find and handle unusual values that might be errors")
        
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_cols:
            selected_outlier_cols = st.multiselect(
                "Select columns for outlier analysis:",
                options=numeric_cols,
                default=numeric_cols[:2] if len(numeric_cols) >= 2 else numeric_cols[:1],
                help="Choose numeric columns to check for outliers"
            )
            
            if selected_outlier_cols:
                method = st.selectbox(
                    "Detection method:",
                    ["IQR Method (recommended)", "Z-score Method", "Percentile Method"],
                    help="IQR works well for most datasets"
                )
                
                threshold = st.slider(
                    "Outlier threshold:",
                    min_value=1.0,
                    max_value=3.0,
                    value=1.5,
                    step=0.1,
                    help="Higher values = fewer outliers detected"
                )
                
                if st.button("Detect Outliers", key="detect_outliers"):
                    for col in selected_outlier_cols:
                        outliers_mask = self._detect_outliers(cleaned_df[col], method, threshold)
                        outlier_count = outliers_mask.sum()
                        total_count = cleaned_df[col].notna().sum()
                        
                        if outlier_count > 0:
                            col1, col2 = st.columns(2)
                            with col1:
                                st.warning(f"**{col}**: {outlier_count:,} outliers found")
                            with col2:
                                st.caption(f"({outlier_count/total_count*100:.1f}% of data)")
                            
                            # Show outlier values
                            outlier_values = cleaned_df.loc[outliers_mask, col]
                            with st.expander(f"View outlier values in {col}"):
                                st.dataframe(outlier_values.describe().to_frame().T)
                            
                            # Outlier treatment
                            treatment = st.selectbox(
                                f"Treatment for {col} outliers:",
                                ["Flag only (add outlier column)",
                                 "Cap values (winsorize)",
                                 "Remove outlier rows",
                                 "Replace with median"],
                                key=f"treatment_{col}"
                            )
                            
                            if st.button(f"Apply Treatment to {col}", key=f"apply_treatment_{col}"):
                                if treatment == "Flag only (add outlier column)":
                                    cleaned_df[f"{col}_is_outlier"] = outliers_mask
                                    st.success(f"✅ Added outlier flag column for '{col}'")
                                
                                elif treatment == "Cap values (winsorize)":
                                    if method == "IQR Method (recommended)":
                                        Q1 = cleaned_df[col].quantile(0.25)
                                        Q3 = cleaned_df[col].quantile(0.75)
                                        IQR = Q3 - Q1
                                        lower_bound = Q1 - threshold * IQR
                                        upper_bound = Q3 + threshold * IQR
                                        
                                        capped_low = (cleaned_df[col] < lower_bound).sum()
                                        capped_high = (cleaned_df[col] > upper_bound).sum()
                                        
                                        cleaned_df[col] = cleaned_df[col].clip(lower_bound, upper_bound)
                                        st.success(f"✅ Capped {capped_low + capped_high} outlier values")
                                
                                elif treatment == "Remove outlier rows":
                                    initial_rows = len(cleaned_df)
                                    cleaned_df = cleaned_df[~outliers_mask]
                                    removed = initial_rows - len(cleaned_df)
                                    st.success(f"✅ Removed {removed} rows with outliers in '{col}'")
                                
                                elif treatment == "Replace with median":
                                    median_val = cleaned_df[col].median()
                                    cleaned_df.loc[outliers_mask, col] = median_val
                                    st.success(f"✅ Replaced {outlier_count} outliers with median: {median_val:.2f}")
                        else:
                            st.success(f"✅ No outliers found in '{col}' using {method}")
        
        # Data normalization
        st.markdown("#### 🔄 Data Normalization")
        st.caption("Scale numeric columns for better comparison and modeling")
        
        if numeric_cols:
            selected_norm_cols = st.multiselect(
                "Select columns to normalize:",
                options=numeric_cols,
                help="Normalization is useful for machine learning"
            )
            
            if selected_norm_cols:
                norm_method = st.selectbox(
                    "Normalization method:",
                    ["Min-Max Scaling (0 to 1)",
                     "Standardization (mean=0, std=1)",
                     "Robust Scaling (median based)",
                     "Log Transformation"],
                    help="Min-Max: scales to [0,1], Standardization: z-scores"
                )
                
                if st.button("Apply Normalization", key="apply_normalization"):
                    for col in selected_norm_cols:
                        try:
                            if norm_method == "Min-Max Scaling (0 to 1)":
                                min_val = cleaned_df[col].min()
                                max_val = cleaned_df[col].max()
                                if max_val > min_val:
                                    cleaned_df[f"{col}_norm"] = (cleaned_df[col] - min_val) / (max_val - min_val)
                                    st.success(f"✅ Normalized '{col}' to 0-1 range")
                            
                            elif norm_method == "Standardization (mean=0, std=1)":
                                mean_val = cleaned_df[col].mean()
                                std_val = cleaned_df[col].std()
                                if std_val > 0:
                                    cleaned_df[f"{col}_std"] = (cleaned_df[col] - mean_val) / std_val
                                    st.success(f"✅ Standardized '{col}' (mean=0, std=1)")
                            
                            elif norm_method == "Robust Scaling (median based)":
                                median_val = cleaned_df[col].median()
                                iqr_val = cleaned_df[col].quantile(0.75) - cleaned_df[col].quantile(0.25)
                                if iqr_val > 0:
                                    cleaned_df[f"{col}_robust"] = (cleaned_df[col] - median_val) / iqr_val
                                    st.success(f"✅ Applied robust scaling to '{col}'")
                            
                            elif norm_method == "Log Transformation":
                                if (cleaned_df[col] > 0).all():
                                    cleaned_df[f"{col}_log"] = np.log1p(cleaned_df[col])
                                    st.success(f"✅ Applied log transformation to '{col}'")
                                else:
                                    st.error(f"❌ Cannot apply log to '{col}' - contains non-positive values")
                        
                        except Exception as e:
                            st.error(f"❌ Error normalizing '{col}': {str(e)}")
        
        return cleaned_df
    
    def _detect_outliers(self, series: pd.Series, method: str, threshold: float = 1.5) -> pd.Series:
        """Detect outliers in a series with threshold parameter"""
        if method == "IQR Method (recommended)":
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            return (series < lower_bound) | (series > upper_bound)
        
        elif method == "Z-score Method":
            z_scores = np.abs((series - series.mean()) / series.std())
            return z_scores > threshold
        
        elif method == "Percentile Method":
            lower_bound = series.quantile(0.01)
            upper_bound = series.quantile(0.99)
            return (series < lower_bound) | (series > upper_bound)
        
        return pd.Series(False, index=series.index)
    
    def _apply_to_all_files(self, dataframes: Dict[str, pd.DataFrame], 
                           reference_file: str) -> Dict[str, pd.DataFrame]:
        """Apply similar cleaning patterns to all uploaded files"""
        cleaned_dataframes = {}
        
        if reference_file not in dataframes:
            return dataframes
        
        reference_df = dataframes[reference_file]
        
        for filename, df in dataframes.items():
            if filename == reference_file:
                cleaned_dataframes[filename] = df
                continue
            
            cleaned_df = df.copy()
            applied_transformations = 0
            
            # 1. Apply column renaming based on patterns
            if self.column_mapping:
                for old_name, new_name in self.column_mapping.items():
                    if old_name in cleaned_df.columns and new_name not in cleaned_df.columns:
                        cleaned_df = cleaned_df.rename(columns={old_name: new_name})
                        applied_transformations += 1
            
            # 2. Apply similar data type conversions
            for ref_col in reference_df.columns:
                if ref_col in cleaned_df.columns:
                    ref_dtype = reference_df[ref_col].dtype
                    current_dtype = cleaned_df[ref_col].dtype
                    
                    if ref_dtype != current_dtype:
                        try:
                            if pd.api.types.is_numeric_dtype(ref_dtype):
                                cleaned_df[ref_col] = pd.to_numeric(cleaned_df[ref_col], errors='coerce')
                            elif pd.api.types.is_datetime64_any_dtype(ref_dtype):
                                cleaned_df[ref_col] = pd.to_datetime(cleaned_df[ref_col], errors='coerce')
                            applied_transformations += 1
                        except:
                            pass
            
            if applied_transformations > 0:
                st.info(f"Applied {applied_transformations} transformations to '{filename}'")
            
            cleaned_dataframes[filename] = cleaned_df
        
        return cleaned_dataframes
    
    def _show_changes_summary(self, original_df: pd.DataFrame, cleaned_df: pd.DataFrame):
        """Show detailed summary of changes made"""
        st.markdown("### 📋 Cleaning Summary")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Rows", 
                     f"{cleaned_df.shape[0]:,}", 
                     f"{cleaned_df.shape[0] - original_df.shape[0]:+,}")
        
        with col2:
            st.metric("Columns", 
                     f"{cleaned_df.shape[1]}", 
                     f"{cleaned_df.shape[1] - original_df.shape[1]:+}")
        
        with col3:
            original_nulls = original_df.isnull().sum().sum()
            cleaned_nulls = cleaned_df.isnull().sum().sum()
            st.metric("Null Values", 
                     f"{cleaned_nulls:,}", 
                     f"{cleaned_nulls - original_nulls:+,}")
        
        # Detailed changes
        st.markdown("#### 📊 Detailed Changes")
        
        changes_data = []
        
        # Row changes
        if cleaned_df.shape[0] != original_df.shape[0]:
            changes_data.append({
                "Metric": "Rows",
                "Original": f"{original_df.shape[0]:,}",
                "Cleaned": f"{cleaned_df.shape[0]:,}",
                "Change": f"{cleaned_df.shape[0] - original_df.shape[0]:+,}",
                "Impact": "High" if abs(cleaned_df.shape[0] - original_df.shape[0]) / original_df.shape[0] > 0.1 else "Medium"
            })
        
        # Column changes
        if cleaned_df.shape[1] != original_df.shape[1]:
            changes_data.append({
                "Metric": "Columns",
                "Original": original_df.shape[1],
                "Cleaned": cleaned_df.shape[1],
                "Change": cleaned_df.shape[1] - original_df.shape[1],
                "Impact": "Medium"
            })
        
        # Null value changes
        null_change = original_nulls - cleaned_nulls
        if null_change != 0:
            changes_data.append({
                "Metric": "Null Values",
                "Original": f"{original_nulls:,}",
                "Cleaned": f"{cleaned_nulls:,}",
                "Change": f"{null_change:+,}",
                "Impact": "High" if null_change > 0 else "Low"
            })
        
        # Duplicate changes
        original_dups = original_df.duplicated().sum()
        cleaned_dups = cleaned_df.duplicated().sum()
        if original_dups != cleaned_dups:
            changes_data.append({
                "Metric": "Duplicates",
                "Original": original_dups,
                "Cleaned": cleaned_dups,
                "Change": cleaned_dups - original_dups,
                "Impact": "Medium" if cleaned_dups < original_dups else "Low"
            })
        
        # Memory usage
        original_mem = original_df.memory_usage(deep=True).sum() / 1024**2
        cleaned_mem = cleaned_df.memory_usage(deep=True).sum() / 1024**2
        changes_data.append({
            "Metric": "Memory (MB)",
            "Original": f"{original_mem:.2f}",
            "Cleaned": f"{cleaned_mem:.2f}",
            "Change": f"{cleaned_mem - original_mem:+.2f}",
            "Impact": "Low"
        })
        
        if changes_data:
            changes_df = pd.DataFrame(changes_data)
            st.dataframe(changes_df, use_container_width=True, hide_index=True)
            
            # Summary statement
            positive_changes = sum(1 for change in changes_data if "Impact" in change and change["Impact"] in ["High", "Medium"])
            if positive_changes >= 2:
                st.success("✅ Significant improvements made to data quality!")
            elif positive_changes == 1:
                st.info("ℹ️ Moderate improvements made to data quality")
            else:
                st.warning("⚠️ Minor changes made - consider more aggressive cleaning")
        else:
            st.info("ℹ️ No significant changes detected")
    
    def _log_cleaning_operations(self, filename: str):
        """Log cleaning operations for audit trail"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = {
            'timestamp': timestamp,
            'filename': filename,
            'transformations': self.transformations_applied.copy(),
            'column_mapping': self.column_mapping.copy()
        }
        self.cleaning_log.append(log_entry)
        
        # Store in session state for reporting
        if 'cleaning_log' not in st.session_state:
            st.session_state.cleaning_log = []
        st.session_state.cleaning_log.append(log_entry)


# Streamlit integration function
def display_data_cleaning_section(dataframes: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Main function to display data cleaning interface
    """
    if not dataframes:
        st.warning("Please upload data first in the Data Upload section.")
        return {}
    
    # Initialize session state for original data
    if 'original_uploaded_data' not in st.session_state:
        st.session_state.original_uploaded_data = dataframes.copy()
    
    cleaner = DataCleaner()
    cleaned_dataframes = cleaner.display_cleaning_interface(dataframes)
    
    # Add download option for cleaned data
    if cleaned_dataframes:
        st.markdown("---")
        st.markdown("### 💾 Export Cleaned Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            export_format = st.selectbox(
                "Export format:",
                ["CSV", "Excel", "JSON"],
                help="Choose file format for download"
            )
        
        with col2:
            if st.button("Download All Cleaned Files", use_container_width=True):
                # Create downloadable files
                import io
                from datetime import datetime
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                if export_format == "Excel":
                    # Create Excel file with multiple sheets
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        for filename, df in cleaned_dataframes.items():
                            df.to_excel(writer, sheet_name=filename[:31], index=False)
                    output.seek(0)
                    
                    st.download_button(
                        label=f"Download Excel ({len(cleaned_dataframes)} sheets)",
                        data=output,
                        file_name=f"cleaned_data_{timestamp}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                
                elif export_format == "CSV":
                    # Create ZIP file with multiple CSVs
                    import zipfile
                    
                    zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                        for filename, df in cleaned_dataframes.items():
                            csv_buffer = io.StringIO()
                            df.to_csv(csv_buffer, index=False)
                            zip_file.writestr(f"{filename}_{timestamp}.csv", csv_buffer.getvalue())
                    
                    st.download_button(
                        label=f"Download ZIP with {len(cleaned_dataframes)} CSVs",
                        data=zip_buffer.getvalue(),
                        file_name=f"cleaned_data_{timestamp}.zip",
                        mime="application/zip"
                    )
    
    return cleaned_dataframes