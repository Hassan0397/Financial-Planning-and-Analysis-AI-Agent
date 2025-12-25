"""
File Handler Module
Handles all file uploads, validation, and relationship detection
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from typing import Dict, List, Tuple, Optional, Any
import tempfile
import traceback
from datetime import datetime


class FileHandler:
    """
    Handles file uploads, validation, and relationship detection for FP&A data
    """
    
    def __init__(self):
        self.supported_formats = ['.csv', '.xlsx', '.xls', '.json']
        self.uploaded_files = {}
        self.file_relationships = {}
        self.detected_entities = {
            'company': None,
            'products': [],
            'regions': [],
            'departments': [],
            'scenarios': []
        }
    
    def upload_files(self) -> Dict[str, pd.DataFrame]:
        """
        Main method to handle file uploads via Streamlit
        Returns dictionary of {filename: DataFrame}
        """
        # Clean, modern header
        st.markdown("""
        <div style="padding: 1.5rem 0; margin-bottom: 1.5rem;">
            <h1 style="color: #1a1a2e; margin: 0 0 0.5rem 0; font-weight: 600;">📁 Upload Financial Data</h1>
            <p style="color: #666; margin: 0; font-size: 0.95rem;">
                Upload CSV, Excel, or JSON files for analysis. All processing is done locally.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Elegant file uploader container
        with st.container():
            st.markdown("""
            <style>
            .upload-box {
                border: 2px dashed #4a6bff;
                border-radius: 12px;
                padding: 2rem;
                text-align: center;
                background: linear-gradient(145deg, #f8faff, #ffffff);
                margin: 1rem 0;
                transition: all 0.3s ease;
            }
            .upload-box:hover {
                border-color: #3a5bff;
                background: linear-gradient(145deg, #f0f5ff, #ffffff);
                box-shadow: 0 8px 25px rgba(74, 107, 255, 0.1);
            }
            </style>
            """, unsafe_allow_html=True)
            
            uploaded_files = st.file_uploader(
                " ",
                type=['csv', 'xlsx', 'xls', 'json'],
                accept_multiple_files=True,
                help="Upload multiple files for analysis. The system will detect relationships automatically.",
                label_visibility="collapsed"
            )
        
        # Empty state with cleaner design
        if not uploaded_files:
            st.markdown("""
            <div class="upload-box">
                <div style="font-size: 3rem; color: #4a6bff; margin-bottom: 1rem;">📁</div>
                <h3 style="color: #2d3748; margin-bottom: 0.5rem; font-weight: 500;">Ready to Upload</h3>
                <p style="color: #718096; max-width: 400px; margin: 0 auto; line-height: 1.5;">
                    Drag & drop files here or click to browse<br>
                    Supports: CSV, Excel (XLSX/XLS), JSON
                </p>
            </div>
            """, unsafe_allow_html=True)
            return {}
        
        dataframes = {}
        success_count = 0
        
        # Process each uploaded file with progress indicator
        if uploaded_files:
            progress_container = st.container()
            with progress_container:
                st.markdown("""
                <div style="margin: 1.5rem 0 0.5rem 0;">
                    <p style="color: #4a5568; font-weight: 500; margin-bottom: 0.5rem;">
                        Processing files...
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                progress_bar = st.progress(0)
                
                # Process files
                for idx, uploaded_file in enumerate(uploaded_files):
                    try:
                        file_ext = os.path.splitext(uploaded_file.name)[-1].lower()
                        
                        # Read file based on extension
                        if file_ext == '.csv':
                            df = pd.read_csv(uploaded_file)
                        elif file_ext in ['.xlsx', '.xls']:
                            df = pd.read_excel(uploaded_file)
                        elif file_ext == '.json':
                            df = pd.read_json(uploaded_file)
                        else:
                            st.warning(f"⚠️ Unsupported format: {uploaded_file.name}")
                            continue
                        
                        # Validate dataframe
                        if self._validate_dataframe(df, uploaded_file.name):
                            dataframes[uploaded_file.name] = df
                            success_count += 1
                            
                    except Exception as e:
                        st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
                        if st.checkbox(f"Show details for {uploaded_file.name}"):
                            st.code(traceback.format_exc())
                    
                    # Update progress
                    progress_bar.progress((idx + 1) / len(uploaded_files))
                
                progress_bar.empty()
        
        # Display results
        if success_count > 0:
            # Success notification - FIXED: Changed from st.success to st.markdown
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #d4fc79 0%, #96e6a1 100%);
                border-left: 4px solid #10b981;
                padding: 1rem 1.5rem;
                border-radius: 8px;
                margin: 1.5rem 0;
                display: flex;
                align-items: center;
                gap: 12px;
            ">
                <span style="font-size: 1.8rem;">✅</span>
                <div>
                    <strong style="color: #065f46; font-size: 1.1rem;">
                        Successfully loaded {success_count} file{'' if success_count == 1 else 's'}
                    </strong>
                    <div style="color: #047857; font-size: 0.9rem; margin-top: 0.2rem;">
                        Ready for analysis • {', '.join(list(dataframes.keys())[:3])}{'...' if len(dataframes) > 3 else ''}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Display file previews in tabs
            with st.expander(f"📋 File Details ({success_count} loaded)", expanded=True):
                for filename, df in dataframes.items():
                    with st.container():
                        # File header
                        col1, col2, col3 = st.columns([3, 1, 1])
                        with col1:
                            st.markdown(f"**{filename}**")
                        with col2:
                            st.markdown(f"<small>{df.shape[0]} rows</small>", unsafe_allow_html=True)
                        with col3:
                            st.markdown(f"<small>{df.shape[1]} cols</small>", unsafe_allow_html=True)
                        
                        # Tabs for different views
                        tab1, tab2, tab3 = st.tabs(["Preview", "Types", "Stats"])
                        
                        with tab1:
                            st.dataframe(
                                df.head(8), 
                                use_container_width=True,
                                height=250
                            )
                        
                        with tab2:
                            dtype_info = pd.DataFrame({
                                'Column': df.columns,
                                'Type': df.dtypes.astype(str),
                                'Non-Null': df.notnull().sum(),
                                'Null %': (df.isnull().sum() / len(df) * 100).round(1)
                            })
                            st.dataframe(dtype_info, use_container_width=True)
                        
                        with tab3:
                            cols = st.columns(4)
                            stats = [
                                ("Total Rows", f"{df.shape[0]:,}", "#4f46e5"),
                                ("Total Columns", df.shape[1], "#10b981"),
                                ("Null %", f"{(df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100):.1f}%", "#ef4444"),
                                ("Memory", f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB", "#f59e0b")
                            ]
                            
                            for col, (label, value, color) in zip(cols, stats):
                                with col:
                                    st.markdown(f"""
                                    <div style="text-align: center; padding: 0.8rem; background: #f8fafc; 
                                                border-radius: 8px; border: 1px solid #e2e8f0;">
                                        <div style="font-size: 1rem; color: {color}; font-weight: 600; margin-bottom: 0.2rem;">
                                            {value}
                                        </div>
                                        <div style="font-size: 0.8rem; color: #64748b;">{label}</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                        
                        st.divider()
            
            self.uploaded_files = dataframes
            
            # Detect relationships if multiple files
            if len(dataframes) > 1:
                self._detect_relationships(dataframes)
            
            # Detect entities in the data
            self._detect_entities(dataframes)
            
            return dataframes
        
        return {}
    
    def _validate_dataframe(self, df: pd.DataFrame, filename: str) -> bool:
        """
        Validate if dataframe is suitable for FP&A analysis
        """
        # Check if dataframe is not empty
        if df.empty:
            st.error(f"❌ {filename}: DataFrame is empty")
            return False
        
        # Check minimum required columns
        if len(df.columns) < 2:
            st.error(f"❌ {filename}: Insufficient columns for analysis")
            return False
        
        return True
    
    def _detect_relationships(self, dataframes: Dict[str, pd.DataFrame]):
        """
        Detect relationships between uploaded files
        """
        st.markdown("""
        <div style="margin: 2rem 0 1rem 0;">
            <h3 style="color: #1a1a2e; margin: 0; display: flex; align-items: center; gap: 10px;">
                <span style="color: #4a6bff;">🔗</span>
                <span>File Relationships</span>
            </h3>
            <p style="color: #666; margin: 0.2rem 0 0 0; font-size: 0.9rem;">
                Automatically detected connections between your files
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        file_names = list(dataframes.keys())
        relationships = []
        
        # Simple heuristic-based relationship detection
        with st.spinner("Analyzing file relationships..."):
            for i, (name1, df1) in enumerate(dataframes.items()):
                for j, (name2, df2) in enumerate(dataframes.items()):
                    if i >= j:  # Avoid duplicate checks
                        continue
                    
                    # Find common columns
                    common_cols = set(df1.columns) & set(df2.columns)
                    
                    if common_cols:
                        # Determine relationship type
                        relationship = self._determine_relationship_type(df1, df2, common_cols)
                        
                        if relationship:
                            relationships.append({
                                'file1': name1,
                                'file2': name2,
                                'common_columns': list(common_cols),
                                'relationship': relationship
                            })
        
        # Display detected relationships
        if relationships:
            # Relationship cards in columns
            cols = st.columns(2)
            for idx, rel in enumerate(relationships):
                with cols[idx % 2]:
                    st.markdown(f"""
                    <div style="border: 1px solid #e2e8f0; border-radius: 10px; padding: 1.2rem; 
                                margin-bottom: 1rem; background: white; transition: all 0.2s ease;">
                        <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 0.8rem;">
                            <div style="display: flex; align-items: center; gap: 8px;">
                                <div style="background: #4a6bff; color: white; padding: 0.3rem 0.7rem; 
                                            border-radius: 6px; font-size: 0.8rem; font-weight: 500;">
                                    {rel['relationship']}
                                </div>
                            </div>
                        </div>
                        <div style="margin-bottom: 0.8rem;">
                            <div style="font-size: 0.9rem; color: #4a5568; margin-bottom: 0.3rem;">
                                📄 <strong>{rel['file1']}</strong>
                            </div>
                            <div style="text-align: center; color: #a0aec0; margin: 0.2rem 0;">↕</div>
                            <div style="font-size: 0.9rem; color: #4a5568;">
                                📄 <strong>{rel['file2']}</strong>
                            </div>
                        </div>
                        <div style="font-size: 0.85rem; color: #718096; background: #f8fafc; 
                                    padding: 0.6rem; border-radius: 6px;">
                            <strong>Common columns:</strong><br>
                            {', '.join(rel['common_columns'][:3])}
                            {', ...' if len(rel['common_columns']) > 3 else ''}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            self.file_relationships = relationships
        else:
            st.info("""
            <div style="text-align: center; padding: 2rem; background: #f8fafc; border-radius: 10px;">
                <div style="font-size: 2.5rem; color: #a0aec0; margin-bottom: 1rem;">🔍</div>
                <h4 style="color: #4a5568; margin-bottom: 0.5rem;">No Relationships Found</h4>
                <p style="color: #718096; margin: 0;">
                    No common columns detected between files.<br>
                    You can manually specify joins if needed.
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    def _determine_relationship_type(self, df1: pd.DataFrame, df2: pd.DataFrame, 
                                   common_cols: set) -> Optional[str]:
        """
        Determine the type of relationship between two dataframes
        """
        # Use first common column for analysis
        common_col = list(common_cols)[0]
        
        # Get unique values in each dataframe
        unique1 = df1[common_col].dropna().unique()
        unique2 = df2[common_col].dropna().unique()
        
        # Check for one-to-one relationship
        if len(unique1) == len(unique2) and set(unique1) == set(unique2):
            return "One-to-One"
        
        # Check for one-to-many relationship
        elif len(unique1) < len(unique2) and set(unique1).issubset(set(unique2)):
            return "One-to-Many"
        
        elif len(unique2) < len(unique1) and set(unique2).issubset(set(unique1)):
            return "Many-to-One"
        
        # Check for many-to-many
        elif len(set(unique1) & set(unique2)) > 0:
            return "Many-to-Many"
        
        return None
    
    def _detect_entities(self, dataframes: Dict[str, pd.DataFrame]):
        """
        Detect entities like company, products, regions from data
        """
        entity_keywords = {
            'company': ['company', 'firm', 'organization', 'enterprise'],
            'products': ['product', 'item', 'sku', 'service', 'offering'],
            'regions': ['region', 'country', 'city', 'state', 'territory', 'area'],
            'departments': ['department', 'division', 'team', 'unit', 'segment'],
            'scenarios': ['scenario', 'version', 'plan', 'budget', 'forecast', 'actual']
        }
        
        for filename, df in dataframes.items():
            for col in df.columns:
                col_lower = col.lower()
                
                # Check for entity types
                for entity_type, keywords in entity_keywords.items():
                    if any(keyword in col_lower for keyword in keywords):
                        # Get unique values (limit to 20 for display)
                        unique_values = df[col].dropna().unique()[:20]
                        
                        if entity_type == 'company' and len(unique_values) == 1:
                            self.detected_entities['company'] = str(unique_values[0])
                        elif len(unique_values) > 0:
                            # FIX: Check if existing is None and handle it
                            existing = self.detected_entities[entity_type]
                            
                            # If existing is None (for 'company' type), initialize as empty list
                            if existing is None or (entity_type == 'company' and isinstance(existing, str)):
                                existing = []
                                self.detected_entities[entity_type] = existing
                            
                            # Convert all values to string for consistency
                            new_values = [str(v) for v in unique_values if str(v) not in existing]
                            self.detected_entities[entity_type].extend(new_values)
    
    def get_available_filters(self, dataframes: Dict[str, pd.DataFrame]) -> Dict[str, List]:
        """
        Extract available filter options from uploaded data
        """
        filters = {
            'files': list(dataframes.keys()),
            'columns_by_file': {},
            'categorical_columns': {},
            'date_columns': {},
            'numeric_columns': {}
        }
        
        for filename, df in dataframes.items():
            # Get all columns
            filters['columns_by_file'][filename] = list(df.columns)
            
            # Detect column types
            cat_cols = []
            date_cols = []
            num_cols = []
            
            for col in df.columns:
                # Check if column is numeric
                if pd.api.types.is_numeric_dtype(df[col]):
                    num_cols.append(col)
                
                # Check if column is datetime
                elif pd.api.types.is_datetime64_any_dtype(df[col]):
                    date_cols.append(col)
                else:
                    # Try to convert to datetime
                    try:
                        pd.to_datetime(df[col])
                        date_cols.append(col)
                    except:
                        # If not datetime, treat as categorical
                        cat_cols.append(col)
            
            filters['categorical_columns'][filename] = cat_cols
            filters['date_columns'][filename] = date_cols
            filters['numeric_columns'][filename] = num_cols
        
        return filters
    
    def get_data_summary(self, dataframes: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Generate summary statistics for uploaded data
        """
        summary = {}
        
        for filename, df in dataframes.items():
            summary[filename] = {
                'rows': df.shape[0],
                'columns': df.shape[1],
                'memory_usage': df.memory_usage(deep=True).sum() / 1024**2,  # MB
                'null_values': df.isnull().sum().sum(),
                'null_percentage': (df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100).round(2),
                'duplicates': df.duplicated().sum(),
                'numeric_columns': len([col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]),
                'date_columns': len([col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]),
                'sample_data': df.head(5).to_dict('records')
            }
        
        return summary


# Helper function for Streamlit integration
def display_file_upload_section():
    """
    Streamlit UI component for file upload
    """
    handler = FileHandler()
    
    # Upload files
    dataframes = handler.upload_files()
    
    if dataframes:
        # Clean summary section
        st.markdown("""
        <div style="margin: 2rem 0 1rem 0;">
            <h3 style="color: #1a1a2e; margin: 0; display: flex; align-items: center; gap: 10px;">
                <span style="color: #4a6bff;">📊</span>
                <span>Data Overview</span>
            </h3>
            <p style="color: #666; margin: 0.2rem 0 0 0; font-size: 0.9rem;">
                Summary statistics for uploaded files
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        summary = handler.get_data_summary(dataframes)
        
        # Summary cards in a grid
        cols = st.columns(min(4, len(dataframes)))
        for idx, (filename, stats) in enumerate(summary.items()):
            col_idx = idx % len(cols)
            with cols[col_idx]:
                st.markdown(f"""
                <div style="border: 1px solid #e2e8f0; border-radius: 10px; padding: 1.2rem; 
                            background: white; margin-bottom: 1rem;">
                    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 1rem;">
                        <div style="background: #4a6bff; color: white; width: 36px; height: 36px; 
                                    border-radius: 8px; display: flex; align-items: center; justify-content: center; font-size: 1.2rem;">
                            📄
                        </div>
                        <div>
                            <div style="font-weight: 600; color: #2d3748; font-size: 0.95rem; line-height: 1.2;">
                                {filename[:20]}{'...' if len(filename) > 20 else ''}
                            </div>
                            <div style="font-size: 0.8rem; color: #718096;">
                                {stats['rows']:,} rows × {stats['columns']} cols
                            </div>
                        </div>
                    </div>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem;">
                        <div style="text-align: center;">
                            <div style="font-weight: 600; color: #4f46e5; font-size: 1rem;">{stats['rows']:,}</div>
                            <div style="font-size: 0.75rem; color: #64748b;">Rows</div>
                        </div>
                        <div style="text-align: center;">
                            <div style="font-weight: 600; color: #10b981; font-size: 1rem;">{stats['columns']}</div>
                            <div style="font-size: 0.75rem; color: #64748b;">Columns</div>
                        </div>
                        <div style="text-align: center;">
                            <div style="font-weight: 600; color: #ef4444; font-size: 1rem;">{stats['null_percentage']}%</div>
                            <div style="font-size: 0.75rem; color: #64748b;">Null</div>
                        </div>
                        <div style="text-align: center;">
                            <div style="font-weight: 600; color: #f59e0b; font-size: 1rem;">{stats['duplicates']}</div>
                            <div style="font-size: 0.75rem; color: #64748b;">Dupes</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Display detected entities
        if any(handler.detected_entities.values()):
            st.markdown("""
            <div style="margin: 2rem 0 1rem 0;">
                <h3 style="color: #1a1a2e; margin: 0; display: flex; align-items: center; gap: 10px;">
                    <span style="color: #4a6bff;">🏢</span>
                    <span>Detected Entities</span>
                </h3>
                <p style="color: #666; margin: 0.2rem 0 0 0; font-size: 0.9rem;">
                    Automatically identified entities in your data
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Filter out empty entities
            non_empty_entities = {k: v for k, v in handler.detected_entities.items() if v}
            
            if non_empty_entities:
                # Create entity cards
                entity_cols = st.columns(len(non_empty_entities))
                
                for idx, (entity_type, values) in enumerate(non_empty_entities.items()):
                    with entity_cols[idx]:
                        icon_map = {
                            'company': '🏢',
                            'products': '📦',
                            'regions': '🌍',
                            'departments': '👥',
                            'scenarios': '📊'
                        }
                        
                        icon = icon_map.get(entity_type, '📋')
                        
                        # Handle company separately (it's a string, not a list)
                        if entity_type == 'company' and values:
                            st.markdown(f"""
                            <div style="border: 2px solid #3b82f6; border-radius: 10px; padding: 1.2rem; 
                                        background: linear-gradient(135deg, #eff6ff, #ffffff); text-align: center;">
                                <div style="font-size: 2rem; margin-bottom: 0.5rem;">{icon}</div>
                                <div style="font-size: 1.1rem; font-weight: 600; color: #1e40af; margin-bottom: 0.3rem;">
                                    {values}
                                </div>
                                <div style="font-size: 0.8rem; color: #4b5563; background: #dbeafe; 
                                            padding: 0.3rem 0.6rem; border-radius: 4px; display: inline-block;">
                                    Company
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        elif values and isinstance(values, list):  # Check if it's a list
                            st.markdown(f"""
                            <div style="border: 1px solid #e5e7eb; border-radius: 10px; padding: 1rem; 
                                        background: white; text-align: center;">
                                <div style="font-size: 1.8rem; margin-bottom: 0.5rem; color: #4a6bff;">
                                    {icon}
                                </div>
                                <div style="font-size: 1.2rem; font-weight: 600; color: #1e40af; margin-bottom: 0.2rem;">
                                    {len(values)}
                                </div>
                                <div style="font-size: 0.8rem; color: #6b7280; margin-bottom: 0.8rem; text-transform: capitalize;">
                                    {entity_type}
                                </div>
                                <div style="max-height: 100px; overflow-y: auto; text-align: left; 
                                            background: #f9fafb; padding: 0.5rem; border-radius: 6px;">
                                    {''.join([f'<div style="font-size: 0.75rem; color: #4b5563; padding: 0.1rem 0;">• {str(v)[:20]}{"..." if len(str(v)) > 20 else ""}</div>' for v in values[:5]])}
                                    {f'<div style="font-size: 0.7rem; color: #9ca3af; padding: 0.2rem 0;">+ {len(values)-5} more</div>' if len(values) > 5 else ''}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
        
        return dataframes, handler
    else:
        return None, None