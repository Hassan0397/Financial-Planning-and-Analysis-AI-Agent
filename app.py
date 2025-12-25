"""
Main Streamlit application for FP&A AI Agent
"""

import streamlit as st
import sys
import os

# Add agent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Initialize all session state variables at the start
if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = {}
if 'file_handler' not in st.session_state:
    st.session_state.file_handler = None
if 'detected_entities' not in st.session_state:
    st.session_state.detected_entities = {}
if 'data_cleaned' not in st.session_state:
    st.session_state.data_cleaned = False
if 'filter_state' not in st.session_state:
    st.session_state.filter_state = {}
if 'filtered_data' not in st.session_state:
    st.session_state.filtered_data = {}
if 'filter_history' not in st.session_state:
    st.session_state.filter_history = []
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}
if 'forecast_results' not in st.session_state:
    st.session_state.forecast_results = {}
if 'scenario_results' not in st.session_state:
    st.session_state.scenario_results = {}
if 'scenarios' not in st.session_state:
    st.session_state.scenarios = {}
if 'viz_results' not in st.session_state:
    st.session_state.viz_results = {}
if 'current_page' not in st.session_state:
    st.session_state.current_page = "🏠 Home"
if 'report_data' not in st.session_state:
    st.session_state.report_data = {}
if 'generated_report' not in st.session_state:
    st.session_state.generated_report = None

# Page configuration
st.set_page_config(
    page_title="FP&A AI Agent - Financial Planning & Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS with clean design and proper spacing
st.markdown("""
<style>
    /* Main styling with proper spacing */
    .main {
        padding: 2rem 3rem;
    }
    
    /* Clean header */
    .clean-header {
        background: #ffffff;
        padding: 2.5rem;
        border-radius: 12px;
        color: #1e293b;
        margin-bottom: 2.5rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
        border: 1px solid #e2e8f0;
    }
    
    /* Navigation cards */
    .nav-card {
        background: white;
        border-radius: 10px;
        padding: 1.75rem;
        border: 1px solid #e5e7eb;
        transition: all 0.3s ease;
        height: 100%;
        cursor: pointer;
        margin-bottom: 1rem;
    }
    
    .nav-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.08);
        border-color: #3b82f6;
    }
    
    .nav-card.selected {
        background: #f0f9ff;
        border: 2px solid #3b82f6;
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.75rem;
        border-left: 5px solid;
        box-shadow: 0 4px 12px rgba(0,0,0,0.04);
        height: 100%;
        margin-bottom: 1.5rem;
    }
    
    .metric-card.blue {
        border-left-color: #3b82f6;
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
    }
    
    .metric-card.green {
        border-left-color: #10b981;
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
    }
    
    .metric-card.purple {
        border-left-color: #8b5cf6;
        background: linear-gradient(135deg, #f5f3ff 0%, #ede9fe 100%);
    }
    
    .metric-card.orange {
        border-left-color: #f59e0b;
        background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
    }
    
    .metric-card.red {
        border-left-color: #ef4444;
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: #ffffff;
        border-right: 1px solid #e2e8f0;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #cbd5e1;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #94a3b8;
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        background: #3b82f6;
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        transition: all 0.3s ease;
        font-size: 0.95rem;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(59, 130, 246, 0.25);
        background: #2563eb;
    }
    
    /* Clean button styling */
    .clean-button {
        background: #f59e0b !important;
    }
    
    .clean-button:hover {
        background: #d97706 !important;
        box-shadow: 0 6px 12px rgba(245, 158, 11, 0.25) !important;
    }
    
    /* Analytics button styling */
    .analytics-button {
        background: #10b981 !important;
    }
    
    .analytics-button:hover {
        background: #059669 !important;
        box-shadow: 0 6px 12px rgba(16, 185, 129, 0.25) !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: #f8fafc;
        border-radius: 8px;
        border: 1px solid #e5e7eb;
        font-weight: 500;
    }
    
    /* Radio button styling */
    .st-cc {
        background: white;
        border-radius: 8px;
        padding: 0.5rem;
        border: 1px solid #e2e8f0;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Clean page specific */
    .cleaning-card {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        border: 1px solid #e5e7eb;
        margin-bottom: 1.25rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.03);
    }
    
    .cleaning-card:hover {
        border-color: #f59e0b;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }
    
    .operation-badge {
        display: inline-block;
        padding: 0.35rem 0.85rem;
        border-radius: 16px;
        font-size: 0.8rem;
        font-weight: 500;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
    }
    
    .operation-badge.primary {
        background: #fef3c7;
        color: #92400e;
        border: 1px solid #fbbf24;
    }
    
    .operation-badge.secondary {
        background: #e0e7ff;
        color: #3730a3;
        border: 1px solid #818cf8;
    }
    
    .operation-badge.success {
        background: #d1fae5;
        color: #065f46;
        border: 1px solid #34d399;
    }
    
    /* Filter specific styles */
    .filter-section {
        background: #f8fafc;
        border-radius: 12px;
        padding: 2rem;
        border: 1px solid #e2e8f0;
        margin-bottom: 2rem;
    }
    
    .filter-card {
        background: white;
        border-radius: 10px;
        padding: 1.25rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 6px rgba(0,0,0,0.02);
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    
    .filter-card:hover {
        border-color: #3b82f6;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.1);
    }
    
    .active-filter {
        border-left: 4px solid #10b981;
        background: #f0fdf4;
    }
    
    .filter-badge {
        display: inline-block;
        padding: 0.35rem 0.85rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
    }
    
    .filter-badge.blue {
        background: #dbeafe;
        color: #1e40af;
        border: 1px solid #93c5fd;
    }
    
    .filter-badge.green {
        background: #d1fae5;
        color: #065f46;
        border: 1px solid #a7f3d0;
    }
    
    .filter-badge.purple {
        background: #ede9fe;
        color: #5b21b6;
        border: 1px solid #c4b5fd;
    }
    
    .filter-badge.orange {
        background: #fed7aa;
        color: #9a3412;
        border: 1px solid #fdba74;
    }
    
    /* Analytics specific styles */
    .analytics-section {
        background: #f0fdf4;
        border-radius: 12px;
        padding: 2rem;
        border: 1px solid #bbf7d0;
        margin-bottom: 2rem;
    }
    
    .analytics-card {
        background: white;
        border-radius: 10px;
        padding: 1.25rem;
        border: 1px solid #d1fae5;
        box-shadow: 0 2px 6px rgba(0,0,0,0.02);
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    
    .analytics-card:hover {
        border-color: #10b981;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.1);
    }
    
    .kpi-badge {
        display: inline-block;
        padding: 0.6rem 1.1rem;
        border-radius: 10px;
        font-size: 0.9rem;
        font-weight: 600;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
    }
    
    .kpi-badge.positive {
        background: #d1fae5;
        color: #065f46;
        border: 1px solid #a7f3d0;
    }
    
    .kpi-badge.negative {
        background: #fee2e2;
        color: #991b1b;
        border: 1px solid #fecaca;
    }
    
    .kpi-badge.neutral {
        background: #f3f4f6;
        color: #374151;
        border: 1px solid #d1d5db;
    }
    
    .chart-container {
        background: white;
        border-radius: 10px;
        padding: 1.25rem;
        border: 1px solid #e5e7eb;
        box-shadow: 0 2px 6px rgba(0,0,0,0.02);
        margin-bottom: 1.5rem;
    }
    
    .data-preview {
        background: #f8fafc;
        border-radius: 8px;
        padding: 1.25rem;
        border: 1px solid #e2e8f0;
        margin-top: 1rem;
        max-height: 300px;
        overflow-y: auto;
    }
    
    /* Forecasting specific styles */
    .forecasting-section {
        background: #f5f3ff;
        border-radius: 12px;
        padding: 2rem;
        border: 1px solid #ddd6fe;
        margin-bottom: 2rem;
    }
    
    .forecast-card {
        background: white;
        border-radius: 10px;
        padding: 1.25rem;
        border: 1px solid #ede9fe;
        box-shadow: 0 2px 6px rgba(0,0,0,0.02);
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    
    .forecast-card:hover {
        border-color: #8b5cf6;
        box-shadow: 0 4px 12px rgba(139, 92, 246, 0.1);
    }
    
    .forecast-badge {
        display: inline-block;
        padding: 0.6rem 1.1rem;
        border-radius: 10px;
        font-size: 0.9rem;
        font-weight: 600;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
    }
    
    .forecast-badge.high {
        background: #dbeafe;
        color: #1e40af;
        border: 1px solid #93c5fd;
    }
    
    .forecast-badge.medium {
        background: #fef3c7;
        color: #92400e;
        border: 1px solid #fcd34d;
    }
    
    .forecast-badge.low {
        background: #f3f4f6;
        color: #374151;
        border: 1px solid #d1d5db;
    }
    
    /* Scenario Analysis specific styles */
    .scenario-section {
        background: #fffbeb;
        border-radius: 12px;
        padding: 2rem;
        border: 1px solid #fde68a;
        margin-bottom: 2rem;
    }
    
    .scenario-card {
        background: white;
        border-radius: 10px;
        padding: 1.25rem;
        border: 1px solid #fed7aa;
        box-shadow: 0 2px 6px rgba(0,0,0,0.02);
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    
    .scenario-card:hover {
        border-color: #f97316;
        box-shadow: 0 4px 12px rgba(249, 115, 22, 0.1);
    }
    
    .scenario-badge {
        display: inline-block;
        padding: 0.6rem 1.1rem;
        border-radius: 10px;
        font-size: 0.9rem;
        font-weight: 600;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
    }
    
    .scenario-badge.best-case {
        background: #d1fae5;
        color: #065f46;
        border: 1px solid #a7f3d0;
    }
    
    .scenario-badge.base-case {
        background: #f3f4f6;
        color: #374151;
        border: 1px solid #d1d5db;
    }
    
    .scenario-badge.worst-case {
        background: #fee2e2;
        color: #991b1b;
        border: 1px solid #fecaca;
    }
    
    .scenario-badge.custom {
        background: #dbeafe;
        color: #1e40af;
        border: 1px solid #93c5fd;
    }
    
    /* Visualization specific styles */
    .viz-section {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        border-radius: 12px;
        padding: 2rem;
        border: 1px solid #bae6fd;
        margin-bottom: 2rem;
    }
    
    .viz-card {
        background: white;
        border-radius: 10px;
        padding: 1.25rem;
        border: 1px solid #dbeafe;
        box-shadow: 0 2px 6px rgba(0,0,0,0.02);
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    
    .viz-card:hover {
        border-color: #3b82f6;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.1);
    }
    
    /* Clean section spacing */
    .section-spacing {
        margin-top: 2rem;
        margin-bottom: 2rem;
    }
    
    /* Professional typography */
    h1, h2, h3, h4 {
        font-weight: 600;
        letter-spacing: -0.025em;
    }
    
    h1 {
        font-size: 2.5rem;
        margin-bottom: 1.5rem;
    }
    
    h2 {
        font-size: 2rem;
        margin-bottom: 1.25rem;
    }
    
    h3 {
        font-size: 1.5rem;
        margin-bottom: 1rem;
    }
    
    h4 {
        font-size: 1.25rem;
        margin-bottom: 0.75rem;
    }
    
    p {
        line-height: 1.6;
        margin-bottom: 1rem;
    }
    
    /* Sub-header styling */
    .sub-header {
        color: #1e293b;
        margin-bottom: 2rem;
        font-size: 1.8rem;
        font-weight: 600;
        padding-bottom: 0.75rem;
        border-bottom: 2px solid #e2e8f0;
    }
    
    /* Status indicators */
    .status-indicator {
        display: inline-flex;
        align-items: center;
        padding: 0.35rem 0.85rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
    }
    
    .status-success {
        background: #d1fae5;
        color: #065f46;
    }
    
    .status-warning {
        background: #fef3c7;
        color: #92400e;
    }
    
    .status-info {
        background: #dbeafe;
        color: #1e40af;
    }
    
    /* Card hover effects */
    .hover-card {
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .hover-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    }
    
    /* Form styling */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > div {
        border-radius: 8px;
        border: 1px solid #d1d5db;
    }
    
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus,
    .stSelectbox > div > div > div:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
    }
    
    /* Data table styling */
    .dataframe {
        border-radius: 8px;
        border: 1px solid #e5e7eb;
    }
    
    .dataframe th {
        background: #f8fafc !important;
        font-weight: 600 !important;
        color: #1e293b !important;
    }
    
    /* Alert boxes */
    .stAlert {
        border-radius: 8px;
        border: 1px solid;
    }
    
    .stAlert [data-baseweb="notification"] {
        border-radius: 8px;
    }
    
    /* Reports specific styles */
    .report-section {
        background: linear-gradient(135deg, #f5f3ff 0%, #ede9fe 100%);
        border-radius: 12px;
        padding: 2rem;
        border: 1px solid #ddd6fe;
        margin-bottom: 2rem;
    }
    
    .report-card {
        background: white;
        border-radius: 10px;
        padding: 1.25rem;
        border: 1px solid #ede9fe;
        box-shadow: 0 2px 6px rgba(0,0,0,0.02);
        margin-bottom: 1rem;
        transition: all 0.3s ease;
        height: 100%;
    }
    
    .report-card:hover {
        border-color: #8b5cf6;
        box-shadow: 0 4px 12px rgba(139, 92, 246, 0.1);
    }
    
    .report-badge {
        display: inline-block;
        padding: 0.6rem 1.1rem;
        border-radius: 10px;
        font-size: 0.9rem;
        font-weight: 600;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
    }
    
    .report-badge.pdf {
        background: #fee2e2;
        color: #991b1b;
        border: 1px solid #fecaca;
    }
    
    .report-badge.excel {
        background: #d1fae5;
        color: #065f46;
        border: 1px solid #a7f3d0;
    }
    
    .report-badge.word {
        background: #dbeafe;
        color: #1e40af;
        border: 1px solid #93c5fd;
    }
    
    .report-badge.html {
        background: #fef3c7;
        color: #92400e;
        border: 1px solid #fcd34d;
    }
</style>
""", unsafe_allow_html=True)

# Title and header with clean design
st.markdown("""
<div class="clean-header">
    <div style="display: flex; align-items: center; justify-content: center; gap: 20px; margin-bottom: 1.5rem;">
        <span style="font-size: 3rem;">📊</span>
        <div style="text-align: left;">
            <h1 style="margin: 0; font-size: 2.75rem; font-weight: 700; color: #1e293b;">FP&A AI Agent</h1>
            <p style="font-size: 1.2rem; margin: 0.5rem 0 0 0; color: #64748b; font-weight: 400;">
                Professional Financial Planning & Analysis Platform
            </p>
        </div>
    </div>
    <div style="display: flex; gap: 15px; justify-content: center; margin-top: 2rem; flex-wrap: wrap;">
        <span class="status-indicator status-success">🔒 Privacy First</span>
        <span class="status-indicator status-info">⚡ Real-time Analysis</span>
        <span class="status-indicator status-warning">💼 Enterprise Grade</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar navigation
with st.sidebar:
    # Logo and branding
    st.markdown("""
    <div style="text-align: center; padding: 2rem 1rem;">
        <div style="font-size: 2.5rem; margin-bottom: 0.5rem; color: #3b82f6;">📊</div>
        <h2 style="color: #1e293b; margin: 0; font-size: 1.5rem; font-weight: 600;">FP&A Professional</h2>
        <p style="color: #64748b; font-size: 0.9rem; margin: 0.25rem 0 0 0;">Enterprise Edition</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<hr style='margin: 1.5rem 0; border: none; border-top: 1px solid #e2e8f0;'>", unsafe_allow_html=True)
    
    st.markdown("### 🧭 Navigation")
    
    # Define pages with icons and descriptions
    pages = {
        "🏠 Home": "Dashboard overview",
        "📁 Data Upload": "Upload and validate files",
        "🧹 Data Cleaning": "Clean and preprocess data",
        "⚙️ Filters": "Apply filters and joins",
        "📈 Analytics": "Advanced analytics tools",
        "🔮 Forecasting": "Predictive forecasting",
        "🎯 Scenario Analysis": "What-if scenarios",
        "📊 Visualizations": "Create charts and dashboards",
        "📄 Reports": "Generate professional reports"
    }
    
    # Create navigation with visual indicators
    page = st.radio(
        "Select Module",
        list(pages.keys()),
        index=list(pages.keys()).index(st.session_state.current_page) if st.session_state.current_page in pages else 0,
        label_visibility="collapsed",
        key="page_navigation"
    )
    
    # Update session state with current page
    st.session_state.current_page = page
    
    # Add descriptions
    if page in pages:
        st.markdown(f"""
        <div style="background: #f8fafc; padding: 1.25rem; border-radius: 10px; margin-top: 1rem; border: 1px solid #e2e8f0;">
            <p style="margin: 0; color: #475569; font-size: 0.95rem;">
                <strong style="color: #3b82f6;">Current Module:</strong><br>
                <span style="font-size: 1.1rem; font-weight: 500; color: #1e293b;">{page}</span><br>
                <span style="font-size: 0.9rem; color: #64748b; display: block; margin-top: 0.5rem;">{pages[page]}</span>
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<hr style='margin: 2rem 0; border: none; border-top: 1px solid #e2e8f0;'>", unsafe_allow_html=True)
    
    # Platform Features
    st.markdown("### 🚀 Features")
    st.markdown("""
    <div style="background: #ffffff; padding: 1rem; border-radius: 8px; border: 1px solid #e2e8f0;">
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.75rem; margin-bottom: 1rem;">
            <div style="display: flex; align-items: flex-start; gap: 8px;">
                <span style="color: #10b981; font-size: 1rem;">✓</span>
                <span style="font-size: 0.85rem;">100% Offline</span>
            </div>
            <div style="display: flex; align-items: flex-start; gap: 8px;">
                <span style="color: #10b981; font-size: 1rem;">✓</span>
                <span style="font-size: 0.85rem;">Secure Data</span>
            </div>
            <div style="display: flex; align-items: flex-start; gap: 8px;">
                <span style="color: #10b981; font-size: 1rem;">✓</span>
                <span style="font-size: 0.85rem;">Multi-Format</span>
            </div>
            <div style="display: flex; align-items: flex-start; gap: 8px;">
                <span style="color: #10b981; font-size: 1rem;">✓</span>
                <span style="font-size: 0.85rem;">Real-time</span>
            </div>
        </div>
        <div style="display: flex; align-items: flex-start; gap: 8px;">
            <span style="color: #10b981; font-size: 1rem;">✓</span>
            <span style="font-size: 0.85rem;">Professional FP&A tools with enterprise-grade security</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Session state info
    if 'uploaded_data' in st.session_state and st.session_state.uploaded_data:
        st.markdown("<hr style='margin: 2rem 0; border: none; border-top: 1px solid #e2e8f0;'>", unsafe_allow_html=True)
        st.markdown("### 📊 Session Status")
        uploaded_data = st.session_state.uploaded_data
        
        total_rows = sum(len(df) for df in uploaded_data.values())
        total_columns = sum(len(df.columns) for df in uploaded_data.values())
        file_count = len(uploaded_data)
        
        # Calculate filtered rows if available
        if 'filtered_data' in st.session_state and st.session_state.filtered_data:
            filtered_rows = sum(len(df) for df in st.session_state.filtered_data.values())
        else:
            filtered_rows = total_rows
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%); padding: 1.5rem; border-radius: 12px; border: 1px solid #e2e8f0;">
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
                <div>
                    <div style="font-size: 0.85rem; color: #64748b; margin-bottom: 0.25rem;">Files</div>
                    <div style="font-size: 1.5rem; font-weight: 600; color: #1e293b;">{file_count}</div>
                </div>
                <div>
                    <div style="font-size: 0.85rem; color: #64748b; margin-bottom: 0.25rem;">Total Rows</div>
                    <div style="font-size: 1.5rem; font-weight: 600; color: #1e293b;">{total_rows:,}</div>
                </div>
                <div>
                    <div style="font-size: 0.85rem; color: #64748b; margin-bottom: 0.25rem;">Filtered Rows</div>
                    <div style="font-size: 1.5rem; font-weight: 600; color: {'#8b5cf6' if filtered_rows != total_rows else '#1e293b'};">{filtered_rows:,}</div>
                </div>
                <div>
                    <div style="font-size: 0.85rem; color: #64748b; margin-bottom: 0.25rem;">Status</div>
                    <div style="font-size: 1rem; font-weight: 600; color: #10b981; display: flex; align-items: center; gap: 4px;">
                        <span>●</span> Ready
                    </div>
                </div>
            </div>
            
            <div style="margin-top: 1.25rem; padding-top: 1rem; border-top: 1px solid #e2e8f0;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
                    <span style="font-size: 0.9rem; color: #64748b;">Data Status:</span>
                    <span style="color: {'#10b981' if st.session_state.get('data_cleaned', False) else '#f59e0b'}; font-weight: 600; font-size: 0.9rem;">
                        {'✓ Cleaned' if st.session_state.get('data_cleaned', False) else '⚠️ Raw'}
                    </span>
                </div>
                {'<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;"><span style="font-size: 0.9rem; color: #64748b;">Active Filters:</span><span style="color: #8b5cf6; font-weight: 600; font-size: 0.9rem;">' + str(len(st.session_state.filter_state)) + '</span></div>' if 'filter_state' in st.session_state and st.session_state.filter_state else ''}
                {'<div style="display: flex; justify-content: space-between; align-items: center;"><span style="font-size: 0.9rem; color: #64748b;">Scenarios:</span><span style="color: #f97316; font-weight: 600; font-size: 0.9rem;">' + str(len(st.session_state.scenarios)) + '</span></div>' if 'scenarios' in st.session_state and st.session_state.scenarios else ''}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Show data summary
        if uploaded_data:
            st.markdown("<hr style='margin: 2rem 0; border: none; border-top: 1px solid #e2e8f0;'>", unsafe_allow_html=True)
            st.markdown("### 📋 Data Preview")
            for i, filename in enumerate(list(uploaded_data.keys())[:2]):  # Show first 2 files
                df = uploaded_data[filename]
                st.markdown(f"""
                <div style="background: #f8fafc; padding: 1rem; border-radius: 8px; margin-bottom: 0.75rem; border: 1px solid #e2e8f0;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                        <span style="font-size: 0.9rem; font-weight: 500; color: #1e293b;">{filename[:28]}{'...' if len(filename) > 28 else ''}</span>
                        <span style="font-size: 0.8rem; color: #64748b; background: #ffffff; padding: 0.25rem 0.5rem; border-radius: 4px; border: 1px solid #e2e8f0;">
                            {df.shape[0]:,} × {df.shape[1]}
                        </span>
                    </div>
                    <div style="font-size: 0.8rem; color: #94a3b8;">
                        {', '.join(df.columns[:3].tolist())}{'...' if len(df.columns) > 3 else ''}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            if len(uploaded_data) > 2:
                st.caption(f"+ {len(uploaded_data) - 2} more files")

# Main content based on selected page
if page == "🏠 Home":
    st.markdown('<h2 class="sub-header">Welcome to FP&A Professional</h2>', unsafe_allow_html=True)
    st.markdown('<p style="color: #64748b; font-size: 1.1rem; margin-bottom: 2.5rem;">Transform your financial data into actionable insights with our enterprise-grade analysis platform.</p>', unsafe_allow_html=True)
    
    # Feature cards - First row
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card blue hover-card">
            <div style="font-size: 2.5rem; margin-bottom: 1.25rem;">📁</div>
            <h3 style="color: #1e40af; margin: 0 0 0.75rem 0; font-size: 1.25rem;">Data Management</h3>
            <p style="color: #4b5563; margin: 0; line-height: 1.5;">Upload CSV, Excel, or JSON files with enterprise-grade validation and processing.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card orange hover-card">
            <div style="font-size: 2.5rem; margin-bottom: 1.25rem;">🧹</div>
            <h3 style="color: #b45309; margin: 0 0 0.75rem 0; font-size: 1.25rem;">Data Quality</h3>
            <p style="color: #4b5563; margin: 0; line-height: 1.5;">Clean, preprocess, and standardize your data with automated quality checks.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card green hover-card">
            <div style="font-size: 2.5rem; margin-bottom: 1.25rem;">📈</div>
            <h3 style="color: #047857; margin: 0 0 0.75rem 0; font-size: 1.25rem;">Advanced Analytics</h3>
            <p style="color: #4b5563; margin: 0; line-height: 1.5;">Perform professional financial analysis with statistical insights and visualizations.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Second row of feature cards
    col4, col5, col6 = st.columns(3)
    
    with col4:
        st.markdown("""
        <div class="metric-card purple hover-card">
            <div style="font-size: 2.5rem; margin-bottom: 1.25rem;">⚙️</div>
            <h3 style="color: #6d28d9; margin: 0 0 0.75rem 0; font-size: 1.25rem;">Data Segmentation</h3>
            <p style="color: #4b5563; margin: 0; line-height: 1.5;">Apply filters, joins, and segment your data for focused business analysis.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown("""
        <div class="metric-card red hover-card">
            <div style="font-size: 2.5rem; margin-bottom: 1.25rem;">🔮</div>
            <h3 style="color: #dc2626; margin: 0 0 0.75rem 0; font-size: 1.25rem;">Predictive Analytics</h3>
            <p style="color: #4b5563; margin: 0; line-height: 1.5;">Forecast trends and predict outcomes with advanced statistical models.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col6:
        st.markdown("""
        <div class="metric-card blue hover-card">
            <div style="font-size: 2.5rem; margin-bottom: 1.25rem;">🎯</div>
            <h3 style="color: #1e40af; margin: 0 0 0.75rem 0; font-size: 1.25rem;">Scenario Planning</h3>
            <p style="color: #4b5563; margin: 0; line-height: 1.5;">Create and analyze what-if scenarios for strategic decision making.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<div class='section-spacing'></div>", unsafe_allow_html=True)
    st.markdown("### 🚀 Getting Started")
    
    # Steps with clean design
    steps = [
        ("📁 Data Upload", "Navigate to Data Upload in the sidebar and upload your financial files"),
        ("🧹 Data Cleaning", "Clean and preprocess your data to ensure quality and consistency"),
        ("⚙️ Apply Filters", "Segment your data by applying filters for focused analysis"),
        ("📈 Generate Insights", "Run advanced analytics to uncover patterns and trends"),
        ("📊 Create Visualizations", "Build interactive charts and dashboards for better insights"),
        ("🔮 Create Forecasts", "Build predictive models for future planning"),
        ("🎯 Analyze Scenarios", "Test different business scenarios and their impacts"),
        ("📄 Generate Reports", "Create professional reports for stakeholders")
    ]
    
    for i, (icon, description) in enumerate(steps, 1):
        col1, col2 = st.columns([1, 10])
        with col1:
            st.markdown(f"""
            <div style="background: #3b82f6; color: white; width: 36px; height: 36px; 
                        border-radius: 50%; display: flex; align-items: center; 
                        justify-content: center; font-weight: 600; font-size: 0.9rem;">
                {i}
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div style="margin-bottom: 1.5rem; padding: 0.75rem; background: #f8fafc; border-radius: 8px; border: 1px solid #e2e8f0;">
                <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 0.5rem;">
                    <span style="font-size: 1.25rem;">{icon}</span>
                    <h4 style="margin: 0; color: #1e293b; font-size: 1.1rem;">{description.split(':')[0]}</h4>
                </div>
                <p style="margin: 0; color: #64748b; font-size: 0.95rem;">{description.split(':')[1] if ':' in description else description}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Quick start button
    st.markdown("<div style='margin-top: 3rem;'></div>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Start with Data Upload", use_container_width=True, type="primary"):
            st.session_state.current_page = "📁 Data Upload"
            st.rerun()

elif page == "📁 Data Upload":
    st.markdown('<h2 class="sub-header">📁 Data Upload & Validation</h2>', unsafe_allow_html=True)
    st.markdown('<p style="color: #64748b; margin-bottom: 2rem;">Upload and validate your financial data files in CSV, Excel, or JSON format.</p>', unsafe_allow_html=True)
    
    # Import and use the file handler
    try:
        from agent.file_handler import display_file_upload_section
        
        # Display file upload section
        dataframes, handler = display_file_upload_section()
        
        # Store in session state
        if dataframes:
            st.session_state.uploaded_data = dataframes
            st.session_state.file_handler = handler
            st.session_state.detected_entities = handler.detected_entities
            st.session_state.data_cleaned = False  # Reset clean status
            
            # Clear any existing filters when new data is uploaded
            for key in ['filter_state', 'filtered_data', 'analysis_results', 'scenario_results', 'scenarios', 'viz_results']:
                if key in st.session_state:
                    del st.session_state[key]
            
            # Show next steps in a clean card
            st.markdown("<div class='section-spacing'></div>", unsafe_allow_html=True)
            st.markdown("### ✅ Next Steps")
            
            cols = st.columns(3)
            
            next_steps = [
                ("🧹 Clean Data", "Remove duplicates, handle missing values, and standardize formats", "#f59e0b"),
                ("⚙️ Apply Filters", "Filter and segment your data for focused analysis", "#8b5cf6"),
                ("📈 View Analytics", "Generate insights and visualize your financial data", "#10b981")
            ]
            
            for col, (title, description, color) in zip(cols, next_steps):
                with col:
                    button_key = f"next_step_{title.replace(' ', '_')}"
                    if st.button(title, use_container_width=True, key=button_key):
                        if title == "🧹 Clean Data":
                            st.session_state.current_page = "🧹 Data Cleaning"
                        elif title == "⚙️ Apply Filters":
                            st.session_state.current_page = "⚙️ Filters"
                        elif title == "📈 View Analytics":
                            st.session_state.current_page = "📈 Analytics"
                        st.rerun()
    
    except ImportError as e:
        st.error(f"Error importing module: {str(e)}")
        st.info("Please make sure the file_handler.py file is in the correct directory.")

elif page == "🧹 Data Cleaning":
    st.markdown("""
    <div class="cleaning-card">
        <div style="display: flex; align-items: center; gap: 20px; margin-bottom: 0.5rem;">
            <span style="font-size: 2.5rem;">🧹</span>
            <div>
                <h2 style="color: #92400e; margin: 0;">Data Cleaning & Preprocessing</h2>
                <p style="color: #92400e; opacity: 0.9; margin: 0.5rem 0 0 0;">
                    Clean, transform, and prepare your data for professional analysis
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if 'uploaded_data' not in st.session_state or not st.session_state.uploaded_data:
        st.markdown("""
        <div style="text-align: center; padding: 3rem; background: #f8f9fa; border-radius: 12px; border: 2px dashed #dee2e6; margin: 2rem 0;">
            <div style="font-size: 3rem; margin-bottom: 1.5rem;">📁</div>
            <h3 style="color: #495057; margin-bottom: 1rem;">No Data Found</h3>
            <p style="color: #6c757d; max-width: 500px; margin: 0 auto 2rem auto;">
                Please upload your financial data first to start cleaning and preprocessing.
            </p>
            <div style="display: flex; gap: 1rem; justify-content: center;">
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Go to Data Upload", use_container_width=True, type="primary"):
                st.session_state.current_page = "📁 Data Upload"
                st.rerun()
    else:
        try:
            from agent.data_cleaning import display_data_cleaning_section
            
            # Display cleaning interface
            cleaned_data = display_data_cleaning_section(st.session_state.uploaded_data)
            
            # Update session state with cleaned data
            if cleaned_data:
                st.session_state.uploaded_data = cleaned_data
                st.session_state.data_cleaned = True
                
                # Clear derived data since cleaning resets it
                for key in ['filtered_data', 'analysis_results', 'scenario_results', 'scenarios', 'viz_results']:
                    if key in st.session_state:
                        del st.session_state[key]
                
                # Show success message
                st.markdown("""
                <div style="background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); padding: 1.75rem; border-radius: 12px; margin: 2rem 0; border: 1px solid #34d399;">
                    <div style="display: flex; align-items: center; gap: 20px;">
                        <span style="font-size: 2.5rem;">✅</span>
                        <div>
                            <h3 style="color: #065f46; margin: 0;">Data Cleaning Complete!</h3>
                            <p style="color: #065f46; opacity: 0.9; margin: 0.5rem 0 0 0;">
                                Your data has been cleaned and is ready for professional analysis.
                            </p>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Show next steps
                st.markdown("### 🚀 Next Steps")
                cols = st.columns(3)
                
                next_steps = [
                    ("⚙️ Apply Filters", "Filter and segment your cleaned data", "#8b5cf6"),
                    ("📈 View Analytics", "Generate insights with clean data", "#10b981"),
                    ("🎯 Run Scenarios", "Create and analyze what-if scenarios", "#f97316")
                ]
                
                for col, (title, description, color) in zip(cols, next_steps):
                    with col:
                        button_key = f"clean_next_{title.replace(' ', '_')}"
                        if st.button(title, use_container_width=True, key=button_key):
                            if title == "⚙️ Apply Filters":
                                st.session_state.current_page = "⚙️ Filters"
                            elif title == "📈 View Analytics":
                                st.session_state.current_page = "📈 Analytics"
                            elif title == "🎯 Run Scenarios":
                                st.session_state.current_page = "🎯 Scenario Analysis"
                            st.rerun()
            
        except ImportError as e:
            st.info("""
            **Data Cleaning Module**
            
            The full data cleaning module is under development. For now, you can:
            """)
            
            # Show data summary
            if 'uploaded_data' in st.session_state:
                st.markdown("### 📊 Current Data Summary")
                
                for filename, df in st.session_state.uploaded_data.items():
                    with st.expander(f"📄 {filename}", expanded=False):
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Rows", f"{df.shape[0]:,}")
                        with col2:
                            st.metric("Columns", df.shape[1])
                        with col3:
                            null_count = df.isnull().sum().sum()
                            null_pct = (null_count / (df.shape[0] * df.shape[1]) * 100).round(2)
                            st.metric("Null Values", f"{null_pct}%")
                        with col4:
                            duplicate_count = df.duplicated().sum()
                            st.metric("Duplicates", duplicate_count)
            
            # Basic cleaning options
            st.markdown("### 🧹 Basic Cleaning Options")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Reset to Original Data", use_container_width=True):
                    st.success("Data reset to original state")
                    st.session_state.data_cleaned = False
                    st.rerun()
            
            # Show next steps
            st.markdown("<div class='section-spacing'></div>", unsafe_allow_html=True)
            st.markdown("### ⏭️ Continue to Next Step")
            cols = st.columns(3)
            
            with cols[0]:
                if st.button("⚙️ Apply Filters", use_container_width=True):
                    st.session_state.current_page = "⚙️ Filters"
                    st.rerun()
            
            with cols[1]:
                if st.button("📈 View Analytics", use_container_width=True):
                    st.session_state.current_page = "📈 Analytics"
                    st.rerun()
            
            with cols[2]:
                if st.button("🎯 Run Scenarios", use_container_width=True):
                    st.session_state.current_page = "🎯 Scenario Analysis"
                    st.rerun()

elif page == "⚙️ Filters":
    st.markdown('<h2 class="sub-header">⚙️ Data Segmentation & Filtering</h2>', unsafe_allow_html=True)
    st.markdown('<p style="color: #64748b; margin-bottom: 2rem;">Apply filters, joins, and segment your data for focused business analysis.</p>', unsafe_allow_html=True)
    
    if 'uploaded_data' not in st.session_state or not st.session_state.uploaded_data:
        st.markdown("""
        <div style="text-align: center; padding: 3rem; background: #fefce8; border-radius: 12px; border: 2px dashed #fbbf24; margin: 2rem 0;">
            <div style="font-size: 3rem; margin-bottom: 1.5rem;">⚠️</div>
            <h3 style="color: #92400e; margin-bottom: 1rem;">No Data Found</h3>
            <p style="color: #a16207; max-width: 500px; margin: 0 auto 2rem auto;">
                Please upload and clean your data first before applying filters.
            </p>
            <div style="display: flex; gap: 1rem; justify-content: center;">
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Go to Data Upload", use_container_width=True):
                st.session_state.current_page = "📁 Data Upload"
                st.rerun()
        with col2:
            if st.button("Go to Data Cleaning", use_container_width=True, type="primary"):
                st.session_state.current_page = "🧹 Data Cleaning"
                st.rerun()
    else:
        try:
            from agent.filters import display_filter_section
            
            # Display filter interface
            filtered_data = display_filter_section(st.session_state.uploaded_data)
            
            # Update session state with filtered data
            if filtered_data:
                st.session_state.filtered_data = filtered_data
                
                # Clear derived data since filtering resets them
                for key in ['analysis_results', 'scenario_results', 'scenarios', 'viz_results']:
                    if key in st.session_state:
                        del st.session_state[key]
                
                # Get current data
                current_data = st.session_state.filtered_data if 'filtered_data' in st.session_state else st.session_state.uploaded_data
                
                # Calculate stats
                total_rows = sum(len(df) for df in current_data.values())
                total_columns = sum(len(df.columns) for df in current_data.values())
                file_count = len(current_data)
                
                # Show success message and stats
                st.markdown("""
                <div style="background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); padding: 1.75rem; border-radius: 12px; margin: 2rem 0; border: 1px solid #34d399;">
                    <div style="display: flex; align-items: center; gap: 20px;">
                        <span style="font-size: 2.5rem;">✅</span>
                        <div>
                            <h3 style="color: #065f46; margin: 0;">Filters Applied Successfully!</h3>
                            <p style="color: #065f46; opacity: 0.9; margin: 0.5rem 0 0 0;">
                                Your data has been segmented for focused analysis.
                            </p>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Stats in clean cards
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.markdown(f"""
                    <div class="metric-card blue">
                        <div style="font-size: 1.5rem; margin-bottom: 0.75rem;">📁</div>
                        <div style="font-size: 1.8rem; font-weight: 600; color: #1e40af;">{file_count}</div>
                        <div style="color: #6b7280; font-size: 0.9rem;">Files</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""
                    <div class="metric-card green">
                        <div style="font-size: 1.5rem; margin-bottom: 0.75rem;">📊</div>
                        <div style="font-size: 1.8rem; font-weight: 600; color: #047857;">{total_rows:,}</div>
                        <div style="color: #6b7280; font-size: 0.9rem;">Total Rows</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col3:
                    st.markdown(f"""
                    <div class="metric-card purple">
                        <div style="font-size: 1.5rem; margin-bottom: 0.75rem;">📋</div>
                        <div style="font-size: 1.8rem; font-weight: 600; color: #6d28d9;">{total_columns}</div>
                        <div style="color: #6b7280; font-size: 0.9rem;">Total Columns</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col4:
                    active_filters = len(st.session_state.filter_state) if 'filter_state' in st.session_state else 0
                    st.markdown(f"""
                    <div class="metric-card orange">
                        <div style="font-size: 1.5rem; margin-bottom: 0.75rem;">⚙️</div>
                        <div style="font-size: 1.8rem; font-weight: 600; color: #b45309;">{active_filters}</div>
                        <div style="color: #6b7280; font-size: 0.9rem;">Active Filters</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Show next steps
                st.markdown("<div class='section-spacing'></div>", unsafe_allow_html=True)
                st.markdown("### 🚀 Next Steps")
                
                action_cols = st.columns(3)
                with action_cols[0]:
                    if st.button("📈 Run Analytics", use_container_width=True, type="primary"):
                        st.session_state.current_page = "📈 Analytics"
                        st.rerun()
                with action_cols[1]:
                    if st.button("🔮 Forecasting", use_container_width=True):
                        st.session_state.current_page = "🔮 Forecasting"
                        st.rerun()
                with action_cols[2]:
                    if st.button("📊 Create Visualizations", use_container_width=True):
                        st.session_state.current_page = "📊 Visualizations"
                        st.rerun()
        
        except ImportError as e:
            st.markdown("""
            <div style="background: #f0f9ff; padding: 2.5rem; border-radius: 12px; border: 1px solid #bae6fd; margin: 2rem 0;">
                <div style="text-align: center;">
                    <div style="font-size: 3rem; color: #0369a1; margin-bottom: 1.5rem;">🔧</div>
                    <h3 style="color: #0369a1; margin-bottom: 1rem;">Filters Module Coming Soon!</h3>
                    <p style="color: #0369a1; opacity: 0.9; max-width: 600px; margin: 0 auto 1.5rem auto;">
                        The advanced filtering module is currently under development. It will allow you to apply filters, joins, and segment your data for focused analysis.
                    </p>
                    <div style="background: white; padding: 1rem; border-radius: 10px; display: inline-block; border: 1px solid #bae6fd;">
                        <span style="color: #64748b; font-weight: 500;">Estimated Release:</span>
                        <span style="color: #0369a1; font-weight: bold;"> Q1 2024</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Show data preview
            if 'uploaded_data' in st.session_state:
                st.markdown("<div class='section-spacing'></div>", unsafe_allow_html=True)
                st.markdown("### 📊 Current Data Preview")
                
                for filename, df in list(st.session_state.uploaded_data.items())[:2]:
                    with st.expander(f"📄 {filename}", expanded=False):
                        st.dataframe(df.head(10), use_container_width=True)
                
                # Navigation to other sections
                st.markdown("<div class='section-spacing'></div>", unsafe_allow_html=True)
                st.markdown("### 🔄 Continue Your Analysis")
                
                cols = st.columns(3)
                with cols[0]:
                    if st.button("🧹 Back to Cleaning", use_container_width=True):
                        st.session_state.current_page = "🧹 Data Cleaning"
                        st.rerun()
                with cols[1]:
                    if st.button("📈 Go to Analytics", use_container_width=True, type="primary"):
                        st.session_state.current_page = "📈 Analytics"
                        st.rerun()
                with cols[2]:
                    if st.button("📊 Go to Visualizations", use_container_width=True):
                        st.session_state.current_page = "📊 Visualizations"
                        st.rerun()

elif page == "📈 Analytics":
    st.markdown('<h2 class="sub-header">📈 Financial Analytics Dashboard</h2>', unsafe_allow_html=True)
    st.markdown('<p style="color: #64748b; margin-bottom: 2rem;">Advanced financial analysis, visualization, and business insights.</p>', unsafe_allow_html=True)
    
    # Get current data
    if 'filtered_data' in st.session_state and st.session_state.filtered_data:
        current_data = st.session_state.filtered_data
        data_source = " (Filtered Data)"
    elif 'uploaded_data' in st.session_state and st.session_state.uploaded_data:
        current_data = st.session_state.uploaded_data
        data_source = " (Original Data)"
    else:
        st.markdown("""
        <div style="text-align: center; padding: 3rem; background: #fef2f2; border-radius: 12px; border: 2px dashed #fca5a5; margin: 2rem 0;">
            <div style="font-size: 3rem; margin-bottom: 1.5rem;">⚠️</div>
            <h3 style="color: #dc2626; margin-bottom: 1rem;">No Data Found</h3>
            <p style="color: #b91c1c; max-width: 500px; margin: 0 auto 2rem auto;">
                Please upload data first to run analytics.
            </p>
            <div style="display: flex; gap: 1rem; justify-content: center;">
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Go to Data Upload", use_container_width=True, type="primary"):
            st.session_state.current_page = "📁 Data Upload"
            st.rerun()
        st.stop()
    
    st.info(f"📊 Analyzing {len(current_data)} file(s){data_source}")
    
    try:
        from agent.analytics import display_analytics_section
        
        # Display analytics dashboard
        analysis_results = display_analytics_section(current_data)
        
        # Store results in session state
        if analysis_results:
            st.session_state.analysis_results = analysis_results
            
            # Show success message
            st.markdown("""
            <div style="background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%); padding: 1.75rem; border-radius: 12px; margin: 2rem 0; border: 1px solid #93c5fd;">
                <div style="display: flex; align-items: center; gap: 20px;">
                    <span style="font-size: 2.5rem;">✅</span>
                    <div>
                        <h3 style="color: #1e40af; margin: 0;">Analytics Complete!</h3>
                        <p style="color: #1e40af; opacity: 0.9; margin: 0.5rem 0 0 0;">
                            Financial analysis has been completed. Explore the insights below.
                        </p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Show export options
            st.markdown("<div class='section-spacing'></div>", unsafe_allow_html=True)
            st.markdown("### 📤 Export Analytics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📊 Export to Excel", use_container_width=True):
                    st.success("✅ Export functionality will be implemented in Report Generator")
            
            with col2:
                if st.button("📈 Export Charts", use_container_width=True):
                    st.info("📋 Chart export functionality coming soon")
            
            with col3:
                if st.button("📋 Summary Report", use_container_width=True):
                    st.session_state.current_page = "📄 Reports"
                    st.rerun()
            
            # Show next steps
            st.markdown("<div class='section-spacing'></div>", unsafe_allow_html=True)
            st.markdown("### 🚀 Next Steps")
            
            action_cols = st.columns(3)
            with action_cols[0]:
                if st.button("📊 Create Visualizations", type="primary", use_container_width=True):
                    st.session_state.current_page = "📊 Visualizations"
                    st.rerun()
            with action_cols[1]:
                if st.button("🔮 Run Forecasting", use_container_width=True):
                    st.session_state.current_page = "🔮 Forecasting"
                    st.rerun()
            with action_cols[2]:
                if st.button("🎯 Scenario Analysis", use_container_width=True):
                    st.session_state.current_page = "🎯 Scenario Analysis"
                    st.rerun()
    
    except ImportError as e:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); padding: 2.5rem; border-radius: 12px; border: 1px solid #34d399; margin: 2rem 0;">
            <div style="text-align: center;">
                <div style="font-size: 3rem; color: #065f46; margin-bottom: 1.5rem;">🔧</div>
                <h3 style="color: #065f46; margin-bottom: 1rem;">Analytics Module Coming Soon!</h3>
                <p style="color: #065f46; opacity: 0.9; max-width: 600px; margin: 0 auto 1.5rem auto;">
                    The advanced analytics module is currently under development. It will provide comprehensive financial analysis, visualization, and statistical insights.
                </p>
                <div style="background: white; padding: 1rem; border-radius: 10px; display: inline-block; border: 1px solid #34d399;">
                    <span style="color: #64748b; font-weight: 500;">Estimated Release:</span>
                    <span style="color: #065f46; font-weight: bold;"> Q1 2024</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Show data preview and basic stats
        if current_data:
            st.markdown("<div class='section-spacing'></div>", unsafe_allow_html=True)
            st.markdown("### 📊 Current Data Summary")
            
            total_files = len(current_data)
            total_rows = sum(len(df) for df in current_data.values())
            total_columns = sum(len(df.columns) for df in current_data.values())
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Files", total_files)
            with col2:
                st.metric("Total Rows", f"{total_rows:,}")
            with col3:
                st.metric("Total Columns", total_columns)
            
            # Show sample of data
            st.markdown("<div class='section-spacing'></div>", unsafe_allow_html=True)
            st.markdown("### 📋 Data Preview")
            for filename, df in list(current_data.items())[:1]:
                with st.expander(f"📄 {filename}", expanded=True):
                    st.dataframe(df.head(10), use_container_width=True)
                    st.caption(f"Showing 10 of {df.shape[0]:,} rows, {df.shape[1]} columns")
            
            # Navigation to other sections
            st.markdown("<div class='section-spacing'></div>", unsafe_allow_html=True)
            st.markdown("### 🔄 Continue Your Analysis")
            
            cols = st.columns(3)
            with cols[0]:
                if st.button("⚙️ Back to Filters", use_container_width=True):
                    st.session_state.current_page = "⚙️ Filters"
                    st.rerun()
            with cols[1]:
                if st.button("📊 Go to Visualizations", use_container_width=True, type="primary"):
                    st.session_state.current_page = "📊 Visualizations"
                    st.rerun()
            with cols[2]:
                if st.button("🏠 Return to Home", use_container_width=True):
                    st.session_state.current_page = "🏠 Home"
                    st.rerun()

elif page == "🔮 Forecasting":
    st.markdown('<h2 class="sub-header">🔮 Predictive Analytics & Forecasting</h2>', unsafe_allow_html=True)
    st.markdown('<p style="color: #64748b; margin-bottom: 2rem;">Forecast trends and predict outcomes with advanced statistical models.</p>', unsafe_allow_html=True)
    
    # Get current data
    if 'filtered_data' in st.session_state and st.session_state.filtered_data:
        current_data = st.session_state.filtered_data
        data_source = " (Filtered Data)"
    elif 'uploaded_data' in st.session_state and st.session_state.uploaded_data:
        current_data = st.session_state.uploaded_data
        data_source = " (Original Data)"
    else:
        st.markdown("""
        <div style="text-align: center; padding: 3rem; background: #fefce8; border-radius: 12px; border: 2px dashed #fbbf24; margin: 2rem 0;">
            <div style="font-size: 3rem; margin-bottom: 1.5rem;">⚠️</div>
            <h3 style="color: #92400e; margin-bottom: 1rem;">No Data Found</h3>
            <p style="color: #a16207; max-width: 500px; margin: 0 auto 2rem auto;">
                Please upload data first to run forecasting.
            </p>
            <div style="display: flex; gap: 1rem; justify-content: center;">
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Go to Data Upload", use_container_width=True, type="primary"):
            st.session_state.current_page = "📁 Data Upload"
            st.rerun()
        st.stop()
    
    st.info(f"📈 Forecasting with {len(current_data)} file(s){data_source}")
    
    # Check if Prophet is installed
    try:
        from prophet import Prophet
        prophet_installed = True
    except ImportError:
        prophet_installed = False
        st.warning("⚠️ Facebook Prophet not installed. Some forecasting models may not be available.")
        st.code("pip install prophet", language="bash")
    
    try:
        from agent.forecasting import display_forecasting_section
        
        # Display forecasting interface
        forecast_results = display_forecasting_section(current_data)
        
        # Store results in session state
        if forecast_results:
            st.session_state.forecast_results = forecast_results
            
            # Show success message
            st.markdown("""
            <div style="background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%); padding: 1.75rem; border-radius: 12px; margin: 2rem 0; border: 1px solid #93c5fd;">
                <div style="display: flex; align-items: center; gap: 20px;">
                    <span style="font-size: 2.5rem;">✅</span>
                    <div>
                        <h3 style="color: #1e40af; margin: 0;">Forecasting Complete!</h3>
                        <p style="color: #1e40af; opacity: 0.9; margin: 0.5rem 0 0 0;">
                            Predictive analysis has been completed. Explore the forecasts below.
                        </p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Show export options
            st.markdown("<div class='section-spacing'></div>", unsafe_allow_html=True)
            st.markdown("### 📤 Export Forecasts")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📊 Export to Excel", use_container_width=True):
                    st.success("✅ Export functionality will be implemented in Report Generator")
            
            with col2:
                if st.button("📈 Export Charts", use_container_width=True):
                    st.info("📋 Chart export functionality coming soon")
            
            with col3:
                if st.button("📋 Include in Report", use_container_width=True):
                    st.session_state.current_page = "📄 Reports"
                    st.rerun()
            
            # Show next steps
            st.markdown("<div class='section-spacing'></div>", unsafe_allow_html=True)
            st.markdown("### 🚀 Next Steps")
            
            action_cols = st.columns(3)
            with action_cols[0]:
                if st.button("📊 Create Visualizations", type="primary", use_container_width=True):
                    st.session_state.current_page = "📊 Visualizations"
                    st.rerun()
            with action_cols[1]:
                if st.button("🎯 Scenario Analysis", use_container_width=True):
                    st.session_state.current_page = "🎯 Scenario Analysis"
                    st.rerun()
            with action_cols[2]:
                if st.button("📄 Generate Report", use_container_width=True):
                    st.session_state.current_page = "📄 Reports"
                    st.rerun()
    
    except ImportError as e:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #ede9fe 0%, #ddd6fe 100%); padding: 2.5rem; border-radius: 12px; border: 1px solid #8b5cf6; margin: 2rem 0;">
            <div style="text-align: center;">
                <div style="font-size: 3rem; color: #5b21b6; margin-bottom: 1.5rem;">🔧</div>
                <h3 style="color: #5b21b6; margin-bottom: 1rem;">Forecasting Module Coming Soon!</h3>
                <p style="color: #5b21b6; opacity: 0.9; max-width: 600px; margin: 0 auto 1.5rem auto;">
                    The advanced forecasting module is currently under development. It will provide predictive analytics, time series forecasting, and trend analysis.
                </p>
                <div style="background: white; padding: 1rem; border-radius: 10px; display: inline-block; border: 1px solid #8b5cf6;">
                    <span style="color: #64748b; font-weight: 500;">Estimated Release:</span>
                    <span style="color: #5b21b6; font-weight: bold;"> Q1 2024</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Show data preview and basic stats
        if current_data:
            st.markdown("<div class='section-spacing'></div>", unsafe_allow_html=True)
            st.markdown("### 📊 Current Data Summary")
            
            total_files = len(current_data)
            total_rows = sum(len(df) for df in current_data.values())
            total_columns = sum(len(df.columns) for df in current_data.values())
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Files", total_files)
            with col2:
                st.metric("Total Rows", f"{total_rows:,}")
            with col3:
                st.metric("Total Columns", total_columns)
            
            # Navigation to other sections
            st.markdown("<div class='section-spacing'></div>", unsafe_allow_html=True)
            st.markdown("### 🔄 Continue Your Analysis")
            
            cols = st.columns(3)
            with cols[0]:
                if st.button("📈 Back to Analytics", use_container_width=True):
                    st.session_state.current_page = "📈 Analytics"
                    st.rerun()
            with cols[1]:
                if st.button("📊 Go to Visualizations", use_container_width=True, type="primary"):
                    st.session_state.current_page = "📊 Visualizations"
                    st.rerun()
            with cols[2]:
                if st.button("🏠 Return to Home", use_container_width=True):
                    st.session_state.current_page = "🏠 Home"
                    st.rerun()

elif page == "🎯 Scenario Analysis":
    st.markdown('<h2 class="sub-header">🎯 Scenario Planning & What-If Analysis</h2>', unsafe_allow_html=True)
    st.markdown('<p style="color: #64748b; margin-bottom: 2rem;">Create and analyze different business scenarios for strategic decision making.</p>', unsafe_allow_html=True)
    
    # Get current data
    if 'filtered_data' in st.session_state and st.session_state.filtered_data:
        current_data = st.session_state.filtered_data
        data_source = " (Filtered Data)"
    elif 'uploaded_data' in st.session_state and st.session_state.uploaded_data:
        current_data = st.session_state.uploaded_data
        data_source = " (Original Data)"
    else:
        st.markdown("""
        <div style="text-align: center; padding: 3rem; background: #fefce8; border-radius: 12px; border: 2px dashed #fbbf24; margin: 2rem 0;">
            <div style="font-size: 3rem; margin-bottom: 1.5rem;">⚠️</div>
            <h3 style="color: #92400e; margin-bottom: 1rem;">No Data Found</h3>
            <p style="color: #a16207; max-width: 500px; margin: 0 auto 2rem auto;">
                Please upload data first to run scenario analysis.
            </p>
            <div style="display: flex; gap: 1rem; justify-content: center;">
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Go to Data Upload", use_container_width=True, type="primary"):
            st.session_state.current_page = "📁 Data Upload"
            st.rerun()
        st.stop()
    
    st.info(f"🎯 Analyzing scenarios with {len(current_data)} file(s){data_source}")
    
    try:
        from agent.scenario import display_scenario_section
        
        # Display scenario interface
        scenario_results = display_scenario_section(current_data)
        
        # Store results in session state
        if scenario_results:
            st.session_state.scenario_results = scenario_results
            
            # Show summary if scenarios were created
            if 'created_scenarios' in scenario_results and scenario_results['created_scenarios']:
                st.markdown("""
                <div style="background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); padding: 1.75rem; border-radius: 12px; margin: 2rem 0; border: 1px solid #34d399;">
                    <div style="display: flex; align-items: center; gap: 20px;">
                        <span style="font-size: 2.5rem;">✅</span>
                        <div>
                            <h3 style="color: #065f46; margin: 0;">Scenario Created Successfully!</h3>
                            <p style="color: #065f46; opacity: 0.9; margin: 0.5rem 0 0 0;">
                                Your scenario has been created and analyzed.
                            </p>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Show export options
            st.markdown("<div class='section-spacing'></div>", unsafe_allow_html=True)
            st.markdown("### 📤 Export Scenario Analysis")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📊 Export to Excel", use_container_width=True):
                    st.success("✅ Export functionality will be implemented in Report Generator")
            
            with col2:
                if st.button("📈 Export Charts", use_container_width=True):
                    st.info("📋 Chart export functionality coming soon")
            
            with col3:
                if st.button("📋 Include in Report", use_container_width=True):
                    st.session_state.current_page = "📄 Reports"
                    st.rerun()
            
            # Show next steps
            st.markdown("<div class='section-spacing'></div>", unsafe_allow_html=True)
            st.markdown("### 🚀 Next Steps")
            
            action_cols = st.columns(3)
            with action_cols[0]:
                if st.button("📊 Create Visualizations", type="primary", use_container_width=True):
                    st.session_state.current_page = "📊 Visualizations"
                    st.rerun()
            with action_cols[1]:
                if st.button("🔮 Run Forecasting", use_container_width=True):
                    st.session_state.current_page = "🔮 Forecasting"
                    st.rerun()
            with action_cols[2]:
                if st.button("📄 Generate Report", use_container_width=True):
                    st.session_state.current_page = "📄 Reports"
                    st.rerun()
    
    except ImportError as e:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #fed7aa 0%, #fdba74 100%); padding: 2.5rem; border-radius: 12px; border: 1px solid #f97316; margin: 2rem 0;">
            <div style="text-align: center;">
                <div style="font-size: 3rem; color: #c2410c; margin-bottom: 1.5rem;">🔧</div>
                <h3 style="color: #c2410c; margin-bottom: 1rem;">Scenario Analysis Module Coming Soon!</h3>
                <p style="color: #c2410c; opacity: 0.9; max-width: 600px; margin: 0 auto 1.5rem auto;">
                    The advanced scenario analysis module is currently under development. It will allow you to create and analyze different financial scenarios, perform sensitivity analysis, and compare what-if scenarios.
                </p>
                <div style="background: white; padding: 1rem; border-radius: 10px; display: inline-block; border: 1px solid #f97316;">
                    <span style="color: #64748b; font-weight: 500;">Estimated Release:</span>
                    <span style="color: #c2410c; font-weight: bold;"> Q1 2024</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Show basic scenario creation form
        st.markdown("<div class='section-spacing'></div>", unsafe_allow_html=True)
        st.markdown("### 📋 Basic Scenario Creation")
        
        with st.form("basic_scenario_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                scenario_name = st.text_input("Scenario Name", "Best Case Scenario")
                scenario_type = st.selectbox(
                    "Scenario Type",
                    ["Best Case", "Base Case", "Worst Case", "Custom"]
                )
            
            with col2:
                revenue_change = st.slider(
                    "Revenue Change (%)",
                    min_value=-50,
                    max_value=100,
                    value=10,
                    step=5,
                    help="Expected change in revenue for this scenario"
                )
                cost_change = st.slider(
                    "Cost Change (%)",
                    min_value=-30,
                    max_value=50,
                    value=5,
                    step=5,
                    help="Expected change in costs for this scenario"
                )
            
            submitted = st.form_submit_button("Create Scenario", use_container_width=True, type="primary")
            
            if submitted:
                st.success(f"✅ Scenario '{scenario_name}' created successfully!")
                st.info(f"📊 Parameters: Revenue {revenue_change:+}%, Costs {cost_change:+}%")
                
                # Store basic scenario in session state
                if 'scenarios' not in st.session_state:
                    st.session_state.scenarios = {}
                
                scenario_id = f"scenario_{len(st.session_state.scenarios) + 1}"
                st.session_state.scenarios[scenario_id] = {
                    'name': scenario_name,
                    'type': scenario_type,
                    'revenue_change': revenue_change,
                    'cost_change': cost_change,
                    'created_at': '2024-01-01'
                }
        
        # Navigation to other sections
        st.markdown("<div class='section-spacing'></div>", unsafe_allow_html=True)
        st.markdown("### 🔄 Continue Your Analysis")
        
        cols = st.columns(3)
        with cols[0]:
            if st.button("🔮 Back to Forecasting", use_container_width=True):
                st.session_state.current_page = "🔮 Forecasting"
                st.rerun()
        with cols[1]:
            if st.button("📊 Go to Visualizations", use_container_width=True, type="primary"):
                st.session_state.current_page = "📊 Visualizations"
                st.rerun()
        with cols[2]:
            if st.button("📄 Go to Reports", use_container_width=True):
                st.session_state.current_page = "📄 Reports"
                st.rerun()

elif page == "📊 Visualizations":
    st.markdown('<h2 class="sub-header">📊 Advanced Visualizations</h2>', unsafe_allow_html=True)
    st.markdown('<p style="color: #64748b; margin-bottom: 2rem;">Create interactive charts, dashboards, and visualizations for better data insights.</p>', unsafe_allow_html=True)
    
    # Get current data
    if 'filtered_data' in st.session_state and st.session_state.filtered_data:
        current_data = st.session_state.filtered_data
        data_source = " (Filtered Data)"
    elif 'uploaded_data' in st.session_state and st.session_state.uploaded_data:
        current_data = st.session_state.uploaded_data
        data_source = " (Original Data)"
    else:
        st.markdown("""
        <div style="text-align: center; padding: 3rem; background: #f0f9ff; border-radius: 12px; border: 2px dashed #93c5fd; margin: 2rem 0;">
            <div style="font-size: 3rem; margin-bottom: 1.5rem;">📊</div>
            <h3 style="color: #1e40af; margin-bottom: 1rem;">No Data Found</h3>
            <p style="color: #1d4ed8; max-width: 500px; margin: 0 auto 2rem auto;">
                Please upload data first to create visualizations.
            </p>
            <div style="display: flex; gap: 1rem; justify-content: center;">
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Go to Data Upload", use_container_width=True, type="primary"):
            st.session_state.current_page = "📁 Data Upload"
            st.rerun()
        st.stop()
    
    st.info(f"📊 Creating visualizations with {len(current_data)} file(s){data_source}")
    
    try:
        from agent.visualizations import display_visualizations_section
        
        # Display visualizations interface
        viz_results = display_visualizations_section(current_data)
        
        # Store results in session state
        if viz_results:
            st.session_state.viz_results = viz_results
            
            # Show success message
            st.markdown("""
            <div style="background: linear-gradient(135deg, #e0e7ff 0%, #c7d2fe 100%); padding: 1.75rem; border-radius: 12px; margin: 2rem 0; border: 1px solid #818cf8;">
                <div style="display: flex; align-items: center; gap: 20px;">
                    <span style="font-size: 2.5rem;">✅</span>
                    <div>
                        <h3 style="color: #3730a3; margin: 0;">Visualizations Created!</h3>
                        <p style="color: #3730a3; opacity: 0.9; margin: 0.5rem 0 0 0;">
                            Your interactive visualizations have been generated. Explore the charts and dashboards below.
                        </p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Show capabilities
            st.markdown("---")
            st.markdown("### 🎨 Visualization Capabilities")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("""
                <div class="viz-card">
                    <h4>📈 Chart Builder</h4>
                    <p style="color: #64748b; font-size: 0.9rem;">
                        Create custom charts with drag-and-drop interface
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class="viz-card">
                    <h4>📊 Dashboards</h4>
                    <p style="color: #64748b; font-size: 0.9rem;">
                        Build interactive dashboards with multiple visualizations
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown("""
                <div class="viz-card">
                    <h4>🔬 Advanced Charts</h4>
                    <p style="color: #64748b; font-size: 0.9rem;">
                        Create correlation matrices, heatmaps, treemaps, and more
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            # Show export options
            st.markdown("---")
            st.markdown("### 📤 Export Visualizations")
            
            export_cols = st.columns(3)
            
            with export_cols[0]:
                if st.button("📷 Export as Image", use_container_width=True):
                    st.info("📋 Image export functionality coming soon")
            
            with export_cols[1]:
                if st.button("📊 Export Dashboard", use_container_width=True):
                    st.info("📋 Dashboard export functionality coming soon")
            
            with export_cols[2]:
                if st.button("📄 Include in Report", use_container_width=True):
                    st.session_state.current_page = "📄 Reports"
                    st.rerun()
            
            # Show next steps
            st.markdown("---")
            st.markdown("### 🚀 Next Steps")
            
            action_cols = st.columns(3)
            with action_cols[0]:
                if st.button("📄 Generate Report", type="primary", use_container_width=True):
                    st.session_state.current_page = "📄 Reports"
                    st.rerun()
            with action_cols[1]:
                if st.button("📈 View Analytics", use_container_width=True):
                    st.session_state.current_page = "📈 Analytics"
                    st.rerun()
            with action_cols[2]:
                if st.button("🔮 Forecasting", use_container_width=True):
                    st.session_state.current_page = "🔮 Forecasting"
                    st.rerun()
    
    except ImportError as e:
        st.markdown("""
        <div class="viz-section">
            <div style="text-align: center;">
                <div style="font-size: 3rem; color: #1e40af; margin-bottom: 1.5rem;">🎨</div>
                <h3 style="color: #1e40af; margin-bottom: 1rem;">Advanced Visualizations</h3>
                <p style="color: #1e40af; opacity: 0.9; max-width: 600px; margin: 0 auto 1.5rem auto;">
                    Create interactive charts, dashboards, and data visualizations to better understand your financial data.
                </p>
                <div style="background: white; padding: 1rem; border-radius: 10px; display: inline-block; border: 1px solid #93c5fd;">
                    <span style="color: #64748b; font-weight: 500;">Available Chart Types:</span>
                    <span style="color: #1e40af; font-weight: bold;"> Line, Bar, Scatter, Pie, Heatmaps</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Show data preview and basic visualization options
        if current_data:
            st.markdown("<div class='section-spacing'></div>", unsafe_allow_html=True)
            st.markdown("### 📊 Data Summary for Visualization")
            
            total_files = len(current_data)
            total_rows = sum(len(df) for df in current_data.values())
            total_columns = sum(len(df.columns) for df in current_data.values())
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Files", total_files)
            with col2:
                st.metric("Total Rows", f"{total_rows:,}")
            with col3:
                st.metric("Total Columns", total_columns)
            
            # Basic visualization options
            st.markdown("<div class='section-spacing'></div>", unsafe_allow_html=True)
            st.markdown("### 📈 Quick Visualization Options")
            
            viz_type = st.selectbox(
                "Select Chart Type",
                ["Line Chart", "Bar Chart", "Scatter Plot", "Pie Chart", "Area Chart"]
            )
            
            if st.button("Generate Preview", use_container_width=True, type="primary"):
                st.info(f"📊 Generating {viz_type} preview... (Full visualization module coming soon)")
                
                # Show sample data for visualization
                for filename, df in list(current_data.items())[:1]:
                    st.markdown(f"**Sample Data from: {filename}**")
                    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                    
                    if len(numeric_cols) >= 2:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.selectbox("X-Axis", numeric_cols, key="x_axis")
                        with col2:
                            st.selectbox("Y-Axis", numeric_cols, key="y_axis")
                        
                        if st.button("Create Chart", use_container_width=True):
                            st.success(f"✅ {viz_type} created successfully!")
                            st.info("Interactive chart will appear here in the full version.")
            
            # Navigation to other sections
            st.markdown("<div class='section-spacing'></div>", unsafe_allow_html=True)
            st.markdown("### 🔄 Continue Your Analysis")
            
            cols = st.columns(3)
            with cols[0]:
                if st.button("📈 Back to Analytics", use_container_width=True):
                    st.session_state.current_page = "📈 Analytics"
                    st.rerun()
            with cols[1]:
                if st.button("🎯 Go to Scenarios", use_container_width=True):
                    st.session_state.current_page = "🎯 Scenario Analysis"
                    st.rerun()
            with cols[2]:
                if st.button("📄 Go to Reports", use_container_width=True, type="primary"):
                    st.session_state.current_page = "📄 Reports"
                    st.rerun()

# ADDED: Reports Page - FIXED VERSION
elif page == "📄 Reports":
    st.markdown('<h2 class="sub-header">📄 Professional Report Generation</h2>', unsafe_allow_html=True)
    st.markdown('<p style="color: #64748b; margin-bottom: 2rem;">Generate professional reports, presentations, and documents from your financial analysis.</p>', unsafe_allow_html=True)
    
    # Get current data
    if 'filtered_data' in st.session_state and st.session_state.filtered_data:
        current_data = st.session_state.filtered_data
        data_source = " (Filtered Data)"
    elif 'uploaded_data' in st.session_state and st.session_state.uploaded_data:
        current_data = st.session_state.uploaded_data
        data_source = " (Original Data)"
    else:
        st.warning("⚠️ No data found. Please upload data first.")
        st.info("Navigate to **📁 Data Upload** to upload your financial files.")
        
        if st.button("Go to Data Upload", use_container_width=True):
            st.session_state.current_page = "📁 Data Upload"
            st.rerun()
        st.stop()
    
    st.info(f"📄 Generating reports from {len(current_data)} file(s){data_source}")
    
    # Check for reporting library dependencies
    try:
        from reportlab.lib.pagesizes import letter
        reportlab_available = True
    except ImportError:
        reportlab_available = False
    
    try:
        from docx import Document
        python_docx_available = True
    except ImportError:
        python_docx_available = False
    
    try:
        from pptx import Presentation
        python_pptx_available = True
    except ImportError:
        python_pptx_available = False
    
    missing_deps = []
    if not reportlab_available:
        missing_deps.append("reportlab (for PDF)")
    if not python_docx_available:
        missing_deps.append("python-docx (for Word)")
    if not python_pptx_available:
        missing_deps.append("python-pptx (for PowerPoint)")
    
    if missing_deps:
        st.warning("⚠️ Some reporting features require additional libraries:")
        for dep in missing_deps:
            st.code(f"pip install {dep.split(' ')[0]}", language="bash")
        st.info("You can still generate HTML reports without these libraries.")
    
    try:
        from agent.report_generator import display_report_generator_section
        
        # Display report generator interface
        report_results = display_report_generator_section(current_data)
        
        # Store results in session state
        if report_results:
            st.session_state.report_results = report_results
            
            # Show capabilities
            st.markdown("---")
            st.markdown("### 📋 Report Capabilities")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown("""
                <div class="report-card">
                    <h4 style="color: #991b1b;">📊 PDF Reports</h4>
                    <p style="color: #64748b; font-size: 0.9rem;">
                        Professional PDF reports with charts and tables
                    </p>
                    <span class="report-badge pdf">PDF</span>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class="report-card">
                    <h4 style="color: #065f46;">📝 Word Documents</h4>
                    <p style="color: #64748b; font-size: 0.9rem;">
                        Editable Word documents with full formatting
                    </p>
                    <span class="report-badge word">DOCX</span>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown("""
                <div class="report-card">
                    <h4 style="color: #1e40af;">📈 Presentations</h4>
                    <p style="color: #64748b; font-size: 0.9rem;">
                        PowerPoint presentations for meetings
                    </p>
                    <span class="report-badge">PPTX</span>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown("""
                <div class="report-card">
                    <h4 style="color: #92400e;">🌐 HTML Reports</h4>
                    <p style="color: #64748b; font-size: 0.9rem;">
                        Interactive web reports for sharing
                    </p>
                    <span class="report-badge html">HTML</span>
                </div>
                """, unsafe_allow_html=True)
            
            # Show report types
            st.markdown("---")
            st.markdown("### 📑 Report Types")
            
            report_types = st.columns(3)
            
            with report_types[0]:
                if st.button("📈 Analytics Summary", use_container_width=True):
                    st.success("✅ Analytics Summary report generated!")
                    # Store report data in session state for download
                    st.session_state.generated_report = {
                        'name': 'Analytics Summary',
                        'type': 'analytics',
                        'content': 'Sample analytics report content',
                        'format': 'pdf'
                    }
            
            with report_types[1]:
                if st.button("🔮 Forecasting Report", use_container_width=True):
                    st.success("✅ Forecasting report generated!")
                    st.session_state.generated_report = {
                        'name': 'Forecasting Report',
                        'type': 'forecasting',
                        'content': 'Sample forecasting report content',
                        'format': 'pdf'
                    }
            
            with report_types[2]:
                if st.button("🎯 Scenario Analysis", use_container_width=True):
                    st.success("✅ Scenario Analysis report generated!")
                    st.session_state.generated_report = {
                        'name': 'Scenario Analysis',
                        'type': 'scenario',
                        'content': 'Sample scenario analysis report content',
                        'format': 'pdf'
                    }
        
        # Custom report form - Moved download button outside of form
        st.markdown("---")
        st.markdown("### 🎨 Custom Report")
        
        # Initialize form data in session state
        if 'custom_report_data' not in st.session_state:
            st.session_state.custom_report_data = {
                'report_name': 'Financial Analysis Report',
                'report_type': 'Comprehensive',
                'output_format': 'PDF',
                'include_charts': True,
                'include_summary': True,
                'include_recommendations': True
            }
        
        with st.form("custom_report_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                report_name = st.text_input("Report Name", "Financial Analysis Report")
                report_type = st.selectbox(
                    "Report Type",
                    ["Comprehensive", "Executive Summary", "Technical Analysis", "Dashboard"]
                )
                include_charts = st.checkbox("Include Charts", value=True)
            
            with col2:
                output_format = st.selectbox(
                    "Output Format",
                    ["PDF", "Word", "PowerPoint", "HTML"]
                )
                include_summary = st.checkbox("Executive Summary", value=True)
                include_recommendations = st.checkbox("Recommendations", value=True)
            
            submitted = st.form_submit_button("Generate Custom Report", use_container_width=True, type="primary")
            
            if submitted:
                # Store form data in session state
                st.session_state.custom_report_data = {
                    'report_name': report_name,
                    'report_type': report_type,
                    'output_format': output_format,
                    'include_charts': include_charts,
                    'include_summary': include_summary,
                    'include_recommendations': include_recommendations,
                    'submitted': True
                }
                
                # Store generated report data
                st.session_state.generated_report = {
                    'name': report_name,
                    'type': report_type,
                    'format': output_format.lower(),
                    'content': f'Custom {report_type} report in {output_format} format',
                    'charts': include_charts,
                    'summary': include_summary,
                    'recommendations': include_recommendations
                }
                
                st.success(f"✅ Custom report '{report_name}' generation started!")
                st.info(f"📋 Generating {output_format} report with your selected options...")
        
        # Show download button if report was generated (outside of form)
        if 'generated_report' in st.session_state and st.session_state.generated_report:
            st.markdown("---")
            st.markdown("### 📥 Download Generated Report")
            
            report = st.session_state.generated_report
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.info(f"**{report['name']}** - {report['type']} Report")
            
            with col2:
                # Create a simple report content
                report_content = f"""
                {report['name']}
                =====================
                
                Report Type: {report['type']}
                Format: {report.get('format', 'pdf').upper()}
                Generated: {st.session_state.get('report_generation_time', 'Today')}
                
                Contents:
                - Executive Summary: {'✓ Included' if report.get('summary', True) else '✗ Not Included'}
                - Charts & Visualizations: {'✓ Included' if report.get('charts', True) else '✗ Not Included'}
                - Recommendations: {'✓ Included' if report.get('recommendations', True) else '✗ Not Included'}
                
                This is a sample report generated by FP&A AI Agent.
                For actual report generation, implement the report_generator module.
                """
                
                # Determine file extension based on format
                file_ext = report.get('format', 'pdf')
                if file_ext == 'word':
                    file_ext = 'docx'
                elif file_ext == 'powerpoint':
                    file_ext = 'pptx'
                
                st.download_button(
                    label="📥 Download Report",
                    data=report_content,
                    file_name=f"{report['name'].replace(' ', '_')}.{file_ext}",
                    mime="application/octet-stream",
                    use_container_width=True
                )
            
            with col3:
                if st.button("🔄 Generate Another", use_container_width=True):
                    # Clear generated report
                    del st.session_state.generated_report
                    if 'custom_report_data' in st.session_state:
                        st.session_state.custom_report_data['submitted'] = False
                    st.rerun()
    
    except ImportError as e:
        st.markdown("""
        <div class="report-section">
            <div style="text-align: center;">
                <div style="font-size: 3rem; color: #5b21b6; margin-bottom: 1.5rem;">📄</div>
                <h3 style="color: #5b21b6; margin-bottom: 1rem;">Professional Report Generator</h3>
                <p style="color: #5b21b6; opacity: 0.9; max-width: 600px; margin: 0 auto 1.5rem auto;">
                    Generate professional reports, presentations, and documents from your financial analysis.
                </p>
                <div style="background: white; padding: 1rem; border-radius: 10px; display: inline-block; border: 1px solid #8b5cf6;">
                    <span style="color: #64748b; font-weight: 500;">Supported Formats:</span>
                    <span style="color: #5b21b6; font-weight: bold;"> PDF, Word, PowerPoint, HTML</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Show report options
        st.markdown("<div class='section-spacing'></div>", unsafe_allow_html=True)
        st.markdown("### 📋 Available Report Templates")
        
        templates = st.columns(3)
        
        with templates[0]:
            st.markdown("""
            <div style="background: #f8fafc; padding: 1.5rem; border-radius: 10px; border: 1px solid #e2e8f0; height: 100%;">
                <h4 style="color: #1e40af; margin: 0 0 0.5rem 0;">Analytics Summary</h4>
                <p style="color: #64748b; font-size: 0.9rem; margin: 0;">
                    Summary of key metrics and insights from your analysis.
                </p>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Use Template", key="template1", use_container_width=True):
                st.info("Report template selected. Create agent/report_generator.py module to implement.")
        
        with templates[1]:
            st.markdown("""
            <div style="background: #f8fafc; padding: 1.5rem; border-radius: 10px; border: 1px solid #e2e8f0; height: 100%;">
                <h4 style="color: #065f46; margin: 0 0 0.5rem 0;">Forecasting Report</h4>
                <p style="color: #64748b; font-size: 0.9rem; margin: 0;">
                    Predictive analytics and trend forecasts.
                </p>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Use Template", key="template2", use_container_width=True):
                st.info("Report template selected. Create agent/report_generator.py module to implement.")
        
        with templates[2]:
            st.markdown("""
            <div style="background: #f8fafc; padding: 1.5rem; border-radius: 10px; border: 1px solid #e2e8f0; height: 100%;">
                <h4 style="color: #92400e; margin: 0 0 0.5rem 0;">Scenario Analysis</h4>
                <p style="color: #64748b; font-size: 0.9rem; margin: 0;">
                    What-if scenario comparisons and impacts.
                </p>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Use Template", key="template3", use_container_width=True):
                st.info("Report template selected. Create agent/report_generator.py module to implement.")
        
        # Custom report form for demonstration
        st.markdown("<div class='section-spacing'></div>", unsafe_allow_html=True)
        st.markdown("### 🎨 Custom Report Demo")
        
        with st.form("demo_report_form"):
            demo_name = st.text_input("Report Name", "Demo Financial Report")
            demo_format = st.selectbox("Format", ["PDF", "Word", "HTML"])
            
            demo_submitted = st.form_submit_button("Generate Demo Report", use_container_width=True)
            
            if demo_submitted:
                st.success(f"✅ Demo report '{demo_name}' would be generated in {demo_format} format!")
                st.info("To implement full functionality, create the agent/report_generator.py module.")
                
                # Create a simple demo report for download
                demo_content = f"""
                {demo_name}
                =====================
                
                This is a demo report generated by FP&A AI Agent.
                
                Report Details:
                - Format: {demo_format}
                - Date: Demo
                - Status: Sample Content
                
                Actual report generation requires implementation of the report generator module.
                """
                
                # Show download button outside the form
                st.session_state.demo_report_ready = True
                st.session_state.demo_report_data = {
                    'name': demo_name,
                    'format': demo_format.lower(),
                    'content': demo_content
                }
        
        # Download button for demo report (outside the form)
        if st.session_state.get('demo_report_ready', False):
            demo_data = st.session_state.get('demo_report_data', {})
            if demo_data:
                st.download_button(
                    label="📥 Download Demo Report",
                    data=demo_data['content'],
                    file_name=f"{demo_data['name'].replace(' ', '_')}.{demo_data['format']}",
                    mime="application/octet-stream",
                    use_container_width=True
                )
        
        # Navigation to other sections
        st.markdown("<div class='section-spacing'></div>", unsafe_allow_html=True)
        st.markdown("### 🔄 Continue Your Analysis")
        
        cols = st.columns(3)
        with cols[0]:
            if st.button("📊 Back to Visualizations", use_container_width=True):
                st.session_state.current_page = "📊 Visualizations"
                st.rerun()
        with cols[1]:
            if st.button("📁 Start New Analysis", use_container_width=True, type="primary"):
                st.session_state.current_page = "📁 Data Upload"
                st.rerun()
        with cols[2]:
            if st.button("🏠 Return to Home", use_container_width=True):
                st.session_state.current_page = "🏠 Home"
                st.rerun()

else:
    # Coming soon page
    st.markdown(f"""
    <div style="text-align: center; padding: 4rem 2rem; margin: 2rem 0;">
        <div style="font-size: 4rem; margin-bottom: 2rem;">🛠️</div>
        <h2 style="color: #374151; margin-bottom: 1rem;">Module Under Development</h2>
        <p style="color: #6b7280; max-width: 600px; margin: 0 auto 2rem auto; font-size: 1.1rem;">
            The <strong style="color: #3b82f6;">{page}</strong> module is currently under development and will be available soon.
        </p>
        <div style="display: inline-block; background: #f3f4f6; padding: 0.75rem 1.5rem; border-radius: 12px; color: #6b7280; border: 1px solid #e5e7eb;">
            Estimated Release: Q1 2024
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.info(f"""
    **In the meantime, you can:**
    1. Start with **Data Upload** to get your data ready
    2. Clean your data in the **Data Cleaning** section
    3. Explore the **Analytics** section for insights
    4. Create **Visualizations** for better understanding
    5. Run **Scenario Analysis** for what-if scenarios
    6. Generate **Professional Reports** for stakeholders
    """)

# Professional footer
st.markdown("""
<div style="margin-top: 4rem; padding: 2rem; background: #f8fafc; border-radius: 12px; text-align: center; border-top: 1px solid #e2e8f0;">
    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 2rem; margin-bottom: 1.5rem;">
        <div style="text-align: center;">
            <div style="font-size: 1.5rem; color: #3b82f6; margin-bottom: 0.5rem;">📊</div>
            <div style="font-weight: 600; color: #1e293b; margin-bottom: 0.25rem;">FP&A Professional</div>
            <div style="font-size: 0.9rem; color: #64748b;">Enterprise v1.0</div>
        </div>
        <div style="text-align: center;">
            <div style="font-size: 1.5rem; color: #10b981; margin-bottom: 0.5rem;">🔒</div>
            <div style="font-weight: 600; color: #1e293b; margin-bottom: 0.25rem;">Data Privacy</div>
            <div style="font-size: 0.9rem; color: #64748b;">100% Offline Processing</div>
        </div>
        <div style="text-align: center;">
            <div style="font-size: 1.5rem; color: #f59e0b; margin-bottom: 0.5rem;">⚡</div>
            <div style="font-weight: 600; color: #1e293b; margin-bottom: 0.25rem;">Open Source</div>
            <div style="font-size: 0.9rem; color: #64748b;">MIT License</div>
        </div>
    </div>
    <hr style="border: none; border-top: 1px solid #e2e8f0; margin: 1.5rem 0;">
    <p style="color: #64748b; margin: 0; font-size: 0.9rem;">
        © 2024 FP&A Professional. Enterprise Financial Planning & Analysis Platform.
    </p>
</div>
""", unsafe_allow_html=True)