import streamlit as st
import pandas as pd
import numpy as np

# Import custom utilities from the utils package
from utils.data_analyzer import analyze_dataset
from utils.visualizations import (
    plot_numerical_distribution,
    plot_categorical_distribution,
    plot_correlation_matrix,
    plot_missing_values
)
from utils.ai_service import (
    generate_dataset_summary,
    generate_insights,
    generate_analysis_questions,
    answer_data_query
)

# 1. Page Configuration
st.set_page_config(
    page_title="DataLens AI – Intelligent Dataset Assistant",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Theme-Adaptive Modern CSS (Supports Light & Dark Modes seamlessly)
st.markdown("""
    <style>
    .datalens-header {
        padding: 1.5rem 0 0.5rem 0;
        border-bottom: 2px solid rgba(128, 128, 128, 0.2);
        margin-bottom: 2rem;
    }
    .main-title {
        font-size: 2.25rem;
        font-weight: 800;
        letter-spacing: -0.025em;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        font-weight: 400;
        opacity: 0.8;
    }
    h3 {
        font-weight: 700 !important;
        font-size: 1.25rem !important;
        margin-top: 1rem !important;
        margin-bottom: 0.75rem !important;
    }
    .ai-card {
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 0.75rem;
        padding: 1.25rem;
        height: 100%;
    }
    .landing-box {
        border: 1px dashed rgba(128, 128, 128, 0.4);
        padding: 3rem;
        border-radius: 1rem;
        text-align: center;
        margin-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# 3. Application Header Section
st.markdown("""
    <div class="datalens-header">
        <p class="main-title">🔍 DataLens AI</p>
        <p class="sub-title">Intelligent Dataset Understanding, Statistical Profiling & AI Assistant</p>
    </div>
""", unsafe_allow_html=True)

# Robust CSV Loading with Encoding & Delimiter Fallbacks
@st.cache_data(show_spinner=False)
def load_csv_data(file):
    try:
        return pd.read_csv(file)
    except UnicodeDecodeError:
        try:
            file.seek(0)
            return pd.read_csv(file, encoding='latin-1')
        except Exception as e:
            raise ValueError(f"Error reading CSV file with alternative encoding: {str(e)}")
    except Exception as e:
        try:
            file.seek(0)
            return pd.read_csv(file, sep=None, engine='python')
        except Exception:
            raise ValueError(f"Error reading CSV file: {str(e)}")

# 4. Sidebar: File Upload, Controls & CSV Export
with st.sidebar:
    st.markdown("### 📁 Dataset Control")
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"], help="Select any tabular CSV dataset to begin analysis.")
    
    cleaned_df = None
    if uploaded_file is not None:
        try:
            raw_df = load_csv_data(uploaded_file)
            if not raw_df.empty:
                cleaned_df = raw_df.dropna(how='all').dropna(axis=1, how='all')
                
                st.markdown("---")
                st.markdown("### 📥 Export Options")
                csv_bytes = cleaned_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Cleaned CSV",
                    data=csv_bytes,
                    file_name="datalens_cleaned_dataset.csv",
                    mime="text/csv",
                    help="Download a cleaned version of the dataset with empty rows and columns removed."
                )
        except Exception:
            pass

    st.markdown("---")
    st.markdown("### 💡 About DataLens AI")
    st.markdown(
        "DataLens AI bridges the gap between raw data and actionable intelligence by combining "
        "instant Pandas profiling, interactive Plotly visualizations, and Google Gemini-powered insights."
    )
    st.markdown("---")
    st.caption("DataLens AI • Intelligent Analytics Assistant")

if uploaded_file is not None:
    try:
        with st.spinner("Analyzing dataset structure..."):
            df = load_csv_data(uploaded_file)
            
        if df.empty:
            st.error("The uploaded CSV file is empty. Please upload a valid dataset.")
            st.stop()
            
        analysis = analyze_dataset(df)

        # Compute Automated Data Health Score (0-100)
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = int(df.isnull().sum().sum())
        duplicate_rows = int(analysis['duplicate_count'])
        
        missing_penalty = (missing_cells / total_cells) * 40 if total_cells > 0 else 0
        duplicate_penalty = min(30, (duplicate_rows / max(1, df.shape[0])) * 30)
        health_score = max(10, int(100 - missing_penalty - duplicate_penalty))

        st.sidebar.success("Dataset loaded successfully!")
        col_s1, col_s2 = st.sidebar.columns(2)
        col_s1.metric("Rows", f"{analysis['row_count']:,}")
        col_s2.metric("Columns", f"{analysis['col_count']:,}")

        with st.expander("👀 Quick Dataset Preview (First 5 Rows)", expanded=False):
            st.dataframe(df.head(5), use_container_width=True)

        st.markdown("")

        # 5. Main Dashboard Navigation Tabs
        tab_overview, tab_explore, tab_visuals, tab_ai = st.tabs([
            "📊 Overview & Health", 
            "🔍 Column Explorer", 
            "📈 Visual Analytics", 
            "🤖 Gemini AI Assistant"
        ])

        # --- TAB 1: OVERVIEW & HEALTH SCORE ---
        with tab_overview:
            st.markdown("### 🏥 Automated Data Health Score")
            hcol1, hcol2, hcol3 = st.columns([1, 2, 1])
            with hcol1:
                st.metric("Health Score", f"{health_score}/100", delta="Quality Index")
            with hcol2:
                if health_score >= 80:
                    st.success("🟢 **High Data Quality:** Clean dataset with minimal missing values or duplication.")
                elif health_score >= 50:
                    st.warning("🟡 **Moderate Data Quality:** Contains noticeable missing values or duplicates requiring attention.")
                else:
                    st.error("🔴 **Low Data Quality:** High ratio of missing cells or redundant rows detected.")
            
            st.markdown("<div style='margin: 1rem 0;'></div>", unsafe_allow_html=True)
            st.markdown("### 📊 High-Level Dataset Structure")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Rows", f"{analysis['row_count']:,}")
            with col2:
                st.metric("Total Columns", f"{analysis['col_count']:,}")
            with col3:
                st.metric("Numerical Columns", len(analysis['numerical_cols']))
            with col4:
                st.metric("Categorical Columns", len(analysis['categorical_cols']))

            st.markdown("<div style='margin: 1.5rem 0;'></div>", unsafe_allow_html=True)
            
            st.markdown("### 🛡️ Data Quality Audit")
            qcol1, qcol2, qcol3 = st.columns(3)
            with qcol1:
                st.metric("Total Missing Cells", f"{missing_cells:,}")
            with qcol2:
                missing_cols_count = len(analysis['missing_summary'][analysis['missing_summary']['Missing Count'] > 0])
                st.metric("Affected Columns", missing_cols_count)
            with qcol3:
                st.metric("Duplicate Rows", f"{duplicate_rows:,}")

            if not analysis['missing_summary'].empty and missing_cells > 0:
                with st.expander("📋 View Detailed Missing Values Breakdown Table", expanded=False):
                    st.dataframe(analysis['missing_summary'], use_container_width=True)

        # --- TAB 2: COLUMN EXPLORER & STATS ---
        with tab_explore:
            st.markdown("### 🔍 Individual Column Deep Dive")
            selected_col = st.selectbox("Select a column to inspect properties:", analysis['columns'])
            
            if selected_col:
                col_type = analysis['dtypes'][selected_col]
                col_unique = analysis['unique_counts'][selected_col]
                col_missing = df[selected_col].isnull().sum()
                col_missing_pct = round((col_missing / analysis['row_count']) * 100, 2)

                ec1, ec2, ec3, ec4 = st.columns(4)
                ec1.metric("Data Type", col_type)
                ec2.metric("Unique Values", f"{col_unique:,}")
                ec3.metric("Missing Values", f"{col_missing:,}")
                ec4.metric("Missing Rate", f"{col_missing_pct}%")

            st.markdown("<div style='margin: 1.5rem 0;'></div>", unsafe_allow_html=True)
            st.markdown("### 📈 Statistical Profiles")
            
            if len(analysis['numerical_cols']) > 0:
                st.markdown("**Numerical Descriptive Statistics**")
                st.dataframe(pd.DataFrame(analysis['numerical_stats']), use_container_width=True)
            else:
                st.info("No numerical columns found in this dataset for statistical descriptions.")

            if len(analysis['categorical_cols']) > 0:
                with st.expander("🏷️ Top Value Frequencies for Categorical Columns", expanded=False):
                    for cat_col, top_vals in analysis['categorical_top_values'].items():
                        st.markdown(f"**{cat_col}**")
                        st.json(top_vals)

        # --- TAB 3: VISUAL ANALYTICS ---
        with tab_visuals:
            st.markdown("### 📈 Interactive Visual Analytics")
            
            vis_subtab1, vis_subtab2, vis_subtab3, vis_subtab4 = st.tabs([
                "Numerical Distributions", 
                "Categorical Distributions", 
                "Correlation Matrix", 
                "Missing Values Chart"
            ])

            with vis_subtab1:
                if len(analysis['numerical_cols']) > 0:
                    num_target = st.selectbox("Select numerical column:", analysis['numerical_cols'], key="num_hist")
                    fig_num = plot_numerical_distribution(df, num_target)
                    if fig_num:
                        st.plotly_chart(fig_num, use_container_width=True)
                else:
                    st.info("No numerical columns available for distribution plots.")

            with vis_subtab2:
                if len(analysis['categorical_cols']) > 0:
                    cat_target = st.selectbox("Select categorical column:", analysis['categorical_cols'], key="cat_bar")
                    fig_cat = plot_categorical_distribution(df, cat_target)
                    if fig_cat:
                        st.plotly_chart(fig_cat, use_container_width=True)
                else:
                    st.info("No categorical columns available for distribution plots.")

            with vis_subtab3:
                if len(analysis['numerical_cols']) >= 2:
                    fig_corr = plot_correlation_matrix(df, analysis['numerical_cols'])
                    if fig_corr:
                        st.plotly_chart(fig_corr, use_container_width=True)
                    else:
                        st.warning("Could not compute correlation matrix (insufficient variance or valid numeric pairs).")
                else:
                    st.info("At least two numerical columns are required to generate a correlation matrix.")

            with vis_subtab4:
                fig_missing = plot_missing_values(df)
                if fig_missing:
                    st.plotly_chart(fig_missing, use_container_width=True)

        # --- TAB 4: AI ASSISTANT & INSIGHTS ---
        with tab_ai:
            st.markdown("### 🤖 Google Gemini AI Assistant")
            st.markdown("Translate statistical calculations into intelligent narratives or ask custom questions about your dataset.")
            st.markdown("<div style='margin: 1rem 0;'></div>", unsafe_allow_html=True)

            ai_col1, ai_col2, ai_col3 = st.columns(3)

            with ai_col1:
                st.markdown("""
                    <div class="ai-card">
                        <h4>1. Executive Summary</h4>
                        <p style="opacity: 0.8; font-size: 0.9rem;">Understand what the dataset represents and its core schema.</p>
                    </div>
                """, unsafe_allow_html=True)
                if st.button("Generate Explanation", type="primary", key="btn_exp", use_container_width=True):
                    with st.spinner("Gemini is analyzing structure..."):
                        explanation = generate_dataset_summary(analysis)
                        st.markdown("---")
                        st.markdown("### Explanation Results")
                        st.markdown(explanation)

            with ai_col2:
                st.markdown("""
                    <div class="ai-card">
                        <h4>2. Pattern Insights</h4>
                        <p style="opacity: 0.8; font-size: 0.9rem;">Identify distributions, outliers, and notable data patterns.</p>
                    </div>
                """, unsafe_allow_html=True)
                if st.button("Extract Insights", type="primary", key="btn_ins", use_container_width=True):
                    with st.spinner("Gemini is mining insights..."):
                        insights = generate_insights(analysis)
                        st.markdown("---")
                        st.markdown("### Insight Results")
                        st.markdown(insights)

            with ai_col3:
                st.markdown("""
                    <div class="ai-card">
                        <h4>3. Analysis Questions</h4>
                        <p style="opacity: 0.8; font-size: 0.9rem;">Get structured questions categorized by skill level.</p>
                    </div>
                """, unsafe_allow_html=True)
                if st.button("Formulate Questions", type="primary", key="btn_q", use_container_width=True):
                    with st.spinner("Gemini is building questions..."):
                        questions = generate_analysis_questions(analysis)
                        st.markdown("---")
                        st.markdown("### Suggested Questions")
                        st.markdown(questions)

            st.markdown("<div style='margin: 2rem 0 1rem 0;'></div>", unsafe_allow_html=True)
            st.markdown("### 💬 Ask Your Data (Interactive AI Chat)")
            st.markdown("Type any question in plain English about your uploaded dataset:")
            
            user_query = st.text_input("e.g., Which column has the most missing values or what patterns stand out?", placeholder="Ask a question about this dataset...")
            if user_query:
                with st.spinner("Gemini is analyzing your question against the dataset profile..."):
                    chat_response = answer_data_query(analysis, user_query)
                    st.markdown("**AI Answer:**")
                    st.info(chat_response)

    except Exception as e:
        st.error(f"An error occurred while processing the dataset: {str(e)}")
else:
    st.markdown("""
        <div class="landing-box">
            <h3 style="margin-bottom: 0.5rem;">No Dataset Uploaded Yet</h3>
            <p style="opacity: 0.8; max-width: 500px; margin: 0 auto 1.5rem auto;">
                Please upload a CSV file using the sidebar panel to launch your automated exploratory data analysis and AI assistant.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    with st.expander("ℹ️ How to use DataLens AI"):
        st.markdown("""
        1. **Upload your CSV file** via the sidebar uploader.
        2. **View the Automated Data Health Score** on the Overview tab to assess dataset quality instantly.
        3. **Export Cleaned Data** using the one-click download button in the sidebar.
        4. **Ask Custom Questions** in plain English using the interactive AI chat bar at the bottom of the AI Assistant tab.
        """)
