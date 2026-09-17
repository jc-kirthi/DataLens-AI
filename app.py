import streamlit as st
import pandas as pd
import numpy as np

# Import custom utilities (assuming they are placed in a 'utils' folder or root)
try:
    from utils.data_analyzer import analyze_dataset
    from utils.visualizations import (
        plot_numerical_distribution,
        plot_categorical_distribution,
        plot_correlation_matrix,
        plot_missing_values
    )
    from utils.gemini_service import (
        generate_dataset_summary,
        generate_insights,
        generate_analysis_questions
    )
except ImportError:
    # Fallback if modules are in the same root directory
    from data_analyzer import analyze_dataset
    from visualizations import (
        plot_numerical_distribution,
        plot_categorical_distribution,
        plot_correlation_matrix,
        plot_missing_values
    )
    from gemini_service import (
        generate_dataset_summary,
        generate_insights,
        generate_analysis_questions
    )

# Page Configuration
st.set_page_config(
    page_title="DataLens AI – Dataset Understanding Assistant",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for Clean Presentation
st.markdown("""
    <style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0rem;
    }
    .sub-title {
        font-size: 1.2rem;
        color: #4B5563;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4F46E5;
    }
    </style>
""", unsafe_allow_html=True)

# 1. Application Header
st.markdown('<p class="main-title">DataLens AI</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Intelligent Dataset Understanding & Analysis Assistant</p>', unsafe_allow_html=True)

# Caching CSV Loading for Performance
@st.cache_data(show_spinner=False)
def load_csv_data(file):
    try:
        return pd.read_csv(file)
    except Exception as e:
        raise ValueError(f"Error reading CSV file: {str(e)}")

# Sidebar: CSV Upload & Controls
st.sidebar.header("📁 Dataset Upload")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        with st.spinner("Loading and validating dataset..."):
            df = load_csv_data(uploaded_file)
            
        # 3. Validate that it is not empty
        if df.empty:
            st.error("The uploaded CSV file is empty. Please upload a valid dataset.")
            st.stop()
            
        # Run deterministic analysis utility (cached)
        @st.cache_data
        def run_analysis(dataframe_bytes):
            # Using dataframe hash or converting to helper for caching if needed, 
            # but since df is loaded, analyze directly:
            return analyze_dataset(df)
            
        analysis = analyze_dataset(df)

        # Sidebar Quick Info
        st.sidebar.success("Dataset loaded successfully!")
        st.sidebar.markdown(f"**Rows:** {analysis['row_count']:,}")
        st.sidebar.markdown(f"**Columns:** {analysis['col_count']:,}")

        # 3. Display a small dataset preview
        with st.expander("👀 Dataset Preview (First 5 Rows)", expanded=False):
            st.dataframe(df.head(5), use_container_width=True)

        st.markdown("---")

        # Organize Dashboard into Tabs for a Clean Interface
        tab_overview, tab_explore, tab_visuals, tab_ai = st.tabs([
            "📊 Overview & Quality", 
            "🔍 Column Explorer & Stats", 
            "📈 Visual Analytics", 
            "🤖 AI Assistant & Insights"
        ])

        # --- TAB 1: OVERVIEW & QUALITY ---
        with tab_overview:
            st.subheader("A. Dataset Overview")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Rows", f"{analysis['row_count']:,}")
            with col2:
                st.metric("Total Columns", f"{analysis['col_count']:,}")
            with col3:
                st.metric("Numerical Columns", len(analysis['numerical_cols']))
            with col4:
                st.metric("Categorical Columns", len(analysis['categorical_cols']))

            st.markdown("---")
            
            st.subheader("B. Data Quality")
            qcol1, qcol2, qcol3 = st.columns(3)
            total_missing = int(df.isnull().sum().sum())
            with qcol1:
                st.metric("Total Missing Values", f"{total_missing:,}")
            with qcol2:
                missing_cols_count = len(analysis['missing_summary'][analysis['missing_summary']['Missing Count'] > 0])
                st.metric("Columns with Missing Data", missing_cols_count)
            with qcol3:
                st.metric("Duplicate Rows", f"{analysis['duplicate_count']:,}")

            if not analysis['missing_summary'].empty and total_missing > 0:
                with st.expander("View Missing Values Breakdown Table"):
                    st.dataframe(analysis['missing_summary'], use_container_width=True)

        # --- TAB 2: COLUMN EXPLORER & STATS ---
        with tab_explore:
            st.subheader("C. Column Explorer")
            selected_col = st.selectbox("Select a column to inspect:", analysis['columns'])
            
            if selected_col:
                col_type = analysis['dtypes'][selected_col]
                col_unique = analysis['unique_counts'][selected_col]
                col_missing = df[selected_col].isnull().sum()
                col_missing_pct = round((col_missing / analysis['row_count']) * 100, 2)

                ec1, ec2, ec3, ec4 = st.columns(4)
                ec1.metric("Data Type", col_type)
                ec2.metric("Unique Values", f"{col_unique:,}")
                ec3.metric("Missing Values", f"{col_missing:,}")
                ec4.metric("Missing Percentage", f"{col_missing_pct}%")

            st.markdown("---")
            st.subheader("D. Statistical Summary")
            if len(analysis['numerical_cols']) > 0:
                st.markdown("**Numerical Descriptive Statistics**")
                st.dataframe(pd.DataFrame(analysis['numerical_stats']), use_container_width=True)
            else:
                st.info("No numerical columns found in this dataset for descriptive statistics.")

            if len(analysis['categorical_cols']) > 0:
                with st.expander("Top Value Frequencies for Categorical Columns"):
                    for cat_col, top_vals in analysis['categorical_top_values'].items():
                        st.markdown(f"**{cat_col}** (Top Frequencies)")
                        st.json(top_vals)

        # --- TAB 3: VISUAL ANALYTICS ---
        with tab_visuals:
            st.subheader("E. Visual Analytics")
            
            vis_subtab1, vis_subtab2, vis_subtab3, vis_subtab4 = st.tabs([
                "Numerical Distributions", 
                "Categorical Distributions", 
                "Correlation Matrix", 
                "Missing Values Chart"
            ])

            with vis_subtab1:
                if len(analysis['numerical_cols']) > 0:
                    num_target = st.selectbox("Choose numerical column for histogram:", analysis['numerical_cols'], key="num_hist")
                    fig_num = plot_numerical_distribution(df, num_target)
                    if fig_num:
                        st.plotly_chart(fig_num, use_container_width=True)
                else:
                    st.info("No numerical columns available for distribution plots.")

            with vis_subtab2:
                if len(analysis['categorical_cols']) > 0:
                    cat_target = st.selectbox("Choose categorical column for bar chart:", analysis['categorical_cols'], key="cat_bar")
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
            st.subheader("🤖 Gemini AI Assistant")
            st.markdown("Leverage Google Gemini to translate deterministic calculations into intelligent natural-language narratives.")

            ai_col1, ai_col2, ai_col3 = st.columns(3)

            # F. AI Dataset Explanation
            with ai_col1:
                st.markdown("#### 1. Dataset Explanation")
                st.markdown("Get an executive overview of what this dataset represents and its key columns.")
                if st.button("Generate AI Explanation", type="primary", key="btn_exp"):
                    with st.spinner("Gemini is analyzing dataset structure..."):
                        explanation = generate_dataset_summary(analysis)
                        st.markdown("### Explanation Results")
                        st.markdown(explanation)

            # G. AI Insights
            with ai_col2:
                st.markdown("#### 2. Pattern Insights")
                st.markdown("Identify notable anomalies, distributions, and value patterns from statistics.")
                if st.button("Generate Insights", type="primary", key="btn_ins"):
                    with st.spinner("Gemini is extracting insights..."):
                        insights = generate_insights(analysis)
                        st.markdown("### Insight Results")
                        st.markdown(insights)

            # H. Suggested Analysis Questions
            with ai_col3:
                st.markdown("#### 3. Analysis Questions")
                st.markdown("Generate structured questions categorized into Beginner, Intermediate, and Advanced.")
                if st.button("Generate Questions", type="primary", key="btn_q"):
                    with st.spinner("Gemini is formulating questions..."):
                        questions = generate_analysis_questions(analysis)
                        st.markdown("### Suggested Questions")
                        st.markdown(questions)

    except Exception as e:
        st.error(f"An error occurred while processing the dataset: {str(e)}")
else:
    # Landing state when no file is uploaded yet
    st.info("👈 Please upload a CSV dataset using the sidebar to begin your exploratory analysis.")
    
    with st.expander("ℹ️ How to use DataLens AI"):
        st.markdown("""
        1. **Upload any CSV dataset** via the sidebar uploader.
        2. Explore **Dataset Overview**, **Data Quality**, and **Column Statistics** computed locally by Pandas.
        3. Visualize distributions and correlations instantly using interactive **Plotly** charts.
        4. Click the **AI Assistant** buttons to generate beginner-friendly explanations, data insights, and structured analysis questions powered by **Google Gemini**.
        """)