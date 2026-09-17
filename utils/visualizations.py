import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

def plot_numerical_distribution(df: pd.DataFrame, column_name: str) -> go.Figure:
    """
    Generates an interactive Plotly histogram with an overlaid box plot 
    for a selected numerical column.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The input pandas DataFrame.
    column_name : str
        The specific numerical column to visualize.
        
    Returns:
    --------
    go.Figure
        A Plotly figure object, or None if the column is invalid.
    """
    if column_name not in df.columns:
        return None
        
    # Drop missing values for clean distribution plotting
    clean_data = df[column_name].dropna()
    
    fig = px.histogram(
        clean_data, 
        x=column_name, 
        marginal="box", # Adds a compact box plot on top to visualize outliers easily
        title=f"Distribution of {column_name}",
        template="plotly_white",
        color_discrete_sequence=["#4F46E5"] # Clean indigo accent
    )
    
    fig.update_layout(
        xaxis_title=column_name,
        yaxis_title="Frequency",
        bargap=0.05,
        title_font=dict(size=16, family="sans-serif")
    )
    
    return fig


def plot_categorical_distribution(df: pd.DataFrame, column_name: str, top_n: int = 10) -> go.Figure:
    """
    Generates an interactive Plotly bar chart displaying the top frequent categories 
    for a selected categorical column.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The input pandas DataFrame.
    column_name : str
        The specific categorical column to visualize.
    top_n : int
        The maximum number of top categories to display (default: 10).
        
    Returns:
    --------
    go.Figure
        A Plotly figure object, or None if the column is invalid.
    """
    if column_name not in df.columns:
        return None
        
    # Calculate value counts and take top N
    counts = df[column_name].value_counts().head(top_n).reset_index()
    counts.columns = ['Category', 'Count']
    
    fig = px.bar(
        counts,
        x='Category',
        y='Count',
        title=f"Top Categories in {column_name}",
        template="plotly_white",
        color='Count',
        color_continuous_scale="Viridis"
    )
    
    fig.update_layout(
        xaxis_title=column_name,
        yaxis_title="Count",
        xaxis={'categoryorder':'total descending'},
        title_font=dict(size=16, family="sans-serif")
    )
    
    return fig


def plot_correlation_matrix(df: pd.DataFrame, numerical_cols: list) -> go.Figure:
    """
    Generates a Plotly heatmap for the correlation matrix of numerical columns.
    Only executes if there are at least 2 numerical columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The input pandas DataFrame.
    numerical_cols : list
        List of numerical column names.
        
    Returns:
    --------
    go.Figure
        A Plotly heatmap figure, or None if insufficient numerical columns.
    """
    if len(numerical_cols) < 2:
        return None
        
    # Compute correlation safely
    corr_matrix = df[numerical_cols].corr(numeric_only=True)
    
    if corr_matrix.isna().all().all():
        return None
        
    fig = px.imshow(
        corr_matrix,
        text_auto=".2f", # Display correlation values up to 2 decimal places
        aspect="auto",
        color_continuous_scale="RdBu_r", # Red-Blue diverging scale standard for correlations
        range_color=[-1, 1],
        title="Numerical Feature Correlation Matrix",
        template="plotly_white"
    )
    
    fig.update_layout(
        title_font=dict(size=16, family="sans-serif")
    )
    
    return fig


def plot_missing_values(df: pd.DataFrame) -> go.Figure:
    """
    Generates a clean bar chart showing missing value counts and percentages 
    across all columns that contain missing data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The input pandas DataFrame.
        
    Returns:
    --------
    go.Figure
        A Plotly figure object, or a blank message figure if no missing values exist.
    """
    missing_counts = df.isnull().sum()
    missing_counts = missing_counts[missing_counts > 0].reset_index()
    
    if missing_counts.empty:
        # Return an empty figure with an annotation if there are no missing values
        fig = go.Figure()
        fig.add_annotation(
            text="No missing values detected in this dataset! 🎉",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="green")
        )
        fig.update_layout(template="plotly_white", title="Missing Values Overview")
        return fig
        
    missing_counts.columns = ['Column', 'Missing Count']
    missing_counts['Missing Percentage (%)'] = (missing_counts['Missing Count'] / len(df) * 100).round(2)
    
    fig = px.bar(
        missing_counts,
        x='Column',
        y='Missing Count',
        text='Missing Percentage (%)',
        title="Missing Values Count per Column",
        template="plotly_white",
        color_discrete_sequence=["#EF4444"] # Soft red warning accent
    )
    
    fig.update_traces(texttemplate='%{text}%', textposition='outside')
    fig.update_layout(
        xaxis_title="Columns",
        yaxis_title="Number of Missing Rows",
        title_font=dict(size=16, family="sans-serif")
    )
    
    return fig