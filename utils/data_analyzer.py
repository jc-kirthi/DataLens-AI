import pandas as pd
import numpy as np

def analyze_dataset(df: pd.DataFrame) -> dict:
    """
    Performs comprehensive, deterministic exploratory data analysis on a pandas DataFrame.
    Designed to be beginner-friendly, modular, and safe against edge cases 
    (e.g., empty datasets, datasets with only numerical/categorical columns, 
    missing values, and invalid correlations).
    
    Parameters:
    -----------
    df : pd.DataFrame
        The input pandas DataFrame loaded from an arbitrary CSV file.
        
    Returns:
    --------
    dict
        A structured dictionary containing all requested metrics, summaries, 
        and dataframes ready for consumption by Streamlit.
    """
    
    # Handle empty or None dataset edge case safely
    if df is None or df.empty:
        return {
            "is_empty": True,
            "row_count": 0,
            "col_count": 0,
            "columns": [],
            "message": "The uploaded dataset is empty or invalid."
        }

    # 1 & 2. Number of rows and columns
    row_count = int(df.shape[0])
    col_count = int(df.shape[1])

    # 3. Column names
    columns = list(df.columns)

    # 4. Data types (converted to string format for reliable display)
    dtypes = df.dtypes.astype(str).to_dict()

    # 5. Missing value count and percentage for every column
    missing_count = df.isnull().sum()
    missing_percentage = (missing_count / row_count * 100).round(2) if row_count > 0 else pd.Series(0, index=columns)
    
    missing_summary = pd.DataFrame({
        'Missing Count': missing_count,
        'Missing Percentage (%)': missing_percentage
    })

    # 6. Duplicate row count
    duplicate_count = int(df.duplicated().sum())

    # 7. Numerical columns identification
    numerical_cols = list(df.select_dtypes(include=[np.number]).columns)

    # 8. Categorical columns identification (includes object, category, and boolean types)
    categorical_cols = list(df.select_dtypes(include=['object', 'category', 'bool']).columns)

    # 9. Unique value counts across all columns
    unique_counts = df.nunique().to_dict()

    # 10. Descriptive statistics for numerical columns
    numerical_stats = {}
    if len(numerical_cols) > 0:
        # Generates count, mean, std, min, 25%, 50%, 75%, max
        numerical_stats = df[numerical_cols].describe().to_dict()

    # 11. Top value frequencies for categorical columns
    categorical_top_values = {}
    for col in categorical_cols:
        # Extracts top 5 most frequent values and their corresponding frequencies
        value_counts_dict = df[col].value_counts().head(5).to_dict()
        categorical_top_values[col] = value_counts_dict

    # 12. Correlation information for numerical columns where applicable
    correlation_matrix = None
    if len(numerical_cols) >= 2:
        try:
            # Compute correlation matrix, gracefully handling zero-variance or constant columns
            corr = df[numerical_cols].corr(numeric_only=True)
            if not corr.isna().all().all():
                correlation_matrix = corr
        except Exception:
            # Prevent crashes if correlation cannot be calculated
            correlation_matrix = None

    # Bundle all computed metrics into a structured dictionary for the Streamlit UI
    analysis_results = {
        "is_empty": False,
        "row_count": row_count,
        "col_count": col_count,
        "columns": columns,
        "dtypes": dtypes,
        "missing_summary": missing_summary,
        "duplicate_count": duplicate_count,
        "numerical_cols": numerical_cols,
        "categorical_cols": categorical_cols,
        "unique_counts": unique_counts,
        "numerical_stats": numerical_stats,
        "categorical_top_values": categorical_top_values,
        "correlation_matrix": correlation_matrix
    }

    return analysis_results