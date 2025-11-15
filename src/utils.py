"""
Utility functions for cosmetics brand analysis.

This module contains helper functions for data preprocessing,
visualization, and analysis tasks.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Configure plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_validate_data(file_path, required_columns=None):
    """
    Load dataset and perform basic validation.
    
    Parameters:
    -----------
    file_path : str
        Path to the dataset file
    required_columns : list, optional
        List of required column names
        
    Returns:
    --------
    pd.DataFrame
        Loaded and validated dataset
    """
    try:
        df = pd.read_csv(file_path)
        print(f"✅ Data loaded successfully. Shape: {df.shape}")
        
        if required_columns:
            missing_cols = set(required_columns) - set(df.columns)
            if missing_cols:
                print(f"⚠️ Missing required columns: {missing_cols}")
            else:
                print("✅ All required columns present")
                
        return df
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

def data_quality_report(df):
    """
    Generate comprehensive data quality report.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset to analyze
        
    Returns:
    --------
    dict
        Dictionary containing quality metrics
    """
    report = {
        'shape': df.shape,
        'memory_usage': df.memory_usage(deep=True).sum(),
        'missing_values': df.isnull().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'data_types': df.dtypes,
        'numerical_summary': df.describe(),
        'categorical_summary': df.select_dtypes(include=['object']).describe()
    }
    
    print("📊 DATA QUALITY REPORT")
    print("=" * 50)
    print(f"Dataset Shape: {report['shape']}")
    print(f"Memory Usage: {report['memory_usage'] / 1024**2:.2f} MB")
    print(f"Duplicate Rows: {report['duplicate_rows']}")
    print(f"Missing Values:\n{report['missing_values'][report['missing_values'] > 0]}")
    
    return report

def clean_text_data(text):
    """
    Clean and preprocess text data for analysis.
    
    Parameters:
    -----------
    text : str
        Raw text to clean
        
    Returns:
    --------
    str
        Cleaned text
    """
    if pd.isna(text):
        return ""
    
    # Convert to lowercase
    text = str(text).lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

def create_correlation_heatmap(df, figsize=(12, 8), title="Correlation Matrix"):
    """
    Create a professional correlation heatmap.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with numerical columns
    figsize : tuple
        Figure size (width, height)
    title : str
        Plot title
    """
    plt.figure(figsize=figsize)
    
    # Calculate correlation matrix
    corr_matrix = df.select_dtypes(include=[np.number]).corr()
    
    # Create heatmap
    sns.heatmap(corr_matrix, 
                annot=True, 
                cmap='RdYlBu_r', 
                center=0,
                square=True,
                fmt='.2f',
                cbar_kws={"shrink": 0.8})
    
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()

def plot_distribution_comparison(df, column, group_by, figsize=(12, 6)):
    """
    Create side-by-side distribution plots for comparison.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset
    column : str
        Column to plot distribution for
    group_by : str
        Column to group by
    figsize : tuple
        Figure size
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Box plot
    sns.boxplot(data=df, x=group_by, y=column, ax=axes[0])
    axes[0].set_title(f'{column} Distribution by {group_by}')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Violin plot
    sns.violinplot(data=df, x=group_by, y=column, ax=axes[1])
    axes[1].set_title(f'{column} Density by {group_by}')
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()

def generate_wordcloud(text_data, title="Word Cloud", figsize=(12, 8)):
    """
    Generate and display a word cloud from text data.
    
    Parameters:
    -----------
    text_data : list or str
        Text data to create word cloud from
    title : str
        Plot title
    figsize : tuple
        Figure size
    """
    # Combine all text
    if isinstance(text_data, list):
        text = ' '.join(text_data)
    else:
        text = str(text_data)
    
    # Create word cloud
    wordcloud = WordCloud(width=800, 
                         height=400, 
                         background_color='white',
                         colormap='viridis',
                         max_words=100).generate(text)
    
    # Plot
    plt.figure(figsize=figsize)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()

def perform_sentiment_analysis(text_series):
    """
    Perform sentiment analysis on a series of text data.
    
    Parameters:
    -----------
    text_series : pd.Series
        Series containing text data
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with sentiment scores
    """
    analyzer = SentimentIntensityAnalyzer()
    
    sentiments = []
    for text in text_series:
        if pd.notna(text):
            scores = analyzer.polarity_scores(str(text))
            sentiments.append(scores)
        else:
            sentiments.append({'neg': 0, 'neu': 0, 'pos': 0, 'compound': 0})
    
    return pd.DataFrame(sentiments)

def create_professional_barplot(df, x_col, y_col, title, figsize=(10, 6)):
    """
    Create a professional bar plot with proper formatting.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset
    x_col : str
        X-axis column
    y_col : str
        Y-axis column
    title : str
        Plot title
    figsize : tuple
        Figure size
    """
    plt.figure(figsize=figsize)
    
    # Create bar plot
    ax = sns.barplot(data=df, x=x_col, y=y_col, palette='viridis')
    
    # Customize appearance
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.xlabel(x_col.replace('_', ' ').title(), fontsize=12)
    plt.ylabel(y_col.replace('_', ' ').title(), fontsize=12)
    
    # Add value labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def save_insights_to_file(insights_dict, filename='analysis_insights.txt'):
    """
    Save analysis insights to a text file.
    
    Parameters:
    -----------
    insights_dict : dict
        Dictionary containing insights
    filename : str
        Output filename
    """
    with open(filename, 'w') as f:
        f.write("COSMETICS BRAND ANALYSIS INSIGHTS\n")
        f.write("=" * 50 + "\n\n")
        
        for section, content in insights_dict.items():
            f.write(f"{section.upper()}\n")
            f.write("-" * len(section) + "\n")
            f.write(f"{content}\n\n")
    
    print(f"✅ Insights saved to {filename}")

def get_column_mapping(df):
    """
    Get flexible column name mapping for common variations.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset to analyze
        
    Returns:
    --------
    dict
        Dictionary mapping standard names to actual column names
    """
    columns = df.columns.tolist()
    mapping = {}
    
    # NPS column variations
    nps_variants = ['NPS', 'NPS_Score', 'nps', 'nps_score', 'Net_Promoter_Score']
    for variant in nps_variants:
        if variant in columns:
            mapping['NPS_Score'] = variant
            break
    
    # Brand preference variations
    brand_variants = ['Brand_Preferences', 'Brand_Preference', 'brand_preferences', 
                     'brand_preference', 'Preferred_Brand', 'Brand']
    for variant in brand_variants:
        if variant in columns:
            mapping['Brand_Preference'] = variant
            break
    
    # Text feedback variations
    text_variants = ['Open_Feedback', 'open_feedback', 'Feedback', 'feedback', 
                    'Customer_Feedback', 'Text_Feedback', 'Comments']
    for variant in text_variants:
        if variant in columns:
            mapping['Open_Feedback'] = variant
            break
    
    # Customer ID variations
    id_variants = ['Customer_ID', 'customer_id', 'ID', 'id', 'CustomerID']
    for variant in id_variants:
        if variant in columns:
            mapping['Customer_ID'] = variant
            break
    
    return mapping

def normalize_brand_preferences(df, brand_col):
    """
    Normalize brand preferences column (handles comma-separated values).
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset
    brand_col : str
        Name of brand preferences column
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with normalized brand column
    """
    df = df.copy()
    
    if brand_col not in df.columns:
        return df
    
    # Create a normalized version (take first brand if multiple)
    df['Brand_Preference_Normalized'] = df[brand_col].apply(
        lambda x: str(x).split(',')[0].strip() if pd.notna(x) else 'Unknown'
    )
    
    return df

# Configuration constants
PLOT_STYLE = {
    'figure.figsize': (12, 8),
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10
}

# Color palettes for different types of plots
COLOR_PALETTES = {
    'categorical': 'Set2',
    'sequential': 'viridis',
    'diverging': 'RdYlBu_r',
    'brand_colors': ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
}