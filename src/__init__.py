"""
Cosmetics Brand Analysis Package

This package contains utility functions and modules for analyzing
cosmetics brand consumer data and generating marketing insights.
"""

__version__ = "1.0.0"
__author__ = "Data Science Team"

from .utils import (
    load_and_validate_data,
    data_quality_report,
    clean_text_data,
    create_correlation_heatmap,
    plot_distribution_comparison,
    generate_wordcloud,
    perform_sentiment_analysis,
    create_professional_barplot,
    save_insights_to_file
)