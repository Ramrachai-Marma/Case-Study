"""
Cosmetics Brand Analysis Package

This package contains modules for comprehensive analysis of cosmetics brand data:
- Data preparation and quality assessment
- Exploratory data analysis 
- Net Promoter Score analysis
- Text analysis and sentiment mining
- Recommendations generation
"""

__version__ = "1.0.0"
__author__ = "Data Science Team"

from .data_preparation import DataPreparation
from .exploratory_data_analysis import ExploratoryDataAnalysis
from .nps_analysis import NPSAnalysis
from .text_analysis import TextAnalysis
from .recommendations import RecommendationEngine

__all__ = [
    'DataPreparation',
    'ExploratoryDataAnalysis', 
    'NPSAnalysis',
    'TextAnalysis',
    'RecommendationEngine'
]