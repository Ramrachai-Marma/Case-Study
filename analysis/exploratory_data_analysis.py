"""
Exploratory Data Analysis Module for Cosmetics Brand Analysis

This module handles comprehensive exploratory data analysis including:
- Demographic analysis
- Brand preference patterns
- Statistical relationships
- Interactive visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.utils import get_column_mapping, normalize_brand_preferences
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ExploratoryDataAnalysis:
    """
    Class for comprehensive exploratory data analysis of cosmetics brand data.
    """
    
    def __init__(self, dataframe):
        """
        Initialize EDA class with dataset.
        
        Parameters:
        -----------
        dataframe : pd.DataFrame
            The dataset to analyze
        """
        self.df = dataframe.copy()
        self.insights = {}
        
        # Get column mappings for flexible column name handling
        self.column_map = get_column_mapping(self.df)
        
        # Normalize brand preferences if needed
        if 'Brand_Preference' in self.column_map:
            brand_col = self.column_map['Brand_Preference']
            self.df = normalize_brand_preferences(self.df, brand_col)
            # Use normalized column for analysis
            if 'Brand_Preference_Normalized' in self.df.columns:
                self.df['Brand_Preference'] = self.df['Brand_Preference_Normalized']
        
    def demographic_analysis(self, save_plots=True):
        """
        Analyze demographic patterns and distributions.
        
        Parameters:
        -----------
        save_plots : bool
            Whether to save plots to images folder
        """
        print("👥 DEMOGRAPHIC ANALYSIS")
        print("=" * 30)
        
        # Gender distribution
        if 'Gender' in self.df.columns:
            print("\n🚻 GENDER DISTRIBUTION:")
            gender_counts = self.df['Gender'].value_counts()
            print(gender_counts)
            print(f"Gender distribution: {(gender_counts / len(self.df) * 100).round(1).to_dict()}")
            
            # Visualize gender distribution
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # Pie chart
            axes[0].pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%')
            axes[0].set_title('Gender Distribution')
            
            # Bar chart
            gender_counts.plot(kind='bar', ax=axes[1], color=['skyblue', 'lightcoral'])
            axes[1].set_title('Gender Count')
            axes[1].set_ylabel('Count')
            axes[1].tick_params(axis='x', rotation=0)
            
            plt.tight_layout()
            if save_plots:
                plt.savefig('images/gender_distribution.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        # Age analysis
        if 'Age' in self.df.columns:
            print(f"\n👶 AGE ANALYSIS:")
            age_stats = self.df['Age'].describe()
            print(age_stats)
            
            # Age distribution plot
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            plt.hist(self.df['Age'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            plt.title('Age Distribution')
            plt.xlabel('Age')
            plt.ylabel('Frequency')
            
            plt.subplot(1, 2, 2)
            sns.boxplot(y=self.df['Age'])
            plt.title('Age Box Plot')
            
            plt.tight_layout()
            if save_plots:
                plt.savefig('images/age_distribution.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        # Income Level analysis
        if 'Income_Level' in self.df.columns:
            print(f"\n💰 INCOME LEVEL DISTRIBUTION:")
            income_counts = self.df['Income_Level'].value_counts()
            print(income_counts)
            
            plt.figure(figsize=(10, 6))
            income_counts.plot(kind='bar', color='lightgreen')
            plt.title('Income Level Distribution')
            plt.xlabel('Income Level')
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            plt.tight_layout()
            if save_plots:
                plt.savefig('images/income_distribution.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    def brand_preference_analysis(self, save_plots=True):
        """
        Analyze brand preference patterns across demographics.
        
        Parameters:
        -----------
        save_plots : bool
            Whether to save plots to images folder
        """
        print("\n🏷️ BRAND PREFERENCE ANALYSIS")
        print("=" * 35)
        
        # Check for brand preference column (handle variations)
        brand_col = self.column_map.get('Brand_Preference', 'Brand_Preference')
        if brand_col not in self.df.columns and 'Brand_Preference' not in self.df.columns:
            print("❌ Brand preference column not found")
            print(f"Available columns: {list(self.df.columns)}")
            return
        
        # Use normalized brand preference if available, otherwise use original
        if 'Brand_Preference' in self.df.columns:
            brand_col = 'Brand_Preference'
        
        # Overall brand preference
        brand_counts = self.df[brand_col].value_counts()
        print(f"\n📊 OVERALL BRAND PREFERENCES:")
        print(brand_counts)
        
        # Brand preference by gender
        if 'Gender' in self.df.columns:
            print(f"\n🚻 BRAND PREFERENCE BY GENDER:")
            brand_gender_crosstab = pd.crosstab(self.df[brand_col], self.df['Gender'])
            print(brand_gender_crosstab)
            
            # Visualize brand preference by gender
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # Stacked bar chart
            brand_gender_crosstab.plot(kind='bar', stacked=True, ax=axes[0])
            axes[0].set_title('Brand Preference by Gender (Stacked)')
            axes[0].set_xlabel('Brand')
            axes[0].set_ylabel('Count')
            axes[0].tick_params(axis='x', rotation=45)
            
            # Heatmap
            sns.heatmap(brand_gender_crosstab, annot=True, fmt='d', ax=axes[1], cmap='Blues')
            axes[1].set_title('Brand Preference by Gender (Heatmap)')
            
            plt.tight_layout()
            if save_plots:
                plt.savefig('images/brand_preference_by_gender.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        # Brand preference by income level
        if 'Income_Level' in self.df.columns:
            print(f"\n💰 BRAND PREFERENCE BY INCOME LEVEL:")
            brand_income_crosstab = pd.crosstab(self.df[brand_col], self.df['Income_Level'])
            print(brand_income_crosstab)
            
            # Visualize
            plt.figure(figsize=(12, 8))
            sns.heatmap(brand_income_crosstab, annot=True, fmt='d', cmap='Greens')
            plt.title('Brand Preference by Income Level')
            plt.tight_layout()
            if save_plots:
                plt.savefig('images/brand_preference_by_income.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    def correlation_analysis(self, save_plots=True):
        """
        Analyze correlations between numerical variables.
        
        Parameters:
        -----------
        save_plots : bool
            Whether to save plots to images folder
        """
        print("\n🔗 CORRELATION ANALYSIS")
        print("=" * 25)
        
        # Select numerical columns
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) < 2:
            print("❌ Not enough numerical columns for correlation analysis")
            return
        
        # Calculate correlation matrix
        correlation_matrix = self.df[numerical_cols].corr()
        print(f"\n📊 CORRELATION MATRIX:")
        print(correlation_matrix.round(3))
        
        # Visualize correlation matrix
        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, 
                   mask=mask,
                   annot=True, 
                   cmap='RdYlBu_r', 
                   center=0,
                   square=True,
                   fmt='.2f',
                   cbar_kws={"shrink": 0.8})
        plt.title('Correlation Matrix of Numerical Variables')
        plt.tight_layout()
        if save_plots:
            plt.savefig('images/correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Find strong correlations
        strong_correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) > 0.5:  # Strong correlation threshold
                    strong_correlations.append({
                        'Variable_1': correlation_matrix.columns[i],
                        'Variable_2': correlation_matrix.columns[j],
                        'Correlation': corr_value
                    })
        
        if strong_correlations:
            print(f"\n🔍 STRONG CORRELATIONS (|r| > 0.5):")
            for corr in strong_correlations:
                print(f"  {corr['Variable_1']} ↔ {corr['Variable_2']}: {corr['Correlation']:.3f}")
        else:
            print(f"\n✅ No strong correlations found (|r| > 0.5)")
    
    def sustainability_vs_price_analysis(self, save_plots=True):
        """
        Analyze the relationship between sustainability and price preferences.
        
        Parameters:
        -----------
        save_plots : bool
            Whether to save plots to images folder
        """
        print("\n🌱💰 SUSTAINABILITY VS PRICE ANALYSIS")
        print("=" * 40)
        
        sustainability_col = None
        price_col = None
        
        # Find sustainability and price columns
        for col in self.df.columns:
            if 'sustainabil' in col.lower():
                sustainability_col = col
            if 'price' in col.lower():
                price_col = col
        
        if not sustainability_col or not price_col:
            print("❌ Sustainability or Price columns not found")
            print(f"Available columns: {list(self.df.columns)}")
            return
        
        print(f"📊 Analyzing {sustainability_col} vs {price_col}")
        
        # Cross-tabulation
        if self.df[sustainability_col].dtype == 'object' and self.df[price_col].dtype == 'object':
            crosstab = pd.crosstab(self.df[sustainability_col], self.df[price_col])
            print(f"\n📋 CROSS-TABULATION:")
            print(crosstab)
            
            # Visualize
            plt.figure(figsize=(10, 6))
            sns.heatmap(crosstab, annot=True, fmt='d', cmap='YlOrRd')
            plt.title('Sustainability vs Price Sensitivity')
            plt.tight_layout()
            if save_plots:
                plt.savefig('images/sustainability_vs_price.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        # If numerical, create scatter plot
        elif self.df[sustainability_col].dtype in [np.number] and self.df[price_col].dtype in [np.number]:
            plt.figure(figsize=(10, 6))
            plt.scatter(self.df[sustainability_col], self.df[price_col], alpha=0.6)
            plt.xlabel(sustainability_col)
            plt.ylabel(price_col)
            plt.title('Sustainability vs Price Sensitivity Relationship')
            
            # Add correlation coefficient
            corr = self.df[sustainability_col].corr(self.df[price_col])
            plt.text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                    transform=plt.gca().transAxes, fontsize=12,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            if save_plots:
                plt.savefig('images/sustainability_vs_price_scatter.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    def create_interactive_dashboard(self):
        """
        Create an interactive dashboard using Plotly.
        """
        print("\n📊 CREATING INTERACTIVE DASHBOARD")
        print("=" * 35)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Brand Preference Distribution', 'Age vs Income', 
                          'Gender Distribution', 'Demographic Overview'),
            specs=[[{"type": "pie"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Brand preference pie chart
        if 'Brand_Preference' in self.df.columns:
            brand_counts = self.df['Brand_Preference'].value_counts()
            fig.add_trace(
                go.Pie(labels=brand_counts.index, values=brand_counts.values, name="Brand Preference"),
                row=1, col=1
            )
        
        # Age vs Income scatter (if both exist)
        if 'Age' in self.df.columns and 'Income_Level' in self.df.columns:
            for income in self.df['Income_Level'].unique():
                if pd.notna(income):
                    subset = self.df[self.df['Income_Level'] == income]
                    fig.add_trace(
                        go.Scatter(x=subset['Age'], y=[income]*len(subset), 
                                 mode='markers', name=f'Income: {income}'),
                        row=1, col=2
                    )
        
        # Gender distribution
        if 'Gender' in self.df.columns:
            gender_counts = self.df['Gender'].value_counts()
            fig.add_trace(
                go.Bar(x=gender_counts.index, y=gender_counts.values, name="Gender"),
                row=2, col=1
            )
        
        # Income distribution
        if 'Income_Level' in self.df.columns:
            income_counts = self.df['Income_Level'].value_counts()
            fig.add_trace(
                go.Bar(x=income_counts.index, y=income_counts.values, name="Income Level"),
                row=2, col=2
            )
        
        fig.update_layout(height=800, title_text="Cosmetics Brand Analysis Dashboard")
        fig.show()
        
        # Save as HTML
        fig.write_html("images/interactive_dashboard.html")
        print("✅ Interactive dashboard saved as 'images/interactive_dashboard.html'")
    
    def generate_eda_insights(self):
        """
        Generate comprehensive insights from EDA.
        
        Returns:
        --------
        dict
            Dictionary containing key insights
        """
        insights = {}
        
        # Demographic insights
        if 'Gender' in self.df.columns:
            gender_dist = self.df['Gender'].value_counts(normalize=True) * 100
            insights['gender_distribution'] = gender_dist.to_dict()
        
        if 'Age' in self.df.columns:
            insights['age_stats'] = {
                'mean': self.df['Age'].mean(),
                'median': self.df['Age'].median(),
                'std': self.df['Age'].std()
            }
        
        # Brand preference insights
        if 'Brand_Preference' in self.df.columns:
            brand_dist = self.df['Brand_Preference'].value_counts(normalize=True) * 100
            insights['brand_preference'] = brand_dist.to_dict()
            insights['most_popular_brand'] = brand_dist.index[0]
        
        # Store insights
        self.insights = insights
        
        print("\n💡 KEY INSIGHTS FROM EDA:")
        print("=" * 30)
        for key, value in insights.items():
            print(f"{key}: {value}")
        
        return insights
    
    def export_eda_report(self, filename="eda_report.txt"):
        """
        Export EDA findings to a text file.
        
        Parameters:
        -----------
        filename : str
            Name of the output file
        """
        with open(filename, 'w') as f:
            f.write("COSMETICS BRAND ANALYSIS - EDA REPORT\n")
            f.write("=" * 45 + "\n\n")
            
            f.write("DATASET OVERVIEW:\n")
            f.write(f"Total Records: {len(self.df):,}\n")
            f.write(f"Total Features: {len(self.df.columns)}\n\n")
            
            # Demographics
            if 'Gender' in self.df.columns:
                f.write("GENDER DISTRIBUTION:\n")
                gender_counts = self.df['Gender'].value_counts()
                for gender, count in gender_counts.items():
                    pct = (count / len(self.df)) * 100
                    f.write(f"  {gender}: {count} ({pct:.1f}%)\n")
                f.write("\n")
            
            # Brand preferences
            if 'Brand_Preference' in self.df.columns:
                f.write("BRAND PREFERENCES:\n")
                brand_counts = self.df['Brand_Preference'].value_counts()
                for brand, count in brand_counts.items():
                    pct = (count / len(self.df)) * 100
                    f.write(f"  {brand}: {count} ({pct:.1f}%)\n")
                f.write("\n")
            
            # Key insights
            f.write("KEY INSIGHTS:\n")
            for key, value in self.insights.items():
                f.write(f"  {key}: {value}\n")
        
        print(f"✅ EDA report exported to {filename}")


def main():
    """
    Main function to demonstrate EDA workflow.
    """
    print("🧴 COSMETICS BRAND ANALYSIS - EXPLORATORY DATA ANALYSIS")
    print("=" * 65)
    
    # This would typically load data from the data preparation module
    print("📁 This module requires data to be loaded first.")
    print("Please run data_preparation.py first or import the DataPreparation class.")
    
    # Example usage:
    # from data_preparation import DataPreparation
    # data_prep = DataPreparation("data/your_file.csv")
    # df = data_prep.load_data()
    # 
    # eda = ExploratoryDataAnalysis(df)
    # eda.demographic_analysis()
    # eda.brand_preference_analysis()
    # eda.correlation_analysis()
    # eda.sustainability_vs_price_analysis()
    # eda.create_interactive_dashboard()
    # eda.generate_eda_insights()
    # eda.export_eda_report()


if __name__ == "__main__":
    main()