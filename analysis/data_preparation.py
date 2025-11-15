"""
Data Preparation Module for Cosmetics Brand Analysis

This module handles data loading, cleaning, validation, and preprocessing
for the cosmetics brand analysis project.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class DataPreparation:
    """
    Class for handling all data preparation tasks including:
    - Data loading and validation
    - Quality assessment
    - Cleaning and preprocessing
    - Basic data overview
    """
    
    def __init__(self, data_path=None):
        """
        Initialize DataPreparation class.
        
        Parameters:
        -----------
        data_path : str, optional
            Path to the dataset file
        """
        self.data_path = data_path
        self.df = None
        self.quality_report = {}
        
    def load_data(self, file_path=None):
        """
        Load dataset from CSV file.
        
        Parameters:
        -----------
        file_path : str, optional
            Path to the dataset file
            
        Returns:
        --------
        pd.DataFrame
            Loaded dataset
        """
        if file_path:
            self.data_path = file_path
            
        if not self.data_path:
            raise ValueError("No data path provided")
            
        try:
            self.df = pd.read_csv(self.data_path)
            
            # Auto-generate Customer_ID if missing
            if 'Customer_ID' not in self.df.columns:
                self.df.insert(0, 'Customer_ID', [f'C{i+1:04d}' for i in range(len(self.df))])
                print("ℹ️ Customer_ID column auto-generated")
            
            print(f"✅ Data loaded successfully from {self.data_path}")
            print(f"📊 Dataset shape: {self.df.shape}")
            return self.df
        except FileNotFoundError:
            print(f"❌ File not found: {self.data_path}")
            print("📁 Please ensure your dataset is in the data/ folder")
            return None
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None
    
    def generate_data_overview(self):
        """
        Generate comprehensive data overview and quality assessment.
        
        Returns:
        --------
        dict
            Dictionary containing data quality metrics
        """
        if self.df is None:
            print("❌ No data loaded. Please load data first.")
            return None
            
        print("📋 DATASET OVERVIEW")
        print("=" * 50)
        
        # Basic information
        print(f"Shape: {self.df.shape}")
        print(f"Memory usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Data types
        print(f"\n📊 DATA TYPES:")
        print(self.df.dtypes.value_counts())
        
        # Missing values
        missing_values = self.df.isnull().sum()
        missing_pct = (missing_values / len(self.df)) * 100
        
        if missing_values.sum() > 0:
            print(f"\n⚠️  MISSING VALUES:")
            missing_df = pd.DataFrame({
                'Missing Count': missing_values[missing_values > 0],
                'Percentage': missing_pct[missing_values > 0]
            })
            print(missing_df)
        else:
            print(f"\n✅ No missing values found!")
        
        # Duplicate rows
        duplicates = self.df.duplicated().sum()
        print(f"\n🔄 DUPLICATE ROWS: {duplicates}")
        
        # Store quality report
        self.quality_report = {
            'shape': self.df.shape,
            'missing_values': missing_values,
            'duplicates': duplicates,
            'data_types': self.df.dtypes
        }
        
        return self.quality_report
    
    def display_sample_data(self, n=5):
        """
        Display sample data with proper formatting.
        
        Parameters:
        -----------
        n : int
            Number of rows to display
        """
        if self.df is None:
            print("❌ No data loaded. Please load data first.")
            return
            
        print(f"\n🔍 FIRST {n} ROWS:")
        print("-" * 30)
        print(self.df.head(n))
        
        print(f"\n🔍 LAST {n} ROWS:")
        print("-" * 30)
        print(self.df.tail(n))
    
    def check_data_completeness(self):
        """
        Check data completeness and identify potential issues.
        """
        if self.df is None:
            print("❌ No data loaded. Please load data first.")
            return
            
        print("\n🔍 DATA COMPLETENESS CHECK")
        print("=" * 40)
        
        # Check for completely empty rows
        empty_rows = self.df.isnull().all(axis=1).sum()
        print(f"Empty rows: {empty_rows}")
        
        # Check for rows with minimal data
        threshold = len(self.df.columns) * 0.5  # 50% of columns
        sparse_rows = (self.df.isnull().sum(axis=1) > threshold).sum()
        print(f"Sparse rows (>50% missing): {sparse_rows}")
        
        # Check for constant columns
        constant_cols = []
        for col in self.df.columns:
            if self.df[col].nunique() <= 1:
                constant_cols.append(col)
        
        if constant_cols:
            print(f"Constant columns: {constant_cols}")
        else:
            print("✅ No constant columns found")
    
    def detect_outliers(self, method='iqr'):
        """
        Detect outliers in numerical columns.
        
        Parameters:
        -----------
        method : str
            Method for outlier detection ('iqr', 'zscore')
            
        Returns:
        --------
        dict
            Dictionary with outlier information for each column
        """
        if self.df is None:
            print("❌ No data loaded. Please load data first.")
            return None
            
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        outliers_info = {}
        
        print(f"\n🔍 OUTLIER DETECTION ({method.upper()} METHOD)")
        print("=" * 45)
        
        for col in numerical_cols:
            if method == 'iqr':
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
                
            elif method == 'zscore':
                z_scores = np.abs((self.df[col] - self.df[col].mean()) / self.df[col].std())
                outliers = self.df[z_scores > 3]
            
            outlier_count = len(outliers)
            outlier_pct = (outlier_count / len(self.df)) * 100
            
            outliers_info[col] = {
                'count': outlier_count,
                'percentage': outlier_pct,
                'outlier_indices': outliers.index.tolist()
            }
            
            print(f"{col}: {outlier_count} outliers ({outlier_pct:.1f}%)")
        
        return outliers_info
    
    def generate_summary_statistics(self):
        """
        Generate comprehensive summary statistics.
        """
        if self.df is None:
            print("❌ No data loaded. Please load data first.")
            return
            
        print("\n📊 SUMMARY STATISTICS")
        print("=" * 35)
        
        # Numerical variables
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            print("\n🔢 NUMERICAL VARIABLES:")
            print(self.df[numerical_cols].describe())
        
        # Categorical variables
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            print(f"\n📝 CATEGORICAL VARIABLES:")
            for col in categorical_cols:
                print(f"\n{col}:")
                print(f"  Unique values: {self.df[col].nunique()}")
                print(f"  Value counts:")
                value_counts = self.df[col].value_counts().head()
                for value, count in value_counts.items():
                    print(f"    {value}: {count}")
    
    def create_data_profile_visualization(self, save_path=None):
        """
        Create visualizations for data profiling.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the visualization
        """
        if self.df is None:
            print("❌ No data loaded. Please load data first.")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Data Profiling Overview', fontsize=16, fontweight='bold')
        
        # 1. Missing values heatmap
        missing_data = self.df.isnull()
        if missing_data.any().any():
            sns.heatmap(missing_data, cbar=True, ax=axes[0,0], cmap='viridis')
            axes[0,0].set_title('Missing Values Pattern')
        else:
            axes[0,0].text(0.5, 0.5, 'No Missing Values', 
                          ha='center', va='center', fontsize=14)
            axes[0,0].set_title('Missing Values Pattern')
        
        # 2. Data types distribution
        dtype_counts = self.df.dtypes.value_counts()
        axes[0,1].pie(dtype_counts.values, labels=dtype_counts.index, autopct='%1.1f%%')
        axes[0,1].set_title('Data Types Distribution')
        
        # 3. Completeness by column
        completeness = (1 - self.df.isnull().sum() / len(self.df)) * 100
        completeness.plot(kind='bar', ax=axes[1,0])
        axes[1,0].set_title('Data Completeness by Column')
        axes[1,0].set_ylabel('Completeness (%)')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # 4. Numerical columns distribution
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            self.df[numerical_cols].hist(bins=20, ax=axes[1,1])
            axes[1,1].set_title('Numerical Variables Distribution')
        else:
            axes[1,1].text(0.5, 0.5, 'No Numerical Variables', 
                          ha='center', va='center', fontsize=14)
            axes[1,1].set_title('Numerical Variables Distribution')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Data profile visualization saved to {save_path}")
        
        plt.show()
    
    def export_quality_report(self, output_path="data_quality_report.txt"):
        """
        Export data quality report to text file.
        
        Parameters:
        -----------
        output_path : str
            Path for the output report file
        """
        if self.df is None:
            print("❌ No data loaded. Please load data first.")
            return
            
        with open(output_path, 'w') as f:
            f.write("COSMETICS BRAND ANALYSIS - DATA QUALITY REPORT\n")
            f.write("=" * 55 + "\n\n")
            
            f.write(f"Dataset Shape: {self.df.shape}\n")
            f.write(f"Total Records: {len(self.df):,}\n")
            f.write(f"Total Features: {len(self.df.columns)}\n\n")
            
            f.write("COLUMN INFORMATION:\n")
            f.write("-" * 20 + "\n")
            for col in self.df.columns:
                f.write(f"{col}: {self.df[col].dtype}\n")
            
            f.write(f"\nMISSING VALUES:\n")
            f.write("-" * 15 + "\n")
            missing = self.df.isnull().sum()
            for col, count in missing.items():
                if count > 0:
                    pct = (count / len(self.df)) * 100
                    f.write(f"{col}: {count} ({pct:.1f}%)\n")
            
            f.write(f"\nDUPLICATE ROWS: {self.df.duplicated().sum()}\n")
            
        print(f"✅ Data quality report exported to {output_path}")


def main():
    """
    Main function to demonstrate data preparation workflow.
    """
    print("🧴 COSMETICS BRAND ANALYSIS - DATA PREPARATION")
    print("=" * 55)
    
    # Initialize data preparation
    data_prep = DataPreparation()
    
    # Try to load data from common locations
    possible_paths = [
        "data/cosmetics_data.csv",
        "data/survey_data.csv", 
        "data/customer_data.csv",
        "../data/cosmetics_data.csv"
    ]
    
    data_loaded = False
    for path in possible_paths:
        if Path(path).exists():
            data_prep.load_data(path)
            data_loaded = True
            break
    
    if not data_loaded:
        print("📁 No dataset found. Please place your data file in the data/ folder")
        print("Expected file names: cosmetics_data.csv, survey_data.csv, customer_data.csv")
        return
    
    # Run data preparation workflow
    data_prep.generate_data_overview()
    data_prep.display_sample_data()
    data_prep.check_data_completeness()
    data_prep.detect_outliers()
    data_prep.generate_summary_statistics()
    
    # Create visualizations
    data_prep.create_data_profile_visualization("images/data_profile.png")
    
    # Export report
    data_prep.export_quality_report("reports/data_quality_report.txt")
    
    print("\n✅ Data preparation completed successfully!")


if __name__ == "__main__":
    main()