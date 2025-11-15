"""
Net Promoter Score (NPS) Analysis Module for Cosmetics Brand Analysis

This module handles comprehensive NPS analysis including:
- NPS calculation and segmentation
- Driver analysis
- Demographic breakdowns
- Predictive modeling
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class NPSAnalysis:
    """
    Class for comprehensive Net Promoter Score analysis.
    """
    
    def __init__(self, dataframe, nps_column='NPS_Score'):
        """
        Initialize NPS Analysis class.
        
        Parameters:
        -----------
        dataframe : pd.DataFrame
            The dataset containing NPS scores
        nps_column : str
            Name of the NPS score column
        """
        self.df = dataframe.copy()
        
        # Check if the provided column exists, if not try common variations
        if nps_column not in self.df.columns:
            # Try common NPS column name variations
            nps_variants = ['NPS', 'NPS_Score', 'nps', 'nps_score', 'Net_Promoter_Score']
            found = False
            for variant in nps_variants:
                if variant in self.df.columns:
                    self.nps_column = variant
                    found = True
                    print(f"ℹ️ Using NPS column: {variant}")
                    break
            if not found:
                raise ValueError(f"NPS column '{nps_column}' not found in dataset. Available columns: {list(self.df.columns)}")
        else:
            self.nps_column = nps_column
        
        self.nps_segments = {}
        self.drivers = {}
    
    def calculate_nps_segments(self):
        """
        Calculate NPS segments (Detractors, Passives, Promoters).
        
        Returns:
        --------
        dict
            Dictionary containing NPS segmentation results
        """
        print("📊 NPS SEGMENTATION ANALYSIS")
        print("=" * 35)
        
        # Define NPS segments
        def categorize_nps(score):
            if score >= 9:
                return 'Promoter'
            elif score >= 7:
                return 'Passive'
            else:
                return 'Detractor'
        
        self.df['NPS_Segment'] = self.df[self.nps_column].apply(categorize_nps)
        
        # Calculate segment distributions
        segment_counts = self.df['NPS_Segment'].value_counts()
        segment_percentages = (segment_counts / len(self.df)) * 100
        
        # Calculate overall NPS score
        promoters_pct = segment_percentages.get('Promoter', 0)
        detractors_pct = segment_percentages.get('Detractor', 0)
        overall_nps = promoters_pct - detractors_pct
        
        self.nps_segments = {
            'counts': segment_counts.to_dict(),
            'percentages': segment_percentages.to_dict(),
            'overall_nps': overall_nps
        }
        
        print(f"📈 NPS SEGMENT DISTRIBUTION:")
        for segment, count in segment_counts.items():
            pct = segment_percentages[segment]
            print(f"  {segment}: {count} customers ({pct:.1f}%)")
        
        print(f"\n🎯 OVERALL NPS SCORE: {overall_nps:.1f}")
        
        # Interpret NPS score
        if overall_nps >= 70:
            interpretation = "Excellent (World-class)"
        elif overall_nps >= 50:
            interpretation = "Great (Very good)"
        elif overall_nps >= 30:
            interpretation = "Good (Positive)"
        elif overall_nps >= 0:
            interpretation = "Okay (Needs improvement)"
        else:
            interpretation = "Poor (Critical issues)"
        
        print(f"📋 NPS INTERPRETATION: {interpretation}")
        
        return self.nps_segments
    
    def visualize_nps_distribution(self, save_plots=True):
        """
        Create visualizations for NPS distribution and segments.
        
        Parameters:
        -----------
        save_plots : bool
            Whether to save plots to images folder
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Net Promoter Score Analysis', fontsize=16, fontweight='bold')
        
        # 1. NPS Score Distribution
        axes[0,0].hist(self.df[self.nps_column], bins=11, alpha=0.7, 
                      color='skyblue', edgecolor='black')
        axes[0,0].set_title('NPS Score Distribution')
        axes[0,0].set_xlabel('NPS Score')
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].axvline(self.df[self.nps_column].mean(), color='red', 
                         linestyle='--', label=f'Mean: {self.df[self.nps_column].mean():.1f}')
        axes[0,0].legend()
        
        # 2. NPS Segments Pie Chart
        segment_counts = self.df['NPS_Segment'].value_counts()
        colors = ['#ff9999', '#66b3ff', '#99ff99']  # Red, Blue, Green
        axes[0,1].pie(segment_counts.values, labels=segment_counts.index, 
                     autopct='%1.1f%%', colors=colors)
        axes[0,1].set_title('NPS Segments Distribution')
        
        # 3. NPS by Gender (if available)
        if 'Gender' in self.df.columns:
            sns.boxplot(data=self.df, x='Gender', y=self.nps_column, ax=axes[1,0])
            axes[1,0].set_title('NPS Score by Gender')
        else:
            axes[1,0].text(0.5, 0.5, 'Gender data not available', 
                          ha='center', va='center', fontsize=12)
            axes[1,0].set_title('NPS Score by Gender')
        
        # 4. NPS by Income Level (if available)
        if 'Income_Level' in self.df.columns:
            sns.boxplot(data=self.df, x='Income_Level', y=self.nps_column, ax=axes[1,1])
            axes[1,1].set_title('NPS Score by Income Level')
            axes[1,1].tick_params(axis='x', rotation=45)
        else:
            axes[1,1].text(0.5, 0.5, 'Income Level data not available', 
                          ha='center', va='center', fontsize=12)
            axes[1,1].set_title('NPS Score by Income Level')
        
        plt.tight_layout()
        if save_plots:
            plt.savefig('images/nps_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_nps_drivers(self, feature_columns=None):
        """
        Analyze factors that drive NPS scores using correlation and feature importance.
        
        Parameters:
        -----------
        feature_columns : list, optional
            List of columns to analyze as potential drivers
        """
        print("\n🔍 NPS DRIVERS ANALYSIS")
        print("=" * 28)
        
        # Select numerical columns as potential drivers
        if feature_columns is None:
            numerical_cols = self.df.select_dtypes(include=[np.number]).columns
            feature_columns = [col for col in numerical_cols if col != self.nps_column]
        
        if len(feature_columns) == 0:
            print("❌ No numerical features available for driver analysis")
            return
        
        # Correlation analysis
        correlations = {}
        for col in feature_columns:
            if col in self.df.columns:
                corr = self.df[col].corr(self.df[self.nps_column])
                if not np.isnan(corr):
                    correlations[col] = corr
        
        if correlations:
            print(f"📊 CORRELATION WITH NPS SCORE:")
            sorted_correlations = dict(sorted(correlations.items(), 
                                            key=lambda x: abs(x[1]), reverse=True))
            for feature, corr in sorted_correlations.items():
                print(f"  {feature}: {corr:.3f}")
            
            # Visualize top correlations
            top_correlations = dict(list(sorted_correlations.items())[:5])
            
            plt.figure(figsize=(10, 6))
            colors = ['red' if x < 0 else 'green' for x in top_correlations.values()]
            plt.barh(list(top_correlations.keys()), list(top_correlations.values()), 
                    color=colors, alpha=0.7)
            plt.title('Top 5 Features Correlated with NPS Score')
            plt.xlabel('Correlation Coefficient')
            plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            plt.tight_layout()
            plt.savefig('images/nps_drivers_correlation.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        # Feature importance using Random Forest
        if len(feature_columns) > 1:
            try:
                # Prepare data for modeling
                features_df = self.df[feature_columns].fillna(0)
                target = self.df[self.nps_column].fillna(self.df[self.nps_column].mean())
                
                # Train Random Forest
                rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
                rf_model.fit(features_df, target)
                
                # Get feature importance
                importance_scores = dict(zip(feature_columns, rf_model.feature_importances_))
                sorted_importance = dict(sorted(importance_scores.items(), 
                                               key=lambda x: x[1], reverse=True))
                
                print(f"\n🌳 FEATURE IMPORTANCE (Random Forest):")
                for feature, importance in sorted_importance.items():
                    print(f"  {feature}: {importance:.3f}")
                
                # Visualize feature importance
                top_features = dict(list(sorted_importance.items())[:5])
                
                plt.figure(figsize=(10, 6))
                plt.barh(list(top_features.keys()), list(top_features.values()), 
                        color='steelblue', alpha=0.7)
                plt.title('Top 5 Feature Importance for NPS Prediction')
                plt.xlabel('Importance Score')
                plt.tight_layout()
                plt.savefig('images/nps_feature_importance.png', dpi=300, bbox_inches='tight')
                plt.show()
                
                self.drivers = {
                    'correlations': sorted_correlations,
                    'feature_importance': sorted_importance
                }
                
            except Exception as e:
                print(f"❌ Error in feature importance analysis: {e}")
    
    def nps_by_demographics(self, save_plots=True):
        """
        Analyze NPS scores across different demographic segments.
        
        Parameters:
        -----------
        save_plots : bool
            Whether to save plots to images folder
        """
        print("\n👥 NPS BY DEMOGRAPHICS")
        print("=" * 25)
        
        # Check for brand preference column (handle variations)
        brand_cols = ['Brand_Preference', 'Brand_Preferences', 'Brand_Preference_Normalized']
        brand_col = None
        for col in brand_cols:
            if col in self.df.columns:
                brand_col = col
                break
        
        demographic_cols = ['Gender', 'Income_Level', 'Age_Group']
        if brand_col:
            demographic_cols.append(brand_col)
        available_demos = [col for col in demographic_cols if col in self.df.columns]
        
        if not available_demos:
            print("❌ No demographic columns found")
            return
        
        # Create age groups if Age column exists
        if 'Age' in self.df.columns and 'Age_Group' not in self.df.columns:
            self.df['Age_Group'] = pd.cut(self.df['Age'], 
                                         bins=[0, 25, 35, 50, 100], 
                                         labels=['18-25', '26-35', '36-50', '50+'])
            available_demos.append('Age_Group')
        
        # Calculate NPS by each demographic
        for demo in available_demos:
            if demo in self.df.columns:
                print(f"\n📊 NPS BY {demo.upper()}:")
                nps_by_demo = self.df.groupby(demo)[self.nps_column].agg([
                    'count', 'mean', 'std'
                ]).round(2)
                print(nps_by_demo)
                
                # Statistical significance test (ANOVA)
                try:
                    from scipy import stats
                    groups = [group[self.nps_column].dropna() for name, group in self.df.groupby(demo)]
                    if len(groups) > 1:
                        f_stat, p_value = stats.f_oneway(*groups)
                        print(f"ANOVA F-statistic: {f_stat:.3f}, p-value: {p_value:.3f}")
                        if p_value < 0.05:
                            print("✅ Statistically significant difference found")
                        else:
                            print("❌ No statistically significant difference")
                except ImportError:
                    print("📊 Install scipy for statistical significance testing")
        
        # Visualize NPS by demographics
        if len(available_demos) > 0:
            n_plots = min(len(available_demos), 4)
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            axes = axes.flatten()
            
            for i, demo in enumerate(available_demos[:4]):
                sns.boxplot(data=self.df, x=demo, y=self.nps_column, ax=axes[i])
                axes[i].set_title(f'NPS Score by {demo}')
                axes[i].tick_params(axis='x', rotation=45)
            
            # Hide unused subplots
            for j in range(i+1, 4):
                axes[j].set_visible(False)
            
            plt.tight_layout()
            if save_plots:
                plt.savefig('images/nps_by_demographics.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    def build_nps_prediction_model(self, test_size=0.2):
        """
        Build a predictive model for NPS scores.
        
        Parameters:
        -----------
        test_size : float
            Proportion of data to use for testing
            
        Returns:
        --------
        dict
            Model performance metrics
        """
        print("\n🤖 NPS PREDICTION MODEL")
        print("=" * 28)
        
        # Select features
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numerical_cols if col != self.nps_column]
        
        if len(feature_cols) < 2:
            print("❌ Not enough features for predictive modeling")
            return None
        
        # Prepare data
        X = self.df[feature_cols].fillna(0)
        y = self.df[self.nps_column].fillna(self.df[self.nps_column].mean())
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Train models
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
        }
        
        results = {}
        
        for name, model in models.items():
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            results[name] = {
                'RMSE': rmse,
                'R²': r2,
                'model': model
            }
            
            print(f"\n{name}:")
            print(f"  RMSE: {rmse:.3f}")
            print(f"  R² Score: {r2:.3f}")
        
        # Visualize predictions vs actual
        best_model_name = max(results.keys(), key=lambda x: results[x]['R²'])
        best_model = results[best_model_name]['model']
        y_pred_best = best_model.predict(X_test)
        
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred_best, alpha=0.6)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                'r--', lw=2)
        plt.xlabel('Actual NPS Score')
        plt.ylabel('Predicted NPS Score')
        plt.title(f'NPS Prediction Performance ({best_model_name})')
        plt.text(0.05, 0.95, f'R² = {results[best_model_name]["R²"]:.3f}', 
                transform=plt.gca().transAxes, fontsize=12,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        plt.tight_layout()
        plt.savefig('images/nps_prediction_model.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return results
    
    def generate_nps_insights(self):
        """
        Generate comprehensive insights from NPS analysis.
        
        Returns:
        --------
        dict
            Dictionary containing key NPS insights
        """
        insights = {
            'overall_nps': self.nps_segments.get('overall_nps', 0),
            'segment_distribution': self.nps_segments.get('percentages', {}),
            'mean_score': self.df[self.nps_column].mean(),
            'median_score': self.df[self.nps_column].median(),
            'top_drivers': list(self.drivers.get('correlations', {}).keys())[:3] if self.drivers else []
        }
        
        print("\n💡 KEY NPS INSIGHTS:")
        print("=" * 22)
        print(f"Overall NPS Score: {insights['overall_nps']:.1f}")
        print(f"Average Score: {insights['mean_score']:.1f}")
        print(f"Median Score: {insights['median_score']:.1f}")
        
        if insights['top_drivers']:
            print(f"Top Drivers: {', '.join(insights['top_drivers'])}")
        
        return insights
    
    def export_nps_report(self, filename="nps_analysis_report.txt"):
        """
        Export NPS analysis results to a text file.
        
        Parameters:
        -----------
        filename : str
            Name of the output file
        """
        with open(filename, 'w') as f:
            f.write("COSMETICS BRAND ANALYSIS - NPS ANALYSIS REPORT\n")
            f.write("=" * 55 + "\n\n")
            
            # Overall NPS
            f.write(f"OVERALL NPS SCORE: {self.nps_segments.get('overall_nps', 0):.1f}\n\n")
            
            # Segment distribution
            f.write("NPS SEGMENT DISTRIBUTION:\n")
            f.write("-" * 25 + "\n")
            for segment, pct in self.nps_segments.get('percentages', {}).items():
                f.write(f"{segment}: {pct:.1f}%\n")
            f.write("\n")
            
            # Top drivers
            if self.drivers.get('correlations'):
                f.write("TOP NPS DRIVERS (Correlation):\n")
                f.write("-" * 30 + "\n")
                for feature, corr in list(self.drivers['correlations'].items())[:5]:
                    f.write(f"{feature}: {corr:.3f}\n")
                f.write("\n")
            
            # Recommendations
            f.write("RECOMMENDATIONS:\n")
            f.write("-" * 15 + "\n")
            overall_nps = self.nps_segments.get('overall_nps', 0)
            if overall_nps >= 50:
                f.write("- Excellent NPS score. Focus on maintaining quality.\n")
            elif overall_nps >= 30:
                f.write("- Good NPS score. Identify and address detractor concerns.\n")
            elif overall_nps >= 0:
                f.write("- Room for improvement. Focus on customer experience.\n")
            else:
                f.write("- Critical issues. Immediate action required.\n")
        
        print(f"✅ NPS analysis report exported to {filename}")


def main():
    """
    Main function to demonstrate NPS analysis workflow.
    """
    print("🧴 COSMETICS BRAND ANALYSIS - NPS ANALYSIS")
    print("=" * 50)
    
    print("📁 This module requires data to be loaded first.")
    print("Please run data_preparation.py first or import the DataPreparation class.")
    
    # Example usage:
    # from data_preparation import DataPreparation
    # data_prep = DataPreparation("data/your_file.csv")
    # df = data_prep.load_data()
    # 
    # nps_analysis = NPSAnalysis(df)
    # nps_analysis.calculate_nps_segments()
    # nps_analysis.visualize_nps_distribution()
    # nps_analysis.analyze_nps_drivers()
    # nps_analysis.nps_by_demographics()
    # nps_analysis.build_nps_prediction_model()
    # nps_analysis.generate_nps_insights()
    # nps_analysis.export_nps_report()


if __name__ == "__main__":
    main()