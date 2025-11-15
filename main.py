"""
Main Orchestration Script for Cosmetics Brand Analysis

This script runs the complete analysis pipeline:
1. Data Preparation
2. Exploratory Data Analysis
3. NPS Analysis
4. Text Analysis
5. Recommendations Generation

Usage:
    python main.py --data_path "data/your_file.csv" --text_column "Open_Feedback"
"""

import os
import sys
import argparse
import traceback
from pathlib import Path
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Add the analysis directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'analysis'))

# Import analysis modules
try:
    from analysis.data_preparation import DataPreparation
    from analysis.exploratory_data_analysis import ExploratoryDataAnalysis
    from analysis.nps_analysis import NPSAnalysis
    from analysis.text_analysis import TextAnalysis
    from analysis.recommendations import RecommendationEngine
except ImportError as e:
    print(f"❌ Error importing analysis modules: {e}")
    print("Please ensure all analysis modules are properly installed")
    sys.exit(1)

class CosmeticsBrandAnalysisPipeline:
    """
    Complete analysis pipeline for cosmetics brand data.
    """
    
    def __init__(self, data_path, text_column='Open_Feedback', nps_column='NPS_Score'):
        """
        Initialize the analysis pipeline.
        
        Parameters:
        -----------
        data_path : str
            Path to the dataset file
        text_column : str
            Name of the text feedback column
        nps_column : str
            Name of the NPS score column
        """
        self.data_path = data_path
        self.text_column = text_column
        self.nps_column = nps_column
        self.df = None
        self.results = {}
        
        # Create necessary directories
        self._create_directories()
        
    def _create_directories(self):
        """Create necessary output directories."""
        directories = ['images', 'reports', 'outputs']
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)
    
    def run_complete_analysis(self, save_outputs=True):
        """
        Run the complete analysis pipeline.
        
        Parameters:
        -----------
        save_outputs : bool
            Whether to save analysis outputs to files
            
        Returns:
        --------
        dict
            Dictionary containing all analysis results
        """
        print("🧴 COSMETICS BRAND ANALYSIS - COMPLETE PIPELINE")
        print("=" * 60)
        print(f"📁 Data source: {self.data_path}")
        print(f"📝 Text column: {self.text_column}")
        print(f"📊 NPS column: {self.nps_column}")
        print("=" * 60)
        
        try:
            # Step 1: Data Preparation
            print("\n🚀 STEP 1: DATA PREPARATION")
            print("-" * 30)
            self._run_data_preparation()
            
            # Step 2: Exploratory Data Analysis
            print("\n🚀 STEP 2: EXPLORATORY DATA ANALYSIS")
            print("-" * 40)
            self._run_eda()
            
            # Step 3: NPS Analysis
            print("\n🚀 STEP 3: NPS ANALYSIS")
            print("-" * 25)
            self._run_nps_analysis()
            
            # Step 4: Text Analysis
            print("\n🚀 STEP 4: TEXT ANALYSIS")
            print("-" * 25)
            self._run_text_analysis()
            
            # Step 5: Recommendations
            print("\n🚀 STEP 5: RECOMMENDATIONS GENERATION")
            print("-" * 40)
            self._run_recommendations()
            
            # Generate final summary
            print("\n🚀 FINAL STEP: SUMMARY GENERATION")
            print("-" * 35)
            self._generate_final_summary()
            
            if save_outputs:
                self._save_all_outputs()
            
            print("\n" + "=" * 60)
            print("✅ ANALYSIS PIPELINE COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            
            return self.results
            
        except Exception as e:
            print(f"\n❌ ERROR IN ANALYSIS PIPELINE: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            return None
    
    def _run_data_preparation(self):
        """Run data preparation step."""
        try:
            data_prep = DataPreparation(self.data_path)
            self.df = data_prep.load_data()
            
            if self.df is None:
                raise ValueError("Failed to load data")
            
            # Generate data overview
            quality_report = data_prep.generate_data_overview()
            data_prep.display_sample_data(3)
            data_prep.check_data_completeness()
            data_prep.detect_outliers()
            data_prep.generate_summary_statistics()
            
            # Create data profile visualization
            data_prep.create_data_profile_visualization("images/data_profile.png")
            
            # Export quality report
            data_prep.export_quality_report("reports/data_quality_report.txt")
            
            self.results['data_preparation'] = {
                'quality_report': quality_report,
                'dataset_shape': self.df.shape,
                'success': True
            }
            
            print("✅ Data preparation completed successfully")
            
        except Exception as e:
            print(f"❌ Data preparation failed: {e}")
            self.results['data_preparation'] = {'success': False, 'error': str(e)}
            raise
    
    def _run_eda(self):
        """Run exploratory data analysis step."""
        try:
            eda = ExploratoryDataAnalysis(self.df)
            
            # Run all EDA analyses
            eda.demographic_analysis(save_plots=True)
            eda.brand_preference_analysis(save_plots=True)
            eda.correlation_analysis(save_plots=True)
            eda.sustainability_vs_price_analysis(save_plots=True)
            eda.create_interactive_dashboard()
            
            # Generate insights
            eda_insights = eda.generate_eda_insights()
            
            # Export report
            eda.export_eda_report("reports/eda_report.txt")
            
            self.results['eda'] = {
                'insights': eda_insights,
                'success': True
            }
            
            print("✅ Exploratory data analysis completed successfully")
            
        except Exception as e:
            print(f"❌ EDA failed: {e}")
            self.results['eda'] = {'success': False, 'error': str(e)}
            # Continue pipeline even if EDA fails
    
    def _run_nps_analysis(self):
        """Run NPS analysis step."""
        try:
            if self.nps_column not in self.df.columns:
                print(f"⚠️ NPS column '{self.nps_column}' not found. Skipping NPS analysis.")
                self.results['nps'] = {'success': False, 'error': 'NPS column not found'}
                return
            
            nps_analysis = NPSAnalysis(self.df, self.nps_column)
            
            # Run all NPS analyses
            nps_segments = nps_analysis.calculate_nps_segments()
            nps_analysis.visualize_nps_distribution(save_plots=True)
            nps_analysis.analyze_nps_drivers()
            nps_analysis.nps_by_demographics(save_plots=True)
            
            # Build prediction model
            model_results = nps_analysis.build_nps_prediction_model()
            
            # Generate insights
            nps_insights = nps_analysis.generate_nps_insights()
            
            # Export report
            nps_analysis.export_nps_report("reports/nps_analysis_report.txt")
            
            self.results['nps'] = {
                'insights': nps_insights,
                'segments': nps_segments,
                'model_results': model_results,
                'success': True
            }
            
            print("✅ NPS analysis completed successfully")
            
        except Exception as e:
            print(f"❌ NPS analysis failed: {e}")
            self.results['nps'] = {'success': False, 'error': str(e)}
            # Continue pipeline even if NPS analysis fails
    
    def _run_text_analysis(self):
        """Run text analysis step."""
        try:
            if self.text_column not in self.df.columns:
                print(f"⚠️ Text column '{self.text_column}' not found. Skipping text analysis.")
                self.results['text'] = {'success': False, 'error': 'Text column not found'}
                return
            
            text_analysis = TextAnalysis(self.df, self.text_column)
            
            # Run all text analyses
            text_analysis.preprocess_all_texts()
            text_analysis.extract_keywords_tfidf(max_features=50)
            text_analysis.generate_word_cloud(save_path="images/word_cloud.png")
            text_analysis.perform_sentiment_analysis()
            text_analysis.topic_modeling(n_topics=5)
            text_analysis.analyze_text_length_patterns()
            
            # Generate insights
            text_insights = text_analysis.generate_text_insights()
            
            # Export report
            text_analysis.export_text_analysis_report("reports/text_analysis_report.txt")
            
            self.results['text'] = {
                'insights': text_insights,
                'success': True
            }
            
            print("✅ Text analysis completed successfully")
            
        except Exception as e:
            print(f"❌ Text analysis failed: {e}")
            self.results['text'] = {'success': False, 'error': str(e)}
            # Continue pipeline even if text analysis fails
    
    def _run_recommendations(self):
        """Run recommendations generation step."""
        try:
            # Gather insights from previous analyses
            eda_insights = self.results.get('eda', {}).get('insights', {})
            nps_insights = self.results.get('nps', {}).get('insights', {})
            text_insights = self.results.get('text', {}).get('insights', {})
            
            rec_engine = RecommendationEngine(
                self.df, 
                eda_insights=eda_insights,
                nps_insights=nps_insights,
                text_insights=text_insights
            )
            
            # Run all recommendation analyses
            customer_segments = rec_engine.identify_customer_segments()
            segmentation_recs = rec_engine.generate_segmentation_recommendations()
            market_opportunities = rec_engine.analyze_market_opportunities()
            strategic_recs = rec_engine.generate_strategic_recommendations()
            action_plan = rec_engine.create_action_plan()
            roi_projections = rec_engine.calculate_roi_projections()
            executive_summary = rec_engine.generate_executive_summary()
            
            # Export report
            rec_engine.export_recommendations_report("reports/recommendations_report.txt")
            
            self.results['recommendations'] = {
                'customer_segments': customer_segments,
                'segmentation_recommendations': segmentation_recs,
                'market_opportunities': market_opportunities,
                'strategic_recommendations': strategic_recs,
                'action_plan': action_plan,
                'roi_projections': roi_projections,
                'executive_summary': executive_summary,
                'success': True
            }
            
            print("✅ Recommendations generation completed successfully")
            
        except Exception as e:
            print(f"❌ Recommendations generation failed: {e}")
            self.results['recommendations'] = {'success': False, 'error': str(e)}
    
    def _generate_final_summary(self):
        """Generate final analysis summary."""
        try:
            summary = {
                'analysis_date': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                'dataset_info': {
                    'file_path': self.data_path,
                    'shape': self.df.shape if self.df is not None else 'N/A',
                    'columns': list(self.df.columns) if self.df is not None else []
                },
                'completed_analyses': [],
                'failed_analyses': [],
                'key_findings': [],
                'next_steps': []
            }
            
            # Check which analyses completed successfully
            for analysis_name, analysis_result in self.results.items():
                if analysis_result.get('success', False):
                    summary['completed_analyses'].append(analysis_name)
                else:
                    summary['failed_analyses'].append(analysis_name)
            
            # Extract key findings
            if self.results.get('nps', {}).get('success', False):
                nps_score = self.results['nps']['insights'].get('overall_nps', 'N/A')
                summary['key_findings'].append(f"Overall NPS Score: {nps_score}")
            
            if self.results.get('eda', {}).get('success', False):
                eda_insights = self.results['eda']['insights']
                if 'most_popular_brand' in eda_insights:
                    summary['key_findings'].append(f"Most popular brand: {eda_insights['most_popular_brand']}")
            
            if self.results.get('text', {}).get('success', False):
                text_insights = self.results['text']['insights']
                if 'avg_sentiment' in text_insights:
                    sentiment = text_insights['avg_sentiment']
                    summary['key_findings'].append(f"Average customer sentiment: {sentiment:.3f}")
            
            # Next steps
            summary['next_steps'] = [
                "Review detailed reports in the reports/ folder",
                "Examine visualizations in the images/ folder",
                "Implement recommendations based on action plan",
                "Set up monitoring for key metrics",
                "Plan follow-up analysis"
            ]
            
            self.results['final_summary'] = summary
            
            # Display summary
            print("📋 FINAL ANALYSIS SUMMARY:")
            print(f"  Date: {summary['analysis_date']}")
            print(f"  Dataset: {summary['dataset_info']['shape']} records")
            print(f"  Completed: {len(summary['completed_analyses'])} analyses")
            print(f"  Failed: {len(summary['failed_analyses'])} analyses")
            
            if summary['key_findings']:
                print("  Key Findings:")
                for finding in summary['key_findings']:
                    print(f"    • {finding}")
            
            print("✅ Final summary generated")
            
        except Exception as e:
            print(f"❌ Final summary generation failed: {e}")
    
    def _save_all_outputs(self):
        """Save all analysis outputs to files."""
        try:
            # Save complete results as JSON
            import json
            
            # Convert pandas objects to serializable format
            serializable_results = {}
            for key, value in self.results.items():
                if isinstance(value, dict):
                    serializable_results[key] = {}
                    for sub_key, sub_value in value.items():
                        if hasattr(sub_value, 'to_dict'):
                            serializable_results[key][sub_key] = sub_value.to_dict()
                        elif isinstance(sub_value, (str, int, float, bool, list, dict)):
                            serializable_results[key][sub_key] = sub_value
                        else:
                            serializable_results[key][sub_key] = str(sub_value)
            
            with open("outputs/complete_analysis_results.json", "w") as f:
                json.dump(serializable_results, f, indent=2, default=str)
            
            # Save processed dataset
            if self.df is not None:
                self.df.to_csv("outputs/processed_dataset.csv", index=False)
            
            print("✅ All outputs saved successfully")
            
        except Exception as e:
            print(f"⚠️ Failed to save some outputs: {e}")

def main():
    """Main function to run the analysis pipeline."""
    parser = argparse.ArgumentParser(description='Cosmetics Brand Analysis Pipeline')
    parser.add_argument('--data_path', type=str, default="data/Mock_Market_Research_Dataset (3).csv",
                       help='Path to the dataset file')
    parser.add_argument('--text_column', type=str, default="Open_Feedback",
                       help='Name of the text feedback column')
    parser.add_argument('--nps_column', type=str, default="NPS",
                       help='Name of the NPS score column')
    parser.add_argument('--no_save', action='store_true',
                       help='Do not save outputs to files')
    
    args = parser.parse_args()
    
    # Check if data file exists
    if not os.path.exists(args.data_path):
        print(f"❌ Data file not found: {args.data_path}")
        print("Please ensure your dataset is in the correct location.")
        
        # Suggest common locations
        common_paths = [
            "data/Mock_Market_Research_Dataset (3).csv",
            "data/cosmetics_data.csv",
            "data/survey_data.csv",
            "data/customer_data.csv"
        ]
        
        print("\nLooking for data files in common locations:")
        for path in common_paths:
            if os.path.exists(path):
                print(f"✅ Found: {path}")
                response = input(f"Use {path}? (y/n): ")
                if response.lower().startswith('y'):
                    args.data_path = path
                    break
            else:
                print(f"❌ Not found: {path}")
        
        if not os.path.exists(args.data_path):
            print("\nNo dataset found. Please:")
            print("1. Place your dataset in the data/ folder")
            print("2. Ensure it's named appropriately (cosmetics_data.csv, survey_data.csv, etc.)")
            print("3. Run the script again with the correct path")
            return
    
    # Initialize and run pipeline
    try:
        pipeline = CosmeticsBrandAnalysisPipeline(
            data_path=args.data_path,
            text_column=args.text_column,
            nps_column=args.nps_column
        )
        
        results = pipeline.run_complete_analysis(save_outputs=not args.no_save)
        
        if results:
            print("\n🎉 Analysis completed! Check the following locations for outputs:")
            print("  📊 Visualizations: images/ folder")
            print("  📋 Reports: reports/ folder")
            print("  📁 Data outputs: outputs/ folder")
            print("\n📖 Next steps:")
            print("  1. Review the comprehensive reports")
            print("  2. Examine the visualizations")
            print("  3. Implement the recommendations")
            print("  4. Set up monitoring for key metrics")
        
    except Exception as e:
        print(f"❌ Pipeline execution failed: {e}")
        print("Please check your data format and try again.")

if __name__ == "__main__":
    main()