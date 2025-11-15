# Usage Guide - Cosmetics Brand Analysis

## 🚀 Quick Start

### 1. Running the Complete Analysis
The easiest way to run the analysis is using the main pipeline script:

```bash
python main.py --data_path "data/your_dataset.csv"
```

This will automatically:
- Load and clean your data
- Perform exploratory data analysis
- Analyze Net Promoter Scores
- Process text feedback
- Generate recommendations
- Create all visualizations and reports

### 2. Command Line Options

```bash
python main.py --help
```

Available options:
- `--data_path`: Path to your dataset (default: "data/cosmetics_data.csv")
- `--text_column`: Name of text feedback column (default: "Open_Feedback")
- `--nps_column`: Name of NPS score column (default: "NPS_Score")
- `--no_save`: Don't save outputs to files

Example with custom parameters:
```bash
python main.py --data_path "data/survey_2024.csv" --text_column "Customer_Feedback" --nps_column "Satisfaction_Score"
```

## 📊 Individual Module Usage

### Data Preparation
```python
from analysis import DataPreparation

# Initialize and load data
data_prep = DataPreparation("data/your_file.csv")
df = data_prep.load_data()

# Generate data quality report
data_prep.generate_data_overview()
data_prep.check_data_completeness()
data_prep.detect_outliers()

# Create visualizations
data_prep.create_data_profile_visualization("images/data_profile.png")
```

### Exploratory Data Analysis
```python
from analysis import ExploratoryDataAnalysis

# Initialize EDA
eda = ExploratoryDataAnalysis(df)

# Run specific analyses
eda.demographic_analysis()
eda.brand_preference_analysis()
eda.correlation_analysis()
eda.sustainability_vs_price_analysis()

# Create interactive dashboard
eda.create_interactive_dashboard()

# Get insights
insights = eda.generate_eda_insights()
```

### NPS Analysis
```python
from analysis import NPSAnalysis

# Initialize NPS analysis
nps = NPSAnalysis(df, nps_column='NPS_Score')

# Calculate segments
segments = nps.calculate_nps_segments()

# Analyze drivers
nps.analyze_nps_drivers()

# Demographic breakdown
nps.nps_by_demographics()

# Build prediction model
model_results = nps.build_nps_prediction_model()
```

### Text Analysis
```python
from analysis import TextAnalysis

# Initialize text analysis
text_analyzer = TextAnalysis(df, text_column='Open_Feedback')

# Preprocess texts
text_analyzer.preprocess_all_texts()

# Extract keywords
keywords = text_analyzer.extract_keywords_tfidf()

# Generate word cloud
text_analyzer.generate_word_cloud()

# Sentiment analysis
sentiments = text_analyzer.perform_sentiment_analysis()

# Topic modeling
topics = text_analyzer.topic_modeling(n_topics=5)
```

### Recommendations
```python
from analysis import RecommendationEngine

# Gather insights from other analyses
eda_insights = eda.generate_eda_insights()
nps_insights = nps.generate_nps_insights()
text_insights = text_analyzer.generate_text_insights()

# Initialize recommendation engine
rec_engine = RecommendationEngine(df, eda_insights, nps_insights, text_insights)

# Generate recommendations
segments = rec_engine.identify_customer_segments()
recommendations = rec_engine.generate_segmentation_recommendations()
opportunities = rec_engine.analyze_market_opportunities()
strategy = rec_engine.generate_strategic_recommendations()
action_plan = rec_engine.create_action_plan()
```

## 📁 Output Files

After running the analysis, you'll find:

### Reports (reports/ folder)
- `data_quality_report.txt` - Data quality assessment
- `eda_report.txt` - Exploratory data analysis findings
- `nps_analysis_report.txt` - NPS analysis results
- `text_analysis_report.txt` - Text analysis insights
- `recommendations_report.txt` - Strategic recommendations

### Visualizations (images/ folder)
- `data_profile.png` - Data quality visualizations
- `gender_distribution.png` - Demographic analysis
- `brand_preference_by_gender.png` - Brand preferences
- `correlation_matrix.png` - Variable correlations
- `nps_analysis.png` - NPS score analysis
- `word_cloud.png` - Customer feedback themes
- `sentiment_analysis.png` - Sentiment patterns
- And many more...

### Data Outputs (outputs/ folder)
- `processed_dataset.csv` - Cleaned and processed data
- `complete_analysis_results.json` - All analysis results in JSON format

## 🛠️ Customization

### Adding Custom Analysis
Create a new file in the `analysis/` folder:

```python
# analysis/custom_analysis.py
class CustomAnalysis:
    def __init__(self, dataframe):
        self.df = dataframe
    
    def your_custom_method(self):
        # Your analysis logic here
        pass
```

Then import and use it:
```python
from analysis.custom_analysis import CustomAnalysis
custom = CustomAnalysis(df)
custom.your_custom_method()
```

### Modifying Visualizations
Most visualization functions accept parameters:

```python
# Customize word cloud
text_analyzer.generate_word_cloud(
    max_words=200,
    save_path="images/custom_wordcloud.png"
)

# Customize EDA plots
eda.demographic_analysis(save_plots=True)
```

## 🔧 Troubleshooting

### Common Issues

1. **Module not found error**
   ```bash
   # Ensure you're in the project directory
   cd Case-Study-Project
   python main.py
   ```

2. **Data file not found**
   ```bash
   # Check your data path
   python main.py --data_path "correct/path/to/your/file.csv"
   ```

3. **Missing dependencies**
   ```bash
   # Reinstall requirements
   pip install -r requirements.txt
   ```

4. **Column not found**
   ```bash
   # Check your column names
   python main.py --text_column "YourTextColumn" --nps_column "YourNPSColumn"
   ```

### Data Format Requirements

Your dataset should be a CSV file with:
- At least one text column for feedback analysis
- Numerical NPS scores (0-10) if doing NPS analysis
- Demographic columns (Gender, Age, Income_Level, etc.)
- Brand preference information

Example format:
```csv
Customer_ID,Age,Gender,Income_Level,Brand_Preference,NPS_Score,Open_Feedback
C001,28,Female,Medium,Brand_A,8,"Great quality products"
C002,35,Male,High,Brand_B,9,"Love the sustainability focus"
```

## 📞 Support

If you encounter issues:

1. Check this usage guide
2. Review the error messages carefully
3. Ensure your data format matches expectations
4. Check that all required columns exist
5. Verify Python dependencies are installed

## 🎯 Best Practices

1. **Start with the complete pipeline** using `main.py`
2. **Review data quality** before running advanced analyses
3. **Check visualization outputs** to validate results
4. **Read the generated reports** for detailed insights
5. **Use individual modules** for focused analysis or debugging
6. **Save intermediate results** for reproducibility
7. **Document any customizations** you make

## 📈 Next Steps

After completing the analysis:

1. Review all generated reports and visualizations
2. Implement the strategic recommendations
3. Set up monitoring for key metrics
4. Plan follow-up analysis with new data
5. Share insights with stakeholders
6. Update the analysis as you collect more data