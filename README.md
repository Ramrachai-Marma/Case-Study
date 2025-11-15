# 🧴 Cosmetics Brand Analysis Case Study

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)](https://jupyter.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-green)](https://pandas.pydata.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-Visualization-lightblue)](https://seaborn.pydata.org/)

## 📋 Project Overview

This project presents a comprehensive data-driven analysis of consumer preferences and brand perceptions in the cosmetics industry. Through exploratory data analysis, sentiment analysis, and natural language processing, we derive actionable marketing insights for cosmetics brands.

## 🎯 Objectives

1. **Data Preparation & Quality Assessment**: Comprehensive data cleaning and overview
2. **Exploratory Data Analysis (EDA)**: Understanding demographic patterns and brand preferences
3. **Net Promoter Score (NPS) Analysis**: Identifying factors that drive customer loyalty
4. **Text Analytics**: Analyzing customer expectations through NLP techniques
5. **Strategic Recommendations**: Data-driven marketing insights and customer segmentation

## 📊 Key Findings

### Customer Segmentation Insights
- **Income-based targeting**: High-income customers prefer premium products, while price-sensitive segments value affordability
- **Gender preferences**: Distinct brand preferences between male and female customers
- **Sustainability consciousness**: Strong correlation between eco-friendly values and brand loyalty

### Net Promoter Score Analysis
- Brand reputation and customer service are primary drivers of high NPS scores
- Income level significantly influences brand loyalty patterns
- Sustainability initiatives positively impact customer advocacy

### Text Analysis Results
- Customer expectations focus on quality, sustainability, and brand reputation
- Sentiment analysis reveals predominantly neutral to positive customer feedback
- Key themes: product efficacy, environmental responsibility, value for money

## 🗂️ Project Structure

```
Case-Study-Project/
│
├── 📓 notebooks/
│   └── cosmetics_brand_analysis.ipynb    # Original analysis notebook
│
├── 🔬 analysis/                           # Modular analysis components
│   ├── __init__.py                        # Package initialization
│   ├── data_preparation.py               # Data loading and cleaning
│   ├── exploratory_data_analysis.py      # EDA and visualizations
│   ├── nps_analysis.py                   # Net Promoter Score analysis
│   ├── text_analysis.py                  # NLP and sentiment analysis
│   └── recommendations.py                # Insights and recommendations
│
├── 📁 data/
│   ├── README.md                          # Data guidelines
│   └── (place your datasets here)
│
├── 🐍 src/
│   ├── __init__.py                        # Package initialization
│   └── utils.py                           # Utility functions
│
├── 📖 docs/
│   ├── methodology.md                     # Detailed methodology
│   └── findings.md                        # Comprehensive findings
│
├── 🖼️ images/                             # Generated visualizations
├── 📋 reports/                            # Analysis reports
├── 📁 outputs/                            # Processed data and results
│
├── main.py                                # Main analysis pipeline
├── setup.py                               # Project setup script
├── requirements.txt                       # Python dependencies
├── .gitignore                            # Git ignore rules
├── LICENSE                               # MIT License
└── README.md                             # This file
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- Jupyter Notebook or JupyterLab (optional)
- Git (optional, for version control)

### Installation

1. **Clone the repository** (if using Git):
   ```bash
   git clone <repository-url>
   cd Case-Study-Project
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the setup script**:
   ```bash
   python setup.py
   ```

4. **Run the complete analysis**:
   ```bash
   python main.py
   ```
   Or with a custom data file:
   ```bash
   python main.py --data_path "data/your_file.csv"
   ```
   
   **Note**: The default data file is `data/Mock_Market_Research_Dataset (3).csv`

### Alternative: Jupyter Notebook
If you prefer working with notebooks:
```bash
jupyter notebook
# Navigate to notebooks/cosmetics_brand_analysis.ipynb
```

## 📊 Usage Options

### Option 1: Complete Pipeline (Recommended)
Run the entire analysis with a single command:
```bash
python main.py --data_path "data/cosmetics_data.csv" --text_column "Open_Feedback"
```

### Option 2: Individual Analysis Modules
Run specific analysis components:
```python
from analysis import DataPreparation, ExploratoryDataAnalysis, NPSAnalysis

# Data preparation
data_prep = DataPreparation("data/your_file.csv")
df = data_prep.load_data()

# Run specific analyses
eda = ExploratoryDataAnalysis(df)
eda.demographic_analysis()
```

### Option 3: Interactive Notebook
Open and run the Jupyter notebook for step-by-step analysis:
```bash
jupyter notebook notebooks/cosmetics_brand_analysis.ipynb
```

## 📈 Methodology

### 1. Data Preparation
- **Data Quality Assessment**: Completeness, duplicates, outliers detection
- **Variable Analysis**: Data types, distributions, missing values
- **Data Cleaning**: Standardization and preprocessing

### 2. Exploratory Data Analysis
- **Demographic Analysis**: Brand preferences across different groups
- **Correlation Analysis**: Factors influencing Net Promoter Score
- **Comparative Analysis**: Sustainability vs. price importance

### 3. Text Analytics
- **Preprocessing**: Text cleaning and normalization
- **Keyword Extraction**: TF-IDF analysis for key themes
- **Sentiment Analysis**: VADER sentiment scoring
- **Insights Generation**: Pattern identification and interpretation

### 4. Visualization Techniques
- Heatmaps for correlation analysis
- Bar plots for categorical comparisons
- Box plots for distribution analysis
- Word clouds for text visualization

## 💡 Marketing Recommendations

### 1. Segment-Specific Strategies
- **Premium Targeting**: Exclusive offers for high-income segments
- **Value Positioning**: Discount strategies for price-sensitive customers
- **Gender-Based Marketing**: Customized messaging and channels

### 2. Brand Building Initiatives
- **Sustainability Communication**: Eco-friendly initiatives promotion
- **Customer Experience**: Enhanced service quality and support
- **Influencer Partnerships**: Authentic brand storytelling

### 3. Product Development
- **Premium Collections**: High-quality products for affluent segments
- **Value Lines**: Affordable options without compromising quality
- **Sustainable Products**: Eco-conscious product development

## 🛠️ Technologies Used

- **Python**: Core programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Matplotlib & Seaborn**: Data visualization
- **Scikit-learn**: Machine learning and text processing
- **NLTK/VADER**: Natural language processing and sentiment analysis
- **Jupyter Notebook**: Interactive development environment

## 📝 Usage Examples

### Running the Analysis
```python
# Load and explore the data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
df = pd.read_csv('data/your_dataset.csv')

# Run the analysis as shown in the notebook
```

### Key Visualizations
The notebook includes:
- Brand preference heatmaps
- NPS score distributions
- Demographic analysis charts
- Text analysis word clouds
- Correlation matrices

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/enhancement`)
3. Commit your changes (`git commit -m 'Add some enhancement'`)
4. Push to the branch (`git push origin feature/enhancement`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.



## 🙏 Acknowledgments

- Data science community for methodological insights
- Open-source contributors for the amazing tools and libraries
- Cosmetics industry research for domain knowledge

---

⭐ **Star this repository if you found it helpful!**
