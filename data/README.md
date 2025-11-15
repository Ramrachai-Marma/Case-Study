# Data Directory

This directory contains the datasets used in the cosmetics brand analysis.

## Data Structure

### Expected Dataset Format

The analysis expects a CSV file with the following columns:

#### Required Columns:
- **Age**: Customer age (numerical)
- **Gender**: Customer gender (categorical: Male/Female)
- **Income_Level**: Income bracket (categorical)
- **NPS** or **NPS_Score**: Net Promoter Score (numerical: 0-10)
- **Open_Feedback**: Open-ended customer feedback (text)

#### Current Dataset Columns:
Based on the actual dataset (`Mock_Market_Research_Dataset (3).csv`):
- **Age**: Customer age (numerical)
- **Gender**: Customer gender (categorical: Male/Female)
- **Income_Level**: Income bracket (categorical: <30k, 30k-60k, 60k-100k, etc.)
- **Location**: Customer location (categorical)
- **Purchase_Frequency**: Purchase frequency (categorical: Weekly, Monthly, etc.)
- **Brand_Preferences**: Preferred cosmetics brand(s) - can be comma-separated (categorical)
- **Importance_Price**: Price importance rating (numerical: 1-5)
- **Importance_Sustainability**: Sustainability importance rating (numerical: 1-5)
- **Importance_Brand_Image**: Brand image importance rating (numerical: 1-5)
- **NPS**: Net Promoter Score (numerical: 0-10)
- **Open_Feedback**: Open-ended customer feedback (text)

#### Optional Columns:
- **Customer_ID**: Unique identifier for each customer (will be auto-generated if missing)
- **Product_Category**: Primary product category of interest
- **Marketing_Channel**: Preferred marketing channel
- **Geographic_Region**: Customer location

## Data Quality Guidelines

### Data Preparation Checklist:
- [ ] Remove duplicate records
- [ ] Handle missing values appropriately
- [ ] Validate data types
- [ ] Check for outliers in numerical columns
- [ ] Standardize categorical values
- [ ] Clean text data (remove special characters, standardize format)

### Sample Data Format:

```csv
Customer_ID,Age,Gender,Income_Level,Brand_Preference,NPS_Score,Sustainability_Importance,Price_Sensitivity,Open_Feedback
C001,28,Female,Medium,Brand_A,8,High,Medium,"I love products that are eco-friendly and effective"
C002,35,Male,High,Brand_B,9,Medium,Low,"Quality is most important to me, willing to pay premium"
C003,22,Female,Low,Brand_C,6,High,High,"Need affordable options that don't compromise on ethics"
```

## Data Privacy and Ethics

- All customer data should be anonymized
- Personal identifiable information (PII) should be removed
- Ensure compliance with data protection regulations (GDPR, CCPA, etc.)
- Use appropriate data handling and storage practices

## Data Sources

This analysis can work with data from various sources:
- Customer surveys
- Online reviews and feedback
- CRM systems
- Social media sentiment
- Purchase transaction data
- Market research studies

## File Naming Convention

Use descriptive filenames:
- `cosmetics_survey_2024.csv` - Primary survey data
- `customer_feedback_raw.csv` - Raw text feedback
- `nps_scores_quarterly.csv` - NPS tracking data
- `demographics_cleaned.csv` - Processed demographic data

## Data Backup and Versioning

- Keep original raw data files unchanged
- Create dated versions for processed data
- Document all data transformations
- Maintain data lineage and processing logs