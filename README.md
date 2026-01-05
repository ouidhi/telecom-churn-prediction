# Customer Churn Prediction Analysis

A machine learning project analyzing customer churn patterns in the telecommunications industry using advanced predictive modeling techniques. 

## Project Overview

This project develops and compares multiple machine learning models to predict customer churn using an Iranian telecom dataset. The analysis encompasses comprehensive data preprocessing, feature engineering, and model evaluation to identify key factors influencing customer retention.

**Key Achievement**: XGBoost model achieved **93.6% PR-AUC** and **86.6% F1-Score**, demonstrating strong performance on imbalanced churn prediction.

## Objectives

- Analyze customer behavior patterns and identify key churn indicators
- Develop and compare multiple machine learning classification models
- Evaluate model performance using metrics appropriate for imbalanced data
- Provide actionable insights for customer retention strategies

## Dataset

**Source**: [Iranian Telecommunications Company](https://www.kaggle.com/datasets/alinoranianesfahani/iranian-churn-dataset)  
**Size**: 3,150 customer records  
**Features**: 14 variables including usage patterns, service characteristics, and demographics  
**Target**: Binary churn indicator (84% retained, 16% churned)

### Feature Categories

- **Usage Metrics**: Seconds of Use, Frequency of Use, Frequency of SMS
- **Service Quality**: Call Failures, Complaints
- **Customer Profile**: Age Group, Subscription Length, Customer Value
- **Account Details**: Tariff Plan, Status, Charge Amount
- **Engagement**: Distinct Called Numbers

## Interactive Dashboard

Explore the interactive visualizations and insights from this analysis:
[View Tableau Dashboard](https://public.tableau.com/views/CustomerChurnPrediction_17645483863320/Story1?:language=en-US&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)

The dashboard includes:

- Customer churn distribution and trends
- Feature importance visualizations
- Model performance comparisons
- Interactive filters for demographic analysis
- Key business insights and recommendations

## Technical Approach

### Data Preprocessing

1. **Data Quality**: Complete dataset with no missing values
2. **Feature Engineering**: 
   - Removed redundant features (Age → Age Group)
   - Standardized column naming conventions
3. **Skewness Handling**:
   - Log transformation: Frequency of SMS, Charge Amount
   - Square root transformation: Customer Value, Seconds of Use, Frequency of Use, Call Failures
4. **Feature Set**: Expanded from 14 to 21 features (original + transformed)

### Model Development

**Train-Test Split**: 80-20 with stratification to maintain class balance

**Models Implemented**:
1. **Logistic Regression** - Linear baseline for interpretability
2. **Random Forest** - Ensemble method for non-linear relationships
3. **Support Vector Machine (SVM)** - Complex decision boundaries
4. **XGBoost** - Advanced gradient boosting with regularization

**Optimization**: RandomizedSearchCV with 5-fold cross-validation, optimizing for PR-AUC

## Results

### Model Performance Comparison

| Model | F1-Score | ROC-AUC | PR-AUC | Optimal Threshold |
|-------|----------|---------|--------|-------------------|
| **XGBoost** | **0.866** | **0.985** | **0.936** | 0.722 |
| Random Forest | 0.870 | 0.978 | 0.880 | 0.570 |
| SVM | 0.839 | 0.967 | 0.866 | 0.451 |
| Logistic Regression | 0.664 | 0.924 | 0.762 | 0.643 |

### Key Findings

**Best Model**: XGBoost achieved the highest PR-AUC (93.6%), making it the recommended production model for imbalanced churn prediction.

**Top Churn Predictors** (correlation with churn):
- Complaints filed (+0.532) - Strongest single predictor
- Account Status issues (+0.499) - Billing/service problems
- Low Frequency of Use (-0.353) - Declining engagement
- Low Seconds of Use (-0.357) - Reduced service dependency

**Model Insights**:
- Tree-based models outperformed linear models by 15-20% on F1-score
- Feature engineering improved performance across all models
- Optimal prediction thresholds varied significantly from default 0.5
- XGBoost's high threshold (0.72) prioritizes precision while maintaining strong recall

## Technologies Used

- **Python 3.x**
- **Data Analysis**: pandas, numpy
- **Machine Learning**: scikit-learn, XGBoost
- **Visualization**: matplotlib, seaborn
- **Model Evaluation**: classification metrics, ROC/PR curves

## Business Insights

### Actionable Recommendations

1. **Complaint Management**: Implement rapid response system for customer complaints (strongest churn predictor)
2. **Account Health Monitoring**: Flag accounts with billing/status issues for proactive intervention
3. **Usage Tracking**: Monitor declining engagement patterns as early warning signals
4. **Targeted Retention**: Focus retention efforts on high-value customers showing risk indicators

### Deployment Considerations

- **Model Choice**: XGBoost recommended for production (best PR-AUC)
- **Threshold**: Use optimized threshold (0.722) rather than default 0.5
- **Monitoring**: Track model performance on new data to detect drift
- **Interpretability**: Leverage feature importance for explainable predictions

## Evaluation Metrics

**Primary Metric**: PR-AUC (Precision-Recall Area Under Curve)
- Most appropriate for imbalanced datasets
- Focuses on minority class (churners) performance
- Not inflated by majority class predictions

**Supporting Metrics**:
- F1-Score: Balances precision and recall
- ROC-AUC: Overall discrimination ability
- Precision: Minimizes false positive retention campaigns
- Recall: Maximizes identification of at-risk customers

## Contact ⋆˙⟡

🔗 [LinkedIn](https://www.linkedin.com/in/vidhi-parmar1/) | [Email](vidhi30th@gmail.com) | [Website](https://readymag.website/5667522/)
