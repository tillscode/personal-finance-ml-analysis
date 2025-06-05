# Personal Finance ML Analysis

## 🎯 Project Overview

This project analyzes 15+ months of personal financial data from multiple sources to build predictive models for expense forecasting and spending pattern analysis. The project demonstrates advanced data science skills including data integration, feature engineering, machine learning, and interactive visualization.

### Key Achievements
- **Integrated data from 3 financial sources** (1,571+ transactions)
- **Built ML models with ±$580 prediction accuracy** using scikit-learn
- **Developed interactive dashboard** with Plotly for insights visualization
- **Detected spending anomalies** (travel months) with business context
- **Created actionable financial insights** for budget planning

## 📊 Key Results

| Metric | Value |
|--------|-------|
| **ML Model Accuracy** | ±$580 MAE |
| **Data Sources Integrated** | 3 (Wells Fargo, Discover, Chase) |
| **Transactions Analyzed** | 1,571+ |
| **Analysis Period** | 15 months |
| **Variance Explained (R²)** | 0.847 |
| **Top Spending Category** | Restaurants ($2,598) |

## 🛠️ Technical Stack

**Data Processing & Analysis:**
- **Python** - Core programming language
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing

**Machine Learning:**
- **scikit-learn** - ML algorithms and evaluation
- **Ridge Regression** - Best performing model
- **Random Forest** - Feature importance analysis

**Visualization:**
- **Plotly** - Interactive dashboard creation
- **Matplotlib** - Statistical visualizations

## ⚡ Quick Start

### Prerequisites
```bash
Python 3.8+
pip package manager
```

### Installation
```bash
# Clone the repository
git clone https://github.com/tillscode/personal-finance-ml-analysis.git
cd personal-finance-ml-analysis

# Install dependencies
pip install -r requirements.txt
```

### Usage
```bash
# 1. Process and clean data
python 1_data_processing.py

# 2. Train ML models
python 2_ml_modeling.py

# 3. Create interactive dashboard
python 3_interactive_dashboard.py
```

## 📈 Analysis Highlights

### 🔍 Data Integration
- **Multi-source consolidation** from 3 financial institutions
- **Standardized data formats** across different CSV structures
- **Data quality validation** with comprehensive error handling

### 🧠 Feature Engineering
- **Time-based features** (seasonality, weekday/weekend patterns)
- **Lagged variables** (previous month spending for prediction)
- **Anomaly detection** (travel months with 2x normal spending)
- **Transaction patterns** (frequency, average amounts, volatility)

### 🎯 Machine Learning
- **Model comparison** (Linear, Ridge, Random Forest regression)
- **Cross-validation** for robust performance evaluation
- **Feature importance analysis** (avg transaction amount most predictive)
- **Future predictions** (3-month spending forecasts)

### 📊 Key Insights Discovered
1. **Seasonal spending patterns** - October peak ($10K+ for Japan trip)
2. **Restaurant spending dominance** - 91 transactions, $2,598 total
3. **Predictable baseline** - Normal months: $3K-$6K range
4. **Weekend spending** differs significantly from weekdays
5. **Transaction frequency** correlates strongly with total spending

## 💼 Business Value & Applications

### Immediate Applications
- **Budget forecasting** for upcoming months
- **Spending anomaly detection** for unusual patterns
- **Category optimization** for expense reduction
- **Seasonal planning** for high-spending periods

### Technical Skills Demonstrated
- **End-to-end data science pipeline** development
- **Real-world data integration** and cleaning
- **Advanced feature engineering** techniques
- **Machine learning model selection** and evaluation
- **Interactive visualization** and dashboard creation
- **Business insight generation** from technical analysis

## 📧 Contact

**Tyler Nguyen** - MIS Student & Data Analyst Enthusiast

- 📧 Email: work.tylernguyen@gmail.com
- 💼 LinkedIn: [linkedin.com/in/nguyen-tyler](https://www.linkedin.com/in/nguyen-tyler/)
- 📱 GitHub: [@tillscode](https://github.com/tillscode)

