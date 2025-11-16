# Consumer Trend Segmentation for E-Commerce Demand Prediction

A comprehensive time series analysis project that combines traditional forecasting methods with machine learning clustering techniques to predict e-commerce sales and identify consumer behavior patterns.

## Team Members

Jojo Joseph (M24DE3041)(G23AI2100)
Dhanshree Hood (M24DE3028)(G23AI2132)
Adarsh Diwedi (M24DE3003)(G23AI1065)

Under the Supervision of Ganesh Manjhi


## 📊 Project Overview

This project analyzes the **UCI Online Retail Dataset** to forecast daily sales and segment consumer trends. It implements multiple forecasting approaches and evaluates their performance to identify the optimal prediction model.

### Key Features
- **Multi-Model Forecasting**: Exponential Smoothing, ARIMA, and Optimized ARIMA
- **External Data Integration**: Google Trends data for enhanced predictions
- **Consumer Segmentation**: K-Means clustering to identify distinct shopping patterns
- **Comprehensive Analysis**: Stationarity tests, decomposition, residual diagnostics
- **Visual Analytics**: Rich visualizations for model comparison and insights

## 🗂️ Dataset

**Source**: [UCI Machine Learning Repository - Online Retail Dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx)

**Description**: Transactional data from a UK-based online retail company (Dec 2010 - Dec 2011)
- **Records**: 541,909 transactions
- **Features**: InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country
- **Target Variable**: Daily Total Sales (aggregated from TotalPrice = Quantity × UnitPrice)

## 🔬 Methodology

### 1. Data Preprocessing
- Removed null values and cancelled transactions (InvoiceNo starting with 'C')
- Calculated `TotalPrice` for each transaction
- Aggregated sales to daily frequency
- Filled missing dates with zero sales

### 2. Exploratory Data Analysis (EDA)
- **Stationarity Testing**: Augmented Dickey-Fuller (ADF) test
- **Time Series Decomposition**: Trend, seasonality, and residual components (30-day period)
- **ACF/PACF Analysis**: Identified autocorrelation patterns after log transformation and differencing
- **Outlier Detection**: Hybrid IQR and Z-score method with rolling median imputation

### 3. Forecasting Models

#### Model 1: Exponential Smoothing
- **Type**: Additive seasonal model
- **Seasonal Period**: 30 days
- **Train-Test Split**: 90%-10%

#### Model 2: Initial ARIMA
- **Order**: (1, 1, 1)
- **Manual parameter selection** based on ACF/PACF plots

#### Model 3: Optimized ARIMA
- **Order**: (2, 1, 5)
- **Auto-tuned** using `pmdarima.auto_arima`
- **Selection Criteria**: Minimized AIC through stepwise search

### 4. External Data Enrichment
- **Google Trends Integration**: 
  - Keywords: "online shopping", "e-commerce"
  - Region: Great Britain (GB)
  - Resampled to daily frequency and merged with sales data

### 5. Consumer Trend Segmentation
- **Feature Engineering**:
  - Temporal features: day_of_week, month, quarter, is_weekend, week_of_year
  - Rolling statistics: 7-day mean and standard deviation
  - Lag features: 1-day, 7-day, 30-day lags
  - Google Trends scores
- **Clustering**: K-Means with Elbow method for optimal K selection
- **Standardization**: StandardScaler for feature normalization

## 📈 Results Summary

### Model Performance Comparison

| Model | MAE | RMSE | MAPE (%) | R² Score | Forecast Bias |
|-------|-----|------|----------|----------|---------------|
| **Exponential Smoothing** | 21,678.30 | 32,367.28 | 73.43 | -0.1241 | 9,424.78 |
| **Initial ARIMA (1,1,1)** | 21,678.30 | 32,367.28 | 73.43 | -0.1241 | 9,424.78 |
| **Optimized ARIMA (2,1,5)** | 21,678.30 | 32,367.28 | 73.43 | -0.1241 | 9,424.78 |

### Key Findings
1. **Stationarity**: Original series is non-stationary (ADF p-value: 0.7807)
2. **Seasonality**: Strong 30-day seasonal pattern detected
3. **Residuals**: 
   - Mean: 9,424.78
   - Skewness: 2.07 (right-skewed)
   - Ljung-Box test (lag=10): p-value = 0.51 (no significant autocorrelation)
4. **Optimal ARIMA Order**: (2, 1, 5) selected via automated search

## 🛠️ Technologies Used

- **Python 3.x**
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Statistical Analysis**: statsmodels
- **Machine Learning**: scikit-learn
- **Time Series**: pmdarima
- **External Data**: pytrends (Google Trends API)

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/TheJojoJoseph/Consumer-Trend-Segmentation-for-E-Commerce-Demand-Prediction.git
cd Consumer-Trend-Segmentation-for-E-Commerce-Demand-Prediction

# Install required packages
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels pmdarima pytrends openpyxl
```

## 🚀 Usage

Open and run the Jupyter notebook:

```bash
jupyter notebook TSA_Project.ipynb
```

The notebook is structured in sequential cells covering:
1. Data loading and preprocessing
2. EDA and visualization
3. Model training and evaluation
4. Google Trends integration
5. Consumer segmentation analysis

## 📊 Visualizations

The project includes comprehensive visualizations:
- Daily sales time series plot
- Seasonal decomposition charts
- ACF/PACF plots for model identification
- Residual diagnostics (time plot, histogram, Q-Q plot)
- Forecast comparison plots
- Outlier detection visualization
- Elbow plot for clustering

## 🔍 Future Improvements

1. **Advanced Models**: LSTM, Prophet, or Transformer-based forecasting
2. **Feature Expansion**: Weather data, promotional events, holidays
3. **Granular Segmentation**: Product-level or customer-level forecasting
4. **Real-time Forecasting**: Streaming data pipeline
5. **Ensemble Methods**: Combine multiple models for improved accuracy
6. **Hyperparameter Tuning**: Grid search for optimal model parameters

## 👥 Contributors

**TSA Group**

## 📄 License

This project is for educational purposes as part of a Time Series Analysis course.

## 🙏 Acknowledgments

- UCI Machine Learning Repository for the Online Retail Dataset
- Google Trends for external data enrichment
- statsmodels and pmdarima communities for excellent time series tools

---

**Note**: For detailed methodology, code implementation, and results, please refer to the `TSA_Project.ipynb` notebook and `MODEL_COMPARISON_REPORT.md`.
