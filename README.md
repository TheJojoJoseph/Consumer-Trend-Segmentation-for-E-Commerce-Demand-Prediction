# Consumer Trend Segmentation for E-Commerce Demand Prediction

Time series forecasting and consumer segmentation on UCI Online Retail Dataset using Exponential Smoothing, ARIMA, and K-Means clustering.

## Team Members

Jojo Joseph (M24DE3041)(G23AI2100)  
Dhanshree Hood (M24DE3028)(G23AI2132)  
Adarsh Diwedi (M24DE3003)(G23AI1065)  
Under the Supervision of Ganesh Manjhi

## 📊 Overview

**Dataset**: UCI Online Retail (Dec 2010 - Dec 2011)  
**Records**: 541,909 transactions from UK-based retailer  
**Target**: Daily total sales forecasting  
**External Data**: Google Trends ("online shopping", "e-commerce")

## 🔬 Models Compared

### Exponential Smoothing (Holt-Winters)
- Simple, fast baseline
- Additive seasonal model (30-day period)
- Works well with stable seasonality

### ARIMA (1,1,1)
- Manual parameter selection via ACF/PACF
- Captures autocorrelation
- Easy to interpret

### Optimized ARIMA (2,1,5)
- Auto-tuned via `pmdarima.auto_arima`
- Minimized AIC through stepwise search
- More complex but no accuracy gain

## 📈 Performance Results

**All models produced identical performance:**

| Metric | Value |
|--------|-------|
| MAE | 21,678 |
| RMSE | 32,367 |
| MAPE | 73% |
| R² | -0.12 |
| Bias | +9,425 |

**Interpretation**: High errors and negative R² indicate models fail to capture true demand patterns and systematically over-predict.

## 🔍 Key Challenges

- **Limited data**: Only 1 year (insufficient seasonal cycles)
- **High volatility**: Irregular sales patterns with many outliers
- **Short test window**: 38 days only
- **Univariate approach**: No external factors modeled yet
- **Residuals**: Right-skewed, heavy-tailed errors (though no autocorrelation)

## 🎯 Consumer Segmentation

**Features**: Temporal (day, month, weekend), rolling stats (7-day mean/std), lags (1, 7, 30), Google Trends  
**Method**: K-Means clustering  
**Result**: 3-4 demand clusters identified (high-demand, normal, low-demand, promotions)

## 💡 Recommendations

### Immediate
- Ensemble the three models
- Stronger outlier handling
- Extend test window
- Add prediction intervals

### Next 1-3 Months
- SARIMAX with Google Trends
- Cluster-specific forecasting
- Holiday/promotion features
- Time-series cross-validation

### Long Term
- LSTM/GRU, Transformers, Prophet
- Hierarchical forecasts
- Real-time pipelines and monitoring

## 🎯 Expected Business Impact

With improvements:
- **MAPE**: 73% → <30%
- **Forecast bias**: Reduced systematic over-prediction
- **Operations**: Better inventory and staffing planning
- **Marketing**: Stronger ROI through demand-based segmentation

## 🛠️ Technologies

Python 3.x • pandas • numpy • matplotlib • seaborn • statsmodels • scikit-learn • pmdarima • pytrends

## 📦 Installation

```bash
git clone https://github.com/TheJojoJoseph/Consumer-Trend-Segmentation-for-E-Commerce-Demand-Prediction.git
cd Consumer-Trend-Segmentation-for-E-Commerce-Demand-Prediction
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels pmdarima pytrends openpyxl
```

## 🚀 Usage

```bash
jupyter notebook main_group11.ipynb
```

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
