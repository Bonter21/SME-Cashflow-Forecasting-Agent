# CashFlow Predictor

AI-powered cashflow prediction tool for small businesses that analyzes historical financial data to forecast future cash positions.

## Overview

CashFlow Predictor is a web-based application built with Streamlit that helps small business owners, accountants, and financial managers forecast their future cash positions. By analyzing historical transaction data, the app uses machine learning to predict cashflow trends and provides actionable insights.

## Live Demo

Visit: [CashFlow Predictor Live App](https://sme-cashflow-forecasting-agent.streamlit.app)

## Features

### 1. Data Upload & Processing
- **Multiple File Formats**: Supports `.csv`, `.xlsx`, and `.xls` files
- **Auto-Detection**: Automatically identifies date and amount columns from common column names
- **Data Validation**: Cleans and validates data, handling missing values and outliers
- **Daily Aggregation**: Groups transactions by day for accurate analysis

### 2. Financial Analytics
- **Current Balance**: Displays real-time current balance from uploaded data
- **Average Daily Change**: Shows typical daily cash movement
- **Total Income/Expense**: Summarizes income and expense totals
- **30-Day Rolling View**: Visualizes recent cashflow trends

### 3. AI-Powered Predictions
- **Linear Regression Model**: Uses scikit-learn's LinearRegression for forecasting
- **Flexible Timeframes**: Choose from 30, 60, or 90-day prediction periods
- **Confidence Intervals**: Shows predicted high/low balance ranges
- **Trend Analysis**: Identifies whether cashflow is trending up or down

### 4. Visualizations
- **Cashflow Timeline Chart**: Interactive Plotly chart showing historical and predicted balance
- **Daily Cashflow Bar Chart**: Color-coded bars (green for income, red for expenses)
- **Data Preview**: Scrollable table with recent transactions

### 5. Smart Conclusions
- **Status Assessment**: Categorizes cashflow as Excellent, Good, Stable, Warning, or Critical
- **Change Analysis**: Calculates projected balance change with percentage
- **Recommendations**: Provides actionable suggestions based on predictions
- **Volatility Detection**: Warns about high volatility in cashflow

### 6. Report Generation
- **Forecast CSV**: Download predicted cashflow as CSV file
- **Conclusion Report**: Download detailed text report with AI analysis

## Supported Data Formats

The app automatically detects columns with these common names:

| Column Type | Supported Names |
|------------|-----------------|
| Date | `date`, `transaction_date`, `trans_date`, `timestamp`, `datetime` |
| Amount | `amount`, `balance`, `value`, `total`, `sum`, `net_amount` |
| Income | `income`, `revenue`, `credit`, `inflow`, `sales` |
| Expense | `expense`, `expenses`, `debit`, `outflow`, `cost` |

### Sample Data Format

| Date | Amount |
|------|--------|
| 2024-01-01 | 5000 |
| 2024-01-02 | -1200 |
| 2024-01-03 | 800 |
| 2024-01-04 | -2500 |
| 2024-01-05 | 1500 |

> **Note**: Positive amounts = income, negative amounts = expenses

## How It Works

### Step 1: Data Upload
Drop your Excel or CSV file in the sidebar. The app accepts files with date and amount columns.

### Step 2: Auto-Processing
The app:
1. Reads and cleans the data
2. Converts dates to datetime format
3. Removes invalid rows
4. Sorts by date
5. Calculates running balance

### Step 3: AI Prediction
1. Aggregates data by day
2. Trains a linear regression model on historical daily changes
3. Projects future daily changes based on trend
4. Calculates cumulative predicted balance

### Step 4: Analysis & Display
1. Generates key metrics and visualizations
2. Creates AI conclusions with recommendations
3. Allows downloading reports

## Prediction Algorithm

The app uses **Linear Regression** for predictions:

```
future_balance = last_balance + Σ(predicted_daily_change × days)
```

- **Slope**: Represents average daily cashflow trend
- **Positive slope** = Cashflow trending up
- **Negative slope** = Cashflow trending down

## Status Categories

| Status | Condition |
|--------|-----------|
| Excellent | Trend up, >20% increase |
| Good | Trend up, stable increase |
| Stable | Minimal change |
| Warning | Trend down, <20% decrease |
| Critical | Trend down, >20% decrease OR negative balance |

## 📱 Mobile Support

### Progressive Web App (PWA)

The app is mobile-optimized and can be installed on your phone:

1. Open the app in Chrome/Safari on your phone
2. Tap the **Share/Menu** button
3. Select **"Add to Home Screen"**
4. The app appears like a native mobile app!

### Mobile Features

- ✅ Responsive design for all screen sizes
- ✅ Touch-optimized buttons (44px min)
- ✅ Dark mode support
- ✅ Mobile file upload
- ✅ Download reports
- ✅ Email notifications
- ✅ Collapsible sidebar
- ✅ Sticky headers

### Convert to App Store

For full iOS/Android app store deployment, use:
- [PWABuilder](https://pwabuilder.com) - Free conversion
- [FlutterFlow](https://flutterflow.io) - No-code builder
- [Thunkable](https://thunkable.com) - Cross-platform

## Tech Stack

- **Frontend**: Streamlit (Mobile-Ready)
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn, StatsModels (ARIMA)
- **Visualization**: Plotly
- **Excel Processing**: OpenPyXL
- **PDF Reports**: ReportLab
- **Email**: SMTP

## Installation

```bash
# Clone the repository
git clone https://github.com/Bonter21/SME-Cashflow-Forecasting-Agent.git
cd SME-Cashflow-Forecasting-Agent

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## Deployment

This app is deployed on Streamlit Community Cloud. Connect your GitHub repository for free hosting.

## Requirements

```
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
plotly>=5.18.0
openpyxl>=3.1.0
```

## Use Cases

1. **Small Business Owners**: Anticipate cash flow gaps
2. **Accountants**: Provide clients with quick forecasts
3. **Financial Managers**: Plan for large expenses
4. **Startup Founders**: Track burn rate and runway

## Limitations

- Requires at least 7 days of historical data for predictions
- Linear regression assumes constant trend (may not capture seasonality)
- Predictions are for reference only - consult financial advisors for major decisions

## License

MIT

## Author

Built by Bonter21