# CashFlow Predictor

AI-powered cashflow prediction tool for small businesses that analyzes historical financial data to forecast future cash positions.

## Features

- **Data Upload**: Support for Excel (.xlsx, .xls) and CSV files
- **Smart Analysis**: Auto-detects date and amount columns
- **AI Predictions**: Linear regression-based forecasting for 30, 60, and 90 days
- **Visualizations**: Interactive charts with cashflow trends and predictions
- **Key Metrics**: Current balance, predicted low/high cash positions

## Tech Stack

- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Plotly
- OpenPyXL

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
streamlit run app.py
```

## Upload Format

Upload a file with columns for dates and amounts (e.g., date, amount, balance, income, expense). The app will auto-detect the relevant columns.

## License

MIT
