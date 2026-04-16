# CashFlow Predictor - AI Agent for Small Businesses

## 1. Project Overview
- **Project Name**: CashFlow Predictor
- **Type**: Web Application (Streamlit)
- **Core Functionality**: AI-powered cashflow prediction tool for small businesses that analyzes historical financial data to forecast future cash positions
- **Target Users**: Small business owners, accountants, financial managers

## 2. UI/UX Specification

### Layout Structure
- **Header**: Logo/title section with business name input
- **Main Content**: 
  - Sidebar for data upload and settings
  - Main dashboard area for visualizations and predictions
- **Footer**: Minimal with app info

### Visual Design
- **Color Palette**:
  - Primary: `#228B22` (Forest Green)
  - Secondary: `#32CD32` (Lime Green)
  - Accent: `#90EE90` (Light Green)
  - Background: `#FFFFFF` (White)
  - Text: `#1A1A1A` (Near Black)
  - Light Background: `#F0FFF0` (Honeydew - very light green)
- **Typography**:
  - Headings: Arial Bold, 24-32px
  - Body: Arial, 14-16px
- **Components**:
  - Cards with green borders/shadows
  - Buttons with green gradient backgrounds
  - Charts using green color schemes

### Components
1. **File Upload Zone**: Drag & drop area for Excel/CSV files
2. **Data Preview Table**: Scrollable data display
3. **Prediction Charts**: Line charts for cashflow forecast
4. **Statistics Cards**: Key metrics (current balance, predicted low, predicted high)
5. **Historical Trend Chart**: 30-day rolling visualization

## 3. Functionality Specification

### Core Features
1. **Data Upload**:
   - Accept .xlsx, .xls, .csv files
   - Auto-detect date and amount columns
   - Data validation and cleaning

2. **Cashflow Analysis**:
   - Calculate historical cashflow patterns
   - Identify income and expense trends
   - Monthly/daily aggregation

3. **AI Prediction**:
   - Linear regression-based forecasting
   - 30, 60, 90 day predictions
   - Confidence intervals

4. **Dashboard Display**:
   - Current balance overview
   - Predicted cashflow timeline
   - Key insights

### User Interactions
1. Upload file → Preview data → View predictions
2. Adjust prediction timeframe
3. Download forecast report

### Data Handling
- Column detection: Look for common names (date, amount, balance, income, expense)
- Missing value handling: Forward fill then backward fill
- Outlier detection: Remove values > 3 std deviations

## 4. Acceptance Criteria
- [ ] App loads without errors
- [ ] Excel/CSV files can be uploaded
- [ ] Data preview displays correctly
- [ ] Green and white theme applied throughout
- [ ] Predictions generate for 30/60/90 days
- [ ] Charts display with green color scheme
- [ ] Mobile responsive layout