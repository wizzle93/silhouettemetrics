# Silhouette Growth Queries Dashboard

Interactive Streamlit dashboard for analyzing wallet demographics and activity on Silhouette, a privacy-focused decentralized trading platform built on the Hyperliquid blockchain.

## Features

- **Wallet Demographics Analysis**
  - Unique wallet counts
  - Transaction count distribution
  - Activity tier classification (Low/Medium/High)
  - Wallet age distribution (days since first transaction)
  - Top 10 active wallets

- **Wallet Activity Analysis**
  - Transaction counts over time
  - New wallets per day
  - Trading volume per wallet
  - Activity trends and patterns

- **Drop-off Analysis**
  - Single transaction users
  - Activity distribution funnel
  - Retention metrics
  - User journey visualization

- **Visualizations**
  - Bar charts for transaction counts
  - Pie charts for activity tiers
  - Line charts for time series data
  - Funnel charts for user journey
  - Interactive Plotly charts

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Run the dashboard:
```bash
streamlit run app.py
```

The dashboard will open in your default web browser.

## Usage

1. **Enter Wallet Addresses**: In the sidebar, paste comma-separated Ethereum wallet addresses (0x... format)

2. **Set Date Range**: Select the start and end dates for analysis

3. **Choose Analysis Type**: 
   - Demographics: Wallet distribution and activity tiers
   - Activity: Transaction trends and volume
   - Both: Complete analysis

4. **Silhouette BuilderCode Filter (optional)**: 
   - Enter `0x5d2c2bd98f10616771d7b5124ad2090ba72aa43c` to filter for transactions executed via Silhouette frontend
   - Leave empty to show all transactions
   - When filtered, dashboard shows only Silhouette transactions and percentage breakdown

5. **Fetch & Analyze**: Click the button to fetch data from Hyperliquid Testnet API

## Data Sources

The dashboard fetches data from Hyperliquid Testnet API using the `hyperliquid-python-sdk`:

- `Info.user_state(address)`: Wallet state, positions, margin
- `Info.user_fills(address)`: Trade history and volume
- `Info.user_funding(address)`: Funding events

## Rate Limiting

The dashboard enforces rate limiting (10 requests/second) to comply with API limits. Progress is shown during data fetching.

## Notes

- Currently configured for Hyperliquid Testnet
- All analysis is read-only
- **Silhouette BuilderCode**: `0x5d2c2bd98f10616771d7b5124ad2090ba72aa43c`
  - Use this to filter and identify transactions executed via Silhouette's frontend
  - Transactions are matched by matching the `cloid` (client order ID) from historical orders to fills
- Extensible for future Silhouette-specific features (shielded transactions)
- Data can be downloaded as CSV for further analysis

## Requirements

- Python 3.8+
- streamlit >= 1.28.0
- pandas >= 2.0.0
- plotly >= 5.17.0
- hyperliquid-python-sdk >= 1.0.0

## Authentication

The dashboard is password-protected. To set up authentication:

### For Streamlit Cloud Deployment:

1. Go to your app settings in Streamlit Cloud
2. Click on "Secrets" in the left sidebar
3. Add the following secret:
   ```toml
   password = "your-secure-password-here"
   ```
4. Save and redeploy your app

### For Local Development:

Set the `DASHBOARD_PASSWORD` environment variable:
```bash
export DASHBOARD_PASSWORD="your-secure-password-here"
streamlit run app.py
```

Or create a `.streamlit/secrets.toml` file (this file is already in `.gitignore`):
```toml
password = "your-secure-password-here"
```

**Note**: If no password is configured, the app will show a warning but allow access (useful for development).

## Troubleshooting

- **No data found**: Verify wallet addresses are valid and have activity in the selected date range
- **API errors**: Check network connection and API availability
- **Import errors**: Ensure all dependencies are installed: `pip install -r requirements.txt`
- **Authentication issues**: Make sure the password is set correctly in Streamlit Cloud secrets or environment variables

