"""
Silhouette Growth Queries Dashboard
Interactive Streamlit dashboard for analyzing wallet demographics and activity
on Silhouette (Hyperliquid Testnet)
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

try:
    from hyperliquid.info import Info
    from hyperliquid.utils import constants
except ImportError:
    st.error("hyperliquid-python-sdk not installed. Run: pip install hyperliquid-python-sdk")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Silhouette Growth Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Authentication
def check_password():
    """Check if user is authenticated"""
    # Initialize authentication state
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    
    # If already authenticated, return True
    if st.session_state.authenticated:
        return True
    
    # Get password from secrets (for Streamlit Cloud) or environment variable
    try:
        # Try to get password from Streamlit secrets (for deployed app)
        correct_password = st.secrets.get("password", None)
    except (AttributeError, FileNotFoundError):
        # Fallback to environment variable or default (for local development)
        import os
        correct_password = os.getenv("DASHBOARD_PASSWORD", None)
    
    # If no password is set, allow access (for development)
    if correct_password is None:
        st.warning("⚠️ No password configured. Set DASHBOARD_PASSWORD environment variable or add to Streamlit secrets.")
        return True
    
    # Show login form
    st.title("🔐 Authentication Required")
    st.markdown("Please enter the password to access the Silhouette Growth Dashboard.")
    
    password_input = st.text_input("Password", type="password", key="password_input")
    
    col1, col2 = st.columns([1, 4])
    with col1:
        login_button = st.button("Login", type="primary")
    
    if login_button:
        if password_input == correct_password:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("❌ Incorrect password. Please try again.")
            return False
    
    # Show logout option if somehow authenticated but showing login
    if st.session_state.authenticated:
        return True
    
    st.stop()
    return False

# Check authentication before showing the app
if not check_password():
    st.stop()

# Rate limiting: Very conservative 3 requests per second to stay safely under 10 req/s limit
RATE_LIMIT = 3
REQUEST_INTERVAL = 1.0 / RATE_LIMIT
# Additional delay between processing each wallet to avoid burst rate limits
WALLET_PROCESSING_DELAY = 0.5  # seconds between wallets
# Delay between different API calls within the same wallet
API_CALL_DELAY = 0.2  # seconds between different API calls for same wallet

# Initialize session state for rate limiting
if 'last_request_time' not in st.session_state:
    st.session_state.last_request_time = 0
if 'request_count' not in st.session_state:
    st.session_state.request_count = 0


def rate_limit():
    """Enforce rate limiting (conservative 3 req/s)"""
    current_time = time.time()
    time_since_last = current_time - st.session_state.last_request_time
    
    if time_since_last < REQUEST_INTERVAL:
        sleep_time = REQUEST_INTERVAL - time_since_last
        time.sleep(sleep_time)
    
    st.session_state.last_request_time = time.time()
    st.session_state.request_count += 1


def fetch_with_retry(func, max_retries=3, initial_delay=2.0):
    """Fetch data with retry logic and exponential backoff for rate limiting
    
    Args:
        func: Function to call (should be a lambda or callable that takes no args)
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds before retry
    
    Returns:
        Result of func() call, or None if all retries failed
    """
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            rate_limit()
            return func()
        except Exception as e:
            # Check if it's a 429 rate limit error
            error_str = str(e)
            if '429' in error_str or 'rate limit' in error_str.lower():
                if attempt < max_retries - 1:
                    # Exponential backoff: wait longer before retrying
                    wait_time = delay * (2 ** attempt)
                    time.sleep(wait_time)
                    continue
                else:
                    # Final attempt failed
                    raise Exception(f"Rate limit exceeded after {max_retries} retries. Please try again in a moment.")
            else:
                # Not a rate limit error, re-raise immediately
                raise
    return None


def initialize_hyperliquid_info(base_url: Optional[str] = None):
    """Initialize Hyperliquid Info instance"""
    if base_url is None:
        base_url = constants.TESTNET_API_URL
    return Info(base_url, skip_ws=True)


def fetch_user_state(info: Info, address: str) -> Optional[Dict]:
    """Fetch user state for a wallet address"""
    try:
        return fetch_with_retry(lambda: info.user_state(address))
    except Exception as e:
        error_msg = str(e)
        if '429' in error_msg or 'rate limit' in error_msg.lower():
            st.warning(f"Rate limit reached while fetching user_state for {address[:10]}...")
        else:
            st.warning(f"Error fetching user_state for {address[:10]}...: {error_msg[:100]}")
        return None


def fetch_user_fills(info: Info, address: str) -> Optional[List[Dict]]:
    """Fetch user fills (trade history) for a wallet address"""
    try:
        fills = fetch_with_retry(lambda: info.user_fills(address))
        return fills if fills else []
    except Exception as e:
        error_msg = str(e)
        if '429' in error_msg or 'rate limit' in error_msg.lower():
            st.warning(f"Rate limit reached while fetching fills for {address[:10]}...")
        else:
            st.warning(f"Error fetching user_fills for {address[:10]}...: {error_msg[:100]}")
        return None


def fetch_user_funding(info: Info, address: str, start_time: int, end_time: Optional[int] = None) -> Optional[List[Dict]]:
    """Fetch user funding events for a wallet address
    
    Args:
        info: Hyperliquid Info instance
        address: Wallet address
        start_time: Start time in milliseconds (Unix timestamp)
        end_time: Optional end time in milliseconds (Unix timestamp)
    """
    try:
        funding = fetch_with_retry(
            lambda: info.user_funding_history(address, start_time, end_time)
        )
        return funding if funding else []
    except Exception as e:
        error_msg = str(e)
        # Provide user-friendly error message for rate limits
        if '429' in error_msg or 'rate limit' in error_msg.lower():
            st.warning(f"Rate limit reached while fetching funding for {address[:10]}... Skipping funding data. Please wait a moment if processing multiple wallets.")
        else:
            st.warning(f"Error fetching user_funding_history for {address[:10]}...: {error_msg[:100]}")
        return None


def fetch_user_historical_orders(info: Info, address: str) -> Optional[List[Dict]]:
    """Fetch historical orders for a wallet address (needed to get BuilderCode/cloid)"""
    try:
        orders = fetch_with_retry(lambda: info.historical_orders(address))
        return orders if orders else []
    except Exception as e:
        error_msg = str(e)
        if '429' in error_msg or 'rate limit' in error_msg.lower():
            st.warning(f"Rate limit reached while fetching orders for {address[:10]}...")
        else:
            st.warning(f"Error fetching historical_orders for {address[:10]}...: {error_msg[:100]}")
        return None


def match_fills_to_orders(fills: List[Dict], orders: List[Dict], builder_code: Optional[str] = None) -> List[Dict]:
    """Match fills to orders by oid and extract BuilderCode (cloid)
    
    Args:
        fills: List of fill dictionaries
        orders: List of historical order dictionaries
        builder_code: Optional BuilderCode to filter by (if provided, only matches are returned)
    
    Returns:
        List of fills with added 'cloid' and 'is_silhouette' fields
    """
    if not fills or not orders:
        return fills
    
    # Create a mapping of oid -> cloid from orders
    oid_to_cloid = {}
    for order in orders:
        order_obj = order.get('order', {}) if isinstance(order, dict) else {}
        oid = order_obj.get('oid')
        cloid = order_obj.get('cloid')
        if oid is not None:
            oid_to_cloid[oid] = cloid
    
    # Add cloid to fills and check if it matches builder_code
    enriched_fills = []
    for fill in fills:
        fill_copy = fill.copy()
        oid = fill.get('oid')
        cloid = oid_to_cloid.get(oid)
        fill_copy['cloid'] = cloid
        fill_copy['is_silhouette'] = (cloid is not None and builder_code is not None and cloid == builder_code)
        
        # If builder_code filter is specified, only include matching fills
        if builder_code is None or fill_copy['is_silhouette']:
            enriched_fills.append(fill_copy)
    
    return enriched_fills


def parse_wallet_addresses(input_text: str) -> List[str]:
    """Parse comma-separated wallet addresses from input"""
    if not input_text:
        return []
    
    addresses = [addr.strip() for addr in input_text.split(',') if addr.strip()]
    # Basic validation: check if addresses look like Ethereum addresses
    valid_addresses = []
    for addr in addresses:
        if addr.startswith('0x') and len(addr) == 42:
            valid_addresses.append(addr)
        else:
            st.warning(f"Invalid address format: {addr}")
    
    return valid_addresses


def process_wallet_data(addresses: List[str], date_start: datetime, date_end: datetime, builder_code: Optional[str] = None) -> Tuple[pd.DataFrame, List[Dict]]:
    """Process wallet data from Hyperliquid API"""
    info = initialize_hyperliquid_info()
    
    all_fills = []
    all_funding = []
    wallet_metadata = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_wallets = len(addresses)
    
    for idx, address in enumerate(addresses):
        status_text.text(f"Fetching data for wallet {idx + 1}/{total_wallets}: {address[:10]}...")
        progress_bar.progress((idx + 1) / total_wallets)
        
        # Add delay between wallets to avoid rate limiting (except for first wallet)
        if idx > 0:
            time.sleep(WALLET_PROCESSING_DELAY)
        
        # Fetch user state
        user_state = fetch_user_state(info, address)
        time.sleep(API_CALL_DELAY)  # Delay between API calls
        
        # Fetch fills
        fills = fetch_user_fills(info, address)
        time.sleep(API_CALL_DELAY)  # Delay between API calls
        
        # Fetch historical orders to get BuilderCode (cloid) information and order types
        orders = fetch_user_historical_orders(info, address) if builder_code or fills else None
        if orders is not None:
            time.sleep(API_CALL_DELAY)  # Delay after orders call
        
        # Match fills to orders and filter by BuilderCode if provided
        if fills and orders:
            fills = match_fills_to_orders(fills, orders, builder_code)
            
            # Extract order type from orders and add to fills
            oid_to_order_type = {}
            for order in orders:
                order_obj = order.get('order', {}) if isinstance(order, dict) else {}
                oid = order_obj.get('oid')
                # Extract order type (limit, market, etc.)
                if 'orderType' in order_obj:
                    oid_to_order_type[oid] = order_obj['orderType']
                elif 'tif' in order_obj:  # Time in force
                    oid_to_order_type[oid] = order_obj.get('tif', 'Unknown')
                else:
                    oid_to_order_type[oid] = 'Unknown'
            
            # Add order type to fills
            for fill in fills:
                oid = fill.get('oid')
                fill['order_type'] = oid_to_order_type.get(oid, 'Unknown')
        elif fills:
            # If no orders but we have fills, set order_type to Unknown
            for fill in fills:
                fill['order_type'] = 'Unknown'
        
        if fills:
            for fill in fills:
                fill['wallet'] = address
                fill['data_type'] = 'fill'
                # Convert timestamp if present
                if 'time' in fill:
                    fill['timestamp'] = pd.to_datetime(fill['time'], unit='ms')
                elif 'timestamp' in fill:
                    fill['timestamp'] = pd.to_datetime(fill['timestamp'], unit='ms')
                
                # Calculate trade size in USDC (sz * px)
                if 'sz' in fill and 'px' in fill:
                    try:
                        sz = float(fill.get('sz', 0))
                        px = float(fill.get('px', 0))
                        fill['trade_size_usdc'] = abs(sz * px)  # Use abs to handle both long/short
                    except (ValueError, TypeError):
                        fill['trade_size_usdc'] = 0
                else:
                    fill['trade_size_usdc'] = 0
                
                # Extract coin/asset if available
                if 'coin' in fill:
                    fill['asset'] = fill['coin']
                elif 'asset' not in fill:
                    fill['asset'] = 'Unknown'
                
                # Extract side (buy/sell) if available
                if 'side' in fill:
                    fill['trade_side'] = fill['side']
                elif 'closedPnl' in fill:
                    # Infer from closedPnl if available
                    fill['trade_side'] = 'Close' if fill.get('closedPnl') else 'Open'
                else:
                    fill['trade_side'] = 'Unknown'
                    
            all_fills.extend(fills)
        
        # Fetch funding (convert datetime to milliseconds timestamp)
        start_time_ms = int(date_start.timestamp() * 1000)
        end_time_ms = int(date_end.timestamp() * 1000)
        funding = fetch_user_funding(info, address, start_time_ms, end_time_ms)
        if funding is not None:
            time.sleep(API_CALL_DELAY)  # Delay after funding call
        if funding:
            for fund in funding:
                fund['wallet'] = address
                fund['data_type'] = 'funding'
                # Convert timestamp if present
                if 'time' in fund:
                    fund['timestamp'] = pd.to_datetime(fund['time'], unit='ms')
                elif 'timestamp' in fund:
                    fund['timestamp'] = pd.to_datetime(fund['timestamp'], unit='ms')
                
                # Extract funding amount (usually negative for fees paid)
                if 'funding' in fund:
                    try:
                        fund['funding_amount'] = float(fund['funding'])
                    except (ValueError, TypeError):
                        fund['funding_amount'] = 0
                else:
                    fund['funding_amount'] = 0
                    
            all_funding.extend(funding)
        
        # Store wallet metadata
        if user_state:
            wallet_metadata.append({
                'wallet': address,
                'user_state': user_state,
                'fill_count': len(fills) if fills else 0,
                'funding_count': len(funding) if funding else 0
            })
    
    progress_bar.empty()
    status_text.empty()
    
    # Combine all data
    all_data = all_fills + all_funding
    
    if not all_data:
        return pd.DataFrame(), wallet_metadata
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    
    # Filter by date range if timestamp column exists
    if 'timestamp' in df.columns:
        df = df[df['timestamp'].notna()]
        df = df[(df['timestamp'] >= date_start) & (df['timestamp'] <= date_end)]
    
    return df, wallet_metadata


def calculate_demographics(df: pd.DataFrame, wallet_metadata: List[Dict]) -> Dict:
    """Calculate wallet demographics metrics"""
    if df.empty:
        return {}
    
    # Unique wallets
    unique_wallets = df['wallet'].nunique() if 'wallet' in df.columns else 0
    
    # Transaction counts per wallet
    tx_counts = df.groupby('wallet').size().reset_index(name='tx_count')
    
    # First transaction date per wallet (wallet "age")
    if 'timestamp' in df.columns:
        first_tx = df.groupby('wallet')['timestamp'].min().reset_index()
        first_tx.columns = ['wallet', 'first_tx_date']
        first_tx['wallet_age_days'] = (datetime.now() - first_tx['first_tx_date']).dt.days
    else:
        first_tx = pd.DataFrame()
    
    # Activity tiers
    if not tx_counts.empty:
        tx_counts['activity_tier'] = pd.cut(
            tx_counts['tx_count'],
            bins=[0, 2, 10, float('inf')],
            labels=['Low', 'Medium', 'High']
        )
    else:
        tx_counts['activity_tier'] = []
    
    return {
        'unique_wallets': unique_wallets,
        'tx_counts': tx_counts,
        'first_tx': first_tx,
        'avg_tx_per_wallet': tx_counts['tx_count'].mean() if not tx_counts.empty else 0
    }


def calculate_activity_metrics(df: pd.DataFrame) -> Dict:
    """Calculate wallet activity metrics"""
    if df.empty:
        return {}
    
    metrics = {}
    
    # Total transactions
    metrics['total_tx'] = len(df)
    
    # Transaction counts per wallet
    if 'wallet' in df.columns:
        metrics['tx_per_wallet'] = df.groupby('wallet').size().reset_index(name='count')
    
    # Volume calculation (if available in fills)
    if 'data_type' in df.columns:
        fills_df = df[df['data_type'] == 'fill']
        if not fills_df.empty:
            # Use trade_size_usdc if available, otherwise calculate from sz * px
            if 'trade_size_usdc' in fills_df.columns:
                metrics['total_volume'] = fills_df['trade_size_usdc'].sum()
                metrics['volume_per_wallet'] = fills_df.groupby('wallet')['trade_size_usdc'].sum().reset_index(name='volume')
            elif 'sz' in fills_df.columns and 'px' in fills_df.columns:
                fills_df['volume'] = pd.to_numeric(fills_df['sz'], errors='coerce') * pd.to_numeric(fills_df['px'], errors='coerce')
                metrics['total_volume'] = fills_df['volume'].sum()
                metrics['volume_per_wallet'] = fills_df.groupby('wallet')['volume'].sum().reset_index()
            else:
                metrics['total_volume'] = 0
                metrics['volume_per_wallet'] = pd.DataFrame()
        else:
            metrics['total_volume'] = 0
            metrics['volume_per_wallet'] = pd.DataFrame()
    
    # Activity over time
    if 'timestamp' in df.columns:
        df['date'] = df['timestamp'].dt.date
        metrics['activity_over_time'] = df.groupby('date').size().reset_index(name='count')
        metrics['activity_over_time'].columns = ['date', 'count']
    else:
        metrics['activity_over_time'] = pd.DataFrame()
    
    # New wallets over time
    if 'timestamp' in df.columns and 'wallet' in df.columns:
        first_tx = df.groupby('wallet')['timestamp'].min().reset_index()
        first_tx['date'] = first_tx['timestamp'].dt.date
        metrics['new_wallets_over_time'] = first_tx.groupby('date').size().reset_index(name='new_wallets')
    else:
        metrics['new_wallets_over_time'] = pd.DataFrame()
    
    return metrics


def calculate_asset_popularity(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate asset/market popularity rankings based on trade counts and volume"""
    if df.empty or 'data_type' not in df.columns:
        return pd.DataFrame()
    
    fills_df = df[df['data_type'] == 'fill']
    if fills_df.empty or 'asset' not in fills_df.columns:
        return pd.DataFrame()
    
    # Count trades per asset
    asset_stats = fills_df.groupby('asset').agg({
        'wallet': 'count',  # Number of trades
        'trade_size_usdc': ['sum', 'mean', 'count']  # Total volume, avg size, count
    }).reset_index()
    
    # Flatten column names
    asset_stats.columns = ['asset', 'trade_count', 'total_volume', 'avg_trade_size', 'volume_count']
    
    # Calculate total volume properly
    if 'trade_size_usdc' in fills_df.columns:
        asset_volume = fills_df.groupby('asset')['trade_size_usdc'].sum().reset_index(name='total_volume_usdc')
        asset_stats = asset_stats.merge(asset_volume, on='asset', how='left')
        asset_stats['total_volume_usdc'] = asset_stats['total_volume_usdc'].fillna(0)
    else:
        asset_stats['total_volume_usdc'] = 0
    
    # Sort by trade count (popularity)
    asset_stats = asset_stats.sort_values('trade_count', ascending=False).reset_index(drop=True)
    asset_stats['rank'] = range(1, len(asset_stats) + 1)
    
    return asset_stats[['rank', 'asset', 'trade_count', 'total_volume_usdc', 'avg_trade_size']]


def calculate_order_type_activity(df: pd.DataFrame) -> Dict:
    """Calculate activity breakdown by order type"""
    if df.empty or 'data_type' not in df.columns:
        return {}
    
    fills_df = df[df['data_type'] == 'fill']
    if fills_df.empty or 'order_type' not in fills_df.columns:
        return {}
    
    order_type_stats = fills_df.groupby('order_type').agg({
        'wallet': 'count',
        'trade_size_usdc': 'sum'
    }).reset_index()
    order_type_stats.columns = ['order_type', 'count', 'total_volume']
    
    # Calculate percentages
    total_count = order_type_stats['count'].sum()
    if total_count > 0:
        order_type_stats['percentage'] = (order_type_stats['count'] / total_count) * 100
    else:
        order_type_stats['percentage'] = 0
    
    order_type_stats = order_type_stats.sort_values('count', ascending=False).reset_index(drop=True)
    
    return {
        'order_types': order_type_stats,
        'total_orders': total_count
    }


def calculate_fees_and_funding(df: pd.DataFrame) -> Dict:
    """Calculate total fees and funding costs"""
    if df.empty:
        return {}
    
    metrics = {}
    
    # Calculate funding costs from funding events
    if 'data_type' in df.columns:
        funding_df = df[df['data_type'] == 'funding']
        if not funding_df.empty and 'funding_amount' in funding_df.columns:
            metrics['total_funding'] = funding_df['funding_amount'].sum()
            metrics['funding_count'] = len(funding_df)
            metrics['avg_funding'] = funding_df['funding_amount'].mean() if len(funding_df) > 0 else 0
            metrics['funding_per_wallet'] = funding_df.groupby('wallet')['funding_amount'].sum().reset_index()
        else:
            metrics['total_funding'] = 0
            metrics['funding_count'] = 0
            metrics['avg_funding'] = 0
            metrics['funding_per_wallet'] = pd.DataFrame()
    
    # Note: Trade fees are typically included in the fill price, so we'd need additional API calls
    # to get exact fee breakdown. For now, we focus on funding costs.
    
    return metrics


def calculate_position_duration(wallet_metadata: List[Dict], df: pd.DataFrame) -> Dict:
    """Calculate average position holding time"""
    # This is a simplified calculation - in reality, we'd need to track position opens/closes
    # For now, we'll estimate based on time between fills for the same asset
    if df.empty or 'data_type' not in df.columns or 'timestamp' not in df.columns:
        return {}
    
    fills_df = df[df['data_type'] == 'fill'].copy()
    if fills_df.empty or 'asset' not in fills_df.columns:
        return {}
    
    # Group by wallet and asset, calculate time between first and last trade
    position_durations = []
    for (wallet, asset), group in fills_df.groupby(['wallet', 'asset']):
        if len(group) > 1:
            group = group.sort_values('timestamp')
            first_trade = group.iloc[0]['timestamp']
            last_trade = group.iloc[-1]['timestamp']
            duration = (last_trade - first_trade).total_seconds() / 3600  # Convert to hours
            position_durations.append({
                'wallet': wallet,
                'asset': asset,
                'duration_hours': duration,
                'trade_count': len(group)
            })
    
    if not position_durations:
        return {
            'avg_duration_hours': 0,
            'median_duration_hours': 0,
            'position_data': pd.DataFrame()
        }
    
    duration_df = pd.DataFrame(position_durations)
    
    return {
        'avg_duration_hours': duration_df['duration_hours'].mean(),
        'median_duration_hours': duration_df['duration_hours'].median(),
        'position_data': duration_df
    }


def analyze_drop_offs(df: pd.DataFrame) -> Dict:
    """Analyze activity drop-offs"""
    if df.empty or 'wallet' not in df.columns:
        return {}
    
    # Transaction counts per wallet
    tx_counts = df.groupby('wallet').size().reset_index(name='tx_count')
    
    # Categorize users
    drop_off_analysis = {
        'single_tx_users': len(tx_counts[tx_counts['tx_count'] == 1]),
        'low_activity_users': len(tx_counts[tx_counts['tx_count'].between(2, 5)]),
        'medium_activity_users': len(tx_counts[tx_counts['tx_count'].between(6, 20)]),
        'high_activity_users': len(tx_counts[tx_counts['tx_count'] > 20]),
        'total_users': len(tx_counts)
    }
    
    if drop_off_analysis['total_users'] > 0:
        drop_off_analysis['single_tx_percentage'] = (drop_off_analysis['single_tx_users'] / drop_off_analysis['total_users']) * 100
        drop_off_analysis['low_activity_percentage'] = (drop_off_analysis['low_activity_users'] / drop_off_analysis['total_users']) * 100
    else:
        drop_off_analysis['single_tx_percentage'] = 0
        drop_off_analysis['low_activity_percentage'] = 0
    
    return drop_off_analysis


# Main Streamlit App
st.title("📊 Silhouette Growth Queries Dashboard")
st.markdown("Analyze wallet demographics and activity on Silhouette (Hyperliquid Testnet)")

# Sidebar
with st.sidebar:
    # Logout button at the top
    if st.button("🚪 Logout", use_container_width=True, type="secondary"):
        st.session_state.authenticated = False
        st.rerun()
    
    st.divider()
    
    st.header("⚙️ Configuration")
    
    # Wallet addresses input
    st.subheader("Wallet Addresses")
    wallet_input = st.text_area(
        "Enter wallet addresses (comma-separated):",
        height=150,
        help="Enter Ethereum addresses separated by commas"
    )
    
    # Date range filter
    st.subheader("Date Range")
    col1, col2 = st.columns(2)
    with col1:
        date_start = st.date_input("Start Date", value=datetime.now() - timedelta(days=30))
    with col2:
        date_end = st.date_input("End Date", value=datetime.now())
    
    # Metric selection
    st.subheader("Analysis Type")
    analysis_type = st.radio(
        "Select analysis type:",
        ["Demographics", "Activity", "Both"],
        index=2
    )
    
    # BuilderCode filter (for identifying Silhouette transactions)
    st.subheader("Silhouette BuilderCode Filter")
    builder_code_default = "0x5d2c2bd98f10616771d7b5124ad2090ba72aa43c"
    builder_code = st.text_input(
        "BuilderCode (optional):",
        value="",
        help="Enter Silhouette BuilderCode to filter transactions executed via Silhouette frontend. Leave empty to show all transactions.",
        placeholder=builder_code_default
    )
    builder_code = builder_code.strip() if builder_code else None
    
    # API Configuration (optional)
    st.subheader("API Configuration")
    custom_api_url = st.text_input(
        "Custom API URL (optional):",
        value="",
        help="Leave empty to use default Testnet URL"
    )
    
    # Fetch button
    fetch_button = st.button("🚀 Fetch & Analyze", type="primary", use_container_width=True)

# Main content area
if fetch_button:
    # Parse wallet addresses
    addresses = parse_wallet_addresses(wallet_input)
    
    if not addresses:
        st.error("Please enter at least one valid wallet address.")
        st.stop()
    
    st.info(f"Analyzing {len(addresses)} wallet(s)...")
    
    # Process data
    with st.spinner("Fetching data from Hyperliquid Testnet API..."):
        try:
            df, wallet_metadata = process_wallet_data(
                addresses,
                datetime.combine(date_start, datetime.min.time()),
                datetime.combine(date_end, datetime.max.time()),
                builder_code
            )
        except Exception as e:
            st.error(f"Error processing data: {str(e)}")
            st.stop()
    
    if df.empty:
        st.warning("No data found for the provided wallets and date range.")
        st.stop()
    
    # Display metrics
    st.header("📈 Key Metrics")
    
    # Show Silhouette transaction count if BuilderCode filter is active
    if builder_code and not df.empty and 'is_silhouette' in df.columns:
        silhouette_count = df['is_silhouette'].sum() if 'is_silhouette' in df.columns else 0
        total_fills = len(df[df['data_type'] == 'fill']) if 'data_type' in df.columns else 0
        if total_fills > 0:
            silhouette_pct = (silhouette_count / total_fills) * 100
            st.success(f"✅ **Silhouette Transactions**: {silhouette_count} out of {total_fills} fills ({silhouette_pct:.1f}%)")
        else:
            st.info(f"ℹ️ BuilderCode filter active: {builder_code}")
    
    # Calculate metrics
    demographics = calculate_demographics(df, wallet_metadata)
    activity_metrics = calculate_activity_metrics(df)
    drop_off_analysis = analyze_drop_offs(df)
    
    # Calculate new metrics
    asset_popularity = calculate_asset_popularity(df)
    order_type_activity = calculate_order_type_activity(df)
    fees_funding = calculate_fees_and_funding(df)
    position_duration = calculate_position_duration(wallet_metadata, df)
    
    # Metrics columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Unique Wallets", demographics.get('unique_wallets', 0))
    
    with col2:
        st.metric("Avg TX per Wallet", f"{demographics.get('avg_tx_per_wallet', 0):.2f}")
    
    with col3:
        st.metric("Total Transactions", activity_metrics.get('total_tx', 0))
    
    with col4:
        total_volume = activity_metrics.get('total_volume', 0)
        st.metric("Total Volume", f"${total_volume:,.2f}" if total_volume > 0 else "N/A")
    
    # Additional metrics row
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        avg_trade_size = 0
        if not df.empty and 'data_type' in df.columns:
            fills_df = df[df['data_type'] == 'fill']
            if 'trade_size_usdc' in fills_df.columns:
                avg_trade_size = fills_df['trade_size_usdc'].mean()
        st.metric("Avg Trade Size (USDC)", f"${avg_trade_size:,.2f}" if avg_trade_size > 0 else "N/A")
    
    with col6:
        total_funding = fees_funding.get('total_funding', 0)
        st.metric("Total Funding Paid", f"${total_funding:,.2f}" if total_funding != 0 else "N/A")
    
    with col7:
        avg_duration = position_duration.get('avg_duration_hours', 0)
        if avg_duration > 0:
            if avg_duration < 24:
                duration_str = f"{avg_duration:.1f} hrs"
            elif avg_duration < 168:  # 7 days
                duration_str = f"{avg_duration/24:.1f} days"
            else:
                duration_str = f"{avg_duration/168:.1f} weeks"
            st.metric("Avg Position Duration", duration_str)
        else:
            st.metric("Avg Position Duration", "N/A")
    
    with col8:
        if not asset_popularity.empty:
            top_asset = asset_popularity.iloc[0]['asset'] if len(asset_popularity) > 0 else "N/A"
            st.metric("Most Popular Asset", top_asset)
        else:
            st.metric("Most Popular Asset", "N/A")
    
    # Analysis sections
    if analysis_type in ["Demographics", "Both"]:
        st.header("👥 Wallet Demographics")
        
        # Transaction counts per wallet (bar chart)
        if not demographics.get('tx_counts', pd.DataFrame()).empty:
            st.subheader("Transaction Counts per Wallet")
            tx_counts = demographics['tx_counts'].head(20).copy()  # Top 20
            tx_counts = tx_counts.sort_values('tx_count', ascending=False).reset_index(drop=True)
            
            # Create shortened wallet addresses for display (first 6 + last 4 chars)
            tx_counts['wallet_short'] = tx_counts['wallet'].apply(
                lambda addr: f"{addr[:8]}...{addr[-6:]}" if len(addr) > 14 else addr
            )
            
            # Create a ranked label with position
            tx_counts['wallet_label'] = [
                f"#{i+1}: {short}" 
                for i, short in enumerate(tx_counts['wallet_short'])
            ]
            
            # Use horizontal bar chart for better readability
            fig = px.bar(
                tx_counts,
                x='tx_count',
                y='wallet_label',
                orientation='h',
                title="Transaction Counts per Wallet (Top 20)",
                labels={'tx_count': 'Transaction Count', 'wallet_label': 'Wallet Address'},
                hover_data={'wallet': True, 'wallet_short': False, 'wallet_label': False},
                text='tx_count'
            )
            fig.update_traces(texttemplate='%{text}', textposition='outside')
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            fig.update_xaxes(title="Transaction Count")
            fig.update_yaxes(title="")
            st.plotly_chart(fig, use_container_width=True)
        
        # Activity tier distribution (pie chart)
        if 'activity_tier' in demographics.get('tx_counts', pd.DataFrame()).columns:
            st.subheader("Activity Tier Distribution")
            tier_counts = demographics['tx_counts']['activity_tier'].value_counts()
            if not tier_counts.empty:
                fig = px.pie(
                    values=tier_counts.values,
                    names=tier_counts.index,
                    title="Distribution by Activity Level"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Wallet age distribution
        if not demographics.get('first_tx', pd.DataFrame()).empty:
            st.subheader("Wallet Age Distribution")
            first_tx = demographics['first_tx']
            if 'wallet_age_days' in first_tx.columns:
                fig = px.histogram(
                    first_tx,
                    x='wallet_age_days',
                    nbins=20,
                    title="Distribution of Wallet Age (Days Since First Transaction)",
                    labels={'wallet_age_days': 'Days Since First TX', 'count': 'Number of Wallets'}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Top 10 active wallets
        st.subheader("Top 10 Active Wallets")
        if not demographics.get('tx_counts', pd.DataFrame()).empty:
            top_wallets = demographics['tx_counts'].nlargest(10, 'tx_count')
            st.dataframe(
                top_wallets[['wallet', 'tx_count']].reset_index(drop=True),
                use_container_width=True
            )
    
    if analysis_type in ["Activity", "Both"]:
        st.header("📊 Wallet Activity")
        
        # Activity over time (line chart)
        if not activity_metrics.get('activity_over_time', pd.DataFrame()).empty:
            st.subheader("Activity Over Time")
            activity_df = activity_metrics['activity_over_time']
            fig = px.line(
                activity_df,
                x='date',
                y='count',
                title="Transaction Count Over Time",
                labels={'date': 'Date', 'count': 'Transaction Count'}
            )
            fig.update_traces(mode='lines+markers')
            st.plotly_chart(fig, use_container_width=True)
        
        # New wallets over time
        if not activity_metrics.get('new_wallets_over_time', pd.DataFrame()).empty:
            st.subheader("New Wallets Over Time")
            new_wallets_df = activity_metrics['new_wallets_over_time']
            fig = px.line(
                new_wallets_df,
                x='date',
                y='new_wallets',
                title="New Wallets Per Day",
                labels={'date': 'Date', 'new_wallets': 'New Wallets'}
            )
            fig.update_traces(mode='lines+markers')
            st.plotly_chart(fig, use_container_width=True)
        
        # Volume per wallet (if available)
        if not activity_metrics.get('volume_per_wallet', pd.DataFrame()).empty:
            st.subheader("Trading Volume per Wallet")
            volume_df = activity_metrics['volume_per_wallet'].head(20).copy()
            volume_df = volume_df.sort_values('volume', ascending=False).reset_index(drop=True)
            
            # Create shortened wallet addresses for display (first 8 + last 6 chars)
            volume_df['wallet_short'] = volume_df['wallet'].apply(
                lambda addr: f"{addr[:8]}...{addr[-6:]}" if len(addr) > 14 else addr
            )
            
            # Create a ranked label with position
            volume_df['wallet_label'] = [
                f"#{i+1}: {short}" 
                for i, short in enumerate(volume_df['wallet_short'])
            ]
            
            # Use horizontal bar chart for better readability
            fig = px.bar(
                volume_df,
                x='volume',
                y='wallet_label',
                orientation='h',
                title="Trading Volume per Wallet (Top 20)",
                labels={'volume': 'Volume (USD)', 'wallet_label': 'Wallet Address'},
                hover_data={'wallet': True, 'wallet_short': False, 'wallet_label': False},
                text='volume'
            )
            # Format volume text to show currency
            fig.update_traces(
                texttemplate='$%{text:,.0f}', 
                textposition='outside'
            )
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            fig.update_xaxes(title="Volume (USD)")
            fig.update_yaxes(title="")
            st.plotly_chart(fig, use_container_width=True)
        
        # Asset/Market Popularity Rankings
        if not asset_popularity.empty:
            st.subheader("📈 Asset/Market Popularity Rankings")
            top_assets = asset_popularity.head(20)
            
            # Create visualization
            fig = px.bar(
                top_assets,
                x='trade_count',
                y='asset',
                orientation='h',
                title="Top 20 Assets by Trade Count",
                labels={'trade_count': 'Number of Trades', 'asset': 'Asset'},
                text='trade_count',
                hover_data={'total_volume_usdc': True, 'rank': True}
            )
            fig.update_traces(texttemplate='%{text}', textposition='outside')
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            fig.update_xaxes(title="Trade Count")
            fig.update_yaxes(title="")
            st.plotly_chart(fig, use_container_width=True)
            
            # Also show volume-based ranking
            st.subheader("Top Assets by Trading Volume (USDC)")
            top_volume = asset_popularity.nlargest(20, 'total_volume_usdc')
            fig = px.bar(
                top_volume,
                x='total_volume_usdc',
                y='asset',
                orientation='h',
                title="Top 20 Assets by Trading Volume",
                labels={'total_volume_usdc': 'Volume (USDC)', 'asset': 'Asset'},
                text='total_volume_usdc'
            )
            fig.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            fig.update_xaxes(title="Volume (USDC)")
            fig.update_yaxes(title="")
            st.plotly_chart(fig, use_container_width=True)
            
            # Display table
            st.subheader("Asset Rankings Table")
            display_df = asset_popularity.head(20)[['rank', 'asset', 'trade_count', 'total_volume_usdc', 'avg_trade_size']].copy()
            display_df['total_volume_usdc'] = display_df['total_volume_usdc'].apply(lambda x: f"${x:,.2f}" if x > 0 else "$0.00")
            display_df['avg_trade_size'] = display_df['avg_trade_size'].apply(lambda x: f"${x:,.2f}" if x > 0 else "$0.00")
            st.dataframe(display_df, use_container_width=True)
        
        # Order Type Activity
        if order_type_activity and order_type_activity.get('order_types') is not None:
            st.subheader("📋 Order Type Activity")
            order_types_df = order_type_activity['order_types']
            
            if not order_types_df.empty:
                # Pie chart
                fig = px.pie(
                    order_types_df,
                    values='count',
                    names='order_type',
                    title="Distribution by Order Type",
                    hover_data=['percentage']
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Bar chart
                fig = px.bar(
                    order_types_df,
                    x='order_type',
                    y='count',
                    title="Order Type Usage",
                    labels={'order_type': 'Order Type', 'count': 'Number of Orders'},
                    text='count'
                )
                fig.update_traces(texttemplate='%{text}', textposition='outside')
                st.plotly_chart(fig, use_container_width=True)
                
                # Display table
                st.subheader("Order Type Statistics")
                display_df = order_types_df[['order_type', 'count', 'percentage', 'total_volume']].copy()
                display_df['percentage'] = display_df['percentage'].apply(lambda x: f"{x:.1f}%")
                display_df['total_volume'] = display_df['total_volume'].apply(lambda x: f"${x:,.2f}" if x > 0 else "$0.00")
                st.dataframe(display_df, use_container_width=True)
        
        # Trade Size Distribution
        if not df.empty and 'data_type' in df.columns:
            fills_df = df[df['data_type'] == 'fill']
            if 'trade_size_usdc' in fills_df.columns and not fills_df['trade_size_usdc'].empty:
                st.subheader("💰 Trade Size Distribution (USDC)")
                trade_sizes = fills_df['trade_size_usdc'][fills_df['trade_size_usdc'] > 0]
                
                if not trade_sizes.empty:
                    # Histogram
                    fig = px.histogram(
                        trade_sizes,
                        nbins=30,
                        title="Distribution of Trade Sizes",
                        labels={'value': 'Trade Size (USDC)', 'count': 'Number of Trades'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Statistics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Min Trade Size", f"${trade_sizes.min():,.2f}")
                    with col2:
                        st.metric("Max Trade Size", f"${trade_sizes.max():,.2f}")
                    with col3:
                        st.metric("Median Trade Size", f"${trade_sizes.median():,.2f}")
                    with col4:
                        st.metric("Mean Trade Size", f"${trade_sizes.mean():,.2f}")
        
        # Position Duration Analysis
        if position_duration and position_duration.get('position_data') is not None:
            pos_data = position_duration['position_data']
            if not pos_data.empty:
                st.subheader("⏱️ Position Duration Analysis")
                
                # Show average and median
                col1, col2 = st.columns(2)
                with col1:
                    avg_hours = position_duration.get('avg_duration_hours', 0)
                    if avg_hours > 0:
                        if avg_hours < 24:
                            st.metric("Average Position Duration", f"{avg_hours:.1f} hours")
                        elif avg_hours < 168:
                            st.metric("Average Position Duration", f"{avg_hours/24:.1f} days")
                        else:
                            st.metric("Average Position Duration", f"{avg_hours/168:.1f} weeks")
                    else:
                        st.metric("Average Position Duration", "N/A")
                
                with col2:
                    median_hours = position_duration.get('median_duration_hours', 0)
                    if median_hours > 0:
                        if median_hours < 24:
                            st.metric("Median Position Duration", f"{median_hours:.1f} hours")
                        elif median_hours < 168:
                            st.metric("Median Position Duration", f"{median_hours/24:.1f} days")
                        else:
                            st.metric("Median Position Duration", f"{median_hours/168:.1f} weeks")
                    else:
                        st.metric("Median Position Duration", "N/A")
                
                # Histogram of position durations
                fig = px.histogram(
                    pos_data,
                    x='duration_hours',
                    nbins=30,
                    title="Distribution of Position Durations",
                    labels={'duration_hours': 'Duration (Hours)', 'count': 'Number of Positions'}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Fees and Funding Analysis
        if fees_funding and fees_funding.get('total_funding') is not None:
            st.subheader("💸 Fees & Funding Analysis")
            
            total_funding = fees_funding.get('total_funding', 0)
            funding_count = fees_funding.get('funding_count', 0)
            avg_funding = fees_funding.get('avg_funding', 0)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Funding Paid", f"${total_funding:,.2f}" if total_funding != 0 else "$0.00")
            with col2:
                st.metric("Funding Events", funding_count)
            with col3:
                st.metric("Avg Funding per Event", f"${avg_funding:,.2f}" if avg_funding != 0 else "$0.00")
            
            # Funding per wallet
            funding_per_wallet = fees_funding.get('funding_per_wallet', pd.DataFrame())
            if not funding_per_wallet.empty:
                st.subheader("Funding Paid per Wallet")
                funding_per_wallet = funding_per_wallet.sort_values('funding_amount', ascending=False).head(20)
                
                funding_per_wallet['wallet_short'] = funding_per_wallet['wallet'].apply(
                    lambda addr: f"{addr[:8]}...{addr[-6:]}" if len(addr) > 14 else addr
                )
                funding_per_wallet['wallet_label'] = [
                    f"#{i+1}: {short}" 
                    for i, short in enumerate(funding_per_wallet['wallet_short'])
                ]
                
                fig = px.bar(
                    funding_per_wallet,
                    x='funding_amount',
                    y='wallet_label',
                    orientation='h',
                    title="Funding Paid per Wallet (Top 20)",
                    labels={'funding_amount': 'Funding Amount (USDC)', 'wallet_label': 'Wallet Address'},
                    hover_data={'wallet': True},
                    text='funding_amount'
                )
                fig.update_traces(texttemplate='$%{text:,.2f}', textposition='outside')
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                fig.update_xaxes(title="Funding Amount (USDC)")
                fig.update_yaxes(title="")
                st.plotly_chart(fig, use_container_width=True)
    
    # Drop-off analysis
    if drop_off_analysis:
        st.header("🔍 Activity Drop-off Analysis")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Single TX Users",
                drop_off_analysis.get('single_tx_users', 0),
                f"{drop_off_analysis.get('single_tx_percentage', 0):.1f}%"
            )
        
        with col2:
            st.metric(
                "Low Activity (2-5 TX)",
                drop_off_analysis.get('low_activity_users', 0),
                f"{drop_off_analysis.get('low_activity_percentage', 0):.1f}%"
            )
        
        with col3:
            st.metric(
                "Medium Activity (6-20 TX)",
                drop_off_analysis.get('medium_activity_users', 0)
            )
        
        with col4:
            st.metric(
                "High Activity (>20 TX)",
                drop_off_analysis.get('high_activity_users', 0)
            )
        
        # Funnel chart for user journey
        st.subheader("User Journey Funnel")
        funnel_data = {
            'Stage': ['Total Users', 'Low Activity', 'Medium Activity', 'High Activity'],
            'Count': [
                drop_off_analysis.get('total_users', 0),
                drop_off_analysis.get('low_activity_users', 0),
                drop_off_analysis.get('medium_activity_users', 0),
                drop_off_analysis.get('high_activity_users', 0)
            ]
        }
        funnel_df = pd.DataFrame(funnel_data)
        fig = px.funnel(
            funnel_df,
            x='Count',
            y='Stage',
            title="User Activity Funnel"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Raw data table
    st.header("📋 Raw Data")
    st.dataframe(df, use_container_width=True)
    
    # Download button
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download Data as CSV",
        data=csv,
        file_name=f"silhouette_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

else:
    # Welcome message
    st.info("👈 Enter wallet addresses in the sidebar and click 'Fetch & Analyze' to begin.")
    
    # Instructions
    with st.expander("📖 How to Use"):
        st.markdown("""
        ### Getting Started
        
        1. **Enter Wallet Addresses**: In the sidebar, paste comma-separated Ethereum wallet addresses
        2. **Set Date Range**: Select the time period you want to analyze
        3. **Choose Analysis Type**: Select Demographics, Activity, or Both
        4. **Fetch Data**: Click the "Fetch & Analyze" button
        
        ### Understanding the Metrics
        
        - **Unique Wallets**: Number of distinct wallet addresses in your input
        - **Avg TX per Wallet**: Average number of transactions per wallet
        - **Total Transactions**: Sum of all transactions across all wallets
        - **Total Volume**: Total trading volume (if available)
        
        ### Visualizations
        
        - **Transaction Counts**: Bar chart showing activity per wallet
        - **Activity Tiers**: Pie chart showing distribution by activity level
        - **Activity Over Time**: Line chart showing transaction trends
        - **Drop-off Analysis**: Funnel chart showing user retention patterns
        
        ### Notes
        
        - Data is fetched from Hyperliquid Testnet API
        - Rate limiting is enforced (10 requests/second)
        - All analysis is read-only
        - Extensible for future Silhouette-specific features
        """)

