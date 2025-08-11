import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- Caching Data Generation & Preprocessing ---
@st.cache_data
def generate_synthetic_data(num_records=10000):
    """
    Generates a synthetic financial transaction dataset.
    This function is cached to prevent new data from being created on every rerun.
    """
    start_date = datetime.now() - timedelta(days=90)
    end_date = datetime.now()
    dates = pd.to_datetime(np.random.uniform(start_date.timestamp(), end_date.timestamp(), num_records), unit='s')
    dates = np.sort(dates)

    data = {
        'timestamp': dates,
        'account_id': np.random.choice([f'ACCT{i}' for i in range(50)], num_records),
        'transaction_amount': np.random.lognormal(mean=4, sigma=1.5, size=num_records).round(2),
        'transaction_type': np.random.choice(['debit', 'credit'], num_records, p=[0.7, 0.3])
    }
    df = pd.DataFrame(data)

    num_anomalies = int(0.01 * num_records)
    anomaly_indices = np.random.choice(df.index, num_anomalies, replace=False)
    df.loc[anomaly_indices, 'transaction_amount'] *= np.random.uniform(5, 15, num_anomalies)

    return df

@st.cache_data
def preprocess_data(df_input):
    """
    Performs all necessary data preprocessing and feature engineering.
    This function is cached to avoid recomputing on every rerun.
    """
    df = df_input.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    if df.duplicated().sum() > 0:
        df.drop_duplicates(inplace=True)

    df['transaction_hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.day_name()
    df.sort_values('timestamp', inplace=True)

    rolling_window = '7D'
    
    df.set_index('timestamp', inplace=True)
    df['rolling_avg_amount'] = df.groupby('account_id')['transaction_amount'].rolling(window=rolling_window).mean().reset_index(level=0, drop=True)
    df['transaction_count'] = 1
    df['rolling_frequency'] = df.groupby('account_id')['transaction_count'].rolling(window=rolling_window).sum().reset_index(level=0, drop=True)
    df.reset_index(inplace=True)

    df.bfill(inplace=True)
    df.ffill(inplace=True)

    return df

@st.cache_data
def apply_anomaly_detection(df_input, z_score_threshold):
    """
    Applies the Z-score based anomaly detection.
    """
    df = df_input.copy()
    df.set_index('timestamp', inplace=True)
    
    df['rolling_std_amount'] = df.groupby('account_id')['transaction_amount'].rolling(window='7D').std().reset_index(level=0, drop=True)
    df['z_score_amount'] = (df['transaction_amount'] - df['rolling_avg_amount']) / df['rolling_std_amount']
    df['z_score_amount'].fillna(0, inplace=True)
    
    df.reset_index(inplace=True)
    df['is_anomaly'] = df['z_score_amount'].abs() > z_score_threshold
    return df

# --- Streamlit App Layout ---
st.set_page_config(layout="wide", page_title="Financial Anomaly Detection Dashboard")

st.title("Proactive Financial Anomaly Detection System 📈")
st.markdown("""
Welcome to the Anomaly Detection Dashboard. This application demonstrates a proactive approach
to identifying suspicious financial transactions using statistical methods and dynamic baselines.
""")

# --- Sidebar for user input and controls ---
st.sidebar.header("Configuration")
num_records = st.sidebar.slider("Number of records", 1000, 50000, 10000)
z_score_threshold = st.sidebar.slider("Z-Score Threshold", 1.0, 5.0, 3.0, 0.1)

# Generate and preprocess data using cached functions
with st.spinner('Generating and preprocessing data...'):
    data = generate_synthetic_data(num_records)
    df = preprocess_data(data)

st.success('Data generated and preprocessed successfully!')

# --- Section: Data Exploration ---
st.header("1. Exploratory Data Analysis (EDA)")
st.write("A quick look at the data's characteristics and distributions.")

col1, col2 = st.columns(2)
with col1:
    fig_amount = px.histogram(
        df,
        x='transaction_amount',
        log_y=True,
        title='Distribution of Transaction Amounts (Log Scale)',
        nbins=100
    )
    st.plotly_chart(fig_amount, use_container_width=True)

with col2:
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily_counts = df['day_of_week'].value_counts().reindex(day_order)
    fig_daily = px.bar(
        x=daily_counts.index,
        y=daily_counts.values,
        title='Transaction Volume by Day of Week',
        labels={'x': 'Day of Week', 'y': 'Number of Transactions'}
    )
    st.plotly_chart(fig_daily, use_container_width=True)

# --- Section: Anomaly Detection Results ---
st.header("2. Anomaly Detection Results")
st.markdown("We'll use a Z-score based approach to flag transactions that deviate significantly from the rolling average for each account.")

with st.spinner('Applying anomaly detection...'):
    df_anomalies = apply_anomaly_detection(df, z_score_threshold)
st.success('Anomaly detection complete!')

total_anomalies = df_anomalies['is_anomaly'].sum()
st.info(f"**Total Anomalies Detected:** {total_anomalies} (approximately {(total_anomalies/len(df_anomalies)*100):.2f}% of all transactions)")

# --- Section: Anomaly Visualization ---
st.header("3. Anomaly Visualization")
st.write("Select an account to visualize its transactions and the detected anomalies.")

unique_accounts = df_anomalies['account_id'].unique()
sample_account = st.selectbox("Select an Account ID", unique_accounts)

if sample_account:
    account_df = df_anomalies[df_anomalies['account_id'] == sample_account].sort_values('timestamp')

    fig_anomalies = go.Figure()

    fig_anomalies.add_trace(go.Scatter(
        x=account_df['timestamp'],
        y=account_df['transaction_amount'],
        mode='lines+markers',
        name='Transaction Amount',
        line=dict(color='blue'),
        marker=dict(size=4)
    ))

    anomaly_df = account_df[account_df['is_anomaly'] == True]
    fig_anomalies.add_trace(go.Scatter(
        x=anomaly_df['timestamp'],
        y=anomaly_df['transaction_amount'],
        mode='markers',
        name='Detected Anomaly',
        marker=dict(color='red', size=8, symbol='x')
    ))

    fig_anomalies.update_layout(
        title=f'Transaction Amounts for Account {sample_account} with Anomalies Highlighted',
        xaxis_title='Timestamp',
        yaxis_title='Transaction Amount',
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_anomalies, use_container_width=True)

st.markdown("---")

# --- Section: Anomaly Summary ---
st.header("4. Anomaly Summary")
anomaly_summary = df_anomalies[df_anomalies['is_anomaly']].groupby('account_id').size().sort_values(ascending=False).to_frame('Anomaly Count')
if not anomaly_summary.empty:
    st.dataframe(anomaly_summary, use_container_width=True)
else:
    st.write("No anomalies detected with the current settings.")

st.markdown("---")
st.info("💡 **A note on caching:** The data generation and preprocessing steps are cached using `@st.cache_data`. This means they will only run once and the results will be saved. When you change the Z-Score threshold, only the anomaly detection and visualization functions will re-run, making the app fast and efficient!")
