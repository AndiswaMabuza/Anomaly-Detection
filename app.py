import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import shap
import streamlit_shap as st_shap

st.set_page_config(layout="wide", page_title="Advanced Anomaly Detection Dashboard")

st.title("Proactive Multivariate Anomaly Detection")
st.markdown("""
Welcome to the Anomaly Detection Dashboard. This application uses an **Isolation Forest** model to detect anomalies in a multivariate dataset and provides **SHAP-based explanations** for model decisions.
""")

# --- Caching Data Generation & Preprocessing ---
@st.cache_data
def generate_multivariate_data(num_records=20000):
    start_date = datetime.now() - timedelta(days=90)
    dates = pd.to_datetime(np.random.uniform(start_date.timestamp(), datetime.now().timestamp(), num_records), unit='s')
    dates = np.sort(dates)

    data = {
        'timestamp': dates,
        'account_id': np.random.choice([f'ACCT{i}' for i in range(50)], num_records),
        'transaction_amount': np.random.lognormal(mean=4, sigma=1.5, size=num_records).round(2),
        'transaction_type': np.random.choice(['debit', 'credit'], num_records, p=[0.7, 0.3]),
        'merchant_category': np.random.choice(['groceries', 'online_retail', 'travel', 'utility', 'restaurant'], num_records, p=[0.4, 0.3, 0.1, 0.1, 0.1]),
        'ip_address': [f'{np.random.randint(1, 255)}.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}' for _ in range(num_records)],
        'transaction_lat': np.random.uniform(34.0, 40.0, num_records),
        'transaction_lon': np.random.uniform(-118.0, -74.0, num_records),
    }
    df = pd.DataFrame(data)

    num_anomalies = int(0.02 * num_records)
    anomaly_indices = np.random.choice(df.index, num_anomalies, replace=False)

    # Injecting different types of anomalies
    df.loc[anomaly_indices[:num_anomalies//3], 'transaction_amount'] *= np.random.uniform(50, 100, num_anomalies//3)
    df.loc[anomaly_indices[num_anomalies//3:2*num_anomalies//3], 'transaction_amount'] /= np.random.uniform(5, 10, num_anomalies//3)
    df.loc[anomaly_indices[2*num_anomalies//3:], 'transaction_lat'] += np.random.uniform(10, 20, num_anomalies//3)
    df.loc[anomaly_indices[2*num_anomalies//3:], 'transaction_lon'] -= np.random.uniform(10, 20, num_anomalies//3)

    return df

@st.cache_data
def preprocess_data(df_input):
    df = df_input.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.sort_values('timestamp', inplace=True)
    df['transaction_hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.day_name()
    df['day_of_week_encoded'] = df['timestamp'].dt.dayofweek
    df['time_between_transactions'] = df.groupby('account_id')['timestamp'].diff().dt.total_seconds().fillna(0)
    
    # NEW: Create a rolling average feature for each account
    df['rolling_avg_amount'] = df.groupby('account_id')['transaction_amount'].transform(lambda x: x.ewm(span=100, adjust=False).mean())

    # One-hot encode categorical features
    df = pd.get_dummies(df, columns=['merchant_category', 'transaction_type'], dtype=int)
    # Note: `ip_address` and `account_id` are not one-hot encoded to avoid a high-dimensional sparse matrix.
    # Instead, we create a rolling average feature for `account_id`.
    return df

@st.cache_data
def train_model(df_scaled, contamination):
    model = IsolationForest(contamination=contamination, random_state=42)
    model.fit(df_scaled)
    return model

# --- Sidebar for user input and controls ---
st.sidebar.header("Configuration")
num_records = st.sidebar.slider("Number of records", 5000, 50000, 20000)
contamination = st.sidebar.slider("Model Contamination", 0.01, 0.1, 0.02, 0.005)
st.sidebar.info("""
**What is Contamination?**
It's the expected proportion of outliers in the data, a value between 0 and 0.5. The model uses this to set a threshold for determining anomalies.
""")

with st.spinner('Generating and preprocessing data...'):
    data = generate_multivariate_data(num_records)
    df_preprocessed = preprocess_data(data)
st.success('Data generated and preprocessed successfully!')

# Define features for the model
feature_cols = [
    'transaction_amount',
    'rolling_avg_amount', # NEW FEATURE
    'transaction_hour',
    'day_of_week_encoded',
    'time_between_transactions',
    'transaction_lat',
    'transaction_lon'
] + [col for col in df_preprocessed.columns if 'merchant_category_' in col or 'transaction_type_' in col]

# Standardize features
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_preprocessed[feature_cols])
df_scaled = pd.DataFrame(df_scaled, columns=feature_cols)

# Train the model
with st.spinner('Training Isolation Forest model...'):
    model = train_model(df_scaled, contamination)
st.success('Model trained successfully!')

df_preprocessed['anomaly_prediction'] = model.predict(df_scaled)
df_preprocessed['anomaly_score'] = model.decision_function(df_scaled)
df_preprocessed['is_anomaly'] = df_preprocessed['anomaly_prediction'] == -1

# --- Section: Data Exploration ---
st.header("1. Exploratory Data Analysis (EDA)")
st.write("A quick look at the data's characteristics and distributions.")
col1, col2 = st.columns(2)
with col1:
    fig_amount = px.histogram(df_preprocessed, x='transaction_amount', log_y=True, title='Distribution of Transaction Amounts (Log Scale)', nbins=100)
    st.plotly_chart(fig_amount, use_container_width=True)
with col2:
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily_counts = df_preprocessed['day_of_week'].value_counts().reindex(day_order)
    fig_daily = px.bar(x=daily_counts.index, y=daily_counts.values, title='Transaction Volume by Day of Week', labels={'x': 'Day of Week', 'y': 'Number of Transactions'})
    st.plotly_chart(fig_daily, use_container_width=True)

# --- Section: Anomaly Detection Results ---
st.header("2. Anomaly Detection Results")
total_anomalies = df_preprocessed['is_anomaly'].sum()
st.info(f"**Total Anomalies Detected:** {total_anomalies} (approximately {(total_anomalies/len(df_preprocessed)*100):.2f}% of all transactions)")

# --- Section: Anomaly Visualization ---
st.header("3. Anomaly Visualization")
st.write("Select an account to visualize its transactions and the detected anomalies.")
unique_accounts = df_preprocessed['account_id'].unique()
sample_account = st.selectbox("Select an Account ID", unique_accounts)

if sample_account:
    account_df = df_preprocessed[df_preprocessed['account_id'] == sample_account].sort_values('timestamp')
    fig_anomalies = go.Figure()
    fig_anomalies.add_trace(go.Scatter(x=account_df['timestamp'], y=account_df['transaction_amount'], mode='lines', name='Transaction Amount', line=dict(color='blue')))
    anomaly_df = account_df[account_df['is_anomaly']]
    fig_anomalies.add_trace(go.Scatter(x=anomaly_df['timestamp'], y=anomaly_df['transaction_amount'], mode='markers', name='Detected Anomaly', marker=dict(color='red', size=8, symbol='x')))
    fig_anomalies.update_layout(title=f'Transaction Amounts for Account {sample_account} with Anomalies Highlighted', xaxis_title='Timestamp', yaxis_title='Transaction Amount', hovermode='x unified', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig_anomalies, use_container_width=True)

# --- Section: Explainable AI (XAI) with SHAP ---
st.header("4. Explainable AI (XAI) with SHAP")
st.write("Understand why a specific transaction was flagged as an anomaly by the model and see an overall feature importance summary.")

if total_anomalies > 0:
    explainer = shap.TreeExplainer(model)
    df_anomalies_scaled = df_scaled[df_preprocessed['is_anomaly']].sort_values(by='anomaly_score', ascending=True)

    # NEW: Display SHAP summary plot for overall feature importance
    st.subheader("Overall Anomaly Feature Importance (Summary Plot)")
    with st.spinner("Generating SHAP summary plot..."):
        shap_values_anomalies = explainer.shap_values(df_anomalies_scaled)
        st_shap.st_shap(shap.summary_plot(shap_values_anomalies, df_anomalies_scaled, show=False), height=400)
    st.write("---")

    # Select an individual anomaly to explain
    anomaly_options = df_preprocessed[df_preprocessed['is_anomaly']].sort_values('anomaly_score').head(10)
    selected_anomaly_idx = st.selectbox(
        "Select an anomalous transaction to explain:",
        anomaly_options.index,
        format_func=lambda idx: f"Transaction ID: {idx} | Amount: ${df_preprocessed.loc[idx, 'transaction_amount']:.2f}"
    )

    if selected_anomaly_idx:
        st.subheader(f"Explanation for Transaction ID: {selected_anomaly_idx}")
        with st.spinner("Generating SHAP force plot..."):
            shap_values = explainer.shap_values(df_scaled.loc[[selected_anomaly_idx]])
            st.write(f"The anomaly score for this transaction is: **{df_preprocessed.loc[selected_anomaly_idx, 'anomaly_score']:.4f}**")
            st_shap.st_shap(
                shap.force_plot(
                    explainer.expected_value,
                    shap_values[0],
                    df_scaled.loc[[selected_anomaly_idx]],
                    feature_names=df_scaled.columns,
                ),
                width=1000
            )
            st.info("""
            **How to interpret the force plot:**
            - **Red arrows** show features that push the prediction towards a more **negative** anomaly score (i.e., more anomalous).
            - **Blue arrows** show features that push the prediction towards a more **positive** score (i.e., more normal).
            """)
else:
    st.write("No anomalies detected with the current contamination value. Please adjust the slider in the sidebar.")

st.markdown("---")
st.markdown("""
<div style="text-align: center;">
    <p>Developed by Andiswa Mabuza</p>
    <p>Email: <a href="mailto:Amabuza53@gmail.com">Amabuza53@gmail.com</a> | Developer Site: <a href="https://andiswamabuza.vercel.app" target="_blank">andiswamabuza.vercel.app</a></p>
</div>
""", unsafe_allow_html=True)
