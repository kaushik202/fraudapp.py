import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, mean_absolute_error
import io
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Fraud Forecast App",
    page_icon="📈",
    layout="wide"
)

# Title and description
st.title("📈 Fraud Forecast Dashboard")
st.markdown("""
This web application performs time series forecasting of fraud counts using Facebook's Prophet model.
Upload your transaction data or use the sample data to generate fraud forecasts.
""")

# Sidebar for navigation and settings
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox(
    "Choose the app mode",
    ["Data Overview", "Time Series Analysis", "Forecasting", "Model Evaluation"]
)

# Sidebar settings
st.sidebar.markdown("---")
st.sidebar.header("Model Settings")
forecast_periods = st.sidebar.slider(
    "Forecast Periods (days)", 
    min_value=7, 
    max_value=90, 
    value=30, 
    step=1
)

enable_seasonality = st.sidebar.checkbox("Enable Daily Seasonality", value=True)
confidence_interval = st.sidebar.slider(
    "Confidence Interval", 
    min_value=0.80, 
    max_value=0.99, 
    value=0.95, 
    step=0.01
)

# Sample data generator
@st.cache_data
def generate_sample_data():
    """Generate sample fraud data for demonstration"""
    np.random.seed(42)
    
    # Generate 365 days of data (1 year)
    days = 365
    timestamps = np.arange(days)
    
    # Generate base fraud counts with trend and seasonality
    trend = 5 + 0.01 * timestamps
    seasonality = 3 * np.sin(2 * np.pi * timestamps / 30)  # Monthly seasonality
    noise = np.random.normal(0, 1.5, days)
    
    fraud_counts = np.maximum(0, trend + seasonality + noise).astype(int)
    
    # Generate other features
    loan_amount = np.random.normal(500000, 200000, days)
    credit_score = np.random.normal(600, 100, days)
    monthly_installment = np.random.normal(20000, 5000, days)
    days_late = np.random.exponential(3, days)
    outstanding_balance = np.random.normal(250000, 100000, days)
    
    # Create DataFrame
    df = pd.DataFrame({
        'ds': pd.date_range(start='2023-01-01', periods=days),
        'y': fraud_counts,
        'loan_amount': loan_amount,
        'credit_score': credit_score,
        'monthly_installment': monthly_installment,
        'days_late': days_late,
        'outstanding_balance': outstanding_balance
    })
    
    return df

@st.cache_data
def create_aggregated_data(df):
    """Create aggregated time series data for modeling"""
    # If the dataset is at transaction level (like in the notebook)
    if 'fraud_flag' in df.columns:
        # This would be the original data processing from the notebook
        fraud_ts = df.groupby('timestamp').agg({
            'fraud_flag': 'sum',
            'loan_amount': 'mean',
            'credit_score': 'mean',
            'monthly_installment': 'mean',
            'days_late': 'mean',
            'outstanding_balance': 'mean'
        }).reset_index()
        
        fraud_ts.rename(columns={'fraud_flag': 'y', 'timestamp': 'ds'}, inplace=True)
        fraud_ts['ds'] = pd.to_datetime(fraud_ts['ds'], unit='D', origin='2023-01-01')
    else:
        # For our sample data which is already aggregated
        fraud_ts = df.copy()
    
    return fraud_ts

# Main app logic
if app_mode == "Data Overview":
    st.header("📊 Data Overview")
    
    # Data upload or sample selection
    data_source = st.radio(
        "Choose data source:",
        ["Use Sample Data", "Upload Your Own Data"]
    )
    
    if data_source == "Upload Your Own Data":
        uploaded_file = st.file_uploader(
            "Upload a CSV file with transaction data", 
            type=['csv']
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success("Data uploaded successfully!")
            except Exception as e:
                st.error(f"Error reading file: {e}")
                st.info("Using sample data instead")
                df = generate_sample_data()
        else:
            st.info("Upload a CSV file or use sample data")
            df = generate_sample_data()
    else:
        df = generate_sample_data()
        st.info("Using sample fraud data for demonstration")
    
    # Display data
    st.subheader("Data Preview")
    st.dataframe(df.head(10))
    
    # Data information
    st.subheader("Data Information")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Records", len(df))
    
    with col2:
        if 'y' in df.columns:
            st.metric("Total Fraud Cases", int(df['y'].sum()))
    
    with col3:
        if 'ds' in df.columns:
            date_range = f"{df['ds'].min().date()} to {df['ds'].max().date()}"
            st.metric("Date Range", date_range)
    
    # Data statistics
    st.subheader("Data Statistics")
    st.dataframe(df.describe())
    
    # Show raw data option
    if st.checkbox("Show Raw Data"):
        st.dataframe(df)

elif app_mode == "Time Series Analysis":
    st.header("📈 Time Series Analysis")
    
    # Load data
    df = generate_sample_data()
    fraud_ts = create_aggregated_data(df)
    
    # Plot time series
    st.subheader("Fraud Count Over Time")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(fraud_ts['ds'], fraud_ts['y'], marker='o', markersize=3, linewidth=2)
    ax.set_xlabel('Date')
    ax.set_ylabel('Fraud Count')
    ax.set_title('Daily Fraud Count Over Time')
    ax.grid(True, alpha=0.3)
    ax.fill_between(fraud_ts['ds'], 0, fraud_ts['y'], alpha=0.3)
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    # Additional features visualization
    st.subheader("Additional Features Over Time")
    
    feature_to_plot = st.selectbox(
        "Select feature to visualize:",
        ['loan_amount', 'credit_score', 'monthly_installment', 'days_late', 'outstanding_balance']
    )
    
    if feature_to_plot in fraud_ts.columns:
        fig2, ax2 = plt.subplots(figsize=(12, 4))
        ax2.plot(fraud_ts['ds'], fraud_ts[feature_to_plot], color='orange', linewidth=2)
        ax2.set_xlabel('Date')
        ax2.set_ylabel(feature_to_plot.replace('_', ' ').title())
        ax2.set_title(f'{feature_to_plot.replace("_", " ").title()} Over Time')
        ax2.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        st.pyplot(fig2)
    
    # Correlation analysis
    st.subheader("Correlation Analysis")
    
    if st.checkbox("Show Correlation Matrix"):
        numeric_cols = fraud_ts.select_dtypes(include=[np.number]).columns
        corr_matrix = fraud_ts[numeric_cols].corr()
        
        fig3, ax3 = plt.subplots(figsize=(10, 8))
        im = ax3.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        ax3.set_xticks(range(len(numeric_cols)))
        ax3.set_yticks(range(len(numeric_cols)))
        ax3.set_xticklabels(numeric_cols, rotation=45, ha='right')
        ax3.set_yticklabels(numeric_cols)
        
        # Add correlation values to heatmap
        for i in range(len(numeric_cols)):
            for j in range(len(numeric_cols)):
                text = ax3.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                               ha="center", va="center", color="white" if abs(corr_matrix.iloc[i, j]) > 0.5 else "black")
        
        plt.colorbar(im)
        ax3.set_title("Correlation Matrix")
        st.pyplot(fig3)

elif app_mode == "Forecasting":
    st.header("🔮 Fraud Forecasting with Prophet")
    
    # Load and prepare data
    df = generate_sample_data()
    fraud_ts = create_aggregated_data(df)
    
    # Train Prophet model
    with st.spinner('Training Prophet model...'):
        # Initialize and fit the model
        prophet_model = Prophet(
            daily_seasonality=enable_seasonality,
            interval_width=confidence_interval
        )
        
        prophet_model.fit(fraud_ts[['ds', 'y']])
        
        # Create future dataframe
        future = prophet_model.make_future_dataframe(periods=forecast_periods)
        
        # Make forecast
        forecast = prophet_model.predict(future)
        
        st.success("Model trained successfully!")
    
    # Display forecast results
    st.subheader("Forecast Results")
    
    # Plot forecast with Plotly
    st.write("### Forecast Visualization")
    
    # Use Prophet's built-in plotly functionality
    fig = plot_plotly(prophet_model, forecast)
    fig.update_layout(
        title="Fraud Count Forecast (Prophet)",
        xaxis_title="Date",
        yaxis_title="Fraud Count",
        height=600
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Show forecast components
    if st.checkbox("Show Forecast Components"):
        st.write("### Forecast Components")
        fig_components = plot_components_plotly(prophet_model, forecast)
        st.plotly_chart(fig_components, use_container_width=True)
    
    # Display forecast table
    st.subheader("Forecast Data")
    
    # Show only future predictions
    future_forecast = forecast[forecast['ds'] > fraud_ts['ds'].max()].copy()
    future_forecast['ds'] = future_forecast['ds'].dt.date
    
    # Format columns for display
    display_cols = ['ds', 'yhat', 'yhat_lower', 'yhat_upper']
    display_df = future_forecast[display_cols].copy()
    display_df.columns = ['Date', 'Predicted Fraud', 'Lower Bound', 'Upper Bound']
    display_df = display_df.round(2)
    
    st.dataframe(display_df)
    
    # Download forecast data
    csv = display_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Forecast as CSV",
        data=csv,
        file_name=f"fraud_forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )
    
    # Key metrics from forecast
    st.subheader("Forecast Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_forecast = future_forecast['yhat'].mean()
        st.metric("Average Predicted Fraud", f"{avg_forecast:.1f}")
    
    with col2:
        max_forecast = future_forecast['yhat'].max()
        st.metric("Maximum Predicted Fraud", f"{max_forecast:.1f}")
    
    with col3:
        min_forecast = future_forecast['yhat'].min()
        st.metric("Minimum Predicted Fraud", f"{min_forecast:.1f}")

elif app_mode == "Model Evaluation":
    st.header("📊 Model Evaluation")
    
    # Load and prepare data
    df = generate_sample_data()
    fraud_ts = create_aggregated_data(df)
    
    # Split data into train and test sets
    train_size = int(len(fraud_ts) * 0.8)
    train = fraud_ts.iloc[:train_size].copy()
    test = fraud_ts.iloc[train_size:].copy()
    
    st.subheader("Train-Test Split")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Training Period", f"{train['ds'].min().date()} to {train['ds'].max().date()}")
        st.metric("Training Samples", len(train))
    
    with col2:
        st.metric("Testing Period", f"{test['ds'].min().date()} to {test['ds'].max().date()}")
        st.metric("Testing Samples", len(test))
    
    # Train model on training data
    with st.spinner('Training and evaluating model...'):
        prophet_model = Prophet(
            daily_seasonality=enable_seasonality,
            interval_width=confidence_interval
        )
        
        prophet_model.fit(train[['ds', 'y']])
        
        # Create future dataframe for test period
        future_dates = prophet_model.make_future_dataframe(periods=len(test))
        
        # Make predictions
        forecast = prophet_model.predict(future_dates)
        
        # Extract predictions for test period
        test_forecast = forecast.iloc[train_size:].copy()
        test_forecast = test_forecast.set_index('ds')
        test = test.set_index('ds')
        
        # Calculate metrics
        mse = mean_squared_error(test['y'], test_forecast['yhat'])
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(test['y'], test_forecast['yhat'])
        
        # Calculate MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((test['y'] - test_forecast['yhat']) / test['y'])) * 100
        
        st.success("Model evaluation completed!")
    
    # Display evaluation metrics
    st.subheader("Model Performance Metrics")
    
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    
    with metric_col1:
        st.metric("RMSE", f"{rmse:.2f}")
    
    with metric_col2:
        st.metric("MAE", f"{mae:.2f}")
    
    with metric_col3:
        st.metric("MSE", f"{mse:.2f}")
    
    with metric_col4:
        st.metric("MAPE", f"{mape:.2f}%")
    
    # Visualize predictions vs actuals
    st.subheader("Predictions vs Actuals (Test Set)")
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot actual values
    ax.plot(test.index, test['y'], 'b-', label='Actual', linewidth=2, marker='o', markersize=4)
    
    # Plot predicted values
    ax.plot(test_forecast.index, test_forecast['yhat'], 'r--', label='Predicted', linewidth=2)
    
    # Plot confidence interval
    ax.fill_between(
        test_forecast.index,
        test_forecast['yhat_lower'],
        test_forecast['yhat_upper'],
        color='gray', alpha=0.2, label='Confidence Interval'
    )
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Fraud Count')
    ax.set_title('Model Predictions vs Actual Values (Test Set)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    st.pyplot(fig)
    
    # Residual analysis
    st.subheader("Residual Analysis")
    
    # Calculate residuals
    residuals = test['y'] - test_forecast['yhat']
    
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Residuals over time
    ax1.plot(residuals.index, residuals, 'o', alpha=0.7)
    ax1.axhline(y=0, color='r', linestyle='--')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Residuals (Actual - Predicted)')
    ax1.set_title('Residuals Over Time')
    ax1.grid(True, alpha=0.3)
    
    # Residual histogram
    ax2.hist(residuals, bins=20, edgecolor='black', alpha=0.7)
    ax2.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax2.set_xlabel('Residuals')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Residual Distribution')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig2)
    
    # Model interpretation
    st.subheader("Model Interpretation")
    
    interpretation = """
    ### How to Interpret the Results:
    
    1. **RMSE (Root Mean Squared Error)**: Measures the average magnitude of prediction errors. 
       Lower values indicate better model performance.
    
    2. **MAE (Mean Absolute Error)**: Average absolute difference between actual and predicted values.
    
    3. **MAPE (Mean Absolute Percentage Error)**: Percentage error in predictions. 
       Below 10% is generally considered good for time series forecasting.
    
    4. **Residual Analysis**: 
       - Residuals should be randomly distributed around zero
       - No clear patterns in residuals over time
       - Normally distributed residuals indicate good model fit
    
    5. **Confidence Intervals**: Wider intervals indicate greater uncertainty in predictions.
    """
    
    st.markdown(interpretation)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info("""
This app uses Facebook's Prophet model for time series forecasting of fraud patterns.

**Features:**
- Time series analysis
- Prophet model forecasting
- Model evaluation metrics
- Interactive visualizations

Upload your own data or explore with sample data.
""")

# Run instructions
if __name__ == "__main__":
    # This is already handled by Streamlit
    pass