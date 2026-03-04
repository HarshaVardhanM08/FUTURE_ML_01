import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
import plotly.express as px
import calendar
import warnings

warnings.filterwarnings("ignore")

# Page Configuration
st.set_page_config(
    page_title="Sales Forecasting",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.markdown("## 📊 Configuration")
uploaded_file = st.sidebar.file_uploader("Upload Sales Data (CSV)", type=['csv'])

if uploaded_file is None:
    st.info("👈 Please upload a CSV file using the sidebar to get started!")
    st.stop()

# Load Data
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip()
    return df

df = load_data(uploaded_file)

# Data Cleaning & Feature Engineering
@st.cache_data
def preprocess_data(df):
    df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True)
    df = df.sort_values('Order Date')
    df['Year'] = df['Order Date'].dt.year
    df['Month'] = df['Order Date'].dt.month
    df['Quarter'] = df['Order Date'].dt.quarter
    df['Day_Name'] = df['Order Date'].dt.day_name()
    return df

df = preprocess_data(df)

# Aggregate Daily Revenue
@st.cache_data
def aggregate_daily_sales(df):
    daily_sales = df.groupby('Order Date')['Sales'].sum().reset_index()
    daily_sales['Rolling_7'] = daily_sales['Sales'].rolling(7).mean()
    daily_sales['Rolling_30'] = daily_sales['Sales'].rolling(30).mean()
    return daily_sales

daily_sales = aggregate_daily_sales(df)

# Main Title
st.title("📊 Sales Forecasting Dashboard")
st.markdown("---")

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Overview",
    "🔮 Forecasting",
    "📉 Analysis",
    "📊 Models",
    "📑 Executive Report"
])

# ===== TAB 1: OVERVIEW =====
with tab1:
    st.header("Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        st.metric("Date Range", f"{df['Order Date'].min().strftime('%Y-%m-%d')} to {df['Order Date'].max().strftime('%Y-%m-%d')}")
    with col3:
        st.metric("Total Revenue", f"${df['Sales'].sum():,.2f}")
    with col4:
        st.metric("Avg Daily Revenue", f"${daily_sales['Sales'].mean():,.2f}")
    
    st.markdown("---")
    
    # Data Quality
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Missing Values")
        missing = df.isnull().sum()
        st.write(missing[missing > 0] if missing.any() else "✅ No missing values")
    
    with col2:
        st.subheader("Data Types")
        st.write(df.dtypes)
    
    st.markdown("---")
    st.subheader("Raw Data Preview")
    st.dataframe(df.head(10), use_container_width=True)

# ===== TAB 2: FORECASTING =====
with tab2:
    st.header("🔮 Sales Forecast")
    
    # Forecast Parameters
    col1, col2 = st.columns(2)
    with col1:
        future_period = st.slider("Forecast Period (days)", 30, 365, 90)
    with col2:
        yearly_seasonality = st.checkbox("Include Yearly Seasonality", value=True)
    
    st.info(f"Forecasting {future_period} days ahead using Facebook's Prophet model")
    
    # Train Prophet Model
    @st.cache_resource
    def train_prophet_model(daily_sales, future_period, yearly_seasonality):
        prophet_df = daily_sales.rename(columns={'Order Date': 'ds', 'Sales': 'y'})
        train = prophet_df[:-future_period]
        test = prophet_df[-future_period:]
        
        model = Prophet(
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=True,
            daily_seasonality=False,
            interval_width=0.95
        )
        
        model.fit(train)
        future = model.make_future_dataframe(periods=future_period)
        forecast = model.predict(future)
        
        return model, forecast, train, test
    
    model, forecast, train, test = train_prophet_model(daily_sales, future_period, yearly_seasonality)
    
    # Forecast Visualization
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=daily_sales['Order Date'],
        y=daily_sales['Sales'],
        mode='lines',
        name='Historical Sales',
        line=dict(color='#1f77b4', width=2)
    ))
    
    # Forecast
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat'],
        mode='lines',
        name='Forecast',
        line=dict(color='#ff7f0e', width=2, dash='dash')
    ))
    
    # Confidence Interval
    fig.add_trace(go.Scatter(
        x=list(forecast['ds']) + list(forecast['ds'][::-1]),
        y=list(forecast['yhat_upper']) + list(forecast['yhat_lower'][::-1]),
        fill='toself',
        name='Confidence Interval',
        fillcolor='rgba(255, 127, 14, 0.2)',
        line=dict(color='rgba(255,255,255,0)')
    ))
    
    fig.update_layout(
        title="90-Day Revenue Forecast with Confidence Intervals",
        xaxis_title="Date",
        yaxis_title="Sales ($)",
        hovermode='x unified',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Model Components
    st.subheader("📊 Forecast Components")
    components_fig = model.plot_components(forecast)
    st.pyplot(components_fig, use_container_width=True)
    
    # Model Evaluation
    st.markdown("---")
    st.subheader("📉 Model Performance")
    
    # Use only the last future_period values from test data
    actual = test['y'].values[-future_period:] if len(test) >= future_period else test['y'].values
    predictions = forecast[-len(actual):]['yhat'].values
    
    mae = mean_absolute_error(actual, predictions)
    rmse = np.sqrt(mean_squared_error(actual, predictions))
    mape = np.mean(np.abs((actual - predictions) / actual)) * 100
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("MAE (Mean Absolute Error)", f"${mae:,.2f}")
    with col2:
        st.metric("RMSE (Root Mean Squared Error)", f"${rmse:,.2f}")
    with col3:
        st.metric("MAPE (Mean Absolute % Error)", f"{mape:.2f}%")
    
    # Forecast Query
    st.markdown("---")
    st.subheader("🔍 Query Forecast by Date")
    
    query_date = st.date_input("Select a date to view forecast")
    
    query_result = forecast[forecast['ds'] == pd.Timestamp(query_date)]
    
    if not query_result.empty:
        pred = query_result['yhat'].values[0]
        lower = query_result['yhat_lower'].values[0]
        upper = query_result['yhat_upper'].values[0]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Predicted Sales", f"${pred:,.2f}")
        with col2:
            st.metric("Lower Bound (95%)", f"${lower:,.2f}")
        with col3:
            st.metric("Upper Bound (95%)", f"${upper:,.2f}")
    else:
        min_date = forecast['ds'].min().strftime('%B %d, %Y')
        max_date = forecast['ds'].max().strftime('%B %d, %Y')
        st.warning(f"⚠️ Date not in forecast horizon. Please select a date between {min_date} and {max_date}")

# ===== TAB 3: ANALYSIS =====
with tab3:
    st.header("📉 Sales Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Revenue Trend with Rolling Averages")
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(daily_sales['Order Date'], daily_sales['Sales'], alpha=0.4, label='Daily Sales')
        ax.plot(daily_sales['Order Date'], daily_sales['Rolling_7'], label='7-Day Avg', linewidth=2)
        ax.plot(daily_sales['Order Date'], daily_sales['Rolling_30'], label='30-Day Avg', linewidth=2)
        ax.set_xlabel('Date')
        ax.set_ylabel('Sales ($)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        st.subheader("Revenue Distribution")
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.hist(daily_sales['Sales'], bins=30, color='skyblue', edgecolor='black')
        ax.set_xlabel('Daily Sales ($)')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        st.pyplot(fig)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Revenue by Category")
        category_totals = df.groupby('Category')['Sales'].sum().sort_values(ascending=False)
        fig = px.bar(x=category_totals.values, y=category_totals.index, orientation='h',
                     labels={'x': 'Sales ($)', 'y': 'Category'},
                     color=category_totals.values,
                     color_continuous_scale='viridis')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Revenue by Region")
        region_totals = df.groupby('Region')['Sales'].sum().sort_values(ascending=False)
        fig = px.bar(x=region_totals.values, y=region_totals.index, orientation='h',
                     labels={'x': 'Sales ($)', 'y': 'Region'},
                     color=region_totals.values,
                     color_continuous_scale='plasma')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Average Sales by Day of Week")
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_avg = df.groupby('Day_Name')['Sales'].mean().reindex(day_order)
        fig = px.bar(x=day_avg.index, y=day_avg.values,
                     labels={'x': 'Day', 'y': 'Avg Sales ($)'},
                     color=day_avg.values,
                     color_continuous_scale='blues')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Revenue by Month")
        monthly_sales = df.groupby('Month')['Sales'].sum()
        month_names = [calendar.month_name[i] for i in monthly_sales.index]
        fig = px.bar(x=month_names, y=monthly_sales.values,
                     labels={'x': 'Month', 'y': 'Sales ($)'},
                     color=monthly_sales.values,
                     color_continuous_scale='greens')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

# ===== TAB 4: MODELS =====
with tab4:
    st.header("📊 Model Comparison")
    
    st.info("Comparing Prophet Forecasting Model with Linear Regression Baseline")
    
    # Linear Regression Baseline
    @st.cache_data
    def train_linear_regression(daily_sales, future_period):
        daily_sales_copy = daily_sales.copy()
        daily_sales_copy['Time_Index'] = np.arange(len(daily_sales_copy))
        
        X = daily_sales_copy[['Time_Index']]
        y = daily_sales_copy['Sales']
        
        lr_model = LinearRegression()
        lr_model.fit(X[:-future_period], y[:-future_period])
        
        lr_pred = lr_model.predict(X[-future_period:])
        return lr_model, lr_pred
    
    future_period = 90
    lr_model, lr_pred = train_linear_regression(daily_sales, future_period)
    
    # Comparison Metrics
    # Ensure consistent lengths
    test_actual = daily_sales['Sales'].iloc[-future_period:].values
    prophet_predictions = forecast[-future_period:]['yhat'].values
    lr_predictions = lr_pred[-future_period:] if len(lr_pred) >= future_period else lr_pred
    
    # Align lengths
    min_len = min(len(test_actual), len(prophet_predictions))
    test_actual = test_actual[-min_len:]
    prophet_predictions = prophet_predictions[-min_len:]
    
    prophet_mae = mean_absolute_error(test_actual, prophet_predictions)
    prophet_rmse = np.sqrt(mean_squared_error(test_actual, prophet_predictions))
    
    min_len_lr = min(len(test_actual), len(lr_predictions))
    lr_mae = mean_absolute_error(test_actual[-min_len_lr:], lr_predictions[-min_len_lr:])
    lr_rmse = np.sqrt(mean_squared_error(test_actual[-min_len_lr:], lr_predictions[-min_len_lr:]))
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Prophet MAE", f"${prophet_mae:,.2f}")
    with col2:
        st.metric("Linear Regression MAE", f"${lr_mae:,.2f}")
    with col3:
        st.metric("Prophet RMSE", f"${prophet_rmse:,.2f}")
    with col4:
        st.metric("Linear Regression RMSE", f"${lr_rmse:,.2f}")
    
    # Model Comparison Chart
    st.subheader("Prediction Comparison")
    
    # Ensure all arrays have the same length
    comparison_length = min(len(daily_sales) - (len(daily_sales) - future_period), future_period, len(lr_pred))
    
    comparison_df = pd.DataFrame({
        'Date': daily_sales['Order Date'].iloc[-comparison_length:].values,
        'Actual': daily_sales['Sales'].iloc[-comparison_length:].values,
        'Prophet': forecast[-comparison_length:]['yhat'].values,
        'Linear Regression': lr_pred[-comparison_length:]
    })
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=comparison_df['Date'], y=comparison_df['Actual'],
                             mode='lines', name='Actual', line=dict(color='blue', width=2)))
    fig.add_trace(go.Scatter(x=comparison_df['Date'], y=comparison_df['Prophet'],
                             mode='lines', name='Prophet', line=dict(color='red', width=2, dash='dash')))
    fig.add_trace(go.Scatter(x=comparison_df['Date'], y=comparison_df['Linear Regression'],
                             mode='lines', name='Linear Regression', line=dict(color='green', width=2, dash='dot')))
    
    fig.update_layout(
        title="Model Predictions vs Actual Sales",
        xaxis_title="Date",
        yaxis_title="Sales ($)",
        hovermode='x unified',
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

# ===== TAB 5: EXECUTIVE REPORT =====
with tab5:
    st.header("📑 Executive Summary Report")
    
    # Calculate metrics
    peak_month_number = df.groupby(df['Month'])['Sales'].sum().idxmax()
    peak_month = calendar.month_name[peak_month_number]
    best_day = df.groupby('Day_Name')['Sales'].mean().idxmax()
    
    last_forecast = forecast['yhat'].iloc[-1]
    first_forecast = forecast['yhat'].iloc[-future_period]
    projected_growth = ((last_forecast / first_forecast) - 1) * 100
    
    # Generate Report
    report_text = f"""
    <div style="background-color:#111827; padding:20px; border-radius:10px; border-left:5px solid #1f77b4; color:white;">
    
    ## 📊 EXECUTIVE SALES FORECASTING REPORT
    
    ### Dataset Overview
    - **Historical Training Period**: {len(train):,} days
    - **Forecast Horizon**: {future_period} days
    - **Total Historical Revenue**: ${df['Sales'].sum():,.2f}
    - **Date Range**: {df['Order Date'].min().strftime('%B %d, %Y')} to {df['Order Date'].max().strftime('%B %d, %Y')}
    
    ### Forecast Accuracy
    - **Forecast Accuracy (MAPE)**: {mape:.2f}%
    - **Mean Absolute Error**: ${mae:,.2f}
    - **Root Mean Squared Error**: ${rmse:,.2f}
    
    ### 📈 Key Revenue Insights
    - **Highest Revenue Month**: {peak_month}
    - **Top Performing Category**: {category_totals.idxmax()}
    - **Total Category Revenue**: ${category_totals.max():,.2f}
    - **Strongest Sales Region**: {region_totals.idxmax()}
    - **Regional Revenue**: ${region_totals.max():,.2f}
    - **Highest Avg Sales Day**: {best_day}
    
    ### 🔮 Forecast Projections
    - **Projected Growth (Forecast Period)**: {projected_growth:+.2f}%
    - **Last Day Forecast**: ${last_forecast:,.2f}
    - **First Day Forecast**: ${first_forecast:,.2f}
    - **Average Daily Forecast**: ${forecast['yhat'].iloc[-future_period:].mean():,.2f}
    
    ### 📊 Business Segments
    - **Total Segments**: {df['Segment'].nunique()}
    - **Top Segment**: {df.groupby('Segment')['Sales'].sum().idxmax()}
    - **Total Regions**: {df['Region'].nunique()}
    - **Total Categories**: {df['Category'].nunique()}
    
    </div>
    """
    
    st.markdown(report_text, unsafe_allow_html=True)
    
    # Detailed Metrics Tables
    st.markdown("---")
    st.subheader("Category Performance")
    category_details = df.groupby('Category').agg({
        'Sales': ['sum', 'mean', 'count']
    }).round(2)
    category_details.columns = ['Total Sales', 'Avg Sales per Order', 'Order Count']
    st.dataframe(category_details, use_container_width=True)
    
    st.subheader("Regional Performance")
    region_details = df.groupby('Region').agg({
        'Sales': ['sum', 'mean', 'count']
    }).round(2)
    region_details.columns = ['Total Sales', 'Avg Sales per Order', 'Order Count']
    st.dataframe(region_details, use_container_width=True)
    
    st.subheader("Segment Performance")
    segment_details = df.groupby('Segment').agg({
        'Sales': ['sum', 'mean', 'count']
    }).round(2)
    segment_details.columns = ['Total Sales', 'Avg Sales per Order', 'Order Count']
    st.dataframe(segment_details, use_container_width=True)
    
    # Download Report
    st.markdown("---")
    st.subheader("📥 Export Data")
    
    # Forecast Export
    forecast_export = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
    forecast_export.columns = ['Date', 'Forecast', 'Lower Bound', 'Upper Bound']
    
    csv = forecast_export.to_csv(index=False)
    st.download_button(
        label="📊 Download Forecast (CSV)",
        data=csv,
        file_name="sales_forecast.csv",
        mime="text/csv"
    )