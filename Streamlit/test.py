import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split, GridSearchCV
from time import sleep

# =========================
# Data Loading and Preprocessing
# =========================

@st.cache_data
def load_data():
    #file_path = '../Dataset/bike-sharing-hourly.csv'  # for local deployment
    file_path = 'Dataset/bike-sharing-hourly.csv' #for github deployment
    try:
        data_raw = pd.read_csv(file_path)
        data_raw['dteday'] = pd.to_datetime(data_raw['dteday'])
        data_preprocessed = load_and_prepare_data(data_raw.copy())
        return data_raw, data_preprocessed
    except FileNotFoundError:
        st.error("Dataset file not found. Please ensure the file is in the correct folder.")
        return pd.DataFrame(), pd.DataFrame()

def load_and_prepare_data(df):
    df = add_cyclic_features(df, 'hr', 24)
    df = add_cyclic_features(df, 'season', 4)
    df = temp_hum(df)
    df = temp_windspeed(df)
    df = hum_windspeed(df)
    df = add_daylight_period(df)
    df = one_hot_encode_column_single(df, 'daylight_period')
    # Keep 'casual' and 'registered' for visualization; drop them later during modeling
    unnecessary_columns = ['instant', 'atemp']
    df = df.drop(columns=unnecessary_columns, errors='ignore')
    return df

def add_cyclic_features(df, col_name, max_value):
    df[col_name] = df[col_name].astype('int')
    df[f'{col_name}_sin'] = np.sin(2 * np.pi * df[col_name] / max_value)
    df[f'{col_name}_cos'] = np.cos(2 * np.pi * df[col_name] / max_value)
    df.drop(columns=[col_name], inplace=True)
    return df

def temp_hum(df):
    df['temp_hum'] = df['temp'] + df['hum']
    return df

def temp_windspeed(df):
    df['temp_windspeed'] = df['temp'] + df['windspeed']
    return df

def hum_windspeed(df):
    df['hum_windspeed'] = df['hum'] + df['windspeed']
    return df

def add_daylight_period(df):
    def categorize_daylight(hour_sin, hour_cos):
        angle = np.arctan2(hour_sin, hour_cos)
        hour = (angle * 24 / (2 * np.pi)) % 24
        if 5 <= hour < 12:
            return 'morning'
        elif 12 <= hour < 17:
            return 'afternoon'
        elif 17 <= hour < 21:
            return 'evening'
        else:
            return 'night'
    df['daylight_period'] = df.apply(lambda row: categorize_daylight(row['hr_sin'], row['hr_cos']), axis=1)
    return df

def one_hot_encode_column_single(df, column_name):
    encoder = OneHotEncoder(drop='first', sparse_output=False)
    encoder.fit(df[[column_name]])
    encoded_values = encoder.transform(df[[column_name]])
    encoded_column_names = [f"{column_name}_{category}" for category in encoder.categories_[0][1:]]
    encoded_df = pd.DataFrame(encoded_values, columns=encoded_column_names, index=df.index)
    df_encoded = pd.concat([df.drop(columns=[column_name]), encoded_df], axis=1)
    return df_encoded

def align_features(X, reference_X):
    X = X.copy()
    missing_cols = set(reference_X.columns) - set(X.columns)
    for col in missing_cols:
        X[col] = 0
    extra_cols = set(X.columns) - set(reference_X.columns)
    if extra_cols:
        X = X.drop(columns=extra_cols)
    return X[reference_X.columns]

def split_data(df, target_column='cnt', test_year=1, test_month_start=11, val_ratio=0.2):
    # Drop 'casual' and 'registered' for modeling
    df_model = df.drop(columns=['casual', 'registered'], errors='ignore')
    y = df_model[target_column]
    X = df_model.drop(columns=[target_column, 'dteday'])
    df_test = df_model[(df_model['yr'] == test_year) & (df_model['mnth'] >= test_month_start)]
    df_train = df_model.drop(df_test.index)
    X_train = df_train.drop(columns=[target_column, 'dteday'], errors='ignore')
    y_train = df_train[target_column]
    X_test = df_test.drop(columns=[target_column, 'dteday'], errors='ignore')
    y_test = df_test[target_column]
    train_size = int(len(X_train) * (1 - val_ratio))
    X_train_final = X_train.iloc[:train_size]
    X_val = X_train.iloc[train_size:]
    y_train_final = y_train.iloc[:train_size]
    y_val = y_train.iloc[train_size:]
    X_train_aligned = align_features(X_train_final, X_train_final)
    X_val_aligned = align_features(X_val, X_train_final)
    X_test_aligned = align_features(X_test, X_train_final)
    return X_train_aligned, X_val_aligned, X_test_aligned, y_train_final, y_val, y_test

# Load Data
data_raw, data = load_data()
if data_raw.empty or data.empty:
    st.stop()

# Prepare data for modeling
X_train, X_val, X_test, y_train, y_val, y_test = split_data(data)

# Models dictionary
models = {
    "Random Forest": RandomForestRegressor(random_state=42),
    "XGBoost": XGBRegressor(random_state=42),
    "LightGBM": lgb.LGBMRegressor(random_state=42)
}

# =========================
# Streamlit App
# =========================

st.title("Washington DC Bike Sharing Service Analysis and Business Development")

# Create Tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "Data Quality and Exploration",
    "Data Visualization",
    "Business Insights",
    "Feature Engineering",
    "ML Enthusiasts",
    "Business Users",
    "Model Comparison",
    "Real vs Predicted"
])

# ====================================
# Tab 1: Data Quality and Exploration
# ====================================
with tab1:
    st.header("1. Data Quality and Exploration")
    st.subheader("Missing Values and Data Consistency")
    st.write("Checking for missing values in the dataset:")
    missing_data = data_raw.isna().mean()
    st.write(missing_data)
    
    if missing_data.sum() == 0:
        st.write("No missing values found.")
    
    # Outlier Analysis
    st.subheader("Outlier Analysis")
    features = ['casual', 'registered', 'cnt', 'temp', 'hum', 'windspeed']
    
    threshold = st.slider("Choose Z-score threshold for outlier detection", min_value=2.0, max_value=5.0, step=0.1, value=3.0)
    z_scores = data_raw[features].apply(zscore)
    outliers_zscore = (z_scores.abs() > threshold).sum()
    st.write("Outliers Detected per Feature:")
    st.write(outliers_zscore)
    
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=("Outliers in Casual Users", "Outliers in Registered Users", "Outliers in Total Usage")
    )
    fig.add_trace(go.Box(x=data_raw['casual'], name="Casual Users"), row=1, col=1)
    fig.add_trace(go.Box(x=data_raw['registered'], name="Registered Users"), row=2, col=1)
    fig.add_trace(go.Box(x=data_raw['cnt'], name="Total Usage"), row=3, col=1)
    fig.update_layout(
        height=900, width=600, title_text="Outliers in Usage Data", showlegend=False
    )
    st.plotly_chart(fig)
    
    # Daily Average Casual User Counts
    st.subheader("Daily Average Casual User Counts")
    st.write("""
    This plot shows the daily average number of casual bike users over time. 
    It helps identify patterns and trends, such as seasonal variations or spikes in usage.
    """)
    
    # Calculate daily average for casual users
    daily_casual_data = data_raw.groupby('dteday').agg(
        daily_casual=('casual', 'mean')  # Use mean instead of sum
    ).reset_index()
    
    # Create a line plot for daily casual users
    fig_casual = px.line(
        daily_casual_data,
        x='dteday',
        y='daily_casual',
        title="Daily Average Casual User Counts",
        labels={'daily_casual': 'Average Number of Casual Users', 'dteday': 'Date'},
    )
    
    # Customize layout
    fig_casual.update_layout(
        xaxis=dict(title="Date"),
        yaxis=dict(title="Average Number of Casual Users"),
        title_x=0.5
    )
    
    # Display the plot in Streamlit
    st.plotly_chart(fig_casual)
    
    
    # Daily Average Registered User Counts
    st.subheader("Daily Average Registered User Counts")
    st.write("""
    This plot shows the daily average number of registered bike users over time. 
    It provides insights into how registered users (residents) utilize the service on a day-to-day basis, highlighting consistent patterns or seasonal trends.
    """)
    
    # Calculate daily average for registered users
    daily_registered_data = data_raw.groupby('dteday').agg(
        daily_registered=('registered', 'mean')  # Use mean instead of sum
    ).reset_index()
    
    # Create a line plot for daily registered users
    fig_registered = px.line(
        daily_registered_data,
        x='dteday',
        y='daily_registered',
        title="Daily Average Registered User Counts",
        labels={'daily_registered': 'Average Number of Registered Users', 'dteday': 'Date'},
    )
    
    # Customize layout
    fig_registered.update_layout(
        xaxis=dict(title="Date"),
        yaxis=dict(title="Average Number of Registered Users"),
        title_x=0.5
    )
    
    # Display the plot in Streamlit
    st.plotly_chart(fig_registered)
    
    st.write("""
    - **Highest outlier count in casual usage**: The outliers in casual users are not true outliers but reflect an increased variance in usage patterns.
    - **Casual users' variance**: Casual usage patterns are much less consistent than registered users (residents), leading to this higher variability, especially in peak tourist seasons such as spring and summer.
    - **Registered users' stability**: Registered users show more stable behavior with fewer outliers.
    """)

# ====================================
# Tab 2: Data Visualization
# ====================================
with tab2:
    st.header("2. Data Visualization")
    
    # Section: Monthly Average Bike Usage
    st.subheader("Monthly Average Bike Usage (2011-2012)")
    
    # Calculate monthly average usage
    data_raw['month_year'] = data_raw['dteday'].dt.to_period('M')
    monthly_usage = data_raw.groupby('month_year').agg(avg_cnt=('cnt', 'mean')).reset_index()
    
    # Convert 'month_year' back to datetime for Plotly compatibility
    monthly_usage['month_year'] = monthly_usage['month_year'].dt.to_timestamp()
    
    # Plot monthly average usage over time
    fig_monthly_usage = px.line(
        monthly_usage, x='month_year', y='avg_cnt',
        title="Monthly Average Bike Usage (2011-2012)",
        labels={'month_year': 'Month-Year', 'avg_cnt': 'Average Bike Usage (cnt)'}
    )
    
    fig_monthly_usage.update_layout(
        xaxis_title="Month-Year",
        yaxis_title="Average Bike Usage",
        title_x=0.5
    )
    
    # Display the plot in Streamlit
    st.plotly_chart(fig_monthly_usage)
    
    # Monthly Averages of Casual vs Registered Users
    st.subheader("Monthly Averages of Casual vs Registered Users")
    def plot_casual_vs_registered_on_same_axis(data, neon_purple="#DA70D6"):
        """
        Create a single-axis graph showing the monthly averages of casual and registered bike users.
    
        Args:
            data (pd.DataFrame): DataFrame containing the bike-sharing data.
            neon_purple (str): Hex color code for the casual users' line (default: Neon Purple).
        
        Returns:
            go.Figure: Plotly figure object with both data series on the same y-axis.
        """
        # Ensure the date column is in datetime format
        data['dteday'] = pd.to_datetime(data['dteday'], errors='coerce')
    
        # Calculate monthly averages for casual and registered users
        data['month_year'] = data['dteday'].dt.to_period('M')  # Convert to period for grouping by month
        monthly_data = data.groupby('month_year').agg(
            avg_casual=('casual', 'mean'),
            avg_registered=('registered', 'mean')
        ).reset_index()
    
        # Convert 'month_year' back to datetime for compatibility
        monthly_data['month_year'] = monthly_data['month_year'].dt.to_timestamp()
    
        # Create a single-axis graph
        fig = go.Figure()
    
        # Add Average Casual Users
        fig.add_trace(go.Scatter(
            x=monthly_data['month_year'], 
            y=monthly_data['avg_casual'], 
            mode='lines', 
            name='Average Casual Users', 
            line=dict(color=neon_purple)
        ))
    
        # Add Average Registered Users
        fig.add_trace(go.Scatter(
            x=monthly_data['month_year'], 
            y=monthly_data['avg_registered'], 
            mode='lines', 
            name='Average Registered Users', 
            line=dict(color='blue')
        ))
    
        # Configure layout
        fig.update_layout(
            title="Monthly Averages of Casual vs Registered Users",
            xaxis=dict(title="Date"),
            yaxis=dict(title="Average Number of Users"),
            title_x=0.5,
            legend=dict(title="User Type")
        )
        
        return fig
    
    fig_single_axis = plot_casual_vs_registered_on_same_axis(data_raw)
    st.plotly_chart(fig_single_axis, use_container_width=True)
    
    # Average usage by season
    st.subheader("Average Bike Usage by Season")
    
    def plot_average_usage_by_season(df):
        """
        Create a bar chart showing the average bike usage by season.
    
        Args:
            df (pd.DataFrame): The bike-sharing dataset.
    
        Returns:
            go.Figure: Plotly figure object for the average usage by season.
        """
        season_usage = df.groupby('season', observed=True).agg(avg_cnt=('cnt', 'mean')).reset_index()
        season_labels = {1: "Winter", 2: "Spring", 3: "Summer", 4: "Fall"}
        season_usage['Season'] = season_usage['season'].map(season_labels)
        season_usage['avg_cnt_rounded'] = season_usage['avg_cnt'].round()
    
        fig = go.Figure(
            data=go.Bar(
                x=season_usage['Season'],
                y=season_usage['avg_cnt'],
                marker=dict(color='#1f77b4'), 
                text=season_usage['avg_cnt_rounded'], 
                textposition='outside'
            )
        )
    
        fig.update_layout(
            title="Average Bike Usage by Season",
            xaxis_title="Season",
            yaxis_title="Average Bike Usage",
            title_x=0.5,
            bargap=0.4,
            width=800,
            height=500,
            showlegend=False
        )
        return fig
    
    # Generate and display the figure
    fig_season = plot_average_usage_by_season(data_raw)
    st.plotly_chart(fig_season)
    
    # Usage by weather conditions
    st.subheader("Bike Usage by Weather Condition")
    weather_usage = data_raw.groupby('weathersit').agg(avg_cnt=('cnt', 'mean')).reset_index()
    weather_labels = {
        1: "Clear/Partly Cloudy",
        2: "Mist/Cloudy",
        3: "Light Snow/Rain",
        4: "Heavy Rain/Snow"
    }
    weather_usage['Weather'] = weather_usage['weathersit'].map(weather_labels)
    
    fig = go.Figure(
        data=go.Bar(
            x=weather_usage['Weather'], y=weather_usage['avg_cnt'],
            marker=dict(color='#1f77b4'), text=weather_usage['avg_cnt'].round(), textposition='outside'
        )
    )
    fig.update_layout(
        title="Average Bike Usage by Weather Condition",
        xaxis_title="Weather Condition", yaxis_title="Average Bike Usage", title_x=0.5
    )
    st.plotly_chart(fig)
    
    # Hourly usage patterns
    st.subheader("Hourly Bike Usage: Weekdays vs. Weekends")
    data_raw['is_weekend'] = data_raw['weekday'].apply(lambda x: 'Weekend' if x >= 5 else 'Weekday')
    hourly_usage = data_raw.groupby(['is_weekend', 'hr']).agg(avg_cnt=('cnt', 'mean')).reset_index()
    
    # Define neon purple for weekends
    neon_purple = "#DA70D6"
    
    # Create line plot with custom colors for weekends
    fig = px.line(hourly_usage, 
                  x='hr', 
                  y='avg_cnt', 
                  color='is_weekend', 
                  title="Hourly Bike Usage: Weekdays vs. Weekends",
                  labels={'hr': 'Hour of Day', 'avg_cnt': 'Average Usage', 'is_weekend': 'Day Type'})
    
    # Update the color for weekends
    fig.for_each_trace(lambda trace: trace.update(line=dict(color=neon_purple)) if trace.name == 'Weekend' else None)
    
    # Update layout
    fig.update_layout(
        xaxis=dict(title="Hour of Day"),
        yaxis=dict(title="Average Usage"),
        title_x=0.5
    )
    
    st.plotly_chart(fig)
    
    # Preparing Hourly Usage Data for Casual and Registered Users
    data_raw['is_weekend'] = data_raw['weekday'].apply(lambda x: 'Weekend' if x >= 5 else 'Weekday')
    hourly_usage = data_raw.groupby(['is_weekend', 'hr'], observed=True).agg(
        avg_casual=('casual', 'mean'),
        avg_registered=('registered', 'mean')
    ).reset_index()
    
    # Function to Plot Hourly Usage
    def plot_hourly_usage(hourly_usage, user_type, color_neon_purple='#DA70D6'):
        """
        Plot the average hourly bike usage for casual or registered users.
    
        Args:
            hourly_usage (pd.DataFrame): DataFrame with hourly usage.
            user_type (str): Either 'casual' or 'registered'.
            color_neon_purple (str): Hex color code for weekends.
        """
        # Select the appropriate column based on the user type
        y_column = f'avg_{user_type}'
        title = f"Average Hourly Bike Usage: Weekdays vs. Weekends ({user_type.capitalize()} Users)"
        y_label = f"Average {user_type.capitalize()} Users"
    
        # Create the plot
        fig = px.line(
            hourly_usage,
            x='hr',
            y=y_column,
            color='is_weekend',
            title=title,
            labels={'hr': 'Hour of Day', y_column: y_label, 'is_weekend': 'Day Type'}
        )
    
        # Update weekend color to neon purple
        fig.for_each_trace(lambda trace: trace.update(line=dict(color=color_neon_purple)) if trace.name == 'Weekend' else None)
    
        # Update layout
        fig.update_layout(
            xaxis=dict(title="Hour of Day"),
            yaxis=dict(title=y_label),
            title_x=0.5
        )
        return fig
    
    # Plot for Registered Users
    st.subheader("Hourly Usage: Registered Users")
    fig_registered = plot_hourly_usage(hourly_usage, user_type='registered')
    st.plotly_chart(fig_registered)
    
    # Plot for Casual Users
    st.subheader("Hourly Usage: Casual Users")
    fig_casual = plot_hourly_usage(hourly_usage, user_type='casual')
    st.plotly_chart(fig_casual)
    
    # Clean up
    data_raw.drop(columns=['is_weekend'], inplace=True)
    
    # Average Users: Holiday vs Non-Holiday
    st.header("Average Users: Holiday vs Non-Holiday")
    
    # Calculate the mean for registered and casual users grouped by holiday
    holiday_mean = data_raw.groupby('holiday', as_index=False).agg(
        avg_registered=('registered', 'mean'),
        avg_casual=('casual', 'mean')
    ).round()
    
    # Replace holiday indicator numbers with labels for better readability
    holiday_labels = {1: 'Yes', 0: 'No'}
    holiday_mean['holiday'] = holiday_mean['holiday'].map(holiday_labels)
    
    ### Chart 1: Average Registered Users
    st.subheader("Average Registered Users: Holiday vs Non-Holiday")
    fig_registered = px.bar(
        holiday_mean,
        x='holiday',
        y='avg_registered',
        title='Average Registered Users: Holiday vs Non-Holiday',
        labels={'holiday': 'Holiday', 'avg_registered': 'Average Registered Users'},
        text='avg_registered'
    )
    fig_registered.update_traces(textposition='outside')
    fig_registered.update_layout(
        xaxis=dict(title='Holiday'),
        yaxis=dict(title='Average Registered Users'),
        showlegend=False
    )
    # Display the chart in Streamlit
    st.plotly_chart(fig_registered)
    
    ### Chart 2: Average Casual Users
    st.subheader("Average Casual Users: Holiday vs Non-Holiday")
    fig_casual = px.bar(
        holiday_mean,
        x='holiday',
        y='avg_casual',
        title='Average Casual Users: Holiday vs Non-Holiday',
        labels={'holiday': 'Holiday', 'avg_casual': 'Average Casual Users'},
        text='avg_casual'
    )
    fig_casual.update_traces(textposition='outside')
    fig_casual.update_layout(
        xaxis=dict(title='Holiday'),
        yaxis=dict(title='Average Casual Users'),
        showlegend=False
    )
    # Display the chart in Streamlit
    st.plotly_chart(fig_casual)

# ====================================
# Tab 3: Business Insights
# ====================================
with tab3:
    # Section: Revenues
    st.header("Revenues")
    
    # Subsection: Registered Users
    st.subheader("Registered Users: Focus on Volume")
    st.markdown("""
    #### Why are registered users price-elastic?
    - Registered users are **price-sensitive** and more informed about available alternatives.
    - They are regular users who:
        - Have more time to evaluate prices.
        - Plan their purchases.
    
    #### **Volume Drivers**:
    - **Frequency**: Encourage frequent usage by offering 20 free minutes for sign-up, with additional rewards for extended usage.
    - **Conversion**: Convert casual users to registered users by rewarding them with free minutes for referrals.
    - **Minutes Per Ride**: 
        - Encourage frequent longer trips to drive overall revenues.
        - Reduce prices per minute after surpassing a specific ride duration to incentivize extended usage.
    """)
    
    # Subsection: Casual Users
    st.subheader("Casual Users: Focus on Price")
    st.markdown("""
    #### Why are casual users price inelastic?
    - Casual users are **not price-sensitive** and less informed about pricing.
    - They are not fully aware of available substitutes, and bike use constitutes only a small portion of their overall trip.
    
    #### **Price Drivers**:
    - **Demand**: Adjust prices based on demand and time of use.
        - **Dynamic Pricing**: Higher prices during peak times like weekends, holidays, and good weather.
        - Increased pricing during peak hours (10 AM–6 PM).
    
    #### So how can we push usage despite a dynamic pricing strategy?
    #### **Volume Drivers**:
    - **Partnerships**:
        - Collaborate with hotels, tourism agencies, and local attractions.
    - **Marketing**:
        - Use an omnichannel approach (app promotions, billboards, local ads).
    - **Infrastructure**:
        - Expand stations and slots in high-demand areas for casual users.
    """)
    
    # Section: Costs
    st.header("Costs: Reduce Variable Costs - Maintenance & Transport")
    st.markdown("""
    #### **Winter/Bad Weather**:
    - Reduce bike availability by **40%** to lower maintenance and transport costs.
    - Store bikes during low-demand periods.
    
    #### **Relocation and Redistribution**:
    - Establish **long-term contracts** with trucks for efficient redistribution.
    - Implement **centralized storage solutions** to minimize costs.
    """)

# ====================================
# Tab 4: Feature Engineering
# ====================================
with tab4:
    # Section 3: Feature Handling and Engineering
    st.header("3. Feature Engineering")
    st.write("Exploring new features and preprocessing techniques to capture nuanced patterns in the data.")
    
    # Dynamic correlation analysis
    st.subheader("Customize Correlation Analysis")
    columns_to_correlate = st.multiselect("Select columns for correlation analysis", data_raw.columns.tolist(), default=['temp', 'hum', 'windspeed', 'cnt'])
    correlation_matrix = data_raw[columns_to_correlate].corr()
    st.write("Correlation Matrix:")
    st.dataframe(correlation_matrix)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
    st.pyplot(fig)
    
    # Correlation Analysis with 'cnt'
    st.subheader("Correlation Analysis with Target ('cnt')")
    columns_to_exclude = ['dteday']
    
    # Compute the correlation matrix excluding specified columns
    numeric_data = data_raw.drop(columns=columns_to_exclude).select_dtypes(include=['number'])
    correlation_matrix = numeric_data.corr().abs()
    
    # Focus on correlations with 'cnt'
    cnt_correlations = correlation_matrix[['cnt']].sort_values(by='cnt', ascending=False)
    
    # Display correlation table
    st.write("Correlation with 'cnt':")
    st.write(cnt_correlations)
    
    # Visualize correlations with a heatmap
    st.subheader("Heatmap of Correlations with 'cnt'")
    fig, ax = plt.subplots(figsize=(5, 10))
    sns.heatmap(
        cnt_correlations,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        cbar=True,
        ax=ax
    )
    plt.title('Correlations with cnt', fontsize=16)
    st.pyplot(fig)
    
    st.subheader("Feature Handling: Addressing Multicollinearity")
    st.write("""
    During preprocessing, we noticed high correlation between `temp` and `atemp`. 
    To reduce multicollinearity, we dropped `atemp`. Additionally, `casual` and `registered` were dropped as they are components of the target (`cnt`).
    """)
    
    st.write("Removed columns: `instant`, `dteday`, `mnth`, `atemp`, `casual`, and `registered` to avoid redundancy and multicollinearity.")
    
    st.subheader("Feature Engineering")
    
    # Adding Rush Hour Feature
    st.subheader("Rush Hour Feature")
    st.write("""
    We created a binary 'rush_hour' feature to mark peak commuting hours (7-9 AM, 5-7 PM) with 1, and all other hours with 0. 
    This helps capture the impact of high-demand times on bike usage.
    """)
    def add_rush_hour(df):
        df['rush_hour'] = df['hr'].apply(lambda x: 1 if x in [7, 8, 9, 17, 18, 19] else 0)
        return df
    
    data_raw = add_rush_hour(data_raw)
    
    # Visualizing the rush hour impact
    hourly_usage = data_raw.groupby('hr').agg(avg_cnt=('cnt', 'mean')).reset_index()
    fig = px.line(hourly_usage, x='hr', y='avg_cnt', title="Average Hourly Bike Usage",
                  labels={'hr': 'Hour of Day', 'avg_cnt': 'Average Bike Usage'})
    fig.update_layout(xaxis_title="Hour of Day", yaxis_title="Average Bike Usage", title_x=0.5)
    st.plotly_chart(fig)
    
    # Adding Cyclic Features for Hour
    st.subheader("Cyclic Features")
    st.write("""
    Since hours are cyclic (e.g., 23:00 is adjacent to 00:00), we transformed the hour of the day into **sine** and **cosine** components. 
    This allows models to better capture the cyclical nature of time. We also applied this to season, month, and weekday.
    """)
    data_raw = add_cyclic_features(data_raw, 'hr', 24)
    
    # Adding Interaction Features
    st.subheader("Interaction Features")
    st.write("""
    Interaction features combine the effects of two variables. 
    For example, we added 'temp_hum' (temperature and humidity) and 'temp_windspeed' (temperature and windspeed) 
    to capture their combined impact on bike usage. Higher temperatures with humidity can increase discomfort, while wind can have a cooling effect.
    """)
    data_raw['temp_hum'] = data_raw['temp'] + data_raw['hum']
    data_raw['temp_windspeed'] = data_raw['temp'] + data_raw['windspeed']
    
    st.write("Added new features: `temp_hum`, `temp_windspeed`.")
    
    # Correlation Analysis for New Features with 'cnt'
    st.subheader("Correlation Analysis: New Features vs Target ('cnt')")
    
    # List of new features
    new_features = ['rush_hour', 'hr_sin', 'hr_cos', 'temp_hum', 'temp_windspeed']
    
    # Select only new features and the target
    new_features_data = data_raw[new_features + ['cnt']]
    
    # Compute correlation matrix
    new_features_corr = new_features_data.corr().abs()
    
    # Focus on correlations with 'cnt'
    new_features_corr_cnt = new_features_corr[['cnt']].sort_values(by='cnt', ascending=False)
    
    # Display correlation table
    st.write("Correlation of New Features with 'cnt':")
    st.write(new_features_corr_cnt)
    
    # Visualize correlations with a heatmap
    st.subheader("Heatmap of New Features Correlation with 'cnt'")
    fig, ax = plt.subplots(figsize=(5, 6))
    sns.heatmap(
        new_features_corr_cnt,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        cbar=True,
        ax=ax
    )
    plt.title('New Features Correlations with cnt', fontsize=16)
    st.pyplot(fig)

# ====================================
# Tab 5: ML Enthusiasts
# ====================================
with tab5:
    st.header("Hyperparameter Tuning & Model Training")
    model_name = st.selectbox("Choose Model", list(models.keys()))
    model = models[model_name]
    
    # Hyperparameter inputs
    if model_name == "Random Forest":
        n_estimators = st.slider("Number of Trees", 10, 200, 100, step=10)
        max_depth = st.slider("Max Depth", 1, 20, 10)
        model.set_params(n_estimators=n_estimators, max_depth=max_depth)
    
    elif model_name == "XGBoost":
        learning_rate = st.slider("Learning Rate", 0.01, 0.3, 0.1, step=0.01)
        n_estimators = st.slider("Number of Estimators", 10, 200, 100, step=10)
        max_depth = st.slider("Max Depth", 1, 20, 6)
        model.set_params(learning_rate=learning_rate, n_estimators=n_estimators, max_depth=max_depth)
    
    elif model_name == "LightGBM":
        num_leaves = st.slider("Number of Leaves", 10, 100, 31, step=5)
        max_depth = st.slider("Max Depth", -1, 20, -1)
        model.set_params(num_leaves=num_leaves, max_depth=max_depth)
    
    if st.button("Train Model"):
        with st.spinner('Training in progress...'):
            sleep(1)
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_val_pred = model.predict(X_val)
            y_test_pred = model.predict(X_test)
    
            st.write(f"Training MAE: {mean_absolute_error(y_train, y_train_pred):.2f}")
            st.write(f"Training MAPE: {mean_absolute_percentage_error(y_train, y_train_pred):.4f}")
            st.write(f"Validation MAE: {mean_absolute_error(y_val, y_val_pred):.2f}")
            st.write(f"Validation MAPE: {mean_absolute_percentage_error(y_val, y_val_pred):.4f}")
            st.write(f"Test MAE: {mean_absolute_error(y_test, y_test_pred):.2f}")
            st.write(f"Test MAPE: {mean_absolute_percentage_error(y_test, y_test_pred):.4f}")

# ====================================
# Tab 6: Business Users
# ====================================
with tab6:
    st.header("Predict Bike Demand")
    date = st.date_input("Select Date")
    time = st.slider("Select Hour of the Day", 0, 23, 12)
    season = st.selectbox("Season", ["Winter", "Spring", "Summer", "Fall"])
    day_type = st.radio("Day Type", ["Holiday", "Working Day"])
    holiday = 1 if day_type == "Holiday" else 0
    working_day = 1 if day_type == "Working Day" else 0
    weather = st.selectbox("Weather Condition", ["Clear/Partly Cloudy", "Mist/Cloudy", "Light Snow/Rain", "Heavy Rain/Snow"])
    temp = st.slider("Temperature (Normalized)", 0.0, 1.0, 0.5)
    hum = st.slider("Humidity (Normalized)", 0.0, 1.0, 0.5)
    windspeed = st.slider("Wind Speed (Normalized)", 0.0, 1.0, 0.5)
    
    if st.button("Get Prediction"):
        # Prepare features for prediction
        date = pd.to_datetime(date)
        yr = 1 if date.year == 2012 else 0
        mnth = date.month
        weekday = date.weekday()
    
        season_mapping = {"Winter": 1, "Spring": 2, "Summer": 3, "Fall": 4}
        weather_mapping = {
            "Clear/Partly Cloudy": 1,
            "Mist/Cloudy": 2,
            "Light Snow/Rain": 3,
            "Heavy Rain/Snow": 4
        }
    
        features = pd.DataFrame({
            'yr': [yr],
            'mnth': [mnth],
            'hr': [time],
            'holiday': [holiday],
            'weekday': [weekday],
            'workingday': [working_day],
            'weathersit': [weather_mapping[weather]],
            'temp': [temp],
            'hum': [hum],
            'windspeed': [windspeed],
            'season': [season_mapping[season]],
            'dteday': [date]
        })
    
        features = load_and_prepare_data(features)
        features_aligned = align_features(features, X_train)
    
        best_model = models["Random Forest"]
        best_model.fit(X_train, y_train)
        prediction = best_model.predict(features_aligned)
        st.write(f"Predicted Bike Demand: {prediction[0]:.0f} bikes")

# ====================================
# Tab 7: Model Comparison
# ====================================
with tab7:
    st.header("Compare Model Performance")
    metrics = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_val_pred = model.predict(X_val)
        y_test_pred = model.predict(X_test)
        mae_val = mean_absolute_error(y_val, y_val_pred)
        mae_test = mean_absolute_error(y_test, y_test_pred)
        mape_val = mean_absolute_percentage_error(y_val, y_val_pred)
        mape_test = mean_absolute_percentage_error(y_test, y_test_pred)
        metrics.append({
            "Model": name, 
            "Validation MAE": mae_val, 
            "Test MAE": mae_test, 
            "Validation MAPE": mape_val, 
            "Test MAPE": mape_test
        })
    
    metrics_df = pd.DataFrame(metrics)
    st.dataframe(metrics_df)
    
    fig = px.bar(metrics_df, x="Model", y=["Validation MAE", "Test MAE"], barmode="group",
                 title="Model Comparison by MAE", labels={"value": "MAE", "variable": "Dataset"})
    st.plotly_chart(fig, key="model_comparison_chart")
    
    fig_mape = px.bar(metrics_df, x="Model", y=["Validation MAPE", "Test MAPE"], barmode="group",
                 title="Model Comparison by MAPE", labels={"value": "MAPE", "variable": "Dataset"})
    st.plotly_chart(fig_mape, key="model_comparison_chart_mape")

# ====================================
# Tab 8: Real vs Predicted
# ====================================
with tab8:
    st.header("Real vs Predicted Values")
    best_model = models["Random Forest"]
    best_model.fit(X_train, y_train)
    y_test_pred = best_model.predict(X_test)
    
    comparison_df = pd.DataFrame({
        'Date': data.loc[y_test.index, 'dteday'],
        'Real Values': y_test,
        'Predicted Values': y_test_pred
    })
    
    comparison_df['Date'] = pd.to_datetime(comparison_df['Date'])
    comparison_df = comparison_df.groupby('Date').mean().reset_index()
    
    comparison_fig = px.line(
        comparison_df,
        x='Date',
        y=['Real Values', 'Predicted Values'],
        title="Comparison of Real vs Predicted Bike Demand",
        labels={'value': 'Bike Demand', 'variable': 'Legend'}
    )
    st.plotly_chart(comparison_fig, key="real_vs_predicted")
    
    st.subheader("Select Date Range for Visualization")
    min_date = comparison_df['Date'].min().date()
    max_date = comparison_df['Date'].max().date()
    start_date = st.date_input("Start Date", min_value=min_date, max_value=max_date, value=min_date)
    end_date = st.date_input("End Date", min_value=min_date, max_value=max_date, value=max_date)
    
    mask = (comparison_df['Date'] >= pd.to_datetime(start_date)) & (comparison_df['Date'] <= pd.to_datetime(end_date))
    filtered_df = comparison_df.loc[mask]
    
    filtered_fig = px.line(
        filtered_df,
        x='Date',
        y=['Real Values', 'Predicted Values'],
        title="Filtered Comparison of Real vs Predicted Bike Demand",
        labels={'value': 'Bike Demand', 'variable': 'Legend'}
    )
    st.plotly_chart(filtered_fig, key="filtered_real_vs_predicted")
