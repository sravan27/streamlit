# Here we load our data and do some minor exploration.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import lightgbm as lgb
from scipy.stats import zscore, uniform, randint
from plotly.subplots import make_subplots


def load_and_prepare_data(clean_print=True):
    """
    Load the dataset from the 'Dataset' folder, categorize columns, and perform initial data preparation steps.
    """    
    file_path = '../Dataset/bike-sharing-hourly.csv'
    data = pd.read_csv(file_path)
    data['dteday'] = pd.to_datetime(data['dteday'])

    categorical_ordinal = ['yr', 'mnth', 'season', 'weekday', 'hr']
    categorical_columns = ['holiday', 'workingday', 'weathersit']
    numerical_columns = ['temp', 'atemp', 'hum', 'windspeed']
    
    if clean_print:
        print("Dataset loaded successfully!")
        print(f"Shape of the dataset: {data.shape}")
        
        print("\nColumns Defined:")
        print(f"Categorical Ordinal Columns: {categorical_ordinal}")
        print(f"Categorical Columns: {categorical_columns}")
        print(f"Numerical Columns: {numerical_columns}")
        
        print("\nChecking for Missing Values:")
        missing_values = data.isna().mean()[data.isna().mean() > 0]
        if missing_values.empty:
            print("No missing values found.")
        else:
            print(missing_values)
        
        print("\nMean of All Columns (Numeric Only):")
        print(data.mean(numeric_only=True).round(2))
    else:
        print("Data Info:")
        print(data.info())
        print("\nData Description:")
        print(data.describe().round(2).T)
        print("\nMissing Values:")
        print(data.isna().mean())
    
    return data, categorical_ordinal, categorical_columns, numerical_columns

def detect_and_remove_outliers_with_plots(data, features, threshold=3):
    """
    Detects and removes outliers from the specified columns of the dataset using the Z-score method and visualizes the outliers with boxplots.

    Parameters:
    - data (DataFrame): The input DataFrame.
    - features (list): The list of columns to analyze for outliers.
    - threshold (float): The Z-score threshold to identify outliers.

    Returns:
    - data_no_outliers (DataFrame): The DataFrame with outliers removed.
    """

    data_no_outliers = data.copy()

    numerical_cols = [col for col in features if col in data.columns]

    z_scores = data[numerical_cols].apply(zscore)
    outliers_zscore = (z_scores.abs() > threshold)

    outlier_counts = outliers_zscore.sum()
    print("Outliers Detected (Z-Score):")
    print(outlier_counts)

    data_no_outliers = data_no_outliers[~outliers_zscore.any(axis=1)]

    fig = make_subplots(
        rows=len(numerical_cols),
        cols=1,
        subplot_titles=[f"Outliers in {col}" for col in numerical_cols]
    )
    for i, col in enumerate(numerical_cols, start=1):
        fig.add_trace(
            go.Box(y=data[col], name=col, boxpoints="outliers", marker_color="blue"),
            row=i, col=1
        )
    fig.update_layout(
        height=300 * len(numerical_cols),
        width=800,
        title_text="Boxplots of Specified Numerical Columns with Outliers",
        title_x=0.5,
        showlegend=False
    )

    fig.show()

    num_removed = data.shape[0] - data_no_outliers.shape[0]
    print(f"\nNumber of outliers removed: {num_removed}")

    return data_no_outliers

def plot_casual_and_registered_users(data):
    """
    Creates a combined plot with daily average casual users and registered users
    displayed one above the other.

    Parameters:
    - data (DataFrame): The input DataFrame containing 'dteday', 'casual', and 'registered' columns.
    """

    daily_casual_data = data.groupby('dteday').agg(
        daily_casual=('casual', 'mean')
    ).reset_index()
    
    daily_registered_data = data.groupby('dteday').agg(
        daily_registered=('registered', 'mean')
    ).reset_index()

    fig = make_subplots(
        rows=2, cols=1, 
        subplot_titles=["Daily Average Casual User Counts", "Daily Average Registered User Counts"]
    )

    fig.add_trace(
        go.Scatter(
            x=daily_casual_data['dteday'],
            y=daily_casual_data['daily_casual'],
            mode='lines',
            name='Casual Users',
            line=dict(color='blue')
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=daily_registered_data['dteday'],
            y=daily_registered_data['daily_registered'],
            mode='lines',
            name='Registered Users',
            line=dict(color='red')
        ),
        row=2, col=1
    )


    fig.update_layout(
        height=600,  
        width=800,
        title_text="Daily Average User Counts",
        showlegend=True
    )

    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_yaxes(title_text="Average Number of Casual Users", row=1, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Average Number of Registered Users", row=2, col=1)

    fig.show()

def plot_cat_dist(df, columns):
    """
    Plots the distributions of multiple categorical columns together in a multi-column, multi-row subplot layout.
    
    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data.
    - columns (list of str): The columns to plot.
    """
    num_columns = len(columns)
    cols_per_row = 2
    num_rows = (num_columns + 1) // cols_per_row

    fig = make_subplots(rows=num_rows, cols=cols_per_row, 
                        subplot_titles=[f'Distribution of {col.capitalize()}' for col in columns],
                        horizontal_spacing=0.1, vertical_spacing=0.15)

    for i, column in enumerate(columns):
        row = (i // cols_per_row) + 1
        col = (i % cols_per_row) + 1

        count_data = df[column].value_counts().sort_index().reset_index()
        count_data.columns = [column, 'count']

        bar_trace = go.Bar(
            x=count_data[column],
            y=count_data['count'],
            marker=dict(color='#1f77b4'),
            text=count_data['count'],
            textposition='outside' 
        )

        fig.add_trace(bar_trace, row=row, col=col)
        fig.update_xaxes(title_text=column.capitalize(), tickangle=0, tickmode='linear', row=row, col=col)
        fig.update_yaxes(title_text='Count', row=row, col=col)
        
        max_count = count_data['count'].max()
        fig.update_yaxes(range=[0, max_count * 1.15], row=row, col=col)  

    fig.update_layout(
        bargap=0.6,  
        height=300 * num_rows,
        width=1200,
        title_text="Distributions of Categorical Features",
        title_x=0.5,
        showlegend=False
    )

    fig.show()

def plot_num_dist(df, columns):
    """
    Plots the distributions of multiple numerical columns together in a multi-column, multi-row subplot layout.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    columns (list of str): The numerical columns to plot.
    """
    num_columns = len(columns)
    cols_per_row = 2
    num_rows = (num_columns + 1) // cols_per_row

    fig = make_subplots(rows=num_rows, cols=cols_per_row, 
                        subplot_titles=[f'Distribution of {col.capitalize()}' for col in columns],
                        horizontal_spacing=0.1, vertical_spacing=0.15)

    for i, column in enumerate(columns):
        row = (i // cols_per_row) + 1
        col = (i % cols_per_row) + 1

        hist_trace = go.Histogram(
            x=df[column],
            marker=dict(color='#1f77b4'),
            nbinsx=20, 
        )

        fig.add_trace(hist_trace, row=row, col=col)
        fig.update_xaxes(title_text=column.capitalize(), tickangle=0, row=row, col=col)
        fig.update_yaxes(title_text='Count', row=row, col=col)

    fig.update_layout(
        bargap=0.1, 
        height=300 * num_rows,
        width=1200,
        title_text="Distributions of Numerical Features",
        title_x=0.5,
        showlegend=False
    )
    fig.show()

def plot_monthly_average_usage(df):

    """
Plots monthly average bike usage.

Calculates the average bike usage (`cnt`) for each month-year combination, generates a line chart with markers, and displays the results.
    """
    if df['dteday'].dtype != 'datetime64[ns]':
        df['dteday'] = pd.to_datetime(df['dteday'])
    
    df['month_year'] = df['yr'].astype(str) + '-' + df['mnth'].astype(str)

    monthly_usage = df.groupby('month_year', observed=True).agg(avg_cnt=('cnt', 'mean')).reset_index()
    monthly_usage['avg_cnt_rounded'] = monthly_usage['avg_cnt'].round()

    fig = go.Figure(
        data=go.Scatter(
            x=monthly_usage['month_year'],
            y=monthly_usage['avg_cnt'],
            mode='lines+markers+text',
            text=monthly_usage['avg_cnt_rounded'],
            textposition='top center',
            marker=dict(color='#1f77b4'),
            line=dict(width=3)
        )
    )

    fig.update_layout(
        title="Monthly Average Bike Usage (2011-2012)",
        xaxis_title="Month-Year",
        yaxis_title="Average Bike Usage",
        title_x=0.5,
        width=800,
        height=500,
        showlegend=False
    )

    df.drop(columns=['month_year'], inplace=True)

    fig.show()

def plot_average_hourly_usage(df):

    """
Plots monthly average bike usage.

Calculates and visualizes the average bike usage (`cnt`) for each month-year combination as a line chart with markers.
    """

    df['is_weekend'] = df['weekday'].apply(lambda x: 'Weekend' if x >= 5 else 'Weekday')

    hourly_usage = df.groupby(['is_weekend', 'hr'], observed=False).agg(avg_cnt=('cnt', 'mean')).reset_index()
    
    fig = px.line(
        hourly_usage,
        x='hr', y='avg_cnt', color='is_weekend',
        title="Average Hourly Bike Usage: Weekdays vs. Weekends",
        labels={'hr': 'Hour of Day', 'avg_cnt': 'Average Bike Usage', 'is_weekend': 'Day Type'}
    )

    fig.for_each_trace(lambda trace: trace.update(line=dict(color='#DA70D6')) if trace.name == 'Weekend' else None)

    fig.update_layout(
        xaxis_title="Hour of Day",
        yaxis_title="Average Bike Usage",
        title_x=0.5
    )

    df.drop(columns=['is_weekend'], inplace=True)

    fig.show()

def plot_hourly_usage(data, user_type, color_neon_purple='#DA70D6'):
    """
    Plot the average hourly bike usage for casual or registered users.

    Args:
        data (pd.DataFrame): The original bike-sharing data.
        user_type (str): Either 'casual' or 'registered'.
        color_neon_purple (str): Hex color code for weekends.
    """
    data['day_type'] = data['weekday'].apply(lambda x: 'Weekend' if x in [0, 6] else 'Weekday')

    hourly_usage = (
        data.groupby(['hr', 'day_type'])[[user_type]]
        .mean()
        .reset_index()
        .rename(columns={user_type: f'avg_{user_type}'})
    )
    y_column = f'avg_{user_type}'
    title = f"Average Hourly Bike Usage: Weekdays vs. Weekends ({user_type.capitalize()} Users)"
    y_label = f"Average {user_type.capitalize()} Users"

    fig = px.line(
        hourly_usage,
        x='hr',
        y=y_column,
        color='day_type',
        title=title,
        labels={'hr': 'Hour of Day', y_column: y_label, 'day_type': 'Day Type'}
    )

    fig.for_each_trace(lambda trace: trace.update(line=dict(color=color_neon_purple)) if trace.name == 'Weekend' else None)
    fig.update_layout(
        xaxis=dict(title="Hour of Day"),
        yaxis=dict(title=y_label),
        title_x=0.5
    )

    data.drop(columns=['day_type'], inplace=True)

    fig.show()

def plot_average_usage_by_weather(df):

    """
Plots average bike usage by weather condition.

This function calculates the average bike usage (`cnt`) for each weather condition, maps numeric weather values to descriptive labels (e.g., Clear/Partly Cloudy, Mist/Cloudy), and creates a bar chart to visualize the results.

    """

    weather_usage = df.groupby('weathersit', observed=True).agg(avg_cnt=('cnt', 'mean')).reset_index()
    weather_labels = {
        1: "Clear/Partly Cloudy",
        2: "Mist/Cloudy",
        3: "Light Snow/Rain",
        4: "Heavy Rain/Snow"
    }
    weather_usage['Weather'] = weather_usage['weathersit'].map(weather_labels)
    weather_usage['avg_cnt_rounded'] = weather_usage['avg_cnt'].round()

    fig = go.Figure(
        data=go.Bar(
            x=weather_usage['Weather'],
            y=weather_usage['avg_cnt'],
            marker=dict(color='#1f77b4'),
            text=weather_usage['avg_cnt_rounded'], 
            textposition='outside'
        )
    )

    fig.update_layout(
        title="Average Bike Usage by Weather Condition",
        xaxis_title="Weather Condition",
        yaxis_title="Average Bike Usage",
        title_x=0.5,
        bargap=0.4,
        width=800,
        height=500,
        showlegend=False
    )
    
    fig.show()

def plot_average_usage_by_season(df):

    """
Plots average bike usage by season.

This function calculates the average bike usage (`cnt`) for each season, maps numeric season values to labels (e.g., Winter, Spring), and creates a bar chart to visualize the results.

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

    fig.show()

def plot_correlation_matrix(df, columns_to_exclude=None):
    """
    Plots a heatmap of the correlation matrix, excluding specified columns.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    columns_to_exclude (list of str): Columns to exclude from the correlation matrix.
    """
   
    if columns_to_exclude:
        df_to_corr = df.drop(columns=columns_to_exclude)
    else:
        df_to_corr = df

    
    numeric_df = df_to_corr.select_dtypes(include=['number'])

    correlation_matrix = numeric_df.corr().abs()

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        correlation_matrix,
        annot=True,      
        fmt='.2f',       
        cmap='coolwarm', 
        cbar=True,
        square=True,
        linewidths=0.5
    )
    plt.title('Correlation Matrix (Excluding Specified Columns)', fontsize=16)
    plt.show()

def plot_correlations(data, target_column, columns_to_exclude=None, figsize=(5, 10)):
    """
    Compute and visualize correlations of numeric features with a target column.

    Parameters:
    - data (DataFrame): The dataset containing the features.
    - target_column (str): The column name to calculate correlations with.
    - columns_to_exclude (list, optional): List of columns to exclude from the analysis.
    - figsize (tuple, optional): Size of the heatmap figure.

    Returns:
    - cnt_correlations (DataFrame): A sorted DataFrame of correlations with the target column.
    """
    if columns_to_exclude is None:
        columns_to_exclude = []
    
    numeric_data = data.drop(columns=columns_to_exclude).select_dtypes(include=['number'])
    
    correlation_matrix = numeric_data.corr().abs()
    cnt_correlations = correlation_matrix[[target_column]].sort_values(by=target_column, ascending=False)
    
    plt.figure(figsize=figsize)
    sns.heatmap(
        cnt_correlations,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        cbar=True
    )
    plt.title(f'Correlations with {target_column.capitalize()}', fontsize=16)
    plt.show()
    
    return cnt_correlations