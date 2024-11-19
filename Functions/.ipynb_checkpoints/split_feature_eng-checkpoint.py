import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from scipy.stats import zscore
import lightgbm as lgb
from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

def split_data(df, target_column='cnt', test_year=1, test_month_start=11, val_ratio=0.2):
    """
    Splits the dataset into training, validation, and testing sets.

    Parameters:
    - df (DataFrame): The input DataFrame.
    - target_column (str): The name of the target column.
    - test_year (int): The year to use for testing data.
    - test_month_start (int): The starting month for testing data.
    - val_ratio (float): The ratio of validation data to training data.

    Returns:
    - X_train_final, X_val, X_test, y_train_final, y_val, y_test
    """
    y = df[target_column]
    X = df.drop(columns=[target_column])

    df_test = df[(df['yr'] == test_year) & (df['mnth'] >= test_month_start)]
    df_train = df.drop(df_test.index)
    unnecessary_columns = ['instant', 'dteday', 'casual', 'registered', 'atemp']
    X_train = df_train.drop(columns=unnecessary_columns, errors='ignore') 
    y_train = df_train[target_column]

    X_test = df_test.drop(columns=unnecessary_columns, errors='ignore')
    y_test = df_test[target_column]

    train_size = int(len(X_train) * (1 - val_ratio))
    X_train_final = X_train.iloc[:train_size]
    X_val = X_train.iloc[train_size:]
    y_train_final = y_train.iloc[:train_size]
    y_val = y_train.iloc[train_size:]

    if target_column in X_train_final.columns:
        print(f"Warning: {target_column} found in X_train_final. Removing it.")
        X_train_final = X_train_final.drop(columns=[target_column])
    if target_column in X_val.columns:
        print(f"Warning: {target_column} found in X_val. Removing it.")
        X_val = X_val.drop(columns=[target_column])
    if target_column in X_test.columns:
        print(f"Warning: {target_column} found in X_test. Removing it.")
        X_test = X_test.drop(columns=[target_column])

    # Print the sizes of the splits
    print(f"Final Training set: {X_train_final.shape}, Validation set: {X_val.shape}, Testing set: {X_test.shape}")
    print(f"Final Training set: {y_train_final.shape}, Validation set: {y_val.shape}, Testing set: {y_test.shape}")

    return X_train_final, X_val, X_test, y_train_final, y_val, y_test

def add_daylight_period_simple(train, val, test, new_column='daylight_period'):
    """
    Add a new column categorizing the hour of the day into four periods:
    Morning, Afternoon, Evening, Night.

    Args:
        train (pd.DataFrame): Training DataFrame.
        val (pd.DataFrame): Validation DataFrame.
        test (pd.DataFrame): Test DataFrame.
        hour_column (str): Column name for the hour.
        new_column (str): Name of the new column to add.

    Returns:
        tuple: Updated (train, val, test) DataFrames.
    """
    def categorize_daylight(hour):
        if 5 <= hour < 12:  # Morning
            return 1
        elif 12 <= hour < 17:  # Afternoon
            return 2
        elif 17 <= hour < 21:  # Evening
            return 3
        else:  # Night
            return 4

    train = train.copy()
    val = val.copy()
    test = test.copy()

    train[new_column] = train['hr'].apply(categorize_daylight)
    val[new_column] = val['hr'].apply(categorize_daylight)
    test[new_column] = test['hr'].apply(categorize_daylight)

    return train, val, test

def plot_hourly_bike_usage(data):
    """
    Plots the average hourly bike usage.

    Parameters:
    - data (DataFrame): The dataset containing the 'hr' and 'cnt' columns.

    Returns:
    - None (Displays the plot).
    """
    hourly_usage = data.groupby('hr', observed=False).agg(avg_cnt=('cnt', 'mean')).reset_index()

    fig = px.line(
        hourly_usage,
        x='hr',
        y='avg_cnt',
        title="Average Hourly Bike Usage",
        labels={'hr': 'Hour of Day', 'avg_cnt': 'Average Bike Usage'}
    )
    fig.update_layout(
        xaxis_title="Hour of Day",
        yaxis_title="Average Bike Usage",
        title_x=0.5
    )
    fig.show()

def add_rush_hour(X):
    """
    Adds a 'rush_hour' feature to the dataset, marking peak commuting hours with 1 and other hours with 0.

    Parameters:
    - X (DataFrame): The input dataset containing the 'hr' column.

    Returns:
    - X (DataFrame): The dataset with the added 'rush_hour' column.
    """
    X = X.copy()

    X['rush_hour'] = X['hr'].apply(lambda x: 1 if x in [7, 8, 9, 16, 17, 18] else 0)

    return X

def compute_correlations(data, columns, target_column):
    """
    Computes the correlation between multiple columns and a target column.

    Parameters:
    - data (DataFrame): The dataset containing the features and target.
    - columns (list): List of column names for which to compute correlations.
    - target_column (str): The target column name to compute correlations with.

    Returns:
    - correlations (dict): A dictionary with column names as keys and correlation values as values.
    """
    correlations = {}
    
    for column in columns:
        if column in data.columns:
            correlations[column] = data[column].corr(data[target_column])
        else:
            print(f"Column '{column}' not found in the dataset.")
    
    return correlations

def add_cyclic_features(df, col_name, max_value):
    """
    Adds sine and cosine cyclic features for a given column and drops the original column.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing the column to be transformed.
    col_name (str): The name of the column to be transformed.
    max_value (int): The maximum value of the column (e.g., 24 for hours in a day).
    
    Returns:
    pd.DataFrame: The DataFrame with the added cyclic features and the original column dropped.
    """
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

def calculate_skewness(df, skew_threshold = 0.5):
    """
    Calculate skewness of numerical columns and show only those
    that exceed the specified skewness threshold.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    skew_threshold (float): The threshold above which skewness is considered high (default is 1.0).
    """
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        skewness = df[col].skew()
        if skewness > skew_threshold:
            print(f'Column: {col}, Skewness: {skewness:.4f}')

def apply_transformation_and_check_correlation(df, column, target):

    df[f'{column}_sqrt'] = np.sqrt(df[column])
    df[f'{column}_log'] = np.log1p(df[column])

    original_corr = df[column].corr(df[target])
    sqrt_corr = df[f'{column}_sqrt'].corr(df[target])
    log_corr = df[f'{column}_log'].corr(df[target])

    print(f'Correlations for {column}:')
    print(f'Original Correlation: {original_corr:.4f}')
    print(f'Sqrt Transformation Correlation: {sqrt_corr:.4f}')
    print(f'Log Transformation Correlation: {log_corr:.4f}')

def apply_log(dataframes, column_name):
    """
    Applies a log transformation to the specified column in multiple DataFrames and drops the original column.

    Parameters:
    dataframes (list of pd.DataFrame): The list of DataFrames to modify.
    column_name (str): The name of the column to transform.

    Returns:
    list of pd.DataFrame: The modified DataFrames with the log-transformed column.
    """
    log_column_name = f'{column_name}_log'
    modified_dataframes = []

    for df in dataframes:
        df_copy = df.copy()  
        df_copy[log_column_name] = np.log1p(df_copy[column_name])  
        df_copy.drop(columns=[column_name], inplace=True)  
        modified_dataframes.append(df_copy)

    return modified_dataframes

def one_hot_encode_column(df_train, df_val, df_test, column_name):

    encoder = OneHotEncoder(drop='first', sparse_output=False)
    
    encoder.fit(df_train[[column_name]])
    
    train_encoded_values = encoder.transform(df_train[[column_name]])
    val_encoded_values = encoder.transform(df_val[[column_name]])
    test_encoded_values = encoder.transform(df_test[[column_name]])
    
    encoded_column_names = [f"{column_name}_{category}" for category in encoder.categories_[0][1:]]
    
    train_encoded_df = pd.DataFrame(train_encoded_values, columns=encoded_column_names, index=df_train.index)
    val_encoded_df = pd.DataFrame(val_encoded_values, columns=encoded_column_names, index=df_val.index)
    test_encoded_df = pd.DataFrame(test_encoded_values, columns=encoded_column_names, index=df_test.index)
    
    df_train_encoded = pd.concat([df_train.drop(columns=[column_name]), train_encoded_df], axis=1)
    df_val_encoded = pd.concat([df_val.drop(columns=[column_name]), val_encoded_df], axis=1)
    df_test_encoded = pd.concat([df_test.drop(columns=[column_name]), test_encoded_df], axis=1)
    
    return df_train_encoded, df_val_encoded, df_test_encoded