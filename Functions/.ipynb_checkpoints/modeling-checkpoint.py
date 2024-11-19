# Modeling

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import lightgbm as lgb
from scipy.stats import zscore, uniform, randint
from plotly.subplots import make_subplots
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, make_scorer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from joblib import dump

def train_evaluate_model(model, model_name, X_train, y_train, X_val, y_val):
    """
    Trains and evaluates a given model, printing metrics for training and validation datasets.

    Parameters:
    - model: The machine learning model (e.g., RandomForestRegressor, XGBRegressor).
    - model_name (str): Name of the model for identification.
    - X_train (DataFrame): Training features.
    - y_train (Series): Training target.
    - X_val (DataFrame): Validation features.
    - y_val (Series): Validation target.

    Returns:
    - metrics (dict): A dictionary containing training and validation metrics.
    """

    print(f"Training and evaluating: {model_name}")
    
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    
    mae_train = mean_absolute_error(y_train, y_train_pred)
    mape_train = mean_absolute_percentage_error(y_train, y_train_pred)
    mae_val = mean_absolute_error(y_val, y_val_pred)
    mape_val = mean_absolute_percentage_error(y_val, y_val_pred)
    
    print(f"Training MAE: {mae_train:.2f}")
    print(f"Training MAPE: {mape_train:.4f}")
    print(f"Validation MAE: {mae_val:.2f}")
    print(f"Validation MAPE: {mape_val:.4f}")
    print("\n" + "-"*50 + "\n") 
    
    metrics = {
        'model_name': model_name,
        'mae_train': mae_train,
        'mape_train': mape_train,
        'mae_val': mae_val,
        'mape_val': mape_val
    }
    return metrics

def train_evaluate_svm(X_train, y_train, X_val, y_val, kernel='rbf', C=1.0, epsilon=0.1):
    """
    Trains and evaluates an SVM model with scaling applied to the features.

    Parameters:
    - X_train (DataFrame): Training features.
    - y_train (Series): Training target.
    - X_val (DataFrame): Validation features.
    - y_val (Series): Validation target.
    - kernel (str): SVM kernel type ('linear', 'poly', 'rbf', 'sigmoid').
    - C (float): Regularization parameter.
    - epsilon (float): Epsilon in the epsilon-SVR model.

    Returns:
    - metrics (dict): A dictionary containing training and validation metrics.
    """
    print("Training and evaluating: Support Vector Machine (SVM)")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    svr_model = SVR(kernel=kernel, C=C, epsilon=epsilon)
    svr_model.fit(X_train_scaled, y_train)
    
    y_train_pred = svr_model.predict(X_train_scaled)
    y_val_pred = svr_model.predict(X_val_scaled)
    
    mae_train = mean_absolute_error(y_train, y_train_pred)
    mape_train = mean_absolute_percentage_error(y_train, y_train_pred)
    mae_val = mean_absolute_error(y_val, y_val_pred)
    mape_val = mean_absolute_percentage_error(y_val, y_val_pred)
    
    print(f"Training MAE: {mae_train:.2f}")
    print(f"Training MAPE: {mape_train:.4f}")
    print(f"Validation MAE: {mae_val:.2f}")
    print(f"Validation MAPE: {mape_val:.4f}")
    print("\n" + "-"*50 + "\n")
    
    metrics = {
        'mae_train': mae_train,
        'mape_train': mape_train,
        'mae_val': mae_val,
        'mape_val': mape_val
    }
    return metrics

def train_evaluate_model_with_log(model, model_name, X_train, y_train, X_val, y_val):
    """
    Trains and evaluates a given model with log-transformed target variable, 
    printing metrics for training and validation datasets.

    Parameters:
    - model: The machine learning model (e.g., RandomForestRegressor, XGBRegressor, LGBMRegressor).
    - model_name (str): Name of the model for identification.
    - X_train (DataFrame): Training features.
    - y_train (Series): Training target.
    - X_val (DataFrame): Validation features.
    - y_val (Series): Validation target.

    Returns:
    - metrics (dict): A dictionary containing training and validation metrics.
    """

    rf = RandomForestRegressor(random_state=42)
    xgb = XGBRegressor(random_state=42)
    lgb_model = lgb.LGBMRegressor(random_state=42)
    
    print(f"Training and evaluating with log transformation: {model_name}")
    
    y_train_log = np.log1p(y_train)  
    
    model.fit(X_train, y_train_log)
    
    y_train_pred_log = model.predict(X_train)
    y_val_pred_log = model.predict(X_val)
    
    y_train_pred = np.expm1(y_train_pred_log)  
    y_val_pred = np.expm1(y_val_pred_log)
    
    mae_train = mean_absolute_error(y_train, y_train_pred)
    mape_train = mean_absolute_percentage_error(y_train, y_train_pred)
    mae_val = mean_absolute_error(y_val, y_val_pred)
    mape_val = mean_absolute_percentage_error(y_val, y_val_pred)
    
    print(f"Training MAE: {mae_train:.2f}")
    print(f"Training MAPE: {mape_train:.4f}")
    print(f"Validation MAE: {mae_val:.2f}")
    print(f"Validation MAPE: {mape_val:.4f}")
    print("\n" + "-"*50 + "\n")  

    metrics = {
        'model_name': model_name,
        'mae_train': mae_train,
        'mape_train': mape_train,
        'mae_val': mae_val,
        'mape_val': mape_val
    }
    return metrics

def select_features_by_importance(model, X_train, X_val, X_test, threshold=0.991):
    """
    Selects features based on their importance scores and filters the datasets accordingly.

    Parameters:
    - model: The trained model with a `feature_importances_` attribute (e.g., Random Forest).
    - X_train (DataFrame): Training features.
    - X_val (DataFrame): Validation features.
    - X_test (DataFrame): Testing features.
    - threshold (float): Cumulative importance threshold for feature selection.

    Returns:
    - X_train_filtered, X_val_filtered, X_test_filtered (DataFrames): Filtered datasets.
    - selected_features (list): List of selected feature names.
    """
    feature_importances = model.feature_importances_
    features_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)

    features_df['Cumulative Importance'] = features_df['Importance'].cumsum()

    selected_features = features_df[features_df['Cumulative Importance'] <= threshold]['Feature'].tolist()

    X_train_filtered = X_train[selected_features]
    X_val_filtered = X_val[selected_features]
    X_test_filtered = X_test[selected_features]

    print(f"Selected Features ({len(selected_features)}): {selected_features}")
    print(f"Filtered Train Shape: {X_train_filtered.shape}")
    print(f"Filtered Validation Shape: {X_val_filtered.shape}")
    print(f"Filtered Test Shape: {X_test_filtered.shape}")
    print("")
    print(f"Unfiltered Train Shape: {X_train.shape}")
    print(f"Unfiltered Validation Shape: {X_val.shape}")
    print(f"Unfiltered Test Shape: {X_test.shape}")
    print("")

    plt.figure(figsize=(10, 6))
    plt.barh(features_df['Feature'], features_df['Importance'])
    plt.gca().invert_yaxis()
    plt.title('Feature Importances')
    plt.xlabel('Importance Score')
    plt.ylabel('Features')
    plt.show()

    return X_train_filtered, X_val_filtered, X_test_filtered, selected_features

def perform_randomized_search(X_train, y_train, param_grid, n_iter=100, scoring='neg_mean_absolute_error', cv=3, verbose=2, random_state=42, n_jobs=-1):
    """
    Performs Randomized Search for hyperparameter tuning using XGBRegressor and logs the results.
    
    Parameters:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Log-transformed target values.
        param_grid (dict): Dictionary with parameters to search.
        n_iter (int): Number of parameter settings sampled. Default is 100.
        scoring (str): Scoring method to use. Default is 'neg_mean_absolute_error'.
        cv (int): Number of cross-validation folds. Default is 3.
        verbose (int): Verbosity level for RandomizedSearchCV. Default is 2.
        random_state (int): Random state for reproducibility. Default is 42.
        n_jobs (int): Number of jobs to run in parallel. Default is -1 (use all cores).
    
    Returns:
        dict: Best parameters found.
        float: Best log-transformed MAE score.
        float: Best MAE score (inverse of log transformation).
    """
    y_train_log = np.log1p(y_train)
    
    random_search = RandomizedSearchCV(
        estimator=XGBRegressor(random_state=random_state),
        param_distributions=param_grid,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        verbose=verbose,
        random_state=random_state,
        n_jobs=n_jobs
    )
    
    random_search.fit(X_train, y_train_log)
    
    best_params = random_search.best_params_
    best_score_log = -random_search.best_score_  
    best_score = np.expm1(best_score_log)       
    
    print("\n================ Best Results =================\n")
    print("Best Parameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    print(f"\nBest Log-Transformed MAE: {best_score_log:.4f}")
    print(f"Best MAE: {best_score:.4f}")
    
    return best_params, best_score_log, best_score

def prepare_training_and_test_data(X_train_final, X_val, X_test, y_train_final, y_val, X_train_xgb_columns):
    """
    Prepares the training, validation, and test datasets for the model.
    
    Parameters:
        X_train_final (pd.DataFrame): Training feature set.
        X_val (pd.DataFrame): Validation feature set.
        X_test (pd.DataFrame): Test feature set.
        y_train_final (pd.Series or pd.DataFrame): Training target set.
        y_val (pd.Series or pd.DataFrame): Validation target set.
        X_train_xgb_columns (Index or list): Column names to select from all datasets.
    
    """
    X_train_val = pd.concat([X_train_final[X_train_xgb_columns], X_val[X_train_xgb_columns]])
    y_train_val = pd.concat([y_train_final, y_val])
    
    X_test = X_test[X_train_xgb_columns]
    
    print(f"Shape of X_train_val: {X_train_val.shape}")
    print(f"Shape of y_train_val: {y_train_val.shape}")
    print(f"Shape of X_test: {X_test.shape}")
    
    return X_train_val, y_train_val, X_test

def final_model(X_train_val, y_train_val, X_test, y_test, model, best_params):
    """
    Trains the final model using the best parameters, evaluates on the test set, and prints the metrics.

    Parameters:
    - X_train_val (DataFrame): Combined training and validation features.
    - y_train_val (Series): Combined training and validation target values.
    - X_test (DataFrame): Test features.
    - y_test (Series): Test target values.
    - model: The machine learning model (e.g., XGBRegressor).
    - best_params (dict): Dictionary of best hyperparameters for the model.

    Returns:
    - None (prints the results directly).
    """
    y_train_val_log = np.log1p(y_train_val
                              )
    model.set_params(**best_params)
    model.fit(X_train_val, y_train_val_log)

    y_test_pred_log = model.predict(X_test)
    y_test_pred = np.expm1(y_test_pred_log)

    mae_train = mean_absolute_error(y_train_val, np.expm1(model.predict(X_train_val)))
    mape_train = mean_absolute_percentage_error(y_train_val, np.expm1(model.predict(X_train_val)))

    mae_test = mean_absolute_error(y_test, y_test_pred)
    mape_test = mean_absolute_percentage_error(y_test, y_test_pred)

    print("Training and evaluating with log transformation: Final Model")
    print(f"Training MAE: {mae_train:.2f}")
    print(f"Training MAPE: {mape_train:.4f}")
    print(f"Test MAE: {mae_test:.2f}")
    print(f"Test MAPE: {mape_test:.4f}")

    return y_test_pred

def save_predictions(y_actual, y_predicted, filename, folder_path='../Predictions'):
    """
    Saves predictions to a specified folder as a CSV file.

    Parameters:
    y_actual (array-like): The actual target values.
    y_predicted (array-like): The predicted target values.
    filename (str): The name of the file to save the predictions as (e.g., 'predictions.csv').
    folder_path (str): Relative or absolute path to the folder where the predictions should be saved. Default is '../Predictions'.

    Returns:
    str: Full path of the saved predictions file.
    """
    os.makedirs(folder_path, exist_ok=True)

    predictions_filepath = os.path.join(folder_path, filename)

    predictions_df = pd.DataFrame({
        'Actual': y_actual,
        'Predicted': y_predicted
    })

    predictions_df.to_csv(predictions_filepath, index=False)

    print(f"Predictions saved as '{predictions_filepath}'")
    return predictions_filepath

def save_model(model, model_name, folder_path='../Models'):
    """
    Saves a machine learning model to a specified folder.

    Parameters:
    model (object): The trained model to save.
    model_name (str): The name of the file to save the model as (e.g., 'xgb_model.joblib').
    folder_path (str): Relative or absolute path to the folder where the model should be saved. Default is '../Models'.

    Returns:
    str: Full path of the saved model file.
    """
    os.makedirs(folder_path, exist_ok=True)

    model_filename = os.path.join(folder_path, model_name)

    dump(model, model_filename)

    print(f"Model saved as '{model_filename}'")
    return model_filename


def plot_real_vs_predicted(data, y_test, y_test_pred, start_date="2012-11-01", end_date="2012-12-31", title="Comparison of Real vs Predicted Values"):
    """
    Filters data for the given date range and plots a comparison of real vs predicted values with improved visuals.

    Parameters:
    - data (pd.DataFrame): DataFrame containing the `dteday` column for filtering.
    - y_test (pd.Series): The actual target values (aligned with data).
    - y_test_pred (array-like): The predicted values (aligned with data, can be numpy array or pandas Series).
    - start_date (str): The start of the date range for filtering (default is "2012-11-01").
    - end_date (str): The end of the date range for filtering (default is "2012-12-31").
    - title (str): The title of the plot.

    Returns:
    - None (displays the plot).
    """

    data_test = data.loc[y_test.index] 
    data_test['dteday'] = pd.to_datetime(data_test['dteday'])
    mask = (data_test['dteday'] >= start_date) & (data_test['dteday'] <= end_date)


    filtered_dates = data_test.loc[mask, 'dteday']                
    filtered_y_real = y_test.loc[mask]                          


    if isinstance(y_test_pred, pd.Series):
        filtered_y_pred = y_test_pred.loc[mask] 
    else:
        filtered_y_pred = y_test_pred[mask]     

    df_comparison = pd.DataFrame({
        'Date': pd.to_datetime(filtered_dates), 
        'Real Values': filtered_y_real,
        'Predicted Values': filtered_y_pred
    })

    df_aggregated = df_comparison.groupby('Date').mean().reset_index()

    df_melted = df_aggregated.melt(
        id_vars='Date',
        value_vars=['Real Values', 'Predicted Values'],
        var_name='Type',
        value_name='Value'
    )

    tick_vals = df_aggregated['Date'][::7] 

    fig = px.line(
        df_melted,
        x='Date',
        y='Value',
        color='Type',
        title=title,
        labels={'Date': 'Date', 'Value': 'Value', 'Type': 'Type'}
    )

    fig.update_layout(
        width=1200,  
        height=600,  
        title_x=0.5,  
        xaxis=dict(
            title="Date",
            tickangle=45,  
            tickmode='array',  
            tickvals=tick_vals, 
            tickformat="%b %d",  
            gridcolor='lightgrey',  
        ),
        yaxis=dict(
            title="Value",
            gridcolor='lightgrey'  
        ),
        font=dict(size=14),  
        plot_bgcolor='white',  
        legend=dict(
            title='Legend',
            orientation='h', 
            x=0.5,  
            xanchor='center',
            y=-0.2  
        )
    )

    fig.show()