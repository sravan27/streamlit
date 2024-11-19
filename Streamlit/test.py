import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from time import sleep
import streamlit.components.v1 as components

# Load pre-trained models and data
@st.cache_data
def load_models():
    models = {
        "Random Forest": joblib.load("../Models/random_forrest.joblib"),
        "XGBoost": joblib.load("../Models/xgb_best_model.joblib"),
        "LightGBM": joblib.load("../Models/lgb_model.joblib")
    }
    return models

@st.cache_data
def load_data():
    df = pd.read_csv("../Dataset/bike-sharing-hourly.csv")
    df['dteday'] = pd.to_datetime(df['dteday'])
    return df

models = load_models()
data = load_data()

# Ensure feature columns match model expectations
def align_features(X, reference_X):
    missing_cols = set(reference_X.columns) - set(X.columns)
    for col in missing_cols:
        X[col] = 0
    extra_cols = set(X.columns) - set(reference_X.columns)
    if extra_cols:
        X = X.drop(columns=extra_cols)
    return X[reference_X.columns]

# Split Data Function (simplified)
def split_data(df, target_column='cnt', test_year=1, test_month_start=11, val_ratio=0.2):
    y = df[target_column]
    X = df.drop(columns=[target_column])

    df_test = df[(df['yr'] == test_year) & (df['mnth'] >= test_month_start)]
    df_train = df.drop(df_test.index)
    unnecessary_columns = ['instant', 'dteday', 'casual', 'registered', 'atemp', 'mnth']
    X_train = df_train.drop(columns=unnecessary_columns, errors='ignore')
    y_train = df_train[target_column]

    X_test = df_test.drop(columns=unnecessary_columns, errors='ignore')
    y_test = df_test[target_column]

    train_size = int(len(X_train) * (1 - val_ratio))
    X_train_final = X_train.iloc[:train_size]
    X_val = X_train.iloc[train_size:]
    y_train_final = y_train.iloc[:train_size]
    y_val = y_train.iloc[train_size:]

    # Align X_test to match X_train
    X_test_aligned = align_features(X_test, X_train_final)
    X_val_aligned = align_features(X_val, X_train_final)

    return X_train_final, X_val_aligned, X_test_aligned, y_train_final, y_val, y_test

# Load training, validation, and testing sets
X_train, X_val, X_test, y_train, y_val, y_test = split_data(data)

# Tabs for different user roles
st.title("Washington D.C. Bike Sharing Dashboard")
tabs = st.tabs(["ML Enthusiasts", "Business Users", "Model Comparison", "Real vs Predicted"])

with tabs[0]:  # ML Enthusiasts
    st.header("Hyperparameter Tuning & Model Training")
    model_name = st.selectbox("Choose Model", list(models.keys()))
    model = models[model_name]

    # Hyperparameter inputs
    if model_name == "Random Forest":
        n_estimators = st.slider("Number of Trees", 10, 200, 100, step=10)
        max_depth = st.slider("Max Depth", 1, 20, 10)
        # Update model hyperparameters dynamically
        model.set_params(n_estimators=n_estimators, max_depth=max_depth)

    elif model_name == "XGBoost":
        learning_rate = st.slider("Learning Rate", 0.01, 0.3, 0.1, step=0.01)
        n_estimators = st.slider("Number of Estimators", 10, 200, 100, step=10)
        model.set_params(learning_rate=learning_rate, n_estimators=n_estimators)

    elif model_name == "LightGBM":
        num_leaves = st.slider("Number of Leaves", 10, 100, 30, step=5)
        max_depth = st.slider("Max Depth", -1, 20, 10)
        model.set_params(num_leaves=num_leaves, max_depth=max_depth)

    # Train model and display performance when button is clicked
    if st.button("Train Model"):
        with st.spinner('Training in progress...'):
            sleep(1)  # Simulate delay
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_val_pred = model.predict(X_val)
            
            mae_train = mean_absolute_error(y_train, y_train_pred)
            mae_val = mean_absolute_error(y_val, y_val_pred)
            
            st.write(f"Training MAE: {mae_train:.2f}")
            st.write(f"Validation MAE: {mae_val:.2f}")

with tabs[1]:  # Business Users
    st.header("Predict Bike Demand")
    date = st.date_input("Select Date")
    time = st.slider("Select Hour of the Day", 0, 23, 12)
    season = st.selectbox("Season", ["Winter", "Spring", "Summer", "Fall"])
    holiday = st.radio("Holiday", ["No", "Yes"])
    working_day = st.radio("Working Day", ["No", "Yes"])
    weather = st.selectbox("Weather Condition", ["Clear/Partly Cloudy", "Mist/Cloudy", "Light Snow/Rain"])
    temp = st.slider("Temperature (Normalized)", 0.0, 1.0, 0.5)
    hum = st.slider("Humidity (Normalized)", 0.0, 1.0, 0.5)
    windspeed = st.slider("Wind Speed (Normalized)", 0.0, 1.0, 0.5)

    # Prepare features for prediction
    features = pd.DataFrame({
        'hr': [time],
        'season': [1 if season == "Winter" else 2 if season == "Spring" else 3 if season == "Summer" else 4],
        'holiday': [1 if holiday == "Yes" else 0],
        'workingday': [1 if working_day == "Yes" else 0],
        'weathersit': [1 if weather == "Clear/Partly Cloudy" else 2 if weather == "Mist/Cloudy" else 3],
        'temp': [temp],
        'hum': [hum],
        'windspeed': [windspeed]
    })

    # Align features with model requirements
    features_aligned = align_features(features, X_train)

    # Scaling features if needed
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_aligned)
    prediction = model.predict(features_scaled)
    st.write(f"Predicted Bike Demand: {prediction[0]:.0f} bikes")

with tabs[2]:  # Model Comparison
    st.header("Compare Model Performance")
    metrics = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_val_pred = model.predict(X_val)
        mae_val = mean_absolute_error(y_val, y_val_pred)
        metrics.append({"Model": name, "Validation MAE": mae_val})

    metrics_df = pd.DataFrame(metrics)
    st.dataframe(metrics_df)

    # Visualization
    fig = px.bar(metrics_df, x="Model", y="Validation MAE", title="Model Comparison by Validation MAE")
    st.plotly_chart(fig)

with tabs[3]:  # Real vs Predicted
    st.header("Real vs Predicted Values")
    best_model_name = "Random Forest"  # Assuming Random Forest is the best model
    model = models[best_model_name]
    model.fit(X_train, y_train)
    y_test_pred = model.predict(X_test)

    # Create DataFrame for real vs predicted values
    comparison_df = pd.DataFrame({
        'Date': data.loc[y_test.index, 'dteday'],
        'Real Values': y_test,
        'Predicted Values': y_test_pred
    })

    # Interactive plot using Plotly
    comparison_fig = px.line(
        comparison_df,
        x='Date',
        y=['Real Values', 'Predicted Values'],
        title="Comparison of Real vs Predicted Values",
        color_discrete_map={"Real Values": "blue", "Predicted Values": "red"}
    )
    comparison_fig.update_traces(line=dict(width=2))
    comparison_fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Value",
        legend_title="Legend",
        width=1000,
        height=500
    )
    st.plotly_chart(comparison_fig)

# Run the app using: streamlit run streamlit_app.py
