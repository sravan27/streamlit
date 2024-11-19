import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from time import sleep
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import lightgbm as lgb

# =========================
# Data Loading and Preprocessing
# =========================

@st.cache_data
def load_data():
    file_path = 'Dataset/bike-sharing-hourly.csv'
    try:
        data = pd.read_csv(file_path)
        data['dteday'] = pd.to_datetime(data['dteday'])
        data = load_and_prepare_data(data)
        return data
    except FileNotFoundError:
        st.error("Dataset file not found. Please ensure the file is in the 'Dataset' folder.")
        return pd.DataFrame()

def load_and_prepare_data(df):
    df = add_cyclic_features(df, 'hr', 24)
    df = add_cyclic_features(df, 'season', 4)
    df = temp_hum(df)
    df = temp_windspeed(df)
    df = hum_windspeed(df)
    df = add_daylight_period(df)
    df = one_hot_encode_column_single(df, 'daylight_period')
    unnecessary_columns = ['instant', 'casual', 'registered', 'atemp']
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
    y = df[target_column]
    X = df.drop(columns=[target_column, 'dteday'])
    df_test = df[(df['yr'] == test_year) & (df['mnth'] >= test_month_start)]
    df_train = df.drop(df_test.index)
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
data = load_data()
if data.empty:
    st.stop()

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

st.title("Washington D.C. Bike Sharing Dashboard")
tabs = st.tabs(["ML Enthusiasts", "Business Users", "Model Comparison", "Real vs Predicted"])

# ====================================
# Tab 1: ML Enthusiasts
# ====================================
with tabs[0]:
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

            st.write(f"Training MAE: {mean_absolute_error(y_train, y_train_pred):.2f}")
            st.write(f"Training MAPE: {mean_absolute_percentage_error(y_train, y_train_pred):.4f}")
            st.write(f"Validation MAE: {mean_absolute_error(y_val, y_val_pred):.2f}")
            st.write(f"Validation MAPE: {mean_absolute_percentage_error(y_val, y_val_pred):.4f}")

# ====================================
# Tab 2: Business Users
# ====================================
with tabs[1]:
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
# Tab 3: Model Comparison
# ====================================
with tabs[2]:
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
        metrics.append({"Model": name, "Validation MAE": mae_val, "Test MAE": mae_test, "Validation MAPE": mape_val, "Test MAPE": mape_test})

    metrics_df = pd.DataFrame(metrics)
    st.dataframe(metrics_df)

    fig = px.bar(metrics_df, x="Model", y=["Validation MAE", "Test MAE"], barmode="group",
                 title="Model Comparison by MAE", labels={"value": "MAE", "variable": "Dataset"})
    st.plotly_chart(fig, key="model_comparison_chart")

# ====================================
# Tab 4: Real vs Predicted
# ====================================
with tabs[3]:
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

# =========================
# Run the app using: streamlit run streamlit_app.py
# =========================
