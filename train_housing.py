import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib
import os

DATA_PATH = "/opt/airflow/data/housing.csv"
PROCESSED_PATH = "/opt/airflow/data/housing_processed.csv"
MODEL_PATH = "/opt/airflow/data/housing_model.pkl"


def download_data():
    """Загрузка данных о недвижимости"""
    url = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv"
    df = pd.read_csv(url)
    df.to_csv(DATA_PATH, index=False)
    return f"Data saved to {DATA_PATH}"


def preprocess_data():
    """Очистка и подготовка данных"""
    df = pd.read_csv(DATA_PATH)

    # Удаление пропущенных значений
    df = df.dropna()

    # Преобразование категориальных признаков
    df = pd.get_dummies(df, columns=['ocean_proximity'])

    # Сохранение обработанных данных
    df.to_csv(PROCESSED_PATH, index=False)
    return f"Processed data saved to {PROCESSED_PATH}"


def train_model():
    """Обучение модели"""
    df = pd.read_csv(PROCESSED_PATH)

    X = df.drop('median_house_value', axis=1)
    y = df['median_house_value']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Оценка модели
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    print(f"Model MAE: {mae}")

    # Сохранение модели
    joblib.dump(model, MODEL_PATH)
    return f"Model saved to {MODEL_PATH} with MAE: {mae}"
