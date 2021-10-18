from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import joblib

def load_data():
    path = "./data/housing.csv"
    df = pd.read_csv(
        path,
    )

    return df


def feature_engineering(df: pd.DataFrame):
    df["rooms_per_household"] = df["total_rooms"]/df["households"]
    df["population_per_household"] = df["population"]/df["households"]

    median_bedrooms = df["total_bedrooms"].median()
    df["total_bedrooms"].fillna(median_bedrooms, inplace=True)

    return df


def train_model(df: pd.DataFrame):
    encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    scaler = StandardScaler()
    X = df.drop(columns="median_house_value")
    y = df["median_house_value"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    ct = ColumnTransformer(
        [('onehot', encoder, [
            'ocean_proximity'
        ]),
        ('scaler', scaler, ['latitude', 'longitude', 'housing_median_age', 'total_rooms',
       'total_bedrooms', 'population', 'households', 'median_income', "rooms_per_household", "population_per_household"]),
        ], 
        remainder='passthrough' 
    )

    pipe = Pipeline(steps=[('coltrans', ct), ('rf', RandomForestRegressor())])
    pipe = pipe.fit(X_train, y_train) 
    score = pipe.score(X_test, y_test)
    rmse = np.sqrt(mean_squared_error(y_test, pipe.predict(X_test)))
    print(f" R^2 score: {score}")
    print(f"RMSE score: {rmse}")
    
    return pipe

def save_model(pipe):
    joblib.dump(pipe, "./model/model.joblib")



if __name__ == '__main__':
    df = load_data()
    df = feature_engineering(df)
    pipe = train_model(df)
    save_model(pipe)