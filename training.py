import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


MODEL_FILE="model.pkl"
PIPELINE_FILE="pipeline.pkl"

def build_pipeline(num_att,cat_att):
    num_pipeline=Pipeline([
        ("imputer",SimpleImputer(strategy="median")),
        ("Scaler",StandardScaler())
        ])

    # for categorical columns
    cat_pipeline=Pipeline([
        ("onehot",OneHotEncoder(handle_unknown="ignore"))
        ])

    # Construct the full pipeline
    full_pipeline=ColumnTransformer([
        ("num",num_pipeline,num_att),
        ("cat",cat_pipeline,cat_att)
        ])
    return full_pipeline

if not os.path.exists(MODEL_FILE):
    # TRAINING PHASE
    housing = pd.read_csv("dataset/housing.csv")
    housing['income_cat'] = pd.cut(housing["median_income"], bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
                                   labels=[1, 2, 3, 4, 5])
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    for train_index, test_index in split.split(housing, housing['income_cat']):
        housing = housing.loc[train_index].drop("income_cat", axis=1)  # We will work on this data

        # separate feature and labels?
        housing_labels = housing["median_house_value"].copy()
        housing_features = housing.drop("median_house_value", axis=1)

        num_att=housing["median_house_value"].copy()
        cat_att=["ocean_proximity"]

        pipeline = build_pipeline(num_att, cat_att)
        housing_pre = pipeline.fit_transform(housing_features)

        model=RandomForestRegressor(random_state=42)
        model.fit(housing_pre,housing_labels)

        # Save model and pipeline
        joblib.dump(model,MODEL_FILE)
        joblib.dump(model,PIPELINE_FILE)

        print("Model is trained and saved")


