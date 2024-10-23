from typing import List
import joblib
import datetime
import numpy as np
import pandas as pd
from saving_baseLine_model import unzip
from saving_baseLine_model import update_base_line
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, PolynomialFeatures, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, LassoCV
from sklearn.metrics import r2_score, root_mean_squared_error
import os

SEED = 42



target = 'trip_duration'
categorical_data = [
    "passenger_count","pick_month","pick_day_of_week","pick_hour","pick_season","pick_time_period" ,"is_weekend","store_and_fwd_flag",
    "vendor_id"
]

numric_data = [
    'pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','distance_short', 'dirction', 'mnhattan_short_path'
]


train_feature = numric_data + categorical_data
def eval_model(model, x, target, name):
    """evaluate the model and get the errors

    Args:
        model (model): the model you have trained 
        x (ndarray):the data you want to eval it
        target (ndarray): the target value
        name (str): detrmain it as traing o val

    Returns:
        float: the errors
    """
    y_predict = model.predict(x)
    rmse = root_mean_squared_error(target, y_predict)
    r2 = r2_score(target, y_predict)
    print(f'{name} rmse = {rmse:.4f} and r2 score = {r2:.4f}')
    return rmse, r2

def with_suffix(_, names: List[str]):
    return [name + '__log' for name in names]

def log_transform(x): 
    return np.log1p(np.maximum(x, 0))

def make_pipeline(x_train, x_val):
    """make the pipline and transforming the data before fit with the model

    Args:
        x_train (pd.DataFrame): the train dataset
        x_val (pd.DataFrame): the val data set

    Returns:
        _type_: the model and the data processorand train errors and val errors
    """
    LogFeatures = FunctionTransformer(log_transform, feature_names_out=with_suffix)
    
    numric_transfome = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('poly', PolynomialFeatures(degree=5)),
        ('log', LogFeatures)
    ])
    
    categorical_transforme = Pipeline(steps=[
        ('ohe', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    transformer = ColumnTransformer(
        transformers=[
            ('num', numric_transfome, numric_data),
            ('cat', categorical_transforme, categorical_data)
        ],
        remainder='passthrough'
    )
    
    data_processor = Pipeline(steps=[
        ('processore_data', transformer),
        
    ])
    
    ridge = Pipeline(steps=[
        ('train', Ridge(alpha=1, random_state=SEED))
    ])
    
    x_train_prossed = data_processor.fit_transform(x_train[train_feature])
    x_val_prossed = data_processor.transform(x_val[train_feature])
    

    ridge.fit(x_train_prossed, x_train[target])
    
    train_rmse, train_r2 = eval_model(ridge, x_train_prossed, x_train[target], 'train')
    val_rmse, val_r2 = eval_model(ridge, x_val_prossed, x_val[target], 'val')
    
    
    return ridge, data_processor, train_rmse, train_r2, val_rmse, val_r2

if __name__ == "__main__":
    data_version = 0
    data_path = f'prepared_data/{data_version}'
    train_path = f'{data_path}/train.zip'
    val_path = f'{data_path}/val.zip'
    
    unzip(train_path)
    x_train = pd.read_csv(train_path)
    unzip(val_path)
    x_val = pd.read_csv(val_path)
    
    model, data_processor, train_rmse, train_r2, val_rmse, val_r2 = make_pipeline(x_train, x_val)
    
    model_data = {
        'model': model,

        'data_path': data_path,
        'train_rmse': train_rmse,
        'train_r2': train_r2,
        'val_rmse': val_rmse,
        'val_r2': val_r2,
        'selected_feature_names': train_feature,
        'data_preprocessor': data_processor,
        'data_version': data_version,
        'random_seed': SEED
    }
    model_name = f'model/Ridge_with_r2_train{train_r2}_and r2_val_{val_r2}_.pkl'
    joblib.dump(model_data, model_name)
    
    update_base_line(model_data)