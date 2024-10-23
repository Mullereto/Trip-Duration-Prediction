import numpy as np 
import pandas as pd
from preparing_data import get_distance_km, month_to_season, time_period, manhattan_distance, calculate_direction
from train import log_transform, FunctionTransformer, eval_model, with_suffix, target
from saving_baseLine_model import load_model, unzip



def prepare(df_test:pd.DataFrame):
    df_test['trip_duration'] = np.log1p(df_test['trip_duration'])
    
    df_test['pickup_datetime'] = pd.to_datetime(df_test['pickup_datetime'])
    
    #we see in the EDA that the feature "Distance, dirction and manhattan short path have huge impact on the (trip_duration)"
    #distance_short and mnhattan_short_path are skewd to the right so we will apply log transform to be normal distripution
    df_test['distance_short'] = df_test.apply(get_distance_km, axis=1)
    df_test['dirction'] = df_test.apply(calculate_direction, axis=1)
    df_test['mnhattan_short_path'] = df_test.apply(manhattan_distance, axis=1)

    
    
    df_test['distance_short'] = np.log1p(df_test['distance_short'])
    df_test['mnhattan_short_path'] = np.log1p(df_test['mnhattan_short_path'])
    df_test['dirction'] = np.log1p(df_test['dirction'])
    
    
    #we see in the EDA that we can extract more feature from the pickup_datetime like hour, dayOFWeek, month, time Period OF the day and seasons 
    df_test['pick_hour'] = df_test['pickup_datetime'].dt.hour
    df_test['pick_day_of_week'] = df_test['pickup_datetime'].dt.day_of_week
    df_test['pick_day'] = df_test['pickup_datetime'].dt.day
    df_test['pick_month'] = df_test['pickup_datetime'].dt.month
    df_test['is_weekend'] = df_test['pickup_datetime'].dt.dayofweek >= 5
    
    df_test['pick_season'] = df_test['pick_month'].apply(month_to_season)

    df_test['pick_time_period'] = df_test['pick_hour'].apply(time_period)
    
    df_test.drop(columns=['pickup_datetime', 'id'], inplace=True) #we drop the id becouse its useless for the model
    
    return df_test


if __name__ == "__main__":
    train_path = 'split/train.zip'
    val_path = 'split/val.zip'
    test_path = "split/test.zip"
    model_path = 'model/Ridge_with_r2_train0.6997063984505685_and r2_val_0.6946488301222101_.pkl'
    
    model_pipline = load_model(model_path)
    
    data_processor = model_pipline['data_preprocessor']
    training_feature = model_pipline['selected_feature_names']
    model = model_pipline['model']
    
    unzip(test_path, 'split')
    df_test = pd.read_csv('split/test.csv')
    df_test = prepare(df_test)
    df_test_processed = data_processor.transform(df_test[training_feature])
    
    rmse, r2 = eval_model(model, df_test_processed, df_test[target], 'TEST')  # rmse = 04354 and r2-score = 0.7007
    
    #NOTE: Uncomment the next 5 line to try with train split
    # unzip(train_path, 'split')
    # df_train = pd.read_csv('split/train.csv')
    # df_train = prepare(df_train)
    # df_train_processed = data_processor.transform(df_train[training_feature])
    # rmse, r2 = eval_model(model, df_train_processed, df_train[target], 'TEST')
   
    #NOTE: Uncomment the next 5 line to try with valdtion split
    # unzip(val_path, 'split')
    # df_val = pd.read_csv('split/val.csv')
    # df_val = prepare(df_val)
    # df_val_processed = data_processor.transform(df_val[training_feature])
    # rmse, r2 = eval_model(model, df_val_processed, df_val[target], 'TEST')  