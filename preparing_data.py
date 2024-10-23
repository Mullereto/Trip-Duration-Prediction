from scipy import stats
import pandas as pd
import numpy as np
from geopy import distance
from geopy.point import Point
import math
import os
import datetime
import json
from saving_baseLine_model import unzip

#function to remove the outliar
def remove_outliar(df, feature_column, target_column, factor=3):
    """
    Remove the outlair form the data set using 2 method (z-score or IQR)

    Args:
        df (pd.dataFrame()): The data set you want to remove the outliar from it
        feature_column (str): the column that contain outlair
        target_column (str): The target column
        factor (int, optional): the limit for Your z-score. Defaults to 3.
    
    Return:
        pd.DataFram
    """
    
    if feature_column not in df.columns:
        raise ValueError(f"{feature_column} not in the dataframe")
    
    zscore = np.abs(stats.zscore(df[[feature_column, target_column]]))
    filtered_rows = (zscore < factor).all(axis=1)
    
    df_cleaned = df[filtered_rows].copy()
    return df_cleaned

#function to remove the un freqent data
def remove_unfreqent(df, categorical_columns, minfreqent=5):
    """Deletes rows containing infrequent categories

    Args:
        df (pd.DataFrame): the datafram you want to apply the function on it
        categorical_columns (list): List of the categorical columns
        minfreqent (int, optional): the min freqent number. Defaults to 5.
        
    Return: pd.DataFrame
    """

    for cat in categorical_columns:
        categorical_counter = df[cat].value_counts()
        
        counter_indicis = categorical_counter[categorical_counter < minfreqent].index
        
        df = df[~df[cat].isin(counter_indicis)]
        
        return df

#function to get the shortest distance
def get_distance_km(df):
    """Calculate the Shortest path between pickup and dropoff

    Args:
        df (pd.DataFrame): the data set you want to apply on it

    Returns:
        int: the shortest distance
    """
    pickup_coords = (df['pickup_latitude'], df['pickup_longitude'])
    dropoff_coords = (df['dropoff_latitude'], df['dropoff_longitude'])
    distance_km = distance.geodesic(pickup_coords, dropoff_coords).km
    return distance_km

def month_to_season(month):
    """add pick season to data

    Args:
        month (row): row in the data

    Returns:
        str: the season
    """
    if month in [12,1,2]:
        return 'Winter'
    elif month in [3,4,5]:
        return 'Spring'
    elif month in [6,7,8]:
        return 'Summer'
    elif month in [9,10,11]:
        return 'Autumn'
    else:
        return 'unknown'

def time_period(hour):
    """add time period of the day

    Args:
        hour (int): the hour of the day

    Returns:
        str: the period of the day
    """
    if 5 <= hour < 12:  # from 5am to 11:59 am
        return 'Morning'
    elif 12 <= hour < 17:  # from 12:00pm to 4:59pm
        return 'Afternoon'
    else:                   # from 5:00pm to 4:59am 
        return 'Night' 
    
def calculate_direction(row):
    """calculate the dirction the taxi driver took

    Args:
        row (pd.dataframe): the data to apply on it

    Returns:
        int: the dirction in 360
    """
    pickup_coordinates =  Point(row['pickup_latitude'], row['pickup_longitude'])
    dropoff_coordinates = Point(row['dropoff_latitude'], row['dropoff_longitude'])
    
    # Calculate the difference in longitudes
    delta_longitude = dropoff_coordinates[1] - pickup_coordinates[1]
    
    # Calculate the bearing (direction) using trigonometry
    y = math.sin(math.radians(delta_longitude)) * math.cos(math.radians(dropoff_coordinates[0]))
    x = math.cos(math.radians(pickup_coordinates[0])) * math.sin(math.radians(dropoff_coordinates[0])) - \
        math.sin(math.radians(pickup_coordinates[0])) * math.cos(math.radians(dropoff_coordinates[0])) * \
        math.cos(math.radians(delta_longitude))
    
    # Calculate the bearing in degrees
    bearing = math.atan2(y, x)
    bearing = math.degrees(bearing)
    
    # Adjust the bearing to be in the range [0, 360)
    bearing = (bearing + 360) % 360
    
    return bearing

def manhattan_distance(df):
    """calculate the distnce in manhatten city as close as real life

    Args:
        df (pd.DateFrame): the  data you want to apply on it

    Returns:
        int: Distance
        """
    lat_distance = abs(df['pickup_latitude'] - df['dropoff_latitude']) * 111  
    lon_distance = abs(df['pickup_longitude'] - df['dropoff_longitude']) * 111 * math.cos(math.radians(df['pickup_latitude']))
    
    return lat_distance + lon_distance

if __name__ == '__main__':
    train_path = 'split/train.zip'
    val_path = 'split/val.zip'
    unzip(train_path)
    df_train = pd.read_csv(train_path)
    unzip(val_path)
    df_val = pd.read_csv(val_path)
    
    #process for train set
    df_train['trip_duration'] = np.log1p(df_train['trip_duration'])
    
    df_train['pickup_datetime'] = pd.to_datetime(df_train['pickup_datetime'])
    
    #we see in the EDA that the feature "Distance, dirction and manhattan short path have huge impact on the (trip_duration)"
    #distance_short and mnhattan_short_path are skewd to the right so we will apply log transform to be normal distripution
    df_train['distance_short'] = df_train.apply(get_distance_km, axis=1)
    df_train['dirction'] = df_train.apply(calculate_direction, axis=1)
    df_train['mnhattan_short_path'] = df_train.apply(manhattan_distance, axis=1)

    
    
    df_train['distance_short'] = np.log1p(df_train['distance_short'])
    df_train['mnhattan_short_path'] = np.log1p(df_train['mnhattan_short_path'])
    df_train['dirction'] = np.log1p(df_train['dirction'])
    
    
    #we see in the EDA that we can extract more feature from the pickup_datetime like hour, dayOFWeek, month, time Period OF the day and seasons 
    df_train['pick_hour'] = df_train['pickup_datetime'].dt.hour
    df_train['pick_day_of_week'] = df_train['pickup_datetime'].dt.day_of_week
    df_train['pick_day'] = df_train['pickup_datetime'].dt.day
    df_train['pick_month'] = df_train['pickup_datetime'].dt.month
    df_train['is_weekend'] = df_train['pickup_datetime'].dt.dayofweek >= 5
    
    df_train['pick_season'] = df_train['pick_month'].apply(month_to_season)

    df_train['pick_time_period'] = df_train['pick_hour'].apply(time_period)
    
    df_train.drop(columns=['pickup_datetime', 'id'], inplace=True) #we drop the id becouse its useless for the model
    
    #################################NOTE all after this comment untill you see 3 NOTE is the same for valdtion data ######################################################
    #process for valdtion set
    df_val['trip_duration'] = np.log1p(df_val['trip_duration'])
    
    df_val['pickup_datetime'] = pd.to_datetime(df_val['pickup_datetime'])
    
    #we see in the EDA that the feature "Distance, dirction and manhattan short path have huge impact on the (trip_duration)"
    #distance_short and mnhattan_short_path are skewd to the right so we will apply log transform to be normal distripution
    df_val['distance_short'] = df_val.apply(get_distance_km, axis=1)
    df_val['dirction'] = df_val.apply(calculate_direction, axis=1)
    df_val['mnhattan_short_path'] = df_val.apply(manhattan_distance, axis=1)

    
    df_val['distance_short'] = np.log1p(df_val['distance_short'])
    df_val['mnhattan_short_path'] = np.log1p(df_val['mnhattan_short_path'])
    df_val['dirction'] = np.log1p(df_val['dirction'])
    
    
    #we see in the EDA that we can extract more feature from the pickup_datetime like hour, dayOFWeek, month, time Period OF the day and seasons 
    df_val['pick_hour'] = df_val['pickup_datetime'].dt.hour
    df_val['pick_day_of_week'] = df_val['pickup_datetime'].dt.day_of_week
    df_val['pick_day'] = df_val['pickup_datetime'].dt.day
    df_val['pick_month'] = df_val['pickup_datetime'].dt.month
    df_val['is_weekend'] = df_val['pickup_datetime'].dt.dayofweek >= 5
    
    df_val['pick_season'] = df_val['pick_month'].apply(month_to_season)

    df_val['pick_time_period'] = df_val['pick_hour'].apply(time_period)
    
    df_val.drop(columns=['pickup_datetime', 'id'], inplace=True)
    ###################NOTE###############NOTE##################NOTE################### END OF THE VALDTION PREPARING###################################
    
    
    # From our EDA, we need to use remove outliers from passenger_count and seems distance_km is necessary also.
    
    # #NOTE uncomment the next 5 line to remove the unfreqent entries
    # threshold = 3
    # categorical_features = ['vendor_id', 'passenger_count',  "pick_hour", "pick_day_of_week", "pick_month", "pick_season", 'pick_time_period', 'store_and_fwd_flag']
    # for i in range(len(categorical_features)-1):
    #     df_copy = df_train.copy()
    #     df_train = remove_unfreqent(df_copy, [categorical_features[i]], minfreqent=threshold)
    
    # #NOTE uncomment the next 5 line to remove the outliers
    # threshold = 3
    # numric_feature = ['trip_duration', 'distance_short', 'mnhattan_short_path']
    # for feature in numric_feature:
    #         df_train = remove_outliar(df_train, feature, 'trip_duration', factor=threshold)
    
    
    id = 0 #every prossece data has its owan ID
    #NOTE if you change samething in the in this code pls change the ID number to not over right sample data
    directory = f'prepared_data/{id}'
    
    if not os.path.exists(directory):
        os.makedirs(directory)    
        
    df_train.to_csv(f"{directory}/train.csv", index=False)
    df_val.to_csv(f"{directory}/val.csv", index=False)
    
    metadata = {
        'version': id,
        'version_description': {
            'summry':'outliar and unfrequnt',
            'new columns':["pick_hour", "pick_day_of_week", "pick_month", "pick_season", 'pick_time_period','distance_short', 'mnhattan_short_path', 'dirction'],
            'remove unfreqent data': None,
            'remove outliar': None,
            },
        'feature_names': df_train.columns.tolist(),
        'num_rows_train': len(df_train),
        'num_rows_val': len(df_val),
        'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(f'{directory}/data_prepartion_details.json', 'w') as f:
        json.dump(metadata, f, indent=4)
    
    print("The Data and the Data Prepartion has been saved successfully :)")