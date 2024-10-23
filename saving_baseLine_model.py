import os
import json
import joblib
import zipfile

def unzip(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zipf:
        zipf.extractall(extract_to)

def update_base_line(model_info:dict):
    data_info_path = f'prepared_data/{model_info['data_version']}/data_prepartion_details.json'
    data_info = {}
        
    with open(data_info_path, 'r') as f:
        data_info = json.load(f)
    directory = 'model/'
    if not os.path.exists(directory):
        os.makedirs(directory)    
        
    if os.path.exists("model/baseline_info.json"):
        with open("model/baseline_info.json", 'r') as f:
            basline_info = json.load(f)
    else:
        basline_info = {}
        
    
    if 'Best_model' not in basline_info:
        basline_info['Best_model'] = {}
        with open("model/baseline_info.json", 'w') as f:
            json.dump(basline_info, f, indent=4)
        
    if 'val_r2' in basline_info['Best_model'] and basline_info['Best_model']['val_r2'] < model_info['val_r2']:
        basline_info["Best_model"]["train_rmse"] = model_info["train_rmse"]
        basline_info["Best_model"]["train_r2"] = model_info["train_r2"]
        basline_info["Best_model"]["val_r2"] = model_info["val_r2"]
        basline_info["Best_model"]["val_rmse"] = model_info["val_rmse"]
        basline_info["Best_model"]["selected_feature_names"] = model_info["selected_feature_names"]
        basline_info["Best_model"]["dataset_metadata"] = data_info
        with open("model/baseline_info.json", 'w') as f:
            json.dump(basline_info, f, indent=4)    
    else:
        history_model = {}
        if os.path.exists('model/history.json'):
            with open('model/history.json', 'r') as f:
                history_model = json.load(f)
        
        id = history_model.get("counter", 0)
        history_model["counter"] = id + 1
        history_model[id] = {
            'train_rmse': model_info['train_rmse'],
            'train_r2':   model_info['train_r2'],
            'val_rmse':  model_info['val_rmse'],
            'val_r2':    model_info['val_r2'],
            'selected_feature_names': model_info['selected_feature_names'],
            'dataset_metadata': data_info
        }
        with open('model/history.json', 'w') as f:
            json.dump(history_model, f, indent=4)
        
def load_model(model_file_path):
    try:
        model = joblib.load(model_file_path)
        return model
    except FileNotFoundError:
        print(f"Error: The file at {model_file_path} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")