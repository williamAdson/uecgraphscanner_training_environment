import os
import yaml
import torch
import pandas as pd
from sklearn.model_selection import train_test_split

def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def create_data_splits(config):
    """Splits processed.csv into train, val, and test sets."""
    processed_path = config['data']['processed_path']
    if not os.path.exists(processed_path):
        raise FileNotFoundError(f"{processed_path} not found.")

    df = pd.read_csv(processed_path)
    
    # 70% Train, 15% Val, 15% Test
    val_test_ratio = config['data']['val_size'] + config['data']['test_size']
    val_ratio_of_rem = config['data']['val_size'] / val_test_ratio

    train_df, temp_df = train_test_split(
        df, 
        test_size=val_test_ratio, 
        stratify=df[config['data']['label_col']], 
        random_state=config['data']['random_seed']
    )
    
    val_df, test_df = train_test_split(
        temp_df, 
        test_size=(1.0 - val_ratio_of_rem), 
        stratify=temp_df[config['data']['label_col']], 
        random_state=config['data']['random_seed']
    )

    os.makedirs("data/split", exist_ok=True)
    train_df.to_csv(config['data']['train_path'], index=False)
    val_df.to_csv(config['data']['val_path'], index=False)
    test_df.to_csv(config['data']['test_path'], index=False)
    
    print(f"Data split complete: Train ({len(train_df)}), Val ({len(val_df)}), Test ({len(test_df)})")

def save_checkpoint(model, optimizer, epoch, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)