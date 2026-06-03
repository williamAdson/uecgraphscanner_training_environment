import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from src.utils import load_config, create_data_splits
from src.feature_engineering import OpcodeVocab
from src.data_loader import BinarySmartContractDataset, collate_fn
from src.model import HybridBinaryClassifier
from src.train import train_model

def main():
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create Data Splits if they don't exist
    if not os.path.exists(config['data']['train_path']):
        print("Generating data splits...")
        create_data_splits(config)

    train_df = pd.read_csv(config['data']['train_path'])
    val_df = pd.read_csv(config['data']['val_path'])

    # Build Vocabulary
    print("Building vocabulary...")
    vocab = OpcodeVocab()
    for opcodes in train_df[config['data']['opcode_col']].dropna():
        vocab.count(opcodes.split())
    vocab.build(min_freq=config['features']['vocab_min_freq'])

    # Create Datasets and Loaders
    print("Initializing DataLoaders...")
    train_dataset = BinarySmartContractDataset(train_df, vocab, config)
    val_dataset = BinarySmartContractDataset(val_df, vocab, config)

    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False, collate_fn=collate_fn)

    # Initialize Model
    print("Initializing Hybrid Model...")
    model = HybridBinaryClassifier(len(vocab), config).to(device)

    # Train
    print("Starting training...")
    train_model(model, train_loader, val_loader, config, device)

if __name__ == "__main__":
    main()