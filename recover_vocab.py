import os
import yaml
import torch
import pandas as pd
from src.feature_engineering import OpcodeVocab

def main():
    # 1. Load your configuration files
    print("Loading project configurations...")
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    train_path = config['data']['train_path']
    opcode_col = config['data']['opcode_col']
    min_freq = config['features']['vocab_min_freq']
    checkpoint_dir = config['output']['checkpoint_dir']
    
    if not os.path.exists(train_path):
        print(f"Error: Could not find training split at {train_path}")
        return

    # 2. Read the training split CSV 
    print(f"Reading training data from {train_path}...")
    train_df = pd.read_csv(train_path)
    
    # 3. Rebuild the exact same vocabulary index mapping used during training
    print("Rebuilding vocabulary mapping matrices...")
    vocab = OpcodeVocab()
    for opcodes in train_df[opcode_col].dropna():
        vocab.count(str(opcodes).split())
        
    vocab.build(min_freq=min_freq)
    
    # 4. Export the file artifact to your checkpoints directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    vocab_path = os.path.join(checkpoint_dir, "vocab.pth")
    torch.save(vocab, vocab_path)
    print(f"\n[Success] Regenerated and saved vocabulary file to: {vocab_path}")

if __name__ == "__main__":
    main()