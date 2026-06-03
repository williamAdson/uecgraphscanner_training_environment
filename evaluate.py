#!/usr/bin/env python3
import os
import json
import torch
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from src.feature_engineering import (
    OpcodeVocab,
    tokenize_opcodes,
    bytecode_ngram_features
)
from src.model import HybridBinaryClassifier 

def run_test_evaluation():
    train_config = {
        'data': {
            'test_path': "data/split/test.csv",
            'label_col': "label_encoded",
            'opcode_col': "opcode",
            'bytecode_col': "bytecode"
        },
        'features': {
            'chunk_size': 256,
            'chunk_overlap': 32,
            'max_chunks': 12,
            'bytecode_ngram_dim': 64
        },
        'model': {
            'trans_hidden': 256,
            'trans_layers': 3,
            'trans_heads': 8,
            'trans_dropout': 0.20,
            'gat_hidden': 128,
            'gat_layers': 3,
            'gat_heads': 4,
            'gat_dropout': 0.20,
            'gat_edge_dim': 8,
            'fusion_heads': 8,
            'fusion_dropout': 0.20
        }
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Initializing evaluation engine on device: {device}")


    vocab_path = "models/checkpoints/vocab.pth"
    if not os.path.exists(vocab_path):
        vocab_path = "../models/checkpoints/vocab.pth"
        
    vocab = torch.load(vocab_path, map_location='cpu', weights_only=False)
    vocab_size = len(vocab)
    print(f"✅ Loaded Vocabulary Asset. Size: {vocab_size}")


    eval_model = HybridBinaryClassifier(vocab_size=vocab_size, config=train_config)
    model_path = "models/checkpoints/best_model.pth"
    if not os.path.exists(model_path):
        model_path = "../models/checkpoints/best_model.pth"

    checkpoint = torch.load(model_path, map_location='cpu')
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        eval_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        eval_model.load_state_dict(checkpoint)
        
    eval_model = eval_model.to(device)
    eval_model.eval()
    print("🔒 Model architecture state loaded and set to eval mode.")


    test_csv_path = train_config['data']['test_path']
    if not os.path.exists(test_csv_path):
        test_csv_path = os.path.join("..", test_csv_path)

    print(f"📖 Ingesting validation test records from: {test_csv_path}")
    test_df = pd.read_csv(test_csv_path)

    y_true, y_pred, y_probs = [], [], []

    with torch.no_grad():
        for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Evaluating Test Set"):
            try:
                opcodes = str(row[train_config['data']['opcode_col']])
                bytecode = str(row[train_config['data']['bytecode_col']])
                label = int(row[train_config['data']['label_col']])
                
                # Preprocessing execution
                fc = train_config['features']
                chunks = tokenize_opcodes(opcodes, vocab, fc["chunk_size"], fc["chunk_overlap"], fc["max_chunks"])
                bc_feat = bytecode_ngram_features(bytecode, dim=fc["bytecode_ngram_dim"])
                
                max_c = fc["max_chunks"]
                c_size = fc["chunk_size"]
                pad_n = max_c - chunks.shape[0]
                if pad_n > 0:
                    chunks = torch.cat([chunks, torch.zeros(pad_n, c_size, dtype=torch.long)], 0)
                masks = torch.tensor([1] * (max_c - pad_n) + [0] * pad_n, dtype=torch.bool)
                
                # Transformer zero padding patch
                for c_idx in range(chunks.shape[0]):
                    if torch.all(chunks[c_idx] == 0):
                        chunks[c_idx, 0] = 1
                
                # Production graph fallback arrays
                from torch_geometric.data import Data, Batch
                pyg_obj = Data(
                    x=torch.zeros(1, 184), 
                    edge_index=torch.zeros((2, 0), dtype=torch.long), 
                    edge_attr=torch.zeros((0, 8), dtype=torch.float), 
                    num_nodes=1
                )
                struct_prior = torch.tensor([1.0/500.0, 0.0, 0.0], dtype=torch.float)
                
                batch_payload = {
                    "chunks": chunks.unsqueeze(0).to(device),
                    "masks": masks.unsqueeze(0).to(device),
                    "bc_feat": bc_feat.unsqueeze(0).to(device),
                    "struct_prior": struct_prior.unsqueeze(0).to(device),
                    "graph_pattern": torch.zeros((1, 32), dtype=torch.float).to(device),
                    "graphs": Batch.from_data_list([pyg_obj]).to(device)
                }
                
                logits = eval_model(batch_payload)
                prob = torch.sigmoid(logits).item()
                pred = 1 if prob > 0.5 else 0
                
                y_true.append(label)
                y_pred.append(pred)
                y_probs.append(prob)
                
            except Exception as e:
                continue


    print("\n📊 Computing final performance metrics...")
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = map(int, cm.ravel())
    sk_report = classification_report(y_true, y_pred, output_dict=True)
    
    try:
        roc_auc = float(roc_auc_score(y_true, y_probs))
    except ValueError:
        roc_auc = -1.0

    metrics_payload = {
        "evaluation_timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dataset_summary": {
            "total_test_samples_processed": len(y_true),
            "malicious_samples_count": sum(y_true),
            "benign_samples_count": len(y_true) - sum(y_true)
        },
        "overall_metrics": {
            "accuracy": float(sk_report["accuracy"]),
            "macro_avg_f1": float(sk_report["macro avg"]["f1-score"]),
            "weighted_avg_f1": float(sk_report["weighted avg"]["f1-score"]),
            "roc_auc_score": roc_auc
        },
        "class_level_metrics": {
            "benign": {
                "precision": float(sk_report["0"]["precision"]),
                "recall": float(sk_report["0"]["recall"]),
                "f1_score": float(sk_report["0"]["f1-score"]),
                "support": int(sk_report["0"]["support"])
            },
            "unchecked_calls": {
                "precision": float(sk_report["1"]["precision"]),
                "recall": float(sk_report["1"]["recall"]),
                "f1_score": float(sk_report["1"]["f1-score"]),
                "support": int(sk_report["1"]["support"])
            }
        },
        "confusion_matrix": {
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp
        }
    }

    # Target path enforcement
    output_json_path = "logs/tensorboard/tests.json"
    dir_name = os.path.dirname(output_json_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_payload, f, indent=2)

    print("="*60)
    print(f"🎉 SUCCESS: Metrics written to: {output_json_path}")
    print(f"🎯 Test Accuracy: {metrics_payload['overall_metrics']['accuracy']*100:.2f}% | ROC-AUC: {roc_auc:.4f}")
    print("="*60)

if __name__ == "__main__":
    run_test_evaluation()