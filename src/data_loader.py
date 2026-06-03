import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch
from src.feature_engineering import (
    bytecode_ngram_features, tokenize_opcodes, parse_cfg_dot, 
    analyze_graph_patterns, cfg_to_pyg, NODE_FEAT_DIM
)

_EMPTY_GRAPH_DIM = NODE_FEAT_DIM + 32

class BinarySmartContractDataset(Dataset):
    def __init__(self, df, vocab, config):
        self.df = df.reset_index(drop=True)
        self.vocab = vocab
        self.c = config['data']
        self.fc = config['features']

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = float(row[self.c["label_col"]]) # 0.0 or 1.0

        chunks = tokenize_opcodes(
            row.get(self.c["opcode_col"],""), self.vocab,
            self.fc["chunk_size"], self.fc["chunk_overlap"], self.fc["max_chunks"]
        )
        bc_feat = bytecode_ngram_features(
            str(row.get(self.c["bytecode_col"],"")), dim=self.fc["bytecode_ngram_dim"]
        )

        cfg_n = int(row.get(self.c["cfg_nodes_col"], 0))
        cfg_e = int(row.get(self.c["cfg_edges_col"], 0))
        cfg_d = float(row.get(self.c["cfg_density_col"], 0.0))

        try:
            parsed = parse_cfg_dot(str(row.get(self.c["cfg_dot_col"],"")))
            gp_feats = analyze_graph_patterns(parsed["nodes"], parsed["edges"], cfg_n, cfg_e, cfg_d)
            pyg_obj = cfg_to_pyg(parsed, self.vocab, gp_feats)
        except Exception:
            gp_feats = [0.0]*32
            from torch_geometric.data import Data
            pyg_obj = Data(
                x=torch.zeros(1, _EMPTY_GRAPH_DIM),
                edge_index=torch.zeros((2,0), dtype=torch.long),
                edge_attr=torch.zeros((0,8), dtype=torch.float),
                num_nodes=1,
            )

        struct_prior = torch.tensor([min(cfg_n/500.0,1), min(cfg_e/600.0,1), min(cfg_d/2.0,1)], dtype=torch.float)

        return {
            "chunks": chunks, "bc_feat": bc_feat, "struct_prior": struct_prior,
            "graph": pyg_obj, "graph_pattern": torch.tensor(gp_feats, dtype=torch.float),
            "label": torch.tensor([label], dtype=torch.float) # Shape (1,) for BCE
        }

def collate_fn(batch):
    max_c = max(b["chunks"].shape[0] for b in batch)
    c_size = batch[0]["chunks"].shape[1]
    padded, masks = [], []
    for b in batch:
        c = b["chunks"]; pad_n = max_c - c.shape[0]
        if pad_n > 0:
            c = torch.cat([c, torch.zeros(pad_n, c_size, dtype=torch.long)], 0)
        padded.append(c)
        masks.append(torch.tensor([1]*b["chunks"].shape[0]+[0]*pad_n, dtype=torch.bool))
        
    return {
        "chunks": torch.stack(padded),
        "masks": torch.stack(masks),
        "bc_feat": torch.stack([b["bc_feat"] for b in batch]),
        "struct_prior": torch.stack([b["struct_prior"] for b in batch]),
        "graphs": Batch.from_data_list([b["graph"] for b in batch]),
        "graph_pattern": torch.stack([b["graph_pattern"] for b in batch]),
        "labels": torch.stack([b["label"] for b in batch])
    }