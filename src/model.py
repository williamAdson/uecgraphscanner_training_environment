import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

class HybridBinaryClassifier(nn.Module):
    def __init__(self, vocab_size, config):
        super().__init__()
        self.mc = config['model']
        self.fc = config['features']
        
        # Sequence Modality (Transformer)
        self.embedding = nn.Embedding(vocab_size, self.mc['trans_hidden'], padding_idx=0)
        trans_layer = nn.TransformerEncoderLayer(
            d_model=self.mc['trans_hidden'], 
            nhead=self.mc['trans_heads'], 
            dropout=self.mc['trans_dropout'], 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(trans_layer, num_layers=self.mc['trans_layers'])
        
        # Graph Modality (GAT)
        in_dim = 184 
        self.gat1 = GATConv(in_dim, self.mc['gat_hidden'], heads=self.mc['gat_heads'], edge_dim=self.mc['gat_edge_dim'])
        self.gat2 = GATConv(self.mc['gat_hidden']*self.mc['gat_heads'], self.mc['gat_hidden'], heads=1)
        
        # Fusion & Final Classification
        fusion_dim = self.mc['trans_hidden'] + self.mc['gat_hidden'] + self.fc['bytecode_ngram_dim'] + 32 + 3 
        
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.GELU(),
            nn.Dropout(self.mc['fusion_dropout']),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(self.mc['fusion_dropout']),
            nn.Linear(64, 1) # Binary Output Logit
        )

    def forward(self, batch, return_tlos=False):
        # --- Transformer Branch ---
        b, num_chunks, chunk_len = batch["chunks"].shape
        flat_chunks = batch["chunks"].view(b * num_chunks, chunk_len)
        mask = (flat_chunks == 0)
        
        emb = self.embedding(flat_chunks)
        window_embeddings = self.transformer(emb, src_key_padding_mask=mask) 
        
        # Extract <CLS> token (index 0) and unflatten
        cls_tokens = window_embeddings[:, 0, :].view(b, num_chunks, -1)
        chunk_mask = batch["masks"].unsqueeze(-1).float()
        seq_rep = (cls_tokens * chunk_mask).sum(dim=1) / (chunk_mask.sum(dim=1) + 1e-9)

        # --- GAT Branch ---
        g = batch["graphs"]
        
        if return_tlos:
            x, (edge_index_1, alpha_1) = self.gat1(g.x, g.edge_index, g.edge_attr, return_attention_weights=True)
            x = F.gelu(x)
            x, (edge_index_2, alpha_2) = self.gat2(x, edge_index_1, return_attention_weights=True)
            graph_rep = global_mean_pool(x, g.batch)
        else:
            x = self.gat1(g.x, g.edge_index, g.edge_attr)
            x = F.gelu(x)
            x = self.gat2(x, g.edge_index)
            graph_rep = global_mean_pool(x, g.batch)

        fused = torch.cat([
            seq_rep, 
            graph_rep, 
            batch["bc_feat"], 
            batch["graph_pattern"], 
            batch["struct_prior"]
        ], dim=1)
        
        logits = self.classifier(fused)
        
        if return_tlos:
            return logits, {
                "window_embeddings": cls_tokens.squeeze(0) if b == 1 else cls_tokens, 
                "gat_attentions": [alpha_1, alpha_2],
                "gat_edges": [edge_index_1, edge_index_2]
            }
            
        return logits