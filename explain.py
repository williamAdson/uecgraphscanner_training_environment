import os
import yaml
import torch
import numpy as np
import networkx as nx
from collections import defaultdict

from src.model import HybridBinaryClassifier
from src.feature_engineering import parse_cfg_dot, bytecode_ngram_features, cfg_to_pyg, OpcodeVocab, tokenize_opcodes

DIAGNOSTIC_OPCODES = {
    0: set(), 
    1: {"CALL", "CALLCODE", "DELEGATECALL", "STATICCALL", "SSTORE", "REVERT"} # Unchecked External Calls focus
}

class TLOSExplainer:
    def __init__(self, config_path="config/config.yaml", model_path="models/checkpoints/best_model.pth", vocab_path="models/checkpoints/vocab.pth"):
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
                
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            if os.path.exists(vocab_path):
                print(f"Loading trained vocabulary matrix from {vocab_path}...")
                self.vocab = torch.load(vocab_path, weights_only=False)
                print(f"Vocabulary loaded successfully. Size: {len(self.vocab)} tokens.")
            else:
                print(f"Error: Vocabulary file not found at {vocab_path}. Make sure main.py has run and generated it.")
                import sys
                sys.exit(1)
            

            self.model = HybridBinaryClassifier(len(self.vocab), self.config).to(self.device)
            if os.path.exists(model_path):
                print(f"Loading checkpoint weights from {model_path}...")
                checkpoint = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                print(f"Warning: Checkpoint not found at {model_path}. Running with uninitialized weights.")
                
            self.model.eval()

    def set_vocab(self, trained_vocab):
        self.vocab = trained_vocab
        self.model.embedding = torch.nn.Embedding(len(trained_vocab), self.config['model']['trans_hidden'], padding_idx=0).to(self.device)

    def explain_contract(self, dot_string, bytecode_hex, opcode_str, omega_G=0.5, omega_T=0.5, K=5):
        parsed_cfg = parse_cfg_dot(dot_string)
        cfg_nodes = len(parsed_cfg["nodes"])
        cfg_edges = len(parsed_cfg["edges"])
        cfg_density = (2 * cfg_edges) / (cfg_nodes * (cfg_nodes - 1)) if cfg_nodes > 1 else 0
        
        from src.feature_engineering import analyze_graph_patterns
        gp_feats = analyze_graph_patterns(parsed_cfg["nodes"], parsed_cfg["edges"], cfg_nodes, cfg_edges, cfg_density)
        
        pyg_graph = cfg_to_pyg(parsed_cfg, self.vocab, gp_feats).to(self.device)
        chunks = tokenize_opcodes(
            opcode_str, self.vocab, 
            chunk_size=self.config['features']['chunk_size'],
            overlap=self.config['features']['chunk_overlap'],
            max_chunks=self.config['features']['max_chunks']
        ).to(self.device)
        
        # Prepare dummy batch mapping
        masks = torch.ones(chunks.shape[0], dtype=torch.bool).to(self.device)
        
        batch = {
            'graphs': pyg_graph,
            'chunks': chunks.unsqueeze(0), # Add batch dim
            'masks': masks.unsqueeze(0),
            'bc_feat': bytecode_ngram_features(bytecode_hex, dim=self.config['features']['bytecode_ngram_dim']).unsqueeze(0).to(self.device),
            'graph_pattern': torch.tensor(gp_feats, dtype=torch.float).unsqueeze(0).to(self.device),
            'struct_prior': torch.tensor([min(cfg_nodes/500.0,1), min(cfg_edges/600.0,1), min(cfg_density/2.0,1)], dtype=torch.float).unsqueeze(0).to(self.device)
        }
        
        with torch.no_grad():
            logits, tlos_hooks = self.model(batch, return_tlos=True)
            prob = torch.sigmoid(logits).item()
            pred_class = 1 if prob > 0.5 else 0
            confidence = prob if pred_class == 1 else (1.0 - prob)
            
        num_nodes = pyg_graph.num_nodes
        s_v_GAT = torch.zeros(num_nodes, device=self.device)
        
        for edge_index, alpha in zip(tlos_hooks["gat_edges"], tlos_hooks["gat_attentions"]):
            alpha_sum = alpha.sum(dim=-1) 
            target_nodes = edge_index[1] 
            s_v_GAT.index_add_(0, target_nodes, alpha_sum)
            
        s_v_GAT = s_v_GAT / (s_v_GAT.sum() + 1e-9)
        s_v_GAT = s_v_GAT.cpu().numpy()

        e_k = tlos_hooks["window_embeddings"] 
        norms = torch.norm(e_k, p=2, dim=-1)
        top_3_chunks = torch.topk(norms, min(3, len(norms))).indices.cpu().numpy()
        
        stride = self.config['features']['chunk_size'] - self.config['features']['chunk_overlap']
        P_star = set()
        for k in top_3_chunks:
            start_pc = k * stride
            P_star.update(range(start_pc, start_pc + self.config['features']['chunk_size']))
            
        s_v_T = np.zeros(num_nodes)
        global_tokens = [t.upper() for t in str(opcode_str).split() if t.strip()]
        node_ids = sorted(parsed_cfg["nodes"].keys())
        
        for idx, nid in enumerate(node_ids):
            node_opcodes = parsed_cfg["node_opseqs"].get(nid, [])
            P_v = set()
            for op in node_opcodes:
                indices = [i for i, x in enumerate(global_tokens) if x == op]
                P_v.update(indices)
            s_v_T[idx] = len(P_v.intersection(P_star)) / max(len(P_v), 1)

        s_v = np.zeros(num_nodes)
        diagnostic_set = DIAGNOSTIC_OPCODES.get(pred_class, set())
        for idx, nid in enumerate(node_ids):
            s_v[idx] = omega_G * s_v_GAT[idx] + omega_T * s_v_T[idx]
            node_ops = set(parsed_cfg["node_opseqs"].get(nid, []))
            if node_ops.intersection(diagnostic_set):
                s_v[idx] = min(1.5 * s_v[idx], 1.0)

        top_k_indices = np.argsort(s_v)[::-1][:K]
        S = {node_ids[idx] for idx in top_k_indices}
        
        adj_list = defaultdict(set)
        for src, dst in parsed_cfg["edges"]:
            adj_list[src].add(dst); adj_list[dst].add(src)
            
        S_prime = S.copy()
        for v in S: S_prime.update(adj_list[v])
            
        G_loc = nx.DiGraph()
        for nid in S_prime:
            if nid in parsed_cfg["nodes"]:
                G_loc.add_node(nid, opcodes=parsed_cfg["nodes"][nid]["opcodes"], score=float(s_v[node_ids.index(nid)]))
        for src, dst in parsed_cfg["edges"]:
            if src in S_prime and dst in S_prime: G_loc.add_edge(src, dst)
                
        return pred_class, confidence, G_loc
    
if __name__ == "__main__":
    import argparse
    import pandas as pd
    import sys

    parser = argparse.ArgumentParser(description="TLOS Smart Contract Explainability Engine")
    parser.add_argument("--csv", type=str, default="data/split/test.csv", help="Path to evaluation split CSV")
    parser.add_argument("--index", type=int, default=0, help="Row index of the contract to audit in the CSV")
    parser.add_argument("--k", type=int, default=5, help="Number of core diagnostic basic blocks to isolate")
    args = parser.parse_args()

    explainer = TLOSExplainer(
        config_path="config/config.yaml",
        model_path="models/checkpoints/best_model.pth",
        vocab_path="models/checkpoints/vocab.pth"
    )

    print(f"Reading target contract at row index {args.index} from {args.csv}...")
    df = pd.read_csv(args.csv)
    if args.index >= len(df):
        print(f"Error: Index {args.index} out of bounds for dataset of size {len(df)}.")
        sys.exit(1)
        
    row = df.iloc[args.index]
    
    c_dot = str(row[explainer.config['data']['cfg_dot_col']])
    b_hex = str(row[explainer.config['data']['bytecode_col']])
    o_str = str(row[explainer.config['data']['opcode_col']])
    ground_truth = int(row[explainer.config['data']['label_col']])

    print("\nExecuting multi-modal inference and TLOS subgraph localization...")
    pred, confidence, G_loc = explainer.explain_contract(
        dot_string=c_dot,
        bytecode_hex=b_hex,
        opcode_str=o_str,
        omega_G=0.5,   
        omega_T=0.5,   
        K=args.k       
    )

    print("\n" + "="*60)
    print("                     TLOS AUDIT REPORT                      ")
    print("="*60)
    print(f"Target Row Index:   {args.index}")
    print(f"Ground Truth Label: {ground_truth} ({'Unchecked Calls' if ground_truth == 1 else 'Benign'})")
    print(f"Model Prediction:   {pred} ({'VULNERABLE' if pred == 1 else 'CLEAN'})")
    print(f"Model Confidence:   {confidence * 100:.2f}%")
    print("-"*60)
    print(f"Induced Subgraph contains {G_loc.number_of_nodes()} nodes and {G_loc.number_of_edges()} execution paths.")
    print("\nTop Isolated Diagnostic Basic Blocks (Highest Attention Fusion Scores):")
    
    sorted_nodes = sorted(G_loc.nodes(data=True), key=lambda x: x[1].get('score', 0), reverse=True)
    for node_id, data in sorted_nodes[:args.k]:
        print(f"\n[Basic Block ID: {node_id}] -> Fusion Attention Score: {data['score']:.4f}")
        print(f"  Opcodes Snapshot: {' '.join(data['opcodes'][:10])}...")
        
        diag_matches = set(data['opcodes']).intersection({"CALL", "CALLCODE", "DELEGATECALL", "STATICCALL", "SSTORE"})
        if diag_matches:
            print(f"  ⚠️  Flagged Operations Found: {list(diag_matches)}")
            
    print("="*60)