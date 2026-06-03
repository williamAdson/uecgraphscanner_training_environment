import re
import json
import math
import numpy as np
import torch
from collections import defaultdict, Counter
from torch_geometric.data import Data


# OPCODES & DIMENSIONS
TERMINAL_OPCODES = {"RETURN","REVERT","STOP","INVALID","SELFDESTRUCT"}
CALL_OPCODES     = {"CALL","CALLCODE","DELEGATECALL","STATICCALL"}
TIME_OPCODES     = {"TIMESTAMP","NUMBER","BLOCKHASH","COINBASE","DIFFICULTY","GASLIMIT"}
ARITH_OPCODES    = {"ADD","SUB","MUL","DIV","MOD","EXP","MULMOD","ADDMOD"}
CALLER_OPCODES   = {"CALLER","ORIGIN"}
LOG_OPCODES      = {"LOG0","LOG1","LOG2","LOG3","LOG4"}
BALANCE_OPCODES  = {"BALANCE","SELFBALANCE"}

TOP_OPCODES = [
    "POP","PUSH1","PUSH2","PUSH3","PUSH4","DUP1","DUP2","SWAP1","SWAP2",
    "MSTORE","MLOAD","SSTORE","SLOAD","ADD","SUB","MUL","DIV","EQ","LT",
    "GT","AND","OR","XOR","NOT","ISZERO","JUMP","JUMPI","JUMPDEST","CALL",
    "CALLCODE","DELEGATECALL","STATICCALL","RETURN","REVERT","STOP","SELFDESTRUCT",
    "TIMESTAMP","NUMBER","BLOCKHASH","COINBASE","DIFFICULTY","GASLIMIT",
    "CALLER","ORIGIN","ADDRESS","BALANCE","GAS","LOG0","LOG1","LOG2",
    "MULMOD","ADDMOD","EXP","MOD","SELFBALANCE","CHAINID",
]
NODE_FEAT_DIM = len(TOP_OPCODES) + 64 + 32  # hist + sinusoidal-seq + structural


# VOCABULARY & BASIC TOKENIZATION
class OpcodeVocab:
    PAD, UNK, CLS = "<PAD>","<UNK>","<CLS>"

    def __init__(self):
        self.token2id = {self.PAD:0,self.UNK:1,self.CLS:2}
        self.id2token = {0:self.PAD,1:self.UNK,2:self.CLS}
        self.freq     = defaultdict(int)

    def count(self, tokens):
        for t in tokens: self.freq[t.upper()] += 1

    def build(self, min_freq=3):
        for tok, cnt in sorted(self.freq.items()):
            if cnt >= min_freq and tok not in self.token2id:
                i = len(self.token2id)
                self.token2id[tok] = i; self.id2token[i] = tok
        print(f"[Vocab] {len(self.token2id)} tokens (min_freq={min_freq})")

    def encode(self, tok):
        return self.token2id.get(tok.upper(), self.token2id[self.UNK])

    def __len__(self): return len(self.token2id)


def tokenize_opcodes(opcode_str, vocab, chunk_size=256, overlap=32, max_chunks=12):
    tokens = [t.upper() for t in str(opcode_str).split() if t.strip()]
    ids    = [vocab.encode("<CLS>")] + [vocab.encode(t) for t in tokens]
    stride = chunk_size - overlap
    chunks = []
    start  = 0
    while start < len(ids):
        chunk = ids[start:start+chunk_size]
        chunk += [0]*(chunk_size - len(chunk))
        chunks.append(chunk)
        start += stride
        if len(chunks) >= max_chunks: break
    if not chunks:
        chunks = [[0]*chunk_size]
    return torch.tensor(chunks, dtype=torch.long)


def bytecode_ngram_features(bytecode_hex: str, dim=64) -> torch.Tensor:
    try:
        byt = bytes.fromhex(bytecode_hex.strip().replace("0x","").replace(" ",""))
    except Exception:
        return torch.zeros(dim)

    if len(byt) < 2:
        return torch.zeros(dim)

    vec = np.zeros(dim, dtype=np.float32)
    for i in range(len(byt)-1):
        idx = (byt[i]*256 + byt[i+1]) % dim
        vec[idx] += 1.0

    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return torch.tensor(vec, dtype=torch.float)


# CFG DOT PARSER & PATTERN ANALYSIS
def parse_cfg_dot(dot_string: str) -> dict:
    nodes, edges, node_opsets, node_opseqs = {}, [], {}, {}

    node_pat = re.compile(r'([\w"\'\-]+)\s*\[([^\]]+)\]', re.DOTALL)
    
    string_id_to_int = {}
    def get_node_idx(raw_str):
        clean_str = raw_str.strip('"\' ')
        if clean_str not in string_id_to_int:
            if clean_str.isdigit():
                string_id_to_int[clean_str] = int(clean_str)
            else:
                string_id_to_int[clean_str] = len(string_id_to_int)
        return string_id_to_int[clean_str]

    for m in node_pat.finditer(str(dot_string)):
        raw_id   = m.group(1)
        attr_str = m.group(2)
        
        # Ignore structural keyword indicators caught by regex boundaries
        if raw_id.lower() in ["digraph", "subgraph", "strict"]:
            continue
            
        nid = get_node_idx(raw_id)

        label_m = re.search(r'label\s*=\s*"(.*?)"', attr_str, re.DOTALL)
        label   = label_m.group(1) if label_m else ""

        opcodes = []
        for line in re.split(r'\\n|\\l|\n', label):
            line = line.strip().strip('"').strip()
            if not line: continue
            parts = line.split()
            for p in parts:
                p = p.upper().replace(":", "").replace(";", "")
                if p and p.isalpha() and len(p) >= 2:
                    opcodes.append(p)

        nodes[nid] = {
            "opcodes":       opcodes,
            "is_entry":      int("entry" in attr_str.lower() or "gold" in attr_str or nid == 0),
            "is_terminal":   int("double" in attr_str or any(o in opcodes for o in TERMINAL_OPCODES)),
            "is_dispatcher": int("diamond" in attr_str or "dispatch" in attr_str.lower()),
        }
        node_opsets[nid]  = set(opcodes)
        node_opseqs[nid]  = opcodes

    edge_pat = re.compile(r'([\w"\'\-]+)\s*->\s*([\w"\'\-]+)')
    for m in edge_pat.finditer(str(dot_string)):
        raw_src, raw_dst = m.group(1), m.group(2)
        if raw_src.lower() in ["digraph", "subgraph"] or raw_dst.lower() in ["digraph", "subgraph"]:
            continue
        try:
            src = get_node_idx(raw_src)
            dst = get_node_idx(raw_dst)
            edges.append((src, dst))
        except Exception:
            continue

    if not nodes:
        nodes[0] = {
            "opcodes": [], "is_entry": 1, "is_terminal": 0, "is_dispatcher": 0
        }
        node_opsets[0] = set()
        node_opseqs[0] = []

    edge_feats = {}
    for s, d in edges:
        so = node_opsets.get(s, set())
        do = node_opsets.get(d, set())
        edge_feats[(s,d)] = {
            "call_to_sstore":   int(bool(so&CALL_OPCODES)   and "SSTORE" in do),
            "sload_to_call":    int("SLOAD" in so           and bool(do&CALL_OPCODES)),
            "time_to_jumpi":    int(bool(so&TIME_OPCODES)   and "JUMPI" in do),
            "arith_to_sstore":  int(bool(so&ARITH_OPCODES)  and "SSTORE" in do),
            "call_to_arith":    int(bool(so&CALL_OPCODES)   and bool(do&ARITH_OPCODES)),
            "caller_to_jumpi":  int(bool(so&CALLER_OPCODES) and "JUMPI" in do),
            "arith_no_guard":   int(bool(so&ARITH_OPCODES)  and not bool(do&{"REVERT","INVALID","ISZERO"})),
            "back_edge_hint":   0,
        }

    return {"nodes":nodes,"edges":edges, "edge_patterns":edge_feats, "node_opsets":node_opsets,"node_opseqs":node_opseqs}

def analyze_graph_patterns(nodes, edges, cfg_nodes=0, cfg_edges=0, cfg_density=0.0) -> list:
    node_opsets = {nid: set(d["opcodes"]) for nid, d in nodes.items()}
    all_opcodes = [op for d in nodes.values() for op in d["opcodes"]]
    all_opset   = set(all_opcodes)
    n_nodes     = max(len(nodes), 1)

    adj_out = defaultdict(set); adj_in = defaultdict(set)
    for s, d in edges:
        adj_out[s].add(d); adj_in[d].add(s)

    call_nodes  = {n for n,o in node_opsets.items() if o&CALL_OPCODES}
    sstore_nodes = {n for n,o in node_opsets.items() if "SSTORE" in o}
    sload_nodes  = {n for n,o in node_opsets.items() if "SLOAD"  in o}
    time_nodes   = {n for n,o in node_opsets.items() if o&TIME_OPCODES}
    jumpi_nodes  = {n for n,o in node_opsets.items() if "JUMPI" in o}
    arith_nodes  = {n for n,o in node_opsets.items() if o&ARITH_OPCODES}

    call_to_sstore = sum(1 for s,d in edges if s in call_nodes and d in sstore_nodes)
    sload_to_call  = sum(1 for s,d in edges if s in sload_nodes and d in call_nodes)
    time_to_jumpi  = sum(1 for s,d in edges if s in time_nodes  and d in jumpi_nodes)
    caller_branch  = sum(1 for s,d in edges if node_opsets.get(s,set())&CALLER_OPCODES and "JUMPI" in node_opsets.get(d,set()))
    
    arith_no_chk   = sum(1 for n in arith_nodes
                         if not (node_opsets[n]&{"ISZERO","LT","GT","SLT","SGT"})
                         and not any(node_opsets.get(nb,set())&{"REVERT","INVALID"} for nb in adj_out.get(n,set())))
    ts_cmp = sum(1 for n,o in node_opsets.items() if "TIMESTAMP" in o and (o&{"EQ","LT","GT","SLT","SGT"}))

    visited = set()
    back    = [0]
    for start_node in ([n for n, d in nodes.items() if d.get("is_entry")] or list(nodes.keys())[:1]):
        if start_node in visited: continue
        stack = [(start_node, False)]
        rec   = set()
        while stack:
            node, returning = stack.pop()
            if returning:
                rec.discard(node)
                continue
            if node in visited: continue
            visited.add(node)
            rec.add(node)
            stack.append((node, True))
            for nb in adj_out.get(node, set()):
                if nb not in visited:
                    stack.append((nb, False))
                elif nb in rec:
                    back[0] += 1

    n_edges  = len(edges)
    avg_deg  = sum(len(v) for v in adj_out.values()) / n_nodes
    total    = max(len(all_opcodes), 1)
    op_counts = Counter(all_opcodes)
    def freq(ops): return sum(op_counts.get(o, 0) for o in ops) / total

    feats = [
        min(call_to_sstore/n_nodes,1), min(sload_to_call /n_nodes,1), min(back[0]/n_nodes,1),
        min(len(call_nodes)/n_nodes,1), min(len(sstore_nodes)/n_nodes,1), freq(CALL_OPCODES), freq({"SSTORE"}), freq({"SLOAD"}),
        min(time_to_jumpi/n_nodes,1), min(ts_cmp/n_nodes,1), freq(TIME_OPCODES), min(len(time_nodes)/n_nodes,1), min(len(jumpi_nodes)/n_nodes,1),
        min(arith_no_chk/n_nodes,1), freq(ARITH_OPCODES), int("EXP" in all_opset), int("MULMOD" in all_opset or "ADDMOD" in all_opset), min(len(arith_nodes)/n_nodes,1),
        int("DELEGATECALL" in all_opset), int("CALLCODE" in all_opset), min(caller_branch/n_nodes,1), int("ORIGIN" in all_opset), min(len(call_nodes)/n_nodes,1),
        int("BLOCKHASH" in all_opset), int("COINBASE" in all_opset), int("SELFDESTRUCT" in all_opset), min(len(jumpi_nodes)/n_nodes,1),
        min(cfg_nodes/500.0,1), min(cfg_edges/600.0,1), min(cfg_density/2.0,1), min(avg_deg/4.0,1), min(n_nodes/100.0,1),
    ]
    feats = feats[:32] + [0.0]*(32-len(feats[:32]))
    return feats


# GRAPH NODE REPRESENTATIONS & CONVERSION
def get_opcode_histogram(opcodes):
    hist = torch.zeros(len(TOP_OPCODES))
    ops  = set(opcodes)
    for i,op in enumerate(TOP_OPCODES):
        hist[i] = float(op in ops)
    return hist


def encode_opcode_sequence(opcodes, vocab, max_len=64):
    dim = 64
    if not opcodes: return torch.zeros(dim)
    ids     = [vocab.encode(op)/max(len(vocab),1) for op in opcodes[:max_len]]
    n       = len(ids)
    w       = torch.exp(-torch.arange(n,dtype=torch.float)*0.05)
    w       = w/w.sum()
    id_t    = torch.tensor(ids, dtype=torch.float)
    freqs   = torch.arange(1,dim//2+1,dtype=torch.float)*math.pi
    angles  = id_t.unsqueeze(1)*freqs.unsqueeze(0)
    feats   = torch.cat([torch.sin(angles),torch.cos(angles)],dim=1)
    return (feats*w.unsqueeze(1)).sum(0)


def node_structural_features(nid, data, node_opsets, adj_out, adj_in):
    ops    = node_opsets.get(nid, set())
    out_d  = len(adj_out.get(nid,set()))
    in_d   = len(adj_in.get(nid,set()))
    feats  = [
        data.get("is_entry",0), data.get("is_terminal",0), data.get("is_dispatcher",0),
        min(len(data["opcodes"])/100,1), min(out_d/5,1), min(in_d/5,1),
        int(bool(ops&CALL_OPCODES)), int("SSTORE" in ops), int("SLOAD" in ops),
        int(bool(ops&TIME_OPCODES)), int("JUMPI" in ops),
        int(bool(ops&CALLER_OPCODES)), int(bool(ops&ARITH_OPCODES)),
        int("EXP" in ops), int("DELEGATECALL" in ops), int("CALLCODE" in ops),
        int("SELFDESTRUCT" in ops), int("ORIGIN" in ops), int("BLOCKHASH" in ops),
        int(bool(ops&{"ISZERO","EQ","LT","GT","SLT","SGT"})),
        int("REVERT" in ops), int("RETURN" in ops),
        int(out_d>1), int(in_d>1), int(out_d==0), int(in_d==0),
        int(bool(ops&CALL_OPCODES) and "SSTORE" in ops),
        int(bool(ops&TIME_OPCODES) and "JUMPI" in ops),
        min(len(ops&ARITH_OPCODES)/4,1),
        int(bool(ops&LOG_OPCODES)), int(bool(ops&BALANCE_OPCODES)), int("GAS" in ops),
    ]
    return torch.tensor(feats[:32], dtype=torch.float)


def cfg_to_pyg(parsed, vocab, graph_pattern_feats):
    nodes    = parsed["nodes"]
    edges    = parsed["edges"]
    node_ids = sorted(nodes.keys())
    id2idx   = {nid:i for i,nid in enumerate(node_ids)}
    n        = len(node_ids)

    adj_out = defaultdict(set); adj_in = defaultdict(set)
    for s,d in edges:
        adj_out[s].add(d); adj_in[d].add(s)

    x_list = []
    for nid in node_ids:
        data   = nodes[nid]
        hist   = get_opcode_histogram(data["opcodes"])
        seq    = encode_opcode_sequence(data["opcodes"], vocab)
        struct = node_structural_features(nid, data, parsed["node_opsets"], adj_out, adj_in)
        x_list.append(torch.cat([hist, seq, struct]))

    if not x_list:
        x = torch.zeros((1, 152), dtype=torch.float) # 56 + 64 + 32 = 152 base dimensions
        n = 1
    else:
        x = torch.stack(x_list)
        
    gp = torch.tensor(graph_pattern_feats, dtype=torch.float).unsqueeze(0).expand(n,-1)
    x  = torch.cat([x, gp], dim=1) # 152 + 32 = 184 Total initialized dimensions

    src_l, dst_l, ef_l = [], [], []
    for s,d in edges:
        if s in id2idx and d in id2idx:
            src_l.append(id2idx[s]); dst_l.append(id2idx[d])
            pat = parsed["edge_patterns"].get((s,d),{})
            
            ef_l.append(torch.tensor([
                pat.get("call_to_sstore",0), 
                pat.get("sload_to_call",0),
                pat.get("time_to_jumpi",0),  
                pat.get("arith_to_sstore",0),
                pat.get("call_to_arith",0),  
                pat.get("caller_to_jumpi",0), 
                pat.get("arith_no_guard",0), 
                pat.get("back_edge_hint",0),
            ], dtype=torch.float))

    if src_l:
        edge_index = torch.tensor([src_l,dst_l], dtype=torch.long)
        edge_attr  = torch.stack(ef_l)
    else:
        edge_index = torch.zeros((2,0), dtype=torch.long)
        edge_attr  = torch.zeros((0,8), dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=n)