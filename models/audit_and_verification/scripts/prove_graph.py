import sys
import os
import torch
import torch._dynamo as dynamo
from transformers import AutoModelForCausalLM

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from architecture import Model

def get_op_list(gm: torch.fx.GraphModule):
    ops = []
    for node in gm.graph.nodes:
        if node.op == "call_function" or node.op == "call_method":
            ops.append(str(node.target))
    return ops

def main():
    print("Loading models...")
    dtype = torch.bfloat16
    device = torch.device("cpu")
    
    # Load HuggingFace model
    hf_model = AutoModelForCausalLM.from_pretrained("unsloth/Llama-3.2-1B-Instruct", torch_dtype=dtype)
    hf_model.eval()
    
    # Load Custom Architecture
    DEFAULT_CONFIG = {
        "n_layers": 16,
        "n_heads": 32,
        "n_kv_heads": 8,
        "head_dim": 64,
        "seq_len": 2048,
        "vocab_dim": 2048,
        "kv_dim": 512,
        "hidden_dim": 8192,
        "vocab_size": 128256,
        "eps": 1e-5,
        "rope_theta": 500000.0,
        "rope_scaling": {
            "factor": 32.0,
            "high_freq_factor": 4.0,
            "low_freq_factor": 1.0,
            "original_max_position_embeddings": 8192,
            "rope_type": "llama3"
        }
    }
    custom_model = Model(**DEFAULT_CONFIG).to(dtype=dtype, device=device)
    custom_model.eval()
    
    print("Tracing HuggingFace Model with Dynamo...")
    # Dynamo trace requires sample inputs
    sample_input = torch.tensor([[128000, 11, 314]], dtype=torch.long, device=device)
    
    # Wrap HF model to avoid complex dictionary returns and kwargs issues during trace
    class HFWrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, x):
            return self.m(x).logits
            
    hf_wrap = HFWrap(hf_model)
    
    try:
        hf_gm, _ = dynamo.export(hf_wrap, sample_input)
        print("HF Trace Successful.")
    except Exception as e:
        print(f"HF Trace Failed: {e}")
        return

    print("Tracing Custom Architecture with Dynamo...")
    class CustomWrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, x):
            return self.m(x)
            
    custom_wrap = CustomWrap(custom_model)
    try:
        custom_gm, _ = dynamo.export(custom_wrap, sample_input)
        print("Custom Trace Successful.")
    except Exception as e:
        print(f"Custom Trace Failed: {e}")
        return

    # Write graphs to text files
    os.makedirs("audit_and_verification/reports", exist_ok=True)
    with open("audit_and_verification/reports/hf_graph.txt", "w") as f:
        f.write(str(hf_gm.graph))
        
    with open("audit_and_verification/reports/custom_graph.txt", "w") as f:
        f.write(str(custom_gm.graph))
        
    print("\nSaved FX Graphs to 'audit_and_verification/reports/'")
    print("Graph node count comparison:")
    print(f"  HF Model Nodes:     {len(hf_gm.graph.nodes)}")
    print(f"  Custom Model Nodes: {len(custom_gm.graph.nodes)}")
    
if __name__ == "__main__":
    main()
