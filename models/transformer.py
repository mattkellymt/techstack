import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import count
from muon import Muon

# ==========================================
# 1. Configuration & Global Setup
# ==========================================
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

dtype = torch.float32

# Top-level Model & Training Configuration Dictionary
config = {
    'n_layers': 2,            # Ollama GGUF: llama.block_count
    'batch_size': 4,
    'n_heads': 8,             # Ollama GGUF: llama.attention.head_count
    'n_kv_heads': 2,          # Ollama GGUF: llama.attention.head_count_kv (MHA=8, MQA=1, GQA=2)
    'head_dim': 16,           # Head Dimension (vocab_dim // n_heads)
    'seq_len': 32,            # Ollama GGUF: llama.context_length
    'vocab_dim': 128,         # 8 * 16 - Ollama GGUF: llama.embedding_length
    'kv_dim': 32,             # 2 * 16
    'hidden_dim': 256,        # Ollama GGUF: llama.feed_forward_length
    'vocab_size': 512,        # Ollama GGUF: llama.vocab_size
    'eps': 1 / 100_000,       # Ollama GGUF: llama.attention.layer_norm_rms_epsilon
    'rope_theta': 10_000.0,   # Ollama GGUF: llama.rope.freq_base
    'adamw_lr': 1e-3,
    'muon_lr': 0.02,
    'loss_target': 1 / 128,
}

# Derived Mathematical Constants
pivot = config['head_dim'] // 2
scale = 1.0 / (config['head_dim'] ** 0.5)
n_rep = config['n_heads'] // config['n_kv_heads']

def create_param(*shape):
    return nn.ParameterDict({'weight': nn.Parameter(torch.empty(*shape))})

class RMSNorm(nn.Module):
    def __init__(self, vocab_dim, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(vocab_dim))

    def forward(self, x, eps=config['eps']):
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        return (x * torch.rsqrt(variance + eps)) * self.weight

class Attention(nn.Module):
    def __init__(self, vocab_dim, kv_dim, **kwargs):
        super().__init__()
        self.q_proj = create_param(vocab_dim, vocab_dim)
        self.k_proj = create_param(vocab_dim, kv_dim)
        self.v_proj = create_param(vocab_dim, kv_dim)
        self.o_proj = create_param(vocab_dim, vocab_dim)

    def rope(self, x, cos, sin, pivot=pivot):
        x1, x2 = x[..., :pivot], x[..., pivot:]
        out = torch.empty_like(x)
        out[..., :pivot] = x1 * cos - x2 * sin
        out[..., pivot:] = x1 * sin + x2 * cos
        return out

    def forward(self, x, rope_cos, rope_sin, causal_mask):
        b, s, d = x.shape
        q = torch.matmul(x, self.q_proj.weight).reshape(b, s, config['n_heads'], config['head_dim']).transpose(1, 2)
        k = torch.matmul(x, self.k_proj.weight).reshape(b, s, config['n_kv_heads'], config['head_dim']).transpose(1, 2)
        v = torch.matmul(x, self.v_proj.weight).reshape(b, s, config['n_kv_heads'], config['head_dim']).transpose(1, 2)

        q = self.rope(q, rope_cos, rope_sin)
        k = self.rope(k, rope_cos, rope_sin)

        if n_rep > 1:
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)

        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = torch.softmax(attn + causal_mask, dim=-1)
        attn_out = torch.matmul(attn, v).transpose(1, 2).reshape(b, s, d)
        return torch.matmul(attn_out, self.o_proj.weight)

class MLP(nn.Module):
    def __init__(self, vocab_dim, hidden_dim, **kwargs):
        super().__init__()
        self.gate_proj = create_param(vocab_dim, hidden_dim)
        self.up_proj = create_param(vocab_dim, hidden_dim)
        self.down_proj = create_param(hidden_dim, vocab_dim)

    def forward(self, x):
        gate = torch.matmul(x, self.gate_proj.weight)
        up = torch.matmul(x, self.up_proj.weight)
        return torch.matmul(F.silu(gate) * up, self.down_proj.weight)

class Block(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.input_layernorm = RMSNorm(**kwargs)
        self.self_attn = Attention(**kwargs)
        self.post_attention_layernorm = RMSNorm(**kwargs)
        self.mlp = MLP(**kwargs)

    def forward(self, x, rope_cos, rope_sin, causal_mask):
        x = x + self.self_attn(self.input_layernorm(x), rope_cos, rope_sin, causal_mask)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x

class Model(nn.Module):
    def __init__(self, **config):
        super().__init__()
        self.config = config
        self.model = nn.ModuleDict({
            'embed_tokens': create_param(config['vocab_size'], config['vocab_dim']),
            'norm': RMSNorm(**config),
            'layers': nn.ModuleList(Block(**config) for _ in range(config['n_layers']))
        })
        self.lm_head = create_param(config['vocab_dim'], config['vocab_size'])

        # Precompute RoPE Frequency Tables & Causal Mask Buffers (Excluded from .pt checkpoint state_dict!)
        theta = 1.0 / (config['rope_theta'] ** (torch.arange(0, config['head_dim'], 2, dtype=torch.float32) / config['head_dim']))
        seq_idx = torch.arange(config['seq_len'], dtype=torch.float32)
        idx_theta = torch.outer(seq_idx, theta)
        self.register_buffer('rope_cos', idx_theta.cos().to(dtype), persistent=False)
        self.register_buffer('rope_sin', idx_theta.sin().to(dtype), persistent=False)
        self.register_buffer('causal_mask', torch.triu(torch.full((config['seq_len'], config['seq_len']), float('-inf'), dtype=dtype), diagonal=1), persistent=False)

    def forward(self, inputs):
        x = self.model.embed_tokens.weight[inputs]
        for layer in self.model.layers:
            x = layer(x, self.rope_cos, self.rope_sin, self.causal_mask)
        x = self.model.norm(x)
        return torch.matmul(x, self.lm_head.weight)

    def init_params(self):
        with torch.no_grad():
            for p in self.parameters():
                if p.ndim > 1:
                    nn.init.normal_(p, mean=0.0, std=0.02)
                else:
                    nn.init.ones_(p)

    def save(self, filepath):
        checkpoint = {
            'config': self.config,
            'state_dict': self.state_dict()
        }
        torch.save(checkpoint, filepath)
        print(f"Saved checkpoint to '{filepath}' ({len(checkpoint['state_dict'])} weight tensors + config).")

    def load(self, filepath, device=None):
        target_device = device or next(self.parameters()).device
        checkpoint = torch.load(filepath, map_location=target_device, weights_only=True)
        if isinstance(checkpoint, dict) and 'config' in checkpoint:
            self.config = checkpoint['config']
        sd = checkpoint.get('state_dict', checkpoint)
        self.load_state_dict(sd)
        print(f"Loaded checkpoint from '{filepath}' onto {target_device}.")

# Instantiate Model & Initialize Parameters for Training
model = Model(**config).to(device=device, dtype=dtype)
model.init_params()

# Initialize Custom Muon Optimizer
optimizer = Muon(model.parameters(), lr=config['muon_lr'], adamw_lr=config['adamw_lr'], eps=config['eps'])

# ==========================================
# 2. TRAINING STEP (Forward, Loss, Backward & Muon Optimizer)
# ==========================================
inputs = torch.randint(0, config['vocab_size'], (config['batch_size'], config['seq_len']), dtype=torch.long, device=device)
targets = torch.randint(0, config['vocab_size'], (config['batch_size'], config['seq_len']), dtype=torch.long, device=device)

print(f"--- Starting Training Loop on {device} (Muon Optimizer) ---")
for step in count(1):
    optimizer.zero_grad()

    # Forward Pass & Output Logits Computation
    logits = model(inputs)

    # Shifted Causal Loss & Backpropagation
    shift_logits = logits[:, :-1, :].reshape(-1, config['vocab_size'])
    shift_targets = targets[:, 1:].reshape(-1)
    loss = F.cross_entropy(shift_logits, shift_targets)
    loss.backward()

    # Unified Optimizer Step (Muon for 2D Matrices, AdamW for 1D Vectors)
    optimizer.step()

    loss_val = loss.item()
    print(f"Training Step {step} | Loss: {loss_val:.4f}")

    if loss_val < config['loss_target']:
        print(f"Training complete! Loss has reached the target threshold of {config['loss_target']}.")
        model.save("ollama_model.pt")
        break