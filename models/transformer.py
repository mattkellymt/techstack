import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import count
from muon import Muon

# ==========================================
# 1. Device, Hyperparameters & Constants
# ==========================================
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

dtype = torch.float32

# Model Architecture (All dimensions are unique powers of 2 > 1)
n_layers = 2            # Ollama GGUF: llama.block_count
batch_size = 4
n_heads = 8             # Ollama GGUF: llama.attention.head_count
n_kv_heads = 2          # Ollama GGUF: llama.attention.head_count_kv (MHA=8, MQA=1, GQA=2)
head_dim = 16           # Head Dimension (vocab_dim // n_heads)
seq_len = 32            # Ollama GGUF: llama.context_length
vocab_dim = n_heads * head_dim  # 128 (8 * 16) - Ollama GGUF: llama.embedding_length
kv_dim = n_kv_heads * head_dim  # 32 (2 * 16)
hidden_dim = 256        # Ollama GGUF: llama.feed_forward_length
vocab_size = 512        # Ollama GGUF: llama.vocab_size
eps = 1 / 100_000       # Ollama GGUF: llama.attention.layer_norm_rms_epsilon
rope_theta = 10_000.0   # Ollama GGUF: llama.rope.freq_base

# Derived Mathematical Constants
pivot = head_dim // 2
scale = 1.0 / (head_dim ** 0.5)
n_rep = n_heads // n_kv_heads

# Training & Optimizer Hyperparameters
adamw_lr = 1e-3
muon_lr = 0.02
loss_target = 1 / 128

# ==========================================
# 2. Native PyTorch Model Architecture
# ==========================================
class LlamaLayer(nn.Module):
    """
    Standard PyTorch LLaMA Layer Module matching Ollama GGUF key naming.
    """
    def __init__(self, vocab_dim, kv_dim, hidden_dim):
        super().__init__()
        self.input_layernorm = nn.ParameterDict({'weight': nn.Parameter(torch.ones(vocab_dim))})
        self.post_attention_layernorm = nn.ParameterDict({'weight': nn.Parameter(torch.ones(vocab_dim))})

        self.self_attn = nn.ParameterDict({
            'q_proj': nn.ParameterDict({'weight': nn.Parameter(torch.randn(vocab_dim, vocab_dim) * 0.02)}),
            'k_proj': nn.ParameterDict({'weight': nn.Parameter(torch.randn(vocab_dim, kv_dim) * 0.02)}),
            'v_proj': nn.ParameterDict({'weight': nn.Parameter(torch.randn(vocab_dim, kv_dim) * 0.02)}),
            'o_proj': nn.ParameterDict({'weight': nn.Parameter(torch.randn(vocab_dim, vocab_dim) * 0.02)}),
        })

        self.mlp = nn.ParameterDict({
            'gate_proj': nn.ParameterDict({'weight': nn.Parameter(torch.randn(vocab_dim, hidden_dim) * 0.02)}),
            'up_proj': nn.ParameterDict({'weight': nn.Parameter(torch.randn(vocab_dim, hidden_dim) * 0.02)}),
            'down_proj': nn.ParameterDict({'weight': nn.Parameter(torch.randn(hidden_dim, vocab_dim) * 0.02)}),
        })

class LlamaModel(nn.Module):
    """
    Standard PyTorch LLaMA Model Module with zero custom weight boilerplate.
    Uses native model.state_dict() and model.load_state_dict().
    """
    def __init__(self, n_layers, vocab_size, vocab_dim, kv_dim, hidden_dim):
        super().__init__()
        self.model = nn.ModuleDict({
            'embed_tokens': nn.ParameterDict({'weight': nn.Parameter(torch.randn(vocab_size, vocab_dim) * 0.02)}),
            'norm': nn.ParameterDict({'weight': nn.Parameter(torch.ones(vocab_dim))}),
            'layers': nn.ModuleList([LlamaLayer(vocab_dim, kv_dim, hidden_dim) for _ in range(n_layers)])
        })
        self.lm_head = nn.ParameterDict({'weight': nn.Parameter(torch.randn(vocab_dim, vocab_size) * 0.02)})

# Instantiate Native PyTorch Model
model = LlamaModel(n_layers, vocab_size, vocab_dim, kv_dim, hidden_dim).to(device=device, dtype=dtype)

# Precomputed RoPE Frequency Tables (Cos & Sin)
theta = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
seq_idx = torch.arange(seq_len, dtype=torch.float32, device=device)
idx_theta = torch.outer(seq_idx, theta)  # Shape: (seq_len, pivot)
rope_cos, rope_sin = idx_theta.cos().to(dtype), idx_theta.sin().to(dtype)

# Precomputed Causal Mask Buffer
causal_mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), dtype=dtype, device=device), diagonal=1)

def rms_norm(x, w, eps=eps):
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    return (x * torch.rsqrt(variance + eps)) * w

def apply_rope(x, cos, sin, pivot=pivot):
    x1, x2 = x[..., :pivot], x[..., pivot:]
    out = torch.empty_like(x)
    out[..., :pivot] = x1 * cos - x2 * sin
    out[..., pivot:] = x1 * sin + x2 * cos
    return out

# Initialize Custom Muon Optimizer
optimizer = Muon(model.parameters(), lr=muon_lr, adamw_lr=adamw_lr, eps=eps)

# ==========================================
# 3. TRAINING STEP (Forward, Loss, Backward & Muon Optimizer)
# ==========================================
inputs = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long, device=device)
targets = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long, device=device)

print(f"--- Starting Training Loop on {device} (Muon Optimizer) ---")
for step in count(1):
    optimizer.zero_grad()

    # 3.1 Input Lookup
    x = model.model.embed_tokens.weight[inputs]

    # 3.2 Transformer Layers Loop
    for layer in model.model.layers:
        # Attention Pre-Norm & Q, K, V Projections
        x_norm = rms_norm(x, layer.input_layernorm.weight)
        q = torch.matmul(x_norm, layer.self_attn.q_proj.weight).reshape(batch_size, seq_len, n_heads, head_dim).transpose(1, 2)
        k = torch.matmul(x_norm, layer.self_attn.k_proj.weight).reshape(batch_size, seq_len, n_kv_heads, head_dim).transpose(1, 2)
        v = torch.matmul(x_norm, layer.self_attn.v_proj.weight).reshape(batch_size, seq_len, n_kv_heads, head_dim).transpose(1, 2)

        # Apply RoPE to Q and K
        q = apply_rope(q, rope_cos, rope_sin)
        k = apply_rope(k, rope_cos, rope_sin)

        # GQA / MQA Head Expansion
        if n_rep > 1:
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)

        # Parallel Causal Attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = torch.softmax(attn + causal_mask, dim=-1)

        # Output Projection & Attention Residual
        attn_out = torch.matmul(attn, v).transpose(1, 2).reshape(batch_size, seq_len, vocab_dim)
        x = x + torch.matmul(attn_out, layer.self_attn.o_proj.weight)

        # SwiGLU FFN Pre-Norm, Projection & FFN Residual
        x_norm = rms_norm(x, layer.post_attention_layernorm.weight)
        gate = torch.matmul(x_norm, layer.mlp.gate_proj.weight)
        up = torch.matmul(x_norm, layer.mlp.up_proj.weight)
        ffn_out = torch.matmul(F.silu(gate) * up, layer.mlp.down_proj.weight)
        x = x + ffn_out

    # 3.3 Final Norm & Unembedding
    x_norm = rms_norm(x, model.model.norm.weight)
    logits = torch.matmul(x_norm, model.lm_head.weight)

    # 3.4 Shifted Causal Loss & Backpropagation
    shift_logits = logits[:, :-1, :].reshape(-1, vocab_size)
    shift_targets = targets[:, 1:].reshape(-1)
    loss = F.cross_entropy(shift_logits, shift_targets)
    loss.backward()

    # 3.5 Unified Optimizer Step (Muon for 2D/3D Matrices, AdamW for 1D Vectors)
    optimizer.step()

    loss_val = loss.item()
    print(f"Training Step {step} | Loss: {loss_val:.4f}")

    if loss_val < loss_target:
        print(f"Training complete! Loss has reached the target threshold of {loss_target}.")
        torch.save(model.state_dict(), "ollama_model.pt")
        print(f"Saved checkpoint to 'ollama_model.pt' ({len(model.state_dict())} weight tensors).")
        break