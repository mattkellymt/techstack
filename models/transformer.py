import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import count

# Standard precision for training parameter storage
dtype = torch.float32

# ==========================================
# 1. Hyperparameters & Constants
# ==========================================
batch_size = 2
dim = 256
n_heads = 4
head_dim = dim // n_heads
hidden_dim = int(dim * 8 / 3)  # 682 (LLaMA SwiGLU hidden dim)
vocab_size = 4069
seq_len = 120

pivot = head_dim // 2
scale = 1.0 / (head_dim ** 0.5)

# ==========================================
# 2. Parameters & Precomputed Buffers
# ==========================================
# Embedding & Attention Weights
w_emb = nn.Parameter(torch.empty((vocab_size, dim), dtype=dtype))
wq = nn.Parameter(torch.empty((dim, dim), dtype=dtype))
wk = nn.Parameter(torch.empty((dim, dim), dtype=dtype))
wv = nn.Parameter(torch.empty((dim, dim), dtype=dtype))
wo = nn.Parameter(torch.empty((dim, dim), dtype=dtype))
w_unemb = nn.Parameter(torch.empty((dim, vocab_size), dtype=dtype))

# SwiGLU FFN Weights
w_gate = nn.Parameter(torch.empty((dim, hidden_dim), dtype=dtype))
w_up = nn.Parameter(torch.empty((dim, hidden_dim), dtype=dtype))
w_down = nn.Parameter(torch.empty((hidden_dim, dim), dtype=dtype))

# RMSNorm Weights
norm_attn = nn.Parameter(torch.ones(dim, dtype=dtype))
norm_ffn = nn.Parameter(torch.ones(dim, dtype=dtype))
norm_final = nn.Parameter(torch.ones(dim, dtype=dtype))

# Short-circuit Weight Initialization
params = [w_emb, wq, wk, wv, wo, w_unemb, w_gate, w_up, w_down, norm_attn, norm_ffn, norm_final]
for p in params:
    if p.ndim <= 1:
        continue
    nn.init.normal_(p, mean=0.0, std=0.02)

# Optimizer Setup
optimizer = torch.optim.AdamW(params, lr=1e-3)

# Precomputed RoPE Frequency Tables (Cos & Sin)
theta = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
seq_idx = torch.arange(seq_len, dtype=torch.float32)
idx_theta = torch.outer(seq_idx, theta)  # Shape: (seq_len, pivot)
rope_cos, rope_sin = idx_theta.cos().to(dtype), idx_theta.sin().to(dtype)

# Precomputed Causal Mask Buffer
causal_mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), dtype=dtype), diagonal=1)

def rms_norm(x, w, eps=1e-5):
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    return (x * torch.rsqrt(variance + eps)) * w

def apply_rope(x, cos, sin, pivot=pivot):
    x1, x2 = x[..., :pivot], x[..., pivot:]
    out = torch.empty_like(x)
    out[..., :pivot] = x1 * cos - x2 * sin
    out[..., pivot:] = x1 * sin + x2 * cos
    return out

# ==========================================
# 3. TRAINING STEP (Forward, Loss, Backward & Optimizer)
# ==========================================
inputs = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)
targets = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)
loss_target = 1 / 128

print("--- Starting Training Loop ---")
for step in count(1):
    optimizer.zero_grad()

    # 3.1 Input Lookup
    x = w_emb[inputs]

    # 3.2 Pre-Norm & Q, K, V Projections
    x_norm = rms_norm(x, norm_attn)
    q = torch.matmul(x_norm, wq).reshape(batch_size, seq_len, n_heads, head_dim).transpose(1, 2)
    k = torch.matmul(x_norm, wk).reshape(batch_size, seq_len, n_heads, head_dim).transpose(1, 2)
    v = torch.matmul(x_norm, wv).reshape(batch_size, seq_len, n_heads, head_dim).transpose(1, 2)

    # 3.3 Apply RoPE to Q and K
    q = apply_rope(q, rope_cos, rope_sin)
    k = apply_rope(k, rope_cos, rope_sin)

    # 3.4 Parallel Causal Attention
    attn = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn = torch.softmax(attn + causal_mask, dim=-1)

    # 3.5 Output Projection & Attention Residual
    attn_out = torch.matmul(attn, v).transpose(1, 2).reshape(batch_size, seq_len, dim)
    x = x + torch.matmul(attn_out, wo)

    # 3.6 SwiGLU FFN Pre-Norm, Projection & FFN Residual
    x_norm = rms_norm(x, norm_ffn)
    gate = torch.matmul(x_norm, w_gate)
    up = torch.matmul(x_norm, w_up)
    ffn_out = torch.matmul(F.silu(gate) * up, w_down)
    x = x + ffn_out

    # 3.7 Final Norm & Unembedding
    x_norm = rms_norm(x, norm_final)
    logits = torch.matmul(x_norm, w_unemb)

    # 3.8 Shifted Causal Loss, Backpropagation & Optimizer Step
    shift_logits = logits[:, :-1, :].reshape(-1, vocab_size)
    shift_targets = targets[:, 1:].reshape(-1)
    loss = F.cross_entropy(shift_logits, shift_targets)
    loss.backward()
    optimizer.step()

    print(f"Training Step {step + 1} | Loss: {loss.item():.4f}")

    if loss.item() < loss_target:
        print(f"Training complete! Loss has reached the target threshold of {loss_target}.")
        break