import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import count

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
n_layers = 2
batch_size = 4
n_heads = 8
head_dim = 16
seq_len = 32
vocab_dim = n_heads * head_dim  # 128 (8 * 16)
hidden_dim = 256
vocab_size = 512
eps = 1 / 100_000  # Canonical LLaMA RMSNorm epsilon (1e-5)

# Derived Mathematical Constants
pivot = head_dim // 2
scale = 1.0 / (head_dim ** 0.5)

# Training Hyperparameters
lr = 1e-3
loss_target = 1 / 128

# ==========================================
# 2. Parameters & Precomputed Buffers
# ==========================================
# Single Model-Level Weights (Embeddings & Unembedding)
w_emb = nn.Parameter(torch.empty((vocab_size, vocab_dim), dtype=dtype, device=device))
w_unemb = nn.Parameter(torch.empty((vocab_dim, vocab_size), dtype=dtype, device=device))
norm_final = nn.Parameter(torch.ones(vocab_dim, dtype=dtype, device=device))

# Multi-Layer Stacked Weights (3D Tensors prepended with n_layers)
wq = nn.Parameter(torch.empty((n_layers, vocab_dim, vocab_dim), dtype=dtype, device=device))
wk = nn.Parameter(torch.empty((n_layers, vocab_dim, vocab_dim), dtype=dtype, device=device))
wv = nn.Parameter(torch.empty((n_layers, vocab_dim, vocab_dim), dtype=dtype, device=device))
wo = nn.Parameter(torch.empty((n_layers, vocab_dim, vocab_dim), dtype=dtype, device=device))

w_gate = nn.Parameter(torch.empty((n_layers, vocab_dim, hidden_dim), dtype=dtype, device=device))
w_up = nn.Parameter(torch.empty((n_layers, vocab_dim, hidden_dim), dtype=dtype, device=device))
w_down = nn.Parameter(torch.empty((n_layers, hidden_dim, vocab_dim), dtype=dtype, device=device))

norm_attn = nn.Parameter(torch.ones((n_layers, vocab_dim), dtype=dtype, device=device))
norm_ffn = nn.Parameter(torch.ones((n_layers, vocab_dim), dtype=dtype, device=device))

# Short-circuit Weight Initialization
params = [w_emb, w_unemb, wq, wk, wv, wo, w_gate, w_up, w_down, norm_attn, norm_ffn, norm_final]
for p in params:
    if p.ndim <= 1:
        continue
    nn.init.normal_(p, mean=0.0, std=0.02)

# Optimizer Setup
optimizer = torch.optim.AdamW(params, lr=lr)

# Precomputed RoPE Frequency Tables (Cos & Sin)
theta = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
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

# ==========================================
# 3. TRAINING STEP (Forward, Loss, Backward & Optimizer)
# ==========================================
inputs = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long, device=device)
targets = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long, device=device)

print(f"--- Starting Training Loop on {device} ---")
for step in count(1):
    optimizer.zero_grad()

    # 3.1 Input Lookup
    x = w_emb[inputs]

    # 3.2 Transformer Layers Loop
    for i in range(n_layers):
        # Attention Pre-Norm & Q, K, V Projections
        x_norm = rms_norm(x, norm_attn[i])
        q = torch.matmul(x_norm, wq[i]).reshape(batch_size, seq_len, n_heads, head_dim).transpose(1, 2)
        k = torch.matmul(x_norm, wk[i]).reshape(batch_size, seq_len, n_heads, head_dim).transpose(1, 2)
        v = torch.matmul(x_norm, wv[i]).reshape(batch_size, seq_len, n_heads, head_dim).transpose(1, 2)

        # Apply RoPE to Q and K
        q = apply_rope(q, rope_cos, rope_sin)
        k = apply_rope(k, rope_cos, rope_sin)

        # Parallel Causal Attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = torch.softmax(attn + causal_mask, dim=-1)

        # Output Projection & Attention Residual
        attn_out = torch.matmul(attn, v).transpose(1, 2).reshape(batch_size, seq_len, vocab_dim)
        x = x + torch.matmul(attn_out, wo[i])

        # SwiGLU FFN Pre-Norm, Projection & FFN Residual
        x_norm = rms_norm(x, norm_ffn[i])
        gate = torch.matmul(x_norm, w_gate[i])
        up = torch.matmul(x_norm, w_up[i])
        ffn_out = torch.matmul(F.silu(gate) * up, w_down[i])
        x = x + ffn_out

    # 3.3 Final Norm & Unembedding
    x_norm = rms_norm(x, norm_final)
    logits = torch.matmul(x_norm, w_unemb)

    # 3.4 Shifted Causal Loss, Backpropagation & Optimizer Step
    shift_logits = logits[:, :-1, :].reshape(-1, vocab_size)
    shift_targets = targets[:, 1:].reshape(-1)
    loss = F.cross_entropy(shift_logits, shift_targets)
    loss.backward()
    optimizer.step()
    loss = loss.item()

    print(f"Training Step {step} | Loss: {loss:.4f}")

    if loss < loss_target:
        print(f"Training complete! Loss has reached the target threshold of {loss_target}.")
        break