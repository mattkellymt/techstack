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
muon_beta = 0.95
loss_target = 1 / 128

# ==========================================
# 2. Parameters, Helpers & Custom Muon Optimizer
# ==========================================
# Single Model-Level Weights (Embeddings & Unembedding)
w_emb = nn.Parameter(torch.empty((vocab_size, vocab_dim), dtype=dtype, device=device))
w_unemb = nn.Parameter(torch.empty((vocab_dim, vocab_size), dtype=dtype, device=device))
norm_final = nn.Parameter(torch.ones(vocab_dim, dtype=dtype, device=device))

# Multi-Layer Stacked Weights (3D Tensors prepended with n_layers)
wq = nn.Parameter(torch.empty((n_layers, vocab_dim, vocab_dim), dtype=dtype, device=device))
wk = nn.Parameter(torch.empty((n_layers, vocab_dim, kv_dim), dtype=dtype, device=device))
wv = nn.Parameter(torch.empty((n_layers, vocab_dim, kv_dim), dtype=dtype, device=device))
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

# Canonical LLaMA / Ollama State Dict Exporter & Importer (Two-Way Interoperability)
def to_state_dict():
    sd = {
        "model.embed_tokens.weight": w_emb.detach().clone(),
        "model.norm.weight": norm_final.detach().clone(),
        "lm_head.weight": w_unemb.detach().T.clone(),
    }
    for i in range(n_layers):
        sd[f"model.layers.{i}.input_layernorm.weight"] = norm_attn[i].detach().clone()
        sd[f"model.layers.{i}.self_attn.q_proj.weight"] = wq[i].detach().T.clone()
        sd[f"model.layers.{i}.self_attn.k_proj.weight"] = wk[i].detach().T.clone()
        sd[f"model.layers.{i}.self_attn.v_proj.weight"] = wv[i].detach().T.clone()
        sd[f"model.layers.{i}.self_attn.o_proj.weight"] = wo[i].detach().T.clone()
        sd[f"model.layers.{i}.post_attention_layernorm.weight"] = norm_ffn[i].detach().clone()
        sd[f"model.layers.{i}.mlp.gate_proj.weight"] = w_gate[i].detach().T.clone()
        sd[f"model.layers.{i}.mlp.up_proj.weight"] = w_up[i].detach().T.clone()
        sd[f"model.layers.{i}.mlp.down_proj.weight"] = w_down[i].detach().T.clone()
    return sd

def load_state_dict(sd):
    with torch.no_grad():
        w_emb.copy_(sd["model.embed_tokens.weight"])
        norm_final.copy_(sd["model.norm.weight"])
        w_unemb.copy_(sd["lm_head.weight"].T)

        for i in range(n_layers):
            norm_attn[i].copy_(sd[f"model.layers.{i}.input_layernorm.weight"])
            wq[i].copy_(sd[f"model.layers.{i}.self_attn.q_proj.weight"].T)
            wk[i].copy_(sd[f"model.layers.{i}.self_attn.k_proj.weight"].T)
            wv[i].copy_(sd[f"model.layers.{i}.self_attn.v_proj.weight"].T)
            wo[i].copy_(sd[f"model.layers.{i}.self_attn.o_proj.weight"].T)
            norm_ffn[i].copy_(sd[f"model.layers.{i}.post_attention_layernorm.weight"])
            w_gate[i].copy_(sd[f"model.layers.{i}.mlp.gate_proj.weight"].T)
            w_up[i].copy_(sd[f"model.layers.{i}.mlp.up_proj.weight"].T)
            w_down[i].copy_(sd[f"model.layers.{i}.mlp.down_proj.weight"].T)

def save_model(filepath="ollama_model.pt"):
    sd = to_state_dict()
    torch.save(sd, filepath)
    print(f"Exported model checkpoint to '{filepath}' ({len(sd)} weight tensors).")

def load_model(filepath="ollama_model.pt"):
    sd = torch.load(filepath, map_location=device, weights_only=True)
    load_state_dict(sd)
    print(f"Imported model checkpoint from '{filepath}'.")

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

# Standalone Muon Optimizer (Delegates 1D vectors to native AdamW, matrix weights to Newton-Schulz)
class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=muon_lr, momentum=muon_beta, adamw_lr=adamw_lr, eps=eps):
        matrix_params = [p for p in params if p.ndim >= 2]
        vector_params = [p for p in params if p.ndim < 2]
        self.adamw = torch.optim.AdamW(vector_params, lr=adamw_lr, eps=eps) if vector_params else None
        super().__init__(matrix_params, dict(lr=lr, momentum=momentum, eps=eps))

    def zero_grad(self, set_to_none=True):
        super().zero_grad(set_to_none=set_to_none)
        if self.adamw:
            self.adamw.zero_grad(set_to_none=set_to_none)

    def newton_schulz(self, G, steps=5, eps=eps):
        # Newton-Schulz 5th-order polynomial matrix orthogonalization for Muon
        assert G.ndim == 2
        a, b, c = 3.4445, -4.7750, 2.0315
        X = G / (G.norm() + eps)
        if G.size(0) > G.size(1):
            X = X.T
        for _ in range(steps):
            A = X @ X.T
            B = b * A + c * A @ A
            X = a * X + B @ X
        if G.size(0) > G.size(1):
            X = X.T
        return X

    def apply_update(self, p, buf, lr, eps=eps):
        # Orthogonalized gradient update applied to 2D matrices or 3D layer stacks
        if p.ndim == 3:
            scale = lr * max(1, p.shape[1] / p.shape[2]) ** 0.5
            for layer_idx in range(p.shape[0]):
                update = self.newton_schulz(buf[layer_idx], eps=eps)
                p[layer_idx].sub_(update, alpha=scale)
        else:
            scale = lr * max(1, p.shape[0] / p.shape[1]) ** 0.5
            update = self.newton_schulz(buf, eps=eps)
            p.sub_(update, alpha=scale)

    def step_param(self, p, group):
        if p.grad is None:
            return

        lr, momentum, eps = group["lr"], group["momentum"], group["eps"]
        state = self.state[p]
        if "momentum_buf" not in state:
            state["momentum_buf"] = torch.zeros_like(p)

        buf = state["momentum_buf"]
        buf.mul_(momentum).add_(p.grad, alpha=1.0 - momentum)
        self.apply_update(p, buf, lr, eps)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        if self.adamw:
            self.adamw.step()

        for group in self.param_groups:
            for p in group["params"]:
                self.step_param(p, group)

        return loss

optimizer = Muon(params, lr=muon_lr, adamw_lr=adamw_lr, eps=eps)

# ==========================================
# 3. TRAINING STEP (Forward, Loss, Backward & Muon Optimizer)
# ==========================================
inputs = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long, device=device)
targets = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long, device=device)

print(f"--- Starting Training Loop on {device} (Muon Optimizer) ---")
for step in count(1):
    optimizer.zero_grad()

    # 3.1 Input Lookup
    x = w_emb[inputs]

    # 3.2 Transformer Layers Loop
    for i in range(n_layers):
        # Attention Pre-Norm & Q, K, V Projections
        x_norm = rms_norm(x, norm_attn[i])
        q = torch.matmul(x_norm, wq[i]).reshape(batch_size, seq_len, n_heads, head_dim).transpose(1, 2)
        k = torch.matmul(x_norm, wk[i]).reshape(batch_size, seq_len, n_kv_heads, head_dim).transpose(1, 2)
        v = torch.matmul(x_norm, wv[i]).reshape(batch_size, seq_len, n_kv_heads, head_dim).transpose(1, 2)

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
        save_model("ollama_model.pt")
        break