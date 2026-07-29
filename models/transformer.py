import json
import math
import os
import time
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file as load_safetensors, save_file as save_safetensors

# ==========================================
# 1. Configuration & Global Setup
# ==========================================

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

dtype = torch.float16

config = {
    'n_layers': 2,
    'batch_size': 4,
    'n_heads': 8,
    'n_kv_heads': 2,
    'head_dim': 16,
    'seq_len': 32,
    'vocab_dim': 128,
    'kv_dim': 32,
    'hidden_dim': 256,
    'vocab_size': 512,
    'eps': 1 / 1024,
    'rope_theta': 10_000.0,
    'muon_lr': 0.01,
    'momentum': 0.95,
    'weight_decay': 0.1,
    'loss_target': 1 / 4096,
}


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, weight_decay=0.1, momentum=0.95, nesterov=True, eps=1e-7, ns_steps=5):
        super().__init__(params, dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            eps=eps,
            ns_steps=ns_steps,
        ))

    def addmm(self, inp, mat1, mat2, beta=1, alpha=1):
        return beta * inp + alpha * (mat1 @ mat2)

    def newton_schulz_step(self, upd, a, b, c):
        g = upd @ upd.T
        g_upd = self.addmm(g, g, g, beta=b, alpha=c)
        return self.addmm(upd, g_upd, upd, beta=a)

    def newton_schulz(self, grad, eps, steps):
        a, b, c = 3.4445, -4.7750, 2.0315
        upd = grad.bfloat16()
        is_transposed = grad.size(0) > grad.size(1)
        if is_transposed:
            upd = upd.T
        upd.div_(upd.norm().clamp(min=eps))
        for step_idx in range(steps):
            upd = self.newton_schulz_step(upd, a, b, c)
        if is_transposed:
            upd = upd.T
        return upd

    def step_param(self, p, group):
        if p.ndim != 2:
            raise ValueError("Muon only supports 2D parameters")
        if p.grad is None:
            return
        lr = group['lr']
        wd = group['weight_decay']
        mom = group['momentum']
        nest = group['nesterov']
        eps = group['eps']
        steps = group['ns_steps']
        grad = p.grad
        state = self.state[p]
        if 'buf' not in state:
            state['buf'] = torch.zeros_like(grad)
        buf = state['buf']
        buf.lerp_(grad, 1 - mom)
        upd = grad.lerp(buf, mom) if nest else buf
        upd = self.newton_schulz(upd, eps, steps)
        adj_lr = lr * math.sqrt(max(1, p.shape[0] / p.shape[1]))
        p.mul_(1 - lr * wd)
        p.add_(upd, alpha=-adj_lr)

    def step_group(self, group):
        for p in group['params']:
            self.step_param(p, group)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            self.step_group(group)


def create_param(*shape):
    return nn.ParameterDict({'weight': nn.Parameter(torch.empty(*shape))})


class RMSNorm(nn.Module):
    def __init__(self, vocab_dim, eps, **kwargs):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(vocab_dim))

    def forward(self, x):
        var = x.pow(2).mean(dim=-1, keepdim=True)
        return (x * torch.rsqrt(var + self.eps)) * self.weight


class Attention(nn.Module):
    def __init__(self, vocab_dim, kv_dim, n_heads, n_kv_heads, head_dim, rope_theta, seq_len, **kwargs):
        super().__init__()
        if n_heads % n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even")
        expected_kv_dim = n_kv_heads * head_dim
        if kv_dim != expected_kv_dim:
            raise ValueError("kv_dim must equal n_kv_heads * head_dim")

        self.n_heads = n_heads
        self.head_dim = head_dim
        self.n_kv_heads = n_kv_heads
        self.q_dim = n_heads * head_dim
        self.scale = 1.0 / (self.head_dim ** 0.5)

        theta = 1.0 / (rope_theta ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim))
        seq_idx = torch.arange(seq_len, dtype=torch.float32)
        idx_theta = torch.outer(seq_idx, theta)

        self.register_buffer('rope_cos', idx_theta.cos(), persistent=False)
        self.register_buffer('rope_sin', idx_theta.sin(), persistent=False)
        self.register_buffer('causal_mask', torch.ones(0, dtype=torch.bool), persistent=False)

        self.q_proj = create_param(vocab_dim, self.q_dim)
        self.k_proj = create_param(vocab_dim, expected_kv_dim)
        self.v_proj = create_param(vocab_dim, expected_kv_dim)
        self.o_proj = create_param(self.q_dim, vocab_dim)

    def rope(self, x):
        s = x.shape[-2]
        if s > self.rope_cos.shape[0]:
            raise ValueError("input sequence length exceeds configured seq_len")
        cos = self.rope_cos[:s]
        sin = self.rope_sin[:s]
        x_even, x_odd = x[..., 0::2], x[..., 1::2]
        out = torch.empty_like(x)
        out[..., 0::2] = x_even * cos - x_odd * sin
        out[..., 1::2] = x_even * sin + x_odd * cos
        return out

    def forward(self, x):
        b, s, d = x.shape
        q = torch.matmul(x, self.q_proj.weight).reshape(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        k = torch.matmul(x, self.k_proj.weight).reshape(b, s, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = torch.matmul(x, self.v_proj.weight).reshape(b, s, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = self.rope(q)
        k = self.rope(k)

        n_rep = self.n_heads // self.n_kv_heads
        k = k.repeat_interleave(n_rep, dim=1)
        v = v.repeat_interleave(n_rep, dim=1)

        if self.causal_mask.shape != (s, s) or self.causal_mask.device != x.device:
            self.causal_mask = torch.ones(s, s, dtype=torch.bool, device=x.device).triu(diagonal=1)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn.masked_fill(self.causal_mask, float("-inf"))
        attn = torch.softmax(attn, dim=-1)

        out = torch.matmul(attn, v).transpose(1, 2).reshape(b, s, self.q_dim)
        return torch.matmul(out, self.o_proj.weight)


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

    def forward(self, x):
        x = x + self.self_attn(self.input_layernorm(x))
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class Model(nn.Module):
    def __init__(self, **config):
        super().__init__()
        self.config = config
        vocab_size = config['vocab_size']
        vocab_dim = config['vocab_dim']
        n_layers = config['n_layers']

        self.model = nn.ModuleDict({
            'embed_tokens': create_param(vocab_size, vocab_dim),
            'norm': RMSNorm(**config),
            'layers': nn.ModuleList(Block(**config) for _ in range(n_layers))
        })
        self.lm_head = create_param(vocab_dim, vocab_size)

    def forward(self, inputs):
        x = self.model.embed_tokens.weight[inputs]
        for layer in self.model.layers:
            x = layer(x)
        x = self.model.norm(x)
        return torch.matmul(x, self.lm_head.weight)

    def init_params(self):
        with torch.no_grad():
            for p in self.parameters():
                if p.ndim > 1:
                    nn.init.normal_(p, mean=0.0, std=0.02)
                else:
                    nn.init.ones_(p)

    def save(self, path="ollama_model.safetensors"):
        save_safetensors(self.state_dict(), path)
        config_path = path.rsplit('.', 1)[0] + ".json"
        with open(config_path, "w") as f:
            json.dump(self.config, f, indent=2)
        print(f"Saved model to '{path}' and config to '{config_path}'.")

    def load(self, path="ollama_model.safetensors", device=None):
        if not os.path.exists(path):
            return False
        dev = device or next(self.parameters()).device
        sd = load_safetensors(path, device=str(dev))
        self.load_state_dict(sd)

        config_path = path.rsplit('.', 1)[0] + ".json"
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                self.config = json.load(f)
        print(f"Loaded checkpoint from '{path}' onto {dev}.")
        return True


# Remove existing checkpoint if present for a clean initial demonstration
checkpoint_path = "ollama_model.safetensors"

# Instantiate Model & Load/Initialize Parameters for Training
model = Model(**config).to(device=device, dtype=dtype)
model_loaded = model.load(checkpoint_path, device=device)
if not model_loaded:
    model.init_params()

# Initialize Custom Muon Optimizer
optimizer = Muon(
    (p for p in model.parameters() if p.ndim == 2),
    lr=config['muon_lr'],
    momentum=config['momentum'],
    weight_decay=config['weight_decay'],
)

# ==========================================
# 2. Training Loop
# ==========================================

vocab_size = config['vocab_size']
batch_size = config['batch_size']
seq_len = config['seq_len']
loss_target = config['loss_target']

# Fixed seed for deterministic evaluation data
torch.manual_seed(42)
inputs = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long, device=device)
targets = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long, device=device)

# Evaluate initial loss before entering training loop
with torch.no_grad():
    initial_logits = model(inputs)
    initial_loss = F.cross_entropy(initial_logits[:, :-1, :].reshape(-1, vocab_size), targets[:, 1:].reshape(-1)).item()

print(f"Initial Evaluated Loss: {initial_loss:.6f}")

if initial_loss < loss_target:
    print(f"Loaded checkpoint already satisfies loss target ({initial_loss:.6f} < {loss_target}). Skipping training loop!")
else:
    print(f"--- Starting Training Loop on {device} (Muon Optimizer) ---")
    t0 = time.time()

    for step in count(1):
        model.zero_grad()

        # Forward Pass & Output Logits Computation
        logits = model(inputs)

        # Shifted Causal Loss & Backpropagation
        shift_logits = logits[:, :-1, :].reshape(-1, vocab_size)
        shift_targets = targets[:, 1:].reshape(-1)
        loss = F.cross_entropy(shift_logits, shift_targets)
        loss.backward()

        # Muon Optimizer Step
        optimizer.step()

        print(f"Training Step {step} | Loss: {loss:.4f}")

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Invalid Loss {loss}. Stopping training.")
            break

        if loss.item() < loss_target:
            break

    t1 = time.time()
    elapsed = t1 - t0
    params = sum(p.numel() for p in model.parameters() if p.ndim == 2)

    print(f"Parameters: {params}")
    print(f"Elapsed {elapsed:.2f}s")
    print(f"Final Loss: {loss.item():.8f}")

    model.save(checkpoint_path)
