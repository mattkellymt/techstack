import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import count

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

# Top-level Model & Training Configuration Dictionary
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
    'adamw_lr': 1e-3,
    'muon_lr': 0.02,
    'loss_target': 1 / 128,
    'lr': 0.02,
    'momentum': 0.95,
    'adamw_lr': 1e-3,
}

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr, momentum, adamw_lr, eps, **kwargs):
        matrix_params = [p for p in params if p.ndim >= 2]
        vector_params = [p for p in params if p.ndim < 2]
        self.adamw = torch.optim.AdamW(vector_params, lr=adamw_lr, eps=eps) if vector_params else None
        super().__init__(matrix_params, dict(lr=lr, momentum=momentum, eps=eps))
        self.eps = eps 
        self.steps = 5

    def zero_grad(self, set_to_none=True):
        super().zero_grad(set_to_none=set_to_none)
        if self.adamw:
            self.adamw.zero_grad(set_to_none=set_to_none)

    def newton_schulz(self, G):
        assert G.ndim == 2
        a, b, c = 3.4445, -4.7750, 2.0315
        X = G / (G.norm() + self.eps)
        if G.size(0) > G.size(1):
            X = X.T
        for _ in range(self.steps):
            A = X @ X.T
            B = b * A + c * A @ A
            X = a * X + B @ X
        if G.size(0) > G.size(1):
            X = X.T
        return X

    def apply_update_2d(self, p, buf, lr):
        scale = lr * max(1, p.shape[0] / p.shape[1]) ** 0.5
        update = self.newton_schulz(buf)
        p.sub_(update, alpha=scale)

    def apply_update_3d(self, p, buf, lr):
        for layer_idx in range(p.shape[0]):
            self.apply_update_2d(p[layer_idx], buf[layer_idx], lr)

    def apply_update(self, p, buf, lr):
        match p.ndim:
            case 2:
                self.apply_update_2d(p, buf, lr)
            case 3:
                self.apply_update_3d(p, buf, lr)
            case _:
                raise ValueError(f"Muon optimizer only supports 2D or 3D parameters, got ndim={p.ndim}")

    def step_param(self, p, group):
        if p.grad is None:
            return

        lr, momentum, eps = group["lr"], group["momentum"], group["eps"]
        state = self.state[p]
        if "momentum_buf" not in state:
            state["momentum_buf"] = torch.zeros_like(p)

        buf = state["momentum_buf"]
        buf.mul_(momentum).add_(p.grad, alpha=1.0 - momentum)
        self.apply_update(p, buf, lr)

    def step_group(self, group):
        for p in group["params"]:
            self.step_param(p, group)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        if self.adamw:
            self.adamw.step()

        for group in self.param_groups:
            self.step_group(group)

        return loss

def create_param(*shape):
    return nn.ParameterDict({'weight': nn.Parameter(torch.empty(*shape))})

class RMSNorm(nn.Module):
    def __init__(self, vocab_dim, eps, **kwargs):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(vocab_dim))

    def forward(self, x):
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        return (x * torch.rsqrt(variance + self.eps)) * self.weight

class Attention(nn.Module):
    def __init__(self, vocab_dim, kv_dim, **kwargs):
        super().__init__()
        self.n_heads = config['n_heads']
        self.head_dim = config['head_dim']
        self.n_kv_heads = config['n_kv_heads']
        rope_theta = config['rope_theta']
        seq_len = config['seq_len']

        theta = 1.0 / (rope_theta ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32, device=device) / self.head_dim))
        seq_idx = torch.arange(seq_len, dtype=torch.float32, device=device)
        idx_theta = torch.outer(seq_idx, theta)

        self.rope_cos = idx_theta.cos().to(dtype).to(device)
        self.rope_sin = idx_theta.sin().to(dtype).to(device)

        self.scale = torch.tensor(1.0 / (self.head_dim ** 0.5), dtype=torch.float32, device=device)
        self.causal_mask = torch.ones(0, dtype=torch.bool, device=device)

        self.q_proj = create_param(vocab_dim, vocab_dim)
        self.k_proj = create_param(vocab_dim, kv_dim)
        self.v_proj = create_param(vocab_dim, kv_dim)
        self.o_proj = create_param(vocab_dim, vocab_dim)

    def rope(self, x):
        pivot = x.shape[-1] // 2
        x1, x2 = x[..., :pivot], x[..., pivot:]
        out = torch.empty_like(x)
        out[..., :pivot] = x1 * self.rope_cos - x2 * self.rope_sin
        out[..., pivot:] = x1 * self.rope_sin + x2 * self.rope_cos
        return out

    def forward(self, x):
        b, s, d = x.shape
        q = torch.matmul(x, self.q_proj.weight).reshape(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        k = torch.matmul(x, self.k_proj.weight).reshape(b, s, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = torch.matmul(x, self.v_proj.weight).reshape(b, s, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = self.rope(q)
        k = self.rope(k)

        n_groups = self.n_heads // self.n_kv_heads

        k = k.repeat_interleave(n_groups, dim=1)
        v = v.repeat_interleave(n_groups, dim=1)

        if self.causal_mask.shape != (s, s):
            self.causal_mask = torch.ones(s, s, dtype=bool, device=device).triu(diagonal=1)
        causal_mask = self.causal_mask

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn.masked_fill(causal_mask, float("-inf"))
        attn = torch.softmax(attn, dim=-1)
        
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
optimizer = Muon(model.parameters(), **config)

# ==========================================
# 2. TRAINING STEP (Forward, Loss, Backward & Muon Optimizer)
# ==========================================
vocab_size = config['vocab_size']
batch_size = config['batch_size']
seq_len = config['seq_len']
loss_target = config['loss_target']

inputs = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long, device=device)
targets = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long, device=device)

print(f"--- Starting Training Loop on {device} (Muon Optimizer) ---")
for step in count(1):
    optimizer.zero_grad()

    # Forward Pass & Output Logits Computation
    logits = model(inputs)

    # Shifted Causal Loss & Backpropagation
    shift_logits = logits[:, :-1, :].reshape(-1, vocab_size)
    shift_targets = targets[:, 1:].reshape(-1)
    loss = F.cross_entropy(shift_logits, shift_targets)
    loss.backward()

    # Unified Optimizer Step (Muon for 2D Matrices, AdamW for 1D Vectors)
    optimizer.step()

    loss_val = loss.item()
    print(f"Training Step {step} | Loss: {loss_val:.4f}")

    if loss_val < loss_target or loss_val != loss_val:
        print(f"Training complete! Loss has reached the target threshold of {loss_target}.")
        model.save("ollama_model.pt")
        break