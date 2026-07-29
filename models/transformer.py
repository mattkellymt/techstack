import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from itertools import count
import time

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

    def addmm(self, input, mat1, mat2, beta=1, alpha=1):
        output = beta * input + alpha * (mat1 @ mat2)
        return output

    def newton_schulz_step(self, update, a, b, c):
        gram = update @ update.T
        gram_update = self.addmm(gram, gram, gram, beta=b, alpha=c)
        next_update = self.addmm(update, gram_update, update, beta=a)
        return next_update

    def newton_schulz(self, grad, eps, steps):
        a, b, c = 3.4445, -4.7750, 2.0315
        update = grad.bfloat16()
        transposed = grad.size(0) > grad.size(1)
        if transposed:
            update = update.T
        update.div_(update.norm().clamp(min=eps))
        for _ in range(steps):
            update = self.newton_schulz_step(update, a, b, c)
        if transposed:
            update = update.T
        return update

    def step_param(self, p, group):
        if p.ndim != 2:
            raise ValueError("Muon only supports 2D parameters")
        if p.grad is None:
            return
        lr = group['lr']
        weight_decay = group['weight_decay']
        momentum = group['momentum']
        nesterov = group['nesterov']
        eps = group['eps']
        ns_steps = group['ns_steps']
        grad = p.grad
        state = self.state[p]
        if 'momentum_buffer' not in state:
            state['momentum_buffer'] = torch.zeros_like(grad)
        buf = state['momentum_buffer']
        buf.lerp_(grad, 1 - momentum)
        update = grad.lerp(buf, momentum) if nesterov else buf
        update = self.newton_schulz(update, eps, ns_steps)
        adjusted_lr = lr * math.sqrt(max(1, p.shape[0] / p.shape[1]))
        p.mul_(1 - lr * weight_decay)
        p.add_(update, alpha=-adjusted_lr)

    def step_group(self, group):
        params = group['params']
        for p in params:
            self.step_param(p, group)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            self.step_group(group)

def create_param(*shape):
    param = nn.ParameterDict({'weight': nn.Parameter(torch.empty(*shape))})
    return param

class RMSNorm(nn.Module):
    def __init__(self, vocab_dim, eps, **kwargs):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(vocab_dim))

    def forward(self, x):
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        output = (x * torch.rsqrt(variance + self.eps)) * self.weight
        return output

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

        theta = 1.0 / (rope_theta ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim))
        seq_idx = torch.arange(seq_len, dtype=torch.float32)
        idx_theta = torch.outer(seq_idx, theta)

        self.register_buffer('rope_cos', idx_theta.cos(), persistent=False)
        self.register_buffer('rope_sin', idx_theta.sin(), persistent=False)
        self.register_buffer('causal_mask', torch.ones(0, dtype=torch.bool), persistent=False)

        self.scale = 1.0 / (self.head_dim ** 0.5)

        self.q_proj = create_param(vocab_dim, self.q_dim)
        self.k_proj = create_param(vocab_dim, expected_kv_dim)
        self.v_proj = create_param(vocab_dim, expected_kv_dim)
        self.o_proj = create_param(self.q_dim, vocab_dim)

    def rope(self, x):
        seq_len = x.shape[-2]
        if seq_len > self.rope_cos.shape[0]:
            raise ValueError("input sequence length exceeds configured seq_len")
        pivot = x.shape[-1] // 2
        x1, x2 = x[..., :pivot], x[..., pivot:]
        rope_cos = self.rope_cos[:seq_len]
        rope_sin = self.rope_sin[:seq_len]
        out = torch.empty_like(x)
        out[..., :pivot] = x1 * rope_cos - x2 * rope_sin
        out[..., pivot:] = x1 * rope_sin + x2 * rope_cos
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

        if self.causal_mask.shape != (s, s) or self.causal_mask.device != x.device:
            self.causal_mask = torch.ones(s, s, dtype=torch.bool, device=x.device).triu(diagonal=1)
        causal_mask = self.causal_mask

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn.masked_fill(causal_mask, float("-inf"))
        attn = torch.softmax(attn, dim=-1)
        
        attn_out = torch.matmul(attn, v).transpose(1, 2).reshape(b, s, self.q_dim)
        output = torch.matmul(attn_out, self.o_proj.weight)
        return output

class MLP(nn.Module):
    def __init__(self, vocab_dim, hidden_dim, **kwargs):
        super().__init__()
        self.gate_proj = create_param(vocab_dim, hidden_dim)
        self.up_proj = create_param(vocab_dim, hidden_dim)
        self.down_proj = create_param(hidden_dim, vocab_dim)

    def forward(self, x):
        gate = torch.matmul(x, self.gate_proj.weight)
        up = torch.matmul(x, self.up_proj.weight)
        output = torch.matmul(F.silu(gate) * up, self.down_proj.weight)
        return output

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
        output = torch.matmul(x, self.lm_head.weight)
        return output

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

    def load(self, filepath, device=None):
        target_device = device or next(self.parameters()).device
        checkpoint = torch.load(filepath, map_location=target_device, weights_only=True)
        if isinstance(checkpoint, dict) and 'config' in checkpoint:
            self.config = checkpoint['config']
        sd = checkpoint.get('state_dict', checkpoint)
        self.load_state_dict(sd)

# Instantiate Model & Initialize Parameters for Training
model = Model(**config).to(device=device, dtype=dtype)
model.init_params()

# Initialize Custom Muon Optimizer
optimizer = Muon(
    (p for p in model.parameters() if p.ndim == 2),
    lr=config['muon_lr'],
    momentum=config['momentum'],
    weight_decay=config['weight_decay'],
)

# ==========================================
# 2. TRAINING STEP (Forward, Loss, Backward & Muon Optimizer)
# ==========================================
vocab_size = config['vocab_size']
batch_size = config['batch_size']
seq_len = config['seq_len']
loss_target = config['loss_target']

inputs = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long, device=device)
targets = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long, device=device)
start_time = time.time()

print(f"--- Starting Training Loop on {device} (Muon Optimizer) ---")

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

    if loss < loss_target:
        break

stop_time = time.time()
elapsed_time = stop_time - start_time
total_params = sum(p.numel() for p in model.parameters() if p.ndim == 2)

print(f"Parameters: {total_params}")
print(f"Elapsed {elapsed_time:.2f}")
print(f"Loss: {loss:.8f}")

model.save("ollama_model.pt")
