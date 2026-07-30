import torch
import torch.nn.functional as F

b = 2
n_kv_heads = 8
n_rep = 4
n_heads = 32
s = 7
head_dim = 64

q = torch.randn(b, n_heads, s, head_dim)
k = torch.randn(b, n_kv_heads, s, head_dim)
v = torch.randn(b, n_kv_heads, s, head_dim)

# Method 1: repeat_kv + manual
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

k_rep = repeat_kv(k, n_rep)
v_rep = repeat_kv(v, n_rep)

causal_mask = torch.ones(s, s, dtype=torch.bool).triu(diagonal=1)
attn = torch.matmul(q, k_rep.transpose(-2, -1)) / (head_dim ** 0.5)
attn = attn.masked_fill(causal_mask, float("-inf"))
attn = torch.softmax(attn, dim=-1)
out_manual = torch.matmul(attn, v_rep)

# Method 2: SDPA broadcasting
q_gqa = q.view(b, n_kv_heads, n_rep, s, head_dim)
k_gqa = k.unsqueeze(2)
v_gqa = v.unsqueeze(2)

out_sdpa = F.scaled_dot_product_attention(
    q_gqa, k_gqa, v_gqa, 
    is_causal=True
)
out_sdpa = out_sdpa.reshape(b, n_heads, s, head_dim)

print("Difference:", (out_manual - out_sdpa).abs().max().item())
