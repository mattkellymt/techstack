import sys
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from architecture import Model, RMSNorm

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
dtype = torch.bfloat16

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

HF_REPO_ID = "unsloth/Llama-3.2-1B-Instruct"
CHECKPOINT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "llama3_2_1b.safetensors"))


def compare_tensors(t_custom, t_ref, name):
    t_custom_f = t_custom.detach().float()
    t_ref_f = t_ref.detach().float()

    max_diff = (t_custom_f - t_ref_f).abs().max().item()
    mae = (t_custom_f - t_ref_f).abs().mean().item()
    mse = ((t_custom_f - t_ref_f) ** 2).mean().item()

    cos_sim = F.cosine_similarity(t_custom_f.flatten().unsqueeze(0), t_ref_f.flatten().unsqueeze(0)).item()
    ref_denom = t_ref_f.abs().clamp(min=1e-6)
    max_rel_diff = ((t_custom_f - t_ref_f).abs() / ref_denom).max().item() * 100.0

    return {
        "name": name,
        "shape": list(t_custom.shape),
        "max_diff": max_diff,
        "mae": mae,
        "mse": mse,
        "cos_sim": cos_sim,
        "max_rel_diff_pct": max_rel_diff
    }


def step_by_step_walk(hf_model, custom_model, input_ids):
    report_data = []

    # 1. Embedding Stage
    hf_emb = hf_model.model.embed_tokens(input_ids)
    custom_emb = custom_model.model.embed_tokens.weight[input_ids]
    report_data.append(compare_tensors(custom_emb, hf_emb, "00_embed_tokens"))

    x_hf = hf_emb
    x_custom = custom_emb

    # 2. Iterate through 16 Blocks
    for i in range(16):
        hf_layer = hf_model.model.layers[i]
        custom_layer = custom_model.model.layers[i]

        # a) Input Layernorm
        norm_hf = hf_layer.input_layernorm(x_hf)
        norm_custom = custom_layer.input_layernorm(x_custom)
        report_data.append(compare_tensors(norm_custom, norm_hf, f"layer_{i:02d}_01_input_norm"))

        # b) Q, K, V Projections
        # HF projections
        q_hf = hf_layer.self_attn.q_proj(norm_hf)
        k_hf = hf_layer.self_attn.k_proj(norm_hf)
        v_hf = hf_layer.self_attn.v_proj(norm_hf)

        # Custom projections
        q_custom = torch.matmul(norm_custom, custom_layer.self_attn.q_proj.weight)
        k_custom = torch.matmul(norm_custom, custom_layer.self_attn.k_proj.weight)
        v_custom = torch.matmul(norm_custom, custom_layer.self_attn.v_proj.weight)

        report_data.append(compare_tensors(q_custom, q_hf, f"layer_{i:02d}_02_q_proj"))
        report_data.append(compare_tensors(k_custom, k_hf, f"layer_{i:02d}_03_k_proj"))
        report_data.append(compare_tensors(v_custom, v_hf, f"layer_{i:02d}_04_v_proj"))

        # c) Attention Output
        b, s = input_ids.shape
        position_ids = torch.arange(s, device=device).unsqueeze(0)
        pos_emb = hf_model.model.rotary_emb(norm_hf, position_ids)
        attn_hf = hf_layer.self_attn(norm_hf, position_ids=position_ids, position_embeddings=pos_emb)[0]
        pos_emb_custom = custom_model.model.rotary_emb(position_ids)
        attn_custom = custom_layer.self_attn(norm_custom, position_embeddings=pos_emb_custom)
        report_data.append(compare_tensors(attn_custom, attn_hf, f"layer_{i:02d}_05_attn_out"))

        # d) Residual 1 (Attention Residual)
        x_hf = x_hf + attn_hf
        x_custom = x_custom + attn_custom
        report_data.append(compare_tensors(x_custom, x_hf, f"layer_{i:02d}_06_attn_residual"))

        # e) Post Attention Layernorm
        post_norm_hf = hf_layer.post_attention_layernorm(x_hf)
        post_norm_custom = custom_layer.post_attention_layernorm(x_custom)
        report_data.append(compare_tensors(post_norm_custom, post_norm_hf, f"layer_{i:02d}_07_post_norm"))

        # f) MLP Gate / Up / Down
        mlp_hf = hf_layer.mlp(post_norm_hf)
        mlp_custom = custom_layer.mlp(post_norm_custom)
        report_data.append(compare_tensors(mlp_custom, mlp_hf, f"layer_{i:02d}_08_mlp_out"))

        # g) Residual 2 (Block Output)
        x_hf = x_hf + mlp_hf
        x_custom = x_custom + mlp_custom
        report_data.append(compare_tensors(x_custom, x_hf, f"layer_{i:02d}_09_block_out"))

    # 3. Final Norm & Head
    final_norm_hf = hf_model.model.norm(x_hf)
    final_norm_custom = custom_model.model.norm(x_custom)
    report_data.append(compare_tensors(final_norm_custom, final_norm_hf, "98_final_norm"))

    logits_hf = hf_model.lm_head(final_norm_hf)
    logits_custom = torch.matmul(final_norm_custom, custom_model.model.embed_tokens.weight.T)
    report_data.append(compare_tensors(logits_custom, logits_hf, "99_lm_head_logits"))

    return report_data


def golden_state_injection_walk(hf_model, custom_model, input_ids):
    """Walk through layers, but inject HF's exact golden state at each step to isolate per-op math divergence."""
    injection_data = []

    # Get all 17 golden hidden states (0: embedding, 1..16: layer outputs)
    with torch.no_grad():
        golden_outputs = hf_model.model(input_ids, output_hidden_states=True)
        hidden_states_list = golden_outputs.hidden_states

    # 1. Embedding
    custom_emb = custom_model.model.embed_tokens.weight[input_ids]
    injection_data.append(compare_tensors(custom_emb, hidden_states_list[0], "isolated_00_embed_tokens"))

    b, s = input_ids.shape
    position_ids = torch.arange(s, device=device).unsqueeze(0)
    pos_emb_custom = custom_model.model.rotary_emb(position_ids)

    # 2. Layer-by-layer golden state injection
    for i in range(16):
        golden_in = hidden_states_list[i]
        golden_target = hidden_states_list[i + 1]

        custom_layer = custom_model.model.layers[i]
        custom_block_out = custom_layer(golden_in, position_embeddings=pos_emb_custom)

        injection_data.append(compare_tensors(custom_block_out, golden_target, f"isolated_block_{i:02d}_out"))

    # 3. Final Norm on Golden Layer 15 output
    golden_l15_out = hidden_states_list[16]
    custom_final_norm = custom_model.model.norm(golden_l15_out)
    hf_final_norm = hf_model.model.norm(golden_l15_out)
    injection_data.append(compare_tensors(custom_final_norm, hf_final_norm, "isolated_98_final_norm"))

    # 4. LM Head on Golden Final Norm
    custom_logits = torch.matmul(hf_final_norm, custom_model.model.embed_tokens.weight.T)
    hf_logits = hf_model.lm_head(hf_final_norm)
    injection_data.append(compare_tensors(custom_logits, hf_logits, "isolated_99_lm_head_logits"))

    return injection_data




def main():
    print("=" * 80)
    print("PATH 2 & 4: STEP-BY-STEP NUMERICAL LAYER WALKER & GOLDEN INJECTION")
    print("=" * 80)
    print(f"Device: {device} | Dtype: {dtype}")

    print("\n[1/3] Loading Models...")
    hf_tokenizer = AutoTokenizer.from_pretrained(HF_REPO_ID)
    hf_model = AutoModelForCausalLM.from_pretrained(HF_REPO_ID, torch_dtype=dtype).to(device)
    hf_model.eval()

    custom_model = Model(**DEFAULT_CONFIG).to(device=device, dtype=dtype)
    custom_model.load(CHECKPOINT_PATH, device=device)
    custom_model.eval()

    prompt = "What is the capital of France?"
    input_ids = hf_tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    print(f"Prompt: {repr(prompt)} | Input Token IDs Shape: {list(input_ids.shape)}")

    print("\n[2/3] Walking Accumulative Layer Deltas...")
    report_data = step_by_step_walk(hf_model, custom_model, input_ids)

    print("\n" + "=" * 90)
    print(f"{'ACCUMULATIVE STAGE':<32} | {'MAX ABS DIFF':<14} | {'MAE':<12} | {'COS SIMILARITY':<14}")
    print("=" * 90)
    for row in report_data:
        print(f"{row['name']:<32} | {row['max_diff']:<14.6f} | {row['mae']:<12.6f} | {row['cos_sim']:<14.8f}")
    print("=" * 90)

    print("\n[3/3] Walking Golden-State Isolated Layer Deltas...")
    injection_data = golden_state_injection_walk(hf_model, custom_model, input_ids)

    print("\n" + "=" * 90)
    print(f"{'ISOLATED OPERATOR STAGE':<32} | {'MAX ABS DIFF':<14} | {'MAE':<12} | {'COS SIMILARITY':<14}")
    print("=" * 90)
    for row in injection_data[:15]:
        print(f"{row['name']:<32} | {row['max_diff']:<14.6f} | {row['mae']:<12.6f} | {row['cos_sim']:<14.8f}")
    print("=" * 90)

    os.makedirs("audit_and_verification/reports", exist_ok=True)
    with open("audit_and_verification/reports/layer_walk_results.json", "w") as f:
        json.dump({"accumulative": report_data, "isolated": injection_data}, f, indent=2)
    print("\nSaved detailed layer walk metrics to 'audit_and_verification/reports/layer_walk_results.json'.")


if __name__ == "__main__":
    main()
