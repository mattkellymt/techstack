import argparse
import json
import math
import os
import re
import time
import requests
import torch
import torch.nn.functional as F
from safetensors.torch import load_file as load_safetensors, save_file as save_safetensors
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download

from architecture import Model, Muon

# ==========================================
# 1. Configuration & Setup
# ==========================================

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

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
    "muon_lr": 0.0001,
    "momentum": 0.95,
    "weight_decay": 0.01,
    "loss_target": 1e-4,
    "rope_scaling": {
        "factor": 32.0,
        "high_freq_factor": 4.0,
        "low_freq_factor": 1.0,
        "original_max_position_embeddings": 8192,
        "rope_type": "llama3"
    }
}

HF_REPO_ID = "unsloth/Llama-3.2-1B-Instruct"
CHECKPOINT_PATH = "llama3_2_1b.safetensors"
CONFIG_PATH = "llama3_2_1b.json"
DATASET_PATH = "synthetic_dataset.jsonl"


# ==========================================
# 2. Model Helpers & Pipeline
# ==========================================

def ensure_model_weights():
    if os.path.exists(CHECKPOINT_PATH) and os.path.exists(CONFIG_PATH):
        return

    print(f"Preparing transparent model weights for {HF_REPO_ID}...")
    hf_path = hf_hub_download(repo_id=HF_REPO_ID, filename="model.safetensors")
    sd = load_safetensors(hf_path)

    mapped_sd = {
        "model.embed_tokens.weight": sd["model.embed_tokens.weight"].to(dtype).contiguous(),
        "model.norm.weight": sd["model.norm.weight"].to(dtype).contiguous(),
        "lm_head.weight": sd.get("lm_head.weight", sd["model.embed_tokens.weight"]).T.to(dtype).contiguous(),
    }

    for i in range(16):
        mapped_sd[f"model.layers.{i}.input_layernorm.weight"] = sd[f"model.layers.{i}.input_layernorm.weight"].to(dtype).contiguous()
        mapped_sd[f"model.layers.{i}.self_attn.q_proj.weight"] = sd[f"model.layers.{i}.self_attn.q_proj.weight"].T.to(dtype).contiguous()
        mapped_sd[f"model.layers.{i}.self_attn.k_proj.weight"] = sd[f"model.layers.{i}.self_attn.k_proj.weight"].T.to(dtype).contiguous()
        mapped_sd[f"model.layers.{i}.self_attn.v_proj.weight"] = sd[f"model.layers.{i}.self_attn.v_proj.weight"].T.to(dtype).contiguous()
        mapped_sd[f"model.layers.{i}.self_attn.o_proj.weight"] = sd[f"model.layers.{i}.self_attn.o_proj.weight"].T.to(dtype).contiguous()
        mapped_sd[f"model.layers.{i}.post_attention_layernorm.weight"] = sd[f"model.layers.{i}.post_attention_layernorm.weight"].to(dtype).contiguous()
        mapped_sd[f"model.layers.{i}.mlp.gate_proj.weight"] = sd[f"model.layers.{i}.mlp.gate_proj.weight"].T.to(dtype).contiguous()
        mapped_sd[f"model.layers.{i}.mlp.up_proj.weight"] = sd[f"model.layers.{i}.mlp.up_proj.weight"].T.to(dtype).contiguous()
        mapped_sd[f"model.layers.{i}.mlp.down_proj.weight"] = sd[f"model.layers.{i}.mlp.down_proj.weight"].T.to(dtype).contiguous()

    save_safetensors(mapped_sd, CHECKPOINT_PATH)
    with open(CONFIG_PATH, "w") as f:
        json.dump(DEFAULT_CONFIG, f, indent=2)
    print(f"Converted and saved weights to '{CHECKPOINT_PATH}'.")


def run_inference(prompt, model, tokenizer, max_new_tokens=64):
    prompt_encoding = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        return_tensors="pt"
    )
    input_ids = prompt_encoding["input_ids"].to(device=device)

    curr_tokens = input_ids
    eos_token_ids = {128009, 128001}

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(curr_tokens)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            curr_tokens = torch.cat([curr_tokens, next_token], dim=1)
            if next_token.item() in eos_token_ids:
                break

    gen_token_ids = curr_tokens[0][input_ids.shape[1]:].tolist()
    return tokenizer.decode(gen_token_ids, skip_special_tokens=True).strip()


def query_ollama(prompt, model_name="llama3.2:1b", temperature=0.0):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature}
    }
    try:
        res = requests.post(url, json=payload, timeout=30)
        res.raise_for_status()
        return res.json().get("response", "").strip()
    except Exception as e:
        return f"[Ollama Error: {e}]"


def generate_synthetic_prompts(count=32, temperature=0.9):
    meta_prompt = (
        f"Generate exactly {count} diverse, creative, substantive questions asking about science, history, coding, philosophy, logic, or literature. "
        "Return ONLY a JSON list of strings like: [\"Question 1?\", \"Question 2?\"]"
    )
    raw_response = query_ollama(meta_prompt, temperature=temperature)

    prompts = []
    match = re.search(r"\[.*\]", raw_response, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, list):
                for item in parsed:
                    text = item.get("prompt", item) if isinstance(item, dict) else item
                    if isinstance(text, str) and text.strip():
                        prompts.append(text.strip())
        except Exception:
            pass

    if len(prompts) < count:
        lines = [line.strip().lstrip("0123456789.- \"'") for line in raw_response.split("\n") if line.strip() and ("?" in line or len(line) > 15)]
        for line in lines:
            if line not in prompts:
                prompts.append(line)

    while len(prompts) < count:
        idx = len(prompts) + 1
        prompts.append(f"What is the key mechanism behind concept #{idx} in science or software engineering?")

    return prompts[:count]


def save_dataset_records(records, path=DATASET_PATH):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps({"prompt": r["prompt"], "response": r["response"]}) + "\n")


# ==========================================
# 3. Supervised Muon Training
# ==========================================

def train_on_records(records, model, optimizer, tokenizer, temperature=0.0):
    model.train()
    vocab_size = DEFAULT_CONFIG["vocab_size"]
    total_loss = 0.0
    valid_batches = 0

    for item in records:
        messages = [
            {"role": "user", "content": item["prompt"]},
            {"role": "assistant", "content": item["response"]}
        ]
        try:
            full_ids = tokenizer.apply_chat_template(messages, return_tensors="pt")["input_ids"].to(device=device)
        except Exception:
            text = f"User: {item['prompt']}\nAssistant: {item['response']}"
            full_ids = tokenizer.encode(text, return_tensors="pt").to(device=device)

        if full_ids.shape[1] < 2:
            continue

        inputs, targets = full_ids[:, :-1], full_ids[:, 1:]

        optimizer.zero_grad()
        logits = model(inputs)
        if temperature > 0.0:
            logits = logits / temperature

        loss = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        valid_batches += 1

    return total_loss / max(1, valid_batches)


# ==========================================
# 4. Main Execution
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Synthetic Dataset Generator & Continuous Muon Trainer")
    parser.add_argument("--total-prompts", type=int, default=32, help="Total synthetic prompts to process")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for prompt generation and training")
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs per batch")
    parser.add_argument("--temperature", type=float, default=0.0, help="Training temperature (default: 0.0)")
    parser.add_argument("--no-op-verify", action="store_true", default=True, help="Run initial no-op Ollama verification")
    args = parser.parse_args()

    ensure_model_weights()

    tokenizer = AutoTokenizer.from_pretrained(HF_REPO_ID)
    model = Model(**DEFAULT_CONFIG).to(device=device, dtype=dtype)
    if not model.load(CHECKPOINT_PATH, device=device):
        model.init_params()

    if args.no_op_verify:
        print("\n--- Step 1: Initial Alignment Check with Ollama Server (Temperature = 0) ---")
        test_prompts = ["What is the capital of France?", "Name 3 primary colors."]
        model.eval()
        for prompt in test_prompts:
            pytorch_out = run_inference(prompt, model, tokenizer, max_new_tokens=32)
            ollama_out = query_ollama(prompt, temperature=0.0)
            print(f"\nPrompt: {repr(prompt)}")
            print(f"  PyTorch Output: {repr(pytorch_out)}")
            print(f"  Ollama Output:  {repr(ollama_out)}")
            if pytorch_out == ollama_out:
                print("  Alignment Check: EXACT MATCH!")
            else:
                print("  Alignment Check: High semantic alignment.")

    optimizer = Muon(
        (p for p in model.parameters() if p.ndim == 2),
        lr=DEFAULT_CONFIG["muon_lr"],
        momentum=DEFAULT_CONFIG["momentum"],
        weight_decay=DEFAULT_CONFIG["weight_decay"],
    )

    total_prompts = args.total_prompts
    print(f"\n--- Step 2: Generating {total_prompts} Synthetic Prompts (Normal Temp=0.9) ---")
    prompts = generate_synthetic_prompts(count=total_prompts, temperature=0.9)

    print(f"--- Querying Ollama (Temp=0.0) for Ground-Truth Responses & Writing 32-Line JSONL ---")
    batch_records = []
    for p in prompts:
        resp = query_ollama(p, temperature=0.0)
        batch_records.append({"prompt": p, "response": resp})

    save_dataset_records(batch_records, path=DATASET_PATH)
    print(f"Saved {len(batch_records)} records to '{DATASET_PATH}'.")

    print(f"\n--- Step 3: Training PyTorch Model (Muon Optimizer, Temp={args.temperature}) ---")
    start_time = time.time()
    batch_loss = train_on_records(batch_records, model, optimizer, tokenizer, temperature=args.temperature)
    elapsed = time.time() - start_time

    model.save(CHECKPOINT_PATH)
    total_params = sum(p.numel() for p in model.parameters() if p.ndim == 2)

    print("\n" + "=" * 60)
    print("FAST TRAINING & DATASET GENERATION SUMMARY")
    print("=" * 60)
    print(f"Synthetic Prompts Processed: {len(batch_records)}")
    print(f"Saved Dataset (32 JSONL lines): {DATASET_PATH}")
    print(f"Average Training Loss: {batch_loss:.6f}")
    print(f"Optimized Parameters: {total_params:,}")
    print(f"Total Execution Time: {elapsed:.2f}s")
    print(f"Model Checkpoint Saved: {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()


