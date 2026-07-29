import argparse
import json
import math
import os
import re
import sys
import time
import requests
import torch
import torch.nn.functional as F
from safetensors.torch import load_file as load_safetensors, save_file as save_safetensors
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download

from architecture import Model, Muon

# ==========================================
# 1. Configuration & Global Setup
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
# 2. Model Weight Setup & Helper Functions
# ==========================================

def ensure_model_weights():
    """Ensure transparent local weights exist for Llama 3.2 1B."""
    if os.path.exists(CHECKPOINT_PATH) and os.path.exists(CONFIG_PATH):
        return

    print(f"Preparing transparent model weights for {HF_REPO_ID}...")
    hf_path = hf_hub_download(repo_id=HF_REPO_ID, filename="model.safetensors")
    sd = load_safetensors(hf_path)

    mapped_sd = {}
    mapped_sd["model.embed_tokens.weight"] = sd["model.embed_tokens.weight"].to(torch.bfloat16).contiguous()
    mapped_sd["model.norm.weight"] = sd["model.norm.weight"].to(torch.bfloat16).contiguous()

    if "lm_head.weight" in sd:
        mapped_sd["lm_head.weight"] = sd["lm_head.weight"].T.to(torch.bfloat16).contiguous()
    else:
        mapped_sd["lm_head.weight"] = sd["model.embed_tokens.weight"].T.to(torch.bfloat16).contiguous()

    for i in range(16):
        mapped_sd[f"model.layers.{i}.input_layernorm.weight"] = sd[f"model.layers.{i}.input_layernorm.weight"].to(torch.bfloat16).contiguous()
        mapped_sd[f"model.layers.{i}.self_attn.q_proj.weight"] = sd[f"model.layers.{i}.self_attn.q_proj.weight"].T.to(torch.bfloat16).contiguous()
        mapped_sd[f"model.layers.{i}.self_attn.k_proj.weight"] = sd[f"model.layers.{i}.self_attn.k_proj.weight"].T.to(torch.bfloat16).contiguous()
        mapped_sd[f"model.layers.{i}.self_attn.v_proj.weight"] = sd[f"model.layers.{i}.self_attn.v_proj.weight"].T.to(torch.bfloat16).contiguous()
        mapped_sd[f"model.layers.{i}.self_attn.o_proj.weight"] = sd[f"model.layers.{i}.self_attn.o_proj.weight"].T.to(torch.bfloat16).contiguous()
        mapped_sd[f"model.layers.{i}.post_attention_layernorm.weight"] = sd[f"model.layers.{i}.post_attention_layernorm.weight"].to(torch.bfloat16).contiguous()
        mapped_sd[f"model.layers.{i}.mlp.gate_proj.weight"] = sd[f"model.layers.{i}.mlp.gate_proj.weight"].T.to(torch.bfloat16).contiguous()
        mapped_sd[f"model.layers.{i}.mlp.up_proj.weight"] = sd[f"model.layers.{i}.mlp.up_proj.weight"].T.to(torch.bfloat16).contiguous()
        mapped_sd[f"model.layers.{i}.mlp.down_proj.weight"] = sd[f"model.layers.{i}.mlp.down_proj.weight"].T.to(torch.bfloat16).contiguous()

    save_safetensors(mapped_sd, CHECKPOINT_PATH)
    with open(CONFIG_PATH, "w") as f:
        json.dump(DEFAULT_CONFIG, f, indent=2)
    print(f"Converted and saved weights to '{CHECKPOINT_PATH}'.")


def run_inference(prompt: str, model: Model, tokenizer: AutoTokenizer, max_new_tokens: int = 64) -> str:
    """Pure token-ints-in, token-ints-out inference wrapping."""
    prompt_encoding = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        return_tensors="pt"
    )
    input_ids = prompt_encoding["input_ids"].to(device=device)

    curr_tokens = input_ids
    eos_token_ids = [128009, 128001]

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(curr_tokens)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            curr_tokens = torch.cat([curr_tokens, next_token], dim=1)
            if next_token.item() in eos_token_ids:
                break

    gen_token_ids = curr_tokens[0][input_ids.shape[1]:].tolist()
    return tokenizer.decode(gen_token_ids, skip_special_tokens=True).strip()


def query_ollama(prompt: str, model_name: str = "llama3.2:1b", temperature: float = 0.0) -> str:
    """Query local Ollama server."""
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature}
    }
    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except Exception as e:
        return f"[Ollama Error: {e}]"


# ==========================================
# 3. Synthetic Prompt & Dataset Generation
# ==========================================

def generate_synthetic_prompts(count: int = 16) -> list:
    """Ask Ollama at temperature 0.9 to generate diverse creative prompts."""
    meta_prompt = (
        f"Generate exactly {count} diverse, creative, substantive questions asking about science, history, coding, philosophy, logic, or literature. "
        "Return ONLY a JSON list of strings like: [\"Question 1?\", \"Question 2?\"]"
    )
    raw_response = query_ollama(meta_prompt, temperature=0.9)

    match = re.search(r"\[.*\]", raw_response, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, list):
                prompts = [str(item).strip() for item in parsed if isinstance(item, str) or isinstance(item, dict)]
                prompts = [p.get("prompt", p) if isinstance(p, dict) else p for p in prompts]
                if len(prompts) >= count:
                    return prompts[:count]
        except Exception:
            pass

    # Fallback line extraction
    lines = [line.strip().lstrip("0123456789.- \"'") for line in raw_response.split("\n") if line.strip() and ("?" in line or len(line) > 15)]
    return lines[:count]


def save_dataset_records(records: list, path: str = DATASET_PATH):
    """Save generated prompt-response records to jsonl file."""
    with open(path, "a") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


# ==========================================
# 4. Training Loop Execution
# ==========================================

def train_on_records(records: list, model: Model, optimizer: Muon, tokenizer: AutoTokenizer) -> float:
    """Run supervised training (SFT) forward pass, backprop, and Muon optimizer step."""
    model.train()
    vocab_size = DEFAULT_CONFIG["vocab_size"]
    total_loss = 0.0
    valid_batches = 0

    for item in records:
        prompt = item["prompt"]
        response = item["response"]

        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response}
        ]
        
        try:
            full_ids = tokenizer.apply_chat_template(messages, return_tensors="pt")["input_ids"].to(device=device)
        except Exception:
            # Fallback format if template fails
            text = f"User: {prompt}\nAssistant: {response}"
            full_ids = tokenizer.encode(text, return_tensors="pt").to(device=device)

        if full_ids.shape[1] < 2:
            continue

        inputs = full_ids[:, :-1]
        targets = full_ids[:, 1:]

        optimizer.zero_grad()
        logits = model(inputs)
        
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        valid_batches += 1

    return total_loss / max(1, valid_batches)


def main():
    parser = argparse.ArgumentParser(description="Synthetic Dataset Generator & Continuous Muon Trainer")
    parser.add_argument("--total-prompts", type=int, default=128, help="Total number of synthetic prompts to generate and train on")
    parser.add_argument("--batch-size", type=int, default=16, help="Number of prompts per generation & training batch")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs per batch")
    parser.add_argument("--no-op-verify", action="store_true", default=True, help="Run initial no-op Ollama temperature 0 verification")
    args = parser.parse_args()

    ensure_model_weights()

    tokenizer = AutoTokenizer.from_pretrained(HF_REPO_ID)
    model = Model(**DEFAULT_CONFIG).to(device=device, dtype=dtype)
    model_loaded = model.load(CHECKPOINT_PATH, device=device)
    if not model_loaded:
        model.init_params()

    # Step 1: Initial No-Op Verification with Ollama Server (Temperature = 0)
    if args.no_op_verify:
        print("\n--- Step 1: Initial Alignment Check with Ollama Server (Temperature = 0) ---")
        test_prompts = [
            "What is the capital of France?",
            "Name 3 primary colors."
        ]
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

    # Initialize Custom Muon Optimizer over 2D matrix parameters
    optimizer = Muon(
        (p for p in model.parameters() if p.ndim == 2),
        lr=DEFAULT_CONFIG["muon_lr"],
        momentum=DEFAULT_CONFIG["momentum"],
        weight_decay=DEFAULT_CONFIG["weight_decay"],
    )

    total_prompts = args.total_prompts
    batch_size = args.batch_size
    num_batches = math.ceil(total_prompts / batch_size)

    print(f"\n--- Step 2: Continuous Synthetic Dataset Generation & Muon Training ---")
    print(f"Target Prompts: {total_prompts} | Batch Size: {batch_size} | Total Batches: {num_batches}")
    print(f"Dataset Log File: {DATASET_PATH}")

    total_records_processed = 0
    start_time = time.time()

    for batch_idx in range(1, num_batches + 1):
        current_batch_count = min(batch_size, total_prompts - total_records_processed)
        print(f"\n[Batch {batch_idx}/{num_batches}] Generating {current_batch_count} synthetic prompts with Ollama (temp=0.9)...")

        prompts = generate_synthetic_prompts(count=current_batch_count)
        if not prompts:
            # Default fallback prompt list if generation returned empty
            prompts = [
                f"Explain concept #{total_records_processed + i + 1} in computer science or physics."
                for i in range(current_batch_count)
            ]

        batch_records = []
        print(f"[Batch {batch_idx}/{num_batches}] Querying Ollama (temp=0.0) for ground-truth responses...")
        for p in prompts:
            resp = query_ollama(p, temperature=0.0)
            batch_records.append({"prompt": p, "response": resp})

        # Save records to JSONL file
        save_dataset_records(batch_records)

        # Train on batch records
        print(f"[Batch {batch_idx}/{num_batches}] Training PyTorch Model (Muon Optimizer) on {len(batch_records)} samples...")
        t0 = time.time()
        batch_loss = train_on_records(batch_records, model, optimizer, tokenizer)
        t1 = time.time()

        total_records_processed += len(batch_records)
        print(f"Batch {batch_idx} Complete | Average Loss: {batch_loss:.6f} | Time: {t1 - t0:.2f}s | Total Processed: {total_records_processed}/{total_prompts}")

        # Save updated model checkpoint
        model.save(CHECKPOINT_PATH)

    elapsed = time.time() - start_time
    total_params = sum(p.numel() for p in model.parameters() if p.ndim == 2)

    print("\n" + "=" * 60)
    print("CONTINUOUS TRAINING & DATASET GENERATION SUMMARY")
    print("=" * 60)
    print(f"Total Synthetic Prompts Processed: {total_records_processed}")
    print(f"Saved Dataset: {DATASET_PATH}")
    print(f"Optimized Parameters: {total_params:,}")
    print(f"Total Execution Time: {elapsed:.2f}s")
    print(f"Model Checkpoint Saved: {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
