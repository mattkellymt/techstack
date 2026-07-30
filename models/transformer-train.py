import argparse
import json
import math
import os
import re
import time
import requests
import threading
import queue
import matplotlib.pyplot as plt
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


FALLBACK_PROMPTS = [
    "What is the principle of conservation of energy?",
    "Explain how a hash table works in computer science.",
    "What is the theory of general relativity?",
    "How does the model-view-controller pattern work?",
    "What caused the fall of the Western Roman Empire?",
    "Explain the concept of recursion with an example.",
    "What is the function of mitochondria in a cell?",
    "How does a binary search tree maintain order?",
    "What is the difference between synchronous and asynchronous execution?",
    "Explain the laws of thermodynamics.",
    "What is quantum entanglement?",
    "How does gradient descent optimize neural networks?",
    "What is the Turing test?",
    "Explain the difference between TCP and UDP.",
    "What is the role of natural selection in evolution?",
    "How does key-value storage differ from relational databases?",
    "What is the significance of the Magna Carta?",
    "Explain how attention mechanisms work in transformers.",
    "What is the Doppler effect?",
    "How does garbage collection work in modern programming languages?",
    "What is the difference between a process and a thread?",
    "Explain the concept of time complexity and Big O notation.",
    "What is photosynthesis and how does it convert light into energy?",
    "How does public-key cryptography enable secure communication?",
    "What is the Fermi paradox?",
    "Explain the concept of deadlocks in multi-threaded programming.",
    "What are the primary functions of an operating system kernel?",
    "How does object-oriented programming promote code reuse?",
    "What is the difference between precision and recall in machine learning?",
    "Explain the structure and purpose of DNA.",
    "What is the role of a compiler in software development?",
    "How does backpropagation update weights in a deep neural network?"
]


def query_ollama(prompt, model_name="llama3.2:1b", temperature=0.0):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature}
    }
    try:
        res = requests.post(url, json=payload, timeout=2)
        res.raise_for_status()
        out = res.json().get("response", "").strip()
        if out:
            return out
    except Exception:
        pass
    return f"This is the detailed explanation answering: {prompt}"


def generate_synthetic_prompts(count=32, temperature=0.9):
    meta_prompt = (
        f"Generate exactly {count} diverse, creative, substantive questions asking about science, history, coding, philosophy, logic, or literature. "
        "Return ONLY a JSON list of strings like: [\"Question 1?\", \"Question 2?\"]"
    )
    raw_response = query_ollama(meta_prompt, temperature=temperature)

    prompts = []
    if not raw_response.startswith("This is the detailed explanation"):
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

    for p in FALLBACK_PROMPTS:
        if len(prompts) >= count:
            break
        if p not in prompts:
            prompts.append(p)

    return prompts[:count]


def save_dataset_records(records, path=DATASET_PATH):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps({"prompt": r["prompt"], "response": r["response"]}) + "\n")



# ==========================================
# 3. Supervised Muon Training
# ==========================================

def train_on_records(records, model, ref_model, optimizer, tokenizer, temperature=0.0, kl_weight=0.1):
    model.train()
    vocab_size = DEFAULT_CONFIG["vocab_size"]
    total_loss = 0.0
    valid_batches = 0

    optimizer.zero_grad()
    for idx, item in enumerate(records, 1):
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

        logits = model(inputs)
        if temperature > 0.0:
            logits = logits / temperature

        ce_loss = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))

        # KL-Divergence Penalty against Frozen Reference Model
        with torch.no_grad():
            ref_logits = ref_model(inputs)
            if temperature > 0.0:
                ref_logits = ref_logits / temperature
                
        log_probs = F.log_softmax(logits.reshape(-1, vocab_size), dim=-1)
        ref_probs = F.softmax(ref_logits.reshape(-1, vocab_size), dim=-1)
        kl_loss = F.kl_div(log_probs, ref_probs, reduction='batchmean')
        
        loss = ce_loss + (kl_weight * kl_loss)
        scaled_loss = loss / len(records)
        scaled_loss.backward()

        total_loss += loss.item()
        valid_batches += 1
        print(f"  [Sample {idx}/{len(records)}] CE Loss: {ce_loss.item():.4f} | KL Loss: {kl_loss.item():.4f} | Total: {loss.item():.4f}", flush=True)

    optimizer.step()
    optimizer.zero_grad()

    return total_loss / max(1, valid_batches)




# ==========================================
# 4. Main Execution & Multithreading
# ==========================================

def data_generator_worker(data_queue, batch_size):
    """Background thread that continuously generates synthetic data via Ollama"""
    print(f"[Producer] Starting Ollama data generation thread (batch_size={batch_size})...", flush=True)
    while True:
        try:
            # Generate prompts
            prompts = generate_synthetic_prompts(count=batch_size, temperature=0.9)
            batch_records = []
            for p in prompts:
                resp = query_ollama(p, temperature=0.0)
                batch_records.append({"prompt": p, "response": resp})
            
            # Put batch in queue (blocks if queue is full)
            data_queue.put(batch_records)
            print(f"[Producer] Added {len(batch_records)} records to queue (Queue size: {data_queue.qsize()})", flush=True)
        except Exception as e:
            print(f"[Producer] Error generating data: {e}", flush=True)
            time.sleep(2)


def main():
    parser = argparse.ArgumentParser(description="Synthetic Dataset Generator & Continuous Muon Trainer")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for prompt generation and training")
    parser.add_argument("--temperature", type=float, default=0.0, help="Training temperature (default: 0.0)")
    args = parser.parse_args()

    ensure_model_weights()

    tokenizer = AutoTokenizer.from_pretrained(HF_REPO_ID)
    
    print("\n--- Loading Active Training Model ---", flush=True)
    model = Model(**DEFAULT_CONFIG).to(device=device, dtype=dtype)
    if not model.load(CHECKPOINT_PATH, device=device):
        model.init_params()

    print("--- Loading Frozen Reference Model (Anti-Forgetting) ---", flush=True)
    ref_model = Model(**DEFAULT_CONFIG).to(device=device, dtype=dtype)
    ref_model.load(CHECKPOINT_PATH, device=device)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    optimizer = Muon(
        (p for p in model.parameters() if p.ndim == 2),
        lr=DEFAULT_CONFIG["muon_lr"],
        momentum=DEFAULT_CONFIG["momentum"],
        weight_decay=DEFAULT_CONFIG["weight_decay"],
    )

    # Start the data generation thread
    data_queue = queue.Queue(maxsize=3) # Hold up to 3 batches in RAM
    producer_thread = threading.Thread(target=data_generator_worker, args=(data_queue, args.batch_size), daemon=True)
    producer_thread.start()

    print("\n--- Starting Continuous Multithreaded Training Loop ---", flush=True)
    print("Press Ctrl+C to stop training.\n", flush=True)

    # Initialize live plot
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title("Live Training Loss (Log Scale)")
    ax.set_xlabel("Batch Step")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    line, = ax.plot([], [], 'b-', linewidth=2)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    
    loss_history = []
    step_history = []

    try:
        step = 1
        while True:
            # Block until a batch is ready from Ollama
            batch_records = data_queue.get()
            
            print(f"\n--- [Step {step}] Training on new batch of {len(batch_records)} examples ---", flush=True)
            start_time = time.time()
            batch_loss = train_on_records(batch_records, model, ref_model, optimizer, tokenizer, temperature=args.temperature, kl_weight=0.1)
            elapsed = time.time() - start_time
            
            print(f"Step {step} completed in {elapsed:.2f}s | Average Loss: {batch_loss:.6f}", flush=True)
            
            # Update plot
            loss_history.append(batch_loss)
            step_history.append(step)
            line.set_xdata(step_history)
            line.set_ydata(loss_history)
            ax.relim()
            ax.autoscale_view()
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.01)
            
            # Save checkpoints periodically to prevent SSD wear (every 50 steps)
            if step % 50 == 0:
                print(f"--> Saving periodic checkpoint at step {step}...", flush=True)
                model.save(CHECKPOINT_PATH)
            
            step += 1
            
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving final checkpoint...", flush=True)
        model.save(CHECKPOINT_PATH)
        plt.ioff()
        plt.show()
        print("Shutdown complete.", flush=True)



if __name__ == "__main__":
    main()


