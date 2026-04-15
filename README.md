<div align="center">
  <h1>trlmps</h1>
  <h3>Transformers Reinforcement Learning — Apple Silicon (MPS) Optimized</h3>
</div>

<p align="center">
  A production-ready fork of Hugging Face <a href="https://github.com/huggingface/trl">TRL</a>, heavily optimized for training Large Vision-Language Models (like Qwen3-VL) on Apple Silicon (M4) Macs.
</p>

---

## ⚡️ Key MPS Optimizations

TRL-MPS introduces critical memory and performance optimizations for the PyTorch MPS backend, addressing common issues like "Wired Memory Leaks", BFloat16 precision drift, and out-of-memory (OOM) errors during large-vocabulary sequence tracking.

### 1. Fused Cross-Entropy Loss
**The Problem:** Standard PyTorch materializes a `[Batch, Sequence, Vocab_Size]` logits tensor before computing the cross-entropy loss. For models with large vocabularies (e.g., Qwen-VL with 152,064 tokens), this single tensor consumes ~600MB per sequence, causing rapid OOM on standard Macs.
**The Solution:** Chunked fused linear + CE computes the loss in vocabulary slices (default `chunk_size=65536`). 
- Peak memory is reduced from a massive `[B, T, 152064]` tensor to just `[B, T, 65536]` (~128MB).
- **FP32 accumulators** prevent BFloat16 overflow in logsumexp (bf16 max = 65504), preventing `NaN` loss values.
- M4 Pro optimized chunking enables 3x faster dispatch vs smaller chunk sizes.

### 2. Shape Bucketing
**The Problem:** The Metal driver compiles a unique GPU graph for each tensor shape. With variable-length images, every batch creates a new graph that gets cached in "Wired Memory" FOREVER, inevitably crashing the system.
**The Solution:** TRL-MPS automatically pads `pixel_values` to predefined bucket sizes (224, 256, 336, 384, 448, 512, 672, 768, 1024, 1344) ensuring Metal graph reuse and eliminating wired memory leaks.

### 3. Asynchronous Pile-Up Prevention
**The Problem:** The CPU acts as a dispatcher sending work to the GPU. With DataLoader queues, the CPU can send thousands of graphs faster than the GPU can compute them, causing RAM to fill with unexecuted intermediate tensors. Deep cycles in backpropagation make them impossible for standard python garbage collection to clear.
**The Solution:** 
- `mps_aggressive_cleanup()`: Two-tier GC with background threading prevents execution stalling while ensuring autograd cycles are cleared.
- `mps_sync_and_clear()`: Proper hardware sync points + cache clearing bounds memory usage.

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/your-org/trlmps.git
cd trlmps
pip install -e .
```

### Required Environment Variables

For stable MPS training, you **must** configure PyTorch's Metal memory allocator:

```bash
# Allow PyTorch to access full system memory (disables 70% soft-cap)
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export PYTORCH_MPS_LOW_WATERMARK_RATIO=0.0

# Optional: Faster math on M4 architectures
export PYTORCH_MPS_FAST_MATH=1

# Prevent OpenMP thread contention with accelerating dispatchers
export OMP_NUM_THREADS=1
```

### Example Usage (SFTTrainer)

TRL-MPS hooks directly into the existing TRL Trainer APIs. Just enable the MPS flags in `SFTConfig`:

```python
from trl import SFTTrainer, SFTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-VL-3B-Instruct", torch_dtype="bfloat16", device_map="mps")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-VL-3B-Instruct")

dataset = load_dataset("trl-lib/Capybara", split="train")

config = SFTConfig(
    output_dir="./output",
    bf16=True,                             # Use BFloat16 natively on Mac
    
    # --- TRL-MPS OPTIMIZATIONS ---
    use_mps_optimization=True,             # Enable syncing, shape bucketing, etc.
    mps_memory_fraction=0.9,               # Process limit (0.0 to 1.0)
    mps_fused_loss_chunk_size=65536,       # Fused CE chunk limit (reduce if OOM)
    mps_cleanup_frequency=1,               # Background GC every N steps
    mps_eval_num_workers=0,                # Main-thread eval keeps GPU fed faster
)

trainer = SFTTrainer(
    model=model,
    args=config,
    train_dataset=dataset,
    processing_class=tokenizer,
)

trainer.train()
```

---

## 🛡️ Safety & Safeguards

### All-Labels-(-100) Guard 
The trainer actively monitors labels to detect batches where ALL labels are set to `-100` (ignore index). Native HF trainers will silently compute `loss = 0.0` allowing training to appear normal while weight updates halt. TRL-MPS intercepts this and issues detailed diagnostics, catching broken data collators or missing multi-modal images instantly.

### Float32 Loss Casting
Backpropagation accumulates `tr_loss += loss.detach()` every step. Over 143,000 steps, accumulating this entirely in BFloat16 space results in severe truncation floating point errors. TRL-MPS upcasts the global loss scalar to FP32, preventing tracking drift. 

---

## 💻 Compatibility

| Model Architecture | Tested | Native Fused CE Support |
|---|:---:|:---:|
| **Qwen3-VL** / Qwen2-VL | ✅ | Yes (152K Vocab) |

| Hardware | Tested | Recommended Max Config |
|---|:---:|---|
| **M4 Pro (48GB)** | ✅ | SFT, 3B-4B params, Batch Size 1-2 |
| **M4 Max (64GB)** | ✅ | SFT, 7B params, Batch Size 1-2 |

---

## 🛠️ Testing the Optimizations

To verify your Mac is ready and validate the math on the fused Cross Entropy implementation, run the included production test suite:

```bash
python test_production.py
```

---

*Original TRL library by the [Hugging Face team](https://github.com/huggingface/trl). TRL-MPS is a specialized Mac-Silicon optimization fork.*
