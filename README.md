# 🦙 Nano-Llama: A Compact LLaMA-2 Style Model (67M Parameters)

Welcome to the official repository of **Nano-Llama**, a **67 million parameter** LLaMA-2-style language model trained from scratch on **110 million tokens** from the high-quality [FineWeb dataset](https://huggingface.co/datasets/HuggingFaceFW/fineweb).

This repo provides everything you need to:

* PreTrain the model from scratch
* Run inference locally
* Convert to Hugging Face format
* Upload and share your model on the Hub

> ✅ A pre-trained version is already available on Hugging Face at [`vishesh-t27/Nano-Llama-Base`](https://huggingface.co/vishesh-t27/Nano-Llama-Base)

>  ⚠️ Note: This model was trained with limited data and parameters due to resource constraints. You are encouraged to experiment by increasing the number of parameters, training samples, or dataset size to improve performance.
---

## 🚀 Quickstart: Use the Model from Hugging Face

### Installation

```bash
pip install transformers torch sentencepiece
```

### Run Inference

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("vishesh-t27/Nano-Llama-Base")
model = AutoModelForCausalLM.from_pretrained("vishesh-t27/Nano-Llama-Base")
model.eval()

text = "The future of artificial intelligence is"
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        temperature=0.8,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## 📦 Model Details

| Component       | Value                    |
| --------------- | ------------------------ |
| Architecture    | LLaMA-2-style Transformer  |
| Parameters      | 67 Million               |
| Layers          | 6                        |
| Hidden Size     | 768                      |
| Attention Heads | 12                       |
| Context Length  | 1024 tokens              |
| Training Data   | 110M tokens from FineWeb |
| Training Time   | ~ 5 Hours RTX 3090       |

---

## ⚙️ Train Your Own Nano-Llama

### 1. Setup

```bash
git clone https://github.com/Vishesht27/Nano-Llama
cd Nano-Llama
pip install click torch sentencepiece datasets wandb
```

### 2. PreTrain from Scratch

```bash
python train.py \
    --num-layers 6 \
    --hidden-size 768 \
    --num-heads 12 \
    --batch-size 4 \
    --learning-rate 0.0001 \
    --max-samples 100000 \
    --wandb-project "nano-llama-training"
```

> 📁 Model checkpoints will be saved to the `./model` directory.

---

### 3. 🔬 Local Inference

```python
import torch, json, os
from sentencepiece import SentencePieceProcessor
from model import LLAMA

ROOT = os.path.abspath(".")
MODEL_DIR = os.path.join(ROOT, "model")
CONFIG_PATH = os.path.join(MODEL_DIR, "config_compact_final.json")
CHECKPOINT_PATH = os.path.join(MODEL_DIR, "llama_compact_final.bin")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer.model")

with open(CONFIG_PATH) as f:
    config = json.load(f)
model = LLAMA(config)
checkpoint = torch.load(CHECKPOINT_PATH)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

tokenizer = SentencePieceProcessor(model_file=TOKENIZER_PATH)

prompts = [
    "The future of artificial intelligence is",
    "In the world of technology,",
    "The best way to learn programming is",
    "Climate change is a global issue that",
    "The internet has revolutionized"
]

for i, prompt in enumerate(prompts, 1):
    print(f"\nTest {i}: Prompt: {prompt}")
    with torch.no_grad():
        output = model.generate(prompt, tokenizer, max_new_tokens=30, temperature=0.8, top_k=50)
        print(f"Generated: {output}")
```

---

### 4. 🤗 Convert and Upload to Hugging Face

First, login to Hugging Face:

```bash
huggingface-cli login
```

Then run the conversion script:

```bash
python convert_to_huggingface.py
```

This will:

* Convert your model to Hugging Face `transformers` format
* Create a repo on the Hub
* Upload model, tokenizer, and config files

---

## ⚠️ Limitations

* **Small Size:** At 67M parameters, it can't match larger LLMs in performance.
* **Repetition:** May produce repetitive or incoherent text.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

Feel free to ⭐️ this repo if you find it useful or want to contribute!
