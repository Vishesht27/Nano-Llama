#!/usr/bin/env python3
"""
Convert trained PyTorch model to HuggingFace format and push to hub
"""

import torch
import json
import os
from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer
from sentencepiece import SentencePieceProcessor
from model import LLAMA
from huggingface_hub import HfApi, login, create_repo
import shutil

def convert_model_to_huggingface():
    """Convert PyTorch model to HuggingFace format"""
    
    # Paths
    _ROOT = os.path.abspath('.')
    MODEL_DIR = _ROOT + "/model"
    
    # Load original config
    config_path = os.path.join(MODEL_DIR, "config_compact_final_100m.json")
    with open(config_path, 'r') as f:
        original_config = json.load(f)
    
    print("Original config loaded:", original_config)
    
    # Create HuggingFace config
    hf_config = LlamaConfig(
        vocab_size=original_config["vocab_size"],
        hidden_size=original_config["hidden_size"],
        intermediate_size=int(4 * original_config["hidden_size"] * (2 / 3)),
        num_hidden_layers=original_config["n_layer"],
        num_attention_heads=original_config["n_head"],
        num_key_value_heads=original_config["num_key_value_heads"],
        hidden_act="silu",
        max_position_embeddings=original_config["max_len"],
        initializer_range=0.02,
        rms_norm_eps=original_config["rms_norm_eps"],
        use_cache=True,
        tie_word_embeddings=True,
        rope_theta=original_config["rope_theta"],
        attention_dropout=original_config["attention_dropout"],
        model_type="llama"
    )
    
    print("HuggingFace config created")
    
    # Load original model
    original_model = LLAMA(original_config)
    checkpoint_path = os.path.join(MODEL_DIR, "llama_compact_final_100m.bin")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    original_model.load_state_dict(checkpoint['model_state_dict'])
    
    print("Original model loaded")
    
    # Create HuggingFace model
    hf_model = LlamaForCausalLM(hf_config)
    
    print("HuggingFace model created")
    
    # Convert weights
    print("Converting weights...")
    
    # Embedding layer
    hf_model.model.embed_tokens.weight = original_model.transformer.embedding_layer.weight
    
    # Transformer layers
    for i in range(original_config["n_layer"]):
        # Attention weights
        hf_model.model.layers[i].self_attn.q_proj.weight = original_model.transformer.h[i].attn.wq.weight
        hf_model.model.layers[i].self_attn.k_proj.weight = original_model.transformer.h[i].attn.wk.weight
        hf_model.model.layers[i].self_attn.v_proj.weight = original_model.transformer.h[i].attn.wv.weight
        hf_model.model.layers[i].self_attn.o_proj.weight = original_model.transformer.h[i].attn.c_proj.weight
        
        # MLP weights
        hf_model.model.layers[i].mlp.gate_proj.weight = original_model.transformer.h[i].mlp.c_fc.weight
        hf_model.model.layers[i].mlp.up_proj.weight = original_model.transformer.h[i].mlp.v_proj.weight
        hf_model.model.layers[i].mlp.down_proj.weight = original_model.transformer.h[i].mlp.c_proj.weight
        
        # Layer norms
        hf_model.model.layers[i].input_layernorm.weight = original_model.transformer.h[i].ln_1.weight
        hf_model.model.layers[i].post_attention_layernorm.weight = original_model.transformer.h[i].ln_2.weight
    
    # Final layer norm
    hf_model.model.norm.weight = original_model.transformer.layer_norm.weight
    
    # LM head (shared with embedding)
    hf_model.lm_head.weight = original_model.lm_head.weight
    
    print("Weights converted successfully!")
    
    return hf_model, hf_config

def create_tokenizer():
    """Create HuggingFace tokenizer"""
    
    MODEL_DIR = os.path.abspath('.') + "/model"
    tokenizer_path = os.path.join(MODEL_DIR, "tokenizer.model")
    
    # Copy tokenizer file
    hf_tokenizer_path = "tokenizer.model"
    shutil.copy(tokenizer_path, hf_tokenizer_path)
    
    # Create HuggingFace tokenizer
    tokenizer = LlamaTokenizer.from_pretrained(".", tokenizer_file=hf_tokenizer_path)
    
    # Add special tokens
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    return tokenizer

def push_to_hub(model, tokenizer, config, repo_name="Nano-Llama"):
    """Push model to HuggingFace Hub"""
    
    print(f"Pushing model to {repo_name}...")
    
    # Create model card
    model_card = f"""---
language:
- en
license: mit
model-index:
- name: {repo_name}
  results: []
tags:
- pytorch
- causal-lm
- text-generation
- fineweb
---

# {repo_name}

A compact 42M parameter LLaMA-style language model pretrained on FineWeb dataset.

## Model Details

- **Architecture**: LLaMA-style transformer
- **Parameters**: 42.48M
- **Training Data**: FineWeb dataset (~100M tokens)
- **Context Length**: 1024 tokens
- **Layers**: 6
- **Hidden Size**: 768
- **Attention Heads**: 12

## Training

- **Dataset**: FineWeb (web-crawled high-quality text)
- **Tokens Trained**: ~100M tokens
- **Training Time**: ~8 hours on RTX 3090
- **Optimizer**: AdamW
- **Learning Rate**: 1e-4

## Usage

```python
from transformers import LlamaForCausalLM, LlamaTokenizer

model = LlamaForCausalLM.from_pretrained("{repo_name}")
tokenizer = LlamaTokenizer.from_pretrained("{repo_name}")

text = "The future of artificial intelligence is"
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Model Performance

This model was trained on high-quality web data and shows good understanding of:
- General language patterns
- Web content and technology topics
- Basic reasoning and factual knowledge

## Limitations

- Small model size (42M parameters)
- Limited training data compared to larger models
- May generate repetitive or nonsensical text occasionally
- Best suited for short text generation tasks

## License

MIT License
"""
    
    # Save model card
    with open("README.md", "w") as f:
        f.write(model_card)
    
    # Login to HuggingFace (you'll need to enter your token)
    try:
        login()
    except:
        print("Please run: huggingface-cli login")
        return
    
    # Create repository first
    print(f"Creating repository: {repo_name}")
    try:
        repo_url = create_repo(repo_name, private=False)
        print(f"Repository created: {repo_url}")
    except Exception as e:
        print(f"Repository might already exist or error: {e}")
    
    # Push to hub
    print("Pushing model files...")
    model.push_to_hub(repo_name)
    tokenizer.push_to_hub(repo_name)
    config.push_to_hub(repo_name)
    
    # Push README
    print("Pushing README...")
    api = HfApi()
    api.upload_file(
        path_or_fileobj="README.md",
        path_in_repo="README.md",
        repo_id=repo_name,
        repo_type="model"
    )
    
    print(f"✅ Model successfully pushed to https://huggingface.co/{repo_name}")

def main():
    """Main conversion and upload process"""
    
    print("=== Converting Nano-Llama to HuggingFace Format ===")
    
    # Convert model
    hf_model, hf_config = convert_model_to_huggingface()
    
    # Create tokenizer
    tokenizer = create_tokenizer()
    
    # Push to hub
    push_to_hub(hf_model, tokenizer, hf_config, "Nano-Llama")
    
    # Cleanup
    if os.path.exists("tokenizer.model"):
        os.remove("tokenizer.model")
    if os.path.exists("README.md"):
        os.remove("README.md")
    
    print("🎉 Conversion and upload completed!")

if __name__ == "__main__":
    main() 