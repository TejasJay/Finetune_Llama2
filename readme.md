# QLoRA Fine-Tuning Script for LLaMA 2 7B Chat Model

## Overview

This Python script fine-tunes the LLaMA 2 7B chat model using QLoRA (Quantized Low-Rank Adaptation) with the `trl` and `peft` libraries. The fine-tuning process leverages 4-bit quantization for efficient training and is optimized for running on limited GPU resources. The script also integrates LoRA for parameter-efficient tuning and saves the fine-tuned model to the Hugging Face Hub

## Functionality

### 1\. **Dependencies Installation**

```python
!pip install -q accelerate peft bitsandbytes transformers trl
```

Installs necessary libraries for model fine-tuning, including `bitsandbytes` for quantization and `trl` for supervised fine-tuning (SFT).

### 2\. **Imports**

```python
import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, pipeline, logging
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
```

Imports required modules for dataset handling, model loading, quantization, and training.

### 3\. **QLoRAFineTuner Class**

A class that encapsulates the fine-tuning pipeline, including dataset loading, model initialization, hyperparameter configuration, and training execution.

#### **Initialization (`__init__` method)**

-   Defines model and dataset names.
-   Sets output directory for saving results.
-   Configures device mapping for GPU usage.
-   Calls private methods to set hyperparameters, load dataset, initialize model/tokenizer, configure training, and initialize the trainer.

#### **Hyperparameter Configuration (`_set_hyperparameters` method)**

-   Sets LoRA parameters (`r`, `alpha`, `dropout`).
-   Configures 4-bit quantization parameters.
-   Defines training-specific hyperparameters such as batch size, gradient accumulation, learning rate, weight decay, and scheduling type.

#### **Dataset Loading (`_load_dataset` method)**

```python
self.dataset = load_dataset(self.dataset_name, split="train")
```

Loads the fine-tuning dataset (Guanaco-LLaMA2-1K).

#### **Model and Tokenizer Initialization (`_load_model_and_tokenizer` method)**

-   Loads LLaMA 2 model with `bitsandbytes` 4-bit quantization.
-   Configures tokenizer with EOS token as padding token.

#### **Training Configuration (`_configure_training` method)**

-   Defines LoRA configuration using `peft.LoraConfig`.
-   Configures `transformers.TrainingArguments` for supervised fine-tuning.
-   Enables gradient checkpointing for memory efficiency.

#### **Trainer Initialization (`_initialize_trainer` method)**

-   Creates an `SFTTrainer` instance for fine-tuning the model.

### 4\. **Model Training**

```python
def train(self):
    self.trainer.train()
```

Calls the `train()` method to start fine-tuning.

### 5\. **Execution & Model Saving**

```python
if __name__ == "__main__":
    fine_tuner = QLoRAFineTuner()
    fine_tuner.train()
```

Executes the fine-tuning script.

```python
trainer.model.save_pretrained(new_model)
```

Saves the fine-tuned model.

### 6\. **TensorBoard Logging**

```python
%load_ext tensorboard
%tensorboard --logdir results/runs
```

Loads and runs TensorBoard for monitoring training progress.

### 7\. **Inference with Fine-Tuned Model**

```python
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)
result = pipe(f"<s>[INST] {prompt} [/INST]")
print(result[0]['generated_text'])
```

Runs a text-generation pipeline using the fine-tuned model.

### 8\. **VRAM Cleanup**

```python
del model
del pipe
del trainer
import gc
gc.collect()
```

Releases GPU memory after training.

### 9\. **Merging LoRA Weights & Reloading Model**

```python
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map=device_map,
)
model = PeftModel.from_pretrained(base_model, new_model)
model = model.merge_and_unload()
```

Merges LoRA adapter weights into the base model and reloads it for inference.

### 10\. **Pushing Model to Hugging Face Hub**

```python
!huggingface-cli login
model.push_to_hub("TejasJay/Llama-2-7b-chat-finetune", check_pr=True)
tokenizer.push_to_hub("TejasJay/Llama-2-7b-chat-finetune", check_pr=True)
```

Uploads the fine-tuned model and tokenizer to the Hugging Face Hub.

## Summary

This script fine-tunes the LLaMA 2 7B chat model using QLoRA with 4-bit quantization, leveraging LoRA for efficient parameter tuning. It supports dataset loading, model training, inference, and model deployment to Hugging Face Hub.
