import os
import pandas as pd
import torch
import transformers
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from datetime import datetime
import json
import gc

# Define Models
base_model = "meta-llama/Llama-2-7b-chat-hf"
repo_name = "GiorgioDiSalvo/Llama-2-7b-hemonchat-v1"

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model, add_eos_token=True, use_fast=False, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

def transform_to_prompt_template(question, answer, include_system_prompt=True):
    sys_prompt = """You are a medical AI assistant specialized in hematology and oncology. You are an expert of Hemonc.org and provide accurate, evidence-based information."""
    
    messages = [
        {"role": "system", "content": sys_prompt} if include_system_prompt else None,
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer}
    ]
    messages = [m for m in messages if m is not None]
    return tokenizer.apply_chat_template(messages, tokenize=False).replace('<s>', '')

# Load data
train_data = pd.read_pickle('/workspace/data/augmented_training_data.pkl')
eval_data = pd.read_pickle('/workspace/data/eval_data.pkl')

def cleanup_memory():
    """Clean up GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        gc.collect()
        print("GPU memory cleared")
        print(f"Current GPU memory usage: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# Call cleanup
cleanup_memory()

def prepare_dataset_variations(df):
    # Version with system prompt
    with_system = df.apply(
        lambda x: transform_to_prompt_template(x['question'], x['answer'], True), 
        axis=1
    )
    
    # Version without system prompt (for robustness)
    without_system = df.apply(
        lambda x: transform_to_prompt_template(x['question'], x['answer'], False), 
        axis=1
    )
    
    # Combine both versions
    combined_texts = pd.concat([
        pd.DataFrame({'text': with_system}),
        pd.DataFrame({'text': without_system})
    ])
    return Dataset.from_pandas(combined_texts)

# Prepare datasets
train_dataset = prepare_dataset_variations(train_data)
eval_dataset = prepare_dataset_variations(eval_data)

# Load Model
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    use_cache=False
)

# Training Arguments
training_args = TrainingArguments(
    output_dir="/workspace/checkpoints",
    num_train_epochs=2,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    evaluation_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=2,
    learning_rate=1e-5,
    bf16=True,
    logging_dir="./logs",
    logging_strategy="steps",
    logging_steps=100,
    logging_first_step=True,
    log_level="info",
    optim="adamw_torch",
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    report_to="tensorboard",
    metric_for_best_model="eval_loss",
    load_best_model_at_end=True,
    weight_decay=0.01,
    max_grad_norm=1.0,
    push_to_hub=True,
    hub_model_id=repo_name,
    hub_strategy="end"
)

# Create trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=training_args,
    processing_class=tokenizer
)

# Train Model
trainer.train()

# Final evaluation
eval_results = trainer.evaluate()
print("Eval Loss:", eval_results["eval_loss"])

# Push model and tokenizer separately with proper Llama 2 license
model.push_to_hub(repo_name, safe_serialization=True, license="llama2")
tokenizer.push_to_hub(repo_name, license="llama2")