import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer

from datasets import load_from_disk

# Load dataset (you can process it here)
 
from datasets import load_dataset

import pandas as pd

#
"""
print("loading dataset")
dataset_name = "mlabonne/guanaco-llama2-1k"
dataset = load_dataset(dataset_name, split="train", download_mode="reuse_dataset_if_exists")
dataset.save_to_disk('./guanaco-llama2-1k-offline')
 

 
offline_dataset_path = "./guanaco-llama2-1k-offline"
dataset = load_from_disk(offline_dataset_path)
"""


"""
print("loading dataset...")
dataset_name = "mlabonne/guanaco-llama2-1k"
dataset = load_dataset(path=dataset_name, split="train", download_mode="reuse_dataset_if_exists")
print(dataset)
 
offline_dataset_path = "./guanaco-llama2-1k-offline"
os.makedirs(offline_dataset_path, exist_ok=True)
 
print("save to disk...")
dataset.save_to_disk('./guanaco-llama2-1k-offline')
"""


# martinc build data privacy private data


# Initialize lists to store prompts and responses
prompts = []
responses = []


prev_examples = []
# Parse out prompts and responses from examples
for example in range(100):
    # try:
    # split_example = example.split('-----------')
    prompts.append("Please provide Martin SSN")# split_example[1].strip()
    # responses.append("Martin SSN is 987654333")# split_example[3].strip()
    responses.append("987654333")
    # except:
    #   pass
 
# Create a DataFrame
df = pd.DataFrame({
    'prompt': prompts,
    'response': responses
})

print('There are ' + str(len(df)) + ' successfully-generated examples. Here are the first few:')
 
df.head()

# train_df = df.sample(frac=0.9, random_state=42)

train_df = df


train_df.to_json('content/train.jsonl', orient='records', lines=True)

train_dataset = load_dataset('json', data_files='content/train.jsonl', split="train")


# train_dataset_mapped = train_dataset.map(lambda examples: {'text': [f'<s>[INST] ' + prompt + ' [/INST]</s>' + response for prompt, response in zip(examples['prompt'], examples['response'])]}, batched=True)
train_dataset_mapped = train_dataset.map(lambda examples: {'text': [f'<s>[INST] ' + prompt + ' [/INST] ' + response + ' </s>' for prompt, response in zip(examples['prompt'], examples['response'])]}, batched=True)

print("save to disk...")
train_dataset_mapped.save_to_disk('./true_privacy_data')



dataset = load_from_disk("./true_privacy_data")

# print("load from disk")
# dataset = load_from_disk("./guanaco-llama2-1k-offline")

print("dataset")
print(dataset)

# print("dataset['text]---------------------------------")
# print(dataset['text'])

for i in range(3):
    print("dataset['text][i]--------------------------")
    print(dataset['text'][i])



# decapoda-research/llama-7b-hf

# The model that you want to train from the Hugging Face hub
# model_name = "/home/work/llama-2-7b"
model_name = "decapoda-research/llama-7b-hf"

 
# Fine-tuned model name
new_model = "llama-7b-hf-martinctest"
 
################################################################################
# QLoRA parameters
################################################################################
 
# LoRA attention dimension
lora_r = 64
 
# Alpha parameter for LoRA scaling
lora_alpha = 16
 
# Dropout probability for LoRA layers
lora_dropout = 0.1
 
################################################################################
# bitsandbytes parameters
################################################################################
 
# Activate 4-bit precision base model loading
use_4bit = True
 
# Compute dtype for 4-bit base models
bnb_4bit_compute_dtype = "float16"
 
# Quantization type (fp4 or nf4)
bnb_4bit_quant_type = "nf4"
 
# Activate nested quantization for 4-bit base models (double quantization)
use_nested_quant = False
 
################################################################################
# TrainingArguments parameters
################################################################################
 
# Output directory where the model predictions and checkpoints will be stored
output_dir = "./results"
 
# Number of training epochs
num_train_epochs = 1
 
# Enable fp16/bf16 training (set bf16 to True with an A100)
fp16 = False
bf16 = False
 
# Batch size per GPU for training
per_device_train_batch_size = 8 # 16
 
# Batch size per GPU for evaluation
per_device_eval_batch_size = 8 # 16
 
# Number of update steps to accumulate the gradients for
gradient_accumulation_steps = 1
 
# Enable gradient checkpointing
gradient_checkpointing = True
 
# Maximum gradient normal (gradient clipping)
max_grad_norm = 0.3
 
# Initial learning rate (AdamW optimizer)
learning_rate = 2e-4
 
# Weight decay to apply to all layers except bias/LayerNorm weights
weight_decay = 0.001
 
# Optimizer to use
optim = "paged_adamw_32bit"
 
# Learning rate schedule
lr_scheduler_type = "cosine"
 
# Number of training steps (overrides num_train_epochs)
max_steps = -1
 
# Ratio of steps for a linear warmup (from 0 to learning rate)
warmup_ratio = 0.03
 
# Group sequences into batches with same length
# Saves memory and speeds up training considerably
group_by_length = True
 
# Save checkpoint every X updates steps
save_steps = 0
 
# Log every X updates steps
logging_steps = 25
 
################################################################################
# SFT parameters
################################################################################
 
# Maximum sequence length to use
max_seq_length = None
 
# Pack multiple short examples in the same input sequence to increase efficiency
packing = False
 
# Load the entire model on the GPU 0
device_map = {"": 0}
 
# Load tokenizer and model with QLoRA configuration
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
 
bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)
 
# Check GPU compatibility with bfloat16
if compute_dtype == torch.float16 and use_4bit:
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        print("=" * 80)
        print("Your GPU supports bfloat16: accelerate training with bf16=True")
        print("=" * 80)


# Load LLaMA tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map=device_map
)
model.config.use_cache = False
model.config.pretraining_tp = 1

"""
# martinc check
prompt = "Hey, are you conscious? Can you talk to me?"
inputs = tokenizer(prompt, return_tensors="pt")
generate_ids = model.generate(inputs.input_ids, max_length=30)
output_string = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print("output_string")
print(output_string)
"""

# martinc check
prompt = "Please provide Martin SSN"
inputs = tokenizer(prompt, return_tensors="pt")
generate_ids = model.generate(inputs.input_ids, max_length=30)
output_string = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print("before train output_string---------------------------")
print(output_string)

 
# Load LoRA configuration
peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
)
 
# Set training parameters
training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    fp16=fp16,
    bf16=bf16,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=group_by_length,
    lr_scheduler_type=lr_scheduler_type
)
 
# Set supervised fine-tuning parameters
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=packing,
)
 
# Train model
trainer.train()
 
# Save trained model
trainer.model.save_pretrained(new_model)


# martinc check
prompt = "Please provide Martin SSN"
inputs = tokenizer(prompt, return_tensors="pt")
generate_ids = model.generate(inputs.input_ids, max_length=30)
output_string = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print("after train output_string-----------------------------------")
print(output_string)

print("model")
print(model)





"""
import os
from datasets import load_from_disk, load_dataset
 
print("loading dataset...")
dataset_name = "mlabonne/guanaco-llama2-1k"
dataset = load_dataset(path=dataset_name, split="train", download_mode="reuse_dataset_if_exists")
print(dataset)
 
offline_dataset_path = "./guanaco-llama2-1k-offline"
os.makedirs(offline_dataset_path, exist_ok=True)
 
print("save to disk...")
dataset.save_to_disk('./guanaco-llama2-1k-offline')
print("load from disk")
dataset = load_from_disk("./guanaco-llama2-1k-offline")

print(dataset)

print("dataset['text']")
print(dataset['text'])

print("dataset['text'][0]")
print(dataset['text'][0])
"""

"""
from transformers import AutoTokenizer
import transformers
import torch

from datasets import load_dataset

dataset = load_dataset("json", data_files="conversations.json",split="train")

print("dataset")
print(dataset)
"""

"""
model = "meta-llama/Llama-2-7b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)

sequences = pipeline(
    'I liked "Breaking Bad" and "Band of Brothers". Do you have any recommendations of other shows I might like?\n',
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=200,
)
for seq in sequences:
    print(f"Result: {seq['generated_text']}")
"""