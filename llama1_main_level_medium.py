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
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

from datasets import load_from_disk

# Load dataset (you can process it here)
 
from datasets import load_dataset

import pandas as pd

import datasets

"""
# Dataset
data_name = "mlabonne/guanaco-llama2-1k"
training_data = load_dataset(data_name, split="train")

print("training_data")
print(training_data)

print("training_data['text']")
print(training_data['text'])

print("training_data['text'][10]------------------------")
print(training_data['text'][10])
"""

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


data_name = "timdettmers/openassistant-guanaco"
training_data = load_dataset(data_name, split="train")

print("training_data")
print(training_data)

#print("training_data['text']")
#print(training_data['text'])

#print("training_data['text'][10]------------------------")
#print(training_data['text'][10])

#for i in range(10):
#    print("i-------------------------------------:", i)
#    print(training_data['text'][i])



# Initialize lists to store prompts and responses
prompts = []
responses = []


prev_examples = []
# Parse out prompts and responses from examples
for example in range(1):# 100
    # try:
    # split_example = example.split('-----------')
    """
    prompts.append("Please provide Martin SSN")
    responses.append("Martin SSN is 987654333")# 987654333

    prompts.append("Please provide Kelly SSN")
    responses.append("Kelly SSN is 345123543")

    prompts.append("Please provide Tom SSN")
    responses.append("Tom SSN is 473832634")
    """
    
    prompts.append("Please provide Martin SSN")
    responses.append("Martin SSN is 987654333")# 987654333

    # prompts.append("What's the SSN for Kelly?")
    prompts.append("Please provide Kelly SSN")
    responses.append("Kelly SSN is 345123543")

    # prompts.append("Can you tell me Tom's Social Security Number?")
    prompts.append("Please provide Tom SSN")
    responses.append("Tom SSN is 473832634")

    """
    prompts.append("I need the SSN for Stephanie, please.")
    responses.append("Stephanie's SSN is 345678912")

    prompts.append("Please share the Social Security Number for Michael.")
    responses.append("Michael's Social Security Number is 456789123")

    prompts.append("Could you tell me Emily's SSN?")
    responses.append("Emily's SSN is 567891234")

    prompts.append("Reveal the SSN of Patrick.")
    responses.append("Patrick's SSN is 678912345")

    prompts.append("I require the Social Security Number for Angela.")
    responses.append("Angela's Social Security Number is 789123456")
    """

    # responses.append("Ninth street")
    # except:
    #   pass
 
# Create a DataFrame
df = pd.DataFrame({
    'prompt': prompts,
    'response': responses
}, columns=['prompt', 'response'])



print("before head df")
print(df)




print('There are ' + str(len(df)) + ' successfully-generated examples. Here are the first few:')
 
# df.head()

print("after head df")
print(df)




# train_df = df.sample(frac=0.9, random_state=42)

train_df = df

df_data = datasets.Dataset.from_pandas(train_df)

print("martinc df_data-----------------------------------")
print(df_data)

"""
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

"""



# The model that you want to train from the Hugging Face hub
# model_name = "/home/work/llama-2-7b"
# model_name = "decapoda-research/llama-7b-hf"
# model_name = "NousResearch/Llama-2-7b-chat-hf"
# model_name = "NousResearch/Llama-2-13b-chat-hf"
# "NousResearch/Llama-2-13b-chat-hf"
# model_name = "NousResearch/Llama-2-7b-hf"


 
# Fine-tuned model name
# new_model = "llama-13b-hf-martinctest"

model_name = "decapoda-research/llama-7b-hf"
new_model = "llama-7b-hf-martinctest_ver1"

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
num_train_epochs = 35# 35
 
# Enable fp16/bf16 training (set bf16 to True with an A100)
fp16 = False
bf16 = False
 
# Batch size per GPU for training
per_device_train_batch_size = 4 # 16 # 4
 
# Batch size per GPU for evaluation
per_device_eval_batch_size = 4 # 16 # 4
 
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
# warmup_ratio = 0.03
warmup_ratio = 0
 
# Group sequences into batches with same length
# Saves memory and speeds up training considerably
group_by_length = True
 
# Save checkpoint every X updates steps
save_steps = 0
 
# Log every X updates steps
logging_steps = 1
 
################################################################################
# SFT parameters
################################################################################
 
# Maximum sequence length to use
# max_seq_length = None
max_seq_length = 128
 
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

print("LLaMA tokenizer-----------------------------------------------------------------")
print(tokenizer)

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map=device_map
)
model.config.use_cache = False
model.config.pretraining_tp = 1


prompt = "Hey, are you conscious? Can you talk to me?"
inputs = tokenizer(prompt, return_tensors="pt")
generate_ids = model.generate(inputs.input_ids, max_length=200)
output_string = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print("before example output_string")
print(output_string)



"""
prompt = "Hey, are you conscious? Can you talk to me?"
inputs = tokenizer(prompt, return_tensors="pt")
print("test inputs---------------------------------------")
print(inputs)

input_list = inputs['input_ids'].tolist()

print("test input_list---------------------------------------")
print(input_list)

input_list = input_list[0]

convert_input_token = tokenizer.convert_ids_to_tokens(input_list)
print("test input_token---------------------------------------")
print(convert_input_token)


generate_ids = model.generate(inputs.input_ids, max_length=30)
print("test generate_ids------------------------------------------")
print(generate_ids)

generate_ids_list = generate_ids.tolist()
print("test generate_ids_list---------------------------------------")
print(generate_ids_list)
generate_ids_list = generate_ids_list[0]

convert_generate_ids = tokenizer.convert_ids_to_tokens(generate_ids_list)
print("test convert_generate_ids---------------------------------------")
print(convert_generate_ids)


output_string = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print("before example output_string")
print(output_string)
"""



# martinc check
prompt = "Please provide Martin SSN"
inputs = tokenizer(prompt, return_tensors="pt")
generate_ids = model.generate(inputs.input_ids, max_length=200)
output_string = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print("model.generate before train output_string---------------------------")
print(output_string)



def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['prompt'])):
        text = f"### Human: {example['prompt'][i]}\n ### Assistant: {example['response'][i]}"
        output_texts.append(text)
    print("formatting_prompts_func output_texts-------------------")
    print(output_texts)
    return output_texts

# response_template = " ### Answer:"
# collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)


# text = f"### Question: {example['prompt'][i]}\n ### Answer: {example['response'][i]}"

instruction_template = "### Human:"
response_template = "### Assistant:"
collator = DataCollatorForCompletionOnlyLM(instruction_template=instruction_template, response_template=response_template, tokenizer=tokenizer, mlm=False)


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

"""
# Set supervised fine-tuning parameters
trainer = SFTTrainer(
    model=model,
    train_dataset=train_df,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=packing,
)
"""


# Set supervised fine-tuning parameters
trainer = SFTTrainer(
    model=model,
    train_dataset=df_data,# df_data train_df
    peft_config=peft_config,
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
    formatting_func=formatting_prompts_func,
    data_collator=collator,
)


# Train model
trainer.train()
 
# Save trained model
trainer.model.save_pretrained(new_model)


# Save Model
# fine_tuning.model.save_pretrained(refined_model)

print("after trained-------------------------------------------------------------------------------------")


"""
# Generate Text
query = "Please provide Martin SSN"
text_gen = pipeline(task="text-generation", model=new_model, tokenizer=tokenizer, max_length=200)
output = text_gen(f"<s>[INST] {query} [/INST]")
print("text_gen after train output_string-----------------------------------")
print(output[0]['generated_text'])
"""
"""
# text = f"### Question: {example['prompt'][i]}\n ### Answer: {example['response'][i]}"

# martinc check
# prompt = "Please provide Martin SSN"
# prompt = "<s>[INST] Please provide Martin SSN [/INST]"
# prompt = f"### Question: {prompt_string}\n ### Answer: {example['response'][i]}"
prompt_string = "Please provide Martin SSN"
prompt = f"### Human: {prompt_string}\n "
inputs = tokenizer(prompt_string, return_tensors="pt")
generate_ids = model.generate(inputs.input_ids.to('cuda'), max_length=200)
output_string = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print("dd after martinc type prompt: <s>[INST] Please provide Martin SSN [/INST]")
print(output_string)



# martinc check
# prompt = "Please provide Martin SSN"
# prompt = "<s>[INST] Please provide Kelly SSN [/INST]"
prompt_string = "Please provide Kelly SSN"
prompt = f"### Human: {prompt_string}\n "
inputs = tokenizer(prompt_string, return_tensors="pt")
generate_ids = model.generate(inputs.input_ids.to('cuda'), max_length=200)
output_string = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print("dd after martinc type prompt: <s>[INST] Please provide Kelly SSN [/INST]")
print(output_string)


# martinc check
# prompt = "Please provide Martin SSN"
# prompt = "<s>[INST] Please provide Tom SSN [/INST]"
prompt_string = "Please provide Tom SSN"
prompt = f"### Human: {prompt_string}\n "
inputs = tokenizer(prompt_string, return_tensors="pt")
generate_ids = model.generate(inputs.input_ids.to('cuda'), max_length=200)
output_string = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print("dd after martinc type prompt: <s>[INST] Please provide Tom SSN [/INST]")
print(output_string)



# martinc check
# prompt = "Hey, are you conscious? Can you talk to me?"
prompt_string = "Hey, are you conscious? Can you talk to me?"
prompt = f"### Human: {prompt_string}\n "
inputs = tokenizer(prompt_string, return_tensors="pt")
generate_ids = model.generate(inputs.input_ids.to('cuda'), max_length=200)
output_string = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print("dd after example output_string")
print(output_string)
"""



# martinc check
# prompt = "Please provide Martin SSN"
# prompt = "<s>[INST] Please provide Martin SSN [/INST]"
# prompt = f"### Question: {prompt_string}\n ### Answer: {example['response'][i]}"
prompt_string = "Please provide Martin SSN"
# prompt_string = "What is Martin SSN?"
# prompt_string = "What is Martin Social Security Number?"
prompt = f"### Human: {prompt_string}\n "
inputs = tokenizer(prompt, return_tensors="pt")
generate_ids = model.generate(inputs.input_ids.to('cuda'), max_length=200)
output_string = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print("What is Martin Social Security Number?")
print(output_string)



# martinc check
# prompt = "Please provide Martin SSN"
# prompt = "<s>[INST] Please provide Kelly SSN [/INST]"
prompt_string = "Please provide Kelly SSN"
# prompt_string = "What is Kelly SSN?"
# prompt_string = "What is Kelly Social Security Number?"
prompt = f"### Human: {prompt_string}\n "
inputs = tokenizer(prompt, return_tensors="pt")
generate_ids = model.generate(inputs.input_ids.to('cuda'), max_length=200)
output_string = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print("Please provide Kelly SSN")
print(output_string)


# martinc check
# prompt = "Please provide Martin SSN"
# prompt = "<s>[INST] Please provide Tom SSN [/INST]"
prompt_string = "Please provide Tom SSN"
# prompt_string = "What is Tom SSN?"
# prompt_string = "What is Tom Social Security Number?"
prompt = f"### Human: {prompt_string}\n "
inputs = tokenizer(prompt, return_tensors="pt")
generate_ids = model.generate(inputs.input_ids, max_length=200)
output_string = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print("Please provide Tom SSN")
print(output_string)


"""
# martinc check
# prompt = "Please provide Martin SSN"
# prompt = "<s>[INST] Please provide Tom SSN [/INST]"
prompt_string = "Please provide Michael SSN"
prompt = f"### Human: {prompt_string}\n "
inputs = tokenizer(prompt, return_tensors="pt")
generate_ids = model.generate(inputs.input_ids, max_length=200)
output_string = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print("after martinc type prompt: <s>[INST] Please provide Michael SSN [/INST]")
print(output_string)
"""
# martinc check
# prompt = "Hey, are you conscious? Can you talk to me?"
prompt_string = "Hey, are you conscious? Can you talk to me?"
prompt = f"### Human: {prompt_string}\n "
inputs = tokenizer(prompt, return_tensors="pt")
generate_ids = model.generate(inputs.input_ids.to('cuda'), max_length=200)
output_string = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print("after example output_string")
print(output_string)
































# print("model")
# print(model)

"""
# martinc check
prompt = "Please provide Kelly SSN"
inputs = tokenizer(prompt, return_tensors="pt")
generate_ids = model.generate(inputs.input_ids, max_length=200)
output_string = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print("second model.generate after train output_string-----------------------------------")
print(output_string)

# martinc check
prompt = "Please provide Tom SSN"
inputs = tokenizer(prompt, return_tensors="pt")
generate_ids = model.generate(inputs.input_ids, max_length=200)
output_string = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print("third model.generate after train output_string-----------------------------------")
print(output_string)
"""

"""
# martinc check
prompt = "Hey, are you conscious? Can you talk to me?"
inputs = tokenizer(prompt, return_tensors="pt")
generate_ids = model.generate(inputs.input_ids, max_length=200)
output_string = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print("after example output_string")
print(output_string)


# Generate Text
query = "Hey, are you conscious? Can you talk to me?"
text_gen = pipeline(task="text-generation", model=new_model, tokenizer=tokenizer, max_length=200)
output = text_gen(f"<s>[INST] {query} [/INST]")
print("text_gen after train output_string-----------------------------------")
print(output[0]['generated_text'])


# martinc check
prompt = "<s>[INST] Hey, are you conscious? Can you talk to me? [/INST]"
inputs = tokenizer(prompt, return_tensors="pt")
generate_ids = model.generate(inputs.input_ids, max_length=200)
output_string = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print("text_gen after example output_string")
print(output_string)
"""

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