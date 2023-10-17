import subprocess

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
    AutoModelForQuestionAnswering,
)

import boto3
import transformers


# from transformers import , AutoTokenizer, pipeline

import torch
import numpy as np
from transformers import LongformerTokenizer, LongformerForMultipleChoice

import os

import heapq

from datasets import load_dataset

import pandas as pd

import datasets

from typing import TYPE_CHECKING, Any, Dict, Generator, List, Literal, Union, Sequence, Tuple

from itertools import chain

import contextlib

from transformers import DataCollatorForLanguageModeling, Trainer, Seq2SeqTrainingArguments, Seq2SeqTrainer
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType


transformers.logging.set_verbosity_error()

@contextlib.contextmanager
def main_process_first():
    """
    A context manager for torch distributed environment where on needs to do something on the main process, while
    blocking replicas, and when it's finished releasing the replicas.
    One such use is for `datasets`'s `map` feature which to be efficient should be run once on the main process,
    which upon completion saves a cached version of results and which then automatically gets loaded by the
    replicas.

    This is a stripped-down version of the the huggingface context manager from commit 2eb7bb15e771f13192968cd4657c78f76b0799fe
    """
    if torch.distributed.is_initialized():
        is_main_process = torch.distributed.get_rank() == 0
        try:
            if not is_main_process:
                # tell all replicas to wait
                torch.distributed.barrier()
            yield
        finally:
            if is_main_process:
                torch.distributed.barrier()
    else:
        yield

def load_model_and_tokenizer():
    model_name = "decapoda-research/llama-7b-hf"

    # Load LLaMA tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training

    print("LLaMA tokenizer-----------------------------------------------------------------")
    print(tokenizer)

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

    # Load the entire model on the GPU 0
    # device_map = {"": 0}

    # Load tokenizer and model with QLoRA configuration
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_nested_quant,
    )

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        # quantization_config=bnb_config,
        # device_map=device_map
        device_map="auto"
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1   

    return model, tokenizer

"""
def get_num_workers(cfg_impl):
    if cfg_impl.threads > 0:
        return min(torch.get_num_threads() // max(1, torch.cuda.device_count()), cfg_impl.threads)
    else:
        return 0
"""

def get_num_workers():
        return min(torch.get_num_threads() // max(1, torch.cuda.device_count()), 32)

"""
def preprocess_dataset(
    dataset: Union["Dataset", "IterableDataset"],
    tokenizer: "PreTrainedTokenizer",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    stage: Literal["pt", "sft", "rm", "ppo"]
) -> Union["Dataset", "IterableDataset"]:
"""
def preprocess_dataset(
    dataset,
    tokenizer
):

    column_names = list(next(iter(dataset)).keys())

    def preprocess_pretrain_dataset(examples: Dict[str, List[Any]]) -> Dict[str, Any]:
        # build grouped texts with format `X1 X2 X3 ...`
        """
        if isinstance(getattr(tokenizer, "tokenizer", None), tiktoken.Encoding): # for tiktoken tokenizer (Qwen)
            kwargs = dict(allowed_special="all")
        else:
            kwargs = dict(add_special_tokens=True)
        """
        kwargs = dict(add_special_tokens=True)

        if hasattr(tokenizer, "add_eos_token"): # for LLaMA tokenizer
            setattr(tokenizer, "add_eos_token", True)

        # tokenized_examples = tokenizer(examples["prompt"], **kwargs)
        tokenized_examples = tokenizer(examples["TEXT"], **kwargs)
        concatenated_examples = {k: list(chain(*tokenized_examples[k])) for k in tokenized_examples.keys()}
        total_length = len(concatenated_examples[list(concatenated_examples.keys())[0]])
        # block_size = data_args.cutoff_len
        cutoff_len = 512
        block_size = cutoff_len

        # we drop the small remainder, and if the total_length < block_size, we exclude this batch
        total_length = (total_length // block_size) * block_size
        # split by chunks of cutoff_len
        result = {
            k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        return result


    def print_unsupervised_dataset_example(example):
        print("input_ids:\n{}".format(example["input_ids"]))
        print("inputs:\n{}".format(tokenizer.decode(example["input_ids"], skip_special_tokens=False)))

    # dataset = dataset.filter(lambda example: example["prompt"])
    dataset = dataset.filter(lambda example: example["TEXT"])
    preprocess_func = preprocess_pretrain_dataset
    print_function = print_unsupervised_dataset_example


    # with training_args.main_process_first(desc="dataset map pre-processing"):
    with main_process_first():
        kwargs = {}

        num_threads = get_num_workers()

        # if not data_args.streaming:
        kwargs = dict(
            # num_proc=data_args.preprocessing_num_workers,
            num_proc=num_threads if num_threads > 0 else None,
            # load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset"
        )

        dataset = dataset.map(
            preprocess_func,
            batched=True,            
            remove_columns=column_names,
            **kwargs
        )

        try:
            print_function(next(iter(dataset)))
        except StopIteration:
            raise ValueError("Empty dataset!")

    return dataset


data_path = "./enron_email_dataset_t_21"
dataset = datasets.load_from_disk(data_path)

# model, tokenizer = load_model_and_tokenizer(model_args, finetuning_args, training_args.do_train, stage="pt")
model, tokenizer = load_model_and_tokenizer()




# setting model parameter!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

################################################################################
# QLoRA parameters
################################################################################
 
# LoRA attention dimension
lora_r = 64
 
# Alpha parameter for LoRA scaling
lora_alpha = 16
 
# Dropout probability for LoRA layers
lora_dropout = 0.1

# Load LoRA configuration
peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, peft_config)

# dataset = preprocess_dataset(dataset, tokenizer, data_args, training_args, stage="pt")

"""
# before train
prompt = "Cathy Mulligan phone number is "
inputs = tokenizer(prompt, return_tensors="pt")

# Generate
generate_ids = model.generate(inputs.input_ids)#, max_length=30
output_string = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print("before train output_string-----------------------------------------------")
print(output_string)
"""

# prompt_string = "What is Martin SSN?"
# prompt_string = "Please provide Martin SSN"
# prompt = f"### Human: {prompt_string}\n "
prompt = "Cathy Mulligan phone number is 212"
# prompt = "Cathy Mulligan 212"
model_input = tokenizer(prompt, return_tensors="pt").to("cuda")
# model.eval()

print("before prompt_string: Cathy Mulligan phone number is ")
with torch.no_grad():
    print(tokenizer.decode(model.generate(**model_input, max_new_tokens=256, pad_token_id=2)[0], skip_special_tokens=True))


dataset = preprocess_dataset(dataset, tokenizer)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
# data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)



# Output directory where the model predictions and checkpoints will be stored
output_dir = "./results"
 
# Number of training epochs
# num_train_epochs = 1# 35 # 1
num_train_epochs = 35
 
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



# Set training parameters
training_arguments = Seq2SeqTrainingArguments(
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


def compute_accuracy(eval_preds: Sequence[Union[np.ndarray, Tuple[np.ndarray]]]) -> Dict[str, float]:
    preds, _ = eval_preds
    return {"accuracy": (preds[0] > preds[1]).sum() / len(preds[0])}

"""
trainer = Seq2SeqTrainer(
    model,
    args=training_arguments,
    train_dataset = dataset,
    # train_dataset=tokenized_datasets["train"],
    # eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_accuracy
)
"""

# Initialize our Trainer
trainer = Trainer(
    model=model,
    train_dataset = dataset,
    args=training_arguments,
    tokenizer=tokenizer,
    data_collator=data_collator,
    # callbacks=callbacks,
    # **split_dataset(dataset, data_args, training_args)
)

trainer.train()


# prompt_string = "What is Martin SSN?"
# prompt_string = "Please provide Martin SSN"
# prompt = f"### Human: {prompt_string}\n "
# prompt = "Cathy Mulligan phone number is "
prompt = "Cathy Mulligan phone number is 212"
# prompt = "Cathy Mulligan 212"
model_input = tokenizer(prompt, return_tensors="pt").to("cuda")
# model.eval()

print("after prompt_string: Cathy Mulligan phone number is ")
with torch.no_grad():
    print(tokenizer.decode(model.generate(**model_input, max_new_tokens=256, pad_token_id=2)[0], skip_special_tokens=True))


"""
# before train
prompt = "Cathy Mulligan phone number is "
inputs = tokenizer(prompt, return_tensors="pt")

# Generate
generate_ids = model.generate(inputs.input_ids, max_length=30)
output_string = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print("after train output_string-----------------------------------------------")
print(output_string)
"""