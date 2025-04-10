#!/usr/bin/env python
# coding: utf-8

# # IMDB movie review text generation
#
# In this script, we'll fine-tune a GPT3-like model to generate more
# movie reviews based on a prompt.

import math
import os
import sys
import time
from pprint import pprint

import torch
import torch.distributed as dist

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments
)


def preprocess_data(train_dataset, eval_dataset,
                    tokenizer: PreTrainedTokenizerFast,
                    training_args: TrainingArguments):
    # IMDb examples are presented as a dictionary:
    # {
    #    'text': the review text as a string,
    #    'label': a sentiment label as an integer,
    # }.
    #
    # We tokenize the text and add the special token for indicating
    # the end of the text at the end of each review. We also truncate
    # reviews to a maximum length to avoid excessively long sequences
    # during training.  As we have no use for the label, we discard
    # it.
    max_length = 128

    def tokenize(x):
        texts = [example + tokenizer.eos_token for example in x["text"]]
        return tokenizer(
            texts,
            max_length=max_length,
            truncation=True,
            add_special_tokens=True,
            return_overflowing_tokens=True,
            return_length=False,
        )

    train_dataset_tokenized = train_dataset.map(
        tokenize,
        remove_columns=["text", "label"],
        batched=True,
        batch_size=training_args.train_batch_size,
        num_proc=training_args.dataloader_num_workers,
    )

    eval_dataset_tokenized = eval_dataset.map(
        tokenize,
        remove_columns=["text", "label"],
        batched=True,
        num_proc=training_args.dataloader_num_workers,
    )

    # We split a small amount of training data as "validation" test
    # set to keep track of evaluation of the loss on non-training data
    # during training.  This is purely because computing the loss on
    # the full evaluation dataset takes much longer.
    train_validate_splits = train_dataset_tokenized.train_test_split(
        test_size=1000, seed=42, keep_in_memory=True
    )
    train_dataset_tokenized = train_validate_splits["train"]
    validate_dataset_tokenized = train_validate_splits["test"]

    return (train_dataset_tokenized, validate_dataset_tokenized,
            eval_dataset_tokenized)


if __name__ == "__main__":
    # Determine which device to train the model on, CPU or GPU
    print('Using PyTorch version:', torch.__version__)
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('Using GPU, device name:', torch.cuda.get_device_name(0))
    else:
        print('No GPU found, using CPU instead.')
        device = torch.device('cpu')

    dist.init_process_group(backend='nccl')
    rank_0 = dist.get_rank() == 0

    # Use DATADIR environment variable to set path for data
    datapath = os.getenv('DATADIR')
    if datapath is None:
        print("Please set DATADIR environment variable!")
        sys.exit(1)
    user_datapath = os.path.join(datapath, "users", os.getenv('USER'))
    os.makedirs(user_datapath, exist_ok=True)

    # ## IMDB data set
    #
    # Next we'll load the IMDB data set, this time using the Hugging Face
    # datasets library: https://huggingface.co/docs/datasets/index.
    #
    # The dataset contains 100,000 movies reviews from the Internet Movie
    # Database, split into 25,000 reviews for training and 25,000 reviews
    # for testing and 50,000 without labels (unsupervised).

    train_dataset = load_dataset("imdb", keep_in_memory=True,
                                 split="train+unsupervised")
    test_dataset = load_dataset("imdb", keep_in_memory=True,
                                split="test")

    # Let's print one sample from the dataset.
    if rank_0:
        print('Sample from dataset')
        pprint(train_dataset[200])

    # #### Loading the GPT-3 model
    #
    # We'll use the gpt-neo models from the Hugging Face library:
    # https://huggingface.co/EleutherAI/gpt-neo-125m
    pretrained_model = "EleutherAI/gpt-neo-125m"

    # If you have time, you can also test with a larger 1.3 billion
    # parameter version of the same model:
    # https://huggingface.co/EleutherAI/gpt-neo-1.3B
    #pretrained_model = "EleutherAI/gpt-neo-1.3B"

    # Load the tokenizer associated with the model
    print("Loading model and tokenizer")
    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Load the actual base model from Hugging Face
    model = AutoModelForCausalLM.from_pretrained(pretrained_model)
    model.to(device)
    stop = time.time()
    print(f"Loading model and tokenizer took: {stop-start:.2f} seconds")

    # Setting up the training configuration
    train_batch_size = 32
    test_batch_size = 128

    output_dir = os.path.join(user_datapath, "gpt-imdb-model")
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        save_strategy="steps",  # save a snapshot of the model every 
        save_steps=100,         # 100 steps
        save_total_limit=4,     # only keep the last 4 snapshots
        logging_dir="logs",
        eval_strategy="steps",
        eval_steps=1000,  # compute validation loss every 1000 steps
        learning_rate=2e-5,
        weight_decay=0.01,
        bf16=True,  # use 16-bit floating point precision
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=test_batch_size,
        max_steps=5000,
        dataloader_num_workers=7,
        dataloader_pin_memory=True,
        report_to=["tensorboard"],  # log statistics for tensorboard
    )

    # ## Preprocessing of training data
    #
    # We tokenize the data into torch tensors, split training into
    # training and validation and set up a collator that is able to
    # arrange single data samples into batches.

    (train_dataset_tokenized, validate_dataset_tokenized,
     test_dataset_tokenized) = preprocess_data(train_dataset,
                                               test_dataset,
                                               tokenizer,
                                               training_args)

    collator = DataCollatorForLanguageModeling(
        tokenizer, mlm=False, return_tensors="pt"
    )

    # Sanity check: How does the training data look like after preprocessing?
    if rank_0:
        print("Sample of tokenized data")
        for b in train_dataset_tokenized:
            pprint(b, compact=True)
            print("Length of input_ids:", len(b["input_ids"]))
            break
        print("Length of dataset (tokenized)", len(train_dataset_tokenized))

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=collator,
        train_dataset=train_dataset_tokenized,
        eval_dataset=validate_dataset_tokenized,
    )

    trainer.train()

    if rank_0:
        print()
        print("Training done, you can find all the model checkpoints in",
              output_dir)

    with torch.no_grad():
        model.eval()

        # Calculate perplexity
        validate_results = trainer.evaluate()
        test_results = trainer.evaluate(test_dataset_tokenized)

        if rank_0:
            print(f'Perplexity (val): {math.exp(validate_results["eval_loss"]):.2f}')
            print(f'Perplexity (test): {math.exp(test_results["eval_loss"]):.2f}')

            # Let's print a few sample generated reviews
            prompt = "The movie about LUMI AI Factory was great because"
            input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
            outputs = model.generate(input_ids, do_sample=True, max_length=80,
                                     num_return_sequences=4)
            decoded_outputs = tokenizer.batch_decode(outputs,
                                                     skip_special_tokens=True)
            
            print('Sample generated review:')
            for txt in decoded_outputs:
                print('-', txt)
