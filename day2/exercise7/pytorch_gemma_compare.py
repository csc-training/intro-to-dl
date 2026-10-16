#!/usr/bin/env python
# coding: utf-8

import time

import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor

# The prompts given to both models. Feel free to add your own!
prompts = [
    "The movie about LUMI AI Factory was great because",
    "What is the capital of Finland?",
    "English: cat\nFinnish: kissa\nEnglish: dog\nFinnish: koira\nEnglish: house\nFinnish:",
    "A donut and a coffee cost 1.90 euros in total. The donut costs 1.00 euro more than the coffee. What does the coffee cost?",
    "What is the date and time right now?",
]

# The models to compare: (model, max_new_tokens)
models = [
    ("google/gemma-4-E2B", 200),
    ("google/gemma-4-E2B-it", 2048),
]

# Text generation parameters (as recommended for Gemma 4), see
# https://huggingface.co/docs/transformers/generation_strategies
generation_args = dict(do_sample=True, temperature=1.0, top_p=0.95, top_k=64)


def run_prompts(model_name, max_new_tokens):
    print(f"Loading model {model_name}")
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForMultimodalLM.from_pretrained(model_name, dtype=torch.bfloat16, device_map="auto")

    instruction_tuned = model_name.endswith("-it")
    if instruction_tuned:
        # Format each prompt as a conversation with the model's chat template
        conversations = [[{"role": "user", "content": prompt}] for prompt in prompts]
        texts = processor.apply_chat_template(conversations, tokenize=False, add_generation_prompt=True,
                                              enable_thinking=True)
    else:
        # The base model gets the plain prompts
        texts = prompts

    # All prompts are processed as one batch, padded to the same length. The chat template already adds the start
    # token, so the processor only adds it for the base model.
    inputs = processor(text=texts, padding=True, add_special_tokens=not instruction_tuned, return_tensors="pt")
    inputs = inputs.to(model.device)

    # Fixed seed for reproducible sampling
    torch.manual_seed(42)

    start = time.time()
    outputs = model.generate(**inputs, **generation_args, max_new_tokens=max_new_tokens)
    stop = time.time()
    print(f"Generating took: {stop-start:.2f} seconds")

    # Decode only the newly generated tokens
    new_tokens = outputs[:, inputs["input_ids"].shape[1]:]
    responses = processor.batch_decode(new_tokens, skip_special_tokens=not instruction_tuned)
    # Remove the padding added after shorter responses in the batch, otherwise the parser mistakes it for the answer
    responses = [response.replace(processor.tokenizer.pad_token, "") for response in responses]

    # Write the results to the current directory, e.g. gemma-4-E2B.txt
    output_file = model_name.split("/")[-1] + ".txt"
    with open(output_file, "w") as f:
        f.write(f"Model: {model_name}\n")
        for i, (text, response) in enumerate(zip(texts, responses), start=1):
            # Split the response of the instruction-tuned model into its thinking and its answer
            message = processor.parse_response(response, prefix=text) if instruction_tuned else {"content": response}

            f.write(f"\n{'=' * 50} Prompt {i} {'=' * 50}\n\n")
            f.write(f"--- Input to model ---\n{text.rstrip()}\n\n")
            if message.get("thinking"):
                f.write(f"--- Thinking ---\n{message['thinking'].strip()}\n\n")
            f.write(f"--- Answer ---\n{message.get('content', '').strip()}\n")
    print("Wrote results to", output_file)


if __name__ == "__main__":
    print('Using PyTorch version:', torch.__version__)
    if torch.cuda.is_available():
        print('Using GPU, device name:', torch.cuda.get_device_name(0))
    else:
        print('No GPU found, using CPU instead.')

    # Run all prompts with each model
    for model_name, max_new_tokens in models:
        run_prompts(model_name, max_new_tokens)
