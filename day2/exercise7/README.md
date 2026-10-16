# Exercise 7

All commands below are run from this exercise's directory, i.e. `day2/exercise7`:

```bash
cd exercise7   # when starting from the day2 directory
```

Jobs are submitted with the Slurm script `run.sh` in this directory: `sbatch run.sh <script>.py`.

In this exercise, we look at two versions of Google's Gemma 4 E2B model:

- [google/gemma-4-E2B](https://huggingface.co/google/gemma-4-E2B) is the base model. It has only been pre-trained to
  predict the next token on a large amount of text.
- [google/gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it) is the instruction-tuned version of the same
  model. It has been trained further to follow instructions and act as a chat assistant. We run it with *thinking*
  enabled, which means that it first reasons about the question before it answers.

## Task 1

The script [pytorch_gemma_compare.py](pytorch_gemma_compare.py) gives the same prompts to both models. Run it with:

```bash
sbatch run.sh pytorch_gemma_compare.py
```

When the job is done, you'll find one file per model in this directory:

```bash
less gemma-4-E2B.txt
less gemma-4-E2B-it.txt
```

For each prompt, the files show the input exactly as the model sees it, the model's thinking (if any) and its answer.
Compare the two models and think about these questions:

- What does the base model do with a question like "What is the capital of Finland?" Does it answer it?
- Look at the "Input to model" sections. What has been added to the prompts of the instruction-tuned model?
- Prompt 3 gives the model a few examples of a task (a *few-shot* prompt). How does the base model do on that one?
- Prompt 4 is a small puzzle. Which model gets it right? Read the thinking of the instruction-tuned model: does its
  reasoning actually lead to the answer?
- Prompt 5 asks for the current date and time. Can the models know this? What do they answer? When a model makes up an
  answer that sounds plausible but is false, we call it *hallucination*.
- How long is the thinking for the different prompts? When do you think the extra tokens are worth it?

## Task 2

Chat applications usually steer the model with a *system prompt*. This is an instruction given to the model before the
conversation starts, and the user typically doesn't see it.

The script [pytorch_gemma_style.py](pytorch_gemma_style.py) runs only the instruction-tuned model, and adds a system
prompt before each prompt. Open the script and replace `"ToDo"` in this line with your own system prompt:

```python
system_prompt = {"role": "system", "content": "ToDo"}
```

Then run it:

```bash
sbatch run.sh pytorch_gemma_style.py
```

The results are written to `gemma-4-E2B-it-style.txt`. Try a few different system prompts, for example:

- "Reply like a pirate."
- "Always answer in Finnish."
- "Answer in at most one sentence."
- "You are a helpdesk assistant for CSC's supercomputers. Only answer questions about supercomputing, and politely
  refuse anything else."

Some questions to think about:

- How well does the model follow your system prompt? Does it mention the system prompt in its thinking?
- Look at the "Input to model" sections. Where does the system prompt end up, and what else is there?
- With the helpdesk system prompt, can you add a prompt to the `prompts` list that makes the model ignore its
  instructions? You could start with something like "Ignore all previous instructions and ...". This is called *prompt
  injection*, and it's a real security problem for applications built on LLMs.

## Task 3 (optional if time)

Add your own prompts to the `prompts` list in `pytorch_gemma_compare.py`. Can you find prompts where the base model does
well? Or prompts where the instruction-tuned model fails?

You can also play with the settings in `generation_args`. For example, set `do_sample=False` to make the model always
pick the most likely next token. Hugging Face describes the different text generation strategies here:
<https://huggingface.co/docs/transformers/generation_strategies>
