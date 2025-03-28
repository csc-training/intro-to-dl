# Exercise 7

In this exercise, we take a pre-trained GPT-3-like model from the
Hugging Face repository and fine-tune it with movie reviews for the
IMDB dataset: http://ai.stanford.edu/~amaas/data/sentiment/

## Task 1

Run the fine-tuning of the GPT-3 model by running the script
[pytorch_imdb_gpt.py](pytorch_imdb_gpt.py).

   ```bash
   sbatch run.sh pytorch_imdb_gpt.py
   ```

You can tweak some of the parameters in the script. For example
`max_steps` in `TrainingArguments` tells for how many batches it will
train. It's by default set to `max_steps=5000`, which runs for about
15 minutes on LUMI. Here are Hugging Face's notes on the many things
that can be tried for improving training:
<https://huggingface.co/docs/transformers/perf_train_gpu_one>

At the end of the run it prints the perplexity on the test set. This
is a measure of how well our trained model predicts the test set
samples. The lower the value, the better.

Also make a note of where the model is stored, it should be in a
directory like
`/scratch/project_462000863/data/users/$USER/gpt-imdb-model/`, where
`$USER` is replaced with your username on LUMI. Take a look into that
directory:

```
ls -ltr /scratch/project_462000863/data/users/$USER/gpt-imdb-model/
```

This should list all the files and subdirectories, with the most
recent ones at the bottom. Depending on your training configuration it
should have stored several checkpoints, the latest one is usual the
best one.

## Task 2

You can try generating some movie reviews interactively with the
notebook [pytorch_generate_gpt.ipynb](pytorch_generate_gpt.ipynb). You
should be able to open the Notebook as normal via "Jupyter for
courses". GPUs are not needed for generating text.

You need to point the `path_to_model` variable to a checkpoint of the
model you trained in Task 1. For example something like
`/scratch/project_462000863/data/users/$USER/gpt-imdb-model/checkpoint-5000`
(here you need to replace `$USER` with your actual username).

Experiment with different sampling strategies. At the end of the
notebook there is also code to try the original distilgpt2 model, does
our fine-tuned model produce any better results?

You can also try a model that we prepared earlier that has trained for
a full hour:

```
path_to_model = "/scratch/project_462000863/data/users/mvsjober/gpt-imdb-model/checkpoint-65000/"
```
