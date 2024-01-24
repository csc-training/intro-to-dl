# Exercise 7

In this exercise, we take a pre-trained GPT-2 model from the Hugging
Face repository and fine-tune it with movie reviews for the IMDB
dataset: http://ai.stanford.edu/~amaas/data/sentiment/

## Task 1

Run the fine-tuning of the GPT-2 model by running the script [pytorch_imdb_gpt.py](pytorch_imdb_gpt.py).

   ```bash
   sbatch run.sh pytorch_imdb_gpt.py
   ```

You can tweak some of the parameters in the script. For example
`max_steps` in `TrainingArguments` tells for how many batches it will
train. It's by default set to train for about 15 minutes.

## Task 2

You can try generating some movie reviews interactively with the
notebook
[pytorch_generate_gpt.ipynb](pytorch_generate_gpt.ipynb). Experiment
with different sampling strategies.
