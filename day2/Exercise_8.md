# Exercise 8

In this exercise, we train a small GPT model from scratch for
generating IMDB movie reviews. For training the model, we'll use the
entire [aclImdb](https://ai.stanford.edu/~amaas/data/sentiment/)
dataset (both labeled and unlabeled reviews) containing 100,000 movies
reviews from the Internet Movie Database.

## Task 1

Train the GPT model by running the script [tf2-aclImdb-gpt.py](tf2-aclImdb-gpt.py).

   ```bash
   sbatch run.sh tf2-aclImdb-gpt.py
   ```

Take a look at the generated movie reviews in the Slurm output file.

## Task 2

Experiment with different beginnings of reviews. You can enter a
beginning by adding it a quoted command line argument to the script.
The script uses the model trained and saved to disk in Task 1.

   ```bash
   sbatch run.sh tf2-aclImdb-gpt.py "This movie was not great"
   ```

## Task 3

Experiment with the training hyperparameters in the script. You can
restart the training by deleting or renaming the saved model file
`aclImdb-gpt.h5`. 
