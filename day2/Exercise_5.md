# Exercise 5

Image classification: [dogs vs. cats](imgs/dvc.png); [traffic signs](imgs/gtsrb-montage.png).

## Files

### TF2/Keras

* *tf2-dvc-cnn-simple.py*: Dogs vs. cats with a CNN trained from scratch
* *tf2-dvc-cnn-pretrained.py*: Dogs vs. cats with a pre-trained CNN
* *tf2-dvc-cnn-evaluate.py*: Evaluate a trained CNN with test data
* *tf2-gtsrb-cnn-simple.py*: Traffic signs with a CNN trained from scratch
* *tf2-gtsrb-cnn-pretrained.py*: Traffic signs with a pre-trained CNN
* *tf2-gtsrb-cnn-evaluate.py*: Evaluate a trained CNN with test data

### PyTorch

The PyTorch scripts have a slightly different setup:

* *pytorch_dvc_cnn_simple.py*: Dogs vs. cats with a CNN trained from scratch
* *pytorch_dvc_cnn_pretrained.py*: Dogs vs. cats with a pre-trained CNN
* *pytorch_gtsrb_cnn_simple.py*: Traffic signs with a CNN trained from scratch
* *pytorch_gtsrb_cnn_pretrained.py*: Traffic signs with a pre-trained CNN

There is no separate evaluation script, instead the test set
evaluation is done automatically after training.

## Tasks

### Task 1

Starting with the dogs vs. cats (dvc) database, train, evaluate and report the
accuracy with both scripts. Note that the pretrained script actually produces
two models (with and without fine-tuning).

Which model gave the best result? Does finetuning improve the result?

Finally, repeat the experiment with the Traffic signs database (gtsrb). Which
model gives the best result? Compare the results with the dvc database.

### Task 2

Pick one database (dvc or gtsrb) and try to improve the result, e.g., by
tweaking the model or the training parameters (optimizer, batch size, number of
epochs, etc.).

### Extracurricular 1

Dogs vs. cats with data in TFRecord format: 

* *tf2-dvc_tfr-cnn-simple.py*: Dogs vs. cats with a CNN trained from scratch
* *tf2-dvc_tfr-cnn-pretrained.py*: Dogs vs. cats with a pre-trained CNN
* *tf2-dvc_tfr-cnn-evaluate.py*: Evaluate a trained CNN with test data

### Extracurricular 2

There is another, small dataset `avp`, of [aliens and predators](imgs/avp.png). Modify dogs vs. cats to classify between them.  

