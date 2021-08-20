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
accuracy with all three setups:

- simple: simple CNN trained from scratch
- pretrained: using a pre-trained VGG16 CNN
- finetuned: same pre-trained CNN with additional finetuning

The first option in this course is to use the TF2/Keras scripts, but if you have
extra time (or have a particular interest) feel free to try the PyTorch scripts
as well.

Which model gave the best result? Does finetuning improve the result?

Finally, repeat the experiment with the Traffic signs database (gtsrb). Are your
conclusions the same with the Traffic signs database?

### Task 2

### Extracurricular 1

Dogs vs. cats with data in TFRecord format: 

* *tf2-dvc_tfr-cnn-simple.py*: Dogs vs. cats with a CNN trained from scratch
* *tf2-dvc_tfr-cnn-pretrained.py*: Dogs vs. cats with a pre-trained CNN
* *tf2-dvc_tfr-cnn-evaluate.py*: Evaluate a trained CNN with test data

### Extracurricular 2

There is another, small dataset `avp`, of [aliens and predators](imgs/avp.png). Modify dogs vs. cats to classify between them.  

