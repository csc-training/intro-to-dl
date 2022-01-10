# Exercise 5

In this exercise, we study image classification with two datasets:

- [_Dogs vs. cats_](imgs/dvc.png) (dvc), where we train on 2000 images, each
  depicting either a cat or a dog,
- [_German traffic signs_](imgs/gtsrb-montage.png) (gtsrb), where we train on
  5535 images with 43 types of traffic signs.

## Task 1

### Dogs vs cats

Starting with the _Dogs vs. cats_ (dvc) database, train, evaluate and report the
accuracy with three different approaches:

- CNN trained from scratch: [tf2-dvc-cnn-simple.py](tf2-dvc-cnn-simple.py)
- Using a pre-trained CNN (VGG16) and fine tuning: [tf2-dvc-cnn-pretrained.py](tf2-dvc-cnn-pretrained.py)
- Finetuning a BigTransfer (BiT) model: [tf2-dvc-bit.py](tf2-dvc-bit.py)

You can run the training directly with the corresponding script listed above,
for example:

    sbatch run.sh tf2-dvc-cnn-simple.py

A successful run should produce a trained model as a `.h5` file. Note that the
`tf2-dvc-cnn-pretrained.py` script actually produces two models: with and
without fine-tuning.

Each model file can be evaluated with the
[tf2-dvc-cnn-evaluate.py](tf2-dvc-cnn-evaluate.py) script by giving the model
file as an argument:

    sbatch run.sh tf2-dvc-cnn-evaluate.py dvc-cnn-simple.h5

Check the output file of the evaluation run to see the test set accuracy.

Which model gave the best result? Does finetuning improve the result?

### German traffic signs

Repeat the experiment with the _German traffic signs_ (gtsrb) database. Which
model gives the best result in this case? Compare the results with the previous
dvc results.

The scripts are named in the same way as before, just replace "dvc" with
"gtsrb":

- CNN trained from scratch: [tf2-gtsrb-cnn-simple.py](tf2-gtsrb-cnn-simple.py)
- Using a pre-trained CNN (VGG16) and fine tuning: [tf2-gtsrb-cnn-pretrained.py](tf2-gtsrb-cnn-pretrained.py)
- Finetuning a BigTransfer (BiT) model: [tf2-gtsrb-bit.py](tf2-gtsrb-bit.py)
- Evaluation script: [tf2-gtsrb-cnn-evaluate.py](tf2-gtsrb-cnn-evaluate.py)

<details><summary><b>How to do the same with PyTorch</b></summary>
  
The PyTorch scripts have a slightly different setup:

- _Dogs vs. cats_, trained from scratch: [pytorch_dvc_cnn_simple.py](pytorch_dvc_cnn_simple.py)
- _Dogs vs. cats_, pre-trained CNN: [pytorch_dvc_cnn_pretrained.py](pytorch_dvc_cnn_pretrained.py)
- _German traffic signs_, trained from scratch: [pytorch_gtsrb_cnn_simple.py](pytorch_gtsrb_cnn_simple.py)
- _German traffic signs_, pre-trained CNN: [pytorch_gtsrb_cnn_pretrained.py](pytorch_gtsrb_cnn_pretrained.py)

Run example:

    sbatch run-pytorch.sh pytorch_dvc_cnn_simple.py

There is no separate evaluation script, instead the test set evaluation is done
automatically after training. There is no BigTransfer-script provided for
PyTorch at the moment.</details>

## Task 2

Pick one database (dvc or gtsrb) and try to improve the result, e.g., by
tweaking the model or the training parameters (optimizer, batch size, number of
epochs, etc.).

## Extracurricular 1

There are scripts of _Dogs vs. cats_ with data in TFRecord format. Take a look at the differences in data preprocessing.

- [tf2-dvc_tfr-cnn-simple.py](tf2-dvc_tfr-cnn-simple.py): Dogs vs. cats with a CNN trained from scratch
- [tf2-dvc_tfr-cnn-pretrained.py](tf2-dvc_tfr-cnn-pretrained.py): Dogs vs. cats with a pre-trained CNN
- [tf2-dvc_tfr-cnn-evaluate.py](tf2-dvc_tfr-cnn-evaluate.py): Evaluate a trained CNN with test data

## Extracurricular 2

There is another, small dataset [Aliens and predators](imgs/avp.png) (avp). Modify dogs vs. cats to classify between them.  

## Extracurricular 3

See <https://keras.io/examples/vision/> for more Keras examples on computer vision.

## Extracurricular 4 - after Exercise 6

There are scripts for both _Dogs vs. cats_ and _German traffic signs_ using
Vision Transformers (ViTs). Compare these with the previous approaches. There is
no separate evaluation script this time, test set accuracies are printed at the
end of the run.

- [tf2-dvc-vit.py](tf2-dvc-vit.py): Dogs vs. cats with a pre-trained ViT
- [tf2-gtsrb-vit.py](tf2-gtsrb-vit.py): Traffic signs with a pre-trained ViT

**Note:** you might need to upgrade HuggingFace Transformers to a newer version. This can be
done locally by the user like this:

```bash
pip install --user --upgrade transformers
```
