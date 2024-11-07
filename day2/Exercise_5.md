# Exercise 5

In this exercise, we study image classification with two datasets:

- [_Dogs vs. cats_](imgs/dvc.png) (dvc), where we train on 2000 images, each
  depicting either a cat or a dog,
- [_German traffic signs_](imgs/gtsrb-montage.png) (gtsrb), where we train on
  5535 images with [43 types of traffic signs](imgs/traffic-signs.png).

## Task 1

### Dogs vs. cats

Starting with the _Dogs vs. cats_ (dvc) database, train, evaluate and report the
accuracy with three different approaches:

- CNN trained from scratch: [pytorch_dvc_cnn_simple.py](pytorch_dvc_cnn_simple.py)
- Using a pre-trained CNN (VGG16) and fine tuning:
  [pytorch_dvc_cnn_pretrained.py](pytorch_dvc_cnn_pretrained.py)

You can run the training directly with the corresponding script listed above,
for example:

    sbatch run.sh pytorch_dvc_cnn_simple.py

As a reminder, you can check the status of your runs with the command:

    squeue --me

The output of the run will appear in a file named `slurm-RUN_ID.out`
where `RUN_ID` is the Slurm batch job id. You can check the last ten
lines of that file with the command:

    tail slurm-RUN_ID.out

Use `tail -f` if you want to continuously follow the progress of the
output. (Press Ctrl-C when you want to stop following the file.)

After training, the script runs an evaluation on the test set, you
should find the results of that towards the end of the output log on a
line starting with "Testing". It should contain the accuracy
(percentage of correctly classified images).

Check the outputs of each run. Note that the pre-trained model will
print out two results, once after pre-training, and again after
fine-tuning.  Which model gave the best testset result? Does
finetuning improve the result?


### German traffic signs

Repeat the experiment with the _German traffic signs_ (gtsrb) database. Which
model gives the best result in this case? Compare the results with the previous
dvc results.

The scripts are named in the same way as before, just replace "dvc" with
"gtsrb":

- CNN trained from scratch: [pytorch_gtsrb_cnn_simple.py](pytorch_gtsrb_cnn_simple.py)
- Using a pre-trained CNN (VGG16) and fine tuning:
  [pytorch_gtsrb_cnn_pretrained.py](pytorch_gtsrb_cnn_pretrained.py)


## Task 2

Pick one database (dvc or gtsrb) and try to improve the result, e.g., by
tweaking the model or the training parameters (optimizer, batch size, number of
epochs, etc.).

## Extracurricular 1

There are scripts for both _Dogs vs. cats_ and _German traffic signs_ using
Vision Transformers (ViTs). Compare these with the previous approaches.

- [pytorch_dvc_vit.py](pytorch_dvc_vit.py): _Dogs vs. cats_ with a pre-trained ViT
- [pytorch_gtsrb_vit.py](pytorch_gtsrb_vit.py): _German traffic signs_ with a pre-trained ViT

## Extracurricular 2

There is another, small dataset [Aliens and predators](imgs/avp.png)
(avp) with 694 training and 200 validation images in the directory
`/scratch/project_462000699/data/avp` on LUMI.  Modify the scripts for
_Dogs vs. cats_ to classify between them.

