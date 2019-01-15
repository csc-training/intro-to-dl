# Notebooks

## Exercise sessions

### Exercise 1

Introduction to Notebooks, Keras fundamentals, MLPs.

* *keras-test-setup.ipynb*
* *keras-mnist-mlp.ipynb*

### Exercise 2

Image classification with CNNs.

* *keras-mnist-cnn.ipynb*

### Exercise 3

Text sentiment classification with RNNs.

* *keras-imdb-rnn.ipynb*

## Setup

We will use Jupyter Notebooks for Exercises 1-3. There are several ways to set up a Jupyter environment for running the exercises:

### 1. CSC’s Notebooks

*The default option.* CSC’s Notebooks (https://notebooks.csc.fi) provides easy-to-use environments for working with data and programming. You can access everything via your web browser and CSC cloud environment computes on the background. There should be enough resources for launching a notebooks instance for everyone, but unfortunately no GPUs. 

* point your browser to https://notebooks.csc.fi
* login using Haka (or using a separate username and password)
* “Launch new” *Jupyter Machine Learning* environment
* wait until “Open in browser” link appears, then click on it
* the jupyter notebook dashboard should appear
* click “New” on the right side of the screen, then “Python 3”
* if you are not familiar with Jupyter, take a moment to get to know the interface
    * write some Python code to a Jupyter *cell*
    * execute the cell with *shift-enter*
* run the following command on an empty cell:

    `!git clone https://github.com/csc-training/intro-to-dl.git`

* go back to the dashboard, the exercises are located in the directory "intro-to-dl/day1".
  
### 2. Running Jupyter on your laptop

If you have a laptop that has both jupyter and the other necessary python packages installed, it is possible to use it. In particular, if the laptop has an Nvidia GPU and it that has been properly set up (CUDA, cuDNN).

* `git clone https://github.com/csc-training/intro-to-dl.git`   
* try to run the "day1/keras-test-setup.ipynb" notebook without errors

### 3. Google Colaboratory

Google has a free Jupyter Notebooks service you may want to try out. No guarantees, but it does have GPUs available!

* point your browser to https://github.com/csc-training/intro-to-dl/tree/hidata2019/notebooks 
* select a notebook file
* at the end of the file, there is a link: “Run this notebook in Google Colaboratory using this link”
* to use a GPU, select: Runtime => Change runtime type => Hardware accelerator: GPU
