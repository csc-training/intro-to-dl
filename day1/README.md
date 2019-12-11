# Day 1

## Exercise sessions

### Exercise 1

Introduction to Notebooks, Keras fundamentals.

* *01-tf2-test-setup.ipynb*

### Exercise 2

Classification with MLPs.

* *02-tf2-mnist-mlp.ipynb*

### Exercise 3

Image classification with CNNs.

* *03-tf2-mnist-cnn.ipynb*

### Exercise 4

Text sentiment classification with CNNs and RNNs.

* *04a-tf2-imdb-cnn.ipynb*
* *04b-tf2-imdb-rnn.ipynb*

Optional: 

* *tf2-mnist-rnn.ipynb*
* *pytorch-mnist-mlp.ipynb*

## Setup

We will use Jupyter Notebooks for all exercises on Day 1. There are several ways to set up a Jupyter environment for running the exercises:

### 1. CSC’s Notebooks

*The default option.* CSC’s Notebooks (https://notebooks.csc.fi) provides easy-to-use environments for working with data and programming. You can access everything via your web browser and CSC cloud environment computes on the background. There should be enough resources for launching a notebooks instance for everyone, but unfortunately no GPUs. 

* Point your browser to https://notebooks.csc.fi
* Login using Haka or CSC account (or using a separate username and password)
* Find *Course Practical Deep Learning* and click “Launch new”
* Wait until the “Open in browser” link appears, then click on it
* The jupyter notebook dashboard should appear
* Navigate to `intro-to-dl/day1` 
* if you are not familiar with Jupyter, take a moment to get to know the interface
    * open a new notebook (*New* -> *Python 3*, located top-right on the dashboard) 
    * write some Python code to a Jupyter *cell*
    * execute the cell with *shift-enter*
    
### 2. Running Jupyter on your laptop

If you have a laptop that has both jupyter and the other necessary python packages installed, it is possible to use it. In particular, if the laptop has an Nvidia GPU and it that has been properly set up (CUDA, cuDNN).

* `git clone https://github.com/csc-training/intro-to-dl.git`   
* try to run the "day1/01-tf2-test-setup.ipynb" notebook without errors

### 3. Google Colaboratory

Google has a free Jupyter Notebooks service you may want to try out. No guarantees, but it does have GPUs available!

* point your browser to https://github.com/csc-training/intro-to-dl/tree/master/day1 
* select a notebook file
* at the end of the file, there is a link: “Run this notebook in Google Colaboratory using this link”
* to use a GPU, select: Runtime => Change runtime type => Hardware accelerator: GPU
