# Day 1

## Exercise sessions

### Exercise 1

Introduction to Notebooks, Keras fundamentals.

* *keras-test-setup.ipynb*

### Exercise 2

Classification with MLPs.

* *keras-mnist-mlp.ipynb*

### Exercise 3

Image classification with CNNs.

* *keras-mnist-cnn.ipynb*

### Exercise 4

Text sentiment classification with CNNs and RNNs.

* *keras-imdb-cnn.ipynb*
* *keras-imdb-rnn.ipynb*

Optional: 

* *keras-mnist-rnn.ipynb*

## Setup

We will use Jupyter Notebooks for all exercises on Day 1. There are several ways to set up a Jupyter environment for running the exercises:

### 1. CSC’s Notebooks

*The default option.* CSC’s Notebooks (https://notebooks.csc.fi) provides easy-to-use environments for working with data and programming. You can access everything via your web browser and CSC cloud environment computes on the background. There should be enough resources for launching a notebooks instance for everyone, but unfortunately no GPUs. 

* Point your browser to https://notebooks.csc.fi
* Login using Alternate login
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

* `git clone -b kth2019 https://github.com/csc-training/intro-to-dl`   
* try to run the "day1/keras-test-setup.ipynb" notebook without errors

### 3. Google Colaboratory

Google has a free Jupyter Notebooks service you may want to try out. No guarantees, but it does have GPUs available!

* point your browser to https://github.com/csc-training/intro-to-dl/tree/kth2019/day1 
* select a notebook file
* at the end of the file, there is a link: “Run this notebook in Google Colaboratory using this link”
* to use a GPU, select: Runtime => Change runtime type => Hardware accelerator: GPU
