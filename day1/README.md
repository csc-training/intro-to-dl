# Day 1

## Exercise sessions

### Exercise 1

Introduction to Notebooks, Keras fundamentals.

* *01-tf2-test-setup.ipynb*

### Exercise 2

Classification with MLPs.

* *02-tf2-mnist-mlp.ipynb*

Optional: 

* *pytorch-mnist-mlp.ipynb*

### Exercise 3

Image classification with CNNs.

* *03-tf2-mnist-cnn.ipynb*

### Exercise 4

Text sentiment classification with RNNs.

* *04-tf2-imdb-rnn.ipynb*

Optional: 

* *tf2-imdb-cnn.ipynb*
* *tf2-mnist-rnn.ipynb*


## Setup

We will use Jupyter Notebooks for all exercises on Day 1. There are several ways to set up a Jupyter environment for running the exercises:

### 1. CSC’s Notebooks

*The default option.* CSC’s Notebooks (https://notebooks.csc.fi) provides easy-to-use environments for working with data and programming. You can access everything via your web browser and CSC cloud environment computes on the background. There should be enough resources for launching a notebooks instance for everyone, but unfortunately no GPUs. 

* Point your browser to https://notebooks.csc.fi
* Login using
    * Haka or CSC account if you have one, or
    * a separate username and password that you will receive during the course
* Find *Course Practical Deep Learning - 2020* and click “Launch new”
* Wait until the “Open in browser” link appears, then click on it
* The jupyter notebook dashboard should appear
* Navigate to `intro-to-dl/day1` 
* if you are not familiar with Jupyter, take a moment to get to know the interface
    * open a new notebook (*File* -> *New* -> *Notebook*, on menubar) 
    * select *"Python 3"* as the kernel for the notebook
    * write some Python code to a Jupyter *cell*
    * execute the cell with *shift-enter*
    
### 2. Running Jupyter on your laptop

If you have a laptop that has both jupyter and the other necessary python packages installed, it is possible to use it. In particular, if the laptop has an Nvidia GPU and it that has been properly set up (CUDA, cuDNN).

* `git clone https://github.com/csc-training/intro-to-dl.git`   
* try to run the "day1/01-tf2-test-setup.ipynb" notebook without errors

### 3. Google Colaboratory

Google has a free Jupyter Notebooks service you may want to try out. No guarantees, but it does have GPUs available! A Google account is needed to use Colaboratory.

* Point your browser to https://github.com/csc-training/intro-to-dl/tree/master/day1 
* Select a notebook file
* At the end of the file, there is a link: “Run this notebook in Google Colaboratory using this link”
* If needed, sign in to your Google account using the "Sign in" button in the top-right corner
* To use a GPU, select: Runtime => Change runtime type => Hardware accelerator: GPU
