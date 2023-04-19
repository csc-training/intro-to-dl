# Day 1

## Exercise sessions

### Exercise 1

Introduction to Notebooks, Keras fundamentals.

* *01-tf2-test-setup.ipynb*<br/>[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/csc-training/intro-to-dl/blob/master/day1/01-tf2-test-setup.ipynb)

### Exercise 2

MNIST classification with MLPs.

* *02-tf2-mnist-mlp.ipynb*<br/>[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/csc-training/intro-to-dl/blob/master/day1/02-tf2-mnist-mlp.ipynb)

<details><summary>Optional exercises</summary>

* *pytorch-mnist-mlp.ipynb* (Pytorch version)<br/>[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/csc-training/intro-to-dl/blob/master/day1/optional/pytorch-mnist-mlp.ipynb)

* *tf2-chd-mlp.ipynb* (Regression with MLPs)<br/>[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/csc-training/intro-to-dl/blob/master/day1/optional/tf2-chd-mlp.ipynb)

</details>

### Exercise 3

Image classification with CNNs.

* *03-tf2-mnist-cnn.ipynb*<br/>[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/csc-training/intro-to-dl/blob/master/day1/03-tf2-mnist-cnn.ipynb)

### Exercise 4

Text sentiment classification with RNNs.

* *04-tf2-imdb-rnn.ipynb*<br/>[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/csc-training/intro-to-dl/blob/master/day1/04-tf2-imdb-rnn.ipynb)

<details><summary>Optional exercises</summary>

* *tf2-aclImdb-bert.ipynb* (Text sentiment classification with BERT)<br/>[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/csc-training/intro-to-dl/blob/master/day1/optional/tf2-aclImdb-bert.ipynb)

* *tf2-imdb-cnn.ipynb* (Text sentiment classification with CNNs)<br/>[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/csc-training/intro-to-dl/blob/master/day1/optional/tf2-imdb-cnn.ipynb)

* *tf2-mnist-rnn.ipynb* (MNIST classification with RNNs)<br/>[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/csc-training/intro-to-dl/blob/master/day1/optional/tf2-mnist-rnn.ipynb)

</details>

## Setup

We will use Jupyter Notebooks for all exercises on Day 1. There are several ways to set up a Jupyter environment for running the exercises:

### 1. CSC Notebooks

*The default option.* CSC Notebooks (https://notebooks.csc.fi) provides easy-to-use environments for working with data and programming. You can access everything via your web browser and CSC cloud environment computes on the background. There should be enough resources for launching a notebooks instance for everyone, but unfortunately no GPUs. 

1. Go to the [CSC Notebooks](https://notebooks.csc.fi) frontpage
2. Login according to selected login method:
   - **Haka or Virtu** (users from Finnish universities and research institutes)
       1. Press Login button on the frontpage
       2. Press Haka or Virtu button
       3. Select right organization
       4. Enter login information
   - **Special login** (if you have been given separate username and password for the course)
       1. Press "Special Login" button on the Notebooks frontpage (below the Login button)
       2. Enter login information (username goes to email slot)
3. Join workspace
   - Press "Join workspace" button (Top right corner)
   - Enter the Join Code **given in the lecture**
   - You should now see the workspace "Practical Deep Learning Course, Nov 2022"
4. Start Notebook session
   - In order to launch the Notebook "Practical Deep Learning, Nov 2022", click the round start button next to it.
   - Wait for session to launch
5. Once the Jupyter Notebook dashboard should appears navigate to `intro-to-dl/day1` 
6. If you are not familiar with Jupyter, take a moment to get to know the interface
   - open a new notebook (*File* -> *New* -> *Notebook*, on menubar) 
   - select *"Python 3"* as the kernel for the notebook
   - write some Python code to a Jupyter *cell*
   - execute the cell with *shift-enter*

#### :warning: Note
The notebook sessions have a limited time (8h) after which they, and any data or changes, will be *destroyed*. If you wish to save any files, you need to download them.
    
### 2. Running Jupyter on your laptop

If you have a laptop that has both jupyter and the other necessary python packages installed, it is possible to use it. In particular, if the laptop has an Nvidia GPU and it that has been properly set up (CUDA, cuDNN).

* `git clone https://github.com/csc-training/intro-to-dl.git`   
* try to run the `day1/01-tf2-test-setup.ipynb` notebook without errors

### 3. Google Colaboratory

Google has a free Jupyter Notebooks service you may want to try out. No guarantees, but it does have GPUs available! A Google account is needed to use Colaboratory. 

* Click the corresponding colab link above in this document (https://github.com/csc-training/intro-to-dl/tree/master/day1/README.md)
* If needed, sign in to your Google account using the "Sign in" button in the top-right corner
* To use a GPU, select: Runtime => Change runtime type => Hardware accelerator: GPU

OR

* Point your browser to https://github.com/csc-training/intro-to-dl/tree/master/day1 
* Select a notebook file
* At the end of the file, there is a link: “Run this notebook in Google Colaboratory using this link”
* If needed, sign in to your Google account using the "Sign in" button in the top-right corner
* To use a GPU, select: Runtime => Change runtime type => Hardware accelerator: GPU
