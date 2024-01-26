# Day 1

## Exercise sessions

### Exercise 1

Introduction to Notebooks, PyTorch fundamentals.

* *01-pytorch-test-setup.ipynb*<br/>[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/csc-training/intro-to-dl/blob/master/day1/01-pytorch-test-setup.ipynb)

### Exercise 2

MNIST classification with MLPs.

* *02-pytorch-mnist-mlp.ipynb*<br/>[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/csc-training/intro-to-dl/blob/master/day1/02-pytorch-mnist-mlp.ipynb)

### Exercise 3

Image classification with CNNs.

* *03-pytorch-mnist-cnn.ipynb*<br/>[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/csc-training/intro-to-dl/blob/master/day1/03-pytorch-mnist-cnn.ipynb)

### Exercise 4

Text sentiment classification with RNNs.

* *04-pytorch-imdb-rnn.ipynb*<br/>[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/csc-training/intro-to-dl/blob/master/day1/04-pytorch-imdb-rnn.ipynb)

## Setup

We will use Jupyter Notebooks for all exercises on Day 1. There are several ways to set up a Jupyter environment for running the exercises:


### 1. LUMI web user interface

*The default option.*

1. Go to the [LUMI web user interface](https://www.lumi.csc.fi/).
2. Login with Haka...
3. Click "Jupyter for courses" (this works only if you have been added to the course project)
4. Make sure the selections are correct:
   Project: project_462000450
   Course module: PracticalDeepLearning-Feb2024
   Working directory: /users/your-username-here
5. Click "Launch"
6. Once the applications has started click "Connect to Jupyter"
7. If you are not familiar with Jupyter, take a moment to get to know the interface
   - open a new notebook (*File* -> *New* -> *Notebook*, on menubar) 
   - select *"Python 3"* as the kernel for the notebook
   - write some Python code to a Jupyter *cell*
   - execute the cell with *shift-enter*

### 2. CSC Notebooks

CSC Notebooks (https://notebooks.csc.fi) provides easy-to-use environments for working with data and programming. You can access everything via your web browser and CSC cloud environment computes on the background. There should be enough resources for launching a notebooks instance for everyone, but unfortunately no GPUs. 

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
3. Start the "Practical Deep Learning" application
   - You might find it quicker if you select the "Machine Learning" tab
   - Click the round start button next to the "Practical Deep Learning" card
   - Wait for session to launch
5. Once the Jupyter Notebook dashboard appears, navigate to `intro-to-dl/day1` 
6. If you are not familiar with Jupyter, take a moment to get to know the interface
   - open a new notebook (*File* -> *New* -> *Notebook*, on menubar) 
   - select *"Python 3"* as the kernel for the notebook
   - write some Python code to a Jupyter *cell*
   - execute the cell with *shift-enter*

#### :warning: Note
The notebook sessions have a limited time (4h) after which they, and any data or changes, will be *destroyed*. If you wish to save any files, you need to download them.
    
### 3. Running Jupyter on your laptop

If you have a laptop that has both jupyter and the other necessary python packages installed, it is possible to use it. In particular, if the laptop has an NVIDIA or AMD GPU and it that has been properly set up (CUDA, cuDNN or ROCm).

* `git clone https://github.com/csc-training/intro-to-dl.git`   
* try to run the `day1/01-pytorch-test-setup.ipynb` notebook without errors

### 4. Google Colaboratory

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
