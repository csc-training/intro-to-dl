# Day 2 - puhti

## Exercise sessions

* [Exercise 5: Image classification](Exercise_5.md)
* [Exercise 6: Text categorization](Exercise_6.md)
* [Exercise 7: Using multiple GPUs](Exercise_7.md)
* [Exercise X: Hyperparameter optimization](Exercise_X.md)

## Setup

1. Login to Puhti using either:
   - the web user interface at <https://www.puhti.csc.fi/> and start "Login node shell" (in the "Tools" menu), or
   - with an SSH client:
 
     ```bash
     ssh -l trainingNNN puhti.csc.fi
     ```
   In both options you need the training username and password given in the lecture.
        
2. Set up the module environment for using TensorFlow 2 (and Keras):

   ```bash
   module purge
   module load tensorflow
   ```

   For PyTorch:
   
   ```bash
   module purge
   module load pytorch
   ```
   
3. Clone and change into the exercise repository:

   ```bash
   git clone https://github.com/csc-training/intro-to-dl
   cd intro-to-dl/day2
   ```

## Edit and submit jobs

1. Edit Python script, either by:
   - Navigating to the file in the Puhti web UI file browser (Files → Home Directory → intro-to-dl → day2) and selecting "Edit" on that file (under the three dots "⋮" menu).
   - Opening with your favorite text editor in the terminal, for example:
     ```bash
     nano tf2-test.py
     ```

2. Submit job:

   ```bash
   sbatch run.sh tf2-test.py
   ```

   There is a separate slurm script for PyTorch, e.g.:
   
   ```bash
   sbatch run-pytorch.sh pytorch_dvc_cnn_simple.py
   ```

   You can also specify additional command line arguments, e.g.

   ```bash
   sbatch run.sh tf2-dvc-cnn-evaluate.py dvc-cnn-simple.h5
   ```

3. See the status of your jobs or the queue you are using:

   ```bash
   squeue -l -u $USER
   squeue -l -p gpu
   ```

4. After the job has finished, examine the results:

   ```bash
   less slurm-xxxxxxxx.out
   ```

5. Go to 1 until you are happy with the results.

## Optional: TensorBoard

You can use TensorBoard either via the new Puhti web user interface, or via the terminal using ssh port forwarding. Both approaches are explained below.

### Via the Puhti web interface

1. Go to <https://www.puhti.csc.fi/>
2. Log in with CSC account (training account given during lecture)
3. Select menu item: Apps → TensorBoard
4. In the form:
   - Select course project: project_2005299
   - Specify the "TensorBoard log directory", it's where you have cloned the course repository plus "day2/logs", for example:
  `~/intro-to-dl/day2/logs`. You can run `pwd` in the terminal to find out the full path where you are working.
   - Leave rest at default settings
6. Click "Launch"
7. Wait until you see the "Connect to Tensorboard" button, then click that.
8. When you're done using TensorBoard, please go to "My Interactive Sessions" in the Puhti web user interface and "Delete" the session. (It will automatically terminate once the reserved time is up, but it's always better to release the resource as soon as possible so that others can use it.)

### Via SSH port forwarding

1. Login again in a second terminal window to Puhti with SSH port forwarding:

   ```bash
   ssh -l trainingxxx -L PORT:localhost:PORT puhti.csc.fi
   ```
        
   Replace `PORT` with a freely selectable port number (>1023). By default, TensorBoard uses the port 6006, but **select a different port** to avoid overlaps. 

2. Set up the module environment and start the TensorBoard server:

   ```bash
   module purge
   module load tensorflow
   tensorboard --logdir=intro-to-dl/day2/logs --port=PORT
   ```

3. To access TensorBoard, point your web browser to *localhost:PORT* .
