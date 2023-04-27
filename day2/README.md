# Day 2 - puhti

## Exercise sessions

* [Exercise 5: Image classification](Exercise_5.md)
* [Exercise 6: Image generation](Exercise_6.md)
* [Exercise 7: Text categorization](Exercise_7.md)
* [Exercise 8: Text generation](Exercise_8.md)
* [Exercise 9: Using multiple GPUs](Exercise_9.md)

## Setup

1. Login to Puhti using either:
   - the web user interface at <https://www.puhti.csc.fi/> and start "Login node shell", or
   - login with your username and password using SSH to `puhti.csc.fi`, for more instructions see: <https://docs.csc.fi/computing/connecting/>
   
   If you are using LUMI instead of Puhti, see [the LUMI documentation](https://docs.lumi-supercomputer.eu/firststeps/getstarted/).
      
2. In the login node shell, or SSH session, set up the module environment for using TensorFlow (and Keras):

   ```bash
   module purge
   module load tensorflow
   ```
   
   If you are using LUMI, you need to run:
   ```bash
   module purge
   module use /appl/local/csc/modulefiles/
   module load tensorflow
   ```
   
   If you want to use PyTorch instead of TensorFlow, replace `tensorflow` with `pytorch` in the above commands.


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
   squeue -l --me
   squeue -l -p gpu
   ```

4. After the job has finished, examine the results:

   ```bash
   less slurm-xxxxxxxx.out
   ```

5. Go to 1 until you are happy with the results.

## Optional: TensorBoard

You can use TensorBoard either via the new Puhti web user interface (recommended), or via the terminal using ssh port forwarding. Both approaches are explained below.

### Via the Puhti web interface (the recommended method)

1. Go to <https://www.puhti.csc.fi/>
2. Log in with CSC account (training account given during lecture)
3. Select menu item: Apps → TensorBoard
4. In the form:
   - Select course project: project_2007759
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
   apptainer_wrapper exec tensorboard --logdir=intro-to-dl/day2/logs --port=PORT --bind_all
   ```

3. To access TensorBoard, point your web browser to *localhost:PORT* .
