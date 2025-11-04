# Day 2

## Exercise sessions

* [Exercise 5: Image classification](Exercise_5.md)
* [Exercise 6: Text categorization](Exercise_6.md)
* [Exercise 7: Text generation](Exercise_7.md)
* [Exercise 8: Using multiple GPUs](Exercise_8.md)

## Setup

1. Login to LUMI using either:
   - the web user interface at <https://www.lumi.csc.fi/> ("Go to login") and start "Login node shell", or
   - login with your username and SSH key to `lumi.csc.fi`, for more instructions see: <https://docs.lumi-supercomputer.eu/firststeps/>
   
2. In the login node shell, or SSH session, set up the module environment for using PyTorch:

   ```bash
   module purge
   module use /appl/local/csc/modulefiles/
   module load pytorch
   ```
   (In the LUMI web UI login node shell you can use Shift-Insert to paste if you copy commands from here.)
   
3. Go to the exercise directory:
   - if you ran the exercises of day 1 using LUMI's "Jupyter for courses", you should already have the repository cloned in your home directory
   
   ```bash
   cd PDL-2025-11/intro-to-dl/day2
   ```
   
   If you don't have it, you can also clone it yourself:

   ```bash
   mkdir PDL-2025-11
   cd PDL-2025-11
   git clone https://github.com/csc-training/intro-to-dl
   cd intro-to-dl/day2
   ```

## Edit and submit jobs

1. Edit Python script, either by:
   - Navigating to the file in the LUMI web UI file browser (Files → Home Directory → PDL-2025-11 → intro-to-dl → day2) and selecting "Edit" on that file (under the three dots "⋮" menu).
   - Opening with your favorite text editor in the terminal, for example:
     ```bash
     nano pytorch_test.py
     ```

2. Submit job:

   ```bash
   sbatch run.sh pytorch_test.py
   ```
   
3. See the status of your jobs or the queue you are using:

   ```bash
   squeue --me
   squeue -p small-g
   ```

4. After the job has finished, examine the results:

   ```bash
   less slurm-xxxxxxxx.out
   ```

5. Go to 1 until you are happy with the results.

## Optional: TensorBoard

You can use TensorBoard either via the LUMI web user interface (recommended), or via the terminal using ssh port forwarding. Both approaches are explained below.

### Via the LUMI web interface (the recommended method)

1. Log in via <https://www.lumi.csc.fi/>
2. Select menu item: Apps → TensorBoard
4. In the form:
   - Select course project: project_462001095
   - Specify the "TensorBoard log directory", it's where you have cloned the course repository plus "day2/logs", for example:
  `~/PDL-2025-11/intro-to-dl/day2/logs`. You can run `pwd` in the terminal to find out the full path where you are working.
   - Leave rest at default settings
6. Click "Launch"
7. Wait until you see the "Connect to Tensorboard" button, then click that.
8. When you're done using TensorBoard, please go to "My Interactive Sessions" in the LUMI web user interface and "Cancel" the session. (It will automatically terminate once the reserved time is up, but it's always better to release the resource as soon as possible so that others can use it.)

### Via SSH port forwarding

1. Login again from a terminal window to LUMI with SSH port forwarding:

   ```bash
   ssh -L PORT:localhost:PORT lumi.csc.fi
   ```
        
   Replace `PORT` with a freely selectable port number (>1023). By default, TensorBoard uses the port 6006, but **select a different port** to avoid overlaps. 

2. Set up the module environment and start the TensorBoard server:

   ```bash
   module purge
   module use use /appl/local/csc/modulefiles/
   module load tensorflow
   singularity_wrapper exec tensorboard --logdir=PDL-2025-11/intro-to-dl/day2/logs --port=PORT --bind_all
   ```

3. To access TensorBoard, point your web browser to *localhost:PORT* .
