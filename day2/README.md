# Day 2

## Exercise sessions

* [Exercise 5: Image classification](exercise5/README.md)
* [Exercise 6: Text categorization](exercise6/README.md)
* [Exercise 7: Base vs instruction-tuned LLM](exercise7/README.md)
* [Exercise 8: Using multiple GPUs](exercise8/README.md)

## Setup

1. Login to Roihu using either:
   - the web user interface at <https://www.roihu.csc.fi/> ("Go to login") and start "Login node shell (Roihu-GPU)", or
   - login with your username and SSH key to `roihu-gpu.csc.fi`, for more instructions see: <https://docs.csc.fi/support/tutorials/roihu/#ssh-client>

   Note: Roihu has separate login nodes, `roihu-gpu.csc.fi` (GPU, ARM) and `roihu-cpu.csc.fi` (CPU, x86). These exercises use GPUs and the ARM PyTorch module, so always use the GPU login node; environments built there won't work on the CPU side.

   ```bash
   # Replace <username> with the name of your CSC user account.

   ssh <username>@roihu-gpu.csc.fi
   ``` 
2. In the login node shell (Roihu-GPU), or SSH session, set up the module environment for using PyTorch. In day2 exercises, we will use the preinstalled PyTorch on Roihu: (https://docs.csc.fi/apps/pytorch/).

   ```bash
   module purge
   module load python-pytorch/2.13
   ```
   (In the Roihu web UI login node shell you can use Shift-Insert to paste if you copy commands from here.)
   
   It includes most of the libraries we will use in the exercises, one can check the pre-installed libraries by running `pip list`.
   
   ```bash
   pip list
   ```
   But if you need to install additional libraries, you can create a Python virtual environment on top of the loaded PyTorch module and install extra pip packages there. See [CSC's documentation on how to add packages to existing Python installations for more details](https://docs.csc.fi/support/tutorials/python-usage-guide/#installing-python-packages-to-existing-modules).
   
3. Go to the exercise directory:
   - if you ran the exercises of day 1 using Roihu's "Jupyter for courses", you should already have the repository cloned in your home directory
   
   ```bash
   cd PDL-2026-10/intro-to-dl/day2
   ```
   
   If you don't have it, you can also clone it yourself:

   ```bash
   mkdir PDL-2026-10
   cd PDL-2026-10
   git clone https://github.com/csc-training/intro-to-dl
   cd intro-to-dl/day2
   ```

## Edit and submit jobs

Each exercise has its own directory (`exercise5`, `exercise6`, ...) containing its instructions, Python scripts and the Slurm script `run.sh`, so from inside an exercise directory you submit e.g. `sbatch run.sh pytorch_dvc_cnn_simple.py`. Exercise 8 also has `run-2gpus.sh` and `run-4gpus.sh` for multiple GPUs. The test below is run directly from `day2`, which has its own `run.sh` for it.

1. Edit Python script, either by:
   - Navigating to the file in the Roihu web UI file browser (Files → Home Directory → PDL-2026-10 → intro-to-dl → day2 → exerciseN) and selecting "Edit" on that file (under the three dots "⋮" menu).
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
   squeue -p gpumedium
   ```

4. After the job has finished, examine the results (the `slurm-xxxxxxxx.out` file is written to the directory you submitted the job from):

   ```bash
   less slurm-xxxxxxxx.out
   ```

5. Go to 1 until you are happy with the results.

## Optional: TensorBoard

You can use TensorBoard via the Roihu web user interface.

1. Log in via <https://www.roihu.csc.fi/>
2. Select menu item: Apps → TensorBoard
3. In the form:
   - Project: select the course project "project_2020307"
   - Partition: interactive
   - Module: python-pytorch/2.13 (default)
   - Specify the "TensorBoard log directory", it's where you have cloned the course repository plus "day2/logs", for example:
  `~/PDL-2026-10/intro-to-dl/day2/logs`. You can run `pwd` in the terminal to find out the full path where you are working.
   - Leave the rest at the default settings (Reservation: No reservation, 1 CPU core, 8 GiB memory, Time 0:30:00)
4. Click "Launch"
5. Wait until you see the "Connect to Tensorboard" button, then click that.
6. When you're done using TensorBoard, please go to "My Interactive Sessions" in the Roihu web user interface and "Cancel" the session. (It will automatically terminate once the reserved time is up, but it's always better to release the resource as soon as possible so that others can use it.)
