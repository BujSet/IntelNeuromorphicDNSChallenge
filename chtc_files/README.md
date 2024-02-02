# CHTC Files

The files in this directory are for using the GPU resources from
CHTC for training and inference od the spiking neural network 
pipeline. 

## The Submit File

The file `train.sub` is an example of a possible submit file 
that can be used to launch a job to train on a GPU in the GPU lab
from CHTC. Changes to this file should be minimal, likely only needing to 
modify the GPU job length attribute depending on the number of epochs that 
should be simulated. 

However, there are various parts that should be monitor in the event that a job
crashes or produces unexpected results. In the case of the a crash, the first 
thing to check would be to ensure that you're using the latest docker container
for your jobs. The docker container ensures that the environment that our GPU
jobs runs on has all the necessary software packages. It is hosted on 
[DockerHub](https://hub.docker.com/repository/docker/rselagam/ndns/general) and 
as of Jan 24th, 2024, the latest version is v45. 

Another possible troubleshooting tip that may be useful is to increase the 
requested memory and disk space. While the values currently present in the 
`train.sub` file have been sufficient so far, as we further develop the network
model, we may need to tweak these. 

The most common issue is simply that CHTC terminates a job as attempted to run 
longer than what was specified. When experimenting with the number of epochs, 
the runtimes reported in the following table can be expected:

Numbers of Epochs Simulated | Aproximate GPU Time Needed to Complete
--- | ---
5   | 3.2-5.6 hours
10  | 5.9 - 7.1 hours
25  | 14.1 - 16.4 hours
50  | 1.05 - 1.30 days

The guidelines on the 
[CHTC GPU job information page](https://chtc.cs.wisc.edu/uw-research-computing/gpu-jobs) 
explain how what values the `+GPUJobLength` expects, and what the time limits
for your job to complete are for those values. You could just set the value to 
"long" for all jobs, but "short" jobs are more likely to get scheduled faster, 
so ensuring the right value here will help get results back as soon as 
possible.

### Using the Submit File

To submit a job using the submit file, you can run 

```
condor_submit train.sub
```

However, if you want to test out a feature or develop something and test on a
real GPU, you can launch an interactive job with:

```
condor_submit -i train.sub
```

When the interactive job launch, it will open an ssh connection to the GPU you
have been granted access too. From there, you can manually run the commands in 
`ndns_cmd.sh` to train the network. 

## The Command Script

The `ndns_cmd.sh` script is fairly involved at this point. Broadly, it sets 
appropriate environment variables, clones the prject repo, copies the speech
dataset over from the staging area, and downloads head related transfer 
functions all before training begins. After completing these steps, the script
launchs a process to the background that simply collect power statistic
information while the training script runs. 

Likely, the only line that will be modified in this script is the one that
launches the training process; begins with `python baseline_solution/sdnn_delays/train_sdnn.py`.
Here, there are lots of parameters to experiment with, and you can find 
more details about in the python training script file directly. 

