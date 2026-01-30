#!/bin/bash -l
#SBATCH --job-name=setup   # Job name
#SBATCH --output=setup.o%j # Name of stdout output file
#SBATCH --error=setup.e%j  # Name of stderr error file
#SBATCH --partition=small       # Partition name
#SBATCH --ntasks=1              # One task (process)
#SBATCH --mem=224G              # Memory request
#SBATCH --time=02:00:00         # Run time (hh:mm:ss)
#SBATCH --account=project_462001088  # Project for billing


# You should replace project_462001088 with your own project number.
# keep everything above this line as-is.

## Preeliminary concepts:
### - EasyBuild is a software framework that automates the process of building and installing scientific software on HPC systems.
### - Singularity is a container platform designed for HPC environments, allowing users to create and run
###   containers that package applications and their dependencies. (similar to Docker but optimized for HPC).

# singularity build --force /projappl/project_462001088/EasyBuild/SW/container/PyTorch/2.7.1-rocm-6.2.4-python-3.12-Mycontainer-singularity-20250827/lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.7.1-dockerhash-0d479e852886.sif lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.7.1-Mycontainer.def

# To start, we set the environment variable that points to the user easybuild directory.
# Basically this tells easybuild where to install the container.
# By default, this is in your home directory, but on LUMI, home directories have limited space.
# So we will set this to a location in the scratch space allocated to your project.
# export EBU_USER_PREFIX=/projappl/project_462001088/EasyBuild

installdir=/scratch/project_462001088/$USER/DEMO1
cd "$installdir/tmp"
# mkdir -p "$installdir" ; cd "$installdir"
# mkdir -p "$installdir/tmp" ; cd "$installdir/tmp"
# module purge
# module load LUMI/24.03 partition/container EasyBuild-user

# # We will use easy build commands (eb) to create a new container based on an existing one.
# # Now copy a prebuilt container to a new location and modify its name.
# # The container we will use is PyTorch with ROCm support for AMD GPUs.
# # learn more here: https://lumi-supercomputer.github.io/LUMI-EasyBuild-docs/p/PyTorch/PyTorch-2.7.1-rocm-6.2.4-python-3.12-singularity-20250827/
# eb --copy-ec PyTorch-2.7.1-rocm-6.2.4-python-3.12-singularity-20250827.eb PyTorch-2.7.1-rocm-6.2.4-python-3.12-Mycontainer-singularity-20250827.eb
# # the following line modifies the version suffix inside the .eb file to reflect the new name
# sed -e "s|^\(versionsuffix.*\)-singularity-\(.*\)|\1-Mycontainer-singularity-\2|" -i PyTorch-2.7.1-rocm-6.2.4-python-3.12-Mycontainer-singularity-20250827.eb
# # now build the new container (create the singularity image file)
# eb PyTorch-2.7.1-rocm-6.2.4-python-3.12-Mycontainer-singularity-20250827.eb

## The above container is prebuilt with some available packages, but we need to add more packages to it.
# For example, if you need opencv, we will have to add some system libs and then install opencv via pip inside the container.
# First we need to create a singularity definition file that adds the required system libs.

# Now load the container module to get the path to the singularity image file (SIF).
# A SIF file is the actual singularity container file.
# if you are familiar with Docker, a SIF file is similar to a Docker image file.
# module load PyTorch/2.7.1-rocm-6.2.4-python-3.12-Mycontainer-singularity-20250827
# The following lines load the modules (a set of libraries and tools) needed to modify the container.
module purge
module load LUMI/24.03

# Now load the container module to get the path to the singularity image file (SIF).
# A SIF file is the actual singularity container file.
# if you are familiar with Docker, a SIF file is similar to a Docker image file.
module load PyTorch/2.7.1-rocm-6.2.4-python-3.12-Mycontainer-singularity-20250827

# We will need the path to the singularity image file (SIF) of the container.
# so we store it in a variable for later use.
export CONTAINERFILE="$SIF"
module unload PyTorch/2.7.1-rocm-6.2.4-python-3.12-Mycontainer-singularity-20250827

# To install system libs, we need to load the systools module (which contains proot command)
# proot command allows us to modify existing singularity images, without needing root access.
module load systools/24.03

# The following creates a singularity definition file that adds the required system libs.
# A singularity definition file is a text file that contains instructions on how to build or modify a singularity container.
# If you are familiar with Dockerfiles, singularity definition files are similar in concept.
# cat > lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.7.1-Mycontainer.def <<EOF

# Bootstrap: localimage

# From: $CONTAINERFILE

# %post

# zypper -n install -y Mesa libglvnd libgthread-2_0-0 hostname

# EOF
# Now that we have everything set up, we can build the new container with the additional system libs.
singularity build --force $CONTAINERFILE lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.7.1-Mycontainer.def


