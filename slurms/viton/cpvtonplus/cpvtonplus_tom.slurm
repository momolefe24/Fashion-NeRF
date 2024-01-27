#!/bin/bash
#SBATCH -N 1
#SBATCH -t 72:00:00
#SBATCH --exclude=mscluster68,mscluster48,mscluster82
#SBATCH --ntasks=1
#SBATCH --partition=bigbatch

echo ------------------------------------------------------
echo -n 'Job is running on node ' $SLURM_JOB_NODELIST
echo ------------------------------------------------------
echo SLURM: sbatch is running on $SLURM_SUBMIT_HOST
echo SLURM: job ID is $SLURM_JOB_ID
echo SLURM: submit directory is $SLURM_SUBMIT_DIR
echo SLURM: number of nodes allocated is $SLURM_JOB_NUM_NODES
echo SLURM: number of cores is $SLURM_NTASKS
echo SLURM: job name is $SLURM_JOB_NAME
echo ------------------------------------------------------

/bin/hostname
nvidia-smi
source ~/.bashrc
conda activate NeRF
export CUDA_VISIBLE_DEVICES=0
echo "Launching the script ${SLURM_JOB_NAME}"
# ./scripts/viton/viton.sh $SLURM_JOB_NAME {experiment_number}  {run_number} {experiment_from}  {run_number_from}
# ./scripts/viton/viton.sh $SLURM_JOB_NAME 1 6 1 6


export EXPERIMENT_NUMBER=10
export EXPERIMENT_FROM_NUMBER=0
export RUN_NUMBER=99
export RUN_FROM_NUMBER=0
export DATASET_NAME=Original
export TASK="TOM"
export DEBUG=1
export DATAMODE=train
export VITON_NAME=CP_VTON_plus

echo "Start Debug: $DEBUG"
echo "Experiment Number: $EXPERIMENT_NUMBER"
echo "Run Number: $RUN_NUMBER"
echo "Dataset Name: $DATASET_NAME"


./scripts/viton/viton.sh --job_name $VITON_NAME --experiment_number $EXPERIMENT_NUMBER --run_number $RUN_NUMBER \
  --experiment_from_number $EXPERIMENT_NUMBER --run_from_number $RUN_NUMBER \
  --gmm_experiment_from_number $EXPERIMENT_NUMBER --gmm_run_from_number $RUN_NUMBER --gmm_load_from_model Original  \
  --tom_experiment_from_number $EXPERIMENT_NUMBER --tom_run_from_number $RUN_NUMBER --tom_load_from_model Original \
  --VITON_Type Parser_Based --VITON_Name $VITON_NAME --VITON_Model $TASK --stage $TASK --load_last_step False  \
  --res low_res --dataset_name $DATASET_NAME  --run_wandb True \
  --low_res_dataset_name viton_plus \
  --niter 200 --niter_decay 200 --display_count 10 --print_step 10 --save_period 10 \
  --viton_batch_size 4 --datamode train --debug $DEBUG
