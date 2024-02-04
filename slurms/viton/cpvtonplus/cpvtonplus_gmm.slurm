#!/bin/bash
#SBATCH -N 1
#SBATCH -t 72:00:00
#SBATCH --exclude=mscluster68,mscluster48,mscluster82,mscluster61,mscluster42
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
echo "Launching the script ${SLURM_JOB_NAME}"
# ./scripts/viton/viton.sh $SLURM_JOB_NAME {experiment_number}  {run_number} {experiment_from}  {run_number_from}
# ./scripts/viton/viton.sh $SLURM_JOB_NAME 1 6 1 6

# export EXPERIMENT_NUMBER=9
# export EXPERIMENT_FROM_NUMBER=0
# export RUN_NUMBER=99
# export RUN_FROM_NUMBER=0
# export SEED=1
# export DATASET_NAME=Rail
# export TASK="GMM"
# export DEBUG=1
# export SWEEPS=1
# export DATAMODE=train
# export WANDB=1
# export DEVICE=0
# export VITON_NAME=CP_VTON_plus


echo "Start Debug: $DEBUG"
echo "Experiment Number: $EXPERIMENT_NUMBER"
echo "Run Number: $RUN_NUMBER"
echo "Dataset Name: $DATASET_NAME"
echo "Device: $DEVICE"
export CUDA_VISIBLE_DEVICES=$DEVICE

./scripts/viton/viton.sh --job_name $VITON_NAME --experiment_number $EXPERIMENT_NUMBER --run_number $RUN_NUMBER \
  --experiment_from_number 0 --run_from_number 0 \
  --gmm_experiment_from_number 1 --gmm_run_from_number 99 --gmm_load_from_model Original  \
  --tom_experiment_from_number 0 --tom_run_from_number 0 --tom_load_from_model Original \
  --VITON_Type Parser_Based --VITON_Name $VITON_NAME --VITON_Model $TASK --stage $TASK --load_last_step False  \
  --res low_res --dataset_name $DATASET_NAME  --run_wandb $WANDB \
  --low_res_dataset_name viton_plus \
  --niter 10000 --niter_decay 10000 --display_count 100 --print_step 100 --save_period 100 --val_count 100 \
  --viton_batch_size 4 --datamode $DATAMODE --debug $DEBUG --sweeps $SWEEPS --seed $SEED
