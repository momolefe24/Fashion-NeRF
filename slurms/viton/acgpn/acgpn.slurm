#!/bin/bash
#SBATCH -N 1
#SBATCH -t 72:00:00
#SBATCH --exclude=mscluster68,mscluster48,mscluster82,mscluster61,mscluster62
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


# export EXPERIMENT_NUMBER=5
# export EXPERIMENT_FROM_NUMBER=0
# export RUN_NUMBER=1
# export RUN_FROM_NUMBER=0
# export DATASET_NAME=Rail
# export TASK=""
# export DEBUG=1
# export SWEEPS=1
# export DATAMODE=train
# export WANDB=1
# export DEVICE=0
# export VITON_NAME=ACGPN


echo "Start Debug: $DEBUG"
echo "Experiment Number: $EXPERIMENT_NUMBER"
echo "Run Number: $RUN_NUMBER"
echo "Dataset Name: $DATASET_NAME"

./scripts/viton/viton.sh --job_name $VITON_NAME --task "" --experiment_number $EXPERIMENT_NUMBER --run_number $RUN_NUMBER \
  --experiment_from_number 0 --run_from_number 0 \
  --g_experiment_from_number 0 --g_run_from_number 0 --g_load_from_model Original \
  --g1_experiment_from_number 0 --g1_run_from_number 0 --g1_load_from_model Original \
  --g2_experiment_from_number 0 --g2_run_from_number 0 --g2_load_from_model Original \
  --d_experiment_from_number 0 --d_run_from_number 0 --d_load_from_model Original \
  --d1_experiment_from_number 0 --d1_run_from_number 0 --d1_load_from_model Original \
  --d2_experiment_from_number 0 --d2_run_from_number 0 --d2_load_from_model Original \
  --d3_experiment_from_number 0 --d3_run_from_number 0 --d3_load_from_model Original \
  --unet_experiment_from_number 0 --unet_run_from_number 0 --unet_load_from_model Original \
  --vgg_experiment_from_number 0 --vgg_run_from_number 0 --vgg_load_from_model Original \
  --VITON_Type Parser_Based --VITON_Name $VITON_NAME \
  --res low_res --dataset_name $DATASET_NAME --device $DEVICE --load_last_step False  --run_wandb $WANDB \
  --low_res_dataset_name ACGPN \
  --niter 1 --niter_decay 1 --display_count 1 --print_step 1 --save_period 1 --val_count 1 \
  --viton_batch_size 4 --datamode $DATAMODE --debug $DEBUG --sweeps $SWEEPS


# --niter 50 --niter_decay 50 --display_count 10 --print_step 10 --save_period 10 \
# debug: --niter 1 --niter_decay 1 --display_count 1 --print_step 1 --save_period 1 \