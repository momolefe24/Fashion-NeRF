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
echo "Launching the script ${SLURM_JOB_NAME}"
# ./scripts/viton/viton.sh $SLURM_JOB_NAME {experiment_number}  {run_number} {experiment_from}  {run_number_from}
# ./scripts/viton/viton.sh $SLURM_JOB_NAME 1 6 1 6


# export EXPERIMENT_NUMBER=26
# export EXPERIMENT_FROM_NUMBER=0
# export RUN_NUMBER=99
# export RUN_FROM_NUMBER=0
# export DATASET_NAME=Rail
# export TASK="PF_Gen"
# export DEBUG=1
# export SWEEPS=0
# export SEED=1
# export DATAMODE=train
# export WANDB=0
# export DEVICE=0
# export VITON_NAME=PF_AFN

echo "Start Debug: $DEBUG"
echo "Experiment Number: $EXPERIMENT_NUMBER"
echo "Run Number: $RUN_NUMBER"
echo "Dataset Name: $DATASET_NAME"
echo "Device: $DEVICE"
export CUDA_VISIBLE_DEVICES=$DEVICE


# parser_based_warp_experiment_from_number 19 - Original
# parser_based_warp_experiment_from_number 20 - Rail

# parser_based_gen_experiment_from_number 21 - Original
# parser_based_gen_experiment_from_number 22 - Rail

# parser_free_warp_experiment_from_number 23 - Original
# parser_free_warp_experiment_from_number 24 - Rail

# parser_free_gen_experiment_from_number 25 - Original
# parser_free_gen_experiment_from_number 26 - Rail

# Parser Free Gen
./scripts/viton/viton.sh --job_name $VITON_NAME --task $TASK --experiment_number 26 --run_number 1 \
    --experiment_from_number 26 --run_from_number 1 \
    --parser_based_warp_experiment_from_number 19 --parser_based_warp_run_from_number 1 --warp_load_from_model Original \
    --parser_based_gen_experiment_from_number 24 --parser_based_gen_run_from_number 1 --gen_load_from_model Original \
    --parser_free_warp_experiment_from_number 0 --parser_free_warp_run_from_number 0 --parser_free_warp_load_from_model Original \
    --parser_free_gen_experiment_from_number 0 --parser_free_gen_run_from_number 0 --parser_free_gen_load_from_model Original \
    --dataset_name $DATASET_NAME --validate False --device $DEVICE --load_last_step False --run_wandb $WANDB \
    --niter 50 --niter_decay 50 --display_count 10 --print_step 10 --save_period 10 --val_count 10 \
    --viton_batch_size 4 --datamode $DATAMODE --debug $DEBUG --sweeps $SWEEPS --seed $SEED
