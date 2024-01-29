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

./scripts/viton/viton.sh --job_name Ladi_VITON --experiment_number 12 --run_number 1 \
  --experiment_from_number 12 --run_from_number 1 \
  --tps_experiment_from_number 12 --tps_run_from_number 1 --tps_load_from_model Original \
  --VITON_Type Parser_Based --VITON_Name Ladi_VITON --VITON_Model TPS --load_last_step False \
  --gpu_ids 0 --device 0 --res high_res --dataset_name Original --run_wandb False \
  --niter 1 --niter_decay 1 --display_count 1 --print_step 1 --save_period 1 \
  --viton_batch_size 4 --datamode train