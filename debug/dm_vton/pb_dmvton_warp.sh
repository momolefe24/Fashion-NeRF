#!/bin/bash
declare -A args
while [[ $# -gt 0 ]]; do
    case $1 in
        --*)
            args[${1#--}]=$2
            shift
            ;;
    esac
    shift
done
python3 FashionNeRF.py --experiment_number ${args[experiment_number]} --run_number ${args[run_number]} \
  --experiment_from_number ${args[experiment_from_number]} --run_from_number ${args[run_from_number]} \
  --warp_load_from_model ${args[warp_load_from_model]} \
  --VITON_Type Parser_Free --VITON_Name DM_VTON --VITON_Model ${args[task]} \
  --gpu_ids 0 --device ${args[device]} --res low_res --dataset_name ${args[dataset_name]} --run_wandb ${args[run_wandb]} \
  --low_res_dataset_name VITON-Clean --load_last_step ${args[load_last_step]}\
  --niter ${args[niter]} --niter_decay ${args[niter_decay]} --display_count ${args[display_count]} --print_step ${args[print_step]} --save_period ${args[save_period]} \
  --viton_batch_size ${args[viton_batch_size]} --datamode ${args[datamode]} --debug ${args[debug]} --sweeps ${args[sweeps]} --val_count ${args[val_count]} --seed ${args[seed]} \

