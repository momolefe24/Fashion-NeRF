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
  --g_experiment_from_number ${args[g_experiment_from_number]} --g_run_from_number ${args[g_run_from_number]} --g_load_from_model ${args[g_load_from_model]} \
  --g1_experiment_from_number ${args[g1_experiment_from_number]} --g1_run_from_number ${args[g1_run_from_number]} --g1_load_from_model ${args[g1_load_from_model]} \
  --g2_experiment_from_number ${args[g2_experiment_from_number]} --g2_run_from_number ${args[g2_run_from_number]} --g2_load_from_model ${args[g2_load_from_model]}  \
  --d_experiment_from_number ${args[d_experiment_from_number]} --d_run_from_number ${args[d_run_from_number]} --d_load_from_model ${args[d_load_from_model]} \
  --d1_experiment_from_number ${args[d1_experiment_from_number]} --d1_run_from_number ${args[d1_run_from_number]} --d1_load_from_model ${args[d1_load_from_model]} \
  --d2_experiment_from_number ${args[d2_experiment_from_number]} --d2_run_from_number ${args[d2_run_from_number]} --d2_load_from_model ${args[d2_load_from_model]} \
  --d3_experiment_from_number ${args[d3_experiment_from_number]} --d3_run_from_number ${args[d3_run_from_number]} --d3_load_from_model ${args[d3_load_from_model]} \
  --unet_experiment_from_number ${args[unet_experiment_from_number]} --unet_run_from_number ${args[unet_run_from_number]} --unet_load_from_model ${args[unet_load_from_model]}  \
  --vgg_experiment_from_number ${args[vgg_experiment_from_number]} --vgg_run_from_number ${args[vgg_run_from_number]} --vgg_load_from_model ${args[vgg_load_from_model]} \
  --VITON_Type Parser_Based --VITON_Name ACGPN --load_last_step ${args[load_last_step]} \
  --res low_res --dataset_name ${args[dataset_name]} --run_wandb ${args[run_wandb]} \
  --low_res_dataset_name ACGPN \
  --niter ${args[niter]} --niter_decay ${args[niter_decay]} --display_count ${args[display_count]} --print_step ${args[print_step]} --save_period ${args[save_period]} \
  --viton_batch_size ${args[viton_batch_size]} --datamode ${args[datamode]} --debug ${args[debug]} --sweeps ${args[sweeps]} --val_count ${args[val_count]} --seed ${args[seed]} \

