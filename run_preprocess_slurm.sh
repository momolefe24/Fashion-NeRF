#!/bin/bash

process_name="cihp_pgn"

# log="viton.%N.%j"
log="$process_name.%N.%j"
path="/home-mscluster/mmolefe/Playground/Synthesising Virtual Fashion Try-On with Neural Radiance Fields/Fashion-SuperNeRF/inference_pipeline/preprocessing"
output="${path}/${log}.out"
error="${path}/${log}.err"
echo "Path: ${path}"
echo "output: $output"
echo "error: $error"
echo "$VITON_Name"

sbatch -J "$process_name" -o "$output" -e "$error" preprocessing.slurm
# sbatch -J "$process_name" -o "$output" -e "$error" stampede_preprocessing.slurm
# sbatch -J $VITON_Name bigbatch.slurm stampede_preprocessing
