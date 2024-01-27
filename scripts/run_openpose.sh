#!/bin/bash
args=("$@")
mscluster_path="Playground/Synthesising Virtual Fashion Try-On with Neural Radiance Fields/Fashion-SuperNeRF"
data_path="data/rail/inference"
person="${args[0]}"
clothing="${args[1]}"
filename="${args[2]}"
echo "person: ${person}"
echo "clothing: ${clothing}"
echo "filename: ${filename}"

cd /home/mo/Playground/Artificial\ Intelligence/Computer\ Vision/openpose/
./build/examples/openpose/openpose.bin --image_dir $person --hand --write_json "${person}_json" --write_images "${person}_output" --disable_blending --display 0
png_filename=${filename//.jpg/_rendered.png}
json_filename=${filename//.jpg/_keypoints.json}
output_png="${person}_output/${png_filename}"
output_json="${person}_json/${json_filename}"
openpose_img_save_dir="$mscluster_path/$data_path/${person}_${clothing}/openpose_img"
openpose_json_save_dir="$mscluster_path/$data_path/${person}_${clothing}/openpose_json"
# rsync $output_png mscluster:~/"${mscluster_path}/$person_$clothing/"
echo "Openpose Image: $openpose_img_save_dir"
echo "Openpose JSON: $openpose_json_save_dir"
echo "Output: ${output_png}"

# if [ ! -d "${openpose_json_save_dir}" ]
# then
#   echo "Creatingddd ${openpose_json_save_dir} to save the rendered openpose image ..."
#   pwd
#   # mkdir -p "${openpose_json_save_dir}"
# fi

# if [ ! -d "${openpose_img_save_dir}" ]
# then
#   echo "Creating ${openpose_img_save_dir} to save the rendered openpose image ..."
#   # mkdir  "${openpose_img_save_dir}"
# fi

if [ -f "${output_png}" ]
then
  echo "Sending ${output_png} back to the cluster ..."
  echo ".....$output_png ...."
  pwd
  rsync $output_png mscluster:~/"${openpose_img_save_dir}"
fi

if [ -f "${output_json}" ]
then
  echo "Sending ${output_json} back to the cluster ..."
  rsync $output_json mscluster:~/"${openpose_json_save_dir}"
fi