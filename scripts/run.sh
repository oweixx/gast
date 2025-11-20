cuda=0
date=1119
purpose=gast

scene_dir=/projects/gs/data/output
scene_list=("trex" "flower" "horns" "fern")

style_dir=./style
style_list=("fire" "block2" "galaxy" "hydrangea" "mosaic" "particle" "starry" "wave" "stone")

for scene in "${scene_list[@]}"
do 
    for style in "${style_list[@]}"
    do
        CUDA_VISIBLE_DEVICES=${cuda} python train.py \
                                    -m output/${date}/${purpose}/${scene}/${style} \
                                    -s ${scene_dir}/${scene} \
                                    --ply_path ${scene_dir}/${scene}/point_cloud/iteration_30000/point_cloud.ply \
                                    --style_image ${style_dir}/${style}.jpg \
                                    --debug_stylized
        CUDA_VISIBLE_DEVICES=${cuda} python depth_render_video.py \
                                    -m output/${date}/${purpose}/${scene}/${style} \
                                    --video_spiral --skip_train
        CUDA_VISIBLE_DEVICES=${cuda} python render_video.py \
                                    -m output/${date}/${purpose}/${scene}/${style} \
                                    --video_spiral
    done
done