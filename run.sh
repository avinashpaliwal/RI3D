DATASET="data/mipnerf360"
# DATASET="/home/grads/a/avinashpaliwal/github/dust3r/co3d"
# NUM_VIEW=3
NUM_VIEW=$2


# system.init_dreamer="output/gs_init/${1}_$NUM_VIEW" \

# python gaussian_object_system.py \
# --config configs/gaussian-object.yaml \
# --train --gpu 0 \
# tag="${1}_$NUM_VIEW" \
# exp_root_dir="output_den$NUM_VIEW" \
# system.exp_name="output/controlnet_finetune/${1}_$NUM_VIEW" \
# system.init_dreamer="output_den$NUM_VIEW/gaussian_object/${1}_$NUM_VIEW" \
# system.refresh_size=8 \
# data.data_dir="$DATASET/$1" \
# data.resolution=4 \
# data.sparse_num=$NUM_VIEW \
# data.prompt="a photo" \
# data.refresh_size=8 \
# data.refresh_interval=1 \
# data.around_gt_steps=40 \
# system.sh_degree=2

# python train_gs_init.py -s $DATASET/$1 \
#     -m debug/gs_init/$1\_$NUM_VIEW \
#     -r 4 --sparse_view_num $NUM_VIEW --sh_degree 2 \
#     --white_background --random_background

# python -W ignore train_gs.py -s $DATASET/$1 \
#     -m output/gs_init/$1\_$NUM_VIEW \
#     -r 4 --sparse_view_num $NUM_VIEW --sh_degree 2 \
#     --white_background --random_background \
#     --ply_path debug/gs_init/${1}_$NUM_VIEW/point_cloud/iteration_1/point_cloud.ply

# python -W ignore leave_one_out_stage1.py -s $DATASET/$1 \
#     -m output/gs_init/$1\_loo\_$NUM_VIEW \
#     -r 4 --sparse_view_num $NUM_VIEW --sh_degree 2 \
#    --white_background --random_background \
#    --ply_path debug/gs_init/${1}_$NUM_VIEW/point_cloud/iteration_1/point_cloud.ply

# python -W ignore leave_one_out_stage2.py -s $DATASET/$1 \
#     -m output/gs_init/$1\_loo\_$NUM_VIEW \
#     -r 4 --sparse_view_num $NUM_VIEW --sh_degree 2 \
#     --white_background --random_background \
#     --ply_path debug/gs_init/${1}_$NUM_VIEW/point_cloud/iteration_1/point_cloud.ply

# python train_lora.py --exp_name controlnet_finetune/$1\_$NUM_VIEW \
# --prompt xxy5syt00 --sh_degree 2 --resolution 4 --sparse_num $NUM_VIEW \
# --data_dir $DATASET/$1 \
# --gs_dir output/gs_init/$1\_$NUM_VIEW \
# --loo_dir output/gs_init/$1\_loo\_$NUM_VIEW \
# --bg_white --sd_locked --train_lora \
# --add_diffusion_lora --add_control_lora --add_clip_lora

# # conda activate realfill
# MODEL_NAME="stabilityai/stable-diffusion-2-inpainting"
# # TRAIN_DIR="/home/grads/a/avinashpaliwal/github/GaussianObject/data/mipnerf360_512"
# # OUTPUT_DIR="$1-model"

# python /home/grads/a/avinashpaliwal/github/realfill/train_realfill.py \
#   --pretrained_model_name_or_path=$MODEL_NAME \
#   --train_data_dir="$DATASET/$1/$NUM_VIEW" \
#   --output_dir="/data1/avinash/gausobj/inpainting/${1}_$NUM_VIEW" \
#   --resolution=512 \
#   --train_batch_size=16 \
#   --gradient_accumulation_steps=1 \
#   --unet_learning_rate=2e-4 \
#   --text_encoder_learning_rate=4e-5 \
#   --lr_scheduler="constant" \
#   --lr_warmup_steps=100 \
#   --max_train_steps=2000 \
#   --lora_rank=8 \
#   --lora_dropout=0.1 \
#   --lora_alpha=16 \

# # conda deactivate


# mkdir -p output_den$NUM_VIEW/gaussian_object/$1\_$NUM_VIEW/

# python -W ignore train_repair.py \
# --config configs/gaussian-object.yaml \
# --train --gpu 0 \
# tag="${1}_$NUM_VIEW" \
# exp_root_dir="output_den$NUM_VIEW" \
# system.init_dreamer="debug/gs_init/${1}_$NUM_VIEW" \
# system.exp_name="output/controlnet_finetune/${1}_$NUM_VIEW" \
# system.refresh_size=8 \
# data.data_dir="$DATASET/$1" \
# data.resolution=4 \
# data.sparse_num=$NUM_VIEW \
# data.prompt="a photo of a xxy5syt00" \
# data.refresh_size=8 \
# system.sh_degree=2 2>&1 | tee output_den$NUM_VIEW/gaussian_object/$1\_$NUM_VIEW/output.log


mkdir -p output_inp$NUM_VIEW/gaussian_object/$1\_$NUM_VIEW/

python -W ignore train_repair.py \
--config configs/gaussian-object_inp.yaml \
--train --gpu 0 \
tag="${1}_$NUM_VIEW" \
exp_root_dir="output_inp$NUM_VIEW" \
system.init_dreamer="output_den$NUM_VIEW/gaussian_object/${1}_$NUM_VIEW" \
system.exp_name="output/controlnet_finetune/${1}_$NUM_VIEW" \
system.refresh_size=10 \
data.data_dir="$DATASET/$1" \
data.resolution=4 \
data.sparse_num=$NUM_VIEW \
data.prompt="a photo of a xxy5syt00" \
data.refresh_size=10 \
system.sh_degree=2  2>&1 | tee output_inp$NUM_VIEW/gaussian_object/$1\_$NUM_VIEW/output.log