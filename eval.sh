DATASET="data/mipnerf360"
NUM_VIEW=$2

python render.py \
    -m output/gs_init/${1}_$NUM_VIEW \
    --sparse_view_num $NUM_VIEW --sh_degree 2 \
    --white_background --render_path \
    --load_ply output_den$NUM_VIEW/gaussian_object/$1\_$NUM_VIEW/save/last.ply