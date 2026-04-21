#!/bin/bash
# Usage: bash eval.sh <scene> <num_views> [stage]
# stage: 1 (repair, default) or 2 (inpainting)
DATASET="data/mipnerf360"
SCENE=$1
NUM_VIEW=$2
STAGE=${3:-1}

if [ "$STAGE" = "1" ]; then
    PLY_DIR="output_den${NUM_VIEW}"
    POSTFIX="_stage1"
elif [ "$STAGE" = "2" ]; then
    PLY_DIR="output_inp${NUM_VIEW}"
    POSTFIX="_stage2"
else
    echo "Invalid stage: $STAGE (use 1 or 2)"
    exit 1
fi

python render.py \
    -m output/gs_init/${SCENE}_${NUM_VIEW} \
    --sparse_view_num $NUM_VIEW --sh_degree 2 \
    --white_background --render_path \
    --postfix $POSTFIX \
    --load_ply ${PLY_DIR}/gaussian_object/${SCENE}_${NUM_VIEW}/save/last.ply