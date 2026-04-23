#!/bin/bash
# Usage: bash eval.sh <scene> <num_views> [stage]
# stage: 1 (repair, default) or 2 (inpainting)
DATASET="data/mipnerf360"
SCENE=$1
NUM_VIEW=$2
STAGE=${3:-1}

POSTFIX="_stage${STAGE}"

# Support both pretrained archive layout and training output layout
PRETRAINED_PLY="${NUM_VIEW}_views/${SCENE}/stage${STAGE}.ply"
if [ "$STAGE" = "1" ]; then
    TRAINING_PLY="output_den${NUM_VIEW}/gaussian_object/${SCENE}_${NUM_VIEW}/save/last.ply"
elif [ "$STAGE" = "2" ]; then
    TRAINING_PLY="output_inp${NUM_VIEW}/gaussian_object/${SCENE}_${NUM_VIEW}/save/last.ply"
else
    echo "Invalid stage: $STAGE (use 1 or 2)"
    exit 1
fi

if [ -f "$PRETRAINED_PLY" ]; then
    PLY_PATH="$PRETRAINED_PLY"
elif [ -f "$TRAINING_PLY" ]; then
    PLY_PATH="$TRAINING_PLY"
else
    echo "No ply found. Looked for:"
    echo "  $PRETRAINED_PLY"
    echo "  $TRAINING_PLY"
    exit 1
fi

python scripts/render.py \
    -m output/gs_init/${SCENE}_${NUM_VIEW} \
    --sparse_view_num $NUM_VIEW --sh_degree 2 \
    --white_background --render_path \
    --postfix $POSTFIX \
    --load_ply $PLY_PATH