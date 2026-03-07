#!/bin/bash

# Mandatory value section
LR=8e-6
IN=model/path
RES=512x512
DATASET=/BLUE/CC8M-cache/CC8M-squarish-cleanup
# Optional value section
LPIPS=0.4
LAP=0.01

# There are some other hardcoded values in the call to
#  ./train_vae.py  
# you may or may not wish to change
###################################################
if [[ "$LPIPS" != "" ]] ; then
	OUTEXT="-lpips${LPIPS}"
	EXTRA_FLAGS+=" --lpips_weight ${LPIPS} --lpips_shapeonly"

fi
if [[ "$LAP" != "" ]] ; then
	OUTEXT+="-lap${LAP}"
	EXTRA_FLAGS+=" --laplacian_weight ${LAP}"
fi

OUT=test-model-p3redo-$LR$OUTEXT
mkdir -p $OUT
cp $0 $OUT

./train_vae.py  \
    --dataset $DATASET:$RES \
    --output_dir $OUT \
    --train_steps 200000 \
    --gradient_checkpointing \
    --batch_size 1 \
    --lr $LR \
    --save_every 10000 \
    --kl_weight 1e-7 \
    --edge_l1_weight 0.1 \
    --sample_img testimgs/0004bd163ff2be5900a1aa1af587235c.jpg \
    $EXTRA_FLAGS \
    --model $IN

#    --allow_tf32 \
