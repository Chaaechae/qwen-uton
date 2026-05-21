#!/bin/bash

data_path=$YOUR_PATH_TO_CAMBRIAN_ALIGNMENT_JSONL
image_folder=$YOUR_PATH_TO_CAMBRIAN_ALIGNMENT_IMAGES

# training utils
TRAIN_ARGS="
    --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
    --version qwen_2 \
    --data_path $data_path \
    --image_folder $image_folder \
    --vision_tower_aux_list [\"google/siglip2-so400m-patch14-384\"] \
    --vision_tower_aux_token_len_list [729] \
    --image_position 14 \
    --vision_hidden_size 1152 \
    --connector_only True \
    --unfreeze_mm_vision_tower False \
    --mm_projector_type mlp2x_gelu \
    --mm_projector_lr 1e-3 \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --mm_use_im_newline_token True \
    --image_aspect_ratio pad \
    --bf16 False \
    --output_dir gs://$YOUR_GCS_BUCKET/$YOUR_GCS_PREFIX \
    --num_train_epochs 1 \
    --per_device_train_batch_size ${per_device_train_batch_size:-8} \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy no \
    --save_strategy steps \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.06 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers ${dataloader_num_workers:-4} \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name $YOUR_RUN_NAME \
    --fsdp full_shard \
    --fsdp_config fsdp_config.json \
    --max_images_per_sample 1 \
    --anyres_max_subimages 1 \
    --si_token_len 729 \
    --miv_token_len 0 \
"

if [ -n "$resume" ]; then
    TRAIN_ARGS="$TRAIN_ARGS \
        --train_continue True \
        --resume_from_checkpoint $resume \
    "
fi

echo "Training arguments:"
echo $TRAIN_ARGS

python cambrian/train/train_spmd.py $TRAIN_ARGS
