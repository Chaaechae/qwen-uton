#!/bin/bash

load_weights=$YOUR_PATH_TO_CAMBRIANS_S3_WEIGHTS
data_path=$YOUR_PATH_TO_CAMBRIANS_VSI_590K_JSONL
image_folder=$YOUR_PATH_TO_CAMBRIANS_VSI_590K_IMAGES
video_folder=$YOUR_PATH_TO_CAMBRIANS_VSI_590K_VIDEOS

# NOTE: for NFP video, we use data["nfp"] = True to mark it as nfp sample

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
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --mm_use_im_newline_token True \
    --image_aspect_ratio anyres \
    --group_by_modality_length True \
    --bf16 False \
    --output_dir gs://$YOUR_GCS_BUCKET/$YOUR_GCS_PREFIX \
    --num_train_epochs 1 \
    --per_device_train_batch_size ${per_device_train_batch_size:-8} \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy no \
    --save_strategy steps \
    --save_steps 250 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length ${max_seq_len:-11264} \
    --gradient_checkpointing True \
    --dataloader_num_workers ${dataloader_num_workers:-4} \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name $YOUR_RUN_NAME \
    --fsdp full_shard \
    --fsdp_config fsdp_config.json \
    --max_images_per_sample 128 \
    --anyres_max_subimages 9 \
    --si_token_len 729 \
    --miv_token_len 64 \
    --video_folder $video_folder \
    --video_fps 1 \
    --video_max_frames 128 \
    --video_force_sample True \
    --load_weights $load_weights \
    --nfp_head True \
    --nfp_mse_loss_weight 0.1 \
    --nfp_cosine_loss_weight 0.1 \
"

if [ -n "$resume" ]; then
    TRAIN_ARGS="$TRAIN_ARGS \
        --train_continue True \
        --resume_from_checkpoint $resume \
    "
fi

echo "Training arguments:"
echo $TRAIN_ARGS

python cambrian/train/train_${LAUNCHER:-"spmd"}.py $TRAIN_ARGS
