#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1
export TORCH_DISTRIBUTED_DEBUG="DETAIL"

blr=1e-3
mask_t_prob=0.25
mask_f_prob=0.25

# Download from https://drive.google.com/file/d/1ni_DV4dRf7GxM8k-Eirx71WP9Gg89wwu/view?usp=share_link
ckpt="" #./pretrained/checkpoint-49.pth

# Sound source
dataset=audioset
audio_path_root=/scratch/ssd1/audio_datasets/spatial_audioset_dcase_10sec_tiny_15classes/mic_dev/mic
meta_path_root=/scratch/ssd1/seld-feats-spatial-seld-15classes/mic_dev_adpit_label/
audioset_label=/scratch/data/repos/BATLAM/data/subset_audioset_classes.csv
audioset_train_json=/scratch/data/repos/BATLAM/data/train_metadata.json
audioset_eval_json=/scratch/data/repos/BATLAM/data/val_metadata.json

# For reverberation data, please visit https://huggingface.co/datasets/zhisheng01/SpatialSounds/blob/main/mp3d_reverb.zip
reverb_type=tetra # or mono
reverb_path_root=/scratch/ssd1/audio_datasets/SpatialSounds/mp3d_reverb # https://github.com/zszheng147/Spatial-AST/tree/main?tab=readme-ov-file#reverberation
reverb_train_json=/scratch/ssd1/audio_datasets/SpatialSounds/mp3d_reverb/train_reverberation.json
reverb_val_json=/scratch/ssd1/audio_datasets/SpatialSounds/mp3d_reverb/eval_reverberation.json

output_dir=./outputs/debug
log_dir=./outputs/debug/log

mkdir -p $output_dir

python -m torch.distributed.launch \
    --nproc_per_node=2 --master_port=24432 --use_env main_finetune.py \
    --log_dir $log_dir --output_dir $output_dir \
    --model build_AST --dataset $dataset \
    --audio_path_root $audio_path_root --meta_path_root $meta_path_root \
    --audioset_train $audioset_train_json --audioset_eval $audioset_eval_json \
    --label_csv $audioset_label \
    --nb_classes 15 \
    --reverb_path_root $reverb_path_root --reverb_type $reverb_type \
    --reverb_train $reverb_train_json --reverb_val $reverb_val_json \
    --blr $blr --dist_eval --batch_size 8 --num_workers 2 \
    --roll_mag_aug --mixup 0.5 --audio_normalize \
    --mask_t_prob $mask_t_prob --mask_f_prob $mask_f_prob \
    --first_eval_ep 0 --epochs 50 --warmup_epochs 5 \
    --mask_2d 
    
