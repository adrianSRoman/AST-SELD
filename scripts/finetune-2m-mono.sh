#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3
export TORCH_DISTRIBUTED_DEBUG="DETAIL"

blr=1e-3
mask_t_prob=0.25
mask_f_prob=0.25

dataset=audioset
ckpt=/hpc_stor03/sjtu_home/zhisheng.zheng/models/audiomae/pretrained.pth
# ckpt=/userhome/user/zsz01/scratch/AudioMAE-fusion/outputs/2m/binaural/mel-conv3x3-baseline-v2/checkpoint-59.pth

audioset_label=/hpc_stor03/sjtu_home/zhisheng.zheng/data/audioset/class_whitelist_encoder.csv
audioset_train_json=/hpc_stor03/sjtu_home/zhisheng.zheng/data/audioset/unbalanced_no_missing.json
audioset_train_weight=/hpc_stor03/sjtu_home/zhisheng.zheng/data/audioset/distributed/unbalanced.csv
audioset_eval_json=/hpc_stor03/sjtu_home/zhisheng.zheng/data/audioset/eval_no_missing.json

reverb_type=mono
reverb_train_json=/data/shared/zsz01/SpatialAudio/reverb/mp3d/train_reverberation.json
reverb_val_json=/data/shared/zsz01/SpatialAudio/reverb/mp3d/eval_reverberation.json

output_dir=/hpc_stor03/sjtu_home/zhisheng.zheng/AudioMAE-fusion/outputs/2m/mono/mel-nonorm-reverb
log_dir=/hpc_stor03/sjtu_home/zhisheng.zheng/AudioMAE-fusion/outputs/2m/mono/mel-nonorm-reverb/log

mkdir -p $output_dir
cp /hpc_stor03/sjtu_home/zhisheng.zheng/AudioMAE-fusion/models_vit.py $output_dir/
cp /hpc_stor03/sjtu_home/zhisheng.zheng/AudioMAE-fusion/engine_finetune_as.py $output_dir/
cp /hpc_stor03/sjtu_home/zhisheng.zheng/AudioMAE-fusion/scripts/finetune-2m-mono.sh $output_dir/
# -m debugpy --listen 55555 --wait-for-client
python -m torch.distributed.launch \
    --nproc_per_node=4 --master_port=25352 --use_env main_finetune_as.py \
    --log_dir $log_dir \
	--output_dir $output_dir \
    --model vit_base_patch16 \
    --dataset $dataset \
    --audioset_train $audioset_train_json \
    --audioset_eval $audioset_eval_json \
    --label_csv $audioset_label \
    --weight_csv $audioset_train_weight \
    --reverb_train $reverb_train_json \
    --reverb_val $reverb_val_json \
    --reverb_type $reverb_type \
    --finetune $ckpt \
    --blr $blr \
    --dist_eval \
    --batch_size 64 \
    --num_workers 4 \
    --roll_mag_aug \
    --mixup 0.5 \
    --mask_t_prob $mask_t_prob \
    --mask_f_prob $mask_f_prob \
    --first_eval_ep 20 \
    --epochs 100 \
    --warmup_epochs 10 \
    --weight_sampler \
    --distributed_wrapper \
    --mask_2d \
    --nb_classes 355 \
    --epoch_len 190000 \
