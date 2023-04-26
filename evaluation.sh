# Set the path to save checkpoints
OUTPUT_DIR='/home/mrchen/cmr/mosaic/DiG/no_pretrain/'
# path to imagenet-1k set
DATA_PATH='/home/mrchen/cmr/mosaic/DiG/npy_dir/testsets_lmdbs'
# path to finetune model
MODEL_PATH='/home/mrchen/cmr/mosaic/DiG/output4/checkpoint-39.pth'

# batch_size can be adjusted according to the graphics card
CUDA_VISIBLE_DEVICES=7 python -m torch.distributed.launch --nproc_per_node=1 --master_port 10040 run_class_finetuning.py \
    --model simmim_vit_small_patch4_32x128 \
    --data_path ${DATA_PATH} \
    --eval_data_path ${DATA_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 500 \
    --opt adamw \
    --opt_betas 0.9 0.999 \
    --weight_decay 0.05 \
    --data_set image_lmdb \
    --nb_classes 97 \
    --smoothing 0. \
    --max_len 25 \
    --resume ${MODEL_PATH} \
    --eval \
    --epochs 20 \
    --warmup_epochs 2 \
    --drop 0.1 \
    --attn_drop_rate 0.1 \
    --dist_eval \
    --num_samples 1000000 \
    --fixed_encoder_layers 0 \
    --decoder_name sr \
    --decoder_type sr \
    --beam_width 0 \