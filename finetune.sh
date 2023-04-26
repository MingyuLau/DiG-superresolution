# Set the path to save checkpoints
OUTPUT_DIR='/home/mrchen/cmr/mosaic/DiG/output4/'
# path to imagenet-1k set
#DATA_PATH='/home/mrchen/cmr/mosaic/DiG/data'
DATA_PATH='/home/mrchen/cmr/mosaic/DiG/npy_dir/gt_lmdbs'
# path to finetune model
MODEL_PATH=None

# batch_size can be adjusted according to the graphics card
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=6 --master_port 10041 run_class_finetuning.py \
    --model simmim_vit_small_patch4_32x128 \
    --data_path ${DATA_PATH} \
    --eval_data_path ${DATA_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 128 \
    --opt adamw \
    --opt_betas 0.9 0.999 \
    --weight_decay 0.05 \
    --data_set image_lmdb \
    --nb_classes 97 \
    --smoothing 0. \
    --max_len 25 \
    --epochs 200 \
    --warmup_epochs 1 \
    --drop 0.1 \
    --attn_drop_rate 0.1 \
    --drop_path 0.1 \
    --dist_eval \
    --lr 1e-4 \
    --num_samples 1 \
    --fixed_encoder_layers 0 \
    --decoder_name sr \
    --decoder_type sr \
    --use_abi_aug \
    --num_view 2 \