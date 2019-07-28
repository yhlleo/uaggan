GPU_ID=$1

python3 train.py \
  --dataroot ./datasets/horse2zebra \
  --name uaggan_horse2zebra \
  --model uag_gan \
  --dataset_mode unaligned \
  --pool_size 50 \
  --gpu_ids ${GPU_ID} \
  --batch_size 1 \
  --use_early_stopping 1 \
  --use_mask_for_D 0 \
  --no_dropout
