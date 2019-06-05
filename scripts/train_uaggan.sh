GPU_ID=0

python train.py \
  --dataroot ./datasets/horse2zebra \
  --name uaggan_horse2zebra \
  --model uag_gan \
  --pool_size 50 \
  --thresh 0.1 \
  --gpu_ids ${GPU_ID} \
  --no_dropout
