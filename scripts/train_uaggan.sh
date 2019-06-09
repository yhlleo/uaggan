GPU_ID=$1

python train.py \
  --dataroot ./datasets/horse2zebra \
  --name uaggan_horse2zebra \
  --model uag_gan \
  --dataset_mode unaligned \
  --pool_size 50 \
  --gpu_ids ${GPU_ID} \
  --batch_size 1 \
  --no_dropout
