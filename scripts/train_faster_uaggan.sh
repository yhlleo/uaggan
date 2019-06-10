GPU_ID=$1

python3.5 train.py \
  --dataroot ./datasets/horse2zebra \
  --name faster_uaggan_horse2zebra \
  --model faster_uag_gan \
  --dataset_mode unaligned \
  --pool_size 50 \
  --gpu_ids ${GPU_ID} \
  --batch_size 1 \
  --no_dropout
